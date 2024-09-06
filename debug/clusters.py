import logging
import torch
from pathlib import Path
import faiss
from experimaestro import Param, Task, pathgenerator, Annotated, LightweightTask, Config, tqdm
import torch.nn as nn
from datamaestro_text.data.ir import DocumentStore, IDItem, SimpleTextItem
from xpmir.text.encoders import (
    TokenizedTextEncoder,
    TokenizedTexts,
    TextsRepresentationOutput,
)
from typing import List
from xpmir.papers.helpers.samplers import (
    msmarco_hofstaetter_ensemble_hard_negatives,
    msmarco_v1_docpairs_sampler,
    msmarco_v1_tests,
    msmarco_v1_validation_dataset,
    prepare_collection,
)

from experimaestro.launcherfinder import find_launcher

from transformers import AutoConfig, PreTrainedModel, AutoTokenizer
from transformers.models.bert import BertModel
from transformers.utils import cached_file
from xpmir.learning import ModuleInitMode
from xpmir.text import TokenizedTextEncoder, TokenizerOptions
from xpmir.text.huggingface import HFModel

from xpmir.text.huggingface import HFMaskedLanguageModel
from xpmir.text.huggingface import HFStringTokenizer, HFTokensEncoder
from colbert import ColbertEncoder


class LinearModel(Config):
    path: Param[Path]
    """Path to the linear module parameters"""


class HFProjectionLayerLoader(LightweightTask):
    linear: Param[LinearModel]
    model: Param[HFMaskedLanguageModel]

    def execute(self):
        logging.info("Replacing output embeddings with %s", self.linear.path)
        
        with self.linear.path.open("rb") as fp:
            linear = torch.load(fp)

        self.model.set_output_embeddings(linear)


class ClusterCompute(Task):
    encoder: Param[TokenizedTextEncoder[str, TextsRepresentationOutput, TokenizedTexts]]
    """The encoder that transforms documents (as strings) to tensors (B x L x D)"""

    path: Annotated[Path, pathgenerator("words.pt")]
    """Contains the linear layer"""
    
    vectors_per_token: Param[int]
    """Number of vectors per token"""

    documents_per_token: Param[int]
    """Number of vectors per token (must be greater than vectors_per_token)"""
    
    documents: Param[DocumentStore]
    """Documents"""

    max_vocab: Param[int]
    """Maximum number of vocabulary"""

    def task_outputs(self) -> LinearModel:
        LinearModel
    

    def clustering(self, vk: torch.Tensor, tokenized_vectors: List[torch.Tensor], max_vocab: int):
        D = tokenized_vectors[0].shape[1]
        expended_vocabulary = [[None]] * len(tokenized_vectors)

        kmeans_scores = torch.zeros_like(vk)
        next_kmeans_scores = torch.zeros_like(vk)
        next_kmeans_centroids = [[None]] * len(tokenized_vectors)

        for v, k in enumerate(vk):
            data = tokenized_vectors[v].numpy()
            kmeans = faiss.Kmeans(D, k, gpu=True)
            kmeans.train(data)
            kmeans_scores[v] = kmeans.obj[-1]
            expended_vocabulary[v] = torch.from_numpy(kmeans.centroids)

            next_kmeans = faiss.Kmeans(D, k + 1, gpu=True)
            next_kmeans.train(data)
            next_kmeans_scores[v] = next_kmeans.obj[-1]
            next_kmeans_centroids[v] = torch.from_numpy(kmeans.centroids)
            
        while vk.sum() * D < max_vocab:
            with tqdm(total=max_vocab, desc=f'Processing - {vk.sum() * D} / {max_vocab}') as clustering_proc:
                v_optim = torch.argmax(next_kmeans_scores - kmeans_scores)
                vk[v_optim] += 1
                kmeans_scores[v_optim] = next_kmeans_scores[v_optim]
                expended_vocabulary[v_optim] = next_kmeans_centroids[v_optim]

                new_kmeans = faiss.Kmeans(D, vk[v_optim], gpu=True)
                new_kmeans.train(tokenized_vectors[v_optim].numpy())

                next_kmeans_scores[v_optim] = new_kmeans.obj[-1]
                next_kmeans_centroids[v_optim] = torch.from_numpy(new_kmeans.centroids)

                clustering_proc.update(1)
        
        return torch.cat(expended_vocabulary)

    def execute(self):
        voc_size = self.encoder.tokenizer.vocabulary_size()
        vk = torch.zeros(voc_size)
        vk += 2

        vocab = self.encoder.tokenizer.get_vocabulary()

        
        # TODO: Select documents_per_token documents (just the ID) for each token

        tokenized_vectors = []
        
        for v in vocab.values():
            with tqdm(total=voc_size, desc=f'Initializing {v} vocabulary') as init:
                sampled_docs = []
                for document in self.documents.iter_sample(randint=None)[self.documents_per_token]:
                    sampled_docs.append(document[SimpleTextItem].text)
                tokenized = self.encoder.tokenize(sampled_docs)
                indices_v = torch.nonzero(tokenized['input_ids'] == v, as_tuple=False)
                self.encoder.forward_tokenized(tokenized)
                data_vectors = self.encoder.model.model.last_hidden_state
                v_embeddings = data_vectors[indices_v[:, 0], indices_v[:, 1], :]
                sampled_v_embedding = v_embeddings[torch.randperm(v_embeddings.shape[0])[:self.vectors_per_token], :]
                tokenized_vectors.append(sampled_v_embedding)
                init.update(1)


        expended_projection = self.clustering(vk=vk, tokenized_vectors=tokenized_vectors, max_vocab=self.max_vocab)

        
        # TODO: get the documents_per_token vectors for each token:
        # - process batch of tokens for efficiency
        # - build vectors_per_token clusters for each token
        # - save this into self.linear.weight.data
        self.linear = nn.Linear(self.encoder.dimension, expended_projection.shape[0])
        self.linear.weight.data = expended_projection.t()

        # Save the tensor
        with self.path.open("wb") as fp:
            torch.save(self.linear, fp)


if __name__ == "__main__":
    # Example of use

    # see colbert.py for colbert_encoder
    launcher = find_launcher('duration=2 days & cuda(mem=24G)')
    documents = prepare_collection("irds.msmarco-passage.documents")

    model_id = "colbert-ir/colbertv2.0"
    test = AutoTokenizer.from_pretrained(model_id)
    tokenizer = HFStringTokenizer.from_pretrained_id(model_id)
    encoder = HFTokensEncoder.C(model=ColbertEncoder.from_pretrained_id(model_id))
    text_encoder = TokenizedTextEncoder.C(encoder=encoder, tokenizer=tokenizer)

    output_embeddings_loader = ClusterCompute(encoder=text_encoder, documents=documents, documents_per_token=2000, vectors_per_token=1000).execute()
    
    # ... 
    
    # outputs = learner.submit(launcher=gpu_launcher_learner, pre_tasks=[
    #     output_embeddings_loader
    # ])