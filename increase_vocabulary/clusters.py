import logging
import torch
import h5py
from pathlib import Path
import faiss
from experimaestro import Param, Task, pathgenerator, Annotated, LightweightTask, Config, tqdm, Meta
from xpmir.learning.devices import DEFAULT_DEVICE, Device
import torch.nn as nn
from datamaestro_text.data.ir import DocumentStore, IDItem, SimpleTextItem, TextItem, create_record
from datamaestro.record import Record
from xpmir.text.encoders import (
    TokenizedTextEncoder,
    TokenizedTexts,
    TextsRepresentationOutput,
)
from xpmir.learning import ModuleInitMode
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
from cuda_optimization import CudaMaxAggregation
from xpmir.rankers import Retriever


class LinearModel(Config):
    path: Param[Path]
    """Path to the linear module parameters"""


class HFProjectionLayerLoader(LightweightTask):
    linear: Param[LinearModel]
    model: Param[HFModel]

    def execute(self):
        logging.info("Replacing output embeddings with %s", self.linear.path)
        
        self.model.initialize(ModuleInitMode.DEFAULT.to_options())

        with self.linear.path.open("rb") as fp:
            linear = torch.load(fp)

        self.model.model.set_output_embeddings(linear)


class ClusterCompute(Task):
    encoder: Param[TokenizedTextEncoder[str, TextsRepresentationOutput, TokenizedTexts]]
    """The encoder that transforms documents (as strings) to tensors (B x L x D)"""

    path: Annotated[Path, pathgenerator("words.pt")]
    """Contains the linear layer"""

    vectors_path: Annotated[Path, pathgenerator("vectors.pt")]
    
    vectors_per_token: Param[int]
    """Number of vectors per token"""

    documents_per_token: Param[int]
    """Number of vectors per token (must be greater than vectors_per_token)"""
    
    documents: Param[DocumentStore]
    """Documents"""

    max_vocab: Param[float]
    """Maximum number of vocabulary"""

    device: Meta[Device] = DEFAULT_DEVICE
    """The device(s) to be used for the model"""

    retriver: Param[Retriever]
    """Retriever to search documents contain a given token"""

    def task_outputs(self, dep) -> LinearModel:
        return dep(LinearModel(path=self.path))

    def batch_sampling(self, docs: List[str], batch_size: int, vocab: int):
        n_batch = len(docs) // batch_size
        batch_embeddings = []
        for i in range(n_batch):   # transform the texts to vectors by batches
            batch_docs = docs[i * batch_size : (i + 1) * batch_size]
            tokenized = self.encoder.tokenize(inputs=batch_docs, options=TokenizerOptions(max_length=256))
            indices_v = torch.nonzero(tokenized.ids == vocab, as_tuple=False)
            self.encoder.forward_tokenized(tokenized)
            data_vectors = self.encoder.encoder.model.model.last_hidden_state
            batch_embeddings.append(data_vectors[indices_v[:, 0], indices_v[:, 1], :].cpu())
            del tokenized, indices_v, data_vectors

        batch_docs = docs[n_batch * batch_size :]
        if len(batch_docs) != 0:
            tokenized = self.encoder.tokenize(inputs=batch_docs, options=TokenizerOptions(max_length=256))
            indices_v = torch.nonzero(tokenized.ids == vocab, as_tuple=False)
            self.encoder.forward_tokenized(tokenized)
            data_vectors = self.encoder.encoder.model.model.last_hidden_state
            batch_embeddings.append(data_vectors[indices_v[:, 0], indices_v[:, 1], :].cpu())
            del tokenized, indices_v, data_vectors
        
        return torch.cat(batch_embeddings)
    
    def kmeans(self, vectors: torch.Tensor, k: int):
        D = vectors.shape[1]
        kmeans = faiss.Kmeans(d=D, k=k, gpu=True)
        kmeans.train(vectors.cpu().numpy())

        return kmeans.obj[-1], torch.from_numpy(kmeans.centroids)

    @torch.no_grad()
    def execute(self):
        self.encoder.initialize(ModuleInitMode.DEFAULT.to_options())
        self.encoder.to(self.device.value)
        voc_size = self.encoder.tokenizer.vocabulary_size()
        vk = torch.zeros(voc_size, dtype=torch.int)
        vk += 1
        vocab = self.encoder.tokenizer.get_vocabulary()
        vector_mask = torch.zeros(voc_size, dtype=torch.int)

        kmeans_scores = torch.zeros(voc_size)
        expended_vocabulary = [None] * voc_size
        next_kmeans_scores = torch.zeros(voc_size)
        next_kmeans_centroids = [None] * voc_size
        
        # TODO: Select documents_per_token documents (just the ID) for each token

        with h5py.File('vectors_data.h5', 'w') as h5f:
            data_set = h5f.create_dataset('tensors', shape=(voc_size, self.vectors_per_token, 768), dtype='float16')

            BATCH_SIZE = 64
            with tqdm(total=voc_size, desc=f'Initializing vocabulary') as init:
                for term, vocab_id in vocab.items():
                    record = create_record(text=term)
                    if "[unused" in term:   # pass special terms
                        continue
                    elif "##" in term:
                        term = term[2:]
                    retrieved_docs = self.retriver.retrieve(record=record)   # get relevant docs by anserini
                    if len(retrieved_docs) == 0:
                        continue
                    docs =  [self.documents.document_ext(doc.document[IDItem].id)[TextItem].text for doc in retrieved_docs]
                    vectors = self.batch_sampling(docs=docs, batch_size=BATCH_SIZE, vocab=vocab_id)    
                    n_vectors = vectors.shape[0]
                    del docs, retrieved_docs
                    vector_mask[vocab_id] = n_vectors
                    if n_vectors == 0:
                        continue
                    elif n_vectors < self.vectors_per_token:
                        vectors = nn.functional.pad(vectors, (0, 0, 0, self.vectors_per_token - vector_mask[vocab_id]), "constant", 0)
                    elif n_vectors > self.vectors_per_token:
                        vector_mask[vocab_id] = self.vectors_per_token
                        vectors = vectors[torch.randperm(n_vectors)[:self.vectors_per_token], :]
                    
                    data_set[vocab_id] = vectors.cpu().numpy()

                    if vector_mask[vocab_id] == 0:
                        kmeans_scores[vocab_id] = 0
                        next_kmeans_scores[vocab_id] = torch.inf
                        vk[vocab_id] = 0

                    elif vector_mask[vocab_id] < 39:  # the minimum amount of data for one cluster in faiss is 39
                        kmeans_scores[vocab_id] = 0
                        expended_vocabulary[vocab_id] = vectors.mean(dim=0).unsqueeze(0)
                        next_kmeans_scores[vocab_id] = torch.inf

                    elif 39 <= vector_mask[vocab_id] < 78:
                        kmeans_scores[vocab_id], expended_vocabulary[vocab_id] = self.kmeans(vectors=vectors[:vector_mask[vocab_id], :], k=1)
                        next_kmeans_scores[vocab_id] = torch.inf
                    
                    else:
                        kmeans_scores[vocab_id], expended_vocabulary[vocab_id] = self.kmeans(vectors=vectors[:vector_mask[vocab_id], :], k=1)
                        next_kmeans_scores[vocab_id], next_kmeans_centroids[vocab_id] = self.kmeans(vectors=vectors[:vector_mask[vocab_id], :], k=2)
                    init.update(1)
                    del vectors

        with open('vector_mask.pt', 'wb') as file:
            torch.save(vector_mask, file)

        target_vocab=int(self.max_vocab * voc_size)
        with tqdm(total=target_vocab - int(vk.sum()), desc=f'Processing - {vk.sum()} => {target_vocab}') as clustering_proc:
            while vk.sum() < target_vocab:
                v_optim = torch.argmax(kmeans_scores - next_kmeans_scores)
                vk[v_optim] += 1
                kmeans_scores[v_optim] = next_kmeans_scores[v_optim]
                expended_vocabulary[v_optim] = next_kmeans_centroids[v_optim]

                with h5py.File('vectors_data.h5', 'r') as h5f:
                    vectors = torch.from_numpy(h5f['tensors'][v_optim])
                if vector_mask[v_optim] < (vk[v_optim] + 1) * 39:
                    next_kmeans_scores[v_optim] = torch.inf
                else:
                    next_kmeans_scores[v_optim], next_kmeans_centroids[v_optim] = self.kmeans(vectors=vectors[:vector_mask[v_optim], :], k=vk[v_optim] + 1)

                clustering_proc.update(1)
                del vectors



        expended_projection = torch.cat(list(map(lambda x: x.unsqueeze(0) if len(x.shape) == 1 else x, filter(lambda x: x is not None, expended_vocabulary))))

        print(vk.sum())
        print(expended_projection.shape)
        
        # TODO: get the documents_per_token vectors for each token:
        # - process batch of tokens for efficiency
        # - build vectors_per_token clusters for each token
        # - save this into self.linear.weight.data
        self.linear = nn.Linear(768, expended_projection.shape[0])
        self.linear.weight.data = expended_projection

        # Save the tensor
        with self.path.open("wb") as fp:
            torch.save(self.linear, fp)


        print(f'test on cluster : {torch.count_nonzero(self.linear.weight) / self.linear.weight.numel()}')
