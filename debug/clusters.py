import logging
import torch
from pathlib import Path
from experimaestro import Param, Task, pathgenerator, Annotated, LightweightTask, Config
import torch.nn as nn
from datamaestro_text.data.ir import DocumentStore, IDItem
from xpmir.text.encoders import (
    TokenizedTextEncoder,
    TokenizedTexts,
    TextsRepresentationOutput,
)
from xpmir.text.huggingface import HFMaskedLanguageModel


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
    
    def task_outputs(self) -> LinearModel:
        LinearModel
    
    def execute(self):
        voc_size = self.encoder.tokenizer.vocabulary_size()

        # TODO: Select documents_per_token documents (just the ID) for each token
        for document in self.documents.iter_sample():
            doc_id = document[IDItem].id
            ...
        
        # TODO: get the documents_per_token vectors for each token:
        # - process batch of tokens for efficiency
        # - build vectors_per_token clusters for each token
        # - save this into self.linear.weight.data
        self.linear = nn.Linear(self.encoder.dimension, self.vectors_per_token * voc_size)
        ...

        # Save the tensor
        with self.path.open("wb") as fp:
            torch.save(self.linear, fp)


if __name__ == "__main__":
    # Example of use

    # see colbert.py for colbert_encoder
    output_embeddings_loader = ClusterCompute(encoder=colbert_encoder, ...).submit(...)
    
    ... 
    
    outputs = learner.submit(launcher=gpu_launcher_learner, pre_tasks=[
        output_embeddings_loader
    ])