import json
import os
from dataclasses import InitVar
from typing import NamedTuple, Union

import torch
import torch.nn as nn
from transformers import AutoConfig, PreTrainedModel
from transformers.models.bert import BertModel
from transformers.utils import cached_file
from xpmir.learning import ModuleInitMode
from xpmir.text import TokenizedTextEncoder
from xpmir.text.huggingface import HFModel

# Look at https://github.com/stanford-futuredata/ColBERT/blob/main/colbert/modeling/hf_colbert.py
# to support more ColBERT architectures


HFConfigName = Union[str, os.PathLike]


class ColbertConfig(NamedTuple):
    """ColBERT configuration when loading a pre-trained ColBERT model"""

    dim: int
    query_maxlen: int
    similarity: str
    attend_to_mask_tokens: bool
    data: dict

    @staticmethod
    def from_pretrained(pretrained_model_name_or_path: HFConfigName):
        resolved_config_file = cached_file(
            pretrained_model_name_or_path, "artifact.metadata"
        )
        config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
        with open(resolved_config_file, "rt") as fp:
            data = json.load(fp)
            kwargs = {key: data[key] for key in ColbertConfig._fields if key != "data"}
            config.colbert = ColbertConfig(**kwargs, data=data)
        return config


class ColbertModel(PreTrainedModel):
    """ColBERT model"""

    DEFAULT_OUTPUT_SIZE = 128

    last_hidden_state: torch.Tensor

    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        self.linear = nn.Linear(config.hidden_size, config.colbert.dim, bias=False)
        self.last_hidden_state = torch.Tensor([])
        
    def forward(self, ids, **kwargs):
        output = self.bert(ids, **kwargs)
        self.last_hidden_state = output.last_hidden_state
        output.last_hidden_state = self.linear(output.last_hidden_state)
        return output


class ColbertEncoder(HFModel):
    model: InitVar[ColbertModel]

    automodel = ColbertModel
    autoconfig = ColbertConfig


# if __name__ == "__main__":
#     # Example of use
#     import logging
#     logging.basicConfig(level=logging.INFO)
#     from xpmir.text.huggingface import HFStringTokenizer, HFTokensEncoder

#     model_id = "colbert-ir/colbertv2.0"
#     tokenizer = HFStringTokenizer.from_pretrained_id(model_id)
#     encoder = HFTokensEncoder.C(model=ColbertEncoder.from_pretrained_id(model_id))
#     text_encoder = TokenizedTextEncoder.C(encoder=encoder, tokenizer=tokenizer)

#     o_text_encoder = text_encoder.instance()
#     o_text_encoder.initialize(ModuleInitMode.DEFAULT.to_options())
    
#     # This should be a 2 x 4 x 128 tensor
#     print(o_text_encoder(["hello world", "another document"]).value.shape)
#     print(o_text_encoder.encoder.model.model.last_hidden_state.shape)

