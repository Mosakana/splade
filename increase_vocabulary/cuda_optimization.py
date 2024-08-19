from xpmir.neural.splade import SpladeTextEncoderV2, Aggregation
from xpmir.text.huggingface.base import HFMaskedLanguageModel
from xpmir.text import TokenizedTexts, TokenizerOptions
from xpmir.learning import ModuleInitOptions
from xpmir.text.encoders import (
    InputType as EncoderInputType,
    TextsRepresentationOutput,
)
from optim_forward_without_iteration import OptimReluMaxLinear
import torch
import torch.nn as nn
from experimaestro import Param


class CudaAggregation(Aggregation):
    """The aggregation function for Splade"""

    def get_output_module(self, linear: nn.Module) -> nn.Module:
        return CudaAggregationModule(linear, self)
    
class CudaAggregationModule(nn.Module):
    def __init__(self, linear: nn.Linear, aggregation: CudaAggregation):
        super().__init__()
        self.linear = linear
        self.aggregation = aggregation

    def forward(self, input: torch.Tensor, mask: torch.Tensor):
        return self.aggregation(input, self.linear, mask)


class CudaMaxAggregation(CudaAggregation):
    """Aggregate using a max"""

    def __call__(self, input, output_embeddings, mask):
        # Get the maximum (masking the values)
        values = OptimReluMaxLinear.apply(input, output_embeddings.weight.t(), output_embeddings.bias, mask.to(input.device))

        # Computes log(1+x)
        return torch.log1p(values.clamp(min=0))
    
class SpladeTextEncoderV2Cuda(SpladeTextEncoderV2):
    k: Param[int]
    
    epsilon: Param[float]

    def __initialize__(self, options: ModuleInitOptions):
        self.encoder.initialize(options)
        self.tokenizer.initialize(options)

        # Adds the aggregation head right away - this could allows
        # optimization e.g. for the Max aggregation method
        output_embeddings = self.encoder.model.get_output_embeddings()
        assert isinstance(
            output_embeddings, nn.Linear
        ), f"Cannot handle output embeddings of class {output_embeddings.__cls__}"

        w = output_embeddings.weight
        b = output_embeddings.bias
        noise = torch.norm(w) / w.numel() * self.epsilon

        w_expanded = w.repeat(self.k, 1)
        b_expanded = b.repeat(self.k)

        w_expanded += torch.randn_like(w_expanded) * noise
        b_expanded += torch.randn_like(b_expanded) * noise

        with torch.no_grad():
            output_embeddings.weight = nn.Parameter(w_expanded, requires_grad=True)
            output_embeddings.bias = nn.Parameter(b_expanded, requires_grad=True)

        del w_expanded
        del b_expanded
        torch.cuda.empty_cache()
        
        self.encoder.model.set_output_embeddings(nn.Identity())

        self.aggregation = self.aggregation.get_output_module(output_embeddings)

    def forward(self, texts: EncoderInputType) -> TextsRepresentationOutput:
        """Returns a batch x vocab tensor"""
        tokenized = self.tokenizer.tokenize(
            texts, options=TokenizerOptions(self.maxlen)
        )

        value = self.aggregation(self.encoder(tokenized).logits, tokenized.mask)
        return TextsRepresentationOutput(value, tokenized)

    
    
    