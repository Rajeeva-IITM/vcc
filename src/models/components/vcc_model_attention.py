from typing import Any, Literal

import torch
import torch.nn as nn
from torch.nn import MultiheadAttention

from src.models.components.basic_vcc_model import ProcessingNN


class NormalizedAttention(nn.Module):
    """
    Simple attention module. Does not support varied dimensions quite yet
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()

        self.norm = nn.LayerNorm(embed_dim)
        self.mhattention = MultiheadAttention(embed_dim, num_heads, dropout)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor):
        """
        Forward through the module
        """
        query = self.norm.forward(query)
        value = self.norm.forward(value)
        key = self.norm.forward(key)

        return self.mhattention.forward(
            query=query, key=key, value=value, need_weights=False
        )


class CellModelAttention(nn.Module):
    """
    Virtual Cell Model
    Model that takes in two inputs (control gene expression and gene knockout status ) and
    processes them individually and returns their vectors.
    The vectors are then fused and then processed by an attention module, the attention output is processed by a decoder
    to predict the perturbed gene expression

    By default, skip connections are enabled so please ensure that the outputs of ko_processor,
    exp_processor are the same as the inputs to attention module and decoder.
    """

    def __init__(
        self,
        ko_processor_args: dict[str, Any],
        exp_processor_args: dict[str, Any],
        # concat_processor_args: dict[str, Any],
        decoder_args: dict[str, Any],
        attention_args: dict[str, Any],
        fusion_type: Literal["sum", "product", "cross_attn", "bilinear"],
        # query: str = "ko_exp" #TODO: Must be able to change what is used as keys and values
    ):
        super().__init__()

        # self.query = query

        self.ko_processor = ProcessingNN(**ko_processor_args)
        self.exp_processor = ProcessingNN(**exp_processor_args)
        # self.concat_processor = ProcessingNN(**concat_processor_args)
        self.decoder = ProcessingNN(**decoder_args)
        self.attention = NormalizedAttention(**attention_args)
        self.fusion = fusion_type
        if self.fusion == "bilinear":
            self.bilinear = nn.Bilinear(
                in1_features=ko_processor_args["output_size"],
                in2_features=exp_processor_args["output_size"],
                out_features=attention_args["embed_dim"],
            )

    def forward(self, inputs: dict[str, torch.Tensor]):
        """
        Performs a forward pass through the model using the provided input tensors.
        Args:
            inputs (Dict[str, torch.Tensor]): A dictionary containing input tensors with keys "ko_vec" and "exp_vec".
        Returns:
            torch.Tensor: The output tensor produced by the model.
        """

        ko_vec: torch.Tensor = inputs["ko_vec"]
        exp_vec: torch.Tensor = inputs["exp_vec"]

        ko_processed = self.ko_processor.forward(ko_vec)
        exp_processed = self.exp_processor.forward(exp_vec)

        # Fusing the two representations

        match self.fusion:
            case "sum":
                fused_representaion = (
                    ko_processed + exp_processed
                )  # Ensure they are of same size
                query = fused_representaion.unsqueeze(0)
                key = fused_representaion.unsqueeze(0)
                value = fused_representaion.unsqueeze(0)

            case "product":
                fused_representaion = (
                    ko_processed * exp_processed
                )  # Ensure they are of same size
                query = fused_representaion.unsqueeze(0)
                key = fused_representaion.unsqueeze(0)
                value = fused_representaion.unsqueeze(0)

            case "cross_attn":
                fused_representaion = self.bilinear.forward(ko_processed, exp_processed)
                query = fused_representaion.unsqueeze(0)
                key = fused_representaion.unsqueeze(0)
                value = fused_representaion.unsqueeze(0)

            case "bilinear":
                query = ko_processed.unsqueeze(0)
                key = exp_processed.unsqueeze(0)
                value = exp_processed.unsqueeze(0)
            case _:
                raise ValueError(
                    "fusion_type should be one of ['sum','product','cross_attn']"
                )

        # Moving onto the attention module

        attn_output, _ = self.attention.forward(
            query=query,
            key=key,
            value=value,
        )  # discarding weights

        output = self.decoder.forward(attn_output.squeeze(0))

        return output


if __name__ == "__main__":
    # Testing
    # Example arguments for each processor
    ko_processor_args = {
        "input_size": 10,
        "hidden_layers": [16, 8],
        "output_size": 4,
        "dropout": 0.2,
        "activation": "relu",
        # "no_processing": True
    }
    exp_processor_args = {
        "input_size": 12,
        "hidden_layers": [18, 8],
        "output_size": 4,
        "dropout": 0.2,
        "activation": "relu",
        # "no_processing": True
    }
    attention_args = {"embed_dim": 4, "num_heads": 1}
    decoder_args = {
        "input_size": 4,
        "hidden_layers": [6],
        "output_size": 2,
        "dropout": 0.2,
        "activation": "relu",
        # "no_processing": True
    }

    # Instantiate the model
    model = CellModelAttention(
        ko_processor_args=ko_processor_args,
        exp_processor_args=exp_processor_args,
        attention_args=attention_args,
        decoder_args=decoder_args,
        fusion_type="sum",
    )

    # Create dummy input tensors
    ko_vec = torch.randn(5, 10)  # batch size 5, input size 10
    exp_vec = torch.randn(5, 12)  # batch size 5, input size 12

    # Forward pass
    output = model({"ko_vec": ko_vec, "exp_vec": exp_vec})
    print("Output shape:", output.shape)
    print("Output:", output)
    print(model)
