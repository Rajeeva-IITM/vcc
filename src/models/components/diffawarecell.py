from typing import Any, Literal

import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch import nn
from torch.nn import MultiheadAttention

from src.models.components.basic_vcc_model import ProcessingNN
from src.utils.process_activation_function import get_activation


class FactorizedDecoder(nn.Module):
    """
        Factorized decoder that predicts perturbed gene expression as:
        y_pred = mu(control) + beta * effect_of_perturbation

    - mu: baseline expression (control reconstruction)
    - effect: learned perturbation effect vector
    - beta: gating mask over genes (DEG probabilities)
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_hidden_layers: int,
        output_size: int,  # Embedding size
        dropout: float,
        activation: str | nn.Module,
        residual_connection: bool = True,
        ensure_output_positive: bool = False,
    ) -> None:
        if dropout < 0 or dropout > 1:
            raise ValueError("Dropout must be between 0 and 1.")

        if input_size < 1 or output_size < 1:
            raise ValueError("Input and output sizes must be greater than 1.")
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.hidden_layers = [hidden_size for _ in range(num_hidden_layers)]
        self.output_size = output_size
        self.dropout = dropout
        self.activation = (
            get_activation(activation) if isinstance(activation, str) else activation
        )
        self.ensure_output_positive = ensure_output_positive
        self.residual_connection = residual_connection

        self.sequence = []

        # Shared backbone
        for _ in range(num_hidden_layers):
            self.sequence.append(
                nn.Linear(
                    self.hidden_size,
                    self.hidden_size,
                )
            )
            self.sequence.append(nn.LayerNorm(self.hidden_size))

            self.sequence.append(self.activation)

        self.sequence.append(nn.Dropout(self.dropout))

        self.input_projection = nn.Linear(self.input_size, self.hidden_size)  # Step 1
        self.output_projection = nn.Linear(self.hidden_size, self.output_size)  # Step 3

        if self.ensure_output_positive:
            self.positive_enforcer = nn.ReLU()  # Step 4 # maybe
        else:
            self.positive_enforcer = nn.Identity()

        self.process_sequence = nn.Sequential(*self.sequence)  # Step 2

        # Separate heads
        self.mu_head = nn.Linear(hidden_size, output_size)  # baseline
        self.effect_head = nn.Linear(hidden_size, output_size)  # perturbation effect
        self.beta_head = nn.Sequential(  # DEG mask
            nn.Linear(hidden_size, output_size), nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor, control_exp: torch.Tensor) -> torch.Tensor:
        projection = self.input_projection(x)
        processed = self.process_sequence(projection)  # Processed input
        if self.residual_connection:
            processed = processed + projection

        mu: torch.Tensor = self.mu_head(processed)
        effect: torch.Tensor = self.effect_head(processed)
        beta: torch.Tensor = self.beta_head(processed)

        y_pred: torch.Tensor = self.positive_enforcer(mu + beta * effect)

        return y_pred


class GatedProcessingNN(ProcessingNN):
    """
    This variant of the ProcessingNN is to be mainly used as a decoder
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Gate head: predicts DEG probability for each gene
        self.gate_projection = nn.Sequential(
            nn.Linear(self.hidden_size, self.output_size), nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor, control_exp: torch.Tensor) -> torch.Tensor:
        if self.no_processing:
            processed: torch.Tensor = self.process_sequence(x)
        else:
            projection = self.input_projection(x)
            processed = self.process_sequence(projection)
            if self.residual_connection:
                processed = processed + projection

        # Raw predicted expression delta
        raw_pred = self.output_projection(processed)
        raw_pred = self.positive_enforcer(raw_pred)

        # DEG gate
        gate = self.gate_projection(processed)

        # Gated output: deviations only where gate allows
        y_pred: torch.Tensor = control_exp + gate * (raw_pred - control_exp)

        return y_pred


class DiffAwareCellModelAttention(nn.Module):
    """
    Virtual Cell Model
    Model that takes in two inputs (control gene expression and gene knockout status ) and
    processes them individually and returns their vectors.
    The vectors are then fused and then processed by an attention module, the attention output is processed by a decoder
    to predict the perturbed gene expression. This version learns differential expression.

    By default, skip connections are enabled so please ensure that the outputs of ko_processor,
    exp_processor are the same as the inputs to attention module and decoder. Don't enforce
    positivity for decoder for this model
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

        # assert decoder_args['ensure_output_positive'] == False, "Decoder outputs only positive values, change it"

        self.ko_processor = ProcessingNN(**ko_processor_args)
        self.exp_processor = ProcessingNN(**exp_processor_args)
        # self.concat_processor = ProcessingNN(**concat_processor_args)
        self.decoder = GatedProcessingNN(**decoder_args)
        self.attention = MultiheadAttention(**attention_args)
        self.fusion = fusion_type
        if self.fusion == "bilinear":
            self.bilinear = nn.Bilinear(
                in1_features=ko_processor_args["output_size"],
                in2_features=exp_processor_args["output_size"],
                out_features=attention_args["embed_dim"],
            )
        self.feature_proj_in = nn.Linear(
            1, attention_args["embed_dim"]
        )  # Projecting of processor out to attn_in
        self.feature_proj_out = nn.Linear(
            attention_args["embed_dim"], 1
        )  # Projection of attn_out to decoder_in

        self.last_attn_weights = torch.Tensor()

    def forward(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Performs a forward pass through the model using the provided input tensors.
        Args:
            inputs (Dict[str, torch.Tensor]): A dictionary containing input tensors with keys "ko_vec" and "exp_vec".
        Returns:
            torch.Tensor: The output tensor produced by the model.
        """

        ko_vec: torch.Tensor = inputs["ko_vec"]  # perturbation embedding
        exp_vec: torch.Tensor = inputs["exp_vec"]  # control expression

        ko_processed = self.ko_processor.forward(ko_vec)
        exp_processed = self.exp_processor.forward(exp_vec)

        # Fusing the two representations

        match self.fusion:
            case "sum":
                fused = ko_processed + exp_processed  # Ensure they are of same size
                query, key, value = fused, fused, fused

            case "product":
                fused = ko_processed * exp_processed  # Ensure they are of same size
                query, key, value = fused, fused, fused

            case "cross_attn":
                query, key, value = ko_processed, exp_processed, exp_processed

            case "bilinear":
                fused = self.bilinear.forward(ko_processed, exp_processed)
                query, key, value = fused, fused, fused
            case _:
                raise ValueError(
                    "fusion_type should be one of ['sum','product','cross_attn', 'bilinear']"
                )

        # changing dimensions from [batch, hidden_dim] -> [hidden_dim, batch, 1]
        # Attention expects the input to be of the form [seq_length, batch_size, embedding_dim]
        # Attention is then calculated for each `token` along the seq_length. This is the context length in LLMs
        query = query.unsqueeze(-1).transpose(0, 1)
        query = self.feature_proj_in(query)
        key = key.unsqueeze(-1).transpose(0, 1)
        key = self.feature_proj_in(key)
        value = value.unsqueeze(-1).transpose(0, 1)
        value = self.feature_proj_in(value)

        # Moving onto the attention module
        attn_output, attn_weights = self.attention(
            query=query,
            key=key,
            value=value,
        )
        self.last_attn_weights: torch.Tensor = attn_weights.detach().cpu()

        # changing dimensions from [hidden_dim, batch_size, embedding_dim] -> [batch, hidden_dim]
        attn_output: torch.Tensor = self.feature_proj_out(attn_output)
        attn_output = attn_output.transpose(0, 1).squeeze(-1)
        output: torch.Tensor = self.decoder(attn_output, exp_vec)  # raw expression

        return output

    def visualize_attention(
        self,
        sample_idx: int = 0,
        head: int = 0,
        avg_heads: bool = False,
        cmap: str = "viridis",
    ):
        """
        Visualize the attention map for a given sample.

        Args:
            sample_idx (int): which sample in the batch to visualize.
            head (int): which attention head to visualize (ignored if avg_heads=True).
            avg_heads (bool): if True, average across all heads.
            cmap (str): colormap for heatmap.
        """
        if self.last_attn_weights is None:
            raise ValueError("No attention weights stored. Run a forward pass first.")

        attn = self.last_attn_weights[sample_idx]  # [num_heads, d_model, d_model]

        if avg_heads:
            attn_matrix = attn.mean(0)  # average across heads
            title = f"Sample {sample_idx} | Avg over {attn.shape[0]} heads"
        else:
            attn_matrix = attn[head]
            title = f"Sample {sample_idx} | Head {head}"

        plt.figure(figsize=(6, 5))
        sns.heatmap(attn_matrix, cmap=cmap, cbar=True)
        plt.title(title)
        plt.xlabel("Key (features)")
        plt.ylabel("Query (features)")
        plt.tight_layout()
        plt.show()


class CellModel(nn.Module):
    """
    Virtual Cell Model
    Model that takes in two inputs processes them individually and returns their vectors.
    The vectors are then concatenated and processed by an encoder.


    """

    def __init__(
        self,
        ko_processor_args: dict[str, Any],
        exp_processor_args: dict[str, Any],
        fused_processor_args: dict[str, Any],
        decoder_args: dict[str, Any],
        fusion_type: Literal["sum", "product", "concat", "bilinear"],
    ):
        # TODO: Need to complete input verification code

        # assert (
        #     fused_processor_args["input_size"]
        #     == ko_processor_args["output_size"] + exp_processor_args["output_size"]
        # ), (
        #     "Mismatch in size between concatenated vector and input of concatenated processor"
        # )

        # assert fused_processor_args["output_size"] == decoder_args["input_size"], (
        #     "Encoder output size should be the same as the decoder input size"
        # )

        super().__init__()

        self.ko_processor = ProcessingNN(**ko_processor_args)
        self.exp_processor = ProcessingNN(**exp_processor_args)
        self.concat_processor = ProcessingNN(**fused_processor_args)
        self.decoder = ProcessingNN(**decoder_args)
        self.fusion = fusion_type
        if self.fusion == "bilinear":
            self.bilinear = nn.Bilinear(
                in1_features=ko_processor_args["output_size"],
                in2_features=exp_processor_args["output_size"],
                out_features=fused_processor_args["output_size"],
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

        ko_processed = self.ko_processor(ko_vec)
        exp_processed = self.exp_processor(exp_vec)

        # Fusing the two representations
        match self.fusion:
            case "sum":
                fused_representaion = (
                    ko_processed + exp_processed
                )  # Ensure they are of same size

            case "product":
                fused_representaion = (
                    ko_processed * exp_processed
                )  # Ensure they are of same size

            case "concat":
                fused_representaion = torch.cat([ko_processed, exp_processed], dim=1)

            case "bilinear":
                fused_representaion = self.bilinear(ko_processed, exp_processed)

            case _:
                raise ValueError(
                    "fusion_type should be one of ['sum','product','cross_attn', 'bilinear']"
                )

        latent = self.concat_processor(fused_representaion)

        output = self.decoder(latent)

        return output
