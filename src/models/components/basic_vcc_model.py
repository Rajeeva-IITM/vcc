from typing import Any, Literal

import torch
from torch import nn

from src.utils.process_activation_function import get_activation

# Contains
# Knockdown information vector processing
# Gene expression vector processing
# Encoder of concatenated parts
# Decoder of encoded parts


class ProcessingNN(
    nn.Module
):  # TODO: Might have to move to this to the components module
    """Build the module for processing vectors

    Parameters
    ----------
    input_size : int
        Input size
    hidden_layers: List[int]
        Sizes of the hidden layers
    output_size : int
        Output size
    dropout : float
        Dropout value
    activation : str | nn.Module
        Activation function
    no_processing: bool, defaults to False
        Flag to indicate if one wants this component to do nothing
    residual_connection: bool, defaults to True
        Whether to include residual connections in the model.
    """

    def __init__(
        self,
        input_size: int,
        hidden_layers: list[int],
        output_size: int,  # Embedding size
        dropout: float,
        activation: str | nn.Module,
        no_processing: bool = False,
        residual_connection: bool = True,
    ):
        # if len(hidden_layers) < 1:
        #     raise ValueError("Number of layers must be greater than 1.")

        if dropout < 0 or dropout > 1:
            raise ValueError("Dropout must be between 0 and 1.")

        if input_size < 1 or output_size < 1:
            raise ValueError("Input and output sizes must be greater than 1.")

        super().__init__()

        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.output_size = output_size
        self.dropout = dropout
        self.activation = (
            get_activation(activation) if isinstance(activation, str) else activation
        )
        self.residual_connection = residual_connection

        self.projection = nn.Linear(self.input_size, self.output_size)

        self.layers = [self.output_size] + self.hidden_layers + [self.output_size]  # type: ignore
        self.sequence = []  # Sequence of layers

        if no_processing:
            self.sequence.append(nn.Identity())

        else:
            if len(self.layers) == 2:  # Makes two layers
                # No hidden layers: just input -> output
                self.sequence.append(nn.LayerNorm(self.layers[0]))  # pre-normalization
                self.sequence.append(
                    nn.Linear(self.layers[0], self.layers[1], dtype=torch.float)
                )
                self.sequence.append(self.activation)
            else:
                self.sequence.append(nn.LayerNorm(self.layers[0]))  # pre-normalization

                for i in range(len(self.layers) - 2):
                    self.sequence.append(
                        nn.Linear(self.layers[i], self.layers[i + 1], dtype=torch.float)
                    )
                    self.sequence.append(self.activation)

                self.sequence.append(nn.Dropout(self.dropout))
                self.sequence.append(
                    nn.Linear(self.layers[-2], self.layers[-1], dtype=torch.float)
                )
                self.sequence.append(self.activation)

        self.sequence = nn.Sequential(*self.sequence)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """This function takes in an input tensor `x` and passes it through the neural network's
        sequence of layers. It then returns the output tensor produced by the forward pass.

        Parameters:
        -----------
        x : tensor
            The input tensor to be passed through the neural network.

        Returns:
        --------
        tensor
            The output tensor produced by the forward pass through the network's layers.
        """

        input = self.projection(x)

        if self.residual_connection:
            return self.sequence(input) + input  # type: ignore
        else:
            return self.sequence(input)


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
                out_features=fused_processor_args["embed_dim"],
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

            case "product":
                fused_representaion = (
                    ko_processed * exp_processed
                )  # Ensure they are of same size

            case "concat":
                fused_representaion = torch.cat([ko_processed, exp_processed], dim=1)

            case "bilinear":
                fused_representaion = self.bilinear.forward(ko_processed, exp_processed)

            case _:
                raise ValueError(
                    "fusion_type should be one of ['sum','product','cross_attn', 'bilinear']"
                )

        latent = self.concat_processor.forward(fused_representaion)

        output = self.decoder.forward(latent)

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
        "no_processing": True,
    }
    exp_processor_args = {
        "input_size": 12,
        "hidden_layers": [18, 8],
        "output_size": 4,
        "dropout": 0.2,
        "activation": "relu",
        "no_processing": True,
    }
    concat_processor_args = {
        "input_size": 22,  # 4 + 4 from previous outputs
        "hidden_layers": [8],
        "output_size": 6,
        "dropout": 0.2,
        "activation": "relu",
        "no_processing": True,
    }
    decoder_args = {
        "input_size": 22,
        "hidden_layers": [6],
        "output_size": 2,
        "dropout": 0.2,
        "activation": "relu",
        "no_processing": True,
    }

    # Instantiate the model
    model = CellModel(
        ko_processor_args=ko_processor_args,
        exp_processor_args=exp_processor_args,
        concat_processor_args=concat_processor_args,
        decoder_args=decoder_args,
    )

    # Create dummy input tensors
    ko_vec = torch.randn(5, 10)  # batch size 5, input size 10
    exp_vec = torch.randn(5, 12)  # batch size 5, input size 12

    # Forward pass
    output = model({"ko_vec": ko_vec, "exp_vec": exp_vec})
    print("Output shape:", output.shape)
    print("Output:", output)
    print(model)
