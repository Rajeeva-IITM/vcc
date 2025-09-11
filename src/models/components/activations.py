import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLU(nn.Module):
    """
    Implements the SwiGLU activation function. Defined as:

    SwiGLU(x) = W1@x * Swish(W2@x)
    """

    def __init__(self, input_dim: int) -> None:
        super().__init__()

        # Create a Linear map to transform the inputs - dim -> 2*dim
        self.linear_map = nn.Linear(in_features=input_dim, out_features=(2 * input_dim))

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        value, gate_input = self.linear_map(X).chunk(
            chunks=2, dim=-1
        )  # output is two tensors of b x d

        calc = value * F.silu(gate_input)

        return calc
