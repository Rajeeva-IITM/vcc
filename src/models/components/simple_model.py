import torch
from torch import nn


class CellModelSimple(nn.Module):
    """
    Very basic, simple model:

    exp_perturbed = b0 + b1*perturbation_embdding
    """

    def __init__(self, num_genes: int, perturbation_embed_dim: int) -> None:
        super().__init__()

        # self.b1 = nn.Parameter(torch.ones(num_genes))
        self.perturbation_effect = nn.Linear(perturbation_embed_dim, num_genes)

    def forward(self, inputs: dict[str, torch.Tensor]):
        """
        Performs a forward pass through the model using the provided input tensors.
        Args:
            inputs (Dict[str, torch.Tensor]): A dictionary containing input tensors with keys "ko_vec" and "exp_vec".
        Returns:
            torch.Tensor: The output tensor produced by the model.
        """

        ko_vec = inputs["ko_vec"]
        exp_vec = inputs["exp_vec"]

        # control_effect = self.b1 * exp_vec
        perturbation_effect: torch.Tensor = self.perturbation_effect(ko_vec)

        y_pred: torch.Tensor = (
            exp_vec + perturbation_effect
        ).relu()  # Ensure positivity

        return y_pred


class CellModelConditioned(nn.Module):
    """
    A model where change in expression is conditioned on perturbation
    """

    def __init__(
        self,
        num_genes: int,
        perturbation_embed_dim: int,
    ) -> None:
        super().__init__()
