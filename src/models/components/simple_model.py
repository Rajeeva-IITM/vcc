import torch
from torch import nn

from src.models.components.basic_vcc_model import ProcessingNN


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


class CellModelFiLMConditioned(nn.Module):
    """
    A model where change in expression is conditioned on perturbation

    y_pred = x + gamma(p) * L(x) + beta(p)

    where gamma, beta, L are small ProcessingNN
    p -> perturbation embedding
    x -> control expression

    """

    def __init__(
        self,
        num_genes: int,
        perturbation_embed_dim: int,
        film_mlp: ProcessingNN,
        control_mlp: ProcessingNN,
    ) -> None:
        super().__init__()

        self.num_genes = num_genes
        self.perturbation_embed_dim = perturbation_embed_dim
        self.film_mlp = film_mlp
        self.control_mlp = control_mlp

        assert film_mlp.input_size == perturbation_embed_dim, (
            "The input of film mlp must be same as the perturbation dim"
        )
        assert film_mlp.output_size == 2 * num_genes, (
            "The output of film mlp must be twice the number of genes"
        )
        assert control_mlp.output_size == control_mlp.input_size == num_genes, (
            "Mismatch between control mlp output to number of genes"
        )

    def forward(self, inputs: dict[str, torch.Tensor]):
        """
        Performs a forward pass through the model using the provided input tensors.
        Args:
            inputs (Dict[str, torch.Tensor]): A dictionary containing input tensors with keys "ko_vec" and "exp_vec".
        Returns:
            torch.Tensor: The output tensor produced by the model.
        """

        ko_vec = inputs["ko_vec"]  # Perturbation embedding
        exp_vec = inputs["exp_vec"]  # Control expression

        # Film processing

        gamma_p, beta_p = self.film_mlp(ko_vec).chunk(2, -1)

        L_x: torch.Tensor = self.control_mlp(exp_vec)

        y_pred: torch.Tensor = exp_vec + (gamma_p * L_x) + beta_p

        return y_pred
