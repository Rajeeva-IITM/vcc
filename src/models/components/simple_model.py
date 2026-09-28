import torch
from torch import nn

from src.models.components.basic_vcc_model import ProcessingNN


class CellModelKernelDelta(nn.Module):
    """Direct pure-shift model: ``y = relu(x_0 + Delta(ko))``.

    The non-flow counterpart to ``flow_model.FlowCellModel``. Instead of regressing a velocity
    and integrating it, it predicts the per-perturbation delta in one shot and adds it to the
    control. This is the ``CellModelSimple`` idea (``x_0 + effect``) with two deliberate
    choices that matter under the challenge's unpaired data:

    * **The delta depends on ``ko`` ONLY, not the individual control cell.** Control and
      perturbed cells are randomly paired, so a delta that saw ``x_0`` would let MSE cancel it
      and collapse to the marginal mean ``mu_pert(ko)`` (losing all cell variance). Constrained
      to ``ko``, the MSE optimum is instead the mean *shift* ``Delta(ko) = mu_pert - mu_ctrl``
      applied to every cell, so the control distribution's spread is preserved and the mean
      lands correctly. It is also robust to control noise: per-cell noise averages out of the
      scored pseudobulk, whereas a control-dependent delta would be attenuated by it.
    * **``Delta`` comes from a smooth, bounded conditioner** (a ``KernelFiLM`` emitting the
      full gene delta, ``output_size = num_genes``). Its convex-hull bound then constrains the
      DE magnitude *directly* -- the anti-overshoot property landing on the exact scored
      quantity -- and its smoothness makes held-out genes interpolate rather than collapse.

    No Euler integration, so no discretisation error and no off-manifold drift; and because
    the model returns expression directly, ``score_local`` predicts it through its ``else``
    (non-``sample``) branch unchanged.

    Parameters
    ----------
    conditioner : nn.Module
        Maps the perturbation embedding ``ko_vec`` (shape ``(B, embed_dim)``) to the gene
        delta (shape ``(B, num_genes)``). A :class:`~src.models.components.flow_model.KernelFiLM`
        with ``output_size = num_genes`` is the intended choice.
    """

    def __init__(self, conditioner: nn.Module) -> None:
        super().__init__()
        self.conditioner = conditioner

    def forward(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Predict perturbed expression as control plus a perturbation-only shift.

        Args:
            inputs (dict): Batch with ``"exp_vec"`` (control expression, model space) and
                ``"ko_vec"`` (perturbation embedding).

        Returns:
            torch.Tensor: Predicted perturbed expression, shape ``(B, num_genes)``, clamped
            non-negative (the inputs are ``log1p`` of a non-negative quantity).
        """
        x_0 = inputs["exp_vec"]
        delta = self.conditioner(inputs["ko_vec"])  # depends on ko only
        return (x_0 + delta).relu()


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

        return y_pred.relu()
