"""Regression guards for the loss functions.

Two loss terms in this repo silently contributed zero gradient for a long time:
``PerturbationSimilarityLoss`` (torchmetrics computes Spearman ranks via
``argsort``, so its output had no ``grad_fn`` at all) and a ``torch.sign``-based
direction term inside ``DiffExpAwareMSELoss`` (piecewise constant). Both carried
non-zero weights in the configs while doing nothing.

These tests assert that every loss reachable from a config is actually
differentiable, and cover the other silent-failure modes found alongside them.
"""

import pytest
import torch
from torch import nn

from src.models.components import loss_functions as lf

BATCH, GENES, EMBED = 16, 64, 8


@pytest.fixture
def batch():
    """A synthetic batch with everything the configured losses read."""
    torch.manual_seed(0)
    return {
        "y_pred": torch.rand(BATCH, GENES, requires_grad=True),
        "y_true": torch.rand(BATCH, GENES),
        "control_exp": torch.rand(BATCH, GENES),
        # Signed embeddings: the GO/Poincare files have negative cosines, which
        # the contrastive loss needs in order to have anything to push apart.
        "gene_embeddings": torch.randn(BATCH, EMBED),
    }


# Every loss referenced by a file in config/model/.
CONFIGURED_LOSSES = [
    lf.MyMSELoss(),
    lf.GenewiseMSELoss(),
    lf.DiffExpAwareMSELoss(temperature=0.25, threshold=0.4),
    lf.DiffGeneBCELoss(pos_weight=10, temperature=0.15, threshold=0.4),
    lf.SoftDiceLoss(temperature=0.25, threshold=0.4),
    lf.SoftJaccardLoss(temperature=0.25, threshold=0.4),
    lf.HybridGeneLoss(alpha=3, temperature=0.1, gamma=1),
    lf.WeightedContrastiveLoss(temperature=0.5, alpha=2),
]


@pytest.mark.parametrize("loss_fn", CONFIGURED_LOSSES, ids=lambda f: type(f).__name__)
def test_loss_is_differentiable(loss_fn, batch):
    """A loss with no gradient path is a no-op dressed up as an objective."""
    out = loss_fn(**batch)

    assert out.grad_fn is not None, (
        f"{type(loss_fn).__name__} is detached from the graph"
    )

    grad = torch.autograd.grad(out, batch["y_pred"], allow_unused=True)[0]
    assert grad is not None, f"{type(loss_fn).__name__} produced no gradient"
    assert torch.isfinite(grad).all(), (
        f"{type(loss_fn).__name__} produced non-finite gradient"
    )
    assert grad.abs().sum() > 0, (
        f"{type(loss_fn).__name__} gradient is identically zero"
    )


def test_removed_losses_stay_removed():
    """Guard against reintroducing the zero-gradient terms."""
    assert not hasattr(lf, "PerturbationSimilarityLoss")
    assert "beta" not in lf.DiffExpAwareMSELoss().__dict__


def test_composite_loss_registers_children():
    """A plain list hides child losses from .parameters() and .to(device)."""
    composite = lf.CompositeLoss([lf.WeightedMAELoss(GENES), lf.MyMSELoss()], [1, 1])

    assert isinstance(composite.loss_functions, nn.ModuleList)
    assert len(list(composite.parameters())) == 1, (
        "learnable loss weights not registered"
    )


def test_composite_loss_moves_buffers_with_module(batch):
    """DiffGeneBCELoss used to pin pos_weight to a hardcoded device."""
    composite = lf.CompositeLoss([lf.DiffGeneBCELoss(pos_weight=10)], [1])
    composite = composite.to(torch.float64)

    assert composite.loss_functions[0].pos_weight.dtype == torch.float64


def test_contrastive_loss_handles_fractional_alpha(batch):
    """torch.pow on a negative base is NaN for non-integer exponents."""
    out = lf.WeightedContrastiveLoss(alpha=1.5)(**batch)

    assert torch.isfinite(out), "fractional alpha produced NaN/inf"


def test_contrastive_loss_needs_signed_embeddings(batch):
    """All-positive embeddings (e.g. quantiles-train_expression) make it a no-op.

    Documents why config/data/dataset_cp10k.yaml uses the Poincare embedding.
    """
    positive_only = batch | {"gene_embeddings": torch.rand(BATCH, EMBED).abs()}
    out = lf.WeightedContrastiveLoss()(**positive_only)

    assert out.abs().item() == pytest.approx(0.0, abs=1e-6)


def test_pos_weight_stays_out_of_state_dict():
    """pos_weight is a config constant, not learned state.

    Making it a *persistent* buffer would add a `criterion.loss_functions.N.pos_weight`
    key and break strict loading of checkpoints saved before it existed.
    """
    composite = lf.CompositeLoss([lf.DiffGeneBCELoss(pos_weight=10)], [1])

    assert composite.state_dict() == {}
    # still moves with the module, which is the reason it is a buffer at all
    assert (
        composite.to(torch.float64).loss_functions[0].pos_weight.dtype == torch.float64
    )


class TestGateSparsityLoss:
    """The gate is the whole masking mechanism; left free it drifts to 1.0."""

    def test_zero_at_target_rate(self, batch):
        """A gate exactly at the target rate costs nothing."""
        loss = lf.GateSparsityLoss(target_rate=0.023)
        gate = torch.full((BATCH, GENES), 0.023)
        assert loss(
            batch["y_pred"], batch["y_true"], gate=gate
        ).item() == pytest.approx(0.0, abs=1e-6)

    def test_penalises_an_open_gate(self, batch):
        """The failure mode: gate near 1.0 means y = raw_pred, no masking."""
        loss = lf.GateSparsityLoss(target_rate=0.023)
        at_target = loss(
            batch["y_pred"], batch["y_true"], torch.full((BATCH, GENES), 0.023)
        )
        wide_open = loss(
            batch["y_pred"], batch["y_true"], torch.full((BATCH, GENES), 0.95)
        )
        assert wide_open > at_target
        assert wide_open > 1.0

    def test_kl_is_two_sided(self, batch):
        """Unlike L1, a gate that is too *closed* is penalised as well."""
        loss = lf.GateSparsityLoss(target_rate=0.1)
        too_closed = loss(
            batch["y_pred"], batch["y_true"], torch.full((BATCH, GENES), 0.001)
        )
        at_target = loss(
            batch["y_pred"], batch["y_true"], torch.full((BATCH, GENES), 0.1)
        )
        assert too_closed > at_target

    def test_selective_gate_is_not_penalised(self, batch):
        """A few genes fully open and the rest closed must be as cheap as uniform.

        This is the point of constraining the mean rather than each element: it
        leaves the model free to choose *which* genes move.
        """
        loss = lf.GateSparsityLoss(target_rate=0.25)
        uniform = torch.full((BATCH, GENES), 0.25)
        selective = torch.zeros(BATCH, GENES)
        selective[:, : GENES // 4] = 1.0
        assert loss(
            batch["y_pred"], batch["y_true"], selective
        ).item() == pytest.approx(
            loss(batch["y_pred"], batch["y_true"], uniform).item(), abs=1e-4
        )

    def test_gradient_reaches_the_gate(self, batch):
        """The penalty must actually be able to move the gate."""
        gate = torch.full((BATCH, GENES), 0.9, requires_grad=True)
        lf.GateSparsityLoss(target_rate=0.023)(
            batch["y_pred"], batch["y_true"], gate=gate
        ).backward()
        assert gate.grad is not None and gate.grad.abs().sum() > 0

    def test_no_gate_is_a_no_op(self, batch):
        """Nets without a gated decoder pass None; this must stay safe in any
        CompositeLoss rather than raising."""
        out = lf.GateSparsityLoss()(batch["y_pred"], batch["y_true"], gate=None)
        assert out.item() == 0.0
        composite = lf.CompositeLoss([lf.MyMSELoss(), lf.GateSparsityLoss()], [1, 0.01])
        plain = lf.CompositeLoss([lf.MyMSELoss()], [1])
        assert composite(
            batch["y_pred"], batch["y_true"], gate=None
        ).item() == pytest.approx(plain(batch["y_pred"], batch["y_true"]).item())

    def test_l1_mode_is_one_sided(self, batch):
        """L1 mode returns the mean activation."""
        loss = lf.GateSparsityLoss(mode="l1")
        gate = torch.full((BATCH, GENES), 0.4)
        assert loss(batch["y_pred"], batch["y_true"], gate).item() == pytest.approx(0.4)

    def test_rejects_bad_config(self):
        """Invalid target rates and modes fail loudly at construction."""
        with pytest.raises(ValueError):
            lf.GateSparsityLoss(target_rate=0.0)
        with pytest.raises(ValueError):
            lf.GateSparsityLoss(mode="nonsense")
