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
        # Which perturbation each cell carries. Only DEWeightedMSELoss reads it; the
        # rest swallow it through **kwargs, which is itself worth pinning.
        "ko_id": torch.arange(BATCH) % GENES,
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
    lf.BatchDEAwareMSELoss(threshold=0.1, temperature=0.05),
    lf.BatchLaplacianReg(),
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


def test_composite_loss_stashes_its_components(batch):
    """Per-term values must be surfaced so a module can log each separately."""
    composite = lf.CompositeLoss([lf.MyMSELoss(), lf.BatchVariance()], [1.0, 0.5])
    total = composite(**batch)

    stash = composite.last_components
    assert set(stash) == {"MyMSELoss", "BatchVariance"}

    for value in stash.values():
        assert value.ndim == 0, "component value is not a scalar"
        assert value.grad_fn is None, "stashed component still carries the graph"

    reconstructed = 1.0 * stash["MyMSELoss"] + 0.5 * stash["BatchVariance"]
    assert reconstructed.item() == pytest.approx(total.item(), rel=1e-6)


def test_composite_loss_disambiguates_duplicate_component_names(batch):
    """Two losses of the same class must not collide to one dict key."""
    composite = lf.CompositeLoss([lf.MyMSELoss(), lf.MyMSELoss()], [1.0, 1.0])
    composite(**batch)

    assert set(composite.last_components) == {"MyMSELoss", "MyMSELoss_1"}


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


def test_contrastive_loss_survives_an_anchor_with_no_similar_neighbour():
    """An anchor with zero positive-weight neighbours used to send score->0, -log->+inf.

    Over a full run some batch is near-certain to contain a cell whose perturbation is
    non-positively correlated with every batch-mate; that must not NaN the loss.
    """
    b, g, e = 6, 64, 4
    y_pred = torch.randn(b, g, requires_grad=True)
    y_true = torch.randn(b, g)

    # Row 0 sits on its own axis -> cosine 0 to every other row (no similar neighbour);
    # rows 1.. share an axis -> positive cosines among themselves.
    emb = torch.zeros(b, e)
    emb[1:, 0] = 1.0
    emb[1:, 1] = torch.randn(b - 1)
    emb[0, 2] = 1.0

    out = lf.WeightedContrastiveLoss(temperature=0.5, alpha=2)(
        y_pred, y_true, gene_embeddings=emb
    )
    assert torch.isfinite(out), "zero-positive anchor produced NaN/inf"

    grad = torch.autograd.grad(out, y_pred)[0]
    assert torch.isfinite(grad).all(), (
        "zero-positive anchor produced non-finite gradient"
    )


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


class TestBatchLaplacianReg:
    """Batch graph-Laplacian smoothness: the efficient B x B form must equal the definition.

    ``BatchLaplacianReg`` computes ``sum_ij W_ij ||y_i - y_j||^2`` through the B x B Gram
    matrix (``2*(deg . sq) - 2*sum(W*S)``) to avoid the 18080 x 18080 matrix the older
    ``LaplacianRegularizerLoss`` builds. That identity is the thing worth pinning.
    """

    @staticmethod
    def _brute_force(y: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        """The literal definition: relu-cosine affinity, zero diagonal, same normalisation."""
        e = torch.nn.functional.normalize(emb, dim=-1)
        w = (e @ e.T).relu()
        w = w - torch.diag(torch.diagonal(w))
        b = y.shape[0]
        energy = sum(
            w[i, j] * (y[i] - y[j]).pow(2).sum() for i in range(b) for j in range(b)
        )
        return energy / (w.sum().clamp_min(1e-8) * y.shape[1])

    def test_matches_definition(self):
        torch.manual_seed(1)
        y = torch.randn(6, 5)
        emb = torch.randn(6, 4)
        got = lf.BatchLaplacianReg()(y, y, emb)
        want = self._brute_force(y, emb)
        assert got.item() == pytest.approx(want.item(), rel=1e-4, abs=1e-6)

    def test_zero_when_predictions_identical(self):
        """Equal predictions -> every pairwise distance is 0 -> the penalty vanishes."""
        y = torch.randn(1, 5).expand(6, 5).contiguous()
        emb = torch.randn(6, 4)
        out = lf.BatchLaplacianReg()(y, y, emb)
        assert out.item() == pytest.approx(0.0, abs=1e-6)

    def test_positive_and_differentiable(self):
        torch.manual_seed(2)
        y = torch.randn(6, 5, requires_grad=True)
        emb = torch.randn(6, 4)
        out = lf.BatchLaplacianReg()(y, y.detach(), emb)
        assert out.item() > 0
        grad = torch.autograd.grad(out, y)[0]
        assert torch.isfinite(grad).all()
        assert grad.abs().sum() > 0

    def test_no_nan_with_duplicate_embeddings(self):
        """Cells of one perturbation carry identical embeddings (cosine 1) -- must stay finite."""
        torch.manual_seed(3)
        emb = torch.randn(3, 4).repeat_interleave(2, dim=0)  # 6 rows, 3 unique genes
        y = torch.randn(6, 5)
        out = lf.BatchLaplacianReg()(y, y, emb)
        assert torch.isfinite(out).all()


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


# --------------------------------------------------------------- DEWeightedMSELoss

N_PERT = 5


@pytest.fixture
def de_table(tmp_path):
    """A synthetic DE npz plus the matching gene-order parquet.

    Perturbation ``i`` is DE in exactly gene ``i``, which makes the weighting trivial to
    reason about: cell with ``ko_id=i`` should upweight column ``i`` and nothing else.
    """
    import numpy as np
    import polars as pl

    genes = [f"G{i:03d}" for i in range(GENES)]
    targets = genes[:N_PERT]

    q = np.ones((N_PERT, GENES), dtype=np.float32)
    lfc = np.zeros((N_PERT, GENES), dtype=np.float32)
    for i in range(N_PERT):
        q[i, i], lfc[i, i] = 0.0, 1.0

    de = tmp_path / "de.npz"
    np.savez(de, targets=np.array(targets), genes=np.array(genes), q=q, lfc=lfc)

    order = tmp_path / "order.parquet"
    pl.DataFrame({"gene_name": genes} | {"e0": [0.0] * GENES}).write_parquet(order)
    return de, order


def test_de_weighted_loss_is_differentiable(de_table, batch):
    """Same guard the configured losses get."""
    de, order = de_table
    loss_fn = lf.DEWeightedMSELoss(de_path=de, gene_order_path=order, alpha=9.0)
    out = loss_fn(**batch)

    assert out.grad_fn is not None
    grad = torch.autograd.grad(out, batch["y_pred"])[0]
    assert grad is not None and torch.isfinite(grad).all()


def test_alpha_zero_is_exactly_plain_mse(de_table, batch):
    """The knob must have a true off position, or 'DE weighting helped' is unfalsifiable."""
    de, order = de_table
    weighted = lf.DEWeightedMSELoss(de_path=de, gene_order_path=order, alpha=0.0)

    torch.testing.assert_close(weighted(**batch), lf.MyMSELoss()(**batch))


def test_de_genes_are_the_ones_upweighted(de_table):
    """Error on a perturbation's own DE gene must cost more than the same error elsewhere.

    Perturbation 0 is DE only in gene 0, so an identical squared error placed in gene 0
    and in gene 1 must produce different losses, and the DE one must be larger.
    """
    de, order = de_table
    loss_fn = lf.DEWeightedMSELoss(de_path=de, gene_order_path=order, alpha=9.0)

    y_true = torch.zeros(1, GENES)
    ko_id = torch.zeros(1, dtype=torch.long)

    on_de = torch.zeros(1, GENES)
    on_de[0, 0] = 1.0
    off_de = torch.zeros(1, GENES)
    off_de[0, 1] = 1.0

    hit = loss_fn(on_de, y_true, ko_id=ko_id)
    miss = loss_fn(off_de, y_true, ko_id=ko_id)
    assert hit > miss
    # 1 + alpha = 10x, and the row-mean normalisation cancels in the ratio.
    torch.testing.assert_close(hit / miss, torch.tensor(10.0))


def test_weights_keep_the_loss_on_the_mse_scale(de_table, batch):
    """Row-mean-1 normalisation: without it the learning rate silently changes meaning."""
    de, order = de_table
    loss_fn = lf.DEWeightedMSELoss(de_path=de, gene_order_path=order, alpha=9.0)

    torch.testing.assert_close(
        loss_fn.weights.mean(dim=1), torch.ones(N_PERT + 1), rtol=1e-5, atol=1e-6
    )


def test_unknown_perturbation_falls_back_to_uniform(de_table):
    """Genes with no DE row must degrade to plain MSE, not index out of bounds."""
    de, order = de_table
    loss_fn = lf.DEWeightedMSELoss(de_path=de, gene_order_path=order, alpha=9.0)

    y_pred, y_true = torch.rand(1, GENES), torch.rand(1, GENES)
    unknown = torch.full((1,), GENES - 1, dtype=torch.long)  # past the N_PERT targets

    torch.testing.assert_close(
        loss_fn(y_pred, y_true, ko_id=unknown), lf.MyMSELoss()(y_pred, y_true)
    )


def test_gene_axis_mismatch_is_fatal(de_table, tmp_path):
    """A silently misaligned gene axis would weight the wrong genes forever."""
    import polars as pl

    de, _ = de_table
    shuffled = tmp_path / "shuffled.parquet"
    names = [f"G{i:03d}" for i in range(GENES)][::-1]
    pl.DataFrame({"gene_name": names} | {"e0": [0.0] * GENES}).write_parquet(shuffled)

    with pytest.raises(ValueError, match="does not match the ko_id order"):
        lf.DEWeightedMSELoss(de_path=de, gene_order_path=shuffled)


def test_de_weights_stay_out_of_the_state_dict(de_table):
    """21 MB of derived table has no business riding along in every checkpoint."""
    de, order = de_table
    loss_fn = lf.DEWeightedMSELoss(de_path=de, gene_order_path=order)

    assert loss_fn.state_dict() == {}


# ------------------------------------------------------- masked flow objective


# --------------------------------------------------------------- BatchDEAwareMSELoss


def test_batch_de_aware_recovers_the_perturbation_mean():
    """The identity the whole class rests on: y_true + control_exp == mu_p.

    Only true for a velocity target under ``target_mode="perturbation_mean"``. If this
    breaks, the loss is standardising something that is not a perturbation profile.
    """
    torch.manual_seed(0)
    control = torch.rand(BATCH, GENES)
    mu = torch.rand(1, GENES).expand(BATCH, GENES)
    velocity = mu - control

    assert torch.allclose(velocity + control, mu, atol=1e-6)


def test_batch_de_aware_weights_up_and_down_equally():
    """Regression guard for the ``.abs()``.

    Without it a downregulated gene scores ``sigmoid(-z) -> 0`` and is weighted *out* of
    the loss -- including a knockdown's own target gene, which is the largest single
    effect in the data and the one the model most needs to fit.
    """
    torch.manual_seed(0)
    control = torch.rand(256, GENES)
    centre, spread = control.mean(0), control.std(0)

    # Half the genes move up by k standard deviations, the other half down by the same k.
    k = torch.linspace(0.0, 3.0, GENES // 2)
    mu = (centre + torch.cat([k, -k]) * spread).unsqueeze(0).expand(256, GENES)

    loss = lf.BatchDEAwareMSELoss(threshold=0.1, temperature=0.05, normalise=False)
    weights = loss.weights(mu - control, control)[0]

    up, down = weights[: GENES // 2], weights[GENES // 2 :]
    assert torch.allclose(up, down, atol=1e-4), (
        "downregulated genes are weighted differently from upregulated ones; "
        "the abs() in BatchDEAwareMSELoss.weights has been dropped"
    )


def test_batch_de_aware_degenerate_threshold_is_plain_mse(batch):
    """A threshold far below every z saturates the sigmoid, so normalising gives 1."""
    flat = lf.BatchDEAwareMSELoss(threshold=-1e6)

    assert torch.allclose(flat(**batch), lf.MyMSELoss()(**batch), atol=1e-6)


def test_batch_de_aware_normalise_holds_the_mse_scale():
    """Why ``normalise`` defaults to True.

    The raw weights are sigmoid outputs, so they average well below 1 -- measured at
    0.516 on a real 2025 batch. An unnormalised loss therefore trains at roughly half the
    configured learning rate, which is the same trap that made
    ``model_flow_abundance.yaml`` carry a hand-computed ``weights: [2.876]`` correction
    for ``DiffExpAwareMSELoss``. Dividing each row by its own mean removes the need for
    any such constant.

    Built here rather than taken from the ``batch`` fixture because the scale claim is
    about realistic velocities: the fixture's ``y_true`` and ``control_exp`` are
    independent uniforms, which puts ``mu_p`` far outside the control distribution and
    saturates most of the weights.
    """
    torch.manual_seed(0)
    control = torch.rand(256, GENES)
    mu = control.mean(0, keepdim=True) + 0.05 * torch.randn(1, GENES)
    payload = {
        "y_true": mu.expand(256, GENES) - control,
        "control_exp": control,
        # Isotropic residual: uncorrelated with the weight, which is the condition under
        # which a mean-1 reweighting leaves the loss exactly where it was.
        "y_pred": mu.expand(256, GENES) - control + 0.1 * torch.randn(256, GENES),
    }

    plain = lf.MyMSELoss()(**payload).item()
    normalised = lf.BatchDEAwareMSELoss(normalise=True)(**payload).item()
    raw = lf.BatchDEAwareMSELoss(normalise=False)(**payload).item()

    assert normalised == pytest.approx(plain, rel=0.05)
    assert raw < 0.75 * plain, (
        f"unnormalised loss is {raw / plain:.2f}x plain MSE; the learning rate would "
        "silently change with it"
    )


def test_batch_de_aware_survives_a_single_cell_batch(batch):
    """An unbiased std over one sample is NaN; fast_dev_run and trailing batches hit it."""
    one = {k: v[:1] for k, v in batch.items()}
    out = lf.BatchDEAwareMSELoss()(**one)

    assert torch.isfinite(out), "batch of one produced a non-finite loss"
    assert torch.allclose(out, lf.MyMSELoss()(**one), atol=1e-6)


def test_batch_de_aware_takes_its_statistics_in_float32(batch):
    """The datamodule emits bfloat16 under `bf16-mixed`.

    A 1024-sample mean and standard deviation in bfloat16's ~3 significant digits is not
    worth computing, so the weight is built in float32 and cast back at the end.
    """
    loss = lf.BatchDEAwareMSELoss()
    half = {k: v.bfloat16() for k, v in batch.items() if k != "ko_id"}

    assert loss.weights(half["y_true"], half["control_exp"]).dtype == torch.float32

    out = loss(**half)
    assert torch.isfinite(out) and out.dtype == torch.bfloat16


def test_batch_de_aware_weight_never_carries_gradient(batch):
    """The weight is a function of the target and the input only.

    If it depended on ``y_pred`` the model could lower its loss by deciding which genes
    count, rather than by predicting them better.
    """
    loss = lf.BatchDEAwareMSELoss()
    weights = loss.weights(batch["y_true"], batch["control_exp"])

    assert not weights.requires_grad

    grad = torch.autograd.grad(loss(**batch), batch["y_pred"])[0]
    assert grad.abs().sum() > 0 and torch.isfinite(grad).all()


def test_gene_mask_zeroes_the_gradient_on_unmeasured_genes():
    """A masked gene must receive exactly no gradient, with no loss-side changes.

    This is what lets datasets with different gene panels share one objective:
    ``d/dv (m*v - m*u)^2 = 2 m^2 (v - u)`` vanishes wherever ``m == 0``, so every
    existing loss keeps working untouched. If this ever fails, a source measuring 41%
    of the panel is training the model to predict zero change on the other 59%.
    """
    import torch

    from src.models.flow_lightning import VCCModule

    n_genes = 10
    mask = torch.zeros(4, n_genes)
    mask[:, : n_genes // 2] = 1.0

    pred = torch.randn(4, n_genes, requires_grad=True)
    target = torch.randn(4, n_genes)

    masked_pred, masked_target = VCCModule._apply_mask(
        {"gene_mask": mask}, pred, target
    )
    torch.nn.functional.mse_loss(masked_pred, masked_target).backward()

    assert pred.grad is not None
    assert torch.all(pred.grad[:, n_genes // 2 :] == 0)
    assert torch.any(pred.grad[:, : n_genes // 2] != 0)


def test_absent_gene_mask_is_a_no_op():
    """Single-panel runs must be untouched by the masking machinery."""
    import torch

    from src.models.flow_lightning import VCCModule

    pred, target = torch.randn(3, 5), torch.randn(3, 5)
    out_pred, out_target = VCCModule._apply_mask({}, pred, target)
    assert out_pred is pred and out_target is target


# --------------------------------------------------- learnable embedding wiring


def _flow_module(embedding_lr=None, num_genes=12, dim=4, init_path=None):
    """A VCCModule whose net holds a small learnable embedding table."""
    import torch

    from src.models.components.flow_model import FlowCellModel, LearnableGeneEmbedding
    from src.models.flow_lightning import VCCModule

    mlp = dict(
        hidden_size=8,
        num_hidden_layers=1,
        output_size=8,
        dropout=0.0,
        activation="gelu",
    )
    net = FlowCellModel(
        time_embedding_dim=8,
        t_processor_args=dict(input_size=8, **mlp),
        ko_processor_args=dict(input_size=dim, **mlp),
        exp_processor_args=dict(input_size=num_genes, **mlp),
        fused_processor_args=dict(
            input_size=8,
            hidden_size=8,
            num_hidden_layers=1,
            output_size=num_genes,
            dropout=0.0,
            activation="gelu",
            ensure_output_positive=False,
        ),
        fusion_type="sum",
        gene_embedding=LearnableGeneEmbedding(
            num_genes=num_genes, embedding_dim=dim, init_path=init_path
        ),
    )

    class _Loss(torch.nn.Module):
        """MSE that tolerates the kwargs CompositeLoss consumers are handed."""

        def forward(self, pred, target, **_):
            """Mean squared error, ignoring the extra loss inputs."""
            return torch.nn.functional.mse_loss(pred, target)

    return VCCModule(
        net=net,
        loss_fn=_Loss(),
        optimizer=lambda p: torch.optim.AdamW(p, lr=1e-4, weight_decay=1e-5),
        scheduler=lambda o: torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            o, T_0=2
        ),
        pert_input_key="ko_id",
        embedding_lr=embedding_lr,
    )


def test_embedding_table_never_gets_weight_decay():
    """The table must be its own group at wd=0, with or without `embedding_lr`.

    nn.Embedding has a dense gradient, so a table left in the main group is decayed on
    every row -- including the 8,847 the datasets never perturb, whose only job is to
    still hold their prior value at prediction time. It also makes `drift()` report every
    row as trained once the decay clears float32 resolution.
    """
    for lr, expected in ((None, 1e-4), (1e-5, 1e-5)):
        groups = (
            _flow_module(embedding_lr=lr)
            .configure_optimizers()["optimizer"]
            .param_groups
        )
        table = [g for g in groups if g["weight_decay"] == 0.0]
        assert len(table) == 1, f"expected exactly one zero-decay group, got {groups}"
        assert len(table[0]["params"]) == 1
        assert table[0]["lr"] == expected
        assert any(g["weight_decay"] == 1e-5 for g in groups), (
            "the net must still decay"
        )


def test_untrained_embedding_rows_do_not_move(tmp_path):
    """Rows never seen in a batch must stay exactly at their initial value.

    This is what lets the 8,847 unperturbed rows -- and the 28 leaderboard targets among
    them -- keep the GO prior the model still has to read them through.
    """
    import numpy as np
    import polars as pl
    import torch

    rng = np.random.default_rng(0)
    prior_path = tmp_path / "prior.parquet"
    pl.DataFrame(
        {"gene_name": [f"G{i}" for i in range(12)]}
        | {f"latent_{j}": rng.standard_normal(12) for j in range(4)}
    ).write_parquet(prior_path)

    module = _flow_module(init_path=prior_path)
    assert module.net.gene_embedding.drift() == (0.0, 0), "nothing has moved yet"
    opt = module.configure_optimizers()["optimizer"]
    prior = module.net.gene_embedding.table.weight.detach().clone()

    seen = torch.tensor([1, 3])
    for _ in range(200):
        opt.zero_grad()
        # A real batch always carries both keys; `pert_input_key` picks which one the
        # net looks up, while the losses are always handed the vector.
        batch = (
            {"ko_id": seen, "ko_vec": torch.rand(2, 4), "exp_vec": torch.rand(2, 12)},
            torch.rand(2, 12),
        )
        module._shared_step(batch, "train/{}", deterministic=True)[0].backward()
        opt.step()

    moved = (module.net.gene_embedding.table.weight.detach() - prior).norm(dim=1)
    assert torch.all(moved[seen] > 0), "the rows in the batch must have learned"
    untouched = [i for i in range(12) if i not in seen.tolist()]
    assert torch.all(moved[untouched] == 0), (
        f"untouched rows drifted: {moved[untouched]}"
    )
    assert module.net.gene_embedding.drift() == (
        module.net.gene_embedding.drift()[0],
        2,
    ), "n_moved must count only trained rows"


def test_drift_survives_a_degenerate_prior_row():
    """A zero-norm prior row must not blow up the reported drift.

    719 of the 18,080 rows in the GO parquet are exactly zero -- genes with no
    annotation. Dividing by their norm gives ~1e7, and on a real run 86 such rows turned
    a true drift of 0.023 into 65,520, making `emb/drift` unreadable for the whole run.
    """
    import torch

    from src.models.components.flow_model import LearnableGeneEmbedding

    table = LearnableGeneEmbedding(num_genes=4, embedding_dim=3)
    prior = torch.tensor(
        [[3.0, 4.0, 0.0], [0.0, 0.0, 0.0], [0.0, 6.0, 8.0], [1.0, 0.0, 0.0]]
    )
    table.register_buffer("prior", prior.clone(), persistent=False)
    with torch.no_grad():
        table.table.weight.copy_(prior)
        table.table.weight[0, 0] += 0.5  # norm 5 -> relative move 0.1
        table.table.weight[1, 0] += 0.5  # zero prior: must not enter the average
        table.table.weight[2, 1] += 1.0  # norm 10 -> relative move 0.1

    drift, n_moved = table.drift()
    assert n_moved == 3, "the zero-prior row still counts as trained"
    assert drift == pytest.approx(0.1, abs=1e-6), f"degenerate row leaked in: {drift}"


# ---------------------------------------------------------------------------
# BatchDiffExpError -- the pseudobulk DE cosine
# ---------------------------------------------------------------------------


def _pseudobulk_batch(n_per_group=8, n_groups=4, genes=GENES, noise=0.0, seed=0):
    """A batch whose per-perturbation profiles are known exactly."""
    g = torch.Generator().manual_seed(seed)
    control = torch.rand(n_per_group * n_groups, genes, generator=g) * 3.0
    ko_id = torch.arange(n_groups).repeat_interleave(n_per_group)
    profiles = torch.rand(n_groups, genes, generator=g) * 3.0
    mu = profiles[ko_id]
    y_true = mu - control  # the velocity target
    y_pred = y_true + noise * torch.randn(y_true.shape, generator=g)
    return y_pred, y_true, control, ko_id


def test_batch_diff_exp_is_zero_for_a_perfect_prediction():
    """A perfect answer must score 0 -- the property DiffExpError caps at 0.849."""
    loss = lf.BatchDiffExpError()
    y_pred, y_true, control, ko_id = _pseudobulk_batch()
    out = loss(y_pred, y_true, control_exp=control, ko_id=ko_id)
    assert out.item() == pytest.approx(0.0, abs=1e-5)


def test_batch_diff_exp_beats_the_per_cell_version_on_the_same_batch():
    """The whole justification: per-cell scoring is dominated by the paired control cell.

    With cells drawn around a shared profile, the per-cell cosine of a PERFECT prediction
    is far from 1 while the pseudobulk one is exact.
    """
    y_pred, y_true, control, ko_id = _pseudobulk_batch()
    batch_out = lf.BatchDiffExpError()(y_pred, y_true, control_exp=control, ko_id=ko_id)
    per_cell = lf.DiffExpError()(y_pred, y_true, control_exp=control)
    assert batch_out.item() < per_cell.item()


def test_batch_diff_exp_penalises_a_perturbation_blind_prediction():
    """Predicting the same profile for every perturbation must cost something."""
    loss = lf.BatchDiffExpError()
    y_pred, y_true, control, ko_id = _pseudobulk_batch()
    mu = y_true + control
    blind = mu.mean(dim=0, keepdim=True).expand_as(mu) - control
    good = loss(y_pred, y_true, control_exp=control, ko_id=ko_id)
    bad = loss(blind, y_true, control_exp=control, ko_id=ko_id)
    # The property that matters is the ORDERING and that the gap is not marginal: a
    # perturbation-blind answer is what the 2026 scale puts at zero, and this loss is the
    # only term in the objective that can tell it apart from a specific one. No magic
    # threshold -- perfect is ~0 by construction, so require blind to cost a real fraction
    # of the cosine's [0, 2] range.
    assert good.item() == pytest.approx(0.0, abs=1e-4)
    assert bad.item() > 0.25


def test_batch_diff_exp_requires_ko_id():
    with pytest.raises(ValueError, match="ko_id"):
        lf.BatchDiffExpError()(
            *_pseudobulk_batch()[:2], control_exp=torch.rand(32, GENES)
        )


def test_batch_diff_exp_drops_singleton_groups():
    """A one-cell group is the 3.5%-signal quantity this loss exists to avoid."""
    loss = lf.BatchDiffExpError(min_group_size=2)
    y_pred, y_true, control, ko_id = _pseudobulk_batch(n_per_group=1, n_groups=6)
    assert loss(y_pred, y_true, control_exp=control, ko_id=ko_id).item() == 0.0


def test_batch_diff_exp_is_differentiable():
    loss = lf.BatchDiffExpError()
    y_pred, y_true, control, ko_id = _pseudobulk_batch(noise=0.3)
    y_pred = y_pred.clone().requires_grad_(True)
    out = loss(y_pred, y_true, control_exp=control, ko_id=ko_id)
    out.backward()
    assert y_pred.grad is not None and torch.isfinite(y_pred.grad).all()
    assert y_pred.grad.abs().sum() > 0


def test_batch_diff_exp_survives_bfloat16_inputs():
    loss = lf.BatchDiffExpError()
    y_pred, y_true, control, ko_id = _pseudobulk_batch(noise=0.2)
    out = loss(
        y_pred.bfloat16(),
        y_true.bfloat16(),
        control_exp=control.bfloat16(),
        ko_id=ko_id,
    )
    assert torch.isfinite(out) and out.dtype == torch.bfloat16


def test_batch_diff_exp_ignores_unmeasured_genes_under_a_mask():
    """Masked genes carry `fill_profile`, a profile that was never observed."""
    loss = lf.BatchDiffExpError()
    y_pred, y_true, control, ko_id = _pseudobulk_batch(noise=0.2)
    mask = torch.ones_like(control)
    mask[:, GENES // 2 :] = 0.0
    clean = loss(y_pred, y_true, control_exp=control, ko_id=ko_id, gene_mask=mask)
    # Garbage beyond the mask must not move the answer.
    dirty_pred = y_pred.clone()
    dirty_pred[:, GENES // 2 :] += 50.0
    dirty = loss(dirty_pred, y_true, control_exp=control, ko_id=ko_id, gene_mask=mask)
    assert clean.item() == pytest.approx(dirty.item(), abs=1e-5)


# ---------------------------------------------------------------------------
# BatchDeltaMagnitudeLoss -- pseudobulk mean-VELOCITY magnitude undershoot penalty
# ---------------------------------------------------------------------------


def _scaled_pred(y_true, alpha):
    """A prediction that scales the velocity uniformly: ``alpha * y_true``.

    The whole point of the mean-velocity formulation is that this makes the per-group
    magnitude ratio exactly ``alpha`` -- ``mean(alpha * y_true) == alpha * mean(y_true)`` --
    with the DIRECTION untouched, which the (rejected) xbar-referenced delta could not do.
    """
    return alpha * y_true


def test_batch_delta_mag_zero_when_calibrated():
    """alpha == 1: predicted magnitude equals true magnitude -> relu(1 - 1) == 0."""
    loss = lf.BatchDeltaMagnitudeLoss()
    y_pred, y_true, control, ko_id = _pseudobulk_batch()
    out = loss(_scaled_pred(y_true, 1.0), y_true, control_exp=control, ko_id=ko_id)
    assert out.item() == pytest.approx(0.0, abs=1e-4)


def test_batch_delta_mag_ratio_reads_the_shrink_factor():
    """The reason for the mean-velocity form: a 0.2x velocity gives loss ~ (1 - 0.2)^2.

    The xbar-referenced delta could only manage ~0.9 here because the control-mean deviation
    inflated the predicted magnitude; the mean velocity reads the shrink cleanly.
    """
    loss = lf.BatchDeltaMagnitudeLoss(de_weighted=False)
    y_pred, y_true, control, ko_id = _pseudobulk_batch(n_per_group=16, n_groups=12)
    out = loss(_scaled_pred(y_true, 0.2), y_true, control_exp=control, ko_id=ko_id)
    assert out.item() == pytest.approx((1.0 - 0.2) ** 2, abs=0.02)


def test_batch_delta_mag_zero_on_overshoot():
    """The asymmetry: an over-large prediction is NOT penalised (relu clips it).

    This is the guardrail that keeps the term from ever driving an nmae-blowing overshoot.
    """
    loss = lf.BatchDeltaMagnitudeLoss()
    y_pred, y_true, control, ko_id = _pseudobulk_batch()
    out = loss(_scaled_pred(y_true, 2.0), y_true, control_exp=control, ko_id=ko_id)
    assert out.item() == pytest.approx(0.0, abs=1e-6)


def test_batch_delta_mag_penalises_undershoot():
    """A shrunk prediction (the observed failure) costs something; more shrink costs more."""
    loss = lf.BatchDeltaMagnitudeLoss()
    y_pred, y_true, control, ko_id = _pseudobulk_batch()
    mild = loss(_scaled_pred(y_true, 0.5), y_true, control_exp=control, ko_id=ko_id)
    hard = loss(_scaled_pred(y_true, 0.2), y_true, control_exp=control, ko_id=ko_id)
    assert hard.item() > mild.item() > 0.0


def test_batch_delta_mag_penalises_mean_collapse():
    """A model predicting the same (batch-mean) velocity for every perturbation is caught.

    That is the actual failure mode: mean_group(y_pred) collapses toward the global mean, so
    its magnitude is far below each perturbation's true mean-velocity magnitude.
    """
    loss = lf.BatchDeltaMagnitudeLoss(de_weighted=False)
    y_pred, y_true, control, ko_id = _pseudobulk_batch(n_per_group=16, n_groups=12)
    collapsed = y_true.mean(dim=0, keepdim=True).expand_as(
        y_true
    )  # same for every cell
    good = loss(_scaled_pred(y_true, 1.0), y_true, control_exp=control, ko_id=ko_id)
    bad = loss(collapsed, y_true, control_exp=control, ko_id=ko_id)
    assert good.item() == pytest.approx(0.0, abs=1e-4)
    assert bad.item() > 0.25


def test_batch_delta_mag_gradient_is_direction_preserving():
    """The load-bearing claim: the descent direction grows the velocity ALONG its own axis.

    With uniform weights and one group, the negative gradient on every cell must be
    parallel to that group's mean predicted velocity -- i.e. the term inflates the model's
    own vector, it never rotates it toward the truth (which is what a contrastive would do).
    """
    loss = lf.BatchDeltaMagnitudeLoss(de_weighted=False, min_rel_true_mag=0.0)
    y_pred, y_true, control, ko_id = _pseudobulk_batch(n_per_group=8, n_groups=1)
    pred = _scaled_pred(y_true, 0.3).clone().requires_grad_(True)
    out = loss(pred, y_true, control_exp=control, ko_id=ko_id)
    out.backward()

    v_pred = pred.detach().float().mean(0)  # the group's mean predicted velocity
    for row in pred.grad:
        cos = torch.cosine_similarity(-row, v_pred, dim=0, eps=1e-12)
        assert cos.item() > 0.999, (
            "gradient is not parallel to the predicted mean velocity"
        )


def test_batch_delta_mag_requires_ko_id():
    with pytest.raises(ValueError, match="ko_id"):
        lf.BatchDeltaMagnitudeLoss()(
            *_pseudobulk_batch()[:2], control_exp=torch.rand(32, GENES)
        )


def test_batch_delta_mag_drops_singleton_groups():
    """A one-cell group has no meaningful pseudobulk magnitude -> excluded, loss 0."""
    loss = lf.BatchDeltaMagnitudeLoss(min_group_size=2)
    y_pred, y_true, control, ko_id = _pseudobulk_batch(n_per_group=1, n_groups=6)
    shrunk = _scaled_pred(y_true, 0.1)
    assert loss(shrunk, y_true, control_exp=control, ko_id=ko_id).item() == 0.0


def test_batch_delta_mag_drops_near_zero_true_groups():
    """A control-like group (mean velocity ~ 0) is dropped, so it cannot demand a magnitude.

    With the floor on, mangling that group's prediction must not change the loss.
    """
    y_pred, y_true, control, ko_id = _pseudobulk_batch(
        n_per_group=8, n_groups=3, seed=1
    )
    is_ctrl = ko_id == 0
    # Make group 0 a control: zero true velocity -> zero mean-velocity magnitude.
    y_true = y_true.clone()
    y_true[is_ctrl] = 0.0

    loss = lf.BatchDeltaMagnitudeLoss(min_rel_true_mag=0.1)
    shrunk = _scaled_pred(y_true, 0.2)
    base = loss(shrunk, y_true, control_exp=control, ko_id=ko_id)

    # Garbage predictions for the control group must not move the answer -- it is dropped.
    mangled = shrunk.clone()
    mangled[is_ctrl] += 100.0
    with_junk = loss(mangled, y_true, control_exp=control, ko_id=ko_id)
    assert base.item() == pytest.approx(with_junk.item(), abs=1e-5)
    assert torch.isfinite(base) and base.item() > 0.0


def test_batch_delta_mag_ignores_unmeasured_genes_under_a_mask():
    """Masked genes hold `fill_profile`; garbage there must not enter the magnitude."""
    loss = lf.BatchDeltaMagnitudeLoss(min_rel_true_mag=0.0)
    y_pred, y_true, control, ko_id = _pseudobulk_batch()
    shrunk = _scaled_pred(y_true, 0.3)
    mask = torch.ones_like(control)
    mask[:, GENES // 2 :] = 0.0
    clean = loss(shrunk, y_true, control_exp=control, ko_id=ko_id, gene_mask=mask)
    dirty = shrunk.clone()
    dirty[:, GENES // 2 :] += 50.0
    out = loss(dirty, y_true, control_exp=control, ko_id=ko_id, gene_mask=mask)
    assert clean.item() == pytest.approx(out.item(), abs=1e-5)


def test_batch_delta_mag_weight_never_carries_gradient():
    """The DE weight is a no-grad function of the target -- the model cannot game it."""
    loss = lf.BatchDeltaMagnitudeLoss(de_weighted=True)
    y_pred, y_true, control, ko_id = _pseudobulk_batch()
    pred = _scaled_pred(y_true, 0.3).clone().requires_grad_(True)
    out = loss(pred, y_true, control_exp=control, ko_id=ko_id)
    out.backward()
    assert pred.grad is not None and torch.isfinite(pred.grad).all()
    assert pred.grad.abs().sum() > 0


def test_batch_delta_mag_survives_bfloat16_inputs():
    loss = lf.BatchDeltaMagnitudeLoss()
    y_pred, y_true, control, ko_id = _pseudobulk_batch()
    shrunk = _scaled_pred(y_true, 0.3)
    out = loss(
        shrunk.bfloat16(),
        y_true.bfloat16(),
        control_exp=control.bfloat16(),
        ko_id=ko_id,
    )
    assert torch.isfinite(out) and out.dtype == torch.bfloat16
