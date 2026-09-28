"""Losses built for flow-velocity targets.

They rely on the ``y_true + control_exp = mu_p`` identity (``velocity_target``), holding under
the datamodule's ``target_mode="perturbation_mean"``. ``DEWeightedMSELoss`` is the exception
that reads a precomputed DE table (``de_mwu_2025.npz``) rather than batch statistics, so it is
the one loss here that is not batch-only.
"""

from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn as nn


class DEWeightedMSELoss(nn.Module):
    """MSE with a per-gene weight that depends on which perturbation the cell carries.

    Roughly 2.3% of genes are differentially expressed for a given perturbation
    (median 418 of 18,080 at ``q < 0.05`` and ``|log2FC| >= 0.25``, against a null floor
    of 26). Plain MSE therefore spends ~98% of its gradient on genes whose correct answer
    is "unchanged", while every metric still costing us on the leaderboard -- ``nmae``,
    ``fid``, ``jac`` -- is computed *only* over DE genes. This reweights the objective
    toward the genes the score actually reads.

    The weight is ``1 + alpha`` on DE genes and ``1`` elsewhere, then divided by its own
    row mean. That normalisation matters: it holds the loss at the same scale as plain
    MSE, so the learning rate keeps its meaning and runs stay comparable. ``alpha = 0``
    reproduces :class:`MyMSELoss` exactly.

    Unlike :class:`DiffExpAwareMSELoss`, this does not infer importance from
    ``y_true - control``; it reads a precomputed Mann-Whitney table. That distinction is
    what makes it usable here, because for the flow model ``y_pred`` and ``y_true`` are
    **velocities**, not expression profiles, and a weight derived from them would be
    measuring the wrong thing. A weight indexed by perturbation and gene is agnostic to
    what the regressed quantity is.

    Parameters
    ----------
    de_path : str | Path
        ``.npz`` from ``scripts/de_mwu_2025.py`` with ``targets``, ``genes``, ``q`` and
        ``lfc``.
    gene_order_path : str | Path
        Parquet whose ``gene_name`` column defines ``ko_id`` -- i.e. the gene-embedding
        table the datamodule reads. Used to map a ``ko_id`` to its row of the DE table,
        and to verify the two gene axes agree rather than assuming it.
    alpha : float
        Extra weight on DE genes. The share of the gradient they receive is
        ``f * (1 + alpha) / (1 + alpha * f)`` for a DE fraction ``f``. Take ``f`` from
        ``extra_repr`` rather than from the per-perturbation median: the DE count is
        badly skewed (2025 median 418 genes, mean 1282, max 8058), so the mean over all
        entries is 7.09% and ``alpha = 9`` puts 43% of the gradient on DE genes, not the
        19% the median would suggest.
    q_cut, lfc_cut : float
        Significance and effect-size thresholds defining "differentially expressed".
    reduction : str | None
        ``"mean"``, ``"sum"``, or ``None`` for the per-cell vector.
    """

    def __init__(
        self,
        de_path: str | Path,
        gene_order_path: str | Path,
        alpha: float = 9.0,
        q_cut: float = 0.05,
        lfc_cut: float = 0.25,
        reduction: str | None = "mean",
    ) -> None:
        super(DEWeightedMSELoss, self).__init__()

        if alpha < 0:
            raise ValueError(f"alpha must be >= 0, got {alpha}")

        self.reduction = reduction
        self.alpha = alpha
        self.q_cut = q_cut
        self.lfc_cut = lfc_cut

        table = np.load(de_path, allow_pickle=False)
        de_genes = [str(g) for g in table["genes"]]
        targets = [str(t) for t in table["targets"]]

        panel = [
            str(g) for g in pl.read_parquet(gene_order_path)["gene_name"].to_list()
        ]
        if de_genes != panel:
            raise ValueError(
                f"the DE table's gene axis ({len(de_genes)} genes) does not match the "
                f"ko_id order from {gene_order_path} ({len(panel)} genes). A column "
                "mismatch here would silently weight the wrong genes, so this is fatal "
                "rather than best-effort."
            )

        mask = (table["q"] < q_cut) & (np.abs(table["lfc"]) >= lfc_cut)
        weights = 1.0 + alpha * mask.astype(np.float32)
        weights /= weights.mean(axis=1, keepdims=True)

        # A trailing all-ones row is the fallback for any gene with no DE entry, so the
        # lookup stays a plain index with no branch and no masking in the hot path.
        weights = np.vstack([weights, np.ones((1, len(panel)), dtype=np.float32)])
        self.register_buffer("weights", torch.from_numpy(weights), persistent=False)

        row_of = np.full(len(panel), len(targets), dtype=np.int64)
        position = {gene: i for i, gene in enumerate(panel)}
        for row, target in enumerate(targets):
            if target in position:
                row_of[position[target]] = row
        self.register_buffer("row_of_ko_id", torch.from_numpy(row_of), persistent=False)

        self.de_fraction = float(mask.mean())
        self.covered = int((row_of != len(targets)).sum())

    def extra_repr(self) -> str:
        """Report what the weighting actually does, for the model summary."""
        f, a = self.de_fraction, self.alpha
        share = f * (1 + a) / (1 + a * f) if f else 0.0
        return (
            f"alpha={a}, q<{self.q_cut}, |lfc|>={self.lfc_cut}, "
            f"DE fraction={100 * f:.2f}%, gradient share on DE genes="
            f"{100 * share:.1f}% (was {100 * f:.2f}%), "
            f"perturbations covered={self.covered}"
        )

    def forward(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        ko_id: torch.Tensor,
        **kwargs,
    ):
        """Weighted MSE, one weight vector per cell chosen by its perturbation.

        Args:
            y_pred (Tensor): Predicted values -- velocity for the flow model.
            y_true (Tensor): Target values, same quantity as ``y_pred``.
            ko_id (Tensor): Gene ids, shape ``(B,)``, indexing ``gene_order_path``.

        Returns:
            Tensor: The loss under ``reduction``.
        """
        weights = self.weights[self.row_of_ko_id[ko_id]].to(y_pred.dtype)
        calc = (weights * (y_pred - y_true) ** 2).mean(dim=-1)

        match self.reduction:
            case "sum":
                return calc.sum()
            case "mean":
                return calc.mean()
            case _:
                return calc


class BatchDEAwareMSELoss(nn.Module):
    """MSE tilted toward differentially expressed genes, using only what is in the batch.

    The DE-aware sibling of :class:`DiffExpAwareMSELoss` that is actually valid for a
    velocity target, and the no-prep-step sibling of :class:`DEWeightedMSELoss`.

    The identity this rests on
    --------------------------
    Under ``target_mode="perturbation_mean"`` the flow's target is the velocity
    ``y_true = mu_p - x_0`` and ``control_exp`` is ``x_0``, so::

        y_true + control_exp = mu_p

    *exactly*. The perturbation's mean profile is therefore already in the batch and needs
    no lookup table indexed by perturbation. What is missing is a reference to measure it
    against and a scale to divide by, and both are batch statistics of ``control_exp``.

    Why ``DiffExpAwareMSELoss`` cannot do this
    ------------------------------------------
    That class weights by ``|y_true - control_exp|``. On an expression model that bracket
    is the true log fold change and the name is honest. Here ``y_true`` is a velocity, so
    it is ``|mu_p - 2 x_0| ~= |x_0|`` -- the control abundance. Measured against the
    Mann-Whitney calls from ``scripts/de_mwu_2025.py``::

        corr(weight, control abundance |x_0|)   +0.953
        corr(weight, the DE call itself)        -0.031   <-- no relationship at all

    and its share of the gradient landing on truly-DE genes is 6.8%, against plain MSE's
    7.1%. It is an abundance prior, not a DE prior.

    Why the standard deviation is in the denominator
    ------------------------------------------------
    **This is the load-bearing part, not numerical hygiene.** Raw movement is mostly a
    restatement of abundance, and abundance does not predict DE::

        corr(control mean, |mu_p - xbar|)   +0.776
        corr(control mean, true DE rate)    -0.137

    A highly expressed gene swings more in log space from counting noise alone, while a
    barely-expressed one can double and still move 0.001. Dividing by the gene's own
    control spread asks "did this move a lot *for a gene like this*", which is the ratio a
    Mann-Whitney or t-test is built from. Measured, it nearly doubles the agreement with
    the real DE calls::

        corr(|mu_p - xbar|,      DE rate)   +0.283
        corr(|mu_p - xbar| / sd, DE rate)   +0.512

    and raises the fraction of each perturbation's top-500 weighted genes that are truly
    DE from 0.346 to 0.445, against a 0.071 base rate.

    A consequence worth having: ``threshold`` then lives in **standard-deviation units**.
    ``DiffExpAwareMSELoss``'s ``threshold=1`` is an absolute value in log1p space, so it is
    silently retuned by anything that rescales the data -- adding auxiliary datasets
    shrinks the depth axis from 18,080 genes to 6,691 and multiplies every value by ~1.31.
    An SD-relative threshold survives that.

    Not yet safe with ``gene_mask``
    -------------------------------
    ``flow_lightning._apply_mask`` zeroes ``y_pred`` and ``y_true`` on genes a source did
    not measure but passes ``control_exp`` through unmasked, so there
    ``y_true + control_exp`` is ``x_0`` rather than ``mu_p`` and the weight is computed on
    a profile that does not exist. The squared error is zero on those genes so the loss
    *value* is unaffected, but ``normalise`` divides by a mean that phantom weights have
    diluted. Single-panel runs never allocate a mask and are unaffected; plumb ``gene_mask``
    into the call before using this with ``data=dataset_multi``.

    Parameters
    ----------
    threshold : float
        Sigmoid anchor, in units of control standard deviations. Genes moving further than
        this get weight > 0.5.
    temperature : float
        Sigmoid sharpness, same units.

        **Pick it against the spread of ``z``, not by analogy with the other losses.**
        Measured on a real 1024-cell batch, ``z = |mu_p - xbar| / sd`` has median 0.032 and
        p95 0.238 -- so at ``temperature=1`` the sigmoid never leaves its linear region
        around 0.5, the weight spans 0.500-0.571, and 7.7% of the gradient lands on DE
        genes against plain MSE's 7.1%. That is plain MSE with extra steps. For a real
        tilt::

            threshold  temperature   mean w   p95/p5 range
                0.0        1.0        0.516        1.1x   <- defaults, ~no effect
                0.0        0.05       0.690        2.0x
                0.10       0.05       0.306        7.9x
                0.15       0.05       0.183       17.9x

    eps : float
        Added to the standard deviation. Genes never detected in a control cell have
        ``sd == 0`` exactly; at ``eps=1e-6`` that saturates 2.3% of entries to weight 1.0,
        handing undetected junk genes the top of a range that only spans [0.5, 1) at the
        default threshold. At ``eps=1e-2`` -- roughly 7% of the median control sd of 0.138
        -- it is 0.0%. Raise this first if the weight looks wrong.
    normalise : bool
        Divide each cell's weight vector by its own mean, holding the loss on the same
        scale as plain MSE so the learning rate keeps its meaning.

        Leave this on. ``DiffExpAwareMSELoss`` does not normalise and its weights average
        0.348, which is why ``config/model/model_flow_abundance.yaml`` carries a
        ``weights: [2.876]`` correction and forty lines explaining it. The mean weight here
        is 0.516 at the defaults, so an unnormalised run would train at roughly half the
        configured learning rate -- and would not be comparable to the baseline it exists
        to be compared against.
    velocity_target : bool
        True when ``y_true`` is a velocity, as it is for
        :class:`~src.models.flow_lightning.VCCModule`; false when ``y_true`` is already an
        expression profile, as in ``vcc_lightning``. Getting this wrong does not raise, it
        silently weights the wrong quantity, so it is stated in config rather than sniffed
        from the shapes.

        The ``mu_p`` identity also needs ``target_mode="perturbation_mean"``. Under
        ``target_mode="cell"`` ``y_true + control_exp`` is one perturbed *cell*, and the
        weight would track that cell's sampling noise instead of the perturbation.
    reduction : str | None
        ``"mean"``, ``"sum"``, or ``None`` for the per-cell vector.
    """

    def __init__(
        self,
        threshold: float = 0.0,
        temperature: float = 1.0,
        eps: float = 1e-6,
        normalise: bool = True,
        velocity_target: bool = True,
        reduction: str | None = "mean",
    ) -> None:
        super(BatchDEAwareMSELoss, self).__init__()

        if temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature}")
        if eps < 0:
            raise ValueError(f"eps must be >= 0, got {eps}")

        self.threshold = threshold
        self.temperature = temperature
        self.eps = eps
        self.normalise = normalise
        self.velocity_target = velocity_target
        self.reduction = reduction

    def extra_repr(self) -> str:
        """Report the settings in SD units, so the model summary says what this does."""
        return (
            f"threshold={self.threshold} sd, temperature={self.temperature} sd, "
            f"eps={self.eps}, normalise={self.normalise}, "
            f"velocity_target={self.velocity_target}"
        )

    @torch.no_grad()
    def weights(
        self,
        y_true: torch.Tensor,
        control_exp: torch.Tensor,
        gene_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Per-gene weight, shape ``(B, n_genes)``, in float32.

        Exposed rather than private so a candidate ``threshold``/``temperature`` can be
        scored against ``de_mwu_2025.npz`` offline, without spending a training run. That
        keeps the Mann-Whitney table a *validator* rather than a runtime dependency.

        Never carries gradient: the weight is a function of the target and the input only,
        so the model cannot lower its loss by changing which genes count.

        Args:
            y_true (torch.Tensor): The target -- a velocity when ``velocity_target``.
            control_exp (torch.Tensor): Control expression ``x_0``, shape ``(B, n_genes)``.
            gene_mask (torch.Tensor | None): 1 on genes the sample's source measured.
                Present only when the run mixes datasets with different gene panels.

        Returns:
            torch.Tensor: Weights, ``(B, n_genes)``, float32, zero on unmeasured genes.
        """
        # float32 regardless of the trainer's precision. Under `bf16-mixed` the datamodule
        # emits bfloat16, whose ~3 significant digits make a 1024-sample mean and standard
        # deviation worthless.
        control = control_exp.float()
        target = y_true.float()

        mu = target + control if self.velocity_target else target

        # An unbiased standard deviation needs two samples, so a batch of one would give
        # NaN weights and a NaN loss. `trainer.fast_dev_run` and a trailing batch both hit
        # this; fall back to uniform, which is plain MSE.
        if control.shape[0] < 2:
            return torch.ones_like(mu)

        if gene_mask is None:
            xbar = control.mean(dim=0, keepdim=True)
            sd = control.std(dim=0, keepdim=True)
        else:
            # Every quantity is taken over measured entries only. Two things go wrong
            # otherwise, both measured on a half-2025 half-Replogle batch:
            #
            #  * `control_exp` is NOT masked upstream -- `_apply_mask` zeroes the
            #    prediction and the target but not the input -- so on an unmeasured gene
            #    it holds `fill_profile`, identical for every aux cell. That contributes
            #    zero variance and deflates `sd`, inflating `z` for everyone; the 2025
            #    rows' weights correlate only 0.79 with the mask-aware ones.
            #  * `normalise` would divide by a row mean diluted by 10,624 phantom genes,
            #    scaling a Replogle row's surviving weights up by 1.65x and handing the
            #    aux sources even more of the gradient than they already take.
            mask = gene_mask.float()
            n = mask.sum(dim=0, keepdim=True)
            xbar = (control * mask).sum(dim=0, keepdim=True) / n.clamp_min(1.0)
            var = ((control - xbar) ** 2 * mask).sum(dim=0, keepdim=True) / (
                n - 1.0
            ).clamp_min(1.0)
            sd = var.sqrt()

        # `.abs()` is not optional. Without it a downregulated gene scores sigmoid(-z) -> 0
        # and is weighted *out* of the loss -- including a knockdown's own target gene,
        # which is the single largest effect in the data.
        z = (mu - xbar).abs() / (sd + self.eps)
        w = torch.sigmoid((z - self.threshold) / self.temperature)

        if gene_mask is None:
            if self.normalise:
                w = w / w.mean(dim=-1, keepdim=True).clamp_min(1e-12)
            return w

        # Mean 1 over the genes this sample's source measured, zero elsewhere. That keeps
        # a partial source's contribution scaled by its coverage exactly as plain MSE
        # scales it -- masked genes still count in the denominator of `.mean(-1)` in
        # `forward`, which is the deliberate behaviour documented on
        # `flow_lightning._apply_mask`.
        w = w * mask
        if self.normalise:
            measured = mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
            w = w / (w.sum(dim=-1, keepdim=True) / measured).clamp_min(1e-12)

        return w

    def forward(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        control_exp: torch.Tensor,
        gene_mask: torch.Tensor | None = None,
        **kwargs,
    ):
        """Weighted MSE, one weight vector per cell derived from the batch.

        Args:
            y_pred (torch.Tensor): Predicted values -- velocity for the flow model.
            y_true (torch.Tensor): Target values, the same quantity as ``y_pred``.
            control_exp (torch.Tensor): Control expression ``x_0``.
            gene_mask (torch.Tensor | None): 1 on genes the sample's source measured.
                ``None`` for single-panel runs, which never allocate one.

        Returns:
            torch.Tensor: The loss under ``reduction``.
        """
        w = self.weights(y_true, control_exp, gene_mask).to(y_pred.dtype)
        calc = (w * (y_pred - y_true) ** 2).mean(dim=-1)

        match self.reduction:
            case "sum":
                return calc.sum()
            case "mean":
                return calc.mean()
            case _:
                return calc


class BatchDiffExpError(nn.Module):
    """Cosine on the PSEUDOBULK differential-expression profile, per perturbation.

    The batch counterpart of `DiffExpError`, and the reason it exists is the same reason
    `BatchDEAwareMSELoss` replaced `DiffExpAwareMSELoss`: the per-cell version measures a
    quantity that is mostly noise.

    Why the per-cell version cannot work
    ------------------------------------
    `DiffExpError` compares `y_pred - control_exp` against `y_true - control_exp`, i.e. it
    scores each cell against ITS OWN paired control cell. Measured on the 2025 cache:

        |mu_p - xbar|^2   (perturbation signal)          17.9
        |x_0  - xbar|^2   (one control cell's deviation) 493.9
        signal fraction of (mu_p - x_0)                  3.5%

    So a PERFECT prediction scores only **0.151** under that cosine -- five sixths of its
    gradient pushes on one cell's sampling noise, which nothing can predict. The same
    arithmetic is why a per-cell delta-cosine error sits near 0.6 and moves so little (this
    was the since-removed `val/sample_diff_exp`; the epoch-level `val/diff_exp_agg` avoids it
    by pooling each perturbation's cells first).

    Averaging the cells of a perturbation first, and referencing the POOLED control mean
    rather than the paired cell, takes that ceiling from 0.151 to 1.0 -- and makes the
    quantity the one four of the six 2026 metrics actually score.

    What it computes
    ----------------
    Within the batch, group cells by `ko_id`; for each group with at least
    `min_group_size` cells::

        mu_hat_p = mean over the group of (control_exp + y_pred)   # the predicted profile
        d_pred   = mu_hat_p - xbar
        d_true   = mu_p     - xbar        where mu_p = y_true + control_exp
        loss     = mean over groups of (1 - cos(d_pred, d_true))

    If the model were perfect, `control_exp + y_pred = mu_p` for every cell and the group
    mean is exact -- so the residual noise here is the MODEL's error, averaged, which is
    precisely what should be penalised.

    `velocity_target` is the `y_true + control_exp = mu_p` identity, which holds only under
    `target_mode="perturbation_mean"`; set it False when `y_true` is already the profile.

    Batch size is load-bearing. The group mean averages the model's error over however many
    cells of that perturbation land in the batch, so at 240 training perturbations::

        batch 1024  ->  ~4 cells/pert   ->  ~13% signal in the pseudobulk
        batch 4096  ->  ~17 cells/pert  ->  ~38%

    Requires `ko_id` in the batch dict. `flow_lightning._shared_step` forwards it; a model
    whose step does not will raise rather than silently score something else.
    """

    def __init__(
        self,
        min_group_size: int = 2,
        eps: float = 1e-8,
        velocity_target: bool = True,
        reduction: str | None = "mean",
    ) -> None:
        super().__init__()
        if min_group_size < 1:
            raise ValueError(f"min_group_size must be >= 1, got {min_group_size}")
        self.min_group_size = min_group_size
        self.eps = eps
        self.velocity_target = velocity_target
        self.reduction = reduction

    def forward(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        control_exp: torch.Tensor,
        ko_id: torch.Tensor | None = None,
        gene_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        if ko_id is None:
            raise ValueError(
                "BatchDiffExpError needs `ko_id` to group cells by perturbation, but the "
                "training step did not pass it. Forward it from the batch dict."
            )
        # float32 throughout: under bf16-mixed a group mean and a cosine over 18,080
        # genes in bfloat16 lose most of their precision.
        control = control_exp.float()
        pred = y_pred.float()
        target = y_true.float()

        mu_true = target + control if self.velocity_target else target
        mu_pred = control + pred if self.velocity_target else pred
        xbar = control.mean(dim=0, keepdim=True)

        uniq, inverse = torch.unique(ko_id, return_inverse=True)
        n_groups = int(uniq.numel())
        counts = torch.zeros(n_groups, device=pred.device, dtype=pred.dtype)
        counts.index_add_(0, inverse, torch.ones_like(inverse, dtype=pred.dtype))

        sum_pred = torch.zeros(
            n_groups, pred.shape[1], device=pred.device, dtype=pred.dtype
        )
        sum_true = torch.zeros_like(sum_pred)
        sum_pred.index_add_(0, inverse, mu_pred)
        sum_true.index_add_(0, inverse, mu_true)
        denom = counts.unsqueeze(1).clamp_min(1.0)

        d_pred = sum_pred / denom - xbar
        d_true = sum_true / denom - xbar

        if gene_mask is not None:
            # `_apply_mask` zeroes the prediction and the target on unmeasured genes but
            # leaves `control_exp` holding `fill_profile` there, so mu would be a profile
            # that was never observed. Restrict the cosine to measured genes.
            mask = gene_mask.float()
            sum_mask = torch.zeros_like(sum_pred)
            sum_mask.index_add_(0, inverse, mask)
            keep = (sum_mask / denom > 0.5).to(d_pred.dtype)
            d_pred = d_pred * keep
            d_true = d_true * keep

        calc = 1.0 - torch.cosine_similarity(d_pred, d_true, dim=-1, eps=self.eps)

        # A singleton group's "pseudobulk" is one cell, i.e. the 3.5%-signal quantity this
        # loss exists to avoid. Drop those groups rather than let them back in.
        keep_group = counts >= self.min_group_size
        if not bool(keep_group.any()):
            return torch.zeros((), device=pred.device, dtype=y_pred.dtype)
        calc = calc[keep_group]

        if self.reduction == "sum":
            return calc.sum().to(y_pred.dtype)
        elif self.reduction == "mean":
            return calc.mean().to(y_pred.dtype)
        return calc.to(y_pred.dtype)


class BatchDeltaMagnitudeLoss(nn.Module):
    """Undershoot penalty on each perturbation's pseudobulk delta MAGNITUDE.

    The magnitude counterpart of :class:`BatchDiffExpError` (which matches the pseudobulk
    delta *direction* via cosine). It exists to counter the MSE shrinkage that leaves the
    flow model direction-correct but magnitude-collapsed: on the HepG2 cross-context val
    (2026-09-12) ``specific_cos`` reached 0.28 (right direction) while ``delta_ratio`` sat
    at 0.23 and ``genevar_ratio`` at 0.02. Under squared error a shrunk prediction is the
    risk-minimising point estimate when the per-perturbation response is uncertain, so the
    model hedges its magnitude; DE-weighting tilts *which genes* count but does nothing
    against the shrinkage itself.

    What it computes
    ----------------
    Within the batch, group cells by ``ko_id`` (as :class:`BatchDiffExpError`); per group
    of at least ``min_group_size`` cells, over measured genes, the group-mean VELOCITIES::

        v_pred = mean_group(y_pred)                        # predicted mean velocity
        v_true = mean_group(y_true)                        # true mean velocity (= mu_p - control_mean)
        m_pred = ||v_pred||_w ,  m_true = ||v_true||_w      # DE-weighted L2 magnitude
        loss   = mean_p relu(1 - m_pred / m_true)^2         # penalise UNDERSHOOT only

    Why the mean VELOCITY, not the delta ``mu_p - xbar``: referencing the batch control mean
    adds a per-group ``(control_group_mean - xbar)`` term that, at ~32 cells, is as large as
    the perturbation signal and does NOT scale with the prediction -- it inflates ``m_pred``
    so the ratio reads ~0.9 even for a 0.2x-shrunk prediction (measured), and the term barely
    pushes. The mean velocity has no such additive term: ``mean(alpha * y_true) ==
    alpha * mean(y_true)`` exactly, so the ratio reads the shrink factor directly, and a
    mean-collapsed model (velocity ~ ``globalmean - control``) has near-zero mean-velocity
    magnitude and is correctly penalised.

    Direction-preserving (the load-bearing property)
    -------------------------------------------------
    The gradient of ``||v_pred||`` w.r.t. ``v_pred`` is ``v_pred / ||v_pred||`` -- it points
    ALONG the model's own predicted mean velocity. So this term inflates the response the
    model already predicts; it can NOT rotate a right direction into a wrong one. That is the
    categorical difference from a contrastive term, which moves predictions apart in a
    direction-agnostic way and manufactured wrong-direction magnitude that broke nmae
    (real-panel -0.283). Magnitude is grown only along the chosen axis, and only when it
    undershoots -- ``relu`` zeroes the term once ``m_pred >= m_true``, so it can never drive
    an overshoot (which would blow up nmae, the property the KernelFiLM convex-hull bound
    protects). ``genevar`` recovers as a byproduct: scaling each perturbation up along its
    own (already correctly-differing, per ``specific_cos``) direction raises the
    between-perturbation variance without ever explicitly forcing perturbations apart.

    DE-weighted magnitude
    ---------------------
    With ``de_weighted`` (default), the norm is weighted by the same SD-relative DE weight
    as :class:`BatchDEAwareMSELoss` (reused by composition), so magnitude is matched WHERE
    THE DE GENES ARE, not on high-abundance counting noise. Per-cell weights are pooled to a
    per-group per-gene weight. Only the weight's per-gene *shape* matters: a global scale
    cancels in the ratio ``m_pred / m_true``.

    Controls and null perturbations are dropped
    -------------------------------------------
    A group whose true magnitude is below ``min_rel_true_mag`` times the batch-mean true
    magnitude is skipped. This excludes the non-targeting control (whose true delta is ~0 by
    construction, and which the ratio would otherwise ask the model to give a spurious
    magnitude) and stabilises the ratio denominator -- without needing to know which
    ``ko_id`` is the control, so it stays batch-only.

    Requires ``ko_id`` in the batch dict, like :class:`BatchDiffExpError`; raises otherwise.

    Parameters
    ----------
    threshold, temperature, de_eps : float
        Passed to the internal :class:`BatchDEAwareMSELoss` used only for its ``.weights()``.
        Ignored when ``de_weighted`` is False.
    de_weighted : bool
        Weight the magnitude by the SD-relative DE weight (True) or use a plain L2 norm
        (False).
    min_group_size : int
        Groups with fewer cells are dropped -- a one-cell "pseudobulk" is the noisy quantity
        this loss exists to avoid.
    min_rel_true_mag : float
        Drop a group whose true magnitude is below this fraction of the batch-mean true
        magnitude (excludes controls / null perturbations). ``0`` keeps every group.
    eps : float
        Numerical floor for the norms and the ratio denominator.
    velocity_target : bool
        The ``y_true + control_exp = mu_p`` identity; True for the flow model.
    reduction : {"mean", "sum", None}
        Over the kept perturbation groups.
    """

    def __init__(
        self,
        threshold: float = 0.15,
        temperature: float = 0.05,
        de_eps: float = 1e-2,
        de_weighted: bool = True,
        min_group_size: int = 2,
        min_rel_true_mag: float = 0.1,
        eps: float = 1e-8,
        velocity_target: bool = True,
        reduction: str | None = "mean",
    ) -> None:
        super().__init__()
        if min_group_size < 1:
            raise ValueError(f"min_group_size must be >= 1, got {min_group_size}")
        if not 0.0 <= min_rel_true_mag < 1.0:
            raise ValueError(
                f"min_rel_true_mag must be in [0, 1), got {min_rel_true_mag}"
            )
        self.de_weighted = de_weighted
        self.min_group_size = min_group_size
        self.min_rel_true_mag = min_rel_true_mag
        self.eps = eps
        self.velocity_target = velocity_target
        self.reduction = reduction
        # Reused only for its no-grad `.weights()`. `normalise=False` because a per-group
        # weight scale cancels in the m_pred/m_true ratio -- only the per-gene shape counts.
        self._de = BatchDEAwareMSELoss(
            threshold=threshold,
            temperature=temperature,
            eps=de_eps,
            normalise=False,
            velocity_target=velocity_target,
        )

    def extra_repr(self) -> str:
        return (
            f"de_weighted={self.de_weighted}, min_group_size={self.min_group_size}, "
            f"min_rel_true_mag={self.min_rel_true_mag}, "
            f"velocity_target={self.velocity_target}"
        )

    def forward(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        control_exp: torch.Tensor,
        ko_id: torch.Tensor | None = None,
        gene_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        if ko_id is None:
            raise ValueError(
                "BatchDeltaMagnitudeLoss needs `ko_id` to group cells by perturbation, but "
                "the training step did not pass it. Forward it from the batch dict."
            )
        # float32 throughout: a group mean over 18,080 genes in bfloat16 loses most of its
        # precision, and the magnitudes here are small differences.
        pred = y_pred.float()
        target = y_true.float()

        uniq, inverse = torch.unique(ko_id, return_inverse=True)
        n_groups = int(uniq.numel())
        counts = torch.zeros(n_groups, device=pred.device, dtype=pred.dtype)
        counts.index_add_(0, inverse, torch.ones_like(inverse, dtype=pred.dtype))
        denom = counts.unsqueeze(1).clamp_min(1.0)

        sum_pred = torch.zeros(
            n_groups, pred.shape[1], device=pred.device, dtype=pred.dtype
        )
        sum_true = torch.zeros_like(sum_pred)
        sum_pred.index_add_(0, inverse, pred)
        sum_true.index_add_(0, inverse, target)

        # Group-mean VELOCITIES, NOT deltas referenced to xbar. Referencing xbar would add
        # a per-group (control_group_mean - xbar) term that, at ~32 cells, is as large as the
        # perturbation signal -- it inflates ||d_pred|| so the ratio sits near 1 even for a
        # heavily shrunk prediction (measured: a 0.2x velocity gave ratio ~0.9). The mean
        # velocity has no such additive term: mean(alpha * y_true) = alpha * mean(y_true)
        # exactly, so the ratio reads the shrink directly, and a mean-collapsed model (velocity
        # ~ globalmean - control) has near-zero mean-velocity magnitude -> correctly penalised.
        d_pred = sum_pred / denom
        d_true = (sum_true / denom).detach()  # target: no gradient

        # Per-gene weight for the norm, pooled from the same SD-relative DE weight the
        # anchor MSE uses. Uniform (ones) when `de_weighted` is off.
        if self.de_weighted:
            w_cell = self._de.weights(y_true, control_exp, gene_mask).to(pred.dtype)
            sum_w = torch.zeros_like(sum_pred)
            sum_w.index_add_(0, inverse, w_cell)
            w_grp = sum_w / denom
        else:
            w_grp = torch.ones_like(sum_pred)

        if gene_mask is not None:
            # `_apply_mask` zeroes the prediction/target on unmeasured genes but leaves
            # `control_exp` holding `fill_profile`, so restrict the norm to genes this
            # group's source actually measured.
            mask = gene_mask.float()
            sum_mask = torch.zeros_like(sum_pred)
            sum_mask.index_add_(0, inverse, mask)
            keep = (sum_mask / denom > 0.5).to(pred.dtype)
            d_pred = d_pred * keep
            d_true = d_true * keep
            w_grp = w_grp * keep

        m_pred = torch.sqrt((w_grp * d_pred.pow(2)).sum(dim=-1) + self.eps)
        m_true = torch.sqrt((w_grp * d_true.pow(2)).sum(dim=-1) + self.eps)

        keep_group = counts >= self.min_group_size
        if self.min_rel_true_mag > 0 and bool(keep_group.any()):
            floor = self.min_rel_true_mag * m_true[keep_group].mean()
            keep_group = keep_group & (m_true > floor)

        if not bool(keep_group.any()):
            return torch.zeros((), device=pred.device, dtype=y_pred.dtype)

        ratio = m_pred[keep_group] / m_true[keep_group].clamp_min(self.eps)
        calc = torch.relu(1.0 - ratio).pow(
            2
        )  # undershoot only; zero once m_pred >= m_true

        if self.reduction == "sum":
            return calc.sum().to(y_pred.dtype)
        elif self.reduction == "mean":
            return calc.mean().to(y_pred.dtype)
        return calc.to(y_pred.dtype)
