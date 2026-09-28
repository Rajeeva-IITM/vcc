"""Lightning module for the rectified-flow model in ``src.models.components.flow_model``.

Differs from :class:`src.models.vcc_lightning.VCCModule` in what is being regressed. That
module predicts the perturbed profile directly and its loss compares ``y_pred`` to ``y``.
This one builds the interpolant ``x_t = (1-t) x_0 + t x_1``, asks the net for the velocity
at that point, and compares it to ``x_1 - x_0``. Predictions come from integrating the
learned field out of the control.

One consequence worth remembering when reading the logs: ``*/loss`` here is a velocity
error and is **not** comparable to the ``*/mse`` of the other models. The comparable
numbers are the ``val/sample_*`` metrics, which integrate a real sample and score it
against the perturbed cell.
"""

from typing import Literal, override

import torch
import torchmetrics
from lightning.pytorch import LightningModule
from lightning.pytorch.utilities import grad_norm
from torch import nn
from torch.optim import Optimizer

from src.models.components.flow_model import FlowCellModel


class VCCModule(LightningModule):
    """Train a velocity field between control and perturbed expression.

    Parameters
    ----------
    net : FlowCellModel | nn.Module
        Velocity field. Must expose ``forward(t, x_t, ko_vec)`` and ``sample(x0, ko_vec,
        num_steps)``.
    loss_fn : nn.Module
        Applied to ``(velocity_pred, velocity_target)``. Note that the DE-aware losses in
        ``loss_functions`` assume their first argument is an expression profile and are
        **not** valid here -- see ``config/model/model_flow.yaml``.

        When the batch carries ``gene_mask`` -- i.e. when training across datasets whose
        gene panels differ -- both arguments are zeroed on unmeasured genes before the
        loss sees them. Since ``d/dv (m*v - m*u)^2 = 2 m^2 (v - u)`` vanishes wherever
        ``m == 0``, no loss function has to know the mask exists.
    optimizer : torch.optim.Optimizer
        Partially-instantiated optimizer.
    scheduler : torch.optim.lr_scheduler.LRScheduler
        Partially-instantiated scheduler.
    num_sampling_steps : int
        Euler steps used for the ``val/sample_*`` metrics and for prediction.
    pert_input_key : {"ko_vec", "ko_id"}
        Which batch key carries the perturbation. ``"ko_vec"`` hands the net a
        precomputed embedding, as every model before the learnable table did.
        ``"ko_id"`` hands it a row index instead, for a net whose ``gene_embedding`` is
        a :class:`~src.models.components.flow_model.LearnableGeneEmbedding`. Stated in
        config rather than sniffed from the net's type: it is one line and it says what
        it means.
    embedding_lr : float | None
        Learning rate for the embedding table alone, with weight decay switched off.
        ``None`` leaves it in the main parameter group. This is the knob that decides
        whether the table *refines* the prior or *replaces* it -- 9,233 of 18,080 rows
        receive gradient when training on the Replogle data, so the remaining rows keep
        their prior value and ``ko_processor`` has to stay fluent in both. Watch
        ``emb/drift``.
    """

    def __init__(
        self,
        net: FlowCellModel | nn.Module,
        loss_fn: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        num_sampling_steps: int = 4,
        pert_input_key: Literal["ko_vec", "ko_id"] = "ko_vec",
        embedding_lr: float | None = None,
    ) -> None:
        super().__init__()

        self.save_hyperparameters(
            logger=False,
        )  # Lightning will ask to ignore nn.Module but it'll cause problems

        self.net = net
        self.criterion = loss_fn
        self.num_sampling_steps = num_sampling_steps
        self.pert_input_key = pert_input_key
        self.embedding_lr = embedding_lr

        self.mae = torchmetrics.functional.mean_absolute_error
        self.mse = torchmetrics.functional.mean_squared_error

        # Per-perturbation delta accumulator for the epoch-level direction/diversity
        # metrics. `None` outside a validation/test epoch so `_sample_metrics` skips it;
        # set to a `{prefix: {ko_id: [sum_dp, sum_dt, count]}}` dict at epoch start.
        self._pert_acc: dict[str, dict[int, list]] | None = None

    def _embedding_parameters(self) -> list[torch.nn.Parameter]:
        """Parameters of the net's learnable embedding table, if it has one."""
        table = getattr(self.net, "gene_embedding", None)
        return (
            [] if table is None else [p for p in table.parameters() if p.requires_grad]
        )

    def configure_optimizers(self):
        """Configuring the optimizers.

        An embedding table always gets its own parameter group with ``weight_decay=0``,
        and at ``embedding_lr`` when that is set. Weight decay is wrong here for a
        structural reason: ``nn.Embedding`` produces a *dense* gradient, so AdamW decays
        every row it holds whether or not that row was in the batch -- including the
        8,847 rows no dataset perturbs, whose whole job is to still hold their prior
        value at prediction time.

        Measured, the drift this causes is ~1e-4 relative over a full run, so it is not
        what would break a submission. What it does break is :meth:`drift`, which
        identifies trained rows by having moved at all: past roughly 120 steps the decay
        exceeds float32 resolution and all 18,080 rows register as moved, so
        ``emb/n_moved`` stops being able to tell you that only the perturbed genes are
        learning.
        """
        embedding = self._embedding_parameters()
        if not embedding:
            optimizer: Optimizer = self.hparams["optimizer"](self.parameters())
        else:
            ids = {id(p) for p in embedding}
            rest = [p for p in self.parameters() if id(p) not in ids]
            group = {"params": embedding, "weight_decay": 0.0}
            if self.embedding_lr is not None:
                group["lr"] = self.embedding_lr
            optimizer = self.hparams["optimizer"]([{"params": rest}, group])

        scheduler: torch.optim.lr_scheduler.LRScheduler = self.hparams["scheduler"](
            optimizer,
            # total_steps=self.trainer.estimated_stepping_batches # For OneCycleLR
        )

        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def forward(
        self,
        t: torch.Tensor,
        x_t: torch.Tensor,
        ko_vec: torch.Tensor,
        context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Evaluate the velocity field.

        Args:
            t (torch.Tensor): Times, shape ``(B, 1)``.
            x_t (torch.Tensor): Point on the path, shape ``(B, n_genes)``.
            ko_vec (torch.Tensor): Perturbation embeddings, shape ``(B, embed_dim)``.
            context (torch.Tensor | None): Cell-context vector, shape ``(B, context_dim)``;
                ``None`` unless the datamodule emits ``context_vec`` and the net uses it.

        Returns:
            torch.Tensor: Velocity, shape ``(B, n_genes)``.
        """
        return self.net(t, x_t, ko_vec, context)

    @staticmethod
    def _apply_mask(
        X: dict[str, torch.Tensor], pred: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Zero both sides on genes the sample's dataset did not measure.

        ``gene_mask`` is present only when the run mixes datasets with different gene
        panels; single-panel runs never allocate it and this is a no-op. Masked genes
        still count in the mean's denominator, which scales each sample's loss by its
        coverage -- a source measuring 41% of the panel therefore contributes about 41%
        of the gradient a fully-measured one does. That is wanted, not a defect: it
        carries proportionally less information.

        Args:
            X (dict): The batch's input dict.
            pred (torch.Tensor): Prediction.
            target (torch.Tensor): Target.

        Returns:
            tuple: ``(pred, target)``, masked if a mask was supplied.
        """
        mask = X.get("gene_mask")
        if mask is None:
            return pred, target
        mask = mask.to(pred.dtype)
        return pred * mask, target * mask

    def _sample_times(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        deterministic: bool,
    ) -> torch.Tensor:
        """Draw one time per cell, shape ``(B, 1)``.

        ``device`` and ``dtype`` are taken from the batch
        Validation and test use a fixed midpoint grid instead of fresh randomness. This is
        not cosmetic -- ``config/callbacks/model_checkpoint.yaml`` monitors ``val/loss``,
        so re-drawing ``t`` every pass would let RNG decide which checkpoint is kept.
        """
        if deterministic:
            times = (
                torch.arange(batch_size, device=device, dtype=dtype) + 0.5
            ) / batch_size
        else:
            times = torch.rand(batch_size, device=device, dtype=dtype)

        return times.unsqueeze(-1)

    def _flow_targets(
        self,
        batch: tuple[dict[str, torch.Tensor], torch.Tensor],
        deterministic: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build the interpolant and the conditional velocity target.

        Args:
            batch: The ``({"ko_vec", "exp_vec"}, y)`` pair from the datamodule.
            deterministic (bool): Use the fixed time grid instead of sampling.

        Returns:
            tuple: ``(t, x_t, u)`` where ``x_t = (1-t) x_0 + t x_1`` and ``u = x_1 - x_0``.
        """
        X, y = batch
        x_0 = X["exp_vec"]  # control
        x_1 = y  # perturbed

        t = self._sample_times(x_0.shape[0], x_0.device, x_0.dtype, deterministic)

        x_t = (1.0 - t) * x_0 + t * x_1
        u = x_1 - x_0

        return t, x_t, u

    def _shared_step(
        self,
        batch: tuple[dict[str, torch.Tensor], torch.Tensor],
        step: Literal["train/{}", "val/{}", "test/{}"],
        deterministic: bool,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Run the flow-matching objective and its cheap per-step metrics.

        Args:
            batch: The ``({"ko_vec", "exp_vec"}, y)`` pair from the datamodule.
            step: Log-key template.
            deterministic (bool): Passed through to the time sampler.

        Returns:
            tuple: ``(loss, log_dict)``.
        """
        X, _ = batch
        t, x_t, u = self._flow_targets(batch, deterministic)

        v_pred = self.forward(t, x_t, X[self.pert_input_key], X.get("context_vec"))
        v_pred, u = self._apply_mask(X, v_pred, u)

        # kwargs kept for CompositeLoss compatibility; plain MSE ignores them.
        loss: torch.Tensor = self.criterion(
            v_pred,
            u,
            control_exp=X["exp_vec"],
            # Deliberately "ko_vec" and not `pert_input_key`: this is an input to the
            # DE-aware losses, not the lookup, and they want a vector either way.
            gene_embeddings=X["ko_vec"],
            # None on single-panel runs, which never allocate a mask. A loss that derives
            # its weights from batch statistics needs it: `_apply_mask` zeroes the
            # prediction and the target on unmeasured genes but leaves `control_exp`
            # holding `fill_profile` there, so an unmasked statistic is taken over a
            # profile that does not exist. Every other loss swallows it via **kwargs.
            gene_mask=X.get("gene_mask"),
            # Needed by any loss that aggregates per perturbation within the batch
            # (`BatchDiffExpError`). Absent from the batch only for models that address
            # the perturbation purely by vector; those losses raise rather than guess.
            ko_id=X.get("ko_id"),
        )

        logs = {
            step.format("loss"): loss,
        }

        # A CompositeLoss stashes its per-term (unweighted) values on the last forward;
        # surface each so wandb shows whether a term (e.g. BatchLaplacianReg) is inert
        # relative to the MSE. Logged here so this one method covers train/val/test; kept
        # off the progress bar to avoid clutter. A plain loss has no stash -> no-op.
        components = getattr(self.criterion, "last_components", None)
        if components:
            self.log_dict(
                {step.format(f"loss_{k}"): v for k, v in components.items()},
                prog_bar=False,
                logger=True,
                on_epoch=True,
                add_dataloader_idx=False,
            )

        # A KernelFiLM conditioner stashes the normalised entropy of its anchor weights:
        # ~1 = weights uniform (collapsing toward the blind baseline), ~0 = one anchor per
        # cell (memorising). This is the direct readout of the axis the kernel head targets;
        # a plain MLP ko_processor has no such attribute -> no-op. Lives on `ko_processor`
        # for single-FiLM and on `trunk` (which mirrors it) for the deep per-layer variant.
        conditioner = getattr(self.net, "ko_processor", None) or getattr(
            self.net, "trunk", None
        )
        entropy = getattr(conditioner, "last_entropy", None)
        if entropy is not None:
            self.log_dict(
                {step.format("kernel_entropy"): entropy},
                prog_bar=False,
                logger=True,
                on_epoch=True,
                add_dataloader_idx=False,
            )

        return loss, logs

    def _sample_metrics(
        self,
        batch: tuple[dict[str, torch.Tensor], torch.Tensor],
        step: Literal["train/{}", "val/{}", "test/{}"],
    ) -> dict[str, torch.Tensor]:
        """Integrate the field from the control and score the sample against the truth.

        This costs ``num_sampling_steps`` extra forward passes, so it runs on validation
        and test only. These are the numbers comparable to the other models' metrics and
        to the leaderboard; the flow loss is not.

        Args:
            batch: The ``({"ko_vec", "exp_vec"}, y)`` pair from the datamodule.
            step: Log-key template.

        Returns:
            dict: Metrics keyed by ``<stage>/sample_<name>``.
        """
        X, y = batch

        y_pred = self.net.sample(
            X["exp_vec"],
            X[self.pert_input_key],
            num_steps=self.num_sampling_steps,
            context=X.get("context_vec"),
        )
        y_pred, y = self._apply_mask(X, y_pred, y)
        # The control has to be masked too: `sample_delta`/`delta_ratio` and the epoch-level
        # delta accumulator all read it as a baseline, and an unmasked control against a masked
        # prediction would score an unmeasured gene as having moved by its full control value.
        control, _ = self._apply_mask(X, X["exp_vec"], X["exp_vec"])

        # Predicted and true deltas from the control (log fold changes). Computed once and
        # cast to float32 -- magnitudes over 18k genes are unstable in bf16.
        dp = (y_pred - control).float()
        dt = (y - control).float()

        pred_delta = dp.abs().mean()
        true_delta = dt.abs().mean()

        metrics = {
            step.format("sample_mse"): self.mse(y_pred, y),
            step.format("sample_mae"): self.mae(y_pred, y),
            # Mean absolute delta from the control. Near zero means the flow has collapsed
            # to "predict no change"; that is the safe floor, not a win.
            step.format("sample_delta"): pred_delta,
            # Magnitude reference + ratio (pred/true |delta|): ~1 calibrated, << 1 collapsed,
            # > 1 overshoot. Both sides use the same per-cell control, so they are comparable.
            step.format("true_delta"): true_delta,
            step.format("delta_ratio"): pred_delta / true_delta.clamp_min(1e-8),
        }

        # Accumulate per-perturbation mean deltas for the epoch-level direction/diversity
        # metrics (see `_flush_pert_metrics`). The val loader is ordered by perturbation and
        # not shuffled, so a batch holds ~one perturbation -- specificity and between-
        # perturbation diversity are undefined *within* a batch and only make sense once the
        # whole epoch's perturbations are pooled. `ko_id` groups the rows; absent for
        # vector-only models, so guard on it.
        ids = X.get("ko_id")
        if ids is not None and self._pert_acc is not None:
            prefix = step.format("").strip("/")  # "val" | "val_prior" | "test"
            acc = self._pert_acc.setdefault(prefix, {})
            uniq, inv = torch.unique(ids, return_inverse=True)
            for j, gid in enumerate(uniq.tolist()):
                m = inv == j
                s_dp, s_dt, c = dp[m].sum(0), dt[m].sum(0), int(m.sum())
                if gid in acc:
                    acc[gid][0] += s_dp
                    acc[gid][1] += s_dt
                    acc[gid][2] += c
                else:
                    acc[gid] = [s_dp, s_dt, c]

        return metrics

    def _flush_pert_metrics(self) -> None:
        """Compute epoch-level direction/diversity from the pooled per-perturbation deltas.

        Called at the end of validation/test. For each stage prefix, the accumulator holds
        one summed (pred delta, true delta, count) per perturbation seen in the epoch. From
        the per-perturbation MEAN deltas ``MP``/``MT`` (P perturbations x G genes):

        * ``diff_exp_agg`` -- ``1 - cos`` of pred vs true mean delta, averaged over
          perturbations. The per-cell control noise is gone; this is the clean per-
          perturbation direction, the closest in-loop proxy for the leaderboard DE cosine.
        * ``specific_cos`` -- the same cosine after subtracting the ACROSS-perturbation
          common delta from both. The shared shift is the easy, transferable component;
          this isolates the perturbation-specific direction (memory: it sits near blind).
        * ``genevar_ratio`` -- pred / true between-perturbation variance. ~1 means the model
          spreads perturbations apart as much as the truth does; << 1 is diversity collapse.
        """
        if not self._pert_acc:
            return
        for prefix, acc in self._pert_acc.items():
            if len(acc) < 2:  # specificity/diversity need >= 2 perturbations
                continue
            MP = torch.stack(
                [v[0] / v[2] for v in acc.values()]
            )  # (P, G) pred mean delta
            MT = torch.stack(
                [v[1] / v[2] for v in acc.values()]
            )  # (P, G) true mean delta
            agg = torch.cosine_similarity(MP, MT).mean()
            spec = torch.cosine_similarity(MP - MP.mean(0), MT - MT.mean(0)).mean()
            ratio = MP.var(0, unbiased=False).mean() / MT.var(
                0, unbiased=False
            ).mean().clamp_min(1e-8)
            self.log_dict(
                {
                    f"{prefix}/diff_exp_agg": 1.0 - agg,
                    f"{prefix}/specific_cos": spec,
                    f"{prefix}/genevar_ratio": ratio,
                    f"{prefix}/n_perturbations": float(len(acc)),
                },
                logger=True,
                add_dataloader_idx=False,
            )
        self._pert_acc = {}

    def training_step(
        self, batch: tuple[dict[str, torch.Tensor], torch.Tensor], batch_idx: int
    ):
        """Training step."""
        loss, logs = self._shared_step(batch, "train/{}", deterministic=False)
        self.log_dict(logs, prog_bar=True, logger=True, on_epoch=True)

        return loss

    def validation_step(
        self,
        batch: tuple[dict[str, torch.Tensor], torch.Tensor],
        batch_idx: int,
        dataloader_idx: int = 0,
    ):
        """Validation step.

        A second dataloader, when the datamodule supplies one, holds perturbations whose
        *embedding rows were never trained* -- see ``prior_val_fraction`` in
        ``vcc_anndata_module``. Its metrics are logged under ``val_prior/`` and are the
        check on whether learning the table destroyed the prior geometry the untrained
        rows still rely on. ``add_dataloader_idx=False`` keeps the key names clean, so
        the prefix carries the meaning rather than a positional suffix.
        """
        prefix = "val/{}" if dataloader_idx == 0 else "val_prior/{}"
        loss, logs = self._shared_step(batch, prefix, deterministic=True)
        self.log_dict(
            logs, prog_bar=True, logger=True, on_epoch=True, add_dataloader_idx=False
        )
        self.log_dict(
            self._sample_metrics(batch, prefix),
            prog_bar=True,
            logger=True,
            on_epoch=True,
            add_dataloader_idx=False,
        )

        return loss

    @override
    def on_validation_epoch_start(self) -> None:
        """Open the per-perturbation delta accumulator for the epoch."""
        self._pert_acc = {}

    @override
    def on_validation_epoch_end(self) -> None:
        """Flush epoch-level metrics and log embedding drift.

        ``emb/n_moved`` should equal the number of perturbations in the training split.
        If it is larger, something is receiving gradient that should not be; if
        ``emb/drift`` climbs, the trained rows are walking away from the prior geometry
        that the untrained ones -- 28 of the 300 leaderboard targets -- still sit in.
        """
        self._flush_pert_metrics()
        table = getattr(self.net, "gene_embedding", None)
        drift = table.drift() if hasattr(table, "drift") else None
        if drift is not None:
            mean_drift, n_moved = drift
            self.log_dict(
                {"emb/drift": float(mean_drift), "emb/n_moved": float(n_moved)},
                logger=True,
            )

    @override
    def on_test_epoch_start(self) -> None:
        """Open the per-perturbation delta accumulator for the epoch."""
        self._pert_acc = {}

    @override
    def on_test_epoch_end(self) -> None:
        """Flush the epoch-level direction/diversity metrics."""
        self._flush_pert_metrics()

    def test_step(
        self, batch: tuple[dict[str, torch.Tensor], torch.Tensor], batch_idx: int
    ):
        """Testing step."""
        loss, logs = self._shared_step(batch, "test/{}", deterministic=True)
        self.log_dict(logs, prog_bar=True, logger=True, on_epoch=True)
        self.log_dict(
            self._sample_metrics(batch, "test/{}"),
            prog_bar=True,
            logger=True,
            on_epoch=True,
        )

        return loss

    def predict_step(self, batch, batch_ix):
        """Integrate from the control and return the predicted expression.

        Returns the same thing as ``vcc_lightning.VCCModule.predict_step`` -- a
        ``(B, n_genes)`` expression tensor -- so ``src/train.py`` can ``torch.cat`` the
        results unchanged.
        """
        X, _ = batch

        return self.net.sample(
            X["exp_vec"],
            X[self.pert_input_key],
            num_steps=self.num_sampling_steps,
            context=X.get("context_vec"),
        )

    @override
    def on_before_optimizer_step(self, optimizer: Optimizer) -> None:
        """Log gradient norms."""
        self.log_dict(grad_norm(self, norm_type=2))
