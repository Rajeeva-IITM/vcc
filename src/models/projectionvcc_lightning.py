from pathlib import Path
from typing import Literal, override

import torch
import torchmetrics
from lightning.pytorch import LightningModule
from lightning.pytorch.utilities import grad_norm
from torch import nn
from torch.optim import Optimizer

from src.models.components import loss_functions as lf
from src.models.components.basic_vcc_model import ProjectorCellModel


class VCCModule(LightningModule):
    """
    PyTorch Lightning module for training, validating, and testing a cell model with composite loss and multiple regression metrics.
        net (CellModel | nn.Module): The neural network model to be trained.
        loss_fn (nn.Module): Loss function to be optimized
        optimizer (torch.optim.Optimizer): Optimizer
        scheduler (torch.optim.lr_scheduler.LRScheduler): Learning Rate Scheduler


    Attributes:
        net (CellModel): The neural network model.
        criterion (CompositeLoss): Loss function combining multiple objectives.
        train_mae, val_mae, test_mae (torchmetrics.MeanAbsoluteError): MAE metrics for train/val/test.
        train_cosine, val_cosine, test_cosine (torchmetrics.CosineSimilarity): Cosine similarity metrics.
        train_mse, val_mse, test_mse (torchmetrics.MeanSquaredError): MSE metrics for train/val/test.

    Methods:
        configure_optimizers: Sets up optimizer and learning rate scheduler.
        forward: Performs a forward pass through the model.
        training_step: Computes loss and updates metrics for a training batch.
        validation_step: Computes loss and updates metrics for a validation batch.
        test_step: Computes loss and updates metrics for a test batch.
    """

    def __init__(
        self,
        net: ProjectorCellModel,
        # lr: float,
        # max_lr: float,
        # weight_decay: float,
        primary_loss: nn.Module,
        contrastive_loss: lf.WeightedContrastiveLoss | nn.Module,
        contrastive_weight: float,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
    ) -> None:
        super().__init__()

        self.save_hyperparameters(
            logger=False,
        )  # Lightning will ask to ignore nn.Module but it'll cause problems

        self.net = net
        self.criterion = primary_loss
        self.contrastive_loss = contrastive_loss
        self.contrastive_weight = contrastive_weight

        self.mae = torchmetrics.functional.mean_absolute_error
        self.mse = torchmetrics.functional.mean_squared_error

        # self.train_mae = torchmetrics.MeanAbsoluteError()
        # # self.train_cosine = torchmetrics.CosineSimilarity(reduction="mean")
        # self.train_mse = torchmetrics.MeanSquaredError()
        # self.train_corr = torchmetrics.SpearmanCorrCoef()
        self.train_genevar = lf.BatchVariance()
        self.train_diffexp = lf.DiffExpError()
        self.train_psl = lf.WeightedContrastiveLoss()

        # self.val_mae = torchmetrics.MeanAbsoluteError()
        # # self.val_cosine = torchmetrics.CosineSimilarity(reduction="mean")
        # self.val_mse = torchmetrics.MeanSquaredError()
        # self.val_corr = torchmetrics.SpearmanCorrCoef()
        self.val_genevar = lf.BatchVariance()
        self.val_diffexp = lf.DiffExpError()
        self.val_psl = lf.WeightedContrastiveLoss()

        # self.test_mae = torchmetrics.MeanAbsoluteError()
        # # self.test_cosine = torchmetrics.CosineSimilarity(reduction="mean")
        # self.test_mse = torchmetrics.MeanSquaredError()
        # self.test_corr = torchmetrics.SpearmanCorrCoef()
        self.test_genevar = lf.BatchVariance()
        self.test_diffexp = lf.DiffExpError()
        self.test_psl = lf.WeightedContrastiveLoss()

    def configure_optimizers(self):
        """Configuring the optimizers."""

        optimizer: Optimizer = self.hparams["optimizer"](self.parameters())

        scheduler: torch.optim.lr_scheduler.LRScheduler = self.hparams["scheduler"](
            optimizer,
            # total_steps=self.trainer.estimated_stepping_batches # For OneCycleLR
        )

        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def forward(self, X: dict[str, torch.Tensor]):
        """
        Performs a forward pass through the network.
        Args:
            X (torch.Tensor): Input tensor.
        Returns:
            torch.Tensor: Output tensor from the network.
        """

        return self.net.forward(X)

    def _calculate_losses(
        self,
        batch: tuple[dict[str, torch.Tensor], torch.Tensor],
        y_pred: torch.Tensor,
        latent: torch.Tensor,
        step: Literal["train/{}", "val/{}", "test/{}"],
    ):
        """
        Calculates the three losses: Primary, contrastive and consistency
        """
        X, y = batch
        primary_loss: torch.Tensor = self.criterion(
            y_pred, y, control_exp=X["exp_vec"], gene_embeddings=X["ko_vec"]
        )  # For some losses
        contrastive_loss: torch.Tensor = self.contrastive_loss.forward(
            y_pred=latent, y_true=latent, gene_embeddings=X["ko_vec"]
        )
        loss = primary_loss + contrastive_loss * self.contrastive_weight

        return {
            step.format("loss"): loss,
            step.format("primary_loss"): primary_loss,
            step.format("contrastive_loss"): contrastive_loss,
        }

    def _calculate_metrics(
        self,
        batch: tuple[dict[str, torch.Tensor], torch.Tensor],
        y_pred: torch.Tensor,
        latent: torch.Tensor,
        step: Literal["train/{}", "val/{}", "test/{}"],
    ):
        X, y = batch

        mae = self.mae(y_pred, y)
        mse = self.mse(y_pred, y)
        cosine = torchmetrics.functional.cosine_similarity(y_pred, y, reduction="mean")

        corr = torchmetrics.functional.spearman_corrcoef(
            y_pred.T, y.T
        ).mean()  # Only creates an approximation but this accumulates in the gpu if left uncomputed

        gene_level_variance = self.train_genevar.forward(y_pred, y)
        diff_exp = self.train_diffexp.forward(y_pred, y, X["exp_vec"])
        psl = self.train_psl.forward(latent, y, X["ko_vec"])

        return {
            step.format("mae"): mae,
            step.format("mse"): mse,
            step.format("cosine"): cosine,
            step.format("corr"): corr,
            step.format("genevar"): gene_level_variance,
            step.format("diff_exp"): diff_exp,
            step.format("pertloss"): psl,
        }

    def training_step(
        self, batch: tuple[dict[str, torch.Tensor], torch.Tensor], batch_idx: int
    ):
        """
        Training step
        """
        # ic(batch)
        X, y = batch
        y_pred, latent = self.forward(X)
        losses = self._calculate_losses(batch, y_pred, latent, "train/{}")
        metrics = self._calculate_metrics(batch, y_pred, latent, "train/{}")

        self.log_dict(losses, prog_bar=True, logger=True, on_epoch=True)
        self.log_dict(metrics, prog_bar=True, logger=True, on_epoch=True)

        return losses["train/loss"]

    def validation_step(
        self, batch: tuple[dict[str, torch.Tensor], torch.Tensor], batch_idx: int
    ):
        """
        Validation step
        """
        X, y = batch
        y_pred, latent = self.forward(X)
        losses = self._calculate_losses(batch, y_pred, latent, "val/{}")
        metrics = self._calculate_metrics(batch, y_pred, latent, "val/{}")

        self.log_dict(losses, prog_bar=True, logger=True, on_epoch=True)
        self.log_dict(metrics, prog_bar=True, logger=True, on_epoch=True)

        return losses["val/loss"]

    def test_step(
        self, batch: tuple[dict[str, torch.Tensor], torch.Tensor], batch_idx: int
    ):
        """
        Testing step
        """
        X, y = batch
        y_pred, latent = self.forward(X)
        losses = self._calculate_losses(batch, y_pred, latent, "test/{}")
        metrics = self._calculate_metrics(batch, y_pred, latent, "test/{}")

        self.log_dict(losses, prog_bar=True, logger=True, on_epoch=True)
        self.log_dict(metrics, prog_bar=True, logger=True, on_epoch=True)

        return losses["test/loss"]

    def predict_step(self, batch, batch_ix):
        """
        Prediction step
        """
        X, _ = batch
        y_pred, projection = self.forward(X)

        return y_pred, projection

    @override
    def on_before_optimizer_step(self, optimizer: Optimizer) -> None:
        self.log_dict(grad_norm(self, norm_type=2))


class VCCModulewithConsistency(VCCModule):
    """
    Same as the previous case, but contains a consistency loss term
    """

    def __init__(
        self,
        net: ProjectorCellModel,
        primary_loss: nn.Module,
        contrastive_loss: lf.WeightedContrastiveLoss | nn.Module,
        contrastive_weight: float,
        consistency_loss: nn.Module,
        consistency_loss_weight: float,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        control_std: torch.Tensor | str | Path,
    ) -> None:
        super().__init__(
            net,
            primary_loss,
            contrastive_loss,
            contrastive_weight,
            optimizer,
            scheduler,
        )

        self.consistency_loss: nn.Module = consistency_loss
        self.consistency_loss_weight: float = consistency_loss_weight

        if isinstance(control_std, (str, Path)):
            self.control_std: torch.Tensor = torch.load(control_std)
        else:
            self.control_std = control_std

        self.mae = torchmetrics.functional.mean_absolute_error
        self.mse = torchmetrics.functional.mean_squared_error

    def _add_noise_to_inputs(self, batch: tuple[dict[str, torch.Tensor], torch.Tensor]):
        """
        Adds noise to inputs for measuring consistency

        Args:
            batch (tuple[dict[str, torch.Tensor], torch.Tensor]): Input batch

        Returns:
            tuple[dict[str, torch.Tensor], torch.Tensor], Noise added batch
        """
        X, y = batch
        noise = torch.normal(
            mean=torch.zeros(X["exp_vec"].size(-1)),
            std=self.control_std,
        ).to(dtype=X["exp_vec"].dtype, device=X["exp_vec"].device)

        augmented_input = ({"exp_vec": X["exp_vec"] + noise, "ko_vec": X["ko_vec"]}, y)
        return augmented_input

    def _calculate_losses(
        self,
        batch: tuple[dict[str, torch.Tensor], torch.Tensor],
        y_pred: torch.Tensor,
        y_pred_noise: torch.Tensor,
        latent: torch.Tensor,
        step: Literal["train/{}", "val/{}", "test/{}"],
    ):
        """
        Calculates the three losses: Primary, contrastive and consistency
        """
        X, y = batch
        primary_loss: torch.Tensor = self.criterion(
            y_pred, y, control_exp=X["exp_vec"], gene_embeddings=X["ko_vec"]
        )  # For some losses
        contrastive_loss: torch.Tensor = self.contrastive_loss.forward(
            y_pred=latent, y_true=latent, gene_embeddings=X["ko_vec"]
        )
        consistency_loss: torch.Tensor = self.consistency_loss.forward(
            y_pred, y_pred_noise
        )
        loss = (
            primary_loss
            + contrastive_loss * self.contrastive_weight
            + consistency_loss * self.consistency_loss_weight
        )

        return {
            step.format("loss"): loss,
            step.format("primary_loss"): primary_loss,
            step.format("contrastive_loss"): contrastive_loss,
            step.format("consistency_loss"): consistency_loss,
        }

    def _calculate_metrics(
        self,
        batch: tuple[dict[str, torch.Tensor], torch.Tensor],
        y_pred: torch.Tensor,
        latent: torch.Tensor,
        step: Literal["train/{}", "val/{}", "test/{}"],
    ):
        X, y = batch

        mae = self.mae(y_pred, y)
        mse = self.mse(y_pred, y)
        cosine = torchmetrics.functional.cosine_similarity(y_pred, y, reduction="mean")

        corr = torchmetrics.functional.spearman_corrcoef(
            y_pred.T, y.T
        ).mean()  # Only creates an approximation but this accumulates in the gpu if left uncomputed

        gene_level_variance = self.train_genevar.forward(y_pred, y)
        diff_exp = self.train_diffexp.forward(y_pred, y, X["exp_vec"])
        psl = self.train_psl.forward(latent, y, X["ko_vec"])

        return {
            step.format("mae"): mae,
            step.format("mse"): mse,
            step.format("cosine"): cosine,
            step.format("corr"): corr,
            step.format("genevar"): gene_level_variance,
            step.format("diff_exp"): diff_exp,
            step.format("pertloss"): psl,
        }

    def configure_optimizers(self):
        """Configuring the optimizers."""

        optimizer: Optimizer = self.hparams["optimizer"](self.parameters())

        scheduler: torch.optim.lr_scheduler.LRScheduler = self.hparams["scheduler"](
            optimizer,
            # total_steps=self.trainer.estimated_stepping_batches # For OneCycleLR
        )

        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def training_step(
        self, batch: tuple[dict[str, torch.Tensor], torch.Tensor], batch_idx: int
    ):
        """
        Training step
        """
        X, y = batch
        X_noise, _ = self._add_noise_to_inputs(batch)
        y_pred, latent = self.forward(X)
        y_pred_noise, latent_noise = self.forward(X_noise)

        losses = self._calculate_losses(batch, y_pred, y_pred_noise, latent, "train/{}")
        metrics = self._calculate_metrics(batch, y_pred, latent, "train/{}")

        self.log_dict(losses, prog_bar=True, logger=True, on_epoch=True)
        self.log_dict(metrics, prog_bar=True, logger=True, on_epoch=True)

        return losses["train/loss"]

    def validation_step(
        self, batch: tuple[dict[str, torch.Tensor], torch.Tensor], batch_idx: int
    ):
        """
        Validation step
        """
        X, y = batch
        X_noise, _ = self._add_noise_to_inputs(batch)
        y_pred, latent = self.forward(X)
        y_pred_noise, latent_noise = self.forward(X_noise)

        losses = self._calculate_losses(batch, y_pred, y_pred_noise, latent, "val/{}")
        metrics = self._calculate_metrics(batch, y_pred, latent, "val/{}")

        self.log_dict(losses, prog_bar=True, logger=True)
        self.log_dict(metrics, prog_bar=True, logger=True)

        return losses["val/loss"]

    def test_step(
        self, batch: tuple[dict[str, torch.Tensor], torch.Tensor], batch_idx: int
    ):
        """
        Testing step
        """
        X, y = batch
        X_noise, _ = self._add_noise_to_inputs(batch)
        y_pred, latent = self.forward(X)
        y_pred_noise, latent_noise = self.forward(X_noise)

        losses = self._calculate_losses(batch, y_pred, y_pred_noise, latent, "test/{}")
        metrics = self._calculate_metrics(batch, y_pred, latent, "test/{}")

        self.log_dict(losses, prog_bar=True, logger=True)
        self.log_dict(metrics, prog_bar=True, logger=True)

        return losses["test/loss"]
