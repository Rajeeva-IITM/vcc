# from typing import Any, Dict, List, Tuple

from typing import Literal, override

import torch
import torchmetrics
from lightning.pytorch import LightningModule
from lightning.pytorch.utilities import grad_norm
from torch import nn
from torch.optim import Optimizer

from src.models.components import loss_functions as lf
from src.models.components.basic_vcc_model import CellModel

# from src.models.components.loss_functions import CompositeLoss


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
        net: CellModel | nn.Module,
        loss_fn: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
    ) -> None:
        super().__init__()

        self.save_hyperparameters(
            logger=False,
        )  # Lightning will ask to ignore nn.Module but it'll cause problems

        self.net = net
        self.criterion = loss_fn
        self.mae = torchmetrics.functional.mean_absolute_error
        self.mse = torchmetrics.functional.mean_squared_error
        self.jaccard_loss = lf.SoftJaccardLoss()
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

    def forward(self, X) -> torch.Tensor:
        """
        Performs a forward pass through the network.
        Args:
            X (torch.Tensor): Input tensor.
        Returns:
            torch.Tensor: Output tensor from the network.
        """

        return self.net(X)

    def _calculate_losses(
        self,
        batch: tuple[dict[str, torch.Tensor], torch.Tensor],
        y_pred: torch.Tensor,
        step: Literal["train/{}", "val/{}", "test/{}"],
    ):
        """
        Calculates the three losses: Primary, contrastive and consistency
        """
        X, y = batch
        loss: torch.Tensor = self.criterion(
            y_pred, y, control_exp=X["exp_vec"], gene_embeddings=X["ko_vec"]
        )  # For some losses

        return {
            step.format("loss"): loss,
        }

    def _calculate_metrics(
        self,
        batch: tuple[dict[str, torch.Tensor], torch.Tensor],
        y_pred: torch.Tensor,
        step: Literal["train/{}", "val/{}", "test/{}"],
    ):
        X, y = batch

        mae = self.mae(y_pred, y)
        mse = self.mse(y_pred, y)
        jaccard_loss = self.jaccard_loss.forward(y_pred, y, X["exp_vec"], X["ko_vec"])
        cosine = torchmetrics.functional.cosine_similarity(y_pred, y, reduction="mean")

        corr = torchmetrics.functional.spearman_corrcoef(
            y_pred.T, y.T
        ).mean()  # Only creates an approximation but this accumulates in the gpu if left uncomputed

        gene_level_variance = self.train_genevar.forward(y_pred, y)
        diff_exp = self.train_diffexp.forward(y_pred, y, X["exp_vec"])
        psl = self.train_psl.forward(y_pred, y, X["ko_vec"])

        return {
            step.format("mae"): mae,
            step.format("mse"): mse,
            step.format("jaccard_loss"): jaccard_loss,
            step.format("cosine"): cosine,
            step.format("corr"): corr,
            step.format("genevar"): gene_level_variance,
            step.format("diff_exp"): diff_exp,
            step.format("pertloss"): psl,
        }

    def training_step(
        self, batch: tuple[dict[str, torch.Tensor], torch.Tensor], batch_idx
    ):
        """
        Training step
        """
        X, y = batch
        y_pred = self.forward(X)
        losses = self._calculate_losses(batch, y_pred, "train/{}")
        metrics = self._calculate_metrics(batch, y_pred, "train/{}")
        self.log_dict(losses, prog_bar=True, logger=True, on_epoch=True)
        self.log_dict(metrics, prog_bar=True, logger=True, on_epoch=True)

        return losses["train/loss"]

    def validation_step(
        self, batch: tuple[dict[str, torch.Tensor], torch.Tensor], batch_idx
    ):
        """
        Validation step
        """
        X, y = batch
        y_pred = self.forward(X)
        losses = self._calculate_losses(batch, y_pred, "val/{}")
        metrics = self._calculate_metrics(batch, y_pred, "val/{}")

        self.log_dict(losses, prog_bar=True, logger=True, on_epoch=True)
        self.log_dict(metrics, prog_bar=True, logger=True, on_epoch=True)

        return losses["val/loss"]

    def test_step(self, batch: tuple[dict[str, torch.Tensor], torch.Tensor], batch_idx):
        """
        Testing step
        """
        X, y = batch
        y_pred = self.forward(X)
        losses = self._calculate_losses(batch, y_pred, "test/{}")
        metrics = self._calculate_metrics(batch, y_pred, "test/{}")

        self.log_dict(losses, prog_bar=True, logger=True, on_epoch=True)
        self.log_dict(metrics, prog_bar=True, logger=True, on_epoch=True)

        return losses["test/loss"]

    def predict_step(self, batch, batch_ix):
        X, _ = batch
        y_pred = self.forward(X)

        return y_pred

    @override
    def on_before_optimizer_step(self, optimizer: Optimizer) -> None:
        self.log_dict(grad_norm(self, norm_type=2))


if __name__ == "__main__":
    # from lightning.pytorch import Trainer

    # from src.data.vcc_datamodule import VCCDataModule

    # torch.set_float32_matmul_precision("high")

    # net = CellModel(  # ko_input, exp_input and decoder_ouput must have same size
    #     ko_processor_args={
    #         "input_size": 18080,
    #         "hidden_layers": [16, 8],
    #         "output_size": 4,
    #         "dropout": 0.2,
    #         "activation": "relu",
    #     },
    #     exp_processor_args={
    #         "input_size": 18080,
    #         "hidden_layers": [18, 8],
    #         "output_size": 4,
    #         "dropout": 0.2,
    #         "activation": "relu",
    #     },
    #     concat_processor_args={
    #         "input_size": 8,  # 4 + 4 from previous outputs
    #         "hidden_layers": [8],
    #         "output_size": 6,
    #         "dropout": 0.2,
    #         "activation": "relu",
    #     },
    #     decoder_args={
    #         "input_size": 6,
    #         "hidden_layers": [6],
    #         "output_size": 18080,
    #         "dropout": 0.2,
    #         "activation": "relu",
    #     },
    # )

    # model = VCCModule(net=net, lr=1e-3, max_lr=1e-2, weight_decay=1e-5)
    # # ic(model.dtype)

    # datamodule = VCCDataModule(
    #     data_path="/storage/bt20d204/vcc-data/vcc_data/processed-data/training_data-counts_uint.parquet",
    #     ko_data_path="/storage/bt20d204/vcc-data/vcc_data/processed-data/training_data-gene_ko_uint.parquet",
    # )

    # trainer = Trainer(
    #     accelerator="gpu", devices=1, fast_dev_run=100, precision="16-mixed"
    # )

    # trainer.fit(model, datamodule=datamodule)
    pass
