# from typing import Any, Dict, List, Tuple

from typing import override

import torch
import torchmetrics
from lightning.pytorch import LightningModule
from lightning.pytorch.utilities import grad_norm
from torch import nn
from torch.optim import Optimizer

# from icecream import ic
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
        # lr: float,
        # max_lr: float,
        # weight_decay: float,
        loss_fn: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
    ) -> None:
        super().__init__()

        self.save_hyperparameters(logger=False, ignore=["net"])

        self.net = net
        self.criterion = loss_fn
        # self.lr = lr
        # self.max_lr = max_lr
        # self.weight_decay = weight_decay

        self.train_mae = torchmetrics.MeanAbsoluteError()
        # self.train_cosine = torchmetrics.CosineSimilarity(reduction="mean")
        self.train_mse = torchmetrics.MeanSquaredError()
        # self.train_corr = torchmetrics.SpearmanCorrCoef()

        self.val_mae = torchmetrics.MeanAbsoluteError()
        # self.val_cosine = torchmetrics.CosineSimilarity(reduction="mean")
        self.val_mse = torchmetrics.MeanSquaredError()
        # self.val_corr = torchmetrics.SpearmanCorrCoef()

        self.test_mae = torchmetrics.MeanAbsoluteError()
        # self.test_cosine = torchmetrics.CosineSimilarity(reduction="mean")
        self.test_mse = torchmetrics.MeanSquaredError()
        # self.test_corr = torchmetrics.SpearmanCorrCoef()

    # def configure_optimizers(self) -> Dict[str, Any]:
    #     """Choose what optimizers and learning-rate schedulers to use in your optimization.
    #     Normally you'd need one. But in the case of GANs or similar you might have multiple.

    #     :Returns: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
    #     """
    #     optimizer = self.hparams.optimizer(params=self.trainer.model.parameters()) # type: ignore
    #     if self.hparams.scheduler is not None: # type: ignore
    #         scheduler = self.hparams.scheduler(optimizer=optimizer) # type: ignore
    #         return {
    #             "optimizer": optimizer,
    #             "lr_scheduler": {
    #                 "scheduler": scheduler,
    #                 "monitor": "val/loss",
    #                 "interval": "epoch",
    #                 "frequency": 1,
    #             },
    #         }
    #     return {"optimizer": optimizer}

    # def configure_optimizers(self):
    #     """Returns the configured optimizers and schedulers for training the model.

    #     Returns:
    #         Tuple[List[torch.optim.Optimizer], List[torch.optim.lr_scheduler._LRScheduler]]: A tuple containing the configured optimizers and schedulers.
    #     """
    #     optimizer = torch.optim.AdamW(
    #         self.parameters(),
    #         lr=self.lr,
    #         weight_decay=self.weight_decay,
    #     )
    #     scheduler = torch.optim.lr_scheduler.OneCycleLR(
    #         optimizer,
    #         max_lr=self.max_lr,
    #         total_steps=self.trainer.estimated_stepping_batches,
    #     )

    #     return [optimizer], [scheduler]
    #

    def configure_optimizers(self):
        """Configuring the optimizers."""

        optimizer: Optimizer = self.hparams["optimizer"](self.parameters())

        scheduler: torch.optim.lr_scheduler.LRScheduler = self.hparams["scheduler"](
            optimizer,
            # total_steps=self.trainer.estimated_stepping_batches # For OneCycleLR
            T_0=10,  # For Cosine Annealing
        )

        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def forward(self, X):
        """
        Performs a forward pass through the network.
        Args:
            X (torch.Tensor): Input tensor.
        Returns:
            torch.Tensor: Output tensor from the network.
        """

        return self.net.forward(X)

    def training_step(self, batch, batch_idx):
        """
        Training step
        """
        # ic(batch)
        X, y = batch
        y_pred = self.forward(X)
        # ic(y, y.shape)
        # ic(y_pred, y_pred.shape)
        loss = self.criterion(y_pred, y)

        self.train_mae.update(y_pred, y)
        self.train_cosine = torchmetrics.functional.cosine_similarity(
            y_pred, y, reduction="mean"
        )
        self.train_mse.update(y_pred, y)

        # self.train_corr.update(y_pred.T, y.T)

        corr = torchmetrics.functional.spearman_corrcoef(
            y_pred.T, y.T
        ).mean()  # Only creates an approximation but this accumulates in the gpu if left uncomputed

        self.log(
            "train/loss", loss, on_epoch=True, on_step=True, prog_bar=True, logger=True
        )

        self.log(
            "train/mae",
            self.train_mae,
            on_epoch=True,
            on_step=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "train/cosine",
            self.train_cosine,
            on_epoch=True,
            on_step=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "train/mse",
            self.train_mse,
            on_epoch=True,
            on_step=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "train/corr", corr, on_epoch=True, on_step=True, prog_bar=True, logger=True
        )

        return loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step
        """
        X, y = batch
        y_pred = self.forward(X)
        loss = self.criterion(y_pred, y)

        self.val_mae.update(y_pred, y)
        self.val_cosine = torchmetrics.functional.cosine_similarity(
            y_pred, y, reduction="mean"
        )
        self.val_mse.update(y_pred, y)
        # self.val_corr.update(y_pred.T, y.T)
        # corr = self.val_corr.compute() # Only creates an approximation but this accumulates in the gpu if left uncomputed
        corr = torchmetrics.functional.spearman_corrcoef(
            y_pred.T, y.T
        ).mean()  # Only creates an approximation but this accumulates in the gpu if left uncomputed

        self.log(
            "val/loss", loss, on_epoch=True, on_step=False, prog_bar=True, logger=True
        )

        self.log(
            "val/mae",
            self.val_mae,
            on_epoch=True,
            on_step=False,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "val/cosine",
            self.val_cosine,
            on_epoch=True,
            on_step=False,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "val/mse",
            self.val_mse,
            on_epoch=True,
            on_step=False,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "val/corr",
            corr,
            on_epoch=True,
            on_step=False,
            prog_bar=True,
            logger=True,
        )

        return loss

    def test_step(self, batch, batch_idx):
        """
        Testing step
        """
        X, y = batch
        y_pred = self.forward(X)
        loss = self.criterion(y_pred, y)

        self.test_mae.update(y_pred, y)
        self.test_cosine = torchmetrics.functional.cosine_similarity(
            y_pred, y, reduction="mean"
        )
        # self.test_mse.update(y_pred, y)
        # self.val_corr.update(y_pred.T, y.T)
        corr = torchmetrics.functional.spearman_corrcoef(
            y_pred.T, y.T
        ).mean()  # Only creates an approximation but this accumulates in the gpu if left uncomputed

        self.log(
            "test/loss", loss, on_epoch=True, on_step=False, prog_bar=True, logger=True
        )

        self.log(
            "test/mae",
            self.test_mae,
            on_epoch=True,
            on_step=False,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "test/cosine",
            self.test_cosine,
            on_epoch=True,
            on_step=False,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "test/mse",
            self.test_mse,
            on_epoch=True,
            on_step=False,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "test/corr",
            corr,
            on_epoch=True,
            on_step=False,
            prog_bar=True,
            logger=True,
        )

        return loss

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
