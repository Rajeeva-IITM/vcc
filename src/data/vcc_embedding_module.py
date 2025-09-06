import gc
import warnings
from pathlib import Path
from typing import Literal

import numpy as np
import polars as pl
import polars.selectors as cs
import rich
import torch
from icecream import ic
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler

from src.utils.data import read_data

# from sklearn.model_selection import train_test_split
from src.utils.gene_train_test_split import gene_train_test_split

console = rich.console.Console()


class VCCEmbeddingDataset(Dataset):
    """
    VCCEmbeddingDataset is a PyTorch Dataset for handling gene expression and knockout data with gene embeddings.
    Args:
        control_expression (pl.DataFrame): DataFrame containing control (unperturbed) gene expression data.
        perturbed_genes (np.ndarray): Array of gene names corresponding to the perturbed (knockout) genes for each sample.
        gene_embeddings (pl.DataFrame): DataFrame containing gene embeddings with a 'gene_name' column for indexing.
        ko_expression (pl.DataFrame | None): DataFrame containing knockout gene expression data. Optional, used during training/validation.
        stage (str | None): Stage indicator, e.g., 'train', 'val', or 'predict'. Determines dataset behavior.
    Attributes:
        control_expression (np.ndarray): Numpy array of control expression data.
        perturbed_genes (np.ndarray): Array of perturbed gene names.
        gene_embeddings (pd.DataFrame): Pandas DataFrame of gene embeddings indexed by gene name.
        ko_expression (np.ndarray | None): Numpy array of knockout expression data, or None if not provided.
        stage (str | None): Current stage of the dataset.

    Methods:
        __len__(): Returns the number of samples in the dataset.
        __getitem__(index): Retrieves a single sample consisting of:
            - A dictionary with:
                'ko_vec': Embedding vector(s) for the perturbed gene(s).
                'exp_vec': Control expression vector for the sample.
            - The corresponding knockout expression vector (or an empty list if in 'predict' stage).

    Raises:
        ValueError: If 'stage' is None.
        AssertionError: If input data dimensions do not match as expected.
    """

    def __init__(
        self,
        control_expression: torch.Tensor,
        perturbed_genes: np.typing.NDArray[np.str_],
        gene_embeddings: dict[str, torch.Tensor],
        ko_expression: torch.Tensor | None,  # In case of predict stage
        stage: Literal["predict", None],
    ) -> None:
        super().__init__()

        self.stage = stage
        self.is_predict_stage = stage == "predict"
        self.control_expression = control_expression
        self.ko_expression = ko_expression
        self.perturbed_genes = perturbed_genes
        self.gene_embeddings = gene_embeddings

    def __len__(self):
        """
        Returns the length of the dataset
        """
        return len(self.perturbed_genes)

    # def _process_gene_embeddings(
    #     self, gene_embeddings: torch.Tensor, perturbed_genes: np.typing.NDArray[np.str_]
    # ):
    #     self.perturbed_genes = perturbed_genes

    #     # Create a dictionary of tensors
    #     # self.gene_embeddings = torch.tensor(gene_embeddings.select(cs.numeric()).to_numpy()).to(torch.float32)
    #     # gene_names = gene_embeddings['gene_name'].to_numpy()
    #     self.gene_to_idx: dict[str, int] = {gene:[idx] for idx,gene in enumerate(self.gene_indices)}

    def __getitem__(
        self, index: int
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor | list[None]]:
        """
        Retrieves a single data sample
        """

        exp_input = self.control_expression[index, :]
        gene_input = self.gene_embeddings[self.perturbed_genes[index]]

        if self.is_predict_stage:
            pred_input = []
        else:
            pred_input = (
                self.ko_expression[index, :]
                if self.ko_expression is not None
                else [None]
            )

        return {"ko_vec": gene_input, "exp_vec": exp_input}, pred_input


class VCCDataModule(LightningDataModule):
    """Data module for VCC data. Two main inputs must be provided:
        - The gene embedding data (does not include non-targeting samples)
        - The expression data for each sample

    Parameters
    ----------

    train_exp_data_path : str | Path
        Path to the dataset
    train_ko_data_path : str | Path
        Path to ko_gene_data
    control_data_path: str | Path
        Path to control expression
    test_exp_data_path: str | Path
        Path to test expression (not available but included for consistency), defaults to None
    test_ko_data_path: str | Path
        Path to test Knockout data, defaults to None
    seed : int, optional
        Random seed, by default 42
    num_workers : int, optional
        Number of workers, by default 4
    batch_size : int, optional
        Batch size, by default 64
    test_size : float, optional
        Fraction for test data, by default 0.2
    """

    def __init__(
        self,
        train_exp_data_path: str | Path,
        gene_embedding_path: str | Path,
        train_ko_data_path: str | Path,
        control_data_path: str | Path,
        test_exp_data_path: str | Path | None = None,
        test_ko_data_path: str | Path | None = None,
        seed: int = 42,
        num_workers: int = 4,
        batch_size: int = 64,
        test_size: float = 0.2,
    ):
        super().__init__()

        self.train_exp_data_path = train_exp_data_path
        self.gene_embedding_path = gene_embedding_path
        self.train_ko_data_path = train_ko_data_path
        self.control_data_path = control_data_path  # Common for both train and test
        self.test_exp_data_path = (
            test_exp_data_path  # Not really available so just pass in control data
        )
        self.test_ko_data_path = test_ko_data_path
        self.seed = seed
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.test_size = test_size

    def setup(self, stage: str | None = None) -> None:
        """
        Pre
        gene_embedding_path : str | Path
        Path to      embedding data.pares and splits the dataset for training and testing.

        Reads expression and knockout gene data, processes and samples control data,
        creates the dataset, and splits it into training and test subsets.
        """

        # Won't change with stage
        control_data = read_data(self.control_data_path)
        gene_embeddings = read_data(self.gene_embedding_path)
        gene_embeddings_torch = gene_embeddings.select(cs.numeric()).to_torch()
        gene_indices: np.typing.NDArray[np.str_] = gene_embeddings[
            "gene_name"
        ].to_numpy()

        gene_embeddings_dict = {
            gene: gene_embeddings_torch[idx] for idx, gene in enumerate(gene_indices)
        }

        match self.trainer.precision:
            case "16-mixed" | "16-true":
                dtype = torch.float16
            case "32-true":
                dtype = torch.float32
            case "64-true":
                dtype = torch.float64
            case "bf16-mixed" | "bf16-true":
                dtype = torch.bfloat16
            case _:
                dtype = torch.float32

        match stage:
            case "predict":
                console.log("Setting up data for prediction")

                if (self.test_exp_data_path is None) | (self.test_ko_data_path is None):
                    raise TypeError(
                        "`test_exp_data_path` and `test_ko_data_path` must be present"
                    )

                console.log("Reading data")
                # test_exp_data = read_data(self.test_exp_data_path).select(cs.numeric())  # type: ignore
                # Pass in file similar pert Validation Counts
                test_ko_data = read_data(self.test_ko_data_path)
                perturbed_genes = np.repeat(
                    test_ko_data["target_gene"].to_numpy().flatten(),
                    repeats=test_ko_data["n_cells"].to_numpy().flatten(),
                )

                length = len(perturbed_genes)  # Number of non control indices

                # Since number of control data points is smaller, sampling with replacement
                if control_data.shape[0] < length:
                    warnings.warn(
                        "Number of samples in control data is lower than the expression. Performing sampling with replacement"
                    )
                    control_data = control_data.select(cs.numeric()).sample(
                        length,  # type: ignore
                        with_replacement=True,
                        seed=self.seed,
                    )

                self.test_data = VCCEmbeddingDataset(
                    control_expression=(control_data.to_torch().to(dtype)),
                    perturbed_genes=perturbed_genes,
                    gene_embeddings=gene_embeddings_dict,
                    ko_expression=None,
                    stage=stage,
                )

                del control_data, test_ko_data
                gc.collect()

            case _:
                console.log("Setting up data")
                ko_exp_data = (
                    read_data(self.train_exp_data_path)
                    .sort("sample_index")
                    .select(cs.numeric())
                )
                ko_gene_data = (
                    read_data(self.train_ko_data_path)
                    .sort("sample_index")
                    .filter(pl.col("target_gene") != "non-targeting")["target_gene"]
                    .to_numpy()
                )

                assert ko_exp_data.shape[0] == ko_gene_data.shape[0], (
                    f"row count mismatch between data and ko_exp_data {ko_exp_data.shape[0]} and {ko_gene_data.shape[0]}"
                )

                # Separate knockout data and control data
                console.log("Expression data processing")

                length = len(ko_gene_data)  # Number of non control indices

                # Since number of control data points is smaller, sampling with replacement
                if control_data.shape[0] < length:
                    warnings.warn(
                        "Number of samples in control data is lower than the expression. Performing sampling with replacement"
                    )
                    control_data = control_data.select(cs.numeric()).sample(
                        length,  # type: ignore
                        with_replacement=True,
                        seed=self.seed,
                    )

                # Assembling the pieces
                console.log("Creating Dataset")
                self.data = VCCEmbeddingDataset(
                    control_expression=(control_data.to_torch().to(dtype)),
                    perturbed_genes=ko_gene_data,
                    gene_embeddings=gene_embeddings_dict,
                    ko_expression=(ko_exp_data).to_torch(),
                    stage=stage,
                )

                console.log("Data splitting")

                gene_names = ko_gene_data.flatten()

                self.train_index, self.val_index = gene_train_test_split(
                    gene_names,  # type: ignore
                    test_size=self.test_size,
                    seed=self.seed,
                )

                console.log("Split data generation")
                self.train_data: Subset[VCCEmbeddingDataset] = Subset(
                    self.data, self.train_index
                )
                self.val_data: Subset[VCCEmbeddingDataset] = Subset(
                    self.data, self.val_index
                )

                del ko_gene_data, ko_exp_data  # Clean up
                gc.collect()

                console.log("Setup Done")

    def train_dataloader(self):
        """
        Creates and returns a DataLoader for the training dataset.

        Returns:
            DataLoader: DataLoader  for the train dataset
        """
        console.log("Creating Training Dataloader")

        # Defining stratified sampling

        unique, counts = np.unique(
            self.train_data.dataset.perturbed_genes[self.train_index],
            return_counts=True,
        )
        weight_dict = dict(zip(unique, [1 / count for count in counts]))
        weights = [
            weight_dict[gene]
            for gene in self.train_data.dataset.perturbed_genes[self.train_index]
        ]
        sampler = WeightedRandomSampler(weights, len(self.train_data))

        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            # shuffle=True,
            pin_memory=True,
            sampler=sampler,
        )

    def val_dataloader(self):
        """
        Creates and returns a DataLoader for the validation dataset.

        Returns:
            DataLoader: DataLoader instance for the validation data.
        """
        console.log("Creating Validation Dataloader")
        return DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
        )

    def test_dataloader(self):  # TODO: Need to change this
        """
        Creates and returns a DataLoader for the test dataset.

        Returns:
            DataLoader: DataLoader for the test dataset.
        """

        console.log("Creating Validation Dataloader")
        return DataLoader(
            self.test_data, batch_size=1, num_workers=self.num_workers, shuffle=False
        )

    def predict_dataloader(self):
        """Creates and returns the prediction dataloader"""
        console.log("Creating prediction dataloader")
        return DataLoader(
            self.test_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )


if __name__ == "__main__":
    dm = VCCDataModule(
        train_exp_data_path="/home/rajeeva/Project/vcc_data/processed-data/training_data-counts_uint.parquet",
        gene_embedding_path="/home/rajeeva/Project/vcc_data/gene_embeddings/PCA-train_expression.parquet",
        train_ko_data_path="/home/rajeeva/Project/vcc_data/training_data-row_metadata.csv",
        control_data_path="/home/rajeeva/Project/vcc_data/processed-data/control_exp_data_uint.parquet",
    )
    dm.setup(stage="fit")

    i = 0
    for X, y in dm.val_dataloader():
        print(len(X), y.shape)
        ic(X, y)
        i += 1
        if i > 5:
            break
