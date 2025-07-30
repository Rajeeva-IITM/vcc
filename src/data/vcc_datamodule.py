import gc
from pathlib import Path
from typing import Dict, Tuple, Union

import numpy as np
import polars as pl
import polars.selectors as cs
import rich
import torch
from lightning.pytorch import LightningDataModule
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset

console = rich.console.Console()


def read_data(path: Union[Path, str], format: str | None = None) -> pl.DataFrame:
    """Reads the data from the given path.

    Args:
        path (Union[Path, str]): The path to the observation data.
        format (str, optional): The format of the data. Defaults to "parquet".

    Returns:
        pl.DataFrame: The data read from the path.
    """

    if format is None:
        path = str(path)
        if path.endswith(".feather"):
            format = "feather"
        elif path.endswith(".parquet"):
            format = "parquet"
        elif path.endswith(".csv"):
            format = "csv"
        else:
            raise ValueError(
                "Could not infer file format from extension. Please specify the format."
            )

    match format:
        case "feather":
            return pl.read_feather(path)
        case "parquet":
            return pl.read_parquet(path)
        case "csv":
            return pl.read_csv(path)
        case _:
            raise NotImplementedError(
                "File format not supported. Most be one of 'feather', 'parquet', 'csv'"
            )


class VCCDataset(Dataset):
    def __init__(
        self,
        ko_expression: pl.DataFrame,
        control_expression: pl.DataFrame,
        ko_gene_data: pl.DataFrame,
        # dtype: torch.dtype | = torch.float32,
    ) -> None:
        """Dataset for holding the expression to_numpy() and the knockout vector

        Parameters
        ----------
        ko_expression : pl.DataFrame
            Expression when gene is knocked out (to be predicted)
        control_expression : pl.DataFrame
            Expression of control data (input to the model)
        ko_gene_data : pl.DataFrame
            Binary vector encoding the knocked out gene
        """
        super().__init__()

        assert (
            ko_expression.shape[0]
            == control_expression.shape[0]
            == ko_gene_data.shape[0]
        ), (
            f"Data not of the same length\
                {ko_expression.shape[0], control_expression.shape[0], ko_gene_data.shape[0]}"
        )

        # print(ko_expression)
        # print(ko_gene_data)
        # print(control_expression)

        # Converting to pandas because polars misbehaves with multiprocessing
        self.ko_expression = ko_expression.to_numpy()
        self.control_expression = control_expression.to_numpy()
        self.ko_gene_data = ko_gene_data.to_numpy()
        # self.dtype = dtype

        del ko_expression, control_expression, ko_gene_data
        gc.collect()

    def __len__(self):
        """
        Returns the number of samples in the dataset.
        Returns:
            int: The number of rows in the ko_expression attribute.
        """

        return self.ko_expression.shape[0]

    def __getitem__(self, index: int) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Retrieves a single data sample from the dataset at the specified index.
        Args:
            index (int): Index of the sample to retrieve.
        Returns:
            Tuple[Dict[str, torch.Tensor], torch.Tensor]:
                A tuple containing a dictionary with knockout gene vector ('ko_vec') and control expression vector ('exp_vec'),
                and the knockout expression tensor as the target.
        """

        ko_input = torch.from_numpy(self.ko_gene_data[index, :]).to(torch.float16)
        exp_input = torch.from_numpy(self.control_expression[index, :]).to(
            torch.float16
        )
        pred_input = torch.from_numpy(self.ko_expression[index, :]).to(torch.float16)

        return {"ko_vec": ko_input, "exp_vec": exp_input}, pred_input


class VCCDataModule(LightningDataModule):
    """Data module for VCC data. Two main inputs must be provided:
        - The gene knockout data (does not include non-targeting samples)
        - The expression data for each sample

    Parameters
    ----------

    data_path : Union[str, Path]
        Path to the dataset
    ko_data_path : Union[str, Path]
        Path to ko_gene_data
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
        data_path: Union[str, Path],
        ko_data_path: Union[str, Path],
        seed: int = 42,
        num_workers: int = 4,
        batch_size: int = 64,
        test_size: float = 0.2,
    ):
        super().__init__()

        self.data_path = data_path
        self.ko_data_path = ko_data_path
        self.seed = seed
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.test_size = test_size

    def setup(self, stage: str | None = None) -> None:
        """
        Prepares and splits the dataset for training and testing.

        Reads expression and knockout gene data, processes and samples control data,
        creates the dataset, and splits it into training and test subsets.
        """
        console.log("Setting up data")

        data = read_data(self.data_path).sort("sample_index")
        ko_gene_data = read_data(self.ko_data_path).sort(
            "sample_index"
        )  # Order both of them

        # assert data.shape[0] == ko_exp_data.shape[0], (
        #     "row count mismatch between data and ko_exp_data"
        # )

        # Separate knockout data and control data
        console.log("Expression data processing")
        control_indices = data.filter(
            ~pl.col("sample_index").is_in(ko_gene_data["sample_index"].implode())
        )["sample_index"]  # identify indices not in ko_gene
        length = ko_gene_data.shape[0]  # Number of non control indices

        # Since number of control data points is smaller, sampling with replacement
        self.control_exp_data = (
            data.filter(pl.col("sample_index").is_in(control_indices.implode()))
            .select(cs.numeric())
            .sample(
                length,  # type: ignore
                with_replacement=True,
                seed=self.seed,
            )
        )

        self.expression_data = data.filter(
            ~pl.col("sample_index").is_in(control_indices.implode())
        ).select(cs.numeric())

        self.ko_gene_data = ko_gene_data.select(cs.numeric())

        del ko_gene_data, data
        gc.collect()

        # Assembling the pieces
        console.log("Creating Dataset")
        self.data = VCCDataset(
            self.expression_data, self.control_exp_data, self.ko_gene_data
        )

        console.log("Data splitting")
        train_index, test_index = train_test_split(
            np.linspace(1, length, length, dtype=int).tolist(),  # type: ignore
            test_size=self.test_size,
            random_state=self.seed,
        )

        console.log("Split data generation")
        self.train_data = Subset(self.data, train_index)
        self.test_data = Subset(self.data, test_index)

        console.log("Setup Done")

    def train_dataloader(self):
        """
        Creates and returns a DataLoader for the training dataset.

        Returns:
            DataLoader: DataLoader  for the train dataset
        """
        console.log("Creating Training Dataloader")
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        """
        Creates and returns a DataLoader for the validation dataset.

        Returns:
            DataLoader: DataLoader instance for the validation data.
        """
        console.log("Creating Validation Dataloader")
        return DataLoader(
            self.test_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
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


if __name__ == "__main__":
    dm = VCCDataModule(
        data_path="/storage/bt20d204/vcc-data/vcc_data/processed-data/training_data-counts_uint.parquet",
        ko_data_path="/storage/bt20d204/vcc-data/vcc_data/processed-data/training_data-gene_ko_uint.parquet",
        # num_workers=2,
    )
    dm.setup()

    for X, y in dm.val_dataloader():
        print(len(X), y.shape)
        break
