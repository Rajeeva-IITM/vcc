import gc
import warnings
from pathlib import Path
from typing import Dict, Tuple, Union

import numpy as np
import polars as pl
import polars.selectors as cs
import rich
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader, Dataset, Subset

# from sklearn.model_selection import train_test_split
from src.utils.gene_train_test_split import gene_train_test_split, get_gene_names

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
        stage: str | None,
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
        stage: str | None
            Indicating whether to initialize the data in predict mode or not
        """
        super().__init__()

        self.stage = stage
        match self.stage:
            case None:
                raise ValueError("`stage` cannot be None ")
            case "predict":
                print("Dataset initialized in prediction mode")
                assert control_expression.shape[0] == ko_gene_data.shape[0], (
                    f"Data not of the same length\
                {control_expression.shape[0], ko_gene_data.shape[0]}"
                )
            case _:
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

        # Converting to numpy because polars misbehaves with multiprocessing
        self.ko_expression = ko_expression.to_numpy()
        self.control_expression = control_expression.to_numpy()
        self.ko_gene_data = ko_gene_data.to_numpy()
        # self.dtype = dtype

        # print(f"Data lengths\
        #         {self.ko_expression.shape, self.control_expression.shape, self.ko_gene_data.shape}")

        del ko_expression, control_expression, ko_gene_data
        gc.collect()

    def __len__(self):
        """
        Returns the number of samples in the dataset.
        Returns:
            int: The number of rows in the ko_expression attribute.
        """

        return len(self.ko_gene_data)

    def __getitem__(
        self, index: int
    ) -> Tuple[Dict[str, np.ndarray], np.ndarray | list]:
        """
        Retrieves a single data sample from the dataset at the specified index.
        Args:
            index (int): Index of the sample to retrieve.
        Returns:
            Tuple[Dict[str, np.ndarray], np.ndarray]:
                A tuple containing a dictionary with knockout gene vector ('ko_vec') and control expression vector ('exp_vec'),
                and the knockout expression tensor as the target.
        """

        ko_input = self.ko_gene_data[index, :].astype(np.float16)
        exp_input = self.control_expression[index, :].astype(np.float16)
        if self.stage == "predict":
            pred_input = []
        else:
            pred_input = self.ko_expression[index, :].astype(np.float16)

        return {"ko_vec": ko_input, "exp_vec": exp_input}, pred_input


class VCCDataModule(LightningDataModule):
    """Data module for VCC data. Two main inputs must be provided:
        - The gene knockout data (does not include non-targeting samples)
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
        Prepares and splits the dataset for training and testing.

        Reads expression and knockout gene data, processes and samples control data,
        creates the dataset, and splits it into training and test subsets.
        """

        match stage:
            case "predict":
                console.log("Setting up data for prediction")

                if (self.test_exp_data_path is None) | (self.test_ko_data_path is None):
                    raise TypeError(
                        "`test_exp_data_path` and `test_ko_data_path` must be present"
                    )

                console.log("Reading data")
                test_exp_data = read_data(self.test_exp_data_path).select(cs.numeric())  # type: ignore
                test_ko_data = read_data(self.test_ko_data_path).select(cs.numeric())  # type: ignore

                control_data = read_data(self.control_data_path)
                length = test_ko_data.shape[0]  # Number of non control indices

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

                self.test_data = VCCDataset(
                    test_exp_data, control_data, test_ko_data, stage
                )

                del control_data, test_ko_data, test_exp_data
                gc.collect()

            case _:
                console.log("Setting up data")
                data = read_data(self.train_exp_data_path).select(cs.numeric())
                ko_gene_data = read_data(self.train_ko_data_path)

                assert data.shape[0] == ko_gene_data.shape[0], (
                    "row count mismatch between data and ko_exp_data"
                )

                # Separate knockout data and control data
                console.log("Expression data processing")

                control_data = read_data(self.control_data_path)
                length = ko_gene_data.shape[0]  # Number of non control indices

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

                ko_gene_data = ko_gene_data.select(cs.numeric())

                # Assembling the pieces
                console.log("Creating Dataset")
                self.data = VCCDataset(data, control_data, ko_gene_data, stage)

                console.log("Data splitting")

                gene_names = get_gene_names(ko_gene_data)

                train_index, val_index = gene_train_test_split(
                    gene_names,  # type: ignore
                    test_size=self.test_size,
                    seed=self.seed,
                )

                console.log("Split data generation")
                self.train_data = Subset(self.data, train_index)
                self.val_data = Subset(self.data, val_index)

                del ko_gene_data, data  # Clean up
                gc.collect()

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
            self.val_data,
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

    def predict_dataloader(self):
        """Creates and returns the prediction dataloader"""
        console.log("Creating prediction dataloader")
        return DataLoader(
            self.test_data, batch_size=1, num_workers=self.num_workers, shuffle=False
        )


if __name__ == "__main__":
    dm = VCCDataModule(
        train_exp_data_path="/storage/bt20d204/vcc-data/processed-data/training_data-counts_uint.parquet",
        train_ko_data_path="/storage/bt20d204/vcc-data/processed-data/training_data-gene_ko_uint.parquet",
        control_data_path="/storage/bt20d204/vcc-data/processed-data/control_exp_data_uint.parquet",
        # num_workers=2,
    )
    dm.setup(stage="fit")

    for X, y in dm.val_dataloader():
        print(len(X), y.shape)
        break
