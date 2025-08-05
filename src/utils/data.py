from pathlib import Path

import numpy as np
import polars as pl
import polars.selectors as cs


def read_data(path: Path | str, format: str | None = None) -> pl.DataFrame:
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
            return pl.read_ipc(path)
        case "parquet":
            return pl.read_parquet(path)
        case "csv":
            return pl.read_csv(path)
        case _:
            raise NotImplementedError(
                "File format not supported. Most be one of 'feather', 'parquet', 'csv'"
            )


def build_embedding_dict(data: pl.DataFrame) -> dict[str, np.ndarray]:
    """
    The data should contain one gene_name column and the rest should be embeddings.

    Args:
        data (pl.DataFrame): The data containing the gene name and embeddings

    Returns:
        dict(str, numpy.ndarray)
    """

    gene_names: pl.Series = data["gene_name"]
    embeddings = data.select(cs.numeric()).to_numpy()
    result = dict(zip(gene_names, embeddings))

    for i, row in enumerate(data.select(cs.numeric()).iter_rows()):
        result[gene_names[i]] = np.array(row)

    return result


if __name__ == "__main__":
    data = read_data("../../../vcc_data/gene_embeddings/PCA-train_expression.parquet")
    result = build_embedding_dict(data)

    print(result)
