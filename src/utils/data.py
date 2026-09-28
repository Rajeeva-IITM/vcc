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


def build_gene_maps(
    genes_src: list[str], genes_dst: list[str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map a destination gene panel onto a source panel, **by name**.

    Two panels of the same genes are not interchangeable: the 2025 and 2026 panels
    share 18,077 genes in a different relative order, so anything positional is
    silently wrong. Everywhere this repo crosses panels -- reading a foreign
    ``.h5ad`` into the model's gene order, or scattering a prediction back to the
    2026 order -- goes through this function.

    Args:
        genes_src (list[str]): The panel being read *from*.
        genes_dst (list[str]): The panel being read *into*; the output order.

    Returns:
        tuple: ``(dst_to_src, present, unmapped_src)``. ``dst_to_src[i]`` is the
        source column for destination gene *i* (0 where absent, so it must always
        be masked by ``present``), and ``unmapped_src`` lists the source columns no
        destination gene covers.
    """
    pos = {gene: i for i, gene in enumerate(genes_src)}
    dst_to_src = np.zeros(len(genes_dst), dtype=np.int64)
    present = np.zeros(len(genes_dst), dtype=bool)
    for i, gene in enumerate(genes_dst):
        j = pos.get(gene)
        if j is not None:
            dst_to_src[i], present[i] = j, True

    covered = np.zeros(len(genes_src), dtype=bool)
    covered[dst_to_src[present]] = True

    return dst_to_src, present, np.flatnonzero(~covered)


if __name__ == "__main__":
    data = read_data("../../../vcc_data/gene_embeddings/PCA-train_expression.parquet")
    result = build_embedding_dict(data)

    print(result)
