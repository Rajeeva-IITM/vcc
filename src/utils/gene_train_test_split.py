# Implementing a better train_test_split to simulate out of distribution (OOD) data

import numpy as np
import polars as pl
import polars.selectors as cs

# from rich.progress import track


def get_gene_names(ko_data: pl.DataFrame):
    """
    Extracts gene names from a Knockout Dataframe
    Args:
        ko_data (pl.DataFrame): A Polars DataFrame containing gene data with numeric columns.
    Returns:
        list: A list of  gene names
    """

    # Vectorized extraction of gene names for each row
    numeric_df = ko_data.select(cs.numeric())
    arr = numeric_df.to_numpy()
    columns = np.array(numeric_df.columns)
    # Find the index of the '1' in each row (assumes only one '1' per row)
    gene_indices = np.argmax(arr == 1, axis=1)
    gene_names = columns[gene_indices]
    return gene_names


def gene_train_test_split(
    gene_names: list[str], test_size: float, seed: int | np.random.RandomState | None
):
    assert 0 < test_size < 1, "test size must be between 0 and 1"

    rng = np.random.default_rng(seed=seed)

    # Need a balancing act between splitting and having sufficient test samples
    unique_genes = np.unique(gene_names)

    # data_test_size = np.round(len(gene_names)*test_size) # Approximate number of samples present in test split #TODO: Balancing act complicated so moving this to later
    gene_test_size = np.ceil(len(unique_genes) * test_size).astype(
        int
    )  # Number of genes not present in the train split

    test_genes = rng.choice(unique_genes, gene_test_size)

    test_indices: list[int] = np.where(np.isin(gene_names, test_genes))[0].tolist()
    train_indices: list[int] = np.where(~np.isin(gene_names, test_genes))[0].tolist()

    return train_indices, test_indices


if __name__ == "__main__":
    gene_names = [
        "geneA",
        "geneB",
        "geneC",
        "geneA",
        "geneD",
        "geneE",
        "geneB",
        "geneF",
    ]
    test_size = 0.3
    seed = 42

    train_indices, test_indices = gene_train_test_split(gene_names, test_size, seed)

    print("Genes:", gene_names)
    print("Train indices:", train_indices)
    print("Test indices:", test_indices)
    print("Train genes:", [gene_names[i] for i in train_indices])
    print("Test genes:", [gene_names[i] for i in test_indices])
