"""Contract tests for the AnnData-native datamodule.

``AnnDataVCCDataModule`` is meant to be a drop-in for the parquet-backed
datamodules: swapping ``data=dataset_anndata`` in must not change the shape or
meaning of a batch. These tests pin the parts of that contract the models and
``src/train.py`` actually depend on, using small synthetic ``.h5ad`` files so
they run without the 15 GB real dataset.
"""

import anndata as ad
import numpy as np
import pandas as pd
import polars as pl
import pytest
import torch
from scipy.sparse import csr_matrix

from src.data.vcc_anndata_module import AnnDataVCCDataModule

N_GENES, EMBED = 40, 8
GENES = [f"G{i:03d}" for i in range(N_GENES)]
PERTURBED = ["G001", "G002", "G003", "G004", "G005"]
N_PER_PERT, N_CONTROL = 6, 30


def _write_h5ad(path, seed=0):
    """Write a small synthetic .h5ad with controls and perturbed cells."""
    rng = np.random.default_rng(seed)
    labels = ["non-targeting"] * N_CONTROL + [
        g for g in PERTURBED for _ in range(N_PER_PERT)
    ]
    # Poisson counts with a per-cell depth factor, so normalisation has
    # something to remove.
    depth = rng.uniform(0.5, 4.0, len(labels))[:, None]
    counts = rng.poisson(depth * rng.uniform(0.5, 5.0, N_GENES)).astype(np.float32)
    counts[0] = 0  # a zero-depth cell: the guard against dividing by zero
    adata = ad.AnnData(
        X=csr_matrix(counts),
        obs=pd.DataFrame(
            {"target_gene": labels}, index=[f"c{i}" for i in range(len(labels))]
        ),
        var=pd.DataFrame(index=GENES),
    )
    adata.write_h5ad(path)
    return path


@pytest.fixture
def paths(tmp_path):
    """Synthetic .h5ad plus a matching gene-embedding parquet."""
    h5ad = _write_h5ad(tmp_path / "data.h5ad")
    emb = tmp_path / "emb.parquet"
    rng = np.random.default_rng(1)
    pl.DataFrame(
        {"gene_name": GENES}
        | {f"latent_{i}": rng.standard_normal(N_GENES) for i in range(EMBED)}
    ).write_parquet(emb)
    return h5ad, emb


@pytest.fixture
def dm(paths):
    """A datamodule set up for fit on the synthetic data."""
    h5ad, emb = paths
    module = AnnDataVCCDataModule(
        data_path=h5ad,
        gene_embedding_path=emb,
        num_workers=0,
        batch_size=4,
        test_size=0.4,
    )
    module.setup("fit")
    return module


def test_batch_contract(dm):
    """The two keys every net's forward() reads, and the target."""
    X, y = dm.data[0]
    assert set(X) == {"ko_vec", "exp_vec"}
    assert X["ko_vec"].shape == (EMBED,)
    assert X["exp_vec"].shape == (N_GENES,)
    assert y.shape == (N_GENES,)


def test_controls_are_excluded_from_targets(dm):
    """Control cells are inputs only; they must never appear as a perturbation."""
    assert "non-targeting" not in dm.data.perturbed_genes
    assert len(dm.data) == len(PERTURBED) * N_PER_PERT


def test_normalisation_hits_target_sum(dm):
    """Every non-empty cell should sum to target_sum once log1p is undone."""
    for i in range(len(dm.data)):
        X, y = dm.data[i]
        assert np.isclose(torch.expm1(y).sum().item(), dm.target_sum, rtol=1e-3)
        assert np.isclose(
            torch.expm1(X["exp_vec"]).sum().item(), dm.target_sum, rtol=1e-3
        )


def test_zero_depth_cell_does_not_produce_nan(paths):
    """The synthetic data has an all-zero cell; it must not divide by zero."""
    h5ad, emb = paths
    dm = AnnDataVCCDataModule(data_path=h5ad, gene_embedding_path=emb, num_workers=0)
    dm.setup("fit")
    assert torch.isfinite(dm.data._row(0)).all()


def test_split_holds_out_whole_genes(dm):
    """The split is OOD by gene, not a random cell split."""
    genes = dm.data.perturbed_genes
    train, val = set(genes[dm.train_index]), set(genes[dm.val_index])
    assert train and val
    assert not train & val


def test_predict_stage_returns_empty_target(paths, tmp_path):
    """src/train.py relies on an empty target and on `perturbed_genes`."""
    h5ad, emb = paths
    counts = tmp_path / "pert_counts.csv"
    pl.DataFrame({"target_gene": PERTURBED[:2], "n_cells": [3, 5]}).write_csv(counts)

    dm = AnnDataVCCDataModule(
        data_path=h5ad, gene_embedding_path=emb, pert_counts_path=counts, num_workers=0
    )
    dm.setup("predict")

    assert len(dm.test_data) == 8
    assert list(dm.test_data.perturbed_genes[:3]) == [PERTURBED[0]] * 3
    X, y = dm.test_data[0]
    assert y == []
    assert X["exp_vec"].shape == (N_GENES,)


def test_gene_list_reorders_columns(paths, tmp_path):
    """`gene_list_path` permutes the panel; the same cell must be permuted too."""
    h5ad, emb = paths
    reversed_genes = GENES[::-1]
    gene_list = tmp_path / "genes.csv"
    pl.DataFrame({"g": reversed_genes}).write_csv(gene_list, include_header=False)

    base = AnnDataVCCDataModule(data_path=h5ad, gene_embedding_path=emb, num_workers=0)
    base.setup("fit")
    flipped = AnnDataVCCDataModule(
        data_path=h5ad, gene_embedding_path=emb, gene_list_path=gene_list, num_workers=0
    )
    flipped.setup("fit")

    assert flipped.gene_names == reversed_genes
    torch.testing.assert_close(base.data._row(0).flip(0), flipped.data._row(0))


def test_mismatched_panels_are_rejected(paths, tmp_path):
    """Concatenating files with different panels must fail loudly, not silently."""
    h5ad, emb = paths
    other = ad.read_h5ad(h5ad)
    other.var_names = [f"X{i:03d}" for i in range(N_GENES)]
    other.write_h5ad(tmp_path / "other.h5ad")

    dm = AnnDataVCCDataModule(
        data_path=[h5ad, tmp_path / "other.h5ad"],
        gene_embedding_path=emb,
        num_workers=0,
    )
    with pytest.raises(ValueError, match="different gene panel"):
        dm.setup("fit")


def test_missing_embedding_is_reported_by_name(paths):
    """Fail in setup, not with a KeyError inside a dataloader worker."""
    h5ad, emb = paths
    trimmed = pl.read_parquet(emb).filter(pl.col("gene_name") != PERTURBED[0])
    short = emb.parent / "short.parquet"
    trimmed.write_parquet(short)

    dm = AnnDataVCCDataModule(data_path=h5ad, gene_embedding_path=short, num_workers=0)
    with pytest.raises(KeyError, match=PERTURBED[0]):
        dm.setup("fit")


def test_control_pairing_is_reproducible(paths):
    """Fixed at setup, per the design: the same seed gives the same pairing."""
    h5ad, emb = paths
    rows = []
    for _ in range(2):
        dm = AnnDataVCCDataModule(
            data_path=h5ad, gene_embedding_path=emb, num_workers=0, seed=7
        )
        dm.setup("fit")
        rows.append(dm.data.control_rows.copy())
    np.testing.assert_array_equal(*rows)


def test_dataloader_yields_stacked_batches(dm):
    """The dataloader collates into the shapes the models expect."""
    X, y = next(iter(dm.train_dataloader()))
    assert X["ko_vec"].shape == (4, EMBED)
    assert X["exp_vec"].shape == (4, N_GENES)
    assert y.shape == (4, N_GENES)


def test_library_size_inverts_to_counts(paths):
    """Predictions come out in log-CP10K; the submission needs raw counts back.

    Depth is not recoverable from the normalised values, so the module keeps it.
    Round-tripping a control cell must return its exact integer counts.
    """
    h5ad, emb = paths
    dm = AnnDataVCCDataModule(data_path=h5ad, gene_embedding_path=emb, num_workers=0)
    dm.setup("fit")

    truth = np.asarray(ad.read_h5ad(h5ad).X.todense(), dtype=np.float32)
    for i in (0, 3, 11):
        row = dm.data.control_rows[i]
        X, _ = dm.data[i]
        counts = np.rint(
            torch.expm1(X["exp_vec"]).numpy()
            * dm.data.control_library_size[i]
            / dm.target_sum
        )
        assert dm.data.control_library_size[i] == truth[row].sum()
        np.testing.assert_array_equal(counts, truth[row])
