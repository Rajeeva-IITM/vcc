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
    """The keys every net's forward() reads, and the target."""
    X, y = dm.data[0]
    assert set(X) == {"ko_vec", "ko_id", "exp_vec"}
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
    # Collated to a (B,) int64 tensor, which is what nn.Embedding requires.
    assert X["ko_id"].shape == (4,) and X["ko_id"].dtype == torch.int64
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


def test_shared_cells_across_files_are_rejected(paths, tmp_path):
    """The 2025 splits share their 38,176 control cells byte for byte.

    Stacking them would count every control three times and silently reweight
    every control-paired sample, so overlapping barcodes must fail loudly.
    """
    h5ad, emb = paths
    same = ad.read_h5ad(h5ad)
    copy = tmp_path / "same_cells.h5ad"
    same.write_h5ad(copy)

    dm = AnnDataVCCDataModule(
        data_path=[h5ad, copy], gene_embedding_path=emb, num_workers=0
    )
    with pytest.raises(ValueError, match="more than one of the files"):
        dm.setup("fit")


def test_distinct_files_still_stack(paths, tmp_path):
    """Files with disjoint cells concatenate normally."""
    h5ad, emb = paths
    other = ad.read_h5ad(h5ad)
    other.obs_names = [f"other_{i}" for i in range(other.n_obs)]
    second = tmp_path / "other_cells.h5ad"
    other.write_h5ad(second)

    dm = AnnDataVCCDataModule(
        data_path=[h5ad, second], gene_embedding_path=emb, num_workers=0
    )
    dm.setup("fit")
    assert len(dm.data) == 2 * len(PERTURBED) * N_PER_PERT


# --------------------------------------------------------------- target_mode


@pytest.fixture
def dm_mean(paths):
    """The same datamodule, but targeting perturbation means."""
    h5ad, emb = paths
    module = AnnDataVCCDataModule(
        data_path=h5ad,
        gene_embedding_path=emb,
        num_workers=0,
        batch_size=4,
        test_size=0.4,
        target_mode="perturbation_mean",
    )
    module.setup("fit")
    return module


def _dense(dataset):
    """Rebuild the normalised matrix the dataset holds as raw CSR buffers."""
    return csr_matrix(
        (dataset.data, dataset.indices, dataset.indptr),
        shape=(dataset.indptr.size - 1, dataset.n_genes),
    ).toarray()


def test_perturbation_mean_matches_a_direct_group_mean(dm_mean, paths):
    """The streaming group-by must agree with averaging the rows by hand.

    ``perturbation_mean`` goes through the source cache, which keeps the means and
    discards the perturbed cells that produced them -- so the only honest check is to
    re-read the original file and normalise it independently.
    """
    h5ad, _ = paths
    adata = ad.read_h5ad(h5ad)
    counts = np.asarray(adata.X.todense(), dtype=np.float64)
    depth = counts.sum(axis=1, keepdims=True)
    depth[depth == 0] = 1.0
    normalised = np.log1p(counts / depth * 1e4)
    labels = adata.obs["target_gene"].to_numpy().astype(str)

    ds = dm_mean.data
    for gene in np.unique(ds.perturbed_genes):
        expected = normalised[labels == gene].mean(axis=0)
        code = ds.pert_codes[np.flatnonzero(ds.perturbed_genes == gene)[0]]
        assert np.allclose(ds.pert_means[code].numpy(), expected, atol=1e-5), gene


def test_targets_are_identical_within_a_perturbation(dm_mean):
    """Every cell of a perturbation now carries the same target."""
    ds = dm_mean.data
    member = np.flatnonzero(ds.perturbed_genes == ds.perturbed_genes[0])
    assert len(member) > 1, "fixture must have several cells per perturbation"
    first = ds[member[0]][1]
    for index in member[1:]:
        assert torch.equal(ds[index][1], first)


def test_target_mode_leaves_the_inputs_alone(dm, dm_mean):
    """Only the target changes; the input contract must be identical.

    The two modes take different loading paths now, so the *pairing* of a sample to a
    control cell is no longer guaranteed to match. What must still hold is that a mean
    -mode batch is indistinguishable from a cell-mode one as far as any net can tell.
    """
    inputs_cell, target_cell = dm.data[0]
    inputs_mean, target_mean = dm_mean.data[0]
    assert set(inputs_cell) == set(inputs_mean)
    for key in ("ko_vec", "exp_vec"):
        assert inputs_cell[key].shape == inputs_mean[key].shape
        assert inputs_cell[key].dtype == inputs_mean[key].dtype
    assert torch.equal(inputs_cell["ko_vec"], inputs_mean["ko_vec"])
    assert not torch.equal(target_cell, target_mean)


def test_cell_mode_is_the_default_and_unchanged(dm):
    """The existing path must keep returning the individual perturbed cell."""
    ds = dm.data
    assert ds.pert_means is None and ds.pert_codes is None
    X = _dense(ds)
    for index in (0, 5, len(ds) - 1):
        assert torch.allclose(
            ds[index][1], torch.from_numpy(X[ds.pert_rows[index]]), atol=1e-6
        )


def test_unknown_target_mode_is_rejected(paths):
    """A typo in the config should fail loudly at construction."""
    h5ad, emb = paths
    with pytest.raises(ValueError, match="target_mode"):
        AnnDataVCCDataModule(
            data_path=h5ad, gene_embedding_path=emb, target_mode="mean"
        )


def test_ko_id_addresses_the_same_gene_as_ko_vec(dm, paths):
    """``ko_id`` must index the embedding parquet's row for that sample's gene.

    This is the assertion the learnable-embedding design rests on:
    ``LearnableGeneEmbedding`` initialises its table straight from this parquet, so an id
    that pointed anywhere else would silently train the wrong row -- with no shape error
    and no visible symptom beyond a model that does not learn.
    """
    _, emb_path = paths
    table = torch.from_numpy(
        pl.read_parquet(emb_path).select(pl.exclude("gene_name")).to_numpy()
    )
    names = pl.read_parquet(emb_path)["gene_name"].to_list()

    for index in (0, 7, len(dm.data) - 1):
        X, _ = dm.data[index]
        gene = dm.data.perturbed_genes[index]
        assert names[X["ko_id"]] == gene
        torch.testing.assert_close(table[X["ko_id"]], X["ko_vec"], check_dtype=False)


def test_ko_id_is_present_in_the_predict_stage(dm, paths):
    """Prediction has no target, but it still has to say which gene to predict."""
    h5ad, emb = paths
    module = AnnDataVCCDataModule(
        data_path=h5ad, gene_embedding_path=emb, num_workers=0, batch_size=4
    )
    module.setup("predict")

    X, y = module.test_data[0]
    assert set(X) == {"ko_vec", "ko_id", "exp_vec"}
    assert y == []


# ------------------------------------------------------------- multiple sources
#
# The second synthetic source differs from the first in every way the real ones do:
# a smaller gene panel with genes the model does not know, a different obs column, a
# different control label, and values that are already log1p-normalised.

AUX_SHARED = GENES[:25]
AUX_GENES = AUX_SHARED + [f"X{i:03d}" for i in range(5)]
AUX_PERTURBED = ["G002", "G003", "G010", "G011"]
AUX_TARGET_SUM = 9715.0


def _write_aux_h5ad(path, seed=7):
    """A second source: partial panel, other column names, already log1p(cp9715)."""
    rng = np.random.default_rng(seed)
    labels = ["control"] * 20 + [g for g in AUX_PERTURBED for _ in range(5)]
    depth = rng.uniform(0.5, 4.0, len(labels))[:, None]
    counts = rng.poisson(depth * rng.uniform(0.5, 5.0, len(AUX_GENES))).astype(
        np.float64
    )
    counts[counts.sum(axis=1) == 0, 0] = 1.0
    normalised = np.log1p(
        counts / counts.sum(axis=1, keepdims=True) * AUX_TARGET_SUM
    ).astype(np.float32)

    adata = ad.AnnData(
        X=csr_matrix(normalised),
        obs=pd.DataFrame(
            {"perturbation": labels}, index=[f"a{i}" for i in range(len(labels))]
        ),
        var=pd.DataFrame(index=AUX_GENES),
    )
    adata.uns["log1p"] = {"base": None}  # how both PrimeFlow files flag this
    adata.write_h5ad(path)
    return path


@pytest.fixture
def multi(paths, tmp_path):
    """A datamodule over two sources with different gene panels."""
    h5ad, emb = paths
    aux = _write_aux_h5ad(tmp_path / "aux.h5ad")
    gene_list = tmp_path / "panel.csv"
    pl.DataFrame({"gene": GENES}).write_csv(gene_list, include_header=False)

    module = AnnDataVCCDataModule(
        data_path=[h5ad, aux],
        gene_embedding_path=emb,
        gene_list_path=gene_list,
        cache_dir=tmp_path / "_cache",
        num_workers=0,
        batch_size=4,
        test_size=0.4,
        target_mode="perturbation_mean",
    )
    module.setup("fit")
    return module


def test_sources_are_autodetected(multi):
    """Column names, control labels and value space are read off each file."""
    first, aux = (spec.resolve() for spec in multi.specs)
    assert (first["pert_key"], first["control_label"], first["delog"]) == (
        "target_gene",
        "non-targeting",
        False,
    )
    assert (aux["pert_key"], aux["control_label"], aux["delog"]) == (
        "perturbation",
        "control",
        True,
    )


def test_norm_axis_is_the_intersection_in_model_order(multi):
    """Depth is measured on genes every source has, ordered by the model panel."""
    assert multi.norm_genes == AUX_SHARED
    assert multi.gene_names == GENES


def test_norm_axis_does_not_depend_on_source_order(paths, tmp_path):
    """Listing the files the other way round must not invalidate the caches."""
    from src.data.source_cache import SourceSpec, resolve_norm_axis

    specs = [
        SourceSpec(paths[0]),
        SourceSpec(_write_aux_h5ad(tmp_path / "aux2.h5ad")),
    ]
    assert resolve_norm_axis(GENES, specs) == resolve_norm_axis(GENES, specs[::-1])


def test_adding_a_source_invalidates_the_existing_cache(paths, tmp_path):
    """A cache built against a wider axis is in different units, not merely stale."""
    from src.data.source_cache import SourceSpec, cache_key, resolve_norm_axis

    alone = SourceSpec(paths[0])
    aux = SourceSpec(_write_aux_h5ad(tmp_path / "aux3.h5ad"))
    _, solo = resolve_norm_axis(GENES, [alone])
    _, both = resolve_norm_axis(GENES, [alone, aux])
    assert solo != both
    assert cache_key(alone, GENES, solo, 1e4) != cache_key(alone, GENES, both, 1e4)


def test_unmeasured_genes_are_masked_not_believed(multi):
    """``gene_mask`` marks exactly the genes that source measured."""
    ds = multi.data
    assert ds.emit_mask, "a partial source must switch the mask on"
    for index in range(len(ds)):
        X, _ = ds[index]
        assert set(X) == {"ko_vec", "ko_id", "exp_vec", "gene_mask"}
        expected = ds.observed[ds.source_of_sample[index]]
        assert torch.equal(X["gene_mask"].bool(), expected)
        if not expected.all():
            assert int(expected.sum()) == len(AUX_SHARED)


def test_single_source_emits_no_mask(dm_mean):
    """One panel means an all-ones mask, which is not worth collating."""
    assert not dm_mean.data.emit_mask
    X, _ = dm_mean.data[0]
    assert "gene_mask" not in X


def test_unobserved_input_is_filled_from_the_reference(multi):
    """Genes a source lacks read as the reference control profile, not as zero.

    Zeros would tell ``exp_processor`` those genes are switched off, which is a
    different claim from "not measured" and one the loss then has to unlearn.
    """
    ds = multi.data
    aux = int(np.flatnonzero(np.array(multi.source_names) == "aux")[0])
    index = int(np.flatnonzero(ds.source_of_sample == aux)[0])
    X, _ = ds[index]
    missing = ~ds.observed[aux]
    assert missing.any()
    torch.testing.assert_close(X["exp_vec"][missing], ds.fill_profile[missing])
    assert (X["exp_vec"][missing] != 0).any()


def test_controls_come_from_the_samples_own_source(multi):
    """Pairing across sources would make the velocity a cell-line difference."""
    ds = multi.data
    bounds = np.cumsum([0] + [c["control_X"].shape[0] for c in _caches(multi)])
    assert bounds[-1] == ds.indptr.size - 1, "controls are one stacked block"
    for index in range(0, len(ds), 7):
        source = ds.source_of_sample[index]
        assert bounds[source] <= ds.control_rows[index] < bounds[source + 1]


def _caches(module):
    """Re-read the caches the datamodule built, for assertions about provenance."""
    from src.data.source_cache import cache_key, cache_path, load_cache

    _, digest = _resolve(module)
    return [
        load_cache(
            cache_path(
                spec,
                module.cache_dir,
                cache_key(spec, module.gene_names, digest, module.target_sum),
            )
        )
        for spec in module.specs
    ]


def _resolve(module):
    """The module's normalisation axis and its digest."""
    from src.data.source_cache import resolve_norm_axis

    return resolve_norm_axis(module.gene_names, module.specs)


def test_validation_stays_on_the_reference_source(multi):
    """Held-out genes leave every source, but only one source is scored."""
    ds = multi.data
    val_genes = set(ds.perturbed_genes[multi.val_index])
    assert val_genes, "the split must hold something out"
    assert set(ds.source_of_sample[multi.val_index]) == {0}
    # No training sample anywhere may use a held-out gene.
    assert not (set(ds.perturbed_genes[multi.train_index]) & val_genes)


def test_epoch_is_split_evenly_between_sources(multi):
    """Equal per source, not proportional to perturbation count."""
    weights = multi._sample_weights()
    sources = multi.data.source_of_sample[multi.train_index]
    share = np.bincount(sources, weights=weights, minlength=2)
    share = share / share.sum()
    assert np.allclose(share, 0.5, atol=1e-9)


def test_source_weights_override_the_even_split(paths, tmp_path):
    """The knob has to actually move the mix."""
    h5ad, emb = paths
    module = AnnDataVCCDataModule(
        data_path=[h5ad, _write_aux_h5ad(tmp_path / "aux4.h5ad")],
        gene_embedding_path=emb,
        gene_list_path=None,
        cache_dir=tmp_path / "_cache4",
        num_workers=0,
        target_mode="perturbation_mean",
        test_size=0.4,
        source_weights={"data": 0.75, "aux4": 0.25},
    )
    module.setup("fit")
    sources = module.data.source_of_sample[module.train_index]
    share = np.bincount(sources, weights=module._sample_weights(), minlength=2)
    assert np.allclose(share / share.sum(), [0.75, 0.25], atol=1e-9)


def test_norm_axis_is_written_beside_the_checkpoints(multi, tmp_path):
    """The checkpoint must record the space it was trained in.

    ``scripts/predict_2026.py`` reads this file to reproduce the input scaling. If it
    is missing the script silently falls back to the full panel, which is right for
    single-dataset runs and wrong -- by a constant factor on every gene -- for these.
    """

    class _Callback:
        dirpath = tmp_path / "run"

    class _Trainer:
        is_global_zero = True
        callbacks = [_Callback()]
        default_root_dir = tmp_path

    multi.trainer = _Trainer()
    multi._write_norm_axis()

    written = (tmp_path / "run" / "norm_axis.csv").read_text().split()
    assert written == multi.norm_genes == AUX_SHARED
