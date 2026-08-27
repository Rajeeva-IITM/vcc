import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import anndata as ad
    import marimo as mo
    import numpy as np
    import polars as pl
    from scipy.sparse import csr_matrix, vstack

    return ad, csr_matrix, mo, np, pl, vstack


@app.cell
def _():
    from pathlib import Path

    from dotenv import dotenv_values

    paths = dotenv_values(Path(__file__).parent.parent / ".env")
    data_path = Path(paths["DATA_DIR"])
    result_path = Path(paths["PROJECT"]) / "results"
    return data_path, result_path


@app.cell
def _(data_path, mo, pl):
    gene_names = pl.read_csv(data_path / "vcc_2026/gene_names.csv")
    pert_counts = pl.read_csv(data_path / "vcc_2026/pert_counts.csv")
    mo.hstack([gene_names, pert_counts])
    return gene_names, pert_counts


@app.cell
def _(ad, data_path):
    adata = ad.read_h5ad(data_path / "vcc_2026/context_A.h5ad")
    adata1 = ad.read_h5ad(data_path / "vcc_2026/context_B.h5ad")
    adata2 = ad.read_h5ad(data_path / "vcc_2026/context_C.h5ad")
    adata
    return adata, adata1, adata2


@app.cell
def _(adata):
    adata.obs
    return


@app.cell
def _(adata):
    adata.obs_names
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Building the `prediction.vcc` object

    Instructions from https://virtualcellchallenge.org/app/datasets

    - One Anndata H5AD file
      - Obs - two column dataframe with `target_gene` and `context` columns
      - Var - the same as gene_names.csv
      - X - raw_counts (400 cells x 300 target genes x 3 contexts = 360000 observations) for 18,533 genes
    """)
    return


@app.cell
def _():
    N_CELLS_PER_PERT = 400  # fixed by the submission spec
    N_SELECT = 1000  # control cells feeding each mean

    MAX_COUNTS_PER_CELL = 1_000_000
    MAX_STORED_ENTRIES = 4_750_000_000
    return MAX_COUNTS_PER_CELL, MAX_STORED_ENTRIES, N_CELLS_PER_PERT, N_SELECT


@app.cell
def _(N_CELLS_PER_PERT, pert_counts, pl):
    # One ordered list of (context, target_gene) pairs. Both `pred_obs` and the row blocks of
    # X are derived from it, so the labels cannot drift out of step with the counts.
    pert_pairs = [
        (context, target_gene)
        for context in ("A", "B", "C")
        for target_gene in pert_counts["target_gene"]
    ]

    pred_obs = (
        pl.DataFrame(pert_pairs, schema=["context", "target_gene"], orient="row")
        .with_columns(n=pl.lit(N_CELLS_PER_PERT))
        .with_columns(pl.col("context", "target_gene").repeat_by("n"))
        .explode("context", "target_gene")
        .select("target_gene", "context")
    )

    pred_obs
    return pert_pairs, pred_obs


@app.cell
def _(adata, adata1, adata2):
    # Keep the AnnData objects: `.to_df()` would materialise an 18,400 x 18,533 pandas frame
    # per context for work that is purely numeric.
    context_map = {"A": adata, "B": adata1, "C": adata2}
    return (context_map,)


@app.cell
def _(N_SELECT, context_map, np, pert_counts):
    def _profile_for_gene(exp_norm, column):
        """Mean profile of the N_SELECT lowest-expressing control cells for one target gene.

        Because every cell was scaled to the context median depth, each one totals that
        depth, so this mean is already a raw-count vector at that depth.
        """
        selected = np.argsort(column, kind="stable")[:N_SELECT]

        return np.median(exp_norm[selected], axis=0)

    profiles = {}

    for _context, _adata in context_map.items():
        _counts = _adata.X
        _depth = np.asarray(_counts.sum(axis=1)).ravel().astype(np.float32)
        _target = np.float32(np.median(_depth))

        # Scale every cell to this context's median depth.
        _exp_norm = (
            _counts.toarray() / _depth[:, None] * 5 * 1e4
        )  # Used for PDS score so going with it
        _gene_index = {_gene: _i for _i, _gene in enumerate(_adata.var_names)}

        for _gene in pert_counts["target_gene"]:
            profiles[(_context, _gene)] = _profile_for_gene(
                _exp_norm, _exp_norm[:, _gene_index[_gene]]
            )

        del _exp_norm

    print(f"{len(profiles)} profiles over {len(context_map)} contexts")
    return (profiles,)


@app.cell
def _(N_CELLS_PER_PERT, csr_matrix, np, pert_pairs, profiles, vstack):
    _blocks = []

    for _context, _gene in pert_pairs:
        # The profile is already on a raw-count scale, so rounding is the only step left.
        # All 400 cells of a perturbation are identical.
        _row = np.rint(np.clip(profiles[(_context, _gene)], 0, None)).astype(np.int32)

        _blocks.append(csr_matrix(np.tile(_row, (N_CELLS_PER_PERT, 1))))

    X = vstack(_blocks, format="csr")
    X.eliminate_zeros()  # explicitly-stored zeros count against the spec cap

    del _blocks

    X
    return (X,)


@app.cell
def _(
    MAX_COUNTS_PER_CELL,
    MAX_STORED_ENTRIES,
    N_CELLS_PER_PERT,
    X,
    adata,
    gene_names,
    np,
    pert_counts,
    pred_obs,
):
    # Every assertion here is a requirement from the submission spec.
    _totals = np.asarray(X.sum(axis=1)).ravel()
    _groups = pred_obs.group_by("context", "target_gene").len()

    assert X.shape == (360_000, 18_533), X.shape
    assert np.all(X.data == np.rint(X.data)), "counts must be whole numbers"
    assert X.data.min() >= 0, "counts must be non-negative"
    assert np.isfinite(X.data).all(), "counts must be finite"
    assert _totals.max() <= MAX_COUNTS_PER_CELL, _totals.max()
    assert X.nnz <= MAX_STORED_ENTRIES, X.nnz
    assert (X.data == 0).sum() == 0, "explicitly-stored zeros count against the cap"

    assert list(adata.var_names) == gene_names["gene_name"].to_list(), (
        "var order must match gene_names.csv"
    )
    assert pred_obs.columns == ["target_gene", "context"], pred_obs.columns
    assert pred_obs.height == X.shape[0], (pred_obs.height, X.shape[0])
    assert _groups["len"].unique().to_list() == [N_CELLS_PER_PERT], (
        "every (context, target_gene) needs exactly 400 cells"
    )
    assert _groups.height == 3 * pert_counts.height, _groups.height

    print(
        f"OK  {X.shape[0]:,} cells x {X.shape[1]:,} genes | "
        f"nnz {X.nnz:,} ({100 * X.nnz / MAX_STORED_ENTRIES:.1f}% of cap) | "
        f"cell totals {_totals.min():,.0f} - {_totals.max():,.0f}"
    )
    return


@app.cell
def _(X, ad, gene_names, pred_obs, result_path):
    from datetime import date

    result = ad.AnnData(
        X=X,
        obs=pred_obs.to_pandas(),
        var=gene_names.to_pandas().set_index("gene_name"),
    )

    _out = result_path / "2026" / f"prediction_2026_{date.today():%d%m%y}.h5ad"
    result.write_h5ad(
        _out, compression="gzip"
    )  # ~16 GB uncompressed, so this takes a while

    print(f"wrote {_out}")
    result
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
