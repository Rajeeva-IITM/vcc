"""Model-free 2026 submission: kernel-ridge regression from Poincare embeddings to delta.

This is the closed-form counterpart of `model_direct_kernel_delta` -- the same pure-shift
idea (`y = relu(x_0 + Delta(gene))`, Delta a function of the perturbation embedding ONLY),
but with the embedding->delta map fit in one linear-algebra step instead of trained by SGD.
No GPU, no checkpoint, no Lightning.

The map
-------
Fit on the 2025 panel. For each of the 300 training perturbations we have the mean
`log1p(cp10k)` profile (`pert_means`) and the mean control profile; their difference is the
model-space mean *shift*

    Delta(g) = mu_pert(g) - mu_ctrl        in R^18080, log1p(cp10k) space

which under the challenge's random control/perturbed pairing is exactly the MSE-optimal
per-cell shift (it lands the perturbed mean while preserving control variance). We regress
Delta on the Poincare embedding e(g) with kernel ridge:

    alpha = (K + lambda I)^{-1} Delta_train           K_ij = k(e_i, e_j)
    Delta_hat(e) = k(e, E_train) @ alpha

An RBF kernel (default) makes this a smooth interpolator over the embedding manifold -- the
same "smooth embedding->response map" the transfer diagnosis calls for, and the arm that
locally clears the blind DE-cosine bar (ridge-rbf ~0.22 vs blind ~0.20). `lambda` and the RBF
bandwidth are picked by closed-form leave-one-out CV on the 300 training deltas, scored by
specific delta-cosine (own gene excluded, the metric the leaderboard is sensitive to).

The submission
--------------
Identical arithmetic to `scripts/predict_2026.py` -- same control draw, same
`src.utils.predict` helpers, same 456-gene control copy -- so the file this writes is
byte-for-byte a valid submission. The only substitution is the per-cell forward: where
`predict_2026` runs a trained net, this adds the fitted `Delta_hat(target)` to the control and
ReLUs. Everything is numpy/scipy on CPU.

Source of the training means: the datamodule's own source cache
(`2025/_cache/adata_2025_all__*.npz`), so there is no 43 GB read and no prep step -- the file
is discovered by content hash. It holds `pert_means`, `pert_names`, and the control matrix,
all in the model panel's gene order and the `target_sum=1e4` log1p(cp10k) space.

Usage
-----
    pixi run python scripts/baseline_poincare_ridge.py
    pixi run python scripts/baseline_poincare_ridge.py --kernel rbf --limit 4   # smoke
"""

from __future__ import annotations

import argparse
import glob
import os
from datetime import date
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import polars as pl
import polars.selectors as cs
import rootutils
from dotenv import load_dotenv
from rich.console import Console
from scipy.sparse import csr_matrix, vstack

rootutils.setup_root(__file__, indicator="pixi.toml", pythonpath=True)

from src.utils.data import build_gene_maps  # noqa: E402
from src.utils.kernel_ridge import fit_kernel_ridge, kernel  # noqa: E402
from src.utils.predict import finalize_counts, to_counts, to_model_space  # noqa: E402

console = Console()

N_CELLS_PER_PERT = 400  # fixed by the submission spec
CONTEXTS = ("A", "B", "C")
MAX_COUNTS_PER_CELL = 1_000_000
MAX_STORED_ENTRIES = 4_750_000_000
EMBEDDING = "gene_embeddings/poincare_go_gaf_logmapped_256.parquet"
CONTROL_LABEL = "non-targeting"


# --------------------------------------------------------------------------------------
# training data, straight from the datamodule cache
# --------------------------------------------------------------------------------------


def load_source_cache(data_dir: Path) -> dict:
    """The 2025 source cache: `pert_means`, `pert_names`, and the control matrix.

    Discovered by glob rather than a remembered path -- the datamodule content-addresses
    it, and there is exactly one 2025 cache. The values are already in the model panel's
    gene order and the `target_sum=1e4` log1p(cp10k) space every prediction uses.
    """
    hits = sorted(glob.glob(str(data_dir / "2025/_cache/adata_2025_all__*.npz")))
    if not hits:
        raise SystemExit(
            "no 2025 source cache under 2025/_cache/. Run any training/setup on "
            "data=dataset_anndata once to build it (it is the datamodule's own cache)."
        )
    if len(hits) > 1:
        console.log(f"[yellow]{len(hits)} caches; using newest[/yellow]")
    path = Path(max(hits, key=lambda p: os.stat(p).st_mtime))
    console.log(f"training means from {path.name}")
    blob = np.load(path, allow_pickle=False)
    control = csr_matrix(
        (blob["control_data"], blob["control_indices"], blob["control_indptr"]),
        shape=tuple(blob["control_shape"]),
    )
    return {
        "pert_means": blob["pert_means"],
        "pert_names": blob["pert_names"].astype(str),
        "control_mean": np.asarray(control.mean(axis=0)).ravel(),
    }


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--target-sum", type=float, default=1e4)
    p.add_argument("--kernel", choices=["rbf", "cosine", "linear"], default="rbf")
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="only the first N targets. Smoke-test only -- NOT a valid submission.",
    )
    return p.parse_args()


def main() -> None:
    """Fit the 2025 kernel-ridge map and write the 2026 submission."""
    args = parse_args()
    load_dotenv()
    data_dir = Path(os.environ["DATA_DIR"])
    rng = np.random.default_rng(args.seed)

    genes_model = pl.read_csv(data_dir / "2025/gene_names.csv", has_header=False)[
        "column_1"
    ].to_list()
    genes_2026 = pl.read_csv(data_dir / "vcc_2026/gene_names.csv")[
        "gene_name"
    ].to_list()
    targets = pl.read_csv(data_dir / "vcc_2026/pert_counts.csv")[
        "target_gene"
    ].to_list()
    if args.limit:
        targets = targets[: args.limit]
        console.log(f"[yellow]--limit {args.limit}: NOT a valid submission[/yellow]")
    console.log(
        f"{len(targets)} targets x {len(CONTEXTS)} contexts x {N_CELLS_PER_PERT} cells"
    )

    # ---- embeddings ----
    emb = pl.read_parquet(data_dir / EMBEDDING)
    vectors = emb.select(cs.numeric()).to_numpy().astype(np.float64)
    emb_row = {g: i for i, g in enumerate(emb["gene_name"].to_numpy())}
    missing = [t for t in targets if t not in emb_row]
    if missing:
        raise SystemExit(f"{len(missing)} targets have no embedding: {missing[:5]}")

    # ---- training deltas (model space) ----
    cache = load_source_cache(data_dir)
    pert_names = cache["pert_names"]
    train_has_emb = np.array([g in emb_row for g in pert_names])
    if not train_has_emb.all():
        console.log(
            f"  {int((~train_has_emb).sum())} training perts lack an embedding, dropped"
        )
    pert_names = pert_names[train_has_emb]
    Delta_train = (cache["pert_means"][train_has_emb] - cache["control_mean"]).astype(
        np.float64
    )
    E_train = vectors[[emb_row[g] for g in pert_names]]
    pos_model = {g: i for i, g in enumerate(genes_model)}
    own_col = np.array([pos_model.get(g, -1) for g in pert_names])
    console.log(
        f"fit on {len(pert_names)} perturbations x {Delta_train.shape[1]:,} genes "
        f"({int((own_col >= 0).sum())} with an own-gene column)"
    )

    # ---- fit ----
    alpha, gamma, rep = fit_kernel_ridge(
        E_train,
        Delta_train,
        own_col,
        args.kernel,
        lambdas=[1e-2, 1e-1, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0],
        gamma_scales=[0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0],
    )
    console.log(
        f"[bold]{rep['kernel']}[/bold] kernel | lambda={rep['lambda']:g} "
        f"gamma_scale={rep['gamma_scale']:g} | LOO specific-cos "
        f"[bold]{rep['loo_specific_cosine']:+.4f}[/bold]"
    )

    # ---- predict 2026 deltas ----
    E_test = vectors[[emb_row[t] for t in targets]]
    K_test = kernel(E_test, E_train, args.kernel, gamma)
    Delta_hat = (K_test @ alpha).astype(np.float32)  # (n_targets, 18080)
    delta_of = {t: Delta_hat[i] for i, t in enumerate(targets)}

    # ---- gene-panel bookkeeping (identical to predict_2026) ----
    model_to_26, present, unmapped_26 = build_gene_maps(genes_2026, genes_model)
    src_cols = model_to_26[present]
    norm_cols = np.arange(
        len(genes_model)
    )  # single-source panel: full-panel depth axis
    console.log(f"  {len(unmapped_26):,} 2026 genes copied from the control cell")

    # ---- build the submission ----
    pairs: list[tuple[str, str]] = []
    blocks: list[csr_matrix] = []
    for context in CONTEXTS:
        path = data_dir / f"vcc_2026/context_{context}.h5ad"
        console.log(f"[bold]context {context}[/bold]: {path.name}")
        adata = ad.read_h5ad(path)
        X26 = csr_matrix(adata.X)
        n_ctrl = X26.shape[0]

        for t_i, target in enumerate(targets):
            rows = rng.choice(n_ctrl, N_CELLS_PER_PERT, replace=False)
            ctrl_full = (
                X26[rows].toarray().astype(np.float32)
            )  # 400 x 18,533 raw counts

            ctrl_model = np.zeros(
                (N_CELLS_PER_PERT, len(genes_model)), dtype=np.float32
            )
            ctrl_model[:, present] = ctrl_full[:, src_cols]
            exp_vec, lib = to_model_space(ctrl_model, norm_cols, args.target_sum)

            # The whole model: control (log1p cp10k) + fitted shift, ReLU'd.
            pred = np.maximum(exp_vec + delta_of[target][None, :], 0.0)
            counts_model = to_counts(pred, lib)

            out_row = np.zeros((N_CELLS_PER_PERT, len(genes_2026)), dtype=np.float64)
            out_row[:, src_cols] = counts_model[:, present]
            out_row[:, unmapped_26] = ctrl_full[:, unmapped_26]

            blocks.append(finalize_counts(out_row))
            pairs.extend([(target, context)] * N_CELLS_PER_PERT)
            if (t_i + 1) % 50 == 0:
                console.log(f"  {t_i + 1}/{len(targets)} targets")
        del adata, X26

    X = vstack(blocks, format="csr")
    X.eliminate_zeros()
    del blocks

    obs = pd.DataFrame(
        {"target_gene": [p[0] for p in pairs], "context": [p[1] for p in pairs]},
        index=[f"cell_{i}" for i in range(X.shape[0])],
    )

    totals = np.asarray(X.sum(axis=1)).ravel()
    assert args.limit or X.shape == (
        len(CONTEXTS) * len(targets) * N_CELLS_PER_PERT,
        len(genes_2026),
    ), X.shape
    assert np.all(X.data == np.rint(X.data)), "counts must be whole numbers"
    assert X.data.min() >= 0 and np.isfinite(X.data).all()
    assert totals.max() <= MAX_COUNTS_PER_CELL, totals.max()
    assert X.nnz <= MAX_STORED_ENTRIES, X.nnz
    assert (X.data == 0).sum() == 0, "stored zeros count against the cap"
    assert obs.shape[0] == X.shape[0]
    assert obs.groupby(["context", "target_gene"]).size().unique().tolist() == [
        N_CELLS_PER_PERT
    ]

    out_path = Path(
        args.out or f"results/2026/baseline_poincare_ridge_{date.today():%d%m%y}.h5ad"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ad.AnnData(X=X, obs=obs, var=pd.DataFrame(index=genes_2026)).write_h5ad(
        out_path, compression="gzip"
    )
    console.log(
        f"[green]wrote[/green] {out_path}  {X.shape[0]:,} x {X.shape[1]:,} | "
        f"nnz {X.nnz:,} ({100 * X.nnz / MAX_STORED_ENTRIES:.1f}% of cap) | "
        f"cell totals {totals.min():,} - {totals.max():,} (median {np.median(totals):,.0f})"
    )


if __name__ == "__main__":
    main()
