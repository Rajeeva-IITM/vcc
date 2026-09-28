"""Multi-dataset Poincaré kernel-ridge baseline: pool 2025 + the two PrimeFlow sources.

Same idea as `scripts/baseline_poincare_ridge.py` -- a closed-form embedding #-> delta map,
then `relu(x_0 + Delta_hat(gene))` on real 2026 controls -- but fit on the *union* of the
three training sources' perturbations (300 + 9,210 + 1,958 = 11,468), for far denser coverage
of the embedding manifold. The bet: the 2025-only fit committed to Poincaré directions that
anti-transferred (val fid -1.44); more perturbations spanning the manifold may make the
kernel's local neighbourhoods informative rather than misleading.

Missing genes
-------------
The two PrimeFlow sources measure only ~7.5k / ~8k of the 18,080 model genes (6,691 shared by
all three). A source's mean shift `Delta_s(g) = mu_pert(g) - mu_ctrl(source)` is therefore
defined only on the genes it measures. The unmeasured genes are imputed as **no change** --
the perturbed profile there equals the control -- so `Delta_s = 0` on them. Because the
prediction rides on real 2026 control cells (`relu(x_0 + Delta_hat)`), a zero shift leaves that
gene sitting at the 2026 control level: i.e. unmeasured genes are imputed with the 2026
control. (The source caches already store `pert_means` and the control as zero on unobserved
columns, so `pert_means - control_mean` is exactly zero there -- the imputation is automatic.)
2025 covers all 18,080 genes, so every gene still has at least the 2025 perturbations behind
it; the PrimeFlow rows only *add* signal on the shared axis and abstain elsewhere.

Depth axis
----------
These caches are normalised over the 6,691 genes common to all three sources (not the full
panel), so the 2026 control cells are normalised over that same axis here -- the multi-source
analogue of `predict_2026.py` reading `norm_axis.csv`. Getting this wrong shifts every input.

Scale
-----
`alpha = (K + lambda I)^{-1} Y` at n=11,468, G=18,080 is a ~10^15-flop matmul and is never
formed: predictions for the 300 targets factor as `Y_hat = [K_test (K+lambda I)^{-1}] Y`, and
hyperparameters are chosen by LOO on a subsample. See `src.utils.kernel_ridge.fit_predict_large`.

Usage
-----
    pixi run python scripts/baseline_poincare_ridge_multi.py
    pixi run python scripts/baseline_poincare_ridge_multi.py --limit 4   # smoke
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
from src.utils.kernel_ridge import fit_predict_large  # noqa: E402
from src.utils.predict import finalize_counts, to_counts, to_model_space  # noqa: E402

console = Console()

N_CELLS_PER_PERT = 400
CONTEXTS = ("A", "B", "C")
MAX_COUNTS_PER_CELL = 1_000_000
MAX_STORED_ENTRIES = 4_750_000_000
EMBEDDING = "gene_embeddings/poincare_go_gaf_logmapped_256.parquet"

# The three sources, in the order dataset_multi lists them.
SOURCES = (
    "adata_2025_all__*.npz",
    "replogle22_k562_preprocessed__*.npz",
    "nadig24_jurkat_preprocessed__*.npz",
)


def load_multi_caches(data_dir: Path) -> list[dict]:
    """Every multi-source cache under `_cache/`, as `{name, pert_means, pert_names,
    control_mean, observed}` dicts. These are normalised over the 6,691-gene shared axis."""
    out = []
    for pat in SOURCES:
        hits = sorted(glob.glob(str(data_dir / "_cache" / pat)))
        if not hits:
            raise SystemExit(
                f"no cache matching _cache/{pat}. Run a `data=dataset_multi` setup once to "
                "build all three source caches (seconds after the first, hours the first time)."
            )
        path = Path(hits[0])
        blob = np.load(path, allow_pickle=False)
        control = csr_matrix(
            (blob["control_data"], blob["control_indices"], blob["control_indptr"]),
            shape=tuple(blob["control_shape"]),
        )
        out.append(
            {
                "name": path.name.split("__")[0],
                "pert_means": blob["pert_means"],
                "pert_names": blob["pert_names"].astype(str),
                "control_mean": np.asarray(control.mean(axis=0)).ravel(),
                "observed": blob["observed"],
            }
        )
        console.log(
            f"  {out[-1]['name']}: {len(out[-1]['pert_names']):,} perts, "
            f"{int(blob['observed'].sum()):,} genes observed"
        )
    return out


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--target-sum", type=float, default=1e4)
    p.add_argument("--kernel", choices=["rbf", "cosine", "linear"], default="rbf")
    p.add_argument(
        "--limit", type=int, default=None, help="first N targets (smoke only)"
    )
    return p.parse_args()


def main() -> None:
    """Fit the pooled multi-source kernel-ridge map and write the 2026 submission."""
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

    emb = pl.read_parquet(data_dir / EMBEDDING)
    vectors = emb.select(cs.numeric()).to_numpy().astype(np.float64)
    emb_row = {g: i for i, g in enumerate(emb["gene_name"].to_numpy())}
    missing = [t for t in targets if t not in emb_row]
    if missing:
        raise SystemExit(f"{len(missing)} targets have no embedding: {missing[:5]}")

    # ---- pool training deltas across sources ----
    console.log("[bold]loading source caches[/bold]")
    caches = load_multi_caches(data_dir)
    shared = np.ones(len(genes_model), dtype=bool)
    for c in caches:
        shared &= c["observed"]
    norm_cols = np.flatnonzero(shared)  # 6,691-gene depth axis these caches use
    console.log(
        f"shared depth axis: {len(norm_cols):,} genes (observed by all sources)"
    )

    pos_model = {g: i for i, g in enumerate(genes_model)}
    E_list, D_list, own_list, names_src = [], [], [], []
    for c in caches:
        keep = np.array([g in emb_row for g in c["pert_names"]])
        names = c["pert_names"][keep]
        # Delta = pert mean - control mean; already 0 on unobserved genes (see module doc).
        delta = (c["pert_means"][keep] - c["control_mean"]).astype(np.float64)
        E_list.append(vectors[[emb_row[g] for g in names]])
        D_list.append(delta)
        own_list.append(np.array([pos_model.get(g, -1) for g in names]))
        names_src.extend([(g, c["name"]) for g in names])
    E_train = np.concatenate(E_list)
    Delta_train = np.concatenate(D_list)
    own_col = np.concatenate(own_list)
    del E_list, D_list
    console.log(
        f"pooled {len(E_train):,} perturbations x {Delta_train.shape[1]:,} genes "
        f"({int((own_col >= 0).sum()):,} with an own-gene column)"
    )

    # ---- fit + predict the 300 targets (no full alpha; see fit_predict_large) ----
    E_test = vectors[[emb_row[t] for t in targets]]
    Delta_hat, rep = fit_predict_large(
        E_train,
        Delta_train,
        E_test,
        own_col,
        args.kernel,
        lambdas=[1.0, 3.0, 10.0, 30.0, 100.0, 300.0],
        gamma_scales=[2.0, 4.0, 8.0, 16.0],
        seed=args.seed,
        log=lambda m: console.log(f"  {m}"),
    )
    console.log(
        f"[bold]{rep['kernel']}[/bold] | lambda={rep['lambda']:g} "
        f"scale={rep['gamma_scale']:g} | subsample LOO specific-cos "
        f"[bold]{rep['loo_specific_cosine']:+.4f}[/bold]"
    )
    delta_of = {t: Delta_hat[i] for i, t in enumerate(targets)}

    # ---- gene-panel bookkeeping (identical to predict_2026) ----
    model_to_26, present, unmapped_26 = build_gene_maps(genes_2026, genes_model)
    src_cols = model_to_26[present]
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
            ctrl_full = X26[rows].toarray().astype(np.float32)

            ctrl_model = np.zeros(
                (N_CELLS_PER_PERT, len(genes_model)), dtype=np.float32
            )
            ctrl_model[:, present] = ctrl_full[:, src_cols]
            # Depth over the SHARED axis -- the space the caches (and hence Delta_hat) live in.
            exp_vec, lib = to_model_space(ctrl_model, norm_cols, args.target_sum)

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
        args.out
        or f"results/2026/baseline_poincare_ridge_multi_{date.today():%d%m%y}.h5ad"
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
