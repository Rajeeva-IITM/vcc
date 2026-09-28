"""Remove the perturbation-independent component from a 2026 submission.

Motivation
----------
The flow model's predicted log-FC is ~98% a single shift applied identically to all
300 targets (measured on ``prediction_2026_280826.h5ad``: common energy 278-410 vs
target-specific 6.3-6.9, i.e. ~7x the amplitude). That is the shape of failure that
scored ``nmae = -2.43`` -- a confidently wrong log-FC on every perturbation -- and it
carries no perturbation information by construction, since it is what remains after
averaging the perturbation away.

The likely source is domain shift: the model learned a large regression-to-the-mean
component pointing at the *2025* mean profile, and the 2026 controls have a different
mean, so the pull becomes a constant with no biological content.

What this does
--------------
Within each context, subtract the mean predicted log1p-CP10K profile and add back the
context's own control mean, then re-invert to counts at each cell's original library
size. Per-target log-FC becomes ``lfc[t] - mean_t(lfc)``: the perturbation-specific
signal is kept in full, the common shift is removed.

Two properties worth knowing:

* Because the mean is taken over all cells of a context and every target contributes
  the same number of cells, ``mean_t(pseudobulk_t)`` equals the mean over all cells --
  so the correction needs one streaming pass, not a per-target grouping.
* The scorer CP10K-normalises each cell, so a uniform factor per cell is invisible.
  Only the gene-wise *pattern* of the shift matters; the library-size renormalisation
  after re-inversion absorbs the scalar part.

Usage
-----
    python scripts/center_submission.py results/2026/prediction_2026_280826.h5ad
"""

import argparse
import os
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from rich.console import Console
from scipy.sparse import csr_matrix, vstack

console = Console()

CONTEXTS = ("A", "B", "C")
N_CELLS_PER_PERT = 400
CHUNK = 4_000
MAX_COUNTS_PER_CELL = 1_000_000
MAX_STORED_ENTRIES = 4_750_000_000


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("submission", help="the .h5ad written by predict_2026.py")
    p.add_argument("--out", default=None)
    p.add_argument("--target-sum", type=float, default=1e4)
    return p.parse_args()


def log_cp10k(X: np.ndarray, target_sum: float) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(log1p CP10K profile, original library size)`` for raw counts."""
    lib = X.sum(axis=1, keepdims=True)
    lib[lib == 0] = 1.0
    return np.log1p(X / lib * target_sum), lib


def main() -> None:
    """Center a submission and write the corrected copy."""
    args = parse_args()
    load_dotenv()
    data_dir = Path(os.environ["DATA_DIR"])

    src = Path(args.submission)
    out_path = Path(args.out or src.with_name(src.stem + "_centered.h5ad"))

    pred = ad.read_h5ad(src, backed="r")
    n_cells, n_genes = pred.shape
    ctx_obs = pred.obs["context"].to_numpy()
    console.log(f"{src.name}: {n_cells:,} x {n_genes:,}")

    # Row ranges are contiguous and context-major by construction; verify rather than
    # assume, since everything below slices by range.
    bounds = {}
    for c in CONTEXTS:
        idx = np.flatnonzero(ctx_obs == c)
        lo, hi = int(idx[0]), int(idx[-1]) + 1
        assert hi - lo == len(idx), f"context {c} rows are not contiguous"
        bounds[c] = (lo, hi)
    console.log(f"context row ranges: { {k: v for k, v in bounds.items()} }")

    # ---- pass 1: per-context mean predicted profile, and the control mean ----------
    shift = {}
    for c in CONTEXTS:
        lo, hi = bounds[c]
        acc = np.zeros(n_genes, dtype=np.float64)
        for s in range(lo, hi, CHUNK):
            e = min(s + CHUNK, hi)
            lg, _ = log_cp10k(
                np.asarray(pred.X[s:e].todense(), dtype=np.float64), args.target_sum
            )
            acc += lg.sum(axis=0)
        pred_mean = acc / (hi - lo)

        ctl = ad.read_h5ad(data_dir / f"vcc_2026/context_{c}.h5ad")
        ctl_lg, _ = log_cp10k(
            np.asarray(ctl.X.todense(), dtype=np.float64), args.target_sum
        )
        ctl_mean = ctl_lg.mean(axis=0)
        del ctl, ctl_lg

        # what every target shares, and therefore what carries no perturbation signal
        shift[c] = pred_mean - ctl_mean
        console.log(
            f"context {c}: common shift RMS {np.sqrt((shift[c] ** 2).mean()):.5f}, "
            f"mean {shift[c].mean():+.5f}, "
            f"max |.| {np.abs(shift[c]).max():.4f}"
        )

    # ---- pass 2: subtract it, re-invert to counts at the original library size -----
    blocks: list[csr_matrix] = []
    clipped_total, cells_total = 0, 0
    for c in CONTEXTS:
        lo, hi = bounds[c]
        for s in range(lo, hi, CHUNK):
            e = min(s + CHUNK, hi)
            X = np.asarray(pred.X[s:e].todense(), dtype=np.float64)
            lg, lib = log_cp10k(X, args.target_sum)

            lg -= shift[c]
            # log1p of a count is >= 0; the subtraction can push low-expression genes
            # below that, and those have to floor at zero rather than go negative.
            clipped_total += int((lg < 0).sum())
            np.clip(lg, 0.0, None, out=lg)

            counts = np.expm1(lg)
            mass = counts.sum(axis=1, keepdims=True)
            mass[mass == 0] = 1.0
            counts *= lib / mass  # back to this cell's own library size

            np.clip(counts, 0, None, out=counts)
            block = csr_matrix(np.rint(counts).astype(np.int32))
            block.eliminate_zeros()
            blocks.append(block)
            cells_total += e - s
        console.log(f"context {c}: {cells_total:,} cells done")

    console.log(
        f"floored at zero: {clipped_total:,} gene-cells "
        f"({100 * clipped_total / (n_cells * n_genes):.2f}%)"
    )

    X = vstack(blocks, format="csr")
    X.eliminate_zeros()
    del blocks

    obs = pred.obs.to_df() if hasattr(pred.obs, "to_df") else pred.obs.copy()
    var = pd.DataFrame(index=list(pred.var_names))

    # Same submission-spec checks predict_2026.py enforces.
    totals = np.asarray(X.sum(axis=1)).ravel()
    assert X.shape == (n_cells, n_genes), X.shape
    assert np.all(X.data == np.rint(X.data)), "counts must be whole numbers"
    assert X.data.min() >= 0 and np.isfinite(X.data).all()
    assert totals.max() <= MAX_COUNTS_PER_CELL, totals.max()
    assert X.nnz <= MAX_STORED_ENTRIES, X.nnz
    assert (X.data == 0).sum() == 0, "stored zeros count against the cap"
    assert obs.shape[0] == X.shape[0]
    assert obs.groupby(["context", "target_gene"]).size().unique().tolist() == [
        N_CELLS_PER_PERT
    ]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    ad.AnnData(X=X, obs=obs, var=var).write_h5ad(out_path, compression="gzip")
    console.log(
        f"[green]wrote[/green] {out_path}  {X.shape[0]:,} x {X.shape[1]:,} | "
        f"nnz {X.nnz:,} ({100 * X.nnz / MAX_STORED_ENTRIES:.1f}% of cap) | "
        f"cell totals {totals.min():,} - {totals.max():,} "
        f"(median {np.median(totals):,.0f})"
    )


if __name__ == "__main__":
    main()
