"""Mann-Whitney U differential expression, every 2025 perturbation vs the shared NTC pool.

Writes two artefacts:

* ``de_counts_2025_mwu.csv`` -- one summary row per perturbation (and per null
  pseudo-perturbation), as before.
* ``de_mwu_2025.npz`` -- the **per-gene** q-values and log2 fold changes, ``(n_pert,
  n_genes)`` each, keyed by ``targets`` and ``genes``. This is what
  ``loss_functions.DEWeightedMSELoss`` reads; the summary counts throw it away.

The gene axis is ``2025/gene_names.csv`` order, which is byte-identical to the row order
of the gene-embedding parquet -- so a column here lines up with a ``ko_id``. The loss
re-checks that rather than trusting it.
"""

import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import anndata as ad
import numpy as np
import polars as pl
from dotenv import load_dotenv
from scipy.stats import mannwhitneyu

load_dotenv()
D = Path(os.environ["DATA_DIR"]) / "2025"
OUT = Path(os.environ["DATA_DIR"]) / "2025"
GENES = pl.read_csv(D / "gene_names.csv", has_header=False, new_columns=["g"])[
    "g"
].to_numpy()
CHUNK, WORKERS = 2000, 24
LFC_CUTS = (0.0, 0.1, 0.25, 0.5, 1.0)

C = None  # control matrix, log1p CP10K  (globals: shared with forks via COW)
XCUR = None  # current file's CSR


def norm(M):
    """CP10K-normalise a sparse count block and log1p it."""
    M = M.toarray().astype(np.float32)
    d = M.sum(1, keepdims=True)
    d[d == 0] = 1.0
    return np.log1p(M / d * 1e4)


def bh(p):
    """Benjamini-Hochberg FDR correction over a vector of p-values."""
    n = p.size
    o = np.argsort(p)
    q = np.minimum.accumulate((p[o] * n / np.arange(1, n + 1))[::-1])[::-1]
    out = np.empty(n)
    out[o] = np.minimum(q, 1.0)
    return out


def test_one(job):
    """job = (label, split, row_indices, ctrl_mask_or_None)"""
    label, split, rows, ctrl_rows = job
    P = norm(XCUR[rows])
    ctrl = C if ctrl_rows is None else C[ctrl_rows]
    pv = np.empty(P.shape[1], dtype=np.float64)
    for s in range(0, P.shape[1], CHUNK):
        e = min(s + CHUNK, P.shape[1])
        pv[s:e] = mannwhitneyu(
            P[:, s:e],
            ctrl[:, s:e],
            axis=0,
            alternative="two-sided",
            method="asymptotic",
        ).pvalue
    pv = np.nan_to_num(pv, nan=1.0)
    q = bh(pv)
    lfc = np.log2((np.expm1(P.mean(0)) + 1e-9) / (np.expm1(ctrl.mean(0)) + 1e-9))
    sig = q < 0.05
    row = {"target_gene": label, "split": split, "n_cells": int(P.shape[0])}
    for c in LFC_CUTS:
        m = sig & (np.abs(lfc) >= c)
        row[f"n_de_lfc{c}"] = int(m.sum())
        row[f"n_up_lfc{c}"] = int((m & (lfc > 0)).sum())
    hit = np.flatnonzero(GENES == label)
    row["self_lfc"] = float(lfc[hit[0]]) if hit.size else float("nan")
    row["self_q"] = float(q[hit[0]]) if hit.size else float("nan")
    # Return the per-gene vectors too. The summary counts above are a lossy projection of
    # exactly this, and it is the full thing the DE-weighted loss needs. float32 keeps the
    # 300 x 18,080 pair at ~44 MB.
    return row, q.astype(np.float32), lfc.astype(np.float32)


def run(jobs, tag):
    """Run a batch of Mann-Whitney jobs in parallel.

    Returns ``(summary_rows, per_gene_q, per_gene_lfc)``, the latter two aligned with
    ``jobs`` order.
    """
    t = time.time()
    out, qs, lfcs = [], [], []
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        for i, (r, q, lfc) in enumerate(ex.map(test_one, jobs), 1):
            out.append(r)
            qs.append(q)
            lfcs.append(lfc)
            if i % 25 == 0:
                print(f"  [{tag}] {i}/{len(jobs)}  {time.time() - t:.0f}s", flush=True)
    return out, qs, lfcs


if __name__ == "__main__":
    t0 = time.time()
    a = ad.read_h5ad(D / "train/adata_Training.h5ad")
    tg = a.obs["target_gene"].astype(str).to_numpy()
    gid = a.obs["guide_id"].astype(str).to_numpy()
    cmask = np.flatnonzero(tg == "non-targeting")
    C = norm(a.X[cmask])
    print(
        f"controls {C.shape} ({C.nbytes / 1e9:.2f} GB) in {time.time() - t0:.0f}s",
        flush=True,
    )

    rows = []
    # ---- null: each NTC guide vs the remaining NTC cells -------------------
    XCUR = a.X
    ntc_g = gid[cmask]
    jobs = []
    for g in np.unique(ntc_g):
        sel = np.flatnonzero(ntc_g == g)
        if sel.size < 200:
            continue
        keep = np.ones(cmask.size, bool)
        keep[sel] = False
        jobs.append((f"NTC::{g[:24]}", "null", cmask[sel], np.flatnonzero(keep)))
    print(f"null pseudo-perturbations: {len(jobs)}", flush=True)
    # Null rows calibrate the false-positive floor in the CSV; they are not perturbations,
    # so their per-gene vectors are dropped rather than written to the npz.
    null_rows, _, _ = run(jobs, "null")
    rows += null_rows

    # ---- real perturbations, split by split --------------------------------
    targets, all_q, all_lfc = [], [], []
    for split, f in (
        ("train", "train/adata_Training.h5ad"),
        ("validation", "validation/adata_Validation.h5ad"),
        ("test", "test/adata_Test.h5ad"),
    ):
        if split != "train":
            del a
            a = ad.read_h5ad(D / f)
            tg = a.obs["target_gene"].astype(str).to_numpy()
        XCUR = a.X
        genes = sorted(set(tg) - {"non-targeting"})
        jobs = [(g, split, np.flatnonzero(tg == g), None) for g in genes]
        print(f"{split}: {len(jobs)} perturbations", flush=True)
        split_rows, qs, lfcs = run(jobs, split)
        rows += split_rows
        targets += genes
        all_q += qs
        all_lfc += lfcs
        pl.DataFrame(rows).write_csv(OUT / "de_counts_2025_mwu.csv")

    pl.DataFrame(rows).write_csv(OUT / "de_counts_2025_mwu.csv")

    q_mat = np.stack(all_q)
    lfc_mat = np.stack(all_lfc)
    assert q_mat.shape == (len(targets), GENES.size), q_mat.shape
    # Explicit unicode dtype, not object: polars' .to_numpy() on a string column gives
    # dtype=object, which npz can only store as a pickle -- and the loss loads with
    # allow_pickle=False on purpose.
    np.savez_compressed(
        OUT / "de_mwu_2025.npz",
        targets=np.asarray(targets, dtype=np.str_),
        genes=GENES.astype(np.str_),
        q=q_mat,
        lfc=lfc_mat,
    )
    sig = (q_mat < 0.05) & (np.abs(lfc_mat) >= 0.25)
    print(
        f"DONE {time.time() - t0:.0f}s -> de_counts_2025_mwu.csv, de_mwu_2025.npz "
        f"{q_mat.shape} | median {np.median(sig.sum(1)):.0f} DE genes "
        f"({100 * sig.mean():.2f}% of all entries)",
        flush=True,
    )
