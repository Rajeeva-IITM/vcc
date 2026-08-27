"""Mann-Whitney U differential expression, every 2025 perturbation vs the shared NTC pool."""

import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import anndata as ad
import numpy as np
import polars as pl
from scipy.stats import mannwhitneyu

D = Path("/home/rajeeva/Project/vcc_data/2025")
OUT = Path(
    "/tmp/claude-1002/-home-rajeeva-Project-vcc/b2a7dbb1-86f0-48f5-9c0a-d5d0bbab6fcd/scratchpad"
)
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
    return row


def run(jobs, tag):
    """Run a batch of Mann-Whitney jobs in parallel and collect the rows."""
    t = time.time()
    out = []
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        for i, r in enumerate(ex.map(test_one, jobs), 1):
            out.append(r)
            if i % 25 == 0:
                print(f"  [{tag}] {i}/{len(jobs)}  {time.time() - t:.0f}s", flush=True)
    return out


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
    rows += run(jobs, "null")

    # ---- real perturbations, split by split --------------------------------
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
        rows += run(jobs, split)
        pl.DataFrame(rows).write_csv(OUT / "de_counts.csv")

    pl.DataFrame(rows).write_csv(OUT / "de_counts.csv")
    print(f"DONE {time.time() - t0:.0f}s -> de_counts.csv", flush=True)
