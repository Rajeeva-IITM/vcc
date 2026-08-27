"""Build the CP10K-normalised expression matrices used for training.

The original pipeline stored ``log1p(raw UMI counts)``. On that scale a cell's
profile is almost entirely determined by its sequencing depth
(``corr(log total UMI, mean expression) = 0.993``), and only ~2% of the per-gene
variance is between-perturbation signal. Since each perturbed cell is paired with
a *randomly drawn* control cell, that depth component is unpredictable by
construction, so the model spends most of its capacity on irreducible noise.

This script applies the standard single-cell normalisation instead --
counts-per-10k followed by ``log1p`` -- and writes the result to a new
``cp10k-processed-data/`` directory, leaving ``log-processed-data/`` untouched so
earlier runs and submissions stay reproducible.

Usage
-----
    python src/prepare_data.py              # write the normalised matrices
    python src/prepare_data.py --calibrate  # only print DE-threshold diagnostics

``DATA_DIR`` is read from the environment or from a ``.env`` file at the repo root.
"""

import argparse
import os
from pathlib import Path

import numpy as np
import polars as pl
import polars.selectors as cs
import torch
from dotenv import load_dotenv
from rich.console import Console

console = Console()

TARGET_SUM = 5e4
CHUNK_ROWS = 20_000  # ~1.4 GB per chunk at 18k float32 columns

# (source under processed-data/, destination under cp10k-processed-data/)
MATRICES = [
    ("training_data-counts_uint.parquet", "training_data-counts.parquet"),
    ("control_exp_data_uint.parquet", "control_exp_data.parquet"),
]


def _numeric_columns(path: Path) -> list[str]:
    schema = pl.scan_parquet(path).collect_schema()
    return [n for n, d in zip(schema.names(), schema.dtypes()) if d.is_numeric()]


def normalise_matrix(src: Path, dst: Path) -> None:
    """Write ``log1p(counts / row_total * 1e4)`` for every numeric column.

    Processed in row chunks: the full matrix is ~183k x 18k, which is 13 GB dense
    in float32, and polars' horizontal sum over 18k columns is slow enough that
    a numpy round-trip per chunk is the faster path.
    """
    genes = _numeric_columns(src)
    total_rows = pl.scan_parquet(src).select(pl.len()).collect().item()
    console.log(f"{src.name}: {total_rows:,} rows x {len(genes):,} genes -> {dst.name}")

    parts: list[pl.DataFrame] = []
    for start in range(0, total_rows, CHUNK_ROWS):
        chunk = pl.scan_parquet(src).slice(start, CHUNK_ROWS).collect()
        counts = chunk.select(genes).to_numpy().astype(np.float32)
        depth = counts.sum(axis=1, keepdims=True)
        # A cell with zero counts would divide by zero; there are none in this
        # dataset, but guard anyway so the script is safe to re-run on new data.
        depth[depth == 0] = 1.0
        normalised = np.log1p(counts / depth * TARGET_SUM)

        part = pl.DataFrame(normalised, schema=genes)
        if "sample_index" in chunk.columns:  # preserve the join key
            part = part.with_columns(chunk["sample_index"]).select(
                "sample_index", *genes
            )
        parts.append(part)
        console.log(f"  rows {start:,}-{min(start + CHUNK_ROWS, total_rows):,}")

    dst.parent.mkdir(parents=True, exist_ok=True)
    pl.concat(parts, how="vertical").write_parquet(dst)
    console.log(f"[green]wrote[/green] {dst}")


def write_control_std(control_path: Path, dst: Path) -> None:
    """Per-gene std of the normalised control cells.

    Consumed by ``projection_model_consistent.yaml`` as the scale of the noise it
    adds for the consistency term. The existing file is on the old log1p(raw)
    scale and would inject noise several times too large here.
    """
    ctrl = pl.read_parquet(control_path).select(cs.numeric()).to_numpy()
    std = torch.from_numpy(ctrl.std(axis=0)).float()
    torch.save(std, dst)
    console.log(f"[green]wrote[/green] {dst}  (mean per-gene std {std.mean():.4f})")


def calibrate(data_dir: Path, n_rows: int = 15_000) -> None:
    """Print the DE-threshold diagnostics used to set the loss configs.

    The DE-aware losses gate on ``|y_true - control_exp|``. Those thresholds were
    tuned against log1p(raw counts); on the CP10K scale the same numbers select
    almost nothing, which silently disables the losses. This reports what each
    threshold actually selects so the configs can be set from evidence.
    """
    rng = np.random.default_rng(0)

    def load(name: str) -> np.ndarray:
        X = (
            pl.read_parquet(data_dir / "processed-data" / name, n_rows=n_rows)
            .select(cs.numeric())
            .to_numpy()
            .astype(np.float32)
        )
        return np.log1p(X / X.sum(axis=1, keepdims=True) * TARGET_SUM)

    pert = load("training_data-counts_uint.parquet")
    ctrl = load("control_exp_data_uint.parquet")
    # Mirror what the loss sees: a perturbed cell minus a randomly paired control.
    delta = np.abs(pert - ctrl[rng.integers(0, len(ctrl), len(pert))])

    console.print("\n[bold]|y_true - control| on the CP10K scale[/bold]")
    for q in (50, 75, 90, 95, 97.5, 99):
        console.print(f"  p{q:<6} {np.percentile(delta, q):.4f}")

    console.print("\n[bold]fraction of gene-cells called DE per threshold[/bold]")
    for t in (0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 1.5):
        console.print(f"  threshold {t:<5} -> {100 * (delta > t).mean():5.2f}%")
    console.print(
        "\nConfigs use threshold=0.4 (~10% DE). The pre-normalisation value of "
        "1.5 would select 0.02% here."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="only print DE-threshold diagnostics, write nothing",
    )
    args = parser.parse_args()

    load_dotenv()
    data_dir = os.environ.get("DATA_DIR")
    if data_dir is None:
        raise SystemExit("DATA_DIR is not set (export it or add it to .env)")
    data_dir = Path(data_dir)

    if args.calibrate:
        calibrate(data_dir)
        return

    out_dir = data_dir / "cp10k-processed-data"
    for src_name, dst_name in MATRICES:
        normalise_matrix(data_dir / "processed-data" / src_name, out_dir / dst_name)

    write_control_std(
        out_dir / "control_exp_data.parquet", out_dir / "control_expression_std.pt"
    )
    calibrate(data_dir)


if __name__ == "__main__":
    main()
