"""Score a trained model on the held-out 2025 validation / test splits.

The 2025 validation (50 perturbations) and test (100 perturbations) gene sets are
disjoint from the 150 the model trains on, so this is a genuine OOD check and the
closest local proxy for the 2026 leaderboard.

Two things this handles that a plain predict does not:

1. **Counts, not log-CP10K.** The model emits ``log1p(counts/depth * target_sum)``.
   It is inverted against the library size of the control cell each prediction
   was anchored to, rescaling the profile to ``target_sum`` first so the cell
   actually carries that library size (see scripts/predict_counts.py).
2. **Control cells are part of the submission.** ``adata_Validation.h5ad`` is
   98,927 cells = 60,751 perturbed + 38,176 non-targeting, and cell-eval needs
   the controls to compute DE and the discrimination score. The real NTC cells
   are passed through as-is; only the perturbed cells are predicted.

Usage
-----
    python scripts/validate_2025.py --ckpt <run>/last.ckpt --split validation
    python scripts/validate_2025.py --ckpt <run>/last.ckpt --split test
    python scripts/validate_2025.py --ckpt <run>/last.ckpt --split both
"""

import argparse
import os
import subprocess
from pathlib import Path

import anndata as ad
import hydra
import numpy as np
import pandas as pd
import rootutils
import torch
from dotenv import load_dotenv
from hydra import compose, initialize_config_dir
from rich.console import Console
from scipy.sparse import csr_matrix, vstack

rootutils.setup_root(__file__, indicator="pixi.toml", pythonpath=True)

from scripts.predict_counts import to_counts  # noqa: E402

console = Console()

BLOCK = 4096

SPLITS = {
    "validation": (
        "2025/validation/adata_Validation.h5ad",
        "2025/validation/pert_counts_Validation.csv",
    ),
    "test": ("2025/test/adata_Test.h5ad", "2025/test/pert_counts_Test.csv"),
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--split", default="validation", choices=[*SPLITS, "both"])
    p.add_argument("--device", default="cuda:1")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--outdir", default="results/2025-eval")
    p.add_argument("--model", default="model_gated_bilinear", help="must match --ckpt")
    p.add_argument("--no-renormalise", action="store_true")
    p.add_argument(
        "--skip-score", action="store_true", help="write the h5ad, do not run cell-eval"
    )
    return p.parse_args()


def predict_split(args, split: str, data_dir: Path) -> Path:
    """Predict one 2025 split, convert to counts, and write a scorable h5ad."""
    h5ad_rel, counts_rel = SPLITS[split]
    real_path = data_dir / h5ad_rel

    with initialize_config_dir(
        config_dir=str(Path.cwd() / "config"), version_base=None
    ):
        conf = compose(
            "train.yaml",
            overrides=[
                "data=dataset_anndata",
                f"model={args.model}",
                # Predict this split's perturbations, using this split's own
                # control cells. (The 38,176 NTC cells are the same set in all
                # three 2025 files, so this is self-consistent either way.)
                f"data.datamodule.data_path=[{real_path}]",
                f"data.datamodule.pert_counts_path={data_dir / counts_rel}",
            ],
        )

    console.log(f"[bold]{split}[/bold]: setting up from {real_path.name}")
    dm = hydra.utils.instantiate(conf.data.datamodule)
    dm.batch_size = args.batch_size
    dm.setup(stage="predict")
    dataset = dm.test_data

    model = hydra.utils.instantiate(conf.model)
    model.load_state_dict(
        torch.load(args.ckpt, map_location="cpu", weights_only=False)["state_dict"]
    )
    model.eval().to(args.device)

    console.log(f"Predicting {len(dataset):,} perturbed cells on {args.device}")
    blocks: list[csr_matrix] = []
    buffer: list[np.ndarray] = []
    seen = 0
    gate_sum, gate_n = 0.0, 0

    def flush():
        nonlocal blocks, buffer, seen
        if not buffer:
            return
        chunk = np.concatenate(buffer)
        blocks.append(
            to_counts(
                chunk,
                dataset.control_library_size[seen : seen + len(chunk)],
                dm.target_sum,
                renormalise=not args.no_renormalise,
            )
        )
        seen += len(chunk)
        buffer = []

    with torch.inference_mode():
        for batch, _ in dm.predict_dataloader():
            batch = {k: v.to(args.device) for k, v in batch.items()}
            out = model(batch)
            if isinstance(out, (tuple, list)):
                out = out[0]
            buffer.append(out.float().cpu().numpy())

            gate = getattr(getattr(model.net, "decoder", None), "last_gate", None)
            if gate is not None and gate.numel():
                gate_sum += gate.float().mean().item() * gate.shape[0]
                gate_n += gate.shape[0]

            if sum(len(b) for b in buffer) >= BLOCK:
                flush()
                console.log(f"  {seen:,}/{len(dataset):,}")
    flush()

    if gate_n:
        mean_gate = gate_sum / gate_n
        console.log(f"mean gate activation: {mean_gate:.4f}")
        if mean_gate > 0.9:
            console.log(
                "[yellow]gate is near 1.0 -- the mask is inactive and the model "
                "is effectively predicting absolute expression[/yellow]"
            )

    X_pred = vstack(blocks, format="csr")
    del blocks

    # Pass the real control cells through untouched; they are part of what the
    # scorer reads, and predicting them was never the task.
    console.log("Appending the real non-targeting cells")
    real = ad.read_h5ad(real_path)
    labels = real.obs["target_gene"].to_numpy().astype(str)
    ntc = np.flatnonzero(labels == "non-targeting")
    X_ntc = csr_matrix(real.X)[ntc].astype(np.int32)

    X = vstack([X_pred, X_ntc], format="csr")
    X.eliminate_zeros()  # stored zeros count against the submission cap

    obs = pd.DataFrame(
        {
            "target_gene": np.concatenate(
                [dataset.perturbed_genes, np.full(len(ntc), "non-targeting")]
            )
        },
        index=[f"cell_{i}" for i in range(X.shape[0])],
    )

    assert X.shape == real.shape, (X.shape, real.shape)
    assert np.all(X.data == np.rint(X.data)), "counts must be whole numbers"
    assert X.data.min() >= 0
    assert set(obs["target_gene"]) == set(labels), "perturbation sets differ"

    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"pred_{split}.h5ad"
    ad.AnnData(X=X, obs=obs, var=pd.DataFrame(index=dm.gene_names)).write_h5ad(
        out_path, compression="gzip"
    )

    totals = np.asarray(X.sum(axis=1)).ravel()
    console.log(
        f"[green]wrote[/green] {out_path}  {X.shape[0]:,} x {X.shape[1]:,} | "
        f"nnz {X.nnz:,} | median total {np.median(totals):,.0f}"
    )
    return out_path


def score(pred_path: Path, real_path: Path, outdir: Path) -> None:
    """Run the official scorer with the VCC metric profile."""
    cmd = [
        ".pixi/envs/default/bin/cell-eval",
        "run",
        "-ap",
        str(pred_path),
        "-ar",
        str(real_path),
        "--profile",
        "vcc",
        "--control-pert",
        "non-targeting",
        "--pert-col",
        "target_gene",
        "-o",
        str(outdir),
    ]
    console.log(f"[bold]scoring[/bold]: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main() -> None:
    """Predict and score each requested split."""
    args = parse_args()
    load_dotenv()
    data_dir = Path(os.environ["DATA_DIR"])

    splits = list(SPLITS) if args.split == "both" else [args.split]
    for split in splits:
        pred = predict_split(args, split, data_dir)
        if not args.skip_score:
            score(
                pred, data_dir / SPLITS[split][0], Path(args.outdir) / f"{split}-eval"
            )


if __name__ == "__main__":
    main()
