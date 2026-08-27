"""Run prediction and write raw integer counts.

The models train on ``log1p(counts / depth * target_sum)``, so ``trainer.predict``
returns log-CP10K, not reads -- past submissions in ``results/prediction_*.h5ad``
went out on that scale (non-integral, max ~6.3). This script inverts back to
counts against the library size of the control cell each prediction was anchored
to, which the datamodule keeps as ``control_library_size``:

    counts = rint(expm1(y_pred) * control_library_size / target_sum)

That round-trips a control cell to its exact original integer counts (verified in
tests/test_anndata_module.py), so the only approximation is the model's own.

Usage
-----
    python scripts/predict_counts.py --ckpt <path/to/checkpoint.ckpt> \
        [--out results/prediction_counts.h5ad] [--device cuda:1] [--batch-size 128]
"""

import argparse
from datetime import date
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

console = Console()

# Rows are converted in blocks: 60,751 x 18,080 is 4.4 GB dense in float32, and
# the sparse result is much smaller than the dense intermediate.
BLOCK = 4096


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True, help="Lightning checkpoint to load")
    parser.add_argument("--out", default=None, help="output .h5ad")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--no-renormalise",
        action="store_true",
        help="skip rescaling each predicted profile to sum to target_sum; cell "
        "totals then drift from the control library size by however much mass "
        "the model happened to predict",
    )
    parser.add_argument(
        "--overrides",
        nargs="*",
        default=["data=dataset_anndata", "model=model_gated_bilinear"],
        help="Hydra overrides; must match the run that produced --ckpt",
    )
    return parser.parse_args()


def to_counts(
    pred: np.ndarray,
    library_size: np.ndarray,
    target_sum: float,
    renormalise: bool = True,
) -> csr_matrix:
    """Invert log-CP10K back to integer counts at the given library sizes.

    Nothing constrains the model to emit a valid CP10K profile, so
    ``expm1(y_pred).sum()`` is not ``target_sum`` -- on an untrained net it came
    out ~37% low, which would drag every cell's depth down with it. Rescaling to
    ``target_sum`` first is what actually makes the predicted cell carry the
    control's library size; it is a per-cell constant, so relative expression
    between genes is untouched.
    """
    counts = np.expm1(pred, dtype=np.float64)
    if renormalise:
        mass = counts.sum(axis=1, keepdims=True)
        mass[mass == 0] = 1.0
        counts *= target_sum / mass
    counts *= library_size[:, None] / target_sum
    # expm1 of a small negative would give a negative count; the models enforce
    # positivity, but clip anyway so the output is always spec-valid.
    np.clip(counts, 0, None, out=counts)
    return csr_matrix(np.rint(counts).astype(np.int32))


def main() -> None:
    """Predict, invert to counts, and write the h5ad."""
    args = parse_args()
    load_dotenv()

    with initialize_config_dir(
        config_dir=str(Path.cwd() / "config"), version_base=None
    ):
        conf = compose("train.yaml", overrides=list(args.overrides))

    console.log(f"Instantiating datamodule: {conf.data.datamodule._target_}")
    datamodule = hydra.utils.instantiate(conf.data.datamodule)
    datamodule.batch_size = args.batch_size
    datamodule.setup(stage="predict")

    dataset = datamodule.test_data
    if getattr(dataset, "control_library_size", None) is None:
        raise SystemExit(
            "The datamodule did not retain library sizes -- this script needs "
            "`data=dataset_anndata` (vcc_anndata_module), which keeps them."
        )

    console.log(f"Loading checkpoint {args.ckpt}")
    model = hydra.utils.instantiate(conf.model)
    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(state["state_dict"])
    model.eval().to(args.device)

    console.log(f"Predicting {len(dataset):,} cells on {args.device}")
    blocks: list[csr_matrix] = []
    gate_sum, gate_n = 0.0, 0

    with torch.inference_mode():
        buffer: list[np.ndarray] = []
        seen = 0
        for batch, _ in datamodule.predict_dataloader():
            batch = {k: v.to(args.device) for k, v in batch.items()}
            out = model(batch)
            if isinstance(out, (tuple, list)):  # projector models return (pred, latent)
                out = out[0]
            buffer.append(out.float().cpu().numpy())

            # The gate is unsupervised; report what it actually did.
            gate = getattr(getattr(model.net, "decoder", None), "last_gate", None)
            if gate is not None and gate.numel():
                gate_sum += gate.float().mean().item() * gate.shape[0]
                gate_n += gate.shape[0]

            if sum(len(b) for b in buffer) >= BLOCK:
                chunk = np.concatenate(buffer)
                blocks.append(
                    to_counts(
                        chunk,
                        dataset.control_library_size[seen : seen + len(chunk)],
                        datamodule.target_sum,
                        renormalise=not args.no_renormalise,
                    )
                )
                seen += len(chunk)
                buffer = []
                console.log(f"  {seen:,}/{len(dataset):,}")

        if buffer:
            chunk = np.concatenate(buffer)
            blocks.append(
                to_counts(
                    chunk,
                    dataset.control_library_size[seen : seen + len(chunk)],
                    datamodule.target_sum,
                    renormalise=not args.no_renormalise,
                )
            )
            seen += len(chunk)

    X = vstack(blocks, format="csr")
    X.eliminate_zeros()  # explicitly-stored zeros count against the submission cap
    del blocks

    if gate_n:
        console.log(f"mean gate activation: {gate_sum / gate_n:.4f}")

    assert X.shape[0] == len(dataset), (X.shape, len(dataset))
    assert np.all(X.data == np.rint(X.data)), "counts must be whole numbers"
    assert X.data.min() >= 0, "counts must be non-negative"

    out_path = Path(args.out or f"results/prediction_counts_{date.today():%d%m%y}.h5ad")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    result = ad.AnnData(
        X=X,
        obs=pd.DataFrame(
            {"target_gene": dataset.perturbed_genes},
            index=[f"cell_{i}" for i in range(X.shape[0])],
        ),
        var=pd.DataFrame(index=datamodule.gene_names),
    )
    result.write_h5ad(out_path, compression="gzip")

    totals = np.asarray(X.sum(axis=1)).ravel()
    ctrl = dataset.control_library_size
    console.log(
        f"control library sizes: median {np.median(ctrl):,.0f} "
        f"(renormalise={not args.no_renormalise})"
    )
    console.log(
        f"[green]wrote[/green] {out_path}  "
        f"{X.shape[0]:,} x {X.shape[1]:,} | nnz {X.nnz:,} | "
        f"cell totals {totals.min():,} - {totals.max():,} (median {np.median(totals):,.0f})"
    )


if __name__ == "__main__":
    main()
