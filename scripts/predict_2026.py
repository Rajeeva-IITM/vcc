"""Build the 2026 submission: raw integer counts for 300 targets x 3 contexts.

The model was trained on the 2025 panel (18,080 genes); the 2026 data has 18,533.
Of those, 18,077 are shared, 3 are 2025-only (dup-symbol artefacts), and 456 are
2026-only. Critically the shared genes are **not in the same relative order** in
the two panels, so columns are mapped by name, never by position.

Per the spec each (target_gene, context) needs 400 cells, giving
300 x 3 x 400 = 360,000 rows of raw counts over the 2026 gene order.

How each cell is built
----------------------
1. Draw a real control cell from that context.
2. Take its counts at the 18,080 model genes (the 3 absent ones read 0),
   CP10K-normalise, and run the model.
3. Invert the prediction to counts against that control cell's library size over
   the model genes -- rescaling the profile to target_sum first, so the predicted
   cell actually carries that library size.
4. Scatter those counts back to their 2026 positions, and for the 456 genes the
   model cannot predict, **copy the control cell's own raw counts**. Since the
   split of library size between predicted and copied genes is the control's, the
   total depth of the output cell matches the control it came from.

Usage
-----
    python scripts/predict_2026.py --ckpt <run>/VCC-epoch=21_step=25784.ckpt
"""

import argparse
from datetime import date
from pathlib import Path

import anndata as ad
import hydra
import numpy as np
import pandas as pd
import polars as pl
import rootutils
import torch
from dotenv import load_dotenv
from hydra import compose, initialize_config_dir
from rich.console import Console
from scipy.sparse import csr_matrix, vstack

rootutils.setup_root(__file__, indicator="pixi.toml", pythonpath=True)

console = Console()

N_CELLS_PER_PERT = 400  # fixed by the submission spec
CONTEXTS = ("A", "B", "C")
MAX_COUNTS_PER_CELL = 1_000_000
MAX_STORED_ENTRIES = 4_750_000_000


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--model", default="model_gated_bilinear", help="must match --ckpt")
    p.add_argument("--device", default="cuda:1")
    p.add_argument("--out", default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--target-sum", type=float, default=1e4)
    return p.parse_args()


def build_gene_maps(genes_2026: list[str], genes_model: list[str]):
    """Map model gene order -> 2026 column index, by NAME.

    Returns ``(model_to_26, present, unmapped_26)`` where ``model_to_26[i]`` is the
    2026 column for model gene *i* (0 where absent, masked by ``present``), and
    ``unmapped_26`` lists the 2026 columns no model gene covers.
    """
    pos = {g: i for i, g in enumerate(genes_2026)}
    model_to_26 = np.zeros(len(genes_model), dtype=np.int64)
    present = np.zeros(len(genes_model), dtype=bool)
    for i, g in enumerate(genes_model):
        j = pos.get(g)
        if j is not None:
            model_to_26[i], present[i] = j, True

    covered = np.zeros(len(genes_2026), dtype=bool)
    covered[model_to_26[present]] = True
    unmapped_26 = np.flatnonzero(~covered)

    console.log(
        f"gene map: {present.sum():,}/{len(genes_model):,} model genes found in the "
        f"2026 panel | {len(unmapped_26):,} 2026 genes copied from control"
    )
    if (~present).sum():
        missing = [genes_model[i] for i in np.flatnonzero(~present)]
        console.log(
            f"  model genes absent from 2026 (predicted, then dropped): {missing}"
        )
    return model_to_26, present, unmapped_26


def main() -> None:
    """Build the 360,000-cell 2026 submission and write it out."""
    args = parse_args()
    load_dotenv()
    import os

    data_dir = Path(os.environ["DATA_DIR"])
    rng = np.random.default_rng(args.seed)

    with initialize_config_dir(
        config_dir=str(Path.cwd() / "config"), version_base=None
    ):
        conf = compose(
            "train.yaml", overrides=["data=dataset_anndata", f"model={args.model}"]
        )

    genes_model = pl.read_csv(data_dir / "2025/gene_names.csv", has_header=False)[
        "column_1"
    ].to_list()
    genes_2026 = pl.read_csv(data_dir / "vcc_2026/gene_names.csv")[
        "gene_name"
    ].to_list()
    targets = pl.read_csv(data_dir / "vcc_2026/pert_counts.csv")[
        "target_gene"
    ].to_list()
    console.log(
        f"{len(targets)} targets x {len(CONTEXTS)} contexts x {N_CELLS_PER_PERT} cells"
    )

    model_to_26, present, unmapped_26 = build_gene_maps(genes_2026, genes_model)
    src_cols = model_to_26[present]  # 2026 columns feeding the model, in model order

    # Perturbation embeddings, same source the datamodule uses.
    emb = pl.read_parquet(
        data_dir / "gene_embeddings/poincare_go_gaf_logmapped_256.parquet"
    )
    import polars.selectors as cs

    vectors = emb.select(cs.numeric()).to_torch()
    embeddings = {g: vectors[i] for i, g in enumerate(emb["gene_name"].to_numpy())}
    missing_emb = [t for t in targets if t not in embeddings]
    if missing_emb:
        raise KeyError(
            f"{len(missing_emb)} targets have no embedding: {missing_emb[:5]}"
        )

    model = hydra.utils.instantiate(conf.model)
    model.load_state_dict(
        torch.load(args.ckpt, map_location="cpu", weights_only=False)["state_dict"]
    )
    model.eval().to(args.device)

    pairs: list[tuple[str, str]] = []
    blocks: list[csr_matrix] = []
    gate_sum, gate_n = 0.0, 0

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

            # Model input: control counts at the model's genes, CP10K + log1p.
            ctrl_model = np.zeros(
                (N_CELLS_PER_PERT, len(genes_model)), dtype=np.float32
            )
            ctrl_model[:, present] = ctrl_full[:, src_cols]
            lib = ctrl_model.sum(axis=1, keepdims=True)
            lib[lib == 0] = 1.0
            exp_vec = np.log1p(ctrl_model / lib * args.target_sum)

            with torch.inference_mode():
                batch = {
                    "ko_vec": embeddings[target]
                    .expand(N_CELLS_PER_PERT, -1)
                    .to(args.device),
                    "exp_vec": torch.from_numpy(exp_vec).to(args.device),
                }
                out = model.net(batch)
                gate = getattr(model.net.decoder, "last_gate", None)
                if gate is not None and gate.numel():
                    gate_sum += gate.float().mean().item()
                    gate_n += 1
                pred = out.float().cpu().numpy()

            # Invert to counts at the control cell's library size over model genes.
            counts_model = np.expm1(pred, dtype=np.float64)
            mass = counts_model.sum(axis=1, keepdims=True)
            mass[mass == 0] = 1.0
            counts_model *= lib / mass  # rescale to target_sum, then to lib

            # Scatter to 2026 order; copy control for genes the model cannot predict.
            out_row = np.zeros((N_CELLS_PER_PERT, len(genes_2026)), dtype=np.float64)
            out_row[:, src_cols] = counts_model[:, present]
            out_row[:, unmapped_26] = ctrl_full[:, unmapped_26]

            np.clip(out_row, 0, None, out=out_row)
            block = csr_matrix(np.rint(out_row).astype(np.int32))
            block.eliminate_zeros()
            blocks.append(block)
            pairs.extend([(target, context)] * N_CELLS_PER_PERT)

            if (t_i + 1) % 50 == 0:
                console.log(f"  {t_i + 1}/{len(targets)} targets")

        del adata, X26

    if gate_n:
        console.log(f"mean gate activation: {gate_sum / gate_n:.4f}")

    X = vstack(blocks, format="csr")
    X.eliminate_zeros()
    del blocks

    obs = pd.DataFrame(
        {"target_gene": [p[0] for p in pairs], "context": [p[1] for p in pairs]},
        index=[f"cell_{i}" for i in range(X.shape[0])],
    )

    # Every assertion below is a submission-spec requirement.
    totals = np.asarray(X.sum(axis=1)).ravel()
    assert X.shape == (
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
        args.out or f"results/2026/prediction_2026_{date.today():%d%m%y}.h5ad"
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
