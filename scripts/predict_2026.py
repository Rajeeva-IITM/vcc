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
   normalise to `target_sum` over the depth axis the model was trained on, and run
   the model. That axis is the full model panel for a single-dataset run and a subset
   for a multi-dataset one; it is read from `norm_axis.csv` beside the checkpoint, so
   a checkpoint carries its own answer and the two cannot drift apart.
3. Invert the prediction to counts against that control cell's library size over
   the model genes -- rescaling the profile to target_sum first, so the predicted
   cell actually carries that library size.
4. Scatter those counts back to their 2026 positions, and for the 456 genes the
   model cannot predict, **copy the control cell's own raw counts**. Since the
   split of library size between predicted and copied genes is the control's, the
   total depth of the output cell matches the control it came from.

Usage
-----
    python scripts/predict_2026.py --model model_flow \\
        --ckpt <run>/VCC-epoch=89_step=217800.ckpt
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

from src.utils.data import build_gene_maps  # noqa: E402
from src.utils.predict import (  # noqa: E402
    finalize_counts,
    load_gene_embeddings,
    to_counts,
    to_model_space,
)

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
    p.add_argument(
        "--gene-embedding-path",
        nargs="+",
        default=None,
        help="perturbation embedding parquet(s), MUST match what --ckpt was trained on. "
        "One path -> used as-is; several -> per-block L2-normalise and concatenate, "
        "aligned on the FIRST path's gene order and zero-filling genes a later block "
        "lacks -- identical to the datamodule's `_load_embeddings`. Defaults to the "
        "256-d Poincare table for back-compat; a trio/concat checkpoint MUST pass its "
        "own list or the ko_vec dimension will not match the conditioner.",
    )
    p.add_argument(
        "--override",
        nargs="+",
        default=[],
        help="extra Hydra overrides the checkpoint was trained with, e.g. "
        "'+model.net.trunk.conditioner.context_dim=6346'. Must match the training run's "
        "CLI overrides or the state_dict will not load.",
    )
    p.add_argument("--device", default="cuda:1")
    p.add_argument("--out", default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--target-sum", type=float, default=1e4)
    p.add_argument(
        "--norm-axis",
        default=None,
        help="headerless CSV of the genes per-cell depth was normalised over during "
        "training. Defaults to `norm_axis.csv` beside --ckpt, and to the full model "
        "panel when there is none -- which is what every checkpoint trained on a single "
        "dataset used.",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="only predict the first N targets. Smoke-test only -- the result is NOT a "
        "valid submission and the spec assertions are skipped.",
    )
    return p.parse_args()


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
            "train.yaml",
            overrides=["data=dataset_anndata", f"model={args.model}", *args.override],
        )

    genes_model = pl.read_csv(
        data_dir / "2025/gene_names.csv", has_header=False
    )[  # TODO: Need to make it general
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

    model_to_26, present, unmapped_26 = build_gene_maps(genes_2026, genes_model)
    src_cols = model_to_26[present]  # 2026 columns feeding the model, in model order
    if (~present).sum():
        missing = [genes_model[i] for i in np.flatnonzero(~present)]
        console.log(
            f"  model genes absent from 2026 (predicted, then dropped): {missing}"
        )
    console.log(f"  {len(unmapped_26):,} 2026 genes copied from the control cell")

    # Depth axis. Training normalised each cell over the genes its datasets had in
    # common, which is the full panel for a single-dataset run and a subset for a
    # multi-dataset one. Getting this wrong shifts every input value, so it is resolved
    # from the run that produced the checkpoint rather than assumed.
    axis_path = (
        Path(args.norm_axis)
        if args.norm_axis
        else Path(args.ckpt).parent / "norm_axis.csv"
    )
    if axis_path.exists():
        norm_names = [
            str(g) for g in pl.read_csv(axis_path, has_header=False).to_series(0)
        ]
        pos = {g: i for i, g in enumerate(genes_model)}
        norm_cols = np.array([pos[g] for g in norm_names if g in pos])
        console.log(
            f"depth axis: {len(norm_cols):,} genes from {axis_path}"
            + ("" if args.norm_axis else " (found beside the checkpoint)")
        )
    else:
        norm_cols = np.arange(len(genes_model))
        console.log(
            f"depth axis: all {len(genes_model):,} model genes -- no {axis_path.name} "
            "beside the checkpoint, which is correct for single-dataset runs"
        )

    # Perturbation embeddings, reproduced exactly as the datamodule built the ko_vec.
    emb_paths = args.gene_embedding_path or [
        str(data_dir / "gene_embeddings/poincare_go_gaf_logmapped_256.parquet")
    ]
    embeddings, emb_dim = load_gene_embeddings(emb_paths)
    console.log(
        f"perturbation embedding: {emb_dim}-d from {len(emb_paths)} source(s): "
        + ", ".join(Path(p).name for p in emb_paths)
    )
    # Row index into the same table. A net holding a LearnableGeneEmbedding is addressed
    # by id rather than by vector; the ids must come from the SAME enumeration the ko_vec
    # (and the training table) used -- i.e. the first embedding source's gene order.
    gene_ids = {g: i for i, g in enumerate(embeddings)}
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
    # "ko_vec" for every model that predates the learnable table.
    pert_key = getattr(model, "pert_input_key", "ko_vec")
    console.log(f"model consumes the perturbation as {pert_key!r}")

    pairs: list[tuple[str, str]] = []
    blocks: list[csr_matrix] = []
    gate_sum, gate_n = 0.0, 0

    for context in CONTEXTS:
        path = data_dir / f"vcc_2026/context_{context}.h5ad"
        console.log(f"[bold]context {context}[/bold]: {path.name}")
        adata = ad.read_h5ad(path)
        X26 = csr_matrix(adata.X)
        n_ctrl = X26.shape[0]

        # Context-conditioned model: this 2026 context's pooled control profile over the norm
        # axis (norm_cols), the same vector the datamodule feeds. One fixed vector per context.
        context_t = None
        # The conditioner is `ko_processor` for single-FiLM nets and `trunk.conditioner`
        # for film_deep ones.
        conditioner = getattr(model.net, "ko_processor", None) or getattr(
            getattr(model.net, "trunk", None), "conditioner", None
        )
        if getattr(conditioner, "context_dim", None) is not None:
            m = min(4000, n_ctrl)
            sub = rng.choice(n_ctrl, m, replace=False)
            cm = np.zeros((m, len(genes_model)), dtype=np.float32)
            cm[:, present] = X26[sub].toarray().astype(np.float32)[:, src_cols]
            ce, _ = to_model_space(cm, norm_cols, args.target_sum)
            cvec = ce[:, norm_cols].mean(axis=0).astype(np.float32)
            context_t = torch.from_numpy(cvec).to(args.device)
            console.log(
                f"  context conditioning: {cvec.shape[0]}-d control profile from {m} controls"
            )

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
            # Two different totals -- see `to_model_space`. `lib` is the cell's depth
            # over the whole model panel and drives the inversion back to counts;
            # the input scaling uses its depth over the training norm axis instead.
            exp_vec, lib = to_model_space(ctrl_model, norm_cols, args.target_sum)

            with torch.inference_mode():
                batch = {
                    "ko_vec": embeddings[target]
                    .expand(N_CELLS_PER_PERT, -1)
                    .to(args.device),
                    "ko_id": torch.full(
                        (N_CELLS_PER_PERT,),
                        gene_ids[target],
                        dtype=torch.long,
                        device=args.device,
                    ),
                    "exp_vec": torch.from_numpy(exp_vec).to(args.device),
                }
                # FlowCellModel learns a velocity field, so predicting means
                # integrating it out of the control -- exactly what predict_step does.
                # The older CellModel variants emit the profile directly. Dispatch on
                # the API so both keep working from one script.
                if hasattr(model.net, "sample"):
                    out = model.net.sample(
                        batch["exp_vec"],
                        batch[pert_key],
                        num_steps=getattr(model, "num_sampling_steps", 4),
                        context=(
                            None
                            if context_t is None
                            else context_t.expand(N_CELLS_PER_PERT, -1)
                        ),
                    )
                else:
                    out = model.net(batch)
                # Only the gated decoder has a gate; the flow net has no `.decoder`.
                gate = getattr(getattr(model.net, "decoder", None), "last_gate", None)
                if gate is not None and gate.numel():
                    gate_sum += gate.float().mean().item()
                    gate_n += 1
                pred = out.float().cpu().numpy()

            # Invert to counts at the control cell's library size over model genes.
            counts_model = to_counts(pred, lib)

            # Scatter to 2026 order; copy control for genes the model cannot predict.
            out_row = np.zeros((N_CELLS_PER_PERT, len(genes_2026)), dtype=np.float64)
            out_row[:, src_cols] = counts_model[:, present]
            out_row[:, unmapped_26] = ctrl_full[:, unmapped_26]

            blocks.append(finalize_counts(out_row))
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
