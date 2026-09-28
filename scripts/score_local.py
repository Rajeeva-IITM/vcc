"""Score a checkpoint with the official 2026 metrics, against a 2025-derived reference.

Why this exists
---------------
Every idea currently costs a leaderboard submission to test, and the last full training
run bought +0.0009 overall. `cell-eval2`'s `vcc2026` preset IS the competition scorer, so
nothing here reimplements a metric; what the 2026 season withholds is the ground truth to
score against, and that is what this script manufactures from the 2025 data.

Absolute values will not match the leaderboard -- a different cell line, a different
panel, a different set of perturbations. Differences between our own checkpoints will,
because both ends of the scale are rebuilt from the same reference.

What it scores on
-----------------
The 60 perturbations the datamodule holds out, recomputed from `(sorted pert names,
test_size, seed)` via the same `gene_train_test_split` training uses.

⚠️ These are NOT the 2025 challenge validation split (overlap 12/50). Training reads all
of `adata_2025_all.h5ad` including the challenge splits, so scoring on
`adata_Validation.h5ad` -- which is what `scripts/validate_2025.py` does -- would leak.
`test_held_out_set_is_not_the_challenge_split` guards this.

Everything is derived
---------------------
The data and model configs come from the wandb config beside the checkpoint, the depth
axis from its `norm_axis.csv`, the held-out genes from the split seed. The reference and
its bundle are content-addressed and built on first use. There is no prep step to
remember and no path to pass.

Usage
-----
    pixi run python scripts/score_local.py --ckpt <run>/last.ckpt
    pixi run python scripts/score_local.py --ckpt a.ckpt --ckpt b.ckpt   # compare
    pixi run python scripts/score_local.py --ckpt <run>/last.ckpt --limit 6   # smoke
"""

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import anndata as ad
import h5py
import hydra
import numpy as np
import pandas as pd
import polars as pl
import rootutils
import torch
from dotenv import load_dotenv
from hydra import compose, initialize_config_dir
from rich.console import Console
from rich.table import Table
from scipy.sparse import csr_matrix, vstack

rootutils.setup_root(__file__, indicator="pixi.toml", pythonpath=True)

from src.data.source_cache import (  # noqa: E402
    CONTROL_LABEL_CANDIDATES,
    PERT_KEY_CANDIDATES,
    _read_var_names,
)
from src.utils.gene_train_test_split import gene_train_test_split  # noqa: E402
from src.utils.predict import (  # noqa: E402
    finalize_counts,
    load_gene_embeddings,
    to_counts,
    to_model_space,
)

console = Console()

CONTROL_LABEL = "non-targeting"
PERT_COL = "target_gene"

# The six the competition averages, in the order we report them.
SCORED = (
    "pds_cosine",
    "expr_mse_unbiased_capped_norm",
    "de_wilcoxon_lfc_nmae",
    "de_wilcoxon_direction_fidelity_yield_raw",
    "de_wilcoxon_direction_reach_raw",
    "de_wilcoxon_sig_jaccard",
)
SHORT = {
    "pds_cosine": "pds",
    "expr_mse_unbiased_capped_norm": "mse",
    "de_wilcoxon_lfc_nmae": "nmae",
    "de_wilcoxon_direction_fidelity_yield_raw": "fid",
    "de_wilcoxon_direction_reach_raw": "reach",
    "de_wilcoxon_sig_jaccard": "jac",
}

# ⚠️ Built ONCE and spliced into every invocation. `score --real-bundle` refuses a
# submission whose run_meta.json disagrees with the bundle manifest on SUBMISSION_PEERS
# (config_digest, comparator, resolved_de_backend, ...), so the bundle build and the
# scoring run must be configured identically. One list is the mechanism that guarantees
# that; two lists would drift and the failure would arrive an hour into a build.
#
#   --pert-col   the preset says `target`; every h5ad in this repo says `target_gene`
#   de.backend   `gpudge` -- the GPU Wilcoxon engine, and the SAME engine the 2026
#                leaderboard runs, so a local score is now faithful to it rather than a
#                CPU substitute. gpudge runs the DE on torch/CUDA INDEPENDENT of `device`
#                below (that knob governs only the pseudobulk step), so it needs no cupy;
#                its deps (scipy>=1.17, polars>=1.38, numpy>=2, torch>=2.5) are all in the
#                env. `auto` would also resolve to gpudge here, but pinning it keeps
#                `resolved_de_backend` -- a SUBMISSION_PEERS field -- fixed. NOTE: switching
#                off `scanpy` changes every DE number, so any bundle previously built with
#                the CPU engine has a different content digest and is rebuilt on first use;
#                do not compare a gpudge score to an old scanpy one.
#
# All three of `pert_col`, `device` and `de.backend` sit in `competition._RULE_EXCLUDED` /
# `_RULE_EXCLUDED_NESTED`, so overriding them still yields a bundle stamped as the
# competition rule rather than a diagnostic one.
#   device       `cpu` -- this knob governs cell-eval2's pseudobulk (grouped-mean) step,
#                NOT the gpudge DE. `cuda` here would take the GPU pseudobulk path, which
#                DOES require cupy (the 'gpu' extra) and raises without it. The pseudobulk
#                is cheap next to the DE, so CPU is fine and keeps cupy out of the env. The
#                expensive Wilcoxon still runs on the GPU via gpudge.
COMMON = [
    "--preset",
    "vcc2026",
    "--pert-col",
    PERT_COL,
    "--set",
    "de.backend=gpudge",
    "--set",
    "device=cpu",
]

CELL_EVAL2 = ".pixi/envs/default/bin/cell-eval2"


# --------------------------------------------------------------------------------------
# checkpoint context, all of it derived
# --------------------------------------------------------------------------------------


def discover_configs(ckpt: Path) -> tuple[str, str]:
    """The `model=` and `data=` hydra overrides the run that produced `ckpt` was given.

    Read from the wandb config beside the checkpoint, which records the literal argv. A
    training run without wandb leaves nothing to read, and guessing here would silently
    score a checkpoint under the wrong depth axis -- so that case is an error naming the
    two flags that fix it.
    """
    cfg = ckpt.parent / "wandb/latest-run/files/config.yaml"
    if not cfg.exists():
        raise SystemExit(
            f"no wandb config at {cfg} -- cannot tell which configs produced this "
            "checkpoint. Pass --model and --data explicitly."
        )
    text = cfg.read_text()
    found = {}
    for line in text.splitlines():
        line = line.strip().lstrip("- ").strip()
        for key in ("model", "data"):
            if line.startswith(f"{key}="):
                found[key] = line.split("=", 1)[1].strip()
    missing = {"model", "data"} - set(found)
    if missing:
        raise SystemExit(
            f"{cfg} records no {sorted(missing)} override. Pass --model/--data explicitly."
        )
    return found["model"], found["data"]


def resolve_norm_cols(ckpt: Path, genes_model: list[str], conf) -> np.ndarray:
    """The training depth axis, as column indices into the model panel.

    A multi-source run normalises over the 6,691 genes its datasets share; a single-source
    run over all 18,080. Getting this wrong shifts every input value the model sees, so it
    is read from the run that produced the checkpoint rather than assumed.
    """
    axis = ckpt.parent / "norm_axis.csv"
    if not axis.exists():
        paths = conf.data.datamodule.data_path
        n_sources = 1 if isinstance(paths, str) else len(paths)
        if n_sources > 1:
            raise SystemExit(
                f"{ckpt.name} trained on {n_sources} sources, so its depth axis is a "
                f"subset of the panel -- but no norm_axis.csv sits beside it. Predicting "
                "on the full panel would shift every input value the model sees."
            )
        console.log(
            f"depth axis: all {len(genes_model):,} model genes -- no norm_axis.csv beside "
            "the checkpoint, which is correct for single-source runs"
        )
        return np.arange(len(genes_model))
    names = [str(g) for g in pl.read_csv(axis, has_header=False).to_series(0)]
    pos = {g: i for i, g in enumerate(genes_model)}
    cols = np.array([pos[g] for g in names if g in pos])
    console.log(f"depth axis: {len(cols):,} genes from {axis}")
    return cols


def load_model(ckpt: Path, model_cfg: str, data_cfg: str, device: str):
    """Instantiate from hydra and load the raw state dict -- the repo's standard idiom."""
    with initialize_config_dir(
        config_dir=str(Path.cwd() / "config"), version_base=None
    ):
        conf = compose(
            "train.yaml", overrides=[f"data={data_cfg}", f"model={model_cfg}"]
        )
    model = hydra.utils.instantiate(conf.model)
    model.load_state_dict(
        torch.load(ckpt, map_location="cpu", weights_only=False)["state_dict"]
    )
    model.eval().to(device)
    return model, conf


# --------------------------------------------------------------------------------------
# the held-out set
# --------------------------------------------------------------------------------------


def read_obs_perturbations(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """`(per-cell labels, sorted non-control categories)` -- via h5py, X untouched.

    The perturbation column and control label are auto-detected per source (2025 uses
    ``target_gene``/``non-targeting``; the PrimeFlow files use ``perturbation``/``control``),
    the same candidate lists ``source_cache`` uses. Controls are relabelled to the module
    ``CONTROL_LABEL`` so every downstream step (reference build, cell-eval2 ``--pert-col``)
    is identical regardless of which source this came from.
    """
    with h5py.File(path, "r") as f:
        obs = f["obs"]
        pert_col = next((k for k in PERT_KEY_CANDIDATES if k in obs), None)
        if pert_col is None:
            raise SystemExit(
                f"{path.name}: no perturbation column; tried {PERT_KEY_CANDIDATES}"
            )
        node = obs[pert_col]
        if isinstance(node, h5py.Group):  # categorical
            cats = np.array([c.decode() for c in node["categories"][:]])
            labels = cats[node["codes"][:]]
        else:  # plain string dataset
            labels = np.array(
                [v.decode() if isinstance(v, bytes) else str(v) for v in node[:]]
            )
            cats = np.unique(labels)
    control = next((c for c in CONTROL_LABEL_CANDIDATES if c in set(cats)), None)
    if control is None:
        raise SystemExit(
            f"{path.name}: no control label in obs[{pert_col!r}]; "
            f"tried {CONTROL_LABEL_CANDIDATES}"
        )
    labels = np.where(labels == control, CONTROL_LABEL, labels)
    return labels, np.sort(cats[cats != control])


def held_out_genes(cats: np.ndarray, test_size: float, seed: int) -> np.ndarray:
    """The datamodule's validation perturbations, recomputed rather than remembered.

    Calls the very function training calls. `gene_train_test_split` takes per-cell labels
    but reduces them with `np.unique` immediately, so handing it the unique categories is
    the same draw -- and using the real function means the two cannot drift.
    """
    _, test_index = gene_train_test_split(cats, test_size=test_size, seed=seed)
    return np.sort(np.unique(cats[test_index]))


# --------------------------------------------------------------------------------------
# the reference, content-addressed
# --------------------------------------------------------------------------------------


def reference_key(source: Path, held: np.ndarray, **knobs) -> str:
    """Hash of everything that changes the reference. Same scheme as `source_cache.py`."""
    import cell_eval2

    stat = source.stat()
    payload = {
        "source": {
            "path": str(source.resolve()),
            "mtime": int(stat.st_mtime),
            "size": stat.st_size,
        },
        "held_out": sorted(held.tolist()),
        "cell_eval2": cell_eval2.__version__,
        # ⚠️ The flag set belongs in the key. Several of these reach `config_digest`,
        # which `score --real-bundle` compares between the bundle manifest and the run --
        # so a bundle built under one flag set can never be found by a lookup under
        # another. Without this the failure is a peer mismatch an hour into a build
        # rather than a cache miss in a millisecond.
        "flags": COMMON,
        **knobs,
    }
    blob = json.dumps(payload, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:16], payload


def gather_rows(path: Path, rows: np.ndarray) -> csr_matrix:
    """Read `rows` out of a 43 GB CSR h5ad without materialising it.

    Row-at-a-time on purpose. The selection is ~10% of the file and scattered, so reading
    contiguous spans instead would pull roughly ten rows for every one wanted.
    """
    rows = np.asarray(rows)
    with h5py.File(path, "r") as f:
        X = f["X"]
        n_genes = int(X.attrs["shape"][1])
        indptr = X["indptr"][:]
        data_d, idx_d = X["data"], X["indices"]
        data, indices, out_indptr = [], [], np.zeros(len(rows) + 1, dtype=np.int64)
        for i, r in enumerate(rows):
            a, b = int(indptr[r]), int(indptr[r + 1])
            if b > a:
                data.append(data_d[a:b])
                indices.append(idx_d[a:b])
            out_indptr[i + 1] = out_indptr[i] + (b - a)
            if (i + 1) % 5000 == 0:
                console.log(f"    {i + 1:,}/{len(rows):,} rows")
    return csr_matrix(
        (
            np.concatenate(data) if data else np.zeros(0, np.float32),
            np.concatenate(indices).astype(np.int32)
            if indices
            else np.zeros(0, np.int32),
            out_indptr,
        ),
        shape=(len(rows), n_genes),
    )


def build_reference(
    source: Path, held: np.ndarray, args, refdir: Path, payload: dict, panel_genes=None
):
    """Slice the 2025 data into a 2026-shaped reference, then build its scale bundle.

    ``panel_genes`` restricts the reference to the model panel: cell-eval2 requires the
    prediction and reference to carry an identical gene set, and the prediction is on the
    panel, so a source measuring genes outside it (a PrimeFlow file) must be trimmed to the
    intersection here. ``None`` (or a source that already spans the panel, e.g. 2025) leaves
    every gene in place -- the previous behaviour.
    """
    refdir.mkdir(parents=True, exist_ok=True)
    # One builder at a time. Two processes sharing a key write into the same half-built
    # bundle directory, and `prep-real-bundle`'s own gates only catch that afterwards.
    lock = refdir / ".build.lock"
    fd = os.open(lock, os.O_CREAT | os.O_RDWR)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        console.log(f"[yellow]another build holds {lock}; waiting for it[/yellow]")
        fcntl.flock(fd, fcntl.LOCK_EX)
    real_path = refdir / "real.h5ad"

    if not real_path.exists():
        labels, _ = read_obs_perturbations(source)
        rng = np.random.default_rng(args.seed)

        keep_rows, keep_labels, dropped = [], [], []
        for gene in held:
            idx = np.flatnonzero(labels == gene)
            if len(idx) < args.min_cells:
                dropped.append((gene, len(idx)))
                continue
            take = min(args.cells_per_pert, len(idx))
            chosen = np.sort(rng.choice(idx, take, replace=False))
            keep_rows.append(chosen)
            keep_labels.extend([gene] * take)

        if dropped:
            console.log(
                f"[yellow]dropped {len(dropped)} perturbation(s) under --min-cells="
                f"{args.min_cells}: "
                + ", ".join(f"{g} ({n})" for g, n in dropped)
                + "[/yellow]"
            )

        ctrl_idx = np.flatnonzero(labels == CONTROL_LABEL)
        n_ctrl = min(args.n_controls, len(ctrl_idx))
        ctrl_rows = np.sort(rng.choice(ctrl_idx, n_ctrl, replace=False))
        keep_rows.append(ctrl_rows)
        keep_labels.extend([CONTROL_LABEL] * n_ctrl)

        rows = np.concatenate(keep_rows)
        n_kept = len(held) - len(dropped)
        console.log(
            f"reference: {len(rows) - n_ctrl:,} perturbed cells over {n_kept} "
            f"perturbations + {n_ctrl:,} controls"
        )
        console.log(f"slicing {source.name} ({source.stat().st_size / 1e9:.1f} GB)")
        X = gather_rows(source, rows)

        with h5py.File(source, "r") as f:
            # Read the var index by whatever key the file nominates (Replogle uses
            # `gene_symbol`, not `_index`); reuses the datamodule's resolver.
            genes = np.array(_read_var_names(f, None))
            is_log1p = "log1p" in f.get("uns", {})
        if is_log1p:
            # cell-eval2's vcc2026 preset pins integer counts, but the PrimeFlow files
            # store log1p(cp). Invert to counts (the source_cache `delog` step), so both the
            # scored reference AND predict()'s control input are raw counts, as for 2025.
            X = X.copy()
            X.data = np.rint(np.expm1(X.data)).astype(np.float32)
            X.eliminate_zeros()
            console.log(
                f"de-logged {source.name} (uns['log1p']) back to integer counts"
            )

        if panel_genes is not None:
            panel_pos = {g: i for i, g in enumerate(panel_genes)}
            keep = np.array([i for i, g in enumerate(genes) if g in panel_pos])
            if len(keep) < len(genes):
                X = X.tocsc()[:, keep].tocsr()
                genes = genes[keep]
                console.log(
                    f"restricted reference to {len(genes):,} genes in the model panel "
                    f"(of {len(panel_pos):,})"
                )

        obs = pd.DataFrame(
            {PERT_COL: keep_labels},
            index=[f"cell_{i}" for i in range(X.shape[0])],
        )
        assert_scoreable(X, obs, genes, side="reference")
        ad.AnnData(X=X, obs=obs, var=pd.DataFrame(index=genes)).write_h5ad(
            real_path, compression="gzip"
        )
        (refdir / "spec.json").write_text(json.dumps(payload, indent=2, sort_keys=True))
        console.log(f"wrote {real_path}")

    base_pred = refdir / "baseline_pred.h5ad"
    if not base_pred.exists():
        console.log("building the perturbation-blind baseline (the 0 end of the scale)")
        run_cell_eval2(
            [
                "baseline",
                "-ar",
                str(real_path),
                "-o",
                str(refdir / "baseline"),
                "--save-pred",
                str(base_pred),
            ]
        )

    bundle = refdir / "bundle"
    if not (bundle / "manifest.json").exists():
        console.log(
            "building the replicate anchor (the 1 end) -- 5 split-half metric runs, "
            "this is the slow part and it happens once"
        )
        if bundle.exists():
            shutil.rmtree(bundle)
        # ⚠️ --baseline takes the baseline PREDICTION h5ad, not the baseline outdir.
        run_cell_eval2(
            [
                "prep-real-bundle",
                "--real",
                str(real_path),
                "--baseline",
                str(base_pred),
                "-o",
                str(bundle),
            ]
        )
    return real_path, bundle


# --------------------------------------------------------------------------------------
# prediction
# --------------------------------------------------------------------------------------


def predict(model, conf, real: ad.AnnData, norm_cols, genes_panel, args, ckpt: Path):
    """Predict every reference perturbation from the reference's own control cells.

    Mirrors `predict_2026.py`: the same distinct-control draw, the same shared arithmetic
    from `src.utils.predict`, so the number this produces describes the pipeline that
    actually generates submissions.

    The reference genes may be a SUBSET of the model panel (a PrimeFlow source like HepG2
    measures ~9.6k of the 18,080). Control cells are scattered up to the panel (genes the
    source lacks -> 0, which is what training's reference-mean fill amounts to when the
    reference source IS that source), the model runs on the full panel, and both the
    prediction and the real controls are written on the reference genes that exist in the
    panel, so the pred and real files share one gene space for cell-eval2. When the
    reference already IS the full panel (2025), the mapping is the identity and this is
    byte-for-byte the previous behaviour.
    """
    ref_genes = [str(g) for g in real.var.index]
    panel_pos = {g: i for i, g in enumerate(genes_panel)}
    common = [g for g in ref_genes if g in panel_pos]
    ref_idx = np.array([i for i, g in enumerate(ref_genes) if g in panel_pos])
    panel_idx = np.array([panel_pos[g] for g in common])
    identity = (
        len(common) == len(genes_panel) == len(ref_genes)
        and ref_idx.size
        and bool(
            np.all(ref_idx == np.arange(len(ref_genes)))
            and np.all(panel_idx == ref_idx)
        )
    )
    if not identity:
        console.log(
            f"reference genes: {len(ref_genes):,}; {len(common):,} in the "
            f"{len(genes_panel):,}-gene model panel (scatter control up, subset output back)"
        )
    # The ko_vec exactly as the datamodule built it: read the SAME gene_embedding_path(s)
    # off the checkpoint's own data config and rebuild via the shared loader, so a multi-
    # block concat (quant+STRING+Poincare, proj_dim heads, ...) is reproduced bit-for-bit.
    # A hardcoded single embedding here silently fed a 256-d ko_vec to a 1024-d conditioner.
    emb_cfg = conf.data.datamodule.gene_embedding_path
    emb_paths = (
        [str(emb_cfg)] if isinstance(emb_cfg, str) else [str(p) for p in emb_cfg]
    )
    embeddings, emb_dim = load_gene_embeddings(emb_paths)
    console.log(
        f"perturbation embedding: {emb_dim}-d from {len(emb_paths)} source(s): "
        + ", ".join(Path(p).name for p in emb_paths)
    )
    # ko_id enumeration must match the training table -- the combined dict already iterates
    # in the first path's gene order, so enumerate() reproduces it.
    gene_ids = {g: i for i, g in enumerate(embeddings)}

    labels = real.obs[PERT_COL].to_numpy().astype(str)
    ctrl_rows = np.flatnonzero(labels == CONTROL_LABEL)
    ctrl_X = csr_matrix(real.X)[ctrl_rows]
    targets = [g for g in pd.unique(labels) if g != CONTROL_LABEL]

    missing = [t for t in targets if t not in embeddings]
    if missing:
        raise SystemExit(f"{len(missing)} targets have no embedding: {missing[:5]}")

    pert_key = getattr(model, "pert_input_key", "ko_vec")
    target_sum = float(conf.data.datamodule.target_sum)
    rng = np.random.default_rng(args.seed)
    blocks, out_labels = [], []

    # Context-conditioned model: rebuild the cell-context vector the datamodule feeds --
    # the source's pooled control profile over the norm axis. norm_cols IS the norm axis (an
    # all-source intersection, so this source measures every one of them). One fixed vector.
    context_t = None
    if (
        getattr(getattr(model.net, "ko_processor", None), "context_dim", None)
        is not None
    ):
        m = min(4000, ctrl_X.shape[0])
        sub = rng.choice(ctrl_X.shape[0], m, replace=False)
        cp = np.zeros((m, len(genes_panel)), dtype=np.float32)
        cp[:, panel_idx] = ctrl_X[sub].toarray()[:, ref_idx]
        ce, _ = to_model_space(cp, norm_cols, target_sum)
        cvec = ce[:, norm_cols].mean(axis=0).astype(np.float32)
        context_t = torch.from_numpy(cvec).to(args.device)
        console.log(
            f"context conditioning: {cvec.shape[0]}-d control profile from {m} controls"
        )

    for t_i, target in enumerate(targets):
        n = int((labels == target).sum())
        if n > len(ctrl_rows):
            raise SystemExit(
                f"{target} needs {n} distinct control cells but the reference holds only "
                f"{len(ctrl_rows)}. Raise --n-controls or lower --cells-per-pert."
            )
        picks = rng.choice(len(ctrl_rows), n, replace=False)
        ctrl_ref = ctrl_X[picks].toarray().astype(np.float32)
        # Scatter the reference's control cells up onto the model panel; genes the source
        # does not measure stay 0. (Identity when the reference already spans the panel.)
        ctrl = np.zeros((n, len(genes_panel)), dtype=np.float32)
        ctrl[:, panel_idx] = ctrl_ref[:, ref_idx]
        exp_vec, lib = to_model_space(ctrl, norm_cols, target_sum)

        with torch.inference_mode():
            batch = {
                "ko_vec": embeddings[target].expand(n, -1).to(args.device),
                "ko_id": torch.full(
                    (n,), gene_ids[target], dtype=torch.long, device=args.device
                ),
                "exp_vec": torch.from_numpy(exp_vec).to(args.device),
            }
            if hasattr(model.net, "sample"):
                out = model.net.sample(
                    batch["exp_vec"],
                    batch[pert_key],
                    num_steps=getattr(model, "num_sampling_steps", 4),
                    context=(None if context_t is None else context_t.expand(n, -1)),
                )
            else:
                out = model.net(batch)
            pred = out.float().cpu().numpy()

        # Model output is on the panel; subset back to the reference genes that live in it,
        # so pred and the real controls below share one gene space. (Identity for 2025.)
        counts = to_counts(pred, lib)
        blocks.append(finalize_counts(counts[:, panel_idx]))
        out_labels.extend([target] * n)
        if (t_i + 1) % 20 == 0:
            console.log(f"    {t_i + 1}/{len(targets)} perturbations")

    # The real control block, passed through untouched. Required, not decorative:
    # `io.validate_pair` demands the two files carry the SAME set of perturbation labels,
    # so the prediction must contain `non-targeting` cells even though `control_source:
    # real` substitutes the real controls for the DE pass anyway. Subset to the same
    # reference-in-panel genes as the prediction blocks.
    blocks.append(ctrl_X[:, ref_idx].astype(np.int32))
    out_labels.extend([CONTROL_LABEL] * len(ctrl_rows))

    X = vstack(blocks, format="csr")
    X.eliminate_zeros()
    obs = pd.DataFrame(
        {PERT_COL: out_labels}, index=[f"cell_{i}" for i in range(X.shape[0])]
    )
    genes = np.asarray(common)
    assert_scoreable(X, obs, genes, side="prediction")
    out = ckpt.parent / f"local_pred_{ckpt.stem}.h5ad"
    ad.AnnData(X=X, obs=obs, var=pd.DataFrame(index=genes)).write_h5ad(
        out, compression="gzip"
    )
    return out


def self_test_prediction(kind: str, real: ad.AnnData, refdir: Path) -> Path:
    """Build a model-free prediction whose score we already know, to calibrate the scale.

    A local score is only worth reading if the scale it sits on is sound, and the scale is
    two measured endpoints with our number somewhere between them. These three arms put
    known quantities through the identical path:

      perfect  the reference itself. A flawless prediction beats the split-half replicate
               that defines the 1 end, so `from_replicate` must come out ABOVE 1.
      blind    the perturbation-blind mean-response arm -- the very predictor the 0 end is
               built from. Must land at approximately 0. Rounded to integers because the
               strict counts path is the one our predictions take; that rounding is the
               only reason it is not exactly 0.
      control  every perturbation predicted as a fresh draw of control cells, i.e. "nothing
               changed". Must be at or below 0.

    Any model score has to sit inside `control <= score < perfect`, with `blind` marking
    where "learned nothing perturbation-specific" falls. A number outside that bracket
    indicts the harness, not the model.
    """
    labels = real.obs[PERT_COL].to_numpy().astype(str)
    genes = np.asarray(real.var.index)

    if kind == "perfect":
        X, out_labels = csr_matrix(real.X).astype(np.int32), list(labels)
    elif kind == "blind":
        arm = ad.read_h5ad(refdir / "baseline_pred.h5ad")
        X = finalize_counts(csr_matrix(arm.X).toarray().astype(np.float64))
        out_labels = list(arm.obs[PERT_COL].to_numpy().astype(str))
    elif kind == "control":
        ctrl = csr_matrix(real.X)[labels == CONTROL_LABEL]
        rng = np.random.default_rng(0)
        blocks, out_labels = [], []
        for target in [g for g in pd.unique(labels) if g != CONTROL_LABEL]:
            n = int((labels == target).sum())
            blocks.append(
                ctrl[rng.choice(ctrl.shape[0], n, replace=False)].astype(np.int32)
            )
            out_labels.extend([target] * n)
        blocks.append(ctrl.astype(np.int32))
        out_labels.extend([CONTROL_LABEL] * ctrl.shape[0])
        X = vstack(blocks, format="csr")
    else:
        raise SystemExit(f"unknown --self-test {kind!r}")

    X.eliminate_zeros()
    obs = pd.DataFrame(
        {PERT_COL: out_labels}, index=[f"cell_{i}" for i in range(X.shape[0])]
    )
    assert_scoreable(X, obs, genes, side=f"self-test {kind}")
    out = refdir / f"selftest_{kind}.h5ad"
    ad.AnnData(X=X, obs=obs, var=pd.DataFrame(index=genes)).write_h5ad(
        out, compression="gzip"
    )
    return out


def assert_scoreable(X, obs, genes, *, side: str) -> None:
    """Everything `cell-eval2` will check, checked here first.

    A violation costs seconds to report at this point and a full Wilcoxon pass to report
    from inside the scorer.
    """
    totals = np.asarray(X.sum(axis=1)).ravel()
    assert X.shape[1] == len(genes), f"{side}: {X.shape[1]} columns, {len(genes)} genes"
    assert obs.shape[0] == X.shape[0], f"{side}: obs/X row mismatch"
    assert np.all(X.data == np.rint(X.data)), (
        f"{side}: the vcc2026 preset pins allow_fractional_counts=false, so X must be "
        "whole numbers"
    )
    assert X.data.min() >= 0 and np.isfinite(X.data).all(), f"{side}: bad values in X"
    assert totals.max() <= 1_000_000, (
        f"{side}: a cell carries {totals.max():,.0f} counts, over the preset's "
        "max_counts_per_cell of 1e6"
    )
    assert CONTROL_LABEL in set(obs[PERT_COL]), f"{side}: no {CONTROL_LABEL!r} cells"


# --------------------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------------------


def run_cell_eval2(argv: list[str]) -> None:
    """Run one cell-eval2 subcommand, surfacing its output if it fails."""
    cmd = [CELL_EVAL2, argv[0], *argv[1:], *COMMON]
    console.log(f"[dim]$ {' '.join(cmd)}[/dim]")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        console.print(proc.stdout)
        console.print(f"[red]{proc.stderr}[/red]")
        if "SUBMISSION_PEERS" in proc.stderr or "bundle=" in proc.stderr:
            console.print(
                "[yellow]The run and the bundle disagree on a scoring peer. The bundle "
                "was built under different settings -- rebuild it with --refresh.[/yellow]"
            )
        raise SystemExit(f"cell-eval2 {argv[0]} failed")


def score(pred: Path, real: Path, bundle: Path, outdir: Path) -> pl.DataFrame:
    """Score `pred` against `real` with cell-eval2 and return the per-metric table."""
    outdir.mkdir(parents=True, exist_ok=True)
    # The real side's DE table is identical for every checkpoint and recomputing it is
    # roughly half the cost of a scoring run. `cache_real`/`cache_pred` are in
    # DIGEST_EXEMPT_FIELDS, so this changes no scoring identity. Passed OUTSIDE COMMON
    # deliberately: COMMON feeds the reference cache key, and adding to it would orphan
    # every bundle already built.
    run_cell_eval2(
        [
            "run",
            "-ap",
            str(pred),
            "-ar",
            str(real),
            "-o",
            str(outdir),
            "--write-degenes",
            "--cache-real",
            str(real.parent / "ce2-real"),
            "--cache-pred",
            str(outdir / "ce2-pred"),
        ]
    )
    scored = outdir / "scored.csv"
    subprocess.run(
        [
            CELL_EVAL2,
            "score",
            "--user-agg",
            str(outdir / "agg_results.csv"),
            "--real-bundle",
            str(bundle),
            "-o",
            str(scored),
        ],
        check=True,
    )
    return pl.read_csv(scored)


# --------------------------------------------------------------------------------------
# diagnostics -- the mechanism, which the six official metrics do not report
# --------------------------------------------------------------------------------------


def log_cp10k(X: csr_matrix) -> np.ndarray:
    """Dense log1p(CP10K) of a raw-count matrix."""
    dense = X.toarray().astype(np.float64)
    lib = dense.sum(axis=1, keepdims=True)
    lib[lib == 0] = 1.0
    return np.log1p(dense / lib * 1e4)


def diagnostics(pred_path: Path, real_path: Path, outdir: Path) -> dict:
    """DE cosine against the blind constant, the overshoot ratio, and DE breadth.

    These are what explained the last three submissions. `pds` and friends say how well we
    scored; these say whether the model has learned anything perturbation-specific at all.
    """
    real, pred = ad.read_h5ad(real_path), ad.read_h5ad(pred_path)
    rl = real.obs[PERT_COL].to_numpy().astype(str)
    pl_ = pred.obs[PERT_COL].to_numpy().astype(str)
    targets = sorted(set(rl) - {CONTROL_LABEL})

    real_ctrl = log_cp10k(csr_matrix(real.X)[rl == CONTROL_LABEL]).mean(axis=0)
    pred_ctrl = log_cp10k(csr_matrix(pred.X)[pl_ == CONTROL_LABEL]).mean(axis=0)

    d_true, d_pred = [], []
    for t in targets:
        d_true.append(log_cp10k(csr_matrix(real.X)[rl == t]).mean(axis=0) - real_ctrl)
        d_pred.append(log_cp10k(csr_matrix(pred.X)[pl_ == t]).mean(axis=0) - pred_ctrl)
    d_true, d_pred = np.array(d_true), np.array(d_pred)

    def cos(a, b):
        na, nb = np.linalg.norm(a, axis=-1), np.linalg.norm(b, axis=-1)
        return float(np.mean(np.sum(a * b, -1) / np.clip(na * nb, 1e-12, None)))

    # The bar the scoring formula sets at zero: one profile, reused for every target.
    blind = np.broadcast_to(d_true.mean(axis=0), d_true.shape)

    out = {
        "de_cosine": cos(d_pred, d_true),
        "de_cosine_blind": cos(blind, d_true),
        "overshoot": float(np.abs(d_pred).mean() / np.abs(d_true).mean()),
    }
    for side in ("pred", "real"):
        f = outdir / f"de_{side}.parquet"
        if f.exists():
            de = pl.read_parquet(f)
            col = "p_adj" if "p_adj" in de.columns else de.columns[-1]
            sig = de.filter(pl.col(col) < 0.05).group_by("target").len()
            out[f"n_sig_{side}"] = float(sig["len"].median()) if sig.height else 0.0
    # Machine-readable copy of the decisive numbers, for sweep orchestration.
    (outdir / "mechanism.json").write_text(json.dumps(out, sort_keys=True))
    return out


# --------------------------------------------------------------------------------------


def report(rows: list[tuple[str, pl.DataFrame, dict]]) -> None:
    """Print one table row per checkpoint with its overall and per-metric scores."""
    table = Table(title="local score -- 2025 held-out perturbations, vcc2026 metrics")
    table.add_column("checkpoint", style="bold")
    for name in ("overall", *[SHORT[m] for m in SCORED]):
        table.add_column(name, justify="right")
    for label, scored, _ in rows:
        col = (
            "from_replicate" if "from_replicate" in scored.columns else "from_baseline"
        )
        by = dict(zip(scored["metric"].to_list(), scored[col].to_list()))
        cells = [f"{by.get('avg_score', float('nan')):+.5f}"]
        cells += ["n/a" if by.get(m) is None else f"{by[m]:+.4f}" for m in SCORED]
        table.add_row(label, *cells)
    console.print(table)

    diag = Table(title="mechanism -- has it learned anything perturbation-specific?")
    diag.add_column("checkpoint", style="bold")
    for name in ("DE cosine", "blind bar", "overshoot", "sig genes", "real sig"):
        diag.add_column(name, justify="right")
    for label, _, d in rows:
        diag.add_row(
            label,
            f"{d['de_cosine']:.3f}",
            f"{d['de_cosine_blind']:.3f}",
            f"{d['overshoot']:.3f}",
            f"{d.get('n_sig_pred', float('nan')):.0f}",
            f"{d.get('n_sig_real', float('nan')):.0f}",
        )
    console.print(diag)
    console.print(
        "[dim]DE cosine at or below the blind bar means the model is not beating "
        "'predict the same response for every perturbation' -- which is exactly what the "
        "scoring formula puts at zero. Overshoot is |predicted change| / |true change|; "
        "1.0 is calibrated.[/dim]"
    )


def main() -> None:
    """Score one or more checkpoints on the 2025 held-out perturbations."""
    load_dotenv()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", action="append", default=[], help="repeatable")
    p.add_argument("--model", default=None, help="default: read from the wandb config")
    p.add_argument("--data", default=None, help="default: read from the wandb config")
    p.add_argument(
        "--source",
        default=None,
        help="reference .h5ad to score against (its held-out perturbations). Default: the "
        "2025 data. Pass a PrimeFlow file (e.g. nadig24_hepg2_preprocessed.h5ad) for a "
        "CROSS-CONTEXT score -- the 2025-referenced default has not predicted the board.",
    )
    p.add_argument("--cells-per-pert", type=int, default=400, help="the 2026 protocol")
    p.add_argument("--n-controls", type=int, default=18_400, help="the 2026 protocol")
    p.add_argument("--min-cells", type=int, default=100, help="drop below this")
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda:1")
    p.add_argument(
        "--limit", type=int, default=None, help="smoke path: N perturbations"
    )
    p.add_argument(
        "--refresh", action="store_true", help="rebuild the reference bundle"
    )
    p.add_argument(
        "--self-test",
        choices=["perfect", "blind", "control"],
        default=None,
        help="score a model-free predictor of known quality to calibrate the scale",
    )
    args = p.parse_args()

    # ⚠️ BEFORE the reference build, not after. Building a reference takes an hour, and
    # a usage error that surfaces on the far side of it is worse than no check at all --
    # it also races whatever legitimate build is already running.
    if not args.ckpt and not args.self_test:
        raise SystemExit("pass --ckpt, or --self-test to calibrate the scale")

    if Path.cwd().name != "vcc":
        console.print(
            "[yellow]hydra resolves config/ from the cwd; run from the repo root[/yellow]"
        )

    data_dir = Path(os.environ["DATA_DIR"])
    source = Path(args.source) if args.source else data_dir / "2025/adata_2025_all.h5ad"
    if not source.exists():
        raise SystemExit(f"--source not found: {source}")
    console.log(f"reference source: {source.name}")

    _, cats = read_obs_perturbations(source)
    held = held_out_genes(cats, args.test_size, args.seed)
    console.log(f"held out {len(held)} of {len(cats)} perturbations (seed {args.seed})")

    # Drop held-out perturbations the embedding table cannot address: predict() needs a
    # ko_vec for every one, and cell-eval2 needs the reference and prediction to carry the
    # SAME perturbation set -- so the filter has to happen here, before the reference is
    # built, not inside predict(). (No-op when every held-out gene has an embedding, as for
    # 2025.) The embedding paths come from the run's own data config.
    panel_genes = None
    if args.ckpt:
        probe_data = args.data or discover_configs(Path(args.ckpt[0]))[1]
        with initialize_config_dir(
            config_dir=str(Path.cwd() / "config"), version_base=None
        ):
            dconf = compose("train.yaml", overrides=[f"data={probe_data}"])
        emb_cfg = dconf.data.datamodule.gene_embedding_path
        emb_paths = (
            [str(emb_cfg)] if isinstance(emb_cfg, str) else [str(p) for p in emb_cfg]
        )
        emb_genes = set(load_gene_embeddings(emb_paths)[0])
        before = len(held)
        held = np.array([g for g in held if g in emb_genes])
        if len(held) < before:
            console.log(
                f"dropped {before - len(held)} held-out perturbation(s) with no embedding"
            )
        # The model panel, so the reference is built on the SAME genes the prediction will
        # carry (cell-eval2 requires an exact gene match, it does not intersect).
        panel_genes = [
            str(g)
            for g in pl.read_csv(
                dconf.data.datamodule.gene_list_path, has_header=False
            ).to_series(0)
        ]

    if args.limit:
        held = held[: args.limit]
        console.log(
            f"[yellow]--limit {args.limit}: smoke path, not a real score[/yellow]"
        )

    key, payload = reference_key(
        source,
        held,
        cells_per_pert=args.cells_per_pert,
        n_controls=args.n_controls,
        min_cells=args.min_cells,
        seed=args.seed,
        test_size=args.test_size,
    )
    refdir = data_dir / "2025/_localscore" / f"ref__{key}"
    if args.refresh and refdir.exists():
        shutil.rmtree(refdir)
    console.log(f"reference {refdir}")

    real_path, bundle = build_reference(
        source, held, args, refdir, payload, panel_genes
    )
    real = ad.read_h5ad(real_path)

    if args.self_test:
        pred_path = self_test_prediction(args.self_test, real, refdir)
        outdir = refdir / f"selftest_score_{args.self_test}"
        scored = score(pred_path, real_path, bundle, outdir)
        report(
            [
                (
                    f"self-test:{args.self_test}",
                    scored,
                    diagnostics(pred_path, real_path, outdir),
                )
            ]
        )
        expected = {
            "perfect": "above +1",
            "blind": "about 0",
            "control": "at or below 0",
        }
        console.print(
            f"[bold]--self-test {args.self_test}[/bold] should score {expected[args.self_test]}. "
            "If it does not, the reference is wrong and every other number here is too."
        )
        return

    rows = []
    for raw in args.ckpt:
        ckpt = Path(raw)
        model_cfg, data_cfg = (args.model, args.data)
        if model_cfg is None or data_cfg is None:
            discovered = discover_configs(ckpt)
            model_cfg = model_cfg or discovered[0]
            data_cfg = data_cfg or discovered[1]
        console.log(f"[bold]{ckpt.name}[/bold]: model={model_cfg} data={data_cfg}")

        model, conf = load_model(ckpt, model_cfg, data_cfg, args.device)
        # The model's INPUT/OUTPUT panel (18,080), which the reference genes may be a
        # subset of (a PrimeFlow source measures fewer). norm_cols and the control input
        # live on this panel; predict() maps the reference genes onto it and back.
        # gene_names.csv has no header -- the default would eat SAMD11 as a column name.
        genes_panel = [
            str(g)
            for g in pl.read_csv(
                conf.data.datamodule.gene_list_path, has_header=False
            ).to_series(0)
        ]
        norm_cols = resolve_norm_cols(ckpt, genes_panel, conf)
        pred_path = predict(model, conf, real, norm_cols, genes_panel, args, ckpt)
        del model
        torch.cuda.empty_cache()

        outdir = ckpt.parent / f"local_score_{ckpt.stem}"
        scored = score(pred_path, real_path, bundle, outdir)
        rows.append(
            (
                ckpt.parent.name + "/" + ckpt.stem,
                scored,
                diagnostics(pred_path, real_path, outdir),
            )
        )

    report(rows)


if __name__ == "__main__":
    sys.exit(main())
