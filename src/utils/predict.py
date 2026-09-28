"""The control cell -> model input -> raw counts arithmetic, in one place.

Extracted from `scripts/predict_2026.py` so the leaderboard submission path and
`scripts/score_local.py` cannot drift apart. If the local scorer's arithmetic differed
from the submission's by so much as the choice of depth axis, it would be measuring a
model we never actually submit -- and the difference would be invisible, because both
sides would look internally consistent.

The two axes
------------
`to_model_space` computes two different totals from the same control cell and they are
not interchangeable:

  `depth`  the cell's depth over the axis the model was TRAINED to normalise on. That is
           the full panel for a single-source run and the 6,691-gene shared subset for a
           multi-source one, read from `norm_axis.csv` beside the checkpoint. It scales
           the model's input.
  `lib`    the cell's depth over the whole model panel. It scales the prediction back to
           counts, so the predicted cell carries the library size of the control it came
           from.

They coincide only when the norm axis is the full panel.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix


def load_gene_embeddings(
    paths: str | list[str],
) -> tuple[dict[str, "object"], int]:
    """Build the gene -> vector table, mirroring ``AnnDataVCCDataModule._load_embeddings``.

    The single source of truth for reconstructing a checkpoint's ``ko_vec`` outside the
    datamodule -- shared by ``predict_2026.py`` (submissions) and ``score_local.py``
    (offline scoring), so both reproduce it bit-for-bit and can never drift.

    A single path is used as-is (no normalisation). Several are per-row L2-normalised per
    block and concatenated, aligned on the FIRST path's gene order with a later block's
    missing genes zero-filled. Returns ``(embeddings, dim)`` where ``embeddings`` iterates
    in the first path's gene order -- so ``enumerate(embeddings)`` gives the same ``ko_id``
    the training table used.
    """
    import polars as pl
    import polars.selectors as cs
    import torch

    if isinstance(paths, str):
        paths = [paths]

    if len(paths) == 1:
        df = pl.read_parquet(paths[0])
        vecs = df.select(cs.numeric()).to_torch().float()
        emb = {str(g): vecs[i] for i, g in enumerate(df["gene_name"].to_numpy())}
        return emb, vecs.shape[1]

    blocks: list[tuple[dict[str, "object"], int]] = []
    base: list[str] | None = None
    for p in paths:
        df = pl.read_parquet(p)
        names = [str(g) for g in df["gene_name"].to_numpy()]
        vecs = torch.nn.functional.normalize(
            df.select(cs.numeric()).to_torch().float(), dim=1
        )
        blocks.append(({g: vecs[i] for i, g in enumerate(names)}, vecs.shape[1]))
        if base is None:
            base = names
    assert base is not None
    dim = sum(d for _, d in blocks)
    combined = {
        g: torch.cat([row[g] if g in row else torch.zeros(d) for row, d in blocks])
        for g in base
    }
    return combined, dim


def to_model_space(
    ctrl_counts: np.ndarray, norm_cols: np.ndarray, target_sum: float
) -> tuple[np.ndarray, np.ndarray]:
    """Raw control counts -> the log1p(CP-`target_sum`) space the model consumes.

    Args:
        ctrl_counts: `(n_cells, n_model_genes)` dense raw counts.
        norm_cols: column indices of the training depth axis.
        target_sum: the datamodule's `target_sum` (1e4 for every current config).

    Returns:
        `(exp_vec, lib)` -- the model input, and the full-panel library size to invert
        against. `lib` keeps its `(n_cells, 1)` shape so it broadcasts in `to_counts`.
    """
    lib = ctrl_counts.sum(axis=1, keepdims=True)
    lib[lib == 0] = 1.0
    depth = ctrl_counts[:, norm_cols].sum(axis=1, keepdims=True)
    depth[depth == 0] = 1.0
    return np.log1p(ctrl_counts / depth * target_sum), lib


def to_counts(pred: np.ndarray, lib: np.ndarray) -> np.ndarray:
    """Model output -> dense float64 counts carrying library size `lib`.

    Deliberately does NOT round: `predict_2026.py` scatters these into the wider 2026
    gene panel and only then rounds, so rounding here would quantise twice. Pass the
    result to `finalize_counts` once it is on its final gene axis.
    """
    counts = np.expm1(pred, dtype=np.float64)
    mass = counts.sum(axis=1, keepdims=True)
    mass[mass == 0] = 1.0
    counts *= lib / mass
    return counts


def finalize_counts(dense: np.ndarray) -> csr_matrix:
    """Dense float counts -> the int32 CSR the submission spec and `cell-eval2` require.

    `cell-eval2`'s `vcc2026` preset pins `input_type: counts` with
    `allow_fractional_counts: false`, so a non-integer matrix is rejected outright.
    Stored zeros are eliminated because they count against the submission's entry cap.
    """
    np.clip(dense, 0, None, out=dense)
    block = csr_matrix(np.rint(dense).astype(np.int32))
    block.eliminate_zeros()
    return block
