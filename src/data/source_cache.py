"""Read heterogeneous perturbation ``.h5ad`` files into one shared gene panel.

The datamodule used to take files that already agreed on everything. Training on
public Perturb-seq data breaks that: the three files currently in use disagree on
the gene panel (18,080 / 8,247 / 8,880 genes), on the ``obs`` column naming the
perturbation, on the label marking controls, and even on what the values *mean* --
the 2025 file holds raw integer counts while both PrimeFlow files are already
``log1p(cp9715)``.

This module absorbs all of that so ``data_path`` can stay a plain list of files.
Each source is read once, reconciled, and cached; afterwards setup is seconds.

Two design points are load-bearing.

**Only the means survive.** In ``target_mode="perturbation_mean"`` the dataset never
reads an individual perturbed cell -- ``__getitem__`` returns ``pert_means[code]``.
So the perturbed rows exist only to *compute* the means and can be dropped straight
afterwards. That is what makes a 71 GB file usable: 1.99 M K562 cells collapse to a
9,210 x 18,080 mean table plus 75,328 control rows, ~3 GB instead of ~75 GB.

**Depth is measured on the shared axis.** Normalising each source over its own panel
would make a gene read ~1.4x higher in K562 than in 2025, because K562's 7,456 genes
have to carry the whole ``target_sum`` between them. Every source is instead scaled
so that the genes *common to all of them* sum to ``target_sum``. Genes outside that
axis keep their true relative abundance under the same per-cell factor. The axis is
derived from the source list by :func:`resolve_norm_axis` and never configured by
hand, so adding a dataset cannot leave a stale one behind.
"""

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import h5py
import numpy as np
from rich.console import Console
from scipy.sparse import csr_matrix, vstack

from src.utils.data import build_gene_maps

console = Console()

#: Tried in order when ``SourceSpec.pert_key`` is not given. ``target_gene`` is the
#: 2025 naming; ``perturbation`` is what the PrimeFlow files use.
PERT_KEY_CANDIDATES = ("target_gene", "perturbation", "condition", "gene")

#: Tried in order when ``SourceSpec.control_label`` is not given, against the values
#: actually present in the resolved perturbation column.
CONTROL_LABEL_CANDIDATES = ("non-targeting", "control", "NT", "ctrl", "non_targeting")

#: Rows per streaming chunk. At K562's ~3,160 nnz/row this is ~79 M nonzeros, so the
#: transient buffers stay near 1 GB.
CHUNK_ROWS = 25_000

CACHE_VERSION = 1


@dataclass
class SourceSpec:
    """One ``.h5ad`` to train on, plus overrides for anything auto-detection gets wrong.

    Every field except ``path`` is optional: the common case in config is a bare path
    string, which Hydra hands over as ``SourceSpec(path=...)``.

    Attributes
    ----------
    path : Path
        The ``.h5ad``.
    name : str | None
        Identifier used in logs, in ``val_source``, and as the cache filename stem.
        Defaults to the file stem.
    pert_key : str | None
        ``obs`` column naming the perturbation. Auto: :data:`PERT_KEY_CANDIDATES`.
    control_label : str | None
        Value of that column marking controls. Auto: :data:`CONTROL_LABEL_CANDIDATES`.
    var_key : str | None
        ``var`` column holding gene symbols. Auto: the ``var`` index.
    delog : bool | None
        Whether ``X`` needs ``expm1`` before it can be treated as counts. Auto: true
        iff the file carries ``uns/log1p``, which is how both PrimeFlow files record
        that they were already normalised.
    """

    path: Path
    name: str | None = None
    pert_key: str | None = None
    control_label: str | None = None
    var_key: str | None = None
    delog: bool | None = None

    #: Filled by :meth:`resolve`; not part of the config surface.
    resolved: dict = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        """Normalise ``path`` and default ``name`` off it."""
        self.path = Path(self.path)
        if self.name is None:
            self.name = self.path.stem

    def resolve(self) -> dict:
        """Fill in every auto-detected field by peeking at the file.

        Reads ``obs`` and ``var`` metadata only -- never ``X`` -- so this is cheap
        enough to call on a 71 GB file during config validation.

        Returns:
            dict: ``{"genes", "pert_key", "control_label", "delog"}``.
        """
        if self.resolved:
            return self.resolved

        with h5py.File(self.path, "r") as handle:
            genes = _read_var_names(handle, self.var_key)

            pert_key = self.pert_key
            if pert_key is None:
                pert_key = next(
                    (k for k in PERT_KEY_CANDIDATES if k in handle["obs"]), None
                )
                if pert_key is None:
                    raise KeyError(
                        f"{self.path.name}: no perturbation column found. Tried "
                        f"{PERT_KEY_CANDIDATES}; obs has "
                        f"{sorted(handle['obs'].keys())}. Set `pert_key` explicitly."
                    )

            labels = _read_obs_categories(handle, pert_key)

            control_label = self.control_label
            if control_label is None:
                control_label = next(
                    (c for c in CONTROL_LABEL_CANDIDATES if c in labels), None
                )
                if control_label is None:
                    raise KeyError(
                        f"{self.path.name}: no control label found in obs[{pert_key!r}]. "
                        f"Tried {CONTROL_LABEL_CANDIDATES}. Set `control_label`."
                    )

            delog = self.delog
            if delog is None:
                delog = "log1p" in handle.get("uns", {})

        self.resolved = {
            "genes": genes,
            "pert_key": pert_key,
            "control_label": control_label,
            "delog": bool(delog),
        }
        return self.resolved

    def fingerprint(self) -> dict:
        """Identity of the file *and* how it is being read, for the cache key."""
        stat = self.path.stat()
        info = self.resolve()
        return {
            "path": str(self.path.resolve()),
            "mtime": int(stat.st_mtime),
            "size": stat.st_size,
            "pert_key": info["pert_key"],
            "control_label": info["control_label"],
            "delog": info["delog"],
        }


def _decode(values) -> np.ndarray:
    """Bytes-or-str array to a numpy array of ``str``."""
    return np.array(
        [v.decode() if isinstance(v, bytes) else str(v) for v in values], dtype=object
    )


def _read_var_names(handle: h5py.File, var_key: str | None) -> list[str]:
    """Gene symbols in file order.

    ``var_key`` defaults to whatever the file nominates as its index -- which is
    ``gene_symbol`` in the Replogle file and ``_index`` in the other two.
    """
    key = var_key or handle["var"].attrs.get("_index", "_index")
    key = key.decode() if isinstance(key, bytes) else str(key)
    if key not in handle["var"]:
        raise KeyError(
            f"var column {key!r} not in {sorted(handle['var'].keys())}; set `var_key`"
        )
    node = handle["var"][key]
    if isinstance(node, h5py.Group):  # categorical
        return _decode(node["categories"][:][node["codes"][:]]).tolist()
    return _decode(node[:]).tolist()


def _read_obs_categories(handle: h5py.File, key: str) -> list[str]:
    """Distinct values of an ``obs`` column, without materialising the column."""
    node = handle["obs"][key]
    if isinstance(node, h5py.Group):
        return _decode(node["categories"][:]).tolist()
    return np.unique(_decode(node[:])).tolist()


def _read_obs_column(handle: h5py.File, key: str) -> np.ndarray:
    """One ``obs`` column as a numpy array of ``str``, for every row."""
    node = handle["obs"][key]
    if isinstance(node, h5py.Group):
        cats = _decode(node["categories"][:])
        return cats[node["codes"][:]]
    return _decode(node[:])


def resolve_norm_axis(
    model_genes: list[str], specs: list[SourceSpec]
) -> tuple[list[str], str]:
    """Genes common to the model panel and to *every* source, in model-panel order.

    This is the axis per-cell depth is measured on, so it has to be identical for
    every source or the same gene reads at a different scale in each. Deriving it
    from the source list rather than taking it from config is deliberate: adding a
    dataset shrinks the axis, and a hand-maintained file would silently not shrink.

    Ordering by model-panel position makes the result a function of the *set* of
    panels, not of the order the files happen to be listed in, so re-ordering
    ``data_path`` does not invalidate the caches.

    Args:
        model_genes (list[str]): The model's output panel.
        specs (list[SourceSpec]): Every source being trained on.

    Returns:
        tuple: ``(axis, digest)`` -- the gene names and a short content hash of them.
    """
    axis = set(model_genes)
    for spec in specs:
        axis &= set(spec.resolve()["genes"])
    if not axis:
        raise ValueError(
            "The listed sources share no genes with the model panel; check "
            "`gene_list_path` and the `var_key` of each source."
        )

    ordered = [gene for gene in model_genes if gene in axis]
    digest = hashlib.sha256("\n".join(ordered).encode()).hexdigest()[:12]
    return ordered, digest


def cache_path(spec: SourceSpec, cache_dir: Path, key: str) -> Path:
    """Where this source's reconciled form lives, given the cache key."""
    return Path(cache_dir) / f"{spec.name}__{key}.npz"


def cache_key(
    spec: SourceSpec, model_genes: list[str], norm_digest: str, target_sum: float
) -> str:
    """Hash of everything that changes what the cache should contain.

    The norm-axis digest is in here, so adding a fourth dataset shrinks the axis and
    invalidates every source's cache at once. That rebuild is the point: a cache built
    against a different axis is not merely stale, it is in a different unit.
    """
    payload = {
        "version": CACHE_VERSION,
        "source": spec.fingerprint(),
        "model_genes": hashlib.sha256("\n".join(model_genes).encode()).hexdigest()[:16],
        "norm_axis": norm_digest,
        "target_sum": float(target_sum),
    }
    blob = json.dumps(payload, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def _read_obs_codes(handle: h5py.File, key: str) -> tuple[np.ndarray, np.ndarray]:
    """An ``obs`` column as ``(codes, categories)``.

    Working in codes rather than strings matters here: deciding *per category* which
    perturbations are controls and which lack an embedding is ~10^4 lookups, while
    the same decision per cell would be ~10^6 string comparisons.
    """
    node = handle["obs"][key]
    if isinstance(node, h5py.Group):
        return node["codes"][:].astype(np.int64), _decode(node["categories"][:])
    values = _decode(node[:])
    categories, codes = np.unique(values, return_inverse=True)
    return codes.astype(np.int64), categories


def build_cache(
    spec: SourceSpec,
    model_genes: list[str],
    norm_genes: list[str],
    embedding_genes: set[str],
    cache_dir: Path,
    target_sum: float,
    log1p: bool = True,
    force: bool = False,
) -> Path:
    """Reconcile one source onto the model panel and write it to ``cache_dir``.

    Streams ``X`` in :data:`CHUNK_ROWS` blocks with h5py; ``ad.read_h5ad`` is never
    called, because the Replogle file would want ~75 GB of CSR to hold rows that are
    about to be averaged away. What is written is the *residue* of that averaging:

    ``control_X``
        Control cells only, scattered into the ``model_genes`` panel and normalised.
    ``pert_means``
        ``(n_perturbations, len(model_genes))`` mean of ``log1p(cp<target_sum>)``.
    ``observed``
        Boolean over the model panel; false where this source has no such gene, and
        therefore where the loss must be masked.

    Args:
        spec (SourceSpec): The source, with auto-detection already resolvable.
        model_genes (list[str]): Output panel; the column order of everything written.
        norm_genes (list[str]): Shared axis per-cell depth is summed over.
        embedding_genes (set[str]): Perturbations lacking one of these are dropped
            here, rather than raising from inside a dataloader worker much later.
        cache_dir (Path): Directory for the ``.npz``.
        target_sum (float): Counts-per-cell over ``norm_genes``.
        log1p (bool): Apply ``log1p`` after depth normalisation.
        force (bool): Rebuild even if a cache with this key exists.

    Returns:
        Path: The written (or reused) ``.npz``.
    """
    info = spec.resolve()
    # Same formula as `resolve_norm_axis`, so a cache built by one path is found
    # by the other.
    digest = hashlib.sha256("\n".join(norm_genes).encode()).hexdigest()[:12]
    key = cache_key(spec, model_genes, digest, target_sum)
    out = cache_path(spec, cache_dir, key)
    if out.exists() and not force:
        console.log(f"[green]cache hit[/green] {spec.name}: {out.name}")
        return out

    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    src_genes = info["genes"]
    n_model = len(model_genes)

    # Model panel -> source panel, by name. `model_cols[k]` is where compact column k
    # lands in the output; `src_cols[k]` is where it was read from.
    model_to_src, present, _ = build_gene_maps(src_genes, model_genes)
    model_cols = np.flatnonzero(present)
    src_cols = model_to_src[present]
    n_compact = len(model_cols)

    compact_of_src = np.full(len(src_genes), -1, dtype=np.int64)
    compact_of_src[src_cols] = np.arange(n_compact)

    is_norm_model = np.zeros(n_model, dtype=bool)
    pos = {gene: i for i, gene in enumerate(model_genes)}
    is_norm_model[[pos[gene] for gene in norm_genes]] = True
    is_norm_compact = is_norm_model[model_cols]

    console.log(
        f"[bold]{spec.name}[/bold]: {n_compact:,}/{n_model:,} model genes present "
        f"({n_compact / n_model:.1%}), depth over {len(norm_genes):,} shared genes"
    )

    with h5py.File(spec.path, "r") as handle:
        codes, categories = _read_obs_codes(handle, info["pert_key"])
        n_cells = len(codes)

        # Decide once per category, then map. Controls and embedding-less
        # perturbations both become -1 in `pert_of_cat`; controls are tracked apart.
        control_cat = np.flatnonzero(categories == info["control_label"])
        if not len(control_cat):
            raise ValueError(
                f"{spec.name}: control label {info['control_label']!r} not present"
            )

        keep = np.array(
            [
                cat != info["control_label"] and cat in embedding_genes
                for cat in categories
            ]
        )
        dropped = int((~keep).sum()) - 1  # minus the control category itself
        pert_names = categories[keep]
        pert_of_cat = np.full(len(categories), -1, dtype=np.int64)
        pert_of_cat[keep] = np.arange(len(pert_names))

        pert_of_row = pert_of_cat[codes]
        is_control = codes == control_cat[0]
        n_control = int(is_control.sum())
        console.log(
            f"  {n_cells:,} cells | {len(pert_names):,} perturbations kept, "
            f"{dropped:,} dropped for want of an embedding | {n_control:,} controls"
        )

        group = handle["X"]
        if not isinstance(group, h5py.Group):
            raise TypeError(f"{spec.name}: X is dense; only CSR is supported")
        indptr = group["indptr"][:].astype(np.int64)

        sums = np.zeros(len(pert_names) * n_compact, dtype=np.float64)
        counts = np.bincount(
            pert_of_row[pert_of_row >= 0], minlength=len(pert_names)
        ).astype(np.float64)
        control_blocks: list[csr_matrix] = []
        control_depth: list[np.ndarray] = []

        for start in range(0, n_cells, CHUNK_ROWS):
            stop = min(start + CHUNK_ROWS, n_cells)
            lo, hi = indptr[start], indptr[stop]
            rows = np.repeat(np.arange(stop - start), np.diff(indptr[start : stop + 1]))
            compact = compact_of_src[group["indices"][lo:hi]]
            keep_nnz = compact >= 0

            rows = rows[keep_nnz]
            compact = compact[keep_nnz]
            vals = group["data"][lo:hi][keep_nnz].astype(np.float64)
            # Both PrimeFlow files store log1p(cp9715). expm1 returns something
            # proportional to counts, which is all depth normalisation needs -- the
            # original library size never enters.
            if info["delog"]:
                np.expm1(vals, out=vals)

            on_axis = is_norm_compact[compact]
            depth = np.bincount(
                rows[on_axis], weights=vals[on_axis], minlength=stop - start
            )
            depth[depth == 0] = 1.0
            vals *= target_sum / depth[rows]
            if log1p:
                np.log1p(vals, out=vals)

            chunk_pert = pert_of_row[start:stop]
            keep_row = chunk_pert[rows] >= 0
            if keep_row.any():
                flat = chunk_pert[rows[keep_row]] * n_compact + compact[keep_row]
                sums += np.bincount(flat, weights=vals[keep_row], minlength=sums.size)

            chunk_ctrl = is_control[start:stop]
            if chunk_ctrl.any():
                local = np.cumsum(chunk_ctrl) - 1
                sel = chunk_ctrl[rows]
                control_blocks.append(
                    csr_matrix(
                        (
                            vals[sel].astype(np.float32),
                            (local[rows[sel]], model_cols[compact[sel]]),
                        ),
                        shape=(int(chunk_ctrl.sum()), n_model),
                    )
                )
                control_depth.append(depth[chunk_ctrl])

            if (start // CHUNK_ROWS) % 10 == 0:
                console.log(f"  {stop:,}/{n_cells:,} cells")

    control_X = (
        control_blocks[0] if len(control_blocks) == 1 else vstack(control_blocks, "csr")
    )
    del control_blocks

    means = np.zeros((len(pert_names), n_model), dtype=np.float32)
    means[:, model_cols] = sums.reshape(len(pert_names), n_compact) / counts[:, None]
    del sums

    np.savez(
        out,
        control_data=control_X.data,
        control_indices=control_X.indices,
        control_indptr=control_X.indptr,
        control_shape=np.array(control_X.shape),
        control_depth=np.concatenate(control_depth),
        pert_means=means,
        pert_names=pert_names.astype(str),
        pert_counts=counts.astype(np.int64),
        observed=present,
    )
    console.log(
        f"[green]wrote[/green] {out.name} ({out.stat().st_size / 2**30:.2f} GB)"
    )
    return out


def load_cache(path: Path) -> dict:
    """Read a cache back, rebuilding the control matrix as CSR."""
    blob = np.load(path, allow_pickle=False)
    control = csr_matrix(
        (blob["control_data"], blob["control_indices"], blob["control_indptr"]),
        shape=tuple(blob["control_shape"]),
    )
    return {
        "control_X": control,
        "control_depth": blob["control_depth"],
        "pert_means": blob["pert_means"],
        "pert_names": blob["pert_names"].astype(str),
        "pert_counts": blob["pert_counts"],
        "observed": blob["observed"],
    }
