"""Datamodule that reads the original ``.h5ad`` files directly.

The other datamodules in this package consume the parquet/CSV artefacts that
``src/prepare_data.py`` derives from the AnnData files: a perturbed-expression
matrix, a control matrix, a row-metadata CSV. All of that already lives inside
the ``.h5ad`` -- controls included, tagged as ``target_gene == "non-targeting"``
-- so this module takes a single path and derives the rest.

It emits exactly the batch structure the existing models consume, so
``data=dataset_anndata`` composes with every ``model=`` option unchanged::

    {"ko_vec": <gene embedding>, "ko_id": <its row>, "exp_vec": <control cell>,
     "gene_mask": <genes this source measured>},
    <perturbed cell>

``ko_id`` addresses the same perturbation as ``ko_vec``, as a row index rather than a
vector, for nets that hold a learnable embedding table
(``flow_model.LearnableGeneEmbedding``). Nets read the keys they need by name, so its
presence is inert for every model that does not want it.

``gene_mask`` is 1 on genes the sample's source actually measured. It appears **only**
when the listed sources disagree on their gene panel -- see :mod:`src.data.source_cache`
-- so single-dataset training pays nothing for it. Nets read the keys they need by name,
so both it and ``ko_id`` are inert for models that do not want them.

Normalisation (counts-per-``target_sum`` then ``log1p``) is applied on the fly
and matches ``src/prepare_data.py``, so the tensors are numerically the same as
the ones ``data=dataset_cp10k`` reads off disk.

Two loading paths
-----------------
``target_mode="cell"`` reads whole matrices into memory, as it always has; every file
listed must already share one gene panel. ``target_mode="perturbation_mean"`` goes
through :mod:`src.data.source_cache` instead, which reconciles differing panels,
value spaces and column names, and -- because that mode never reads an individual
perturbed cell -- keeps only the per-perturbation means and the controls. That is what
makes a 71 GB file trainable; it is also why ``"cell"`` cannot use those caches.
"""

import gc
from pathlib import Path
from typing import Literal

import anndata as ad
import numpy as np
import polars as pl
import polars.selectors as cs
import rich
import torch
from lightning.pytorch import LightningDataModule
from scipy.sparse import csr_matrix, vstack
from torch.utils.data import (
    DataLoader,
    Dataset,
    Sampler,
    Subset,
    WeightedRandomSampler,
)

from src.data.source_cache import (
    SourceSpec,
    build_cache,
    load_cache,
    resolve_norm_axis,
)
from src.utils.data import read_data
from src.utils.gene_train_test_split import gene_train_test_split

console = rich.console.Console()

# The scale the cp10k-processed-data/ parquets on disk were actually written at,
# verified cell-by-cell against them. Note that src/prepare_data.py currently has
# TARGET_SUM = 5e4 -- an uncommitted local edit that was never re-run, and that
# disagrees with its own docstring, the directory name, and the data it produced.
TARGET_SUM = 1e4


class AnnDataVCCDataset(Dataset):
    """Pairs of (control cell, perturbed cell) drawn from a shared CSR matrix.

    The matrix is held sparse and one row is densified per item. For the 2025
    training file that is 15.5 GB resident instead of the 16 GB a dense copy
    would need -- the file is 48% dense, so the saving is small, but it avoids
    ever holding two full-size matrices at once.

    Parameters
    ----------
    X : csr_matrix
        Already-normalised expression, cells x genes. Shared by reference with
        the datamodule; never modified here.
    pert_rows : np.ndarray | None
        Row index of the perturbed cell for each sample. ``None`` in the
        ``predict`` stage, where no target exists.
    control_rows : np.ndarray
        Row index of the control cell paired with each sample.
    perturbed_genes : np.ndarray
        Name of the gene knocked out in each sample. Read by ``src/train.py``
        off ``datamodule.test_data`` to colour the UMAP.
    gene_embeddings : dict[str, torch.Tensor]
        Gene name -> embedding vector, **in the row order of the embedding file**.
        ``ko_id`` is a gene's position in this mapping, which is therefore its row in
        that file -- the same indexing ``LearnableGeneEmbedding(init_path=...)`` builds
        its table with. Deriving the id from this dict rather than taking it as a
        separate argument makes it structurally impossible for ``ko_vec`` and ``ko_id``
        to address different genes.
    library_size : np.ndarray | None
        Original per-cell total counts, indexed like ``X``. Kept so a prediction
        can be inverted back to counts against the control cell it was anchored
        to; exposed as ``control_library_size``, not put in the batch, so the
        model contract is unchanged.
    stage : {"predict", None}
        ``"predict"`` returns an empty list as the target.
    dtype : torch.dtype
        Dtype of the emitted expression vectors.
    pert_means : torch.Tensor | None
        ``(n_perturbations, n_genes)`` table of mean profiles. When given, the target
        is the perturbation's mean rather than the individual perturbed cell.
    pert_codes : np.ndarray | None
        Row of ``pert_means`` for each sample. Required with ``pert_means``.
    """

    def __init__(
        self,
        X: csr_matrix,
        pert_rows: np.ndarray | None,
        control_rows: np.ndarray,
        perturbed_genes: np.typing.NDArray[np.str_],
        gene_embeddings: dict[str, torch.Tensor],
        stage: Literal["predict", None] = None,
        dtype: torch.dtype = torch.float32,
        library_size: np.ndarray | None = None,
        pert_means: torch.Tensor | None = None,
        pert_codes: np.ndarray | None = None,
    ) -> None:
        super().__init__()

        self.stage = stage
        self.is_predict_stage = stage == "predict"

        if not self.is_predict_stage and pert_rows is None:
            raise ValueError("`pert_rows` is required outside the predict stage")
        if pert_rows is not None and len(pert_rows) != len(control_rows):
            raise ValueError(
                f"pert/control row counts differ: {len(pert_rows)} vs {len(control_rows)}"
            )
        if len(perturbed_genes) != len(control_rows):
            raise ValueError(
                f"gene labels do not match row count: "
                f"{len(perturbed_genes)} vs {len(control_rows)}"
            )

        # Held as the three raw CSR buffers: `X[i].toarray()` builds and throws
        # away a sparse matrix per item, which dominates the cost of __getitem__.
        self.data = X.data
        self.indices = X.indices
        self.indptr = X.indptr
        self.n_genes = X.shape[1]

        self.pert_rows = pert_rows
        self.control_rows = control_rows
        self.perturbed_genes = perturbed_genes
        self.gene_embeddings = gene_embeddings
        # Insertion order is the embedding file's row order; see the class docstring.
        self.gene_to_id = {gene: i for i, gene in enumerate(gene_embeddings)}
        self.dtype = dtype

        # Aligned with __getitem__ order, so predictions can be converted with
        #   counts = rint(expm1(y_pred) * control_library_size[:, None] / target_sum)
        self.control_library_size = (
            None if library_size is None else library_size[control_rows]
        )

        # When set, __getitem__ returns the perturbation's mean profile instead of the
        # individual perturbed cell. See AnnDataVCCDataModule's `target_mode`.
        if (pert_means is None) != (pert_codes is None):
            raise ValueError("`pert_means` and `pert_codes` must be given together")
        if pert_codes is not None and len(pert_codes) != len(perturbed_genes):
            raise ValueError(
                f"pert_codes length {len(pert_codes)} does not match "
                f"{len(perturbed_genes)} samples"
            )
        self.pert_means = pert_means
        self.pert_codes = pert_codes

    def __len__(self) -> int:
        """Number of (control, perturbed) pairs."""
        return len(self.perturbed_genes)

    def _row(self, row: int) -> torch.Tensor:
        """Densify one CSR row."""
        start, end = self.indptr[row], self.indptr[row + 1]
        out = np.zeros(self.n_genes, dtype=np.float32)
        out[self.indices[start:end]] = self.data[start:end]
        return torch.from_numpy(out).to(self.dtype)

    def __getitem__(
        self, index: int
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor | list[None]]:
        """Return one ``({ko_vec, ko_id, exp_vec}, target)`` sample."""
        gene = self.perturbed_genes[index]
        exp_input = self._row(self.control_rows[index])
        gene_input = self.gene_embeddings[gene]

        if self.is_predict_stage:
            pred_input = []
        elif self.pert_means is not None:
            # Precomputed and shared across workers; the collate_fn copies it into the
            # stacked batch, so handing back a row view is safe.
            pred_input = self.pert_means[self.pert_codes[index]]  # type: ignore[index]
        else:
            pred_input = self._row(self.pert_rows[index])  # type: ignore[index]

        return {
            "ko_vec": gene_input,
            # A plain int; the default collate stacks these into a (B,) int64 tensor,
            # which is what nn.Embedding wants.
            "ko_id": self.gene_to_id[gene],
            "exp_vec": exp_input,
        }, pred_input


class MultiSourceVCCDataset(Dataset):
    """Samples drawn from several datasets reconciled onto one gene panel.

    The counterpart to :class:`AnnDataVCCDataset` for
    ``target_mode="perturbation_mean"``. It never holds a perturbed cell: the target is
    always a row of ``pert_means``, which is the whole reason the source caches can throw
    the perturbed rows away.

    Two invariants are worth stating because violating either silently produces a
    plausible-looking but meaningless objective:

    - **A sample's control comes from its own source.** ``control_rows`` is built per
      source by the datamodule. Pairing a Replogle mean with a 2025 control would make
      ``u = mu_p - x_0`` a cell-line difference rather than a perturbation effect.
    - **Unmeasured genes are filled, not zeroed.** A source covering 41% of the panel
      would otherwise hand ``exp_processor`` an input that is 84% exact zeros, against
      52% for a 2025 control. ``fill_profile`` (the reference source's pooled control
      mean) keeps the input plausible; ``gene_mask`` stops the loss from believing it.

    Parameters
    ----------
    control_X : csr_matrix
        Every source's control cells, stacked, already on the model panel.
    control_rows : np.ndarray
        Row of ``control_X`` paired with each sample. Same source as the sample.
    pert_means : torch.Tensor
        ``(n_perturbations, n_genes)`` mean profiles, stacked across sources.
    pert_codes : np.ndarray
        Row of ``pert_means`` for each sample.
    perturbed_genes : np.ndarray
        Perturbed gene name per sample; also read by ``src/train.py`` for the UMAP.
    source_of_sample : np.ndarray
        Index into ``observed`` for each sample.
    gene_embeddings : dict[str, torch.Tensor]
        Gene name -> vector, in the embedding file's row order. ``ko_id`` is a gene's
        position in this mapping; deriving it here rather than passing it separately
        makes it impossible for ``ko_vec`` and ``ko_id`` to address different genes.
    observed : np.ndarray
        ``(n_sources, n_genes)`` bool. Emitted as ``gene_mask`` when any row is partial.
    fill_profile : torch.Tensor | None
        Value used for genes a source did not measure. ``None`` leaves them at zero.
    dtype : torch.dtype
        Dtype of the emitted expression vectors.
    stage : {"predict", None}
        ``"predict"`` returns an empty list as the target.
    """

    def __init__(
        self,
        control_X: csr_matrix,
        control_rows: np.ndarray,
        pert_means: torch.Tensor | None,
        pert_codes: np.ndarray | None,
        perturbed_genes: np.typing.NDArray[np.str_],
        source_of_sample: np.ndarray,
        gene_embeddings: dict[str, torch.Tensor],
        observed: np.ndarray,
        fill_profile: torch.Tensor | None = None,
        dtype: torch.dtype = torch.float32,
        stage: Literal["predict", None] = None,
        control_depth: np.ndarray | None = None,
        source_context: np.ndarray | None = None,
    ) -> None:
        super().__init__()

        self.stage = stage
        self.is_predict_stage = stage == "predict"
        if not self.is_predict_stage and (pert_means is None or pert_codes is None):
            raise ValueError("`pert_means` and `pert_codes` are required to train")
        for name, arr in (
            ("control_rows", control_rows),
            ("source_of_sample", source_of_sample),
        ):
            if len(arr) != len(perturbed_genes):
                raise ValueError(
                    f"{name} length {len(arr)} does not match "
                    f"{len(perturbed_genes)} samples"
                )

        self.data = control_X.data
        self.indices = control_X.indices
        self.indptr = control_X.indptr
        self.n_genes = control_X.shape[1]

        self.control_rows = control_rows
        self.pert_means = pert_means
        self.pert_codes = pert_codes
        self.perturbed_genes = perturbed_genes
        self.source_of_sample = source_of_sample
        self.gene_embeddings = gene_embeddings
        self.gene_to_id = {gene: i for i, gene in enumerate(gene_embeddings)}
        self.dtype = dtype

        self.observed = torch.from_numpy(np.ascontiguousarray(observed))
        # All-ones masks carry no information and cost ~9 MB per batch to collate, so
        # the key is omitted entirely unless some source is genuinely partial.
        self.emit_mask = not bool(observed.all())
        self.mask = self.observed.to(dtype) if self.emit_mask else None
        self.fill_profile = None if fill_profile is None else fill_profile.to(dtype)

        self.control_library_size = (
            None if control_depth is None else control_depth[control_rows]
        )

        # Per-source cell-context vector (pooled control profile over the shared norm axis).
        # None unless context conditioning is on; emitted per sample as `context_vec`.
        self.source_context = (
            None
            if source_context is None
            else torch.from_numpy(np.ascontiguousarray(source_context)).to(dtype)
        )

    def __len__(self) -> int:
        """Number of samples."""
        return len(self.perturbed_genes)

    def _row(self, row: int) -> torch.Tensor:
        """Densify one control row."""
        start, end = self.indptr[row], self.indptr[row + 1]
        out = np.zeros(self.n_genes, dtype=np.float32)
        out[self.indices[start:end]] = self.data[start:end]
        return torch.from_numpy(out).to(self.dtype)

    def __getitem__(
        self, index: int
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor | list[None]]:
        """Return one ``({ko_vec, ko_id, exp_vec, [gene_mask]}, target)`` sample."""
        gene = self.perturbed_genes[index]
        source = self.source_of_sample[index]
        exp_input = self._row(self.control_rows[index])

        if self.fill_profile is not None and self.emit_mask:
            exp_input = torch.where(self.observed[source], exp_input, self.fill_profile)

        inputs = {
            "ko_vec": self.gene_embeddings[gene],
            "ko_id": self.gene_to_id[gene],
            "exp_vec": exp_input,
        }
        if self.emit_mask:
            inputs["gene_mask"] = self.mask[source]  # type: ignore[index]
        if self.source_context is not None:
            inputs["context_vec"] = self.source_context[source]

        if self.is_predict_stage:
            return inputs, []
        return inputs, self.pert_means[self.pert_codes[index]]  # type: ignore[index]


class PerturbationGroupedBatchSampler(Sampler):
    """Batches of ``perts_per_batch`` perturbations x ``cells_per_pert`` cells each.

    Why this exists
    ---------------
    Any per-perturbation *pseudobulk* loss (``BatchDiffExpError``,
    ``BatchDeltaMagnitudeLoss``) needs several cells of the SAME perturbation in a batch to
    average out the control-cell deviation, which is ~28x the perturbation signal per cell
    (``BatchDiffExpError`` docstring: 493.9 vs 17.9). The default cell-shuffled
    ``WeightedRandomSampler`` draws ~one cell per perturbation when the mixture has ~19k
    perturbations, so ``min_group_size`` drops nearly every group and the pseudobulk term
    is inert. This sampler instead draws whole perturbations, then a fixed number of that
    perturbation's cells (different control pairings, same target mean), giving each batch
    ``perts_per_batch`` clean pseudobulks.

    Sampling matches the default path's marginal: a perturbation is drawn with probability
    ``group_weight`` (each perturbation gets its source's ``source_weights`` share divided
    equally among that source's perturbations), so the per-source epoch mix is preserved.
    Perturbations are drawn WITHOUT replacement within a batch (no duplicate perturbation in
    one batch); a group's cells are drawn with replacement only when it holds fewer than
    ``cells_per_pert``.

    Yields flat lists of positions into the training ``Subset`` -- the same index space the
    ``WeightedRandomSampler`` path uses -- so nothing downstream changes.
    """

    def __init__(
        self,
        group_indices: list["np.ndarray"],
        group_weight: torch.Tensor,
        perts_per_batch: int,
        cells_per_pert: int,
        num_batches: int,
        seed: int = 0,
    ) -> None:
        if perts_per_batch < 1 or cells_per_pert < 1:
            raise ValueError("perts_per_batch and cells_per_pert must be >= 1")
        if len(group_indices) == 0:
            raise ValueError("no perturbation groups to sample from")
        self.group_indices = group_indices
        self.group_weight = group_weight
        self.perts_per_batch = min(perts_per_batch, len(group_indices))
        self.cells_per_pert = cells_per_pert
        self.num_batches = int(num_batches)
        # One persistent generator so successive epochs (each a fresh __iter__) differ; it
        # lives in the main process, since a batch_sampler runs there, not in the workers.
        self._gen = torch.Generator().manual_seed(int(seed))

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self):
        for _ in range(self.num_batches):
            groups = torch.multinomial(
                self.group_weight,
                self.perts_per_batch,
                replacement=False,
                generator=self._gen,
            )
            batch: list[int] = []
            for g in groups.tolist():
                cells = self.group_indices[g]
                replace = len(cells) < self.cells_per_pert
                pick = (
                    torch.randint(
                        len(cells), (self.cells_per_pert,), generator=self._gen
                    )
                    if replace
                    else torch.randperm(len(cells), generator=self._gen)[
                        : self.cells_per_pert
                    ]
                )
                batch.extend(int(cells[i]) for i in pick.tolist())
            yield batch


class AnnDataVCCDataModule(LightningDataModule):
    """Datamodule sourced from ``.h5ad`` files rather than processed parquet.

    Unlike the other datamodules, this one does not take separate train, test
    and control paths. Controls are found inside the data (by ``control_label``)
    and the train/validation split is a held-out-gene split over whatever
    perturbations the files contain.

    Parameters
    ----------
    data_path : str | Path | list[str | Path | dict]
        One ``.h5ad``, or several to train on together. In ``target_mode="cell"`` their
        gene panels must already match. In ``target_mode="perturbation_mean"`` they need
        not: each entry may also be a mapping of :class:`~src.data.source_cache.SourceSpec`
        fields, and everything that can be auto-detected (the perturbation column, the
        control label, the gene-symbol column, whether values are already ``log1p``) is.
        Adding a dataset is adding a line here and nothing else.
    gene_embedding_path : str | Path | list[str | Path]
        Parquet with a ``gene_name`` column plus numeric embedding columns; or a list of
        them to L2-normalise per block and concatenate (see :meth:`_load_embeddings`).
    gene_list_path : str | Path | None
        Headerless CSV of gene names. If given, columns are reordered to this
        panel -- the hook for reconciling the 2025 and 2026 gene sets, which
        overlap heavily but are not in the same order.
    pert_counts_path : str | Path | None
        CSV of ``target_gene, n_cells`` defining what to generate in the predict
        stage. Falls back to the validation split's own genes and counts.
    control_label : str
        ``obs["target_gene"]`` value marking control cells.
    target_sum : float
        Counts-per-cell to normalise to before ``log1p``.
    log1p : bool
        Apply ``log1p`` after depth normalisation.
    seed : int
        Seeds both the control pairing and the gene split.
    num_workers, batch_size : int
        Dataloader settings.
    test_size : float
        Fraction of *genes* held out for validation.
    cache_dir : str | Path | None
        Where reconciled sources are cached. Defaults to ``_cache/`` beside the first
        file. The cache key covers the source files, the model panel and the derived
        normalisation axis, so it can never be stale -- adding a dataset shrinks the
        axis and rebuilds everything rather than mixing two normalisations.
    source_weights : dict[str, float] | None
        Share of each epoch given to each source, by name. ``None`` means equal shares.
        Equal is the default rather than proportional-to-perturbations because the 2025
        file is 2.6% of the perturbations but the only source covering all 18,080 genes;
        weighting by perturbation count would leave the other 10,624 genes receiving
        gradient 2.6% of the time.
    val_source : str | None
        Name of the source validation is scored on. Held-out genes are withheld from
        *every* source's training set regardless; this only decides where ``val/*`` is
        measured. Defaults to the first source, which keeps the numbers comparable to
        single-dataset runs.
    unobserved_fill : {"reference_mean", "zero"}
        What to put in genes a source did not measure. ``"reference_mean"`` uses the
        validation source's pooled control profile, so an aux control still looks like a
        cell; the loss is masked there either way, so this adds no gradient.
    epoch_size : int | None
        Samples drawn per epoch. Defaults to the validation source's perturbed-cell
        count times the number of sources, so each source contributes about what it
        would have contributed alone.
    rebuild_cache : bool
        Rebuild source caches even when a matching one exists.
    prior_val_fraction : float
        Of the genes held out of the reference source, what fraction is *also* withheld
        from every other source. Only matters for a model that learns its perturbation
        embedding, where it decides what validation measures.

        A held-out gene whose row still trains on Replogle reproduces the leaderboard
        situation for 272 of the 300 2026 targets: the perturbation was seen elsewhere,
        and the question is whether that transfers to this context. A gene withheld
        everywhere reproduces the situation for the remaining 28, whose rows never move
        and which therefore depend on the prior geometry surviving.

        ``1.0`` (default) withholds all of them everywhere, which is the single-metric
        behaviour every earlier run had. Anything in ``(0, 1)`` splits the held-out
        genes and exposes a second validation dataloader, logged under ``val_prior/``.
    reference_val_only : bool
        Hold the *entire* validation source out of training instead of only a
        ``test_size`` gene slice. Every one of its perturbations becomes a held-out
        gene, so none of its rows ever train and its context is fully unseen -- the
        genuine cross-context transfer test. Pair with ``prior_val_fraction=0.0`` to
        keep those genes training on the *other* sources (the board's "perturbation
        seen elsewhere, this context unseen" case, 272 of its 300 targets); leave it at
        ``1.0`` to also withhold them everywhere (the harder 28/300 case). The source
        still contributes its controls, its pooled reference-mean fill, and the whole
        validation set -- it just contributes no training gradient. ``False`` (default)
        is the ``test_size`` split every earlier run used.
    target_mode : {"cell", "perturbation_mean"}
        What the target is. ``"cell"`` (default) pairs each control with one
        individual perturbed cell. ``"perturbation_mean"`` replaces that with the
        perturbation's mean profile.

        The second exists because the first is almost all noise. Controls are paired
        at random, so the per-cell target carries the full cell-to-cell variability of
        both cells; measured on the 2025 data, knowing the perturbation explains only
        1.8% of its variance (velocity MSE 0.0545 against a 0.0555 predict-nothing
        floor). Averaging ~1,250 cells per perturbation leaves the signal untouched
        while cutting the noise by that factor, taking the signal fraction to ~96%.

        The MSE optimum is unchanged either way -- ``E[x1 | p]`` is the mean -- so this
        is a variance reduction, not a different objective. What does change is the
        prediction: every cell of a perturbation now maps to the same profile, and
        within-perturbation variance in log space goes to zero. The pseudobulk is
        identical, so the four DE-based challenge metrics are unaffected.
    """

    def __init__(
        self,
        data_path: str | Path | list[str | Path],
        gene_embedding_path: str | Path | list[str | Path],
        gene_list_path: str | Path | None = None,
        pert_counts_path: str | Path | None = None,
        control_label: str = "non-targeting",
        target_sum: float = TARGET_SUM,
        log1p: bool = True,
        seed: int = 42,
        num_workers: int = 8,
        batch_size: int = 128,
        test_size: float = 0.2,
        target_mode: Literal["cell", "perturbation_mean"] = "cell",
        cache_dir: str | Path | None = None,
        source_weights: dict[str, float] | None = None,
        val_source: str | None = None,
        unobserved_fill: Literal["reference_mean", "zero"] = "reference_mean",
        epoch_size: int | None = None,
        rebuild_cache: bool = False,
        prior_val_fraction: float = 1.0,
        group_by_perturbation: bool = False,
        perts_per_batch: int = 32,
        cells_per_pert: int = 32,
        reference_val_only: bool = False,
        emit_context: bool = False,
    ) -> None:
        super().__init__()

        if target_mode not in ("cell", "perturbation_mean"):
            raise ValueError(
                f"target_mode must be 'cell' or 'perturbation_mean', got {target_mode!r}"
            )
        self.target_mode = target_mode

        self.data_paths: list = (
            [data_path]
            if isinstance(data_path, (str, Path, dict, SourceSpec))
            else list(data_path)
        )
        if not self.data_paths:
            raise ValueError("`data_path` is empty")

        self.gene_embedding_path = gene_embedding_path
        self.gene_list_path = gene_list_path
        self.pert_counts_path = pert_counts_path
        self.control_label = control_label
        self.target_sum = target_sum
        self.log1p = log1p
        self.seed = seed
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.test_size = test_size
        self.source_weights = source_weights
        self.unobserved_fill = unobserved_fill
        self.epoch_size = epoch_size
        self.rebuild_cache = rebuild_cache
        if not 0.0 <= prior_val_fraction <= 1.0:
            raise ValueError(
                f"prior_val_fraction must be in [0, 1], got {prior_val_fraction}"
            )
        self.prior_val_fraction = prior_val_fraction
        self.val_prior_index: list[int] = []

        if group_by_perturbation and target_mode != "perturbation_mean":
            raise ValueError(
                "group_by_perturbation requires target_mode='perturbation_mean' -- the "
                "pseudobulk losses it feeds rely on the mu_p identity."
            )
        if perts_per_batch < 1 or cells_per_pert < 1:
            raise ValueError("perts_per_batch and cells_per_pert must be >= 1")
        self.group_by_perturbation = group_by_perturbation
        self.perts_per_batch = perts_per_batch
        self.cells_per_pert = cells_per_pert
        self.reference_val_only = reference_val_only
        self.emit_context = emit_context

        # Only meaningful in perturbation_mean mode; harmless to build either way.
        self.specs = [
            entry
            if isinstance(entry, SourceSpec)
            else SourceSpec(**entry)
            if isinstance(entry, dict)
            else SourceSpec(path=entry)
            for entry in self.data_paths
        ]
        self.source_names = [str(spec.name) for spec in self.specs]
        if len(set(self.source_names)) != len(self.source_names):
            raise ValueError(f"source names are not unique: {self.source_names}")
        self.val_source = val_source or self.source_names[0]
        if self.val_source not in self.source_names:
            raise ValueError(
                f"val_source {self.val_source!r} is not one of {self.source_names}"
            )
        self.cache_dir = (
            Path(cache_dir)
            if cache_dir is not None
            else Path(self.specs[0].path).parent / "_cache"
        )
        self.norm_genes: list[str] = []
        self.source_of_sample: np.ndarray = np.empty(0, dtype=np.int64)

        self.gene_names: list[str] = []
        self._cache: (
            tuple[csr_matrix, np.typing.NDArray[np.str_], np.ndarray] | None
        ) = None
        self.library_size: np.ndarray = np.empty(0)

    # ------------------------------------------------------------------ setup

    def _resolve_dtype(self) -> torch.dtype:
        """Match the trainer's precision.

        Read defensively: this module is runnable standalone, where no trainer
        is attached.
        """
        precision = getattr(getattr(self, "trainer", None), "precision", None)
        match precision:
            case "16-mixed" | "16-true":
                return torch.float16
            case "64-true":
                return torch.float64
            case "bf16-mixed" | "bf16-true":
                return torch.bfloat16
            case _:
                return torch.float32

    def _load_matrix(self) -> tuple[csr_matrix, np.typing.NDArray[np.str_]]:
        """Read every file, check the panels agree, and stack them.

        Lightning calls ``setup`` once per stage, so the result is cached: a
        predict run following a fit would otherwise re-read from disk and
        briefly hold two copies of a ~15 GB matrix.
        """
        if self._cache is not None:
            console.log("Reusing the matrix loaded for the previous stage")
            X, labels, self.library_size = self._cache
            return X, labels

        blocks: list[csr_matrix] = []
        labels: list[np.ndarray] = []
        barcodes: list[np.ndarray] = []

        for path in self.data_paths:
            console.log(f"Reading {path}")
            adata = ad.read_h5ad(path)

            var_names = [str(name) for name in adata.var_names]
            if not self.gene_names:
                self.gene_names = var_names
            elif var_names != self.gene_names:
                raise ValueError(
                    f"{path} has a different gene panel than {self.data_paths[0]}: "
                    f"{len(var_names)} vs {len(self.gene_names)} genes, or the same "
                    "genes in a different order. Files listed together must already "
                    "agree -- `gene_list_path` permutes one shared panel, it cannot "
                    "reconcile two different ones."
                )

            blocks.append(csr_matrix(adata.X))
            labels.append(adata.obs["target_gene"].to_numpy().astype(str))
            barcodes.append(adata.obs_names.to_numpy().astype(str))

            del adata
            gc.collect()

        X = blocks[0] if len(blocks) == 1 else vstack(blocks, format="csr")
        del blocks
        gc.collect()

        self._check_no_duplicate_cells(np.concatenate(barcodes))
        return X, np.concatenate(labels)

    @staticmethod
    def _check_no_duplicate_cells(barcodes: np.typing.NDArray[np.str_]) -> None:
        """Refuse to stack files that share cells.

        The three 2025 split files each contain the *same* 38,176 non-targeting
        cells, so listing them together would count every control three times --
        27.6% of the dataset instead of 9.2% -- and silently reweight every
        control-paired sample. Fail loudly instead; ``scripts/merge_2025.py``
        builds a properly de-duplicated single file.
        """
        uniq, counts = np.unique(barcodes, return_counts=True)
        dupes = counts > 1
        if not dupes.any():
            return
        n = int(dupes.sum())
        raise ValueError(
            f"{n:,} cell barcodes appear in more than one of the files listed in "
            f"`data_path` (e.g. {uniq[dupes][:3].tolist()}). Stacking them would "
            "double-count those cells. If these are the 2025 splits, build the "
            "merged dataset with `python scripts/merge_2025.py` and point "
            "`data_path` at its output instead."
        )

    def _finish_load(self, X: csr_matrix, labels) -> csr_matrix:
        """Reindex and normalise, then cache -- both are idempotent per instance."""
        if self._cache is None:
            if self.gene_list_path is not None:
                X = self._reorder_genes(X)
            # Depth is measured on the panel that is actually kept, so a
            # gene_list_path that drops columns is reflected in the library size.
            self.library_size = self._normalise(X)
            self._cache = (X, labels, self.library_size)
        return X

    def _reorder_genes(self, X: csr_matrix) -> csr_matrix:
        """Permute columns into the order given by ``gene_list_path``."""
        # The 2025 gene_names.csv has no header row; reading it with the default
        # silently consumes SAMD11 as a column name and yields 18,079 genes.
        wanted = (
            read_data(self.gene_list_path)  # type: ignore[arg-type]
            if str(self.gene_list_path).endswith(".parquet")
            else pl.read_csv(self.gene_list_path, has_header=False)  # type: ignore[arg-type]
        )
        wanted_names = [str(name) for name in wanted.to_series(0)]

        position = {name: i for i, name in enumerate(self.gene_names)}
        missing = [name for name in wanted_names if name not in position]
        if missing:
            raise ValueError(
                f"{len(missing)} genes in {self.gene_list_path} are absent from the "
                f"data, e.g. {missing[:5]}"
            )

        order = np.array([position[name] for name in wanted_names])
        if order.size == len(self.gene_names) and np.all(
            order == np.arange(order.size)
        ):
            console.log("Gene list already matches the data order; no reindex needed")
            return X

        console.log(f"Reordering {len(self.gene_names)} -> {len(wanted_names)} genes")
        X = X.tocsc()[:, order].tocsr()
        self.gene_names = wanted_names
        return X

    def _normalise(self, X: csr_matrix) -> np.ndarray:
        """``log1p(counts / depth * target_sum)``, in place on the CSR values.

        ``log1p(0) == 0``, so the transform maps structural zeros to zeros and
        the sparsity pattern is preserved exactly -- no dense matrix is built.

        Returns the per-cell library sizes. They are not recoverable from the
        normalised values afterwards, and inverting a prediction back to counts
        needs them: ``expm1(y_pred) * library_size / target_sum``.
        """
        console.log("Normalising expression")
        depth = np.asarray(X.sum(axis=1)).ravel()
        depth[depth == 0] = (
            1.0  # no such cells here, but re-running on new data is safe
        )

        X.data /= np.repeat(depth, np.diff(X.indptr))
        X.data *= self.target_sum
        if self.log1p:
            np.log1p(X.data, out=X.data)

        return depth

    @staticmethod
    def _perturbation_means(
        X: csr_matrix,
        pert_rows: np.ndarray,
        perturbed_genes: np.typing.NDArray[np.str_],
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, np.ndarray]:
        """Mean expression profile per perturbation, plus each sample's row into it.

        Computed on the **normalised** matrix, so these are means of
        ``log1p(CP10K)`` -- not ``log1p`` of the mean count. That matches the space
        the model predicts in and the pseudobulk definition the DE metrics use. The
        two are not equal and the alternative is defensible, so the choice is
        deliberate rather than incidental.

        A sparse group-by rather than 300 fancy-index slices: build a
        (n_perturbations x n_cells) averaging operator and multiply once.

        Returns ``(means, codes)`` where ``means[codes[i]]`` is the target for sample
        ``i``. At 300 x 18,080 float32 the table is ~22 MB, so it is held dense and
        shared copy-on-write with the forked dataloader workers.
        """
        names, codes = np.unique(perturbed_genes, return_inverse=True)
        counts = np.bincount(codes)

        averager = csr_matrix(
            (1.0 / counts[codes], (codes, pert_rows)),
            shape=(len(names), X.shape[0]),
        )
        means = (averager @ X).toarray()

        console.log(
            f"Perturbation-mean targets: {len(names)} perturbations, "
            f"{counts.min()}-{counts.max()} cells each (median {int(np.median(counts))})"
        )
        return torch.from_numpy(means).to(dtype), codes

    def _load_embeddings(self) -> dict[str, torch.Tensor]:
        """Load the perturbation embedding table as a gene -> vector dict.

        ``gene_embedding_path`` is one parquet, or a **list** of them to concatenate. Each
        source is L2-normalised per row before concatenation, so a low-dimensional block is
        not swamped by a high-dimensional one in a cosine kernel -- both contribute equally
        and the model's learned anchors decide the weighting. Rows align on the FIRST
        source's gene set (its order fixes ``ko_id``); a gene a later source lacks is
        zero-filled for that block, an explicit "no signal" token.
        """
        paths = self.gene_embedding_path
        if isinstance(paths, (str, Path)):
            embeddings = read_data(paths)
            vectors = embeddings.select(cs.numeric()).to_torch().float()
            return {
                str(gene): vectors[idx]
                for idx, gene in enumerate(embeddings["gene_name"].to_numpy())
            }

        blocks: list[tuple[dict[str, torch.Tensor], int]] = []
        base_genes: list[str] | None = None
        for p in paths:
            df = read_data(p)
            names = [str(g) for g in df["gene_name"].to_numpy()]
            # Per-row L2 so each block contributes equally to a downstream cosine; a zero
            # row (a later source's fill) normalises to zero and stays a null token.
            vecs = torch.nn.functional.normalize(
                df.select(cs.numeric()).to_torch().float(), dim=1
            )
            blocks.append(({g: vecs[i] for i, g in enumerate(names)}, vecs.shape[1]))
            if base_genes is None:
                base_genes = names
        assert base_genes is not None
        combined = {
            g: torch.cat(
                [row[g] if g in row else torch.zeros(dim) for row, dim in blocks]
            )
            for g in base_genes
        }
        console.log(
            f"Combined embedding: {len(combined)} genes x "
            f"{sum(dim for _, dim in blocks)} dims from {len(paths)} sources "
            f"(per-block L2-normalised)"
        )
        return combined

    # -------------------------------------------------------- multi-source setup

    def _model_panel(self) -> list[str]:
        """The gene panel the model reads and writes.

        ``gene_list_path`` when given -- that is what pins the output to the 18,080
        genes ``scripts/predict_2026.py`` expects -- otherwise the first source's own
        panel, which reproduces single-dataset behaviour.
        """
        if self.gene_list_path is None:
            return list(self.specs[0].resolve()["genes"])

        wanted = (
            read_data(self.gene_list_path)
            if str(self.gene_list_path).endswith(".parquet")
            # The 2025 gene_names.csv has no header row; reading it with the default
            # silently consumes SAMD11 as a column name and yields 18,079 genes.
            else pl.read_csv(self.gene_list_path, has_header=False)
        )
        return [str(name) for name in wanted.to_series(0)]

    def _write_norm_axis(self) -> None:
        """Record the normalisation axis beside the checkpoints.

        The axis is derived from the source list, so it is not written down anywhere in
        the config -- but ``scripts/predict_2026.py`` has to reproduce it exactly or the
        input scale shifts silently between training and prediction. Dropping a copy in
        the run directory makes the checkpoint self-describing, and lets the prediction
        script find it without being told.
        """
        # `LightningDataModule.trainer` is a plain attribute defaulting to None, not a
        # property that raises, so this is safe before attach and when running the
        # module standalone.
        trainer = getattr(self, "trainer", None)
        if trainer is None or not trainer.is_global_zero:
            return

        targets = {
            Path(cb.dirpath)
            for cb in trainer.callbacks
            if getattr(cb, "dirpath", None) is not None
        } or {Path(trainer.default_root_dir)}

        for directory in targets:
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / "norm_axis.csv"
            path.write_text("\n".join(self.norm_genes) + "\n")
            console.log(f"Wrote the normalisation axis to {path}")

    def _setup_multi(self, stage: str | None = None) -> None:
        """Load every source through its cache and build the combined dataset."""
        dtype = self._resolve_dtype()
        rng = np.random.default_rng(self.seed)

        model_genes = self._model_panel()
        gene_embeddings = self._load_embeddings()

        self.norm_genes, digest = resolve_norm_axis(model_genes, self.specs)
        console.log(
            f"Normalisation axis: {len(self.norm_genes):,} genes shared by all "
            f"{len(self.specs)} source(s) and the {len(model_genes):,}-gene model panel "
            f"[{digest}]"
        )
        if len(self.norm_genes) < len(model_genes):
            console.log(
                f"[yellow]{len(model_genes) - len(self.norm_genes):,} model genes are "
                "not measured everywhere; depth is summed over the shared axis only so "
                "the same gene reads at one scale in every source[/yellow]"
            )
        self._write_norm_axis()

        caches = [
            load_cache(
                build_cache(
                    spec,
                    model_genes,
                    self.norm_genes,
                    set(gene_embeddings),
                    self.cache_dir,
                    self.target_sum,
                    self.log1p,
                    self.rebuild_cache,
                )
            )
            for spec in self.specs
        ]

        self.gene_names = model_genes
        reference = self.source_names.index(self.val_source)

        control_blocks = [cache["control_X"] for cache in caches]
        control_X = (
            control_blocks[0]
            if len(control_blocks) == 1
            else vstack(control_blocks, format="csr")
        )
        control_start = np.cumsum([0] + [block.shape[0] for block in control_blocks])

        # Cell-context vector per source: pooled control profile over the SHARED norm axis
        # (commensurable across sources and with the predict panel; avoids the per-source
        # coverage confound of the full 18,080). Fed to the conditioner when emit_context.
        context_of_source = None
        if self.emit_context:
            pos = {gene: i for i, gene in enumerate(model_genes)}
            norm_idx = np.array([pos[g] for g in self.norm_genes], dtype=np.int64)
            context_of_source = np.stack(
                [np.asarray(b.mean(axis=0)).ravel()[norm_idx] for b in control_blocks]
            ).astype(np.float32)
            console.log(
                f"context conditioning: {context_of_source.shape[1]}-d control profile "
                f"per source over the norm axis"
            )
        self.library_size = np.concatenate([c["control_depth"] for c in caches])

        pert_means = torch.from_numpy(
            np.concatenate([cache["pert_means"] for cache in caches])
        ).to(dtype)
        pert_names = np.concatenate([cache["pert_names"] for cache in caches])
        pert_counts = np.concatenate([cache["pert_counts"] for cache in caches])
        pert_source = np.repeat(
            np.arange(len(caches)), [len(cache["pert_names"]) for cache in caches]
        )
        observed = np.stack([cache["observed"] for cache in caches])

        for i, name in enumerate(self.source_names):
            console.log(
                f"  {name}: {len(caches[i]['pert_names']):,} perturbations, "
                f"{control_blocks[i].shape[0]:,} controls, "
                f"{int(observed[i].sum()):,}/{len(model_genes):,} genes measured"
            )

        # Genes a source did not measure are filled rather than left at zero: an input
        # that is 84% exact zeros against a reference cell's 52% is a domain shift the
        # exp_processor would have to spend capacity on. The loss is masked there
        # regardless, so this adds no gradient -- it only removes a false "gene is off".
        fill_profile = None
        if self.unobserved_fill == "reference_mean" and not observed.all():
            fill_profile = torch.from_numpy(
                np.asarray(
                    control_blocks[reference].mean(axis=0), dtype=np.float32
                ).ravel()
            )

        if stage == "predict":
            console.log("Setting up data for prediction")
            if self.pert_counts_path is not None:
                counts = read_data(self.pert_counts_path)
                target_genes = np.repeat(
                    counts["target_gene"].to_numpy(), counts["n_cells"].to_numpy()
                ).astype(str)
            else:
                _, val_genes = self._reference_split(caches[reference])
                target_genes = np.repeat(
                    caches[reference]["pert_names"], caches[reference]["pert_counts"]
                )
                target_genes = target_genes[np.isin(target_genes, val_genes)]

            self._check_embeddings(target_genes, gene_embeddings)
            lo, hi = control_start[reference], control_start[reference + 1]

            self.test_data = MultiSourceVCCDataset(
                control_X=control_X,
                control_rows=rng.integers(lo, hi, size=len(target_genes)),
                pert_means=None,
                pert_codes=None,
                perturbed_genes=target_genes,
                source_of_sample=np.full(len(target_genes), reference),
                gene_embeddings=gene_embeddings,
                observed=observed,
                fill_profile=fill_profile,
                dtype=dtype,
                stage="predict",
                control_depth=self.library_size,
                source_context=context_of_source,
            )
            console.log(f"Prediction set: {len(self.test_data):,} cells")
            gc.collect()
            return

        self._check_embeddings(pert_names, gene_embeddings)

        # One sample per perturbed cell, exactly as the single-source loader produced,
        # so epoch length and per-perturbation weighting keep their old meaning even
        # though the cells themselves were averaged away.
        perturbed_genes = np.repeat(pert_names, pert_counts)
        pert_codes = np.repeat(np.arange(len(pert_names)), pert_counts)
        self.source_of_sample = np.repeat(pert_source, pert_counts)

        control_rows = np.empty(len(perturbed_genes), dtype=np.int64)
        for i in range(len(caches)):
            rows = self.source_of_sample == i
            control_rows[rows] = rng.integers(
                control_start[i], control_start[i + 1], size=int(rows.sum())
            )

        console.log("Creating Dataset")
        self.data = MultiSourceVCCDataset(
            control_X=control_X,
            control_rows=control_rows,
            pert_means=pert_means,
            pert_codes=pert_codes,
            perturbed_genes=perturbed_genes,
            source_of_sample=self.source_of_sample,
            gene_embeddings=gene_embeddings,
            observed=observed,
            fill_profile=fill_profile,
            dtype=dtype,
            control_depth=self.library_size,
            source_context=context_of_source,
        )

        console.log("Data splitting")
        if self.reference_val_only:
            # The whole validation context is unseen: every reference perturbation is a
            # held-out gene, so `excluded` masks all reference rows out of training below
            # while the aux sources are untouched (with prior_val_fraction=0.0, they keep
            # training these genes -- the board's seen-elsewhere/unseen-here case).
            val_genes = np.unique(caches[reference]["pert_names"])
            console.log(
                f"reference_val_only: holding all {len(val_genes):,} "
                f"{self.val_source!r} perturbations out of training"
            )
        else:
            _, val_genes = self._reference_split(caches[reference])
        prior_genes, covered_genes = self._split_held_out(val_genes)

        is_reference = self.source_of_sample == reference
        # A gene in `prior_genes` leaves every source, so its embedding row never moves
        # and predicting it depends entirely on the prior. A gene in `covered_genes`
        # leaves the reference source only: the aux datasets still train its row, which
        # is exactly the position 272 of the 300 leaderboard targets are in. Expression
        # for both is only ever scored on the reference source, so neither leaks.
        excluded = np.where(
            is_reference,
            np.isin(perturbed_genes, val_genes),
            np.isin(perturbed_genes, prior_genes),
        )
        self.train_index = np.flatnonzero(~excluded).tolist()

        primary = covered_genes if len(covered_genes) else prior_genes
        self.val_index = np.flatnonzero(
            is_reference & np.isin(perturbed_genes, primary)
        ).tolist()
        self.val_prior_index = (
            np.flatnonzero(
                is_reference & np.isin(perturbed_genes, prior_genes)
            ).tolist()
            if len(covered_genes) and len(prior_genes)
            else []
        )

        if self.epoch_size is None:
            # One source's worth of cells per source, so adding a dataset lengthens the
            # epoch rather than diluting what every existing source contributes to it.
            self.epoch_size = int(caches[reference]["pert_counts"].sum()) * len(caches)
            console.log(f"Epoch size: {self.epoch_size:,} samples")

        self.train_data: Subset[MultiSourceVCCDataset] = Subset(
            self.data, self.train_index
        )
        self.val_data: Subset[MultiSourceVCCDataset] = Subset(self.data, self.val_index)
        self.val_prior_data: Subset[MultiSourceVCCDataset] = Subset(
            self.data, self.val_prior_index
        )
        console.log(
            f"Train {len(self.train_index):,} cells / "
            f"{len(np.unique(perturbed_genes[self.train_index])):,} genes across "
            f"{len(caches)} source(s), val {len(self.val_index):,} cells / "
            f"{len(primary):,} genes on {self.val_source!r}"
        )
        if self.val_prior_index:
            console.log(
                f"val_prior: {len(self.val_prior_index):,} cells / "
                f"{len(prior_genes):,} genes withheld from every source, so their "
                "embedding rows never train"
            )
        console.log("Setup Done")
        gc.collect()

    def _split_held_out(self, val_genes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Divide the held-out genes into rows that never train and rows the aux data does.

        Drawn from a stream offset from ``seed`` so that changing this split cannot
        disturb the gene split itself, which several runs are already compared against.

        Args:
            val_genes (np.ndarray): Genes held out of the reference source.

        Returns:
            tuple: ``(prior_genes, covered_genes)``, a partition of ``val_genes``.
        """
        n_prior = int(round(len(val_genes) * self.prior_val_fraction))
        if n_prior >= len(val_genes):
            return val_genes, np.array([], dtype=val_genes.dtype)
        if n_prior == 0:
            return np.array([], dtype=val_genes.dtype), val_genes

        rng = np.random.default_rng(self.seed + 1)
        prior = np.sort(rng.choice(val_genes, n_prior, replace=False))
        return prior, np.setdiff1d(val_genes, prior)

    def _reference_split(self, cache: dict) -> tuple[np.ndarray, np.ndarray]:
        """Held-out genes, chosen from the validation source alone.

        Splitting over the union of every source's perturbations would make the number
        of reference-source genes in validation a random variable -- and the 2025 file
        contributes only 300 of ~11,500 perturbations, so that variance is large. Taking
        the split from the reference source with the same seed reproduces the exact gene
        set a single-dataset run held out, which is what makes ``val/*`` comparable
        across runs. Genes held out here are then withheld from every source.
        """
        labels = np.repeat(cache["pert_names"], cache["pert_counts"])
        train_index, val_index = gene_train_test_split(
            labels, test_size=self.test_size, seed=self.seed
        )
        return np.unique(labels[train_index]), np.unique(labels[val_index])

    def setup(self, stage: str | None = None) -> None:
        """Load, normalise and split the data for the requested stage.

        Dispatches on ``target_mode``. ``"perturbation_mean"`` goes through the source
        caches, which is the only path that can reconcile differing gene panels;
        ``"cell"`` keeps the original whole-matrix loader, because it needs the
        individual perturbed cells the caches deliberately discard.
        """
        if self.target_mode == "perturbation_mean":
            self._setup_multi(stage)
        else:
            self._setup_legacy(stage)

    def _setup_legacy(self, stage: str | None = None) -> None:
        """Original single-panel loader; every file must share one gene panel."""
        dtype = self._resolve_dtype()
        rng = np.random.default_rng(self.seed)

        X, labels = self._load_matrix()
        console.log(f"Loaded {X.shape[0]:,} cells x {X.shape[1]:,} genes")

        X = self._finish_load(X, labels)

        is_control = labels == self.control_label
        control_rows = np.flatnonzero(is_control)
        pert_rows = np.flatnonzero(~is_control)
        console.log(
            f"{len(control_rows):,} control cells, {len(pert_rows):,} perturbed cells"
        )

        if len(control_rows) == 0:
            raise ValueError(
                f"No cells with target_gene == {self.control_label!r}; "
                "set `control_label` to match this dataset"
            )

        perturbed_genes = labels[pert_rows]
        gene_embeddings = self._load_embeddings()

        match stage:
            case "predict":
                console.log("Setting up data for prediction")

                if self.pert_counts_path is not None:
                    counts = read_data(self.pert_counts_path)
                    target_genes = np.repeat(
                        counts["target_gene"].to_numpy(),
                        counts["n_cells"].to_numpy(),
                    ).astype(str)
                else:
                    # Nothing told us what to predict, so reproduce the held-out
                    # genes at their observed abundance.
                    _, val_index = gene_train_test_split(
                        perturbed_genes, test_size=self.test_size, seed=self.seed
                    )
                    target_genes = perturbed_genes[val_index]

                self._check_embeddings(target_genes, gene_embeddings)

                self.test_data = AnnDataVCCDataset(
                    X=X,
                    pert_rows=None,
                    control_rows=rng.choice(control_rows, size=len(target_genes)),
                    perturbed_genes=target_genes,
                    gene_embeddings=gene_embeddings,
                    stage="predict",
                    dtype=dtype,
                    library_size=self.library_size,
                )
                console.log(f"Prediction set: {len(self.test_data):,} cells")

            case _:
                if len(pert_rows) == 0:
                    raise ValueError(
                        "No perturbed cells found -- this looks like a controls-only "
                        "file, which can only be used with stage='predict'"
                    )

                self._check_embeddings(perturbed_genes, gene_embeddings)

                # target_mode is "cell" here by construction -- `setup` routes
                # "perturbation_mean" to `_setup_multi`. Kept so this loader still works
                # standalone (see `__main__`) and so `_perturbation_means` stays tested.
                pert_means, pert_codes = None, None
                if self.target_mode == "perturbation_mean":
                    pert_means, pert_codes = self._perturbation_means(
                        X, pert_rows, perturbed_genes, dtype
                    )

                console.log("Creating Dataset")
                # Drawn once so every epoch sees the same pairing, matching
                # vcc_embedding_module.py.
                self.data = AnnDataVCCDataset(
                    X=X,
                    pert_rows=pert_rows,
                    control_rows=rng.choice(control_rows, size=len(pert_rows)),
                    perturbed_genes=perturbed_genes,
                    gene_embeddings=gene_embeddings,
                    stage=None,
                    dtype=dtype,
                    library_size=self.library_size,
                    pert_means=pert_means,
                    pert_codes=pert_codes,
                )

                console.log("Data splitting")
                self.train_index, self.val_index = gene_train_test_split(
                    perturbed_genes,
                    test_size=self.test_size,
                    seed=self.seed,
                )

                self.train_data: Subset[AnnDataVCCDataset] = Subset(
                    self.data, self.train_index
                )
                self.val_data: Subset[AnnDataVCCDataset] = Subset(
                    self.data, self.val_index
                )
                console.log(
                    f"Train {len(self.train_index):,} cells / "
                    f"{len(np.unique(perturbed_genes[self.train_index])):,} genes, "
                    f"val {len(self.val_index):,} cells / "
                    f"{len(np.unique(perturbed_genes[self.val_index])):,} genes"
                )
                console.log("Setup Done")

        gc.collect()

    @staticmethod
    def _check_embeddings(
        genes: np.typing.NDArray[np.str_], embeddings: dict[str, torch.Tensor]
    ) -> None:
        """Fail here rather than with a KeyError inside a dataloader worker."""
        missing = sorted(set(genes.tolist()) - embeddings.keys())
        if missing:
            raise KeyError(
                f"{len(missing)} perturbed genes have no embedding, e.g. {missing[:5]}"
            )

    # ------------------------------------------------------------ dataloaders

    def _sample_weights(self) -> np.ndarray:
        """Draw weight per training sample: equal per source, then per perturbation.

        Three nested corrections, innermost first. Dividing by a perturbation's cell
        count makes every perturbation equally likely within its source -- that is the
        original behaviour. Dividing by the source's perturbation count then makes every
        *source* equally likely, which matters because pooling the sources would leave
        the 2025 file at 2.6% of draws while being the only one that measures 10,624 of
        the 18,080 genes. ``source_weights`` overrides the last step.

        Counting is per ``(source, gene)`` pair, not per gene: 279 perturbations appear
        in both the 2025 and Replogle files and are genuinely different measurements.
        """
        genes = self.data.perturbed_genes[self.train_index]
        sources = (
            self.source_of_sample[self.train_index]
            if self.source_of_sample.size
            else np.zeros(len(genes), dtype=np.int64)
        )

        _, gene_codes = np.unique(genes, return_inverse=True)
        key = sources * (gene_codes.max() + 1) + gene_codes
        uniq_key, key_codes, cells_per_pert = np.unique(
            key, return_inverse=True, return_counts=True
        )
        perts_per_source = np.bincount(
            sources[np.unique(key_codes, return_index=True)[1]],
            minlength=len(self.source_names),
        ).astype(np.float64)
        perts_per_source[perts_per_source == 0] = 1.0

        share = np.ones(len(self.source_names), dtype=np.float64)
        if self.source_weights is not None:
            missing = set(self.source_weights) - set(self.source_names)
            if missing:
                raise KeyError(
                    f"source_weights names no such source: {sorted(missing)}"
                )
            share = np.array(
                [float(self.source_weights.get(n, 0.0)) for n in self.source_names]
            )
            if share.sum() <= 0:
                raise ValueError("source_weights must not sum to zero")
        share = share / share.sum()

        weights = share[sources] / perts_per_source[sources] / cells_per_pert[key_codes]

        if len(self.source_names) > 1:
            realised = np.bincount(
                sources, weights=weights, minlength=len(self.source_names)
            )
            realised = realised / realised.sum()
            console.log(
                "Epoch mix: "
                + ", ".join(f"{n} {p:.1%}" for n, p in zip(self.source_names, realised))
                + f" over {len(uniq_key):,} (source, perturbation) pairs"
            )
        return weights

    def _train_pert_groups(self) -> tuple[list[np.ndarray], torch.Tensor]:
        """Per-(source, perturbation) cell-index lists and their sampling weights.

        Mirrors :meth:`_sample_weights`' grouping but returns whole groups, for
        :class:`PerturbationGroupedBatchSampler`. The group weight is the source's
        ``source_weights`` share split equally across that source's perturbations -- i.e.
        the marginal the per-cell weights give a perturbation, before the ``/cells_per_pert``
        that only balances cells *within* a perturbation. Indices are positions into the
        training ``Subset``.
        """
        genes = self.data.perturbed_genes[self.train_index]
        sources = (
            self.source_of_sample[self.train_index]
            if self.source_of_sample.size
            else np.zeros(len(genes), dtype=np.int64)
        )
        _, gene_codes = np.unique(genes, return_inverse=True)
        key = sources * (gene_codes.max() + 1) + gene_codes
        uniq_key, key_codes = np.unique(key, return_inverse=True)
        n_groups = len(uniq_key)

        # All cells of a group share a source, so a scatter-assign leaves each group's
        # source correct regardless of write order.
        group_source = np.zeros(n_groups, dtype=np.int64)
        group_source[key_codes] = sources

        perts_per_source = np.bincount(
            group_source, minlength=len(self.source_names)
        ).astype(np.float64)
        perts_per_source[perts_per_source == 0] = 1.0

        share = np.ones(len(self.source_names), dtype=np.float64)
        if self.source_weights is not None:
            missing = set(self.source_weights) - set(self.source_names)
            if missing:
                raise KeyError(
                    f"source_weights names no such source: {sorted(missing)}"
                )
            share = np.array(
                [float(self.source_weights.get(n, 0.0)) for n in self.source_names]
            )
            if share.sum() <= 0:
                raise ValueError("source_weights must not sum to zero")
        share = share / share.sum()

        group_weight = share[group_source] / perts_per_source[group_source]
        group_weight = group_weight / group_weight.sum()

        counts = np.bincount(key_codes, minlength=n_groups)
        order = np.argsort(key_codes, kind="stable")
        group_indices = np.split(order, np.cumsum(counts)[:-1])
        return group_indices, torch.tensor(group_weight, dtype=torch.double)

    def train_dataloader(self):
        """Training dataloader, balanced across sources and perturbations."""
        console.log("Creating Training Dataloader")

        if self.group_by_perturbation:
            group_indices, group_weight = self._train_pert_groups()
            per_batch = self.perts_per_batch * self.cells_per_pert
            num_batches = max(1, (self.epoch_size or len(self.train_data)) // per_batch)
            batch_sampler = PerturbationGroupedBatchSampler(
                group_indices,
                group_weight,
                perts_per_batch=self.perts_per_batch,
                cells_per_pert=self.cells_per_pert,
                num_batches=num_batches,
                seed=self.seed,
            )
            console.log(
                f"Grouped batches: {batch_sampler.perts_per_batch} perts x "
                f"{self.cells_per_pert} cells = "
                f"{batch_sampler.perts_per_batch * self.cells_per_pert}/batch, "
                f"{num_batches:,} batches/epoch over {len(group_indices):,} "
                "(source, perturbation) groups"
            )
            return DataLoader(
                self.train_data,
                num_workers=self.num_workers,
                pin_memory=True,
                batch_sampler=batch_sampler,
                persistent_workers=self.num_workers > 0,
            )

        num_samples = self.epoch_size or len(self.train_data)
        sampler = WeightedRandomSampler(self._sample_weights().tolist(), num_samples)

        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            sampler=sampler,
            # Re-forking a process holding a 40 GB matrix once per epoch costs
            # real CPU in page-table work alone; keep the workers alive instead.
            persistent_workers=self.num_workers > 0,
        )

    def _val_loader(self, data) -> DataLoader:
        """One validation dataloader over a held-out subset."""
        return DataLoader(
            data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self):
        """Validation dataloader(s) over the held-out genes.

        Returns a *list* only when ``prior_val_fraction`` splits the held-out genes, in
        which case index 1 holds the genes whose embedding rows never trained and the
        module logs it under ``val_prior/``. Returning a bare loader otherwise keeps
        every model that does not know about the second one working unchanged.
        """
        console.log("Creating Validation Dataloader")
        primary = self._val_loader(self.val_data)
        if not self.val_prior_index:
            return primary
        console.log("Creating val_prior Dataloader (untrained embedding rows)")
        return [primary, self._val_loader(self.val_prior_data)]

    def test_dataloader(self):
        """Test dataloader over the prediction dataset."""
        console.log("Creating Test Dataloader")
        return DataLoader(
            self.test_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def predict_dataloader(self):
        """Prediction dataloader over the prediction dataset."""
        console.log("Creating prediction dataloader")
        return DataLoader(
            self.test_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )


if __name__ == "__main__":
    dm = AnnDataVCCDataModule(
        data_path="/home/rajeeva/Project/vcc_data/2025/train/adata_Training.h5ad",
        gene_embedding_path="/home/rajeeva/Project/vcc_data/gene_embeddings/poincare_go_gaf_logmapped_256.parquet",
        num_workers=2,
    )
    dm.setup(stage="fit")

    for X, y in dm.val_dataloader():
        print(
            f"ko_vec {tuple(X['ko_vec'].shape)}  "
            f"exp_vec {tuple(X['exp_vec'].shape)}  y {tuple(y.shape)}"
        )
        print(f"control cell total counts: {torch.expm1(X['exp_vec'][0]).sum():,.0f}")
        break
