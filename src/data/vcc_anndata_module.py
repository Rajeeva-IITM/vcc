"""Datamodule that reads the original ``.h5ad`` files directly.

The other datamodules in this package consume the parquet/CSV artefacts that
``src/prepare_data.py`` derives from the AnnData files: a perturbed-expression
matrix, a control matrix, a row-metadata CSV. All of that already lives inside
the ``.h5ad`` -- controls included, tagged as ``target_gene == "non-targeting"``
-- so this module takes a single path and derives the rest.

It emits exactly the batch structure the existing models consume, so
``data=dataset_anndata`` composes with every ``model=`` option unchanged::

    {"ko_vec": <gene embedding>, "exp_vec": <control cell>}, <perturbed cell>

Normalisation (counts-per-``target_sum`` then ``log1p``) is applied on the fly
and matches ``src/prepare_data.py``, so the tensors are numerically the same as
the ones ``data=dataset_cp10k`` reads off disk.
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
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler

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
        Gene name -> embedding vector.
    library_size : np.ndarray | None
        Original per-cell total counts, indexed like ``X``. Kept so a prediction
        can be inverted back to counts against the control cell it was anchored
        to; exposed as ``control_library_size``, not put in the batch, so the
        model contract is unchanged.
    stage : {"predict", None}
        ``"predict"`` returns an empty list as the target.
    dtype : torch.dtype
        Dtype of the emitted expression vectors.
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
        self.dtype = dtype

        # Aligned with __getitem__ order, so predictions can be converted with
        #   counts = rint(expm1(y_pred) * control_library_size[:, None] / target_sum)
        self.control_library_size = (
            None if library_size is None else library_size[control_rows]
        )

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
        """Return one ``({ko_vec, exp_vec}, target)`` sample."""
        exp_input = self._row(self.control_rows[index])
        gene_input = self.gene_embeddings[self.perturbed_genes[index]]

        if self.is_predict_stage:
            pred_input = []
        else:
            pred_input = self._row(self.pert_rows[index])  # type: ignore[index]

        return {"ko_vec": gene_input, "exp_vec": exp_input}, pred_input


class AnnDataVCCDataModule(LightningDataModule):
    """Datamodule sourced from ``.h5ad`` files rather than processed parquet.

    Unlike the other datamodules, this one does not take separate train, test
    and control paths. Controls are found inside the data (by ``control_label``)
    and the train/validation split is a held-out-gene split over whatever
    perturbations the files contain.

    Parameters
    ----------
    data_path : str | Path | list[str | Path]
        One ``.h5ad``, or several to concatenate. Their gene panels must match.
    gene_embedding_path : str | Path
        Parquet with a ``gene_name`` column plus numeric embedding columns.
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
    """

    def __init__(
        self,
        data_path: str | Path | list[str | Path],
        gene_embedding_path: str | Path,
        gene_list_path: str | Path | None = None,
        pert_counts_path: str | Path | None = None,
        control_label: str = "non-targeting",
        target_sum: float = TARGET_SUM,
        log1p: bool = True,
        seed: int = 42,
        num_workers: int = 8,
        batch_size: int = 128,
        test_size: float = 0.2,
    ) -> None:
        super().__init__()

        self.data_paths: list[str | Path] = (
            [data_path] if isinstance(data_path, (str, Path)) else list(data_path)
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

            del adata
            gc.collect()

        X = blocks[0] if len(blocks) == 1 else vstack(blocks, format="csr")
        del blocks
        gc.collect()

        return X, np.concatenate(labels)

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

    def _load_embeddings(self) -> dict[str, torch.Tensor]:
        """Load the perturbation embedding table as a gene -> vector dict."""
        embeddings = read_data(self.gene_embedding_path)
        vectors = embeddings.select(cs.numeric()).to_torch()
        return {
            gene: vectors[idx]
            for idx, gene in enumerate(embeddings["gene_name"].to_numpy())
        }

    def setup(self, stage: str | None = None) -> None:
        """Load, normalise and split the data for the requested stage."""
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

    def train_dataloader(self):
        """Training dataloader, sampling genes inversely to their cell count."""
        console.log("Creating Training Dataloader")

        genes = self.data.perturbed_genes[self.train_index]
        unique, counts = np.unique(genes, return_counts=True)
        weight_dict = dict(zip(unique, 1 / counts))
        sampler = WeightedRandomSampler(
            [weight_dict[gene] for gene in genes], len(self.train_data)
        )

        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            sampler=sampler,
        )

    def val_dataloader(self):
        """Validation dataloader over the held-out genes."""
        console.log("Creating Validation Dataloader")
        return DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
        )

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
