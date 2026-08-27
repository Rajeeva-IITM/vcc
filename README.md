# Virtual Cell Challenge (VCC)

A deep learning pipeline for modeling gene expression changes under genetic perturbations (gene knockouts). Given a control (unperturbed) gene expression profile and a perturbation indicator, the model predicts the resulting gene expression.

## Project Structure

```
vcc/
├── src/
│   ├── train.py                         # Main training entry point (Hydra)
│   ├── prepare_data.py                  # CP10K normalisation + DE-threshold calibration
│   ├── data/
│   │   ├── vcc_anndata_module.py        # DataModule reading .h5ad directly (no preprocessing)
│   │   ├── vcc_datamodule.py            # Standard PyTorch Lightning DataModule
│   │   └── vcc_embedding_module.py      # DataModule using pre-computed gene embeddings
│   ├── models/
│   │   ├── vcc_lightning.py             # Lightning module wrapper
│   │   ├── projectionvcc_lightning.py   # Lightning module with projector head
│   │   └── components/
│   │       ├── basic_vcc_model.py       # CellModel, ProjectorCellModel, CellModelFiLM
│   │       ├── vcc_model_attention.py   # Attention-based model variants
│   │       ├── simple_model.py          # Baseline models
│   │       ├── diffawarecell.py         # Differential expression aware model
│   │       ├── loss_functions.py        # Custom loss functions (15+)
│   │       ├── metrics.py              # Custom evaluation metrics
│   │       └── activations.py          # SwiGLU activation
│   ├── utils/
│   │   ├── data.py                      # File reading (parquet, feather, CSV via polars)
│   │   ├── gene_train_test_split.py     # Gene-level OOD train/test splitting
│   │   ├── process_activation_function.py
│   │   ├── umap_utilities.py            # UMAP visualization utilities
│   │   └── model_params_from_study.py   # Optuna study parameter extraction
│   └── tuning/
│       └── tune_attention.py            # Hyperparameter tuning with Optuna
├── config/                              # Hydra configuration files
│   ├── train.yaml                       # Main training config
│   ├── tune.yaml                        # Hyperparameter tuning config
│   ├── eval.yaml                        # Evaluation config
│   ├── paths/defaults.yaml              # Path configuration (from .env)
│   ├── data/
│   │   ├── dataset_cp10k.yaml           # DEFAULT: CP10K data + GO embeddings
│   │   ├── dataset_anndata.yaml         # Same data, read straight from .h5ad
│   │   ├── dataset_embedding.yaml       # Legacy: log1p(raw) + quantile embeddings
│   │   └── dataset.yaml                 # Legacy one-hot KO variant (unmaintained)
│   ├── model/                           # Model architecture configs
│   │   ├── model.yaml                   # Default CellModel (bilinear fusion)
│   │   ├── model_attention.yaml         # Attention-based model
│   │   ├── model_gated_bilinear.yaml    # Bilinear + gated delta decoder, no attention
│   │   ├── model_attention_mse.yaml     # model_attention with a plain-MSE loss
│   │   ├── model_film.yaml              # FiLM conditioning model
│   │   └── model_simple.yaml            # Simple baseline
│   ├── trainer/trainer.yaml             # PyTorch Lightning trainer settings
│   ├── logging/wandb.yaml               # Weights & Biases logging
│   └── callbacks/                       # Lightning callbacks (checkpointing, early stopping, etc.)
├── scripts/                             # Analysis, evaluation and submission scripts
│   ├── de_mwu_2025.py                   # Mann-Whitney DE counts per perturbation
│   ├── predict_counts.py                # Predictions -> raw integer counts
│   ├── validate_2025.py                 # Score a checkpoint on the 2025 val/test splits
│   └── predict_2026.py                  # Build the 2026 submission (.h5ad, raw counts)
├── tests/                               # pytest suite
│   ├── test_losses.py                   # Loss differentiability / behaviour guards
│   └── test_anndata_module.py           # Datamodule contract tests
├── notebook/                            # Jupyter notebooks for analysis
│   ├── preliminary-data-analysis.ipynb  # Data exploration
│   ├── evaluation.ipynb                 # Model evaluation and metrics
│   ├── gene_expression_embeddings.ipynb # Gene embedding analysis
│   ├── model_debugging.ipynb            # Model debugging
│   └── contrastive-loss.ipynb           # Loss function exploration
├── pixi.toml                            # Pixi dependency specification
├── pixi.lock                            # Locked dependency versions
├── pyproject.toml                       # Python project config
├── .env                                 # Environment variables (your local paths)
└── .env.example                         # Template for .env
```

## Requirements

- Linux (x86_64)
- CUDA 12.0+ compatible GPU
- [Pixi](https://pixi.sh) package manager

### Key Dependencies

| Package | Purpose |
|---------|---------|
| PyTorch (GPU) | Deep learning framework |
| PyTorch Lightning | Training loop and utilities |
| Hydra | Configuration management |
| Polars | Data loading (parquet, feather, CSV) |
| Optuna | Hyperparameter tuning |
| Weights & Biases | Experiment tracking |
| scikit-learn | Train/test splitting |
| torchmetrics | Evaluation metrics |

See `pixi.toml` for the full list of dependencies and version constraints.

## Installation

1. **Install Pixi** (if not already installed):

   ```bash
   curl -fsSL https://pixi.sh/install.sh | bash
   ```

2. **Clone the repository and install dependencies**:

   ```bash
   git clone <repo-url>
   cd vcc
   pixi install
   ```

   This installs all dependencies (including CUDA-enabled PyTorch) into a local `.pixi/` environment.

## Setup

1. **Create your `.env` file** by copying the example:

   ```bash
   cp .env.example .env
   ```

2. **Edit `.env`** and set the paths for your system:

   ```dotenv
   DATA_DIR=/path/to/your/data
   RUN_DIR=/path/to/your/training-runs
   LOG_DIR=/path/to/your/logs
   PROJECT=/path/to/vcc
   ```

   - `DATA_DIR` -- Directory containing the input data files (see [Data Format](#data-format) below)
   - `RUN_DIR` -- Where training run outputs (W&B logs, predictions) are saved
   - `LOG_DIR` -- Where PyTorch Lightning logs are stored
   - `PROJECT` -- Path to this repository root

3. **Prepare your data** in the directory pointed to by `DATA_DIR`. See [Data Format](#data-format) for the expected structure.

## Data Format

The pipeline expects data files organized under `DATA_DIR` as follows:

```
$DATA_DIR/
├── processed-data/
│   ├── training_data-gene_ko_uint.parquet   # Binary knockout vectors (genes as columns)
│   ├── validation_data-gene_ko.parquet      # Test knockout vectors
│   ├── training_data-row_metadata.csv       # Row metadata (for embedding variant)
│   └── pert_counts_Validation.csv           # Test perturbation counts (for embedding variant)
├── cp10k-processed-data/                    # Generated by src/prepare_data.py
│   ├── training_data-counts.parquet         # Training expression, CP10K + log1p
│   ├── control_exp_data.parquet             # Control expression, CP10K + log1p
│   └── control_expression_std.pt            # Per-gene std (consistency model only)
├── log-processed-data/                      # Legacy: log1p(raw counts) -- see Known Limitations
│   ├── training_data-counts.parquet
│   └── control_exp_data.parquet
└── gene_embeddings/
    └── poincare_go_gaf_logmapped_256.parquet  # GO-derived perturbation embeddings
```

Only `processed-data/` and `gene_embeddings/` are inputs you supply. Everything under
`cp10k-processed-data/` is generated:

```bash
python src/prepare_data.py
```

### Reading `.h5ad` directly (no preprocessing)

`data=dataset_anndata` skips all of the above and reads the original AnnData files.
Controls live inside each file (`target_gene == "non-targeting"`), so it needs one
path rather than five, and CP10K + `log1p` is applied on the fly. It is numerically
identical to `dataset_cp10k` -- verified cell-by-cell against the parquets at
`atol=0` -- so the two are interchangeable.

```
$DATA_DIR/
├── 2025/
│   ├── gene_names.csv                   # 18,080 rows, NO header
│   ├── train/adata_Training.h5ad        # 221,273 cells (183,097 perturbed + 38,176 NTC)
│   ├── validation/adata_Validation.h5ad # 98,927 cells, 50 perturbations
│   └── test/adata_Test.h5ad             # 171k cells, 100 perturbations
├── vcc_2026/
│   ├── context_{A,B,C}.h5ad             # 18,400 control cells each, 18,533 genes
│   ├── gene_names.csv                   # 18,533 rows, HAS a header
│   └── pert_counts.csv                  # the 300 target genes
└── gene_embeddings/
    └── poincare_go_gaf_logmapped_256.parquet
```

The 2025 train file holds ~15.5 GB resident as sparse CSR. Listing several `.h5ad`
paths under `data_path` concatenates them (their gene panels must match), which
raises coverage from 150 to 300 perturbations at proportionally more memory.
`gene_list_path` permutes the columns into a given panel's order -- needed because
the 2025 and 2026 panels share 18,077 genes but in **completely different
positions**.

**File descriptions:**

- **Knockout vectors**: Binary parquet files where each column is a gene name. A value of 1 indicates the gene was knocked out.
- **Expression data**: Numerical parquet files with a `sample_index` column plus one column per gene. Values are counts-per-10k followed by `log1p`; `expm1(row).sum() == 1e4`.
- **Control data**: Baseline expression profiles from unperturbed cells. A control cell is paired with each perturbed cell once, when the DataModule is set up -- not re-drawn per epoch.
- **Gene embeddings**: One `gene_name` column plus 256 numeric columns, one row per gene. These are the only route by which the model can generalise to the 50 validation perturbations, none of which appear in training.

## Usage

### Training

Activate the Pixi environment and run training:

```bash
pixi shell
python src/train.py
```

Or run directly without entering the shell:

```bash
pixi run python src/train.py
```

This loads the default configuration from `config/train.yaml`, which uses:
- The standard `CellModel` with bilinear fusion
- CP10K-normalised expression + Poincare GO perturbation embeddings (`dataset_cp10k`)
- `bf16-mixed` precision on GPU
- AdamW optimizer with cosine annealing
- Composite loss (DiffExpAwareMSE + MSE)
- 100 epochs, batch size 128

Run `python src/prepare_data.py` once before the first training run.

### Selecting a Model Architecture

Override the model config using Hydra's command-line syntax:

```bash
# Attention-based model
python src/train.py model=model_attention

# FiLM conditioning model
python src/train.py model=model_film

# Simple baseline
python src/train.py model=model_simple

# Gated delta decoder, bilinear fusion, no attention
python src/train.py model=model_gated_bilinear
```

### Selecting a Data Source

```bash
# Default: CP10K parquets produced by src/prepare_data.py
python src/train.py data=dataset_cp10k

# Same numbers, read straight from the original .h5ad (no preprocessing step)
python src/train.py data=dataset_anndata
```

### Tests

```bash
pixi run python -m pytest tests/ -q
```

### Overriding Configuration

Any config value can be overridden from the command line:

```bash
# Change batch size and learning rate
python src/train.py data.datamodule.batch_size=64 model.optimizer.lr=1e-4

# Change number of epochs
python src/train.py trainer.max_epochs=50

# Use a different GPU
python src/train.py trainer.devices=[1]

# Disable W&B logging (log locally only)
python src/train.py logging.wandb.offline=True

# Resume from a checkpoint
python src/train.py ckpt_path=/path/to/checkpoint.ckpt
```

### Hyperparameter Tuning

Run Optuna-based hyperparameter search:

```bash
python src/tuning/tune_attention.py
```

This uses `config/tune.yaml` and runs 200 trials by default, optimizing validation loss with TPE sampling. Results are stored in a journal file under `RUN_DIR`.

### Notebooks

Jupyter notebooks for data exploration and model evaluation are in `notebook/`. Launch them with:

```bash
pixi shell
jupyter notebook
```

| Notebook | Description |
|----------|-------------|
| `preliminary-data-analysis.ipynb` | Exploratory analysis of the perturbation dataset |
| `evaluation.ipynb` | Evaluate trained models and compute metrics |
| `gene_expression_embeddings.ipynb` | Analyze and visualize gene embeddings |
| `model_debugging.ipynb` | Debug model outputs and intermediate representations |
| `contrastive-loss.ipynb` | Explore contrastive loss function behavior |

## Model Architectures

The project includes several neural network architectures, all configured via YAML:

- **CellModel** (`model.yaml`): The default architecture. Processes knockout vectors and expression profiles through separate MLPs, fuses them (bilinear by default), and decodes to predicted expression.
- **Attention Model** (`model_attention.yaml`): Uses multi-head cross-attention for fusion instead of bilinear. Includes a differential expression aware variant.
- **Gated Bilinear Model** (`model_gated_bilinear.yaml`): Bilinear fusion with the gated delta decoder and **no attention**. Predicts `y = control_exp + gate * (raw_pred - control_exp)`, so the network outputs a masked residual rather than an absolute profile. `GateSparsityLoss` holds the gate near the measured DE rate; without it the gate drifts to ~0.87 and the expression collapses back to `y = raw_pred`. Note that `model_attention` runs its attention module *regardless* of `fusion_type`, so setting `fusion_type: bilinear` there does not remove it -- this config does.
- **FiLM Model** (`model_film.yaml`): Uses Feature-wise Linear Modulation -- the perturbation signal modulates the expression processing via learned scale and shift parameters.
- **Simple Model** (`model_simple.yaml`): Lightweight baseline for comparison.

All models support configurable:
- Layer sizes and depths for each processing stage
- Activation functions (ReLU, GELU, SiLU, SwiGLU)
- Dropout rates
- Residual connections
- Fusion strategies (sum, product, concat, bilinear, cross-attention)

## Evaluation

The pipeline uses a **gene-level train/test split** rather than a random sample split. This means some genes are held out entirely during training, providing an out-of-distribution (OOD) evaluation of the model's ability to generalize to unseen perturbations.

After training, `src/train.py` automatically generates predictions and UMAP visualizations of the results.

### Scoring against the competition metrics

`val/loss` is an MSE proxy, not the challenge metric. To score a checkpoint on the
held-out 2025 splits with `cell-eval` (perturbation discrimination, DE overlap, MAE):

```bash
python scripts/validate_2025.py --ckpt <run>/VCC-epoch=NN_step=NNNN.ckpt --split both
```

This predicts the split, inverts the predictions to raw counts, appends the real
non-targeting cells (the 2025 scorer requires them), and runs `cell-eval run
--profile vcc`. Budget ~30 min for validation and ~2 h for test; the DE computation
is CPU-bound and dominates.

**Read `discrimination_score_l1` against the right floor.** It is `1 - rank/n_perts`,
so **random is ~0.5**, not 0. A model that predicts the same effect for every
perturbation produces one distance ordering for all of them and scores ~0.51 *by
construction*. Numbers near 0.5 mean the model is not distinguishing perturbations,
however good its MSE looks.

### Building a 2026 submission

```bash
python scripts/predict_2026.py --ckpt <run>/checkpoint.ckpt      # -> results/2026/*.h5ad
pixi run vcc prep -i results/2026/<file>.h5ad \
    -g $DATA_DIR/vcc_2026/gene_names.csv \
    --perts $DATA_DIR/vcc_2026/pert_counts.csv --context-col context
pixi run vcc submit results/2026/<file>.prep.vcc
```

`predict_2026.py` maps the two gene panels **by name**, predicts the 18,077 shared
genes, and copies the control cell's own counts for the 456 genes the model cannot
predict. Each of the 400 cells per (target, context) comes from a different control
cell, so the cells are non-degenerate.

Use `vcc prep`, **not** `cell-eval prep` -- see Known Limitations.

## Experiment Tracking

Training is logged to [Weights & Biases](https://wandb.ai) by default. To use it:

1. Create a free W&B account
2. Run `wandb login` and paste your API key
3. Runs will appear under the project `VirtualCellChallenge`

To disable W&B and log only locally:

```bash
python src/train.py logging.wandb.offline=True
```

## Known Limitations

This project was built for scientific exploration, not as a production package.
Read this section before trusting any result from it.

### Fixed, but relevant to older runs

Runs and submissions produced before these fixes are affected:

- **No library-size normalisation.** Training data was `log1p(raw UMI counts)`.
  On that scale `corr(log total UMI, mean expression) = 0.993` -- a cell's
  profile was almost entirely its sequencing depth, and only ~2% of per-gene
  variance was between-perturbation signal. Since each perturbed cell is paired
  with a randomly drawn control cell, that depth term was unpredictable in
  principle. `src/prepare_data.py` now applies CP10K + `log1p`.
- **Two loss terms contributed exactly zero gradient.** `PerturbationSimilarityLoss`
  built on `torchmetrics.spearman_corrcoef`, whose ranks come from `argsort`, so
  its output had no `grad_fn` at all; and a `torch.sign` direction term inside
  `DiffExpAwareMSELoss` was piecewise constant. Both carried non-zero weights.
  Both are removed; `tests/test_losses.py` guards against reintroducing them.
- **`CompositeLoss` held its children in a plain list**, so they were invisible to
  `.parameters()` and `.to(device)`. Any loss with a parameter or buffer never
  trained. This is why several configs used to hardcode `device: cuda:3`.
- **The contrastive loss was a no-op with the configured embedding.**
  `quantiles-train_expression.parquet` has no negative pairwise cosines, so
  `WeightedContrastiveLoss` returned exactly zero. The default now uses the
  Poincare GO embedding.
- **DE thresholds are scale-dependent.** They are calibrated for CP10K
  (`threshold: 0.4`, ~10% of gene-cells called DE). The pre-normalisation values
  of 1.0-1.5 select 0.02% on this scale, silently disabling every DE-aware loss.
  Re-run `python src/prepare_data.py --calibrate` if you change the normalisation.

- **Submission scale.** Predictions come out of the model in log-CP10K, and older
  submissions (`results/prediction_*.h5ad`) were written on that scale with no
  `expm1` -- non-integral values capped around 6.3. **2026 scores in counts space.**
  `scripts/predict_counts.py` and `scripts/predict_2026.py` invert properly:
  rescale `expm1(y_pred)` to `target_sum`, then multiply by the paired control
  cell's library size. The rescale is load-bearing -- the model is not constrained
  to emit a valid CP10K profile, and skipping it made predicted depth 37% too low.

### Open

- **Use `vcc prep`, not `cell-eval prep`, for 2026.** `cell-eval` 0.5.43 is 2025-era
  and breaks four ways on 2026 input: `EXPECTED_GENE_DIM` is 18080 not 18533;
  `MAX_CELL_DIM` is 100000 against a 360000-cell submission; it reads the gene list
  with `has_header=False` (the 2026 `gene_names.csv` *has* a header); and
  `run_prep` parses `--ntc-name`/`--output-pert-col`/`--output-celltype-col` but
  never forwards them to `strip_anndata`, so the NTC check cannot be relaxed and
  `context` is silently renamed `celltype`. Worst of all it **log-normalises
  unconditionally**, destroying the counts the 2026 scorer wants. `vcc prep`
  defaults to `--require-counts` and `--reject-controls` and verifies targets
  against the official per-context list.
- **Plain MSE cannot see perturbation identity.** Measured on the 30 held-out genes:
  predicting the paired control cell gives MSE 0.0558, predicting a *constant*
  global control-mean profile gives 0.0285, and the per-perturbation oracle gives
  0.0277. The entire perturbation-specific signal is worth **0.0008 MSE -- 2.8% of
  the achievable reduction**; the other 97% is denoising. A trained model at
  `val/mse` 0.032 is therefore *beaten by a constant*. Optimise the DE-aware losses
  or supervise the gate, and judge runs on `cell-eval`, not `val/loss`.
- **The gate learns one static mask.** With `GateSparsityLoss` the gate settles
  around 0.42 and is bimodal (~41% of genes hard-closed), but its per-gene standard
  deviation *across cells* is 0.0082 -- essentially the same mask for every
  perturbation. Sparsity constrains *how many* genes may move, not *which ones* for
  a given knockout; that needs per-perturbation DE supervision.
- **`src/prepare_data.py` has `TARGET_SUM = 5e4`,** but the parquets in
  `cp10k-processed-data/` were written at `1e4`, which is what the file's own
  docstring, the directory name, and the `threshold: 0.4` calibration (~10% DE) all
  assume. The `5e4` is an edit that was never re-run. `dataset_anndata` defaults to
  `1e4` to match the data actually on disk.
- **`config/data/dataset_embedding.yaml` has a broken path.**
  `test_exp_data_path` points at `log-processed-data/control_exp_data_uint.parquet`,
  which has never existed; the file lives under `processed-data/`.
- **No early stopping.** `config/callbacks/` has no `EarlyStopping`, and
  `max_epochs: 100` is far past convergence -- a representative run's best
  `val/loss` came at epoch 21 and the remaining 78 epochs only widened the
  train/val gap. Add it, or take the `save_top_k` checkpoint rather than `last.ckpt`.
- **`data=dataset` (the one-hot KO path) is dead.** Its knockout matrix is
  18,080-wide while every model expects a 256-dim embedding, and
  `processed-data/validation_data-gene_ko.parquet` is malformed: 183,097 rows of
  which only the first 60,751 carry a perturbation and 122,346 are all-zero
  padding. Use `dataset_cp10k`.
- **`src/tuning/tune_attention.py` does not run.** It sets pre-refactor
  `hidden_layers`/`max_lr` keys, samples `weight_decay` under the name
  `"learning_rate"` (so the two were always equal), and
  `conf.get("optuna.sampler")` returns `None`, meaning the configured sampler was
  never used. Results from that study should not be trusted.
- **Architectural caveats.** The attention models treat each latent *dimension* as
  a token via `Linear(1, d)` with no positional encoding, so attention is
  permutation-equivariant over features. `SwiGLU` instances are shared across
  layers of a `ProcessingNN`, tying their weights. The consistency model adds one
  noise vector broadcast across the whole batch rather than per cell.
