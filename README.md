# Virtual Cell Challenge (VCC)

A deep learning pipeline for modeling gene expression changes under genetic perturbations (gene knockouts). Given a control (unperturbed) gene expression profile and a perturbation indicator, the model predicts the resulting gene expression.

This project was built for scientific analysis and exploration, not as a production-ready package.

## Project Structure

```
vcc/
├── src/
│   ├── train.py                         # Main training entry point (Hydra)
│   ├── data/
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
│   │   ├── dataset.yaml                 # Standard dataset config
│   │   └── dataset_embedding.yaml       # Embedding-based dataset config
│   ├── model/                           # Model architecture configs
│   │   ├── model.yaml                   # Default CellModel (bilinear fusion)
│   │   ├── model_attention.yaml         # Attention-based model
│   │   ├── model_film.yaml              # FiLM conditioning model
│   │   ├── model_simple.yaml            # Simple baseline
│   │   └── best_model_attention.yaml    # Best model from Optuna tuning
│   ├── trainer/trainer.yaml             # PyTorch Lightning trainer settings
│   ├── logging/wandb.yaml               # Weights & Biases logging
│   └── callbacks/                       # Lightning callbacks (checkpointing, early stopping, etc.)
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
   RESULT_DIR=/path/to/your/results
   ```

   - `DATA_DIR` -- Directory containing the input data files (see [Data Format](#data-format) below)
   - `RUN_DIR` -- Where training run outputs (W&B logs, predictions) are saved
   - `LOG_DIR` -- Where PyTorch Lightning logs are stored
   - `PROJECT` -- Path to this repository root
   - `RESULT_DIR` -- Where final results are saved

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
├── log-processed-data/
│   ├── training_data-counts.parquet         # Training expression data (log-transformed)
│   ├── control_exp_data.parquet             # Control/baseline expression samples
│   └── control_exp_data_uint.parquet        # Control expression (uint, for embedding variant)
└── gene_embeddings/
    └── quantiles-train_expression.parquet   # Pre-computed gene embeddings (for embedding variant)
```

**File descriptions:**

- **Knockout vectors**: Binary parquet files where each column is a gene name. A value of 1 indicates the gene was knocked out.
- **Expression data**: Numerical parquet files with gene names as columns. Values represent (log-transformed) expression levels.
- **Control data**: Baseline expression profiles from unperturbed cells. During training, a control sample is randomly drawn for each perturbation.
- **Gene embeddings** (optional): Pre-computed embeddings used by the `dataset_embedding` data variant.

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
- `bf16-mixed` precision on GPU
- AdamW optimizer with cosine annealing
- Composite loss (DiffExpAwareMSE + PerturbationSimilarity + MSE)
- 100 epochs, batch size 128

### Selecting a Model Architecture

Override the model config using Hydra's command-line syntax:

```bash
# Attention-based model
python src/train.py model=model_attention

# FiLM conditioning model
python src/train.py model=model_film

# Simple baseline
python src/train.py model=model_simple

# Best tuned attention model
python src/train.py model=best_model_attention
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

## Experiment Tracking

Training is logged to [Weights & Biases](https://wandb.ai) by default. To use it:

1. Create a free W&B account
2. Run `wandb login` and paste your API key
3. Runs will appear under the project `VirtualCellChallenge`

To disable W&B and log only locally:

```bash
python src/train.py logging.wandb.offline=True
```
