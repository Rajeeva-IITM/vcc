"""Join the three 2025 splits into one .h5ad.

The 2025 data ships as three files whose perturbation sets are disjoint -- 150
train, 50 validation, 100 test -- so together they cover 300 perturbations
instead of the 150 a model trained on the train split alone ever sees.

The catch: **the 38,176 non-targeting cells are the same cells in all three
files**, identical down to their barcodes. Concatenating naively yields 491,046
cells in which every control appears three times, making controls 27.6% of the
dataset instead of 9.2% and tripling their weight in any control-paired sampling.
This script keeps one copy, giving 414,694 cells.

Output layout (raw integer counts, CSR, 18,080 genes in the shared panel order):

    414,694 cells = 376,518 perturbed (300 perturbations) + 38,176 controls

``obs`` carries the original ``target_gene``, ``guide_id`` and ``batch``, plus a
``split`` column recording provenance -- ``train`` / ``validation`` / ``test``
for perturbed cells and ``control`` for the shared non-targeting block -- so the
original splits can be reconstructed exactly.

Usage
-----
    python scripts/merge_2025.py [-o $DATA_DIR/2025/adata_2025_all.h5ad]
"""

import argparse
import gc
import os
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from rich.console import Console
from scipy.sparse import csr_matrix, vstack

console = Console()

CONTROL_LABEL = "non-targeting"

SPLITS = (
    ("train", "2025/train/adata_Training.h5ad"),
    ("validation", "2025/validation/adata_Validation.h5ad"),
    ("test", "2025/test/adata_Test.h5ad"),
)

OBS_KEEP = ("target_gene", "guide_id", "batch")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("-o", "--output", default=None, help="output .h5ad")
    p.add_argument(
        "--compress",
        action="store_true",
        help="gzip the output; much smaller but slow to write and to read back",
    )
    return p.parse_args()


def main() -> None:
    """Merge the three splits, de-duplicating the shared control cells."""
    args = parse_args()
    load_dotenv()
    data_dir = Path(os.environ["DATA_DIR"])
    out = Path(args.output or data_dir / "2025/adata_2025_all.h5ad")

    blocks: list[csr_matrix] = []
    obs_parts: list[pd.DataFrame] = []
    control_block: csr_matrix | None = None
    control_obs: pd.DataFrame | None = None
    control_names: set[str] | None = None
    genes: list[str] | None = None

    for split, rel in SPLITS:
        path = data_dir / rel
        console.log(f"[bold]{split}[/bold]: {path.name}")
        adata = ad.read_h5ad(path)

        var_names = [str(g) for g in adata.var_names]
        if genes is None:
            genes = var_names
        elif var_names != genes:
            raise ValueError(
                f"{path.name} has a different gene panel or order than "
                f"{SPLITS[0][1]} -- the splits must share one panel to be joined"
            )

        labels = adata.obs["target_gene"].to_numpy().astype(str)
        is_ctrl = labels == CONTROL_LABEL
        X = csr_matrix(adata.X)

        # Controls are byte-identical across the three files; keep the first
        # copy and assert the rest match rather than trusting the claim.
        names = set(adata.obs_names[is_ctrl])
        if control_block is None:
            control_block = X[np.flatnonzero(is_ctrl)]
            control_obs = adata.obs.loc[is_ctrl, list(OBS_KEEP)].copy()
            control_obs["split"] = "control"
            control_names = names
            console.log(
                f"  kept {is_ctrl.sum():,} control cells (shared by all splits)"
            )
        elif names != control_names:
            raise ValueError(
                f"{path.name} has a different control cell set than "
                f"{SPLITS[0][1]}; de-duplication by barcode is not valid here"
            )
        else:
            console.log(f"  dropped {is_ctrl.sum():,} duplicate control cells")

        pert = np.flatnonzero(~is_ctrl)
        blocks.append(X[pert])
        part = adata.obs.iloc[pert][list(OBS_KEEP)].copy()
        part["split"] = split
        obs_parts.append(part)
        console.log(
            f"  kept {len(pert):,} perturbed cells, "
            f"{part['target_gene'].nunique()} perturbations"
        )

        del adata, X
        gc.collect()

    assert control_block is not None and control_obs is not None

    console.log("Stacking")
    X = vstack([*blocks, control_block], format="csr")
    obs = pd.concat([*obs_parts, control_obs])
    del blocks, obs_parts, control_block
    gc.collect()

    n_pert = int((obs["target_gene"] != CONTROL_LABEL).sum())
    n_ctrl = int((obs["target_gene"] == CONTROL_LABEL).sum())

    assert X.shape[0] == obs.shape[0], (X.shape, obs.shape)
    assert obs.index.is_unique, "duplicate cell barcodes after the merge"
    assert n_ctrl == 38_176, n_ctrl
    assert obs.loc[obs["split"] != "control", "target_gene"].nunique() == 300
    assert np.all(X.data == np.rint(X.data)), "expected raw integer counts"

    console.log(f"Writing {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    ad.AnnData(X=X, obs=obs, var=pd.DataFrame(index=genes)).write_h5ad(
        out, compression="gzip" if args.compress else None
    )

    console.log(
        f"[green]wrote[/green] {out}  {X.shape[0]:,} x {X.shape[1]:,} | "
        f"nnz {X.nnz:,} | {n_pert:,} perturbed + {n_ctrl:,} control | "
        f"300 perturbations"
    )


if __name__ == "__main__":
    main()
