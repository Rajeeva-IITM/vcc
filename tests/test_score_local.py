"""Guards for the local scorer's two ways of being silently wrong.

1. Scoring the wrong perturbations. The datamodule's held-out set is NOT the 2025
   challenge validation split, and training reads all of `adata_2025_all.h5ad`, so
   scoring on the challenge split would grade the model on data it trained on.
2. Handing `cell-eval2` a file it will reject after a full Wilcoxon pass rather than
   at the door.
"""

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix

from scripts.score_local import assert_scoreable, held_out_genes

# Verified during planning against the real 300 categories of adata_2025_all at
# seed=42, test_size=0.2. If this list changes, every previously recorded local score
# was measured on a different set and is no longer comparable.
EXPECTED = """
ARID3A ATP1B1 ATP6V0C BIRC2 BRPF1 CALM3 CENPB CSK CSNK1E CTNNB1 CXCL12 DHX36 DOT1L
FDPS HMGXB4 HTATSF1 INSIG1 IRF3 KIF20A KLHDC2 LAP3 MAP1B MAST2 MAT2A MAU2 MED13
METTL3 MORN2 NDUFB6 NEK2 NREP NSDHL OXA1L PBX3 PHF12 PHF14 POLB PRCP PSME1 RAB3B
RFX5 RNF40 SAFB2 SBDS SET SHPRH SIN3B SIX4 SLC5A6 SMARCB1 STAG2 SUPT4H1 SUPV3L1 SV2A
THAP11 TRAPPC6A UGDH UQCRQ ZNF581 ZNF598
""".split()

DATA = "/home/rajeeva/Project/vcc_data/2025/adata_2025_all.h5ad"


def _categories():
    """Perturbation categories of the 2025 data."""
    from scripts.score_local import read_obs_perturbations

    return read_obs_perturbations(DATA)[1]


@pytest.mark.skipif(
    not __import__("os").path.exists(DATA), reason="2025 data not present"
)
def test_held_out_set_is_stable():
    """The held-out split matches the pinned 60-gene list."""
    held = held_out_genes(_categories(), test_size=0.2, seed=42)
    assert list(held) == sorted(EXPECTED)
    assert len(held) == 60


@pytest.mark.skipif(
    not __import__("os").path.exists(DATA), reason="2025 data not present"
)
def test_held_out_set_is_not_the_challenge_split():
    """The leakage guard. `validate_2025.py` scores the challenge splits; we must not."""
    import polars as pl

    held = set(held_out_genes(_categories(), test_size=0.2, seed=42))
    challenge = set(
        pl.read_csv(
            "/home/rajeeva/Project/vcc_data/2025/validation/pert_counts_Validation.csv"
        )["target_gene"].to_list()
    )
    assert len(held & challenge) == 12, (
        "the held-out set drifted toward the challenge split; if these ever coincide "
        "the local score is measuring training data"
    )


def test_held_out_set_is_a_pure_function_of_its_inputs():
    """The held-out split depends only on the gene set, fraction and seed."""
    cats = np.array([f"G{i:03d}" for i in range(300)])
    a = held_out_genes(cats, 0.2, 42)
    b = held_out_genes(cats, 0.2, 42)
    c = held_out_genes(cats[::-1], 0.2, 42)  # order must not matter
    assert list(a) == list(b) == list(c)
    assert list(a) != list(held_out_genes(cats, 0.2, 43))


def _frame(n=6, g=5, label="A"):
    """A small valid (X, obs, genes) triple with a control group."""
    X = csr_matrix(np.array([[1, 0, 2, 0, 3]] * n, dtype=np.int32))
    obs = pd.DataFrame({"target_gene": [label] * (n - 2) + ["non-targeting"] * 2})
    return X, obs, np.array([f"g{i}" for i in range(g)])


def test_assert_scoreable_passes_a_well_formed_pair():
    """A well-formed submission passes."""
    assert_scoreable(*_frame(), side="test")


def test_assert_scoreable_rejects_fractional_counts():
    """Non-integer counts are rejected."""
    X, obs, genes = _frame()
    X = X.astype(np.float64)
    X.data[0] = 1.5
    with pytest.raises(AssertionError, match="whole numbers"):
        assert_scoreable(X, obs, genes, side="test")


def test_assert_scoreable_rejects_a_missing_control_group():
    """A submission without non-targeting cells is rejected."""
    X, obs, genes = _frame()
    obs["target_gene"] = "A"
    with pytest.raises(AssertionError, match="non-targeting"):
        assert_scoreable(X, obs, genes, side="test")


def test_assert_scoreable_rejects_cells_over_the_count_cap():
    """A cell above the per-cell count cap is rejected."""
    X, obs, genes = _frame()
    X = X.astype(np.int64)
    X.data[0] = 2_000_000
    with pytest.raises(AssertionError, match="max_counts_per_cell"):
        assert_scoreable(X, obs, genes, side="test")
