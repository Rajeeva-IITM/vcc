"""PerturbationGroupedBatchSampler: shape, grouping, weighting, epoch variation.

The sampler is what makes the pseudobulk-per-perturbation losses trainable: the default
cell-shuffled sampler draws ~one cell per perturbation at ~19k perturbations, so a
``min_group_size=2`` pseudobulk term is inert. These tests pin the properties that matter
for that -- each batch is a fixed number of perturbations x a fixed number of their cells,
drawn on the source-weighted marginal -- without needing any data on disk.
"""

import numpy as np
import torch

from src.data.vcc_anndata_module import PerturbationGroupedBatchSampler


def _disjoint_groups(n_groups=50, size=64):
    """Groups whose index ranges are disjoint, so index -> group is unambiguous."""
    group_indices = [np.arange(g * size, g * size + size) for g in range(n_groups)]
    of_index = {}
    for g, idx in enumerate(group_indices):
        for i in idx:
            of_index[int(i)] = g
    return group_indices, of_index


def _uniform_weight(n):
    return torch.full((n,), 1.0 / n, dtype=torch.double)


def test_batch_shape_and_count():
    """Every batch is perts_per_batch x cells_per_pert; there are num_batches of them."""
    gi, _ = _disjoint_groups()
    s = PerturbationGroupedBatchSampler(
        gi,
        _uniform_weight(len(gi)),
        perts_per_batch=8,
        cells_per_pert=16,
        num_batches=10,
        seed=0,
    )
    batches = list(s)
    assert len(batches) == 10 == len(s)
    for b in batches:
        assert len(b) == 8 * 16


def test_each_batch_is_distinct_perts_with_fixed_cells():
    """A batch holds exactly perts_per_batch distinct groups, cells_per_pert cells each."""
    gi, of_index = _disjoint_groups()
    s = PerturbationGroupedBatchSampler(
        gi,
        _uniform_weight(len(gi)),
        perts_per_batch=8,
        cells_per_pert=16,
        num_batches=20,
        seed=1,
    )
    for b in s:
        groups = [of_index[i] for i in b]
        counts = {g: groups.count(g) for g in set(groups)}
        assert len(counts) == 8, "a batch did not hold 8 distinct perturbations"
        assert all(c == 16 for c in counts.values()), "cells-per-pert not honoured"


def test_all_indices_are_valid():
    gi, of_index = _disjoint_groups()
    s = PerturbationGroupedBatchSampler(
        gi, _uniform_weight(len(gi)), 8, 16, num_batches=15, seed=2
    )
    for b in s:
        assert all(i in of_index for i in b)


def test_small_group_samples_with_replacement():
    """A group with fewer cells than cells_per_pert still contributes cells_per_pert."""
    # One tiny group (3 cells) + larger ones; ask for 16 cells each.
    gi = [np.arange(0, 3)] + [np.arange(3 + g * 64, 3 + g * 64 + 64) for g in range(10)]
    s = PerturbationGroupedBatchSampler(
        gi,
        _uniform_weight(len(gi)),
        perts_per_batch=len(gi),
        cells_per_pert=16,
        num_batches=1,
        seed=3,
    )
    (b,) = list(s)
    tiny = [i for i in b if i < 3]
    assert len(tiny) == 16, "tiny group did not fill cells_per_pert via replacement"
    assert set(tiny) <= {0, 1, 2}


def test_weighting_biases_group_frequency():
    """A heavily weighted group is drawn far more than a light one over many batches."""
    gi, of_index = _disjoint_groups(n_groups=20, size=64)
    w = torch.full((20,), 1.0, dtype=torch.double)
    w[0] = 100.0  # group 0 heavily favoured
    w = w / w.sum()
    s = PerturbationGroupedBatchSampler(
        gi, w, perts_per_batch=5, cells_per_pert=8, num_batches=200, seed=4
    )
    freq = np.zeros(20)
    for b in s:
        for g in {of_index[i] for i in b}:
            freq[g] += 1
    assert freq[0] > 3 * freq[1:].mean(), "sampling ignored the group weights"


def test_epochs_differ_but_run_is_reproducible():
    """Two epochs (two __iter__ calls) differ; a fresh sampler with the same seed repeats."""
    gi, _ = _disjoint_groups()
    s = PerturbationGroupedBatchSampler(
        gi, _uniform_weight(len(gi)), 8, 16, num_batches=5, seed=7
    )
    epoch1 = list(s)
    epoch2 = list(s)
    assert epoch1 != epoch2, "successive epochs produced identical batches"

    s2 = PerturbationGroupedBatchSampler(
        gi, _uniform_weight(len(gi)), 8, 16, num_batches=5, seed=7
    )
    assert list(s2) == epoch1, "same seed did not reproduce the first epoch"


def test_perts_per_batch_capped_at_group_count():
    """Asking for more perturbations than exist clamps rather than erroring."""
    gi, _ = _disjoint_groups(n_groups=6, size=32)
    s = PerturbationGroupedBatchSampler(
        gi,
        _uniform_weight(6),
        perts_per_batch=32,
        cells_per_pert=8,
        num_batches=3,
        seed=0,
    )
    assert s.perts_per_batch == 6
    for b in s:
        assert len(b) == 6 * 8


def test_rejects_bad_config():
    gi, _ = _disjoint_groups(n_groups=4)
    import pytest

    with pytest.raises(ValueError):
        PerturbationGroupedBatchSampler(gi, _uniform_weight(4), 0, 8, 5)
    with pytest.raises(ValueError):
        PerturbationGroupedBatchSampler([], torch.tensor([]), 4, 8, 5)
