"""Pins `src.utils.predict` to the arithmetic it was extracted from.

`scripts/predict_2026.py` produced every submission we have scored, so the extraction
must be byte-identical, not merely equivalent-looking. The reference implementations
below are transcribed verbatim from that script as it stood before the refactor
(lines 202-261); if they and the helpers ever disagree, the local scorer has silently
started measuring a different model than the one we submit.
"""

import numpy as np
import pytest

from src.utils.predict import finalize_counts, to_counts, to_model_space


def _reference_to_model_space(ctrl_model, norm_cols, target_sum):
    """predict_2026.py as it was, verbatim."""
    lib = ctrl_model.sum(axis=1, keepdims=True)
    lib[lib == 0] = 1.0
    depth = ctrl_model[:, norm_cols].sum(axis=1, keepdims=True)
    depth[depth == 0] = 1.0
    exp_vec = np.log1p(ctrl_model / depth * target_sum)
    return exp_vec, lib


def _reference_to_counts(pred, lib):
    """predict_2026.py as it was, verbatim."""
    counts_model = np.expm1(pred, dtype=np.float64)
    mass = counts_model.sum(axis=1, keepdims=True)
    mass[mass == 0] = 1.0
    counts_model *= lib / mass
    return counts_model


def _fixture(seed, n_cells=8, n_genes=40, n_norm=25):
    """Random Poisson control counts and a random norm axis."""
    rng = np.random.default_rng(seed)
    ctrl = rng.poisson(3.0, size=(n_cells, n_genes)).astype(np.float32)
    norm_cols = np.sort(rng.choice(n_genes, n_norm, replace=False))
    return ctrl, norm_cols


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_to_model_space_matches_the_original(seed):
    """to_model_space reproduces the pre-extraction arithmetic exactly."""
    ctrl, norm_cols = _fixture(seed)
    got_exp, got_lib = to_model_space(ctrl.copy(), norm_cols, 1e4)
    want_exp, want_lib = _reference_to_model_space(ctrl.copy(), norm_cols, 1e4)
    assert np.array_equal(got_exp, want_exp)
    assert np.array_equal(got_lib, want_lib)


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_to_counts_matches_the_original(seed):
    """to_counts reproduces the pre-extraction arithmetic exactly."""
    ctrl, norm_cols = _fixture(seed)
    _, lib = to_model_space(ctrl.copy(), norm_cols, 1e4)
    pred = (
        np.random.default_rng(seed + 100)
        .uniform(0, 6, size=ctrl.shape)
        .astype(np.float32)
    )
    assert np.array_equal(
        to_counts(pred, lib.copy()), _reference_to_counts(pred, lib.copy())
    )


def test_to_counts_preserves_the_control_library_size():
    """The whole point of `lib`: a predicted cell carries its control's depth."""
    ctrl, norm_cols = _fixture(7)
    _, lib = to_model_space(ctrl.copy(), norm_cols, 1e4)
    pred = np.random.default_rng(7).uniform(0, 6, size=ctrl.shape).astype(np.float32)
    counts = to_counts(pred, lib)
    assert np.allclose(counts.sum(axis=1), lib.ravel())


def test_zero_depth_cells_do_not_divide_by_zero():
    """An all-zero cell yields finite inputs and a unit library size."""
    ctrl, norm_cols = _fixture(11)
    ctrl[0] = 0.0  # an empty cell: both lib and depth would be 0
    exp_vec, lib = to_model_space(ctrl, norm_cols, 1e4)
    assert np.isfinite(exp_vec).all()
    assert lib[0] == 1.0


def test_off_axis_genes_do_not_scale_the_input():
    """`depth` is summed over `norm_cols` only -- mass outside it must not rescale."""
    n_genes, norm_cols = 40, np.arange(25)
    ctrl = np.ones((4, n_genes), dtype=np.float32)
    base, _ = to_model_space(ctrl.copy(), norm_cols, 1e4)
    loaded = ctrl.copy()
    loaded[:, 30:] = 500.0  # piles mass strictly off the axis
    after, _ = to_model_space(loaded, norm_cols, 1e4)
    assert np.allclose(base[:, :25], after[:, :25])


def test_finalize_counts_yields_integer_csr_without_stored_zeros():
    """finalize_counts rounds, clips negatives and stores no zeros."""
    dense = np.array([[1.4, 0.0, -3.0, 2.6], [0.2, 9.9, 0.0, 0.0]])
    block = finalize_counts(dense)
    assert block.dtype == np.int32
    assert np.array_equal(block.toarray(), np.array([[1, 0, 0, 3], [0, 10, 0, 0]]))
    assert (block.data == 0).sum() == 0
    assert block.data.min() >= 0
