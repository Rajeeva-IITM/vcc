"""KernelFiLM structural / backward-compatibility tests.

The one real risk of the `proj_dim` knob is checkpoint loading: `predict_2026.py` and
`predict_counts.py` load with the default `strict=True`, so the identity-metric path
(`proj_dim=None`) must add no parameter and keep `keys` at `(K, input_size)` -- otherwise every
pre-`proj_dim` checkpoint fails to load. These tests pin that invariant, mirroring the
"stays out of the state_dict" pattern in tests/test_losses.py.
"""

import torch

from src.models.components.flow_model import KernelFiLM


def test_default_kernelfilm_statedict_is_unchanged():
    """proj_dim=None: no query_proj key, keys in input space -> old checkpoints still load."""
    kf = KernelFiLM(input_size=256, output_size=512, num_anchors=128)
    keys = set(kf.state_dict())
    assert not any(k.startswith("query_proj") for k in keys), keys
    assert kf.keys.shape == (128, 256)
    assert kf.proj_dim is None


def test_proj_dim_adds_projection_and_shrinks_keys():
    """proj_dim=d: a query_proj weight (d, input_size) appears and keys move to the d-space."""
    kf = KernelFiLM(input_size=256, output_size=512, num_anchors=128, proj_dim=64)
    sd = kf.state_dict()
    assert "query_proj.weight" in sd
    assert tuple(sd["query_proj.weight"].shape) == (64, 256)
    assert kf.keys.shape == (128, 64)


def test_default_statedict_round_trips_strict():
    """A proj_dim=None module loads another proj_dim=None module's state_dict under strict=True."""
    src = KernelFiLM(input_size=256, output_size=512, num_anchors=64)
    with torch.no_grad():
        src.values.copy_(torch.randn_like(src.values))
    dst = KernelFiLM(input_size=256, output_size=512, num_anchors=64)
    dst.load_state_dict(src.state_dict())  # strict=True by default
    assert torch.equal(dst.values, src.values)


def test_proj_dim_is_zero_init_identity():
    """Zero-value init -> zero output, so FiLM starts at passthrough regardless of proj_dim."""
    kf = KernelFiLM(input_size=256, output_size=512, num_anchors=64, proj_dim=32).eval()
    out = kf(torch.randn(8, 256))
    assert out.shape == (8, 512)
    assert torch.count_nonzero(out) == 0


def test_proj_dim_output_stays_in_convex_hull():
    """The anti-overshoot (bounded-magnitude) property survives the learned metric."""
    kf = KernelFiLM(input_size=128, output_size=64, num_anchors=48, proj_dim=16).eval()
    with torch.no_grad():
        kf.values.copy_(torch.randn_like(kf.values))
    out = kf(torch.randn(16, 128))
    lo, hi = kf.values.min(dim=0).values, kf.values.max(dim=0).values
    assert (out >= lo - 1e-5).all() and (out <= hi + 1e-5).all()
