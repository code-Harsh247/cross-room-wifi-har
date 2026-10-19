"""Tests for the SSL augmentations (Task 2, gate T2)."""
from __future__ import annotations

import pytest
import torch

from xroom import augment as A
from xroom.config import load_task2_config

CFG = load_task2_config()["augment"]


def _x(b=16, t=120, c=242, seed=0):
    return torch.randn(b, t, c, generator=torch.Generator().manual_seed(seed))


def _g(seed=0):
    return torch.Generator().manual_seed(seed)


FUNCS = {
    "crop_resize": A.crop_resize,
    "noise": A.gaussian_noise,
    "subcarrier_mask": A.subcarrier_mask,
    "time_mask": A.time_mask,
    "gain": A.row_gain,
    "format": A.format_jitter,
}


@pytest.mark.parametrize("name", FUNCS)
def test_shape_dtype_finite(name):
    x = _x()
    y = FUNCS[name](x, _g(), **{**CFG[name], "p": 1.0})
    assert y.shape == x.shape and y.dtype == x.dtype and torch.isfinite(y).all()
    assert not torch.equal(y, x)


@pytest.mark.parametrize("name", FUNCS)
def test_identity_at_p0(name):
    x = _x()
    assert torch.equal(FUNCS[name](x, _g(), **{**CFG[name], "p": 0.0}), x)


@pytest.mark.parametrize("name", FUNCS)
def test_deterministic(name):
    x = _x()
    a = FUNCS[name](x, _g(5), **CFG[name])
    b = FUNCS[name](x, _g(5), **CFG[name])
    assert torch.equal(a, b)


def test_full_pipeline_identity_and_determinism():
    x = _x()
    off = {k: {**v, "p": 0.0} for k, v in CFG.items()}
    assert torch.equal(A.augment(x, _g(), off), x)
    v1, v2 = A.make_views(x, _g(1), CFG)
    w1, w2 = A.make_views(x, _g(1), CFG)
    assert torch.equal(v1, w1) and torch.equal(v2, w2) and not torch.equal(v1, v2)


@pytest.mark.parametrize("name,axis", [("subcarrier_mask", 2), ("time_mask", 1)])
def test_masked_fraction_in_configured_range(name, axis):
    x = torch.ones(64, 150, 242)
    y = FUNCS[name](x, _g(), **{**CFG[name], "p": 1.0})
    other = 1 if axis == 2 else 2
    zero_pos = (y == 0).all(dim=other)  # (B, n): positions zeroed across the other axis
    frac = zero_pos.float().mean(dim=1)
    n = x.shape[axis]
    lo = CFG[name]["width"][0] / n
    hi = CFG[name]["blocks"][1] * CFG[name]["width"][1] / n
    assert (frac >= lo - 1e-6).all() and (frac <= hi + 1e-6).all()
    # nothing outside the masked positions changed
    assert torch.equal(y[y != 0], x[y != 0])


def test_crop_resize_full_span_is_identity():
    x = _x()
    assert torch.allclose(A.crop_resize(x, _g(), p=1.0, min_frac=1.0), x, atol=1e-4)  # float32 interpolation weights


def test_smooth_rows_keeps_constant_signal():
    x = torch.full((2, 30, 5), 3.0)
    assert torch.allclose(A.smooth_rows(x, 5), x)
