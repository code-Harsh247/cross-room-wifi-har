"""Tests for the Task 3 hand-off API (Task 2, gate T10)."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from xroom.config import load_task2_config
from xroom.encoder import FrozenEncoder, build_encoder, embed, load_encoder
from xroom.ssl_data import NormStats

CFG = load_task2_config()


def _toy_encoder():
    torch.manual_seed(0)
    sc = np.arange(2, 244)
    stats = NormStats(sc, np.ones(242, np.float32), np.full(242, 0.5, np.float32))
    return FrozenEncoder(build_encoder(CFG["encoder"]).eval(), stats, "cpu")


def test_embed_shape_finite_deterministic_any_length():
    fe = _toy_encoder()
    x = np.random.default_rng(0).uniform(1, 100, size=(10, 150, 256)).astype(np.float16)
    a, b = embed(fe, x, batch=4), embed(fe, x)
    assert a.shape == (10, CFG["encoder"]["embed_dim"]) and a.dtype == np.float32 and np.isfinite(a).all()
    np.testing.assert_allclose(a, b, atol=1e-5)  # batching does not change the result
    assert embed(fe, x[:, :50]).shape == (10, CFG["encoder"]["embed_dim"])
    assert embed(fe, x[:0]).shape == (0, CFG["encoder"]["embed_dim"])


def test_missing_run_is_a_clear_error():
    with pytest.raises(FileNotFoundError):
        load_encoder(CFG, "no_such_run")


def test_published_default_encoder():
    try:
        fe = load_encoder(CFG, "default")
    except FileNotFoundError:
        pytest.skip("default encoder not published (scripts/task2_publish_default.py)")
    from xroom.dataset import load_windows

    x, _ = load_windows(CFG)
    a, b = embed(fe, x[:10]), embed(fe, x[:10])
    assert a.shape == (10, 128) and np.isfinite(a).all()
    assert np.array_equal(a, b)
    assert not fe.model.training and not any(p.requires_grad for p in fe.model.parameters())
    assert "No activity labels were used" in (fe.run_dir / "PROVENANCE.md").read_text()
    # the same weights the probe used (simclr_s0) give the same embeddings
    src = load_encoder(CFG, "simclr_s0", device=fe.device)
    assert np.abs(embed(src, x[:200]) - embed(fe, x[:200])).max() < 1e-5
