"""Tests for Task 3 adaptation building blocks (U3: AdaBN)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from xroom.adapt import adabn
from xroom.config import load_task3_config
from xroom.encoder import FrozenEncoder, build_encoder, embed, load_encoder

CFG = load_task3_config()


def toy_setup(seed: int = 0):
    """A small real-architecture encoder (CPU), random windows and metadata for both rooms."""
    torch.manual_seed(seed)
    stats = load_encoder(CFG, device="cpu").stats
    model = build_encoder(CFG["encoder"], len(stats.data_sc)).eval()
    enc = FrozenEncoder(model, stats, "cpu")
    rng = np.random.default_rng(seed)
    x = rng.uniform(10, 1000, size=(16, 48, 256)).astype(np.float32)
    meta = pd.DataFrame({"room": ["A"] * 8 + ["B"] * 8,
                         "split": (["dev"] * 6 + ["val"] * 2) * 2})
    return enc, x, meta


def bn_buffers(fe: FrozenEncoder) -> dict:
    return {k: v.clone() for k, v in fe.model.state_dict().items() if "running" in k}


def test_adabn_leaves_original_untouched():
    enc, x, meta = toy_setup()
    before = {k: v.clone() for k, v in enc.model.state_dict().items()}
    adabn(enc, x, meta, "A", batch=4)
    after = enc.model.state_dict()
    assert all(torch.equal(before[k], after[k]) for k in before)
    assert not enc.model.training


def test_adabn_changes_bn_stats_only():
    enc, x, meta = toy_setup()
    ad = adabn(enc, x, meta, "A", batch=4)
    orig, new = enc.model.state_dict(), ad.model.state_dict()
    for k in orig:
        if "running_mean" in k or "running_var" in k:
            assert not torch.equal(orig[k], new[k]), k  # re-estimated
        elif "num_batches_tracked" not in k:
            assert torch.equal(orig[k], new[k]), k  # weights unchanged
    assert not ad.model.training and all(not p.requires_grad for p in ad.model.parameters())


def test_adabn_uses_only_that_rooms_dev_windows():
    enc, x, meta = toy_setup()
    ref = bn_buffers(adabn(enc, x, meta, "A", batch=4))
    x2 = x.copy()
    x2[(meta["split"] == "val").to_numpy()] *= 3.0  # validation windows of both rooms
    x2[(meta["room"] == "B").to_numpy()] += 50.0  # every Room B window
    same = bn_buffers(adabn(enc, x2, meta, "A", batch=4))
    assert all(torch.equal(ref[k], same[k]) for k in ref)
    x3 = x.copy()
    a_dev = ((meta["room"] == "A") & (meta["split"] == "dev")).to_numpy()
    # new random content (a pure rescale would be invisible: inputs are per-row normalized)
    x3[a_dev] = np.random.default_rng(9).uniform(10, 1000, size=x3[a_dev].shape).astype(np.float32)
    changed = bn_buffers(adabn(enc, x3, meta, "A", batch=4))
    assert any(not torch.equal(ref[k], changed[k]) for k in ref if "running_mean" in k)


def test_adabn_deterministic_and_embeds():
    enc, x, meta = toy_setup()
    a, b = adabn(enc, x, meta, "B", batch=4), adabn(enc, x, meta, "B", batch=4)
    za, zb = embed(a, x), embed(b, x)
    assert za.shape == (16, enc.dim) and np.isfinite(za).all()
    assert np.array_equal(za, zb)


def test_adabn_rejects_room_without_dev_windows():
    enc, x, meta = toy_setup()
    with pytest.raises(ValueError):
        adabn(enc, x, meta, "C")


def test_cached_default_embeddings_match_encoder():
    path = Path(CFG["paths"]["task3_dir"]) / "emb" / "default.npy"
    if not path.exists():
        pytest.skip("embedding cache not built (scripts/task3_embed.py)")
    from xroom.dataset import load_windows
    x, _ = load_windows(CFG)
    cached = np.load(path)
    idx = np.r_[0:32, len(x) - 32:len(x)]
    fresh = embed(load_encoder(CFG), x[idx])
    assert cached.shape == (len(x), 128) and cached.dtype == np.float32
    assert np.abs(cached[idx] - fresh).max() < 1e-5
