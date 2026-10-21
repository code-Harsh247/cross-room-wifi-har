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


# ---------- U4: alignment and refinement ----------

import inspect  # noqa: E402

from xroom.adapt import coral_apply, coral_fit, per_room_standardize, refine_prototypes  # noqa: E402


def test_per_room_standardize_uses_dev_stats_only():
    rng = np.random.default_rng(0)
    f = np.vstack([rng.normal(5, 3, (40, 8)), rng.normal(-2, 0.5, (40, 8))])
    rooms = np.array(["A"] * 40 + ["B"] * 40)
    dev = np.tile(np.r_[np.ones(30, bool), np.zeros(10, bool)], 2)
    z = per_room_standardize(f, rooms, dev)
    for r in ("A", "B"):
        m = (rooms == r) & dev
        np.testing.assert_allclose(z[m].mean(0), 0, atol=1e-5)
        np.testing.assert_allclose(z[m].std(0), 1, atol=1e-5)
    f2 = f.copy()
    f2[~dev] += 100.0  # change validation rows only
    z2 = per_room_standardize(f2, rooms, dev)
    np.testing.assert_allclose(z2[dev], z[dev], atol=1e-6)  # statistics unchanged
    with pytest.raises(ValueError):
        per_room_standardize(f, rooms, np.zeros(80, bool))


def test_coral_matches_target_covariance():
    rng = np.random.default_rng(1)
    d = 5
    a = rng.normal(size=(d, d))
    b = rng.normal(size=(d, d))
    src = rng.normal(size=(4000, d)) @ a + 3.0
    tgt = rng.normal(size=(4000, d)) @ b - 1.0
    fit = coral_fit(src, tgt, eps=1e-8)
    out = coral_apply(src, fit)
    np.testing.assert_allclose(np.cov(out, rowvar=False), np.cov(tgt, rowvar=False), atol=1e-3)
    np.testing.assert_allclose(out.mean(0), tgt.mean(0), atol=1e-3)


def _clusters(rng, centres, n=60, noise=0.15):
    x = np.vstack([c + noise * rng.normal(size=(n, len(c))) for c in centres])
    y = np.repeat(np.arange(len(centres)), n)
    return x, y


def test_refinement_moves_toward_true_centres_and_keeps_pinned_class():
    rng = np.random.default_rng(2)
    true = np.eye(4)[:3] * 3  # three well-separated directions in 4-d
    x, y = _clusters(rng, true)
    start = true + np.array([[0, 1.5, 0, 0.8], [0.9, 0, 0, 0.8], [1.2, 0, 0, 1.0]])  # biased start
    unit = lambda v: v / np.linalg.norm(v, axis=1, keepdims=True)  # noqa: E731
    support = x[y == 2][:3]
    refined = refine_prototypes(start, x, support, np.full(3, 2), iters=10, temperature=0.1)
    before = (unit(start) * unit(true)).sum(1)
    after = (refined * unit(true)).sum(1)
    assert (after > before).all() and (after > 0.98).all()
    assert np.argmax(refined @ unit(true).T, axis=1).tolist() == [0, 1, 2]  # classes keep identity
    again = refine_prototypes(start, x, support, np.full(3, 2), iters=10, temperature=0.1)
    assert np.array_equal(refined, again)  # deterministic


def test_adaptation_functions_take_no_activity_labels():
    for fn in (per_room_standardize, coral_fit, adabn):
        params = set(inspect.signature(fn).parameters)
        assert not params & {"y", "labels", "activity", "y_train"}, fn.__name__
    assert set(inspect.signature(refine_prototypes).parameters) == {
        "protos", "unlabeled", "support", "support_idx", "iters", "temperature"}
