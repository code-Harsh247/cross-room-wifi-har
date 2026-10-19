"""Tests for the SSL data pipeline (Task 2, gate T1)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from xroom.config import load_task2_config
from xroom.ssl_data import (CropSampler, NormStats, apply_norm, assert_dev_only, dev_capture_ids,
                            fit_dev_stats, prepare)

CFG = load_task2_config()
D = CFG["ssl_data"]


def _toy(seed=0):
    """6 dev + 2 val captures per room; val captures have a very different scale."""
    rng = np.random.default_rng(seed)
    rows, seqs = [], {}
    for room, length in (("A", 150), ("B", 500)):
        for i in range(8):
            cid = f"{room}/c{i}"
            split = "val" if i >= 6 else "dev"
            fold = "holdout" if i == 5 else "train"
            scale = 100.0 if split == "val" else 1.0
            seqs[cid] = (scale * rng.uniform(1, 100, size=(length, 256))).astype(np.float16)
            rows.append({"capture_id": cid, "room": room, "activity": "sit", "split": split, "dev_fold": fold})
    return pd.DataFrame(rows), seqs


def _sampler(table, seqs, seed=0, ids=None):
    stats = fit_dev_stats(table, seqs, CFG)
    ids = ids if ids is not None else dev_capture_ids(table, seqs)
    return CropSampler(ids, table, seqs, stats, batch_size=32, crop_min=D["crop_min"],
                       crop_max=D["crop_max"], room_balance=True, seed=seed)


def test_no_validation_capture_in_200_batches():
    table, seqs = _toy()
    s = _sampler(table, seqs)
    seen = set()
    for _ in range(200):
        seen |= set(s.batch()[1])
    val_ids = set(table.loc[table["split"] == "val", "capture_id"])
    assert seen and not (seen & val_ids)


def test_injected_validation_id_is_refused():
    table, seqs = _toy()
    ids = dev_capture_ids(table, seqs) + ["B/c7"]
    with pytest.raises(AssertionError):
        _sampler(table, seqs, ids=ids)
    with pytest.raises(AssertionError):
        assert_dev_only(["A/c6"], table)


def test_batch_shape_dtype_and_finite():
    table, seqs = _toy()
    s = _sampler(table, seqs)
    for _ in range(20):
        x, ids = s.batch()
        assert x.shape[0] == 32 and x.shape[2] == 242 and len(ids) == 32
        assert D["crop_min"] <= x.shape[1] <= D["crop_max"]
        assert x.dtype == np.float32 and np.isfinite(x).all()


def test_room_balance_over_200_batches():
    table, seqs = _toy()
    s = _sampler(table, seqs)
    rooms = np.concatenate([s.batch_rooms(s.batch()[1]) for _ in range(200)])
    assert 0.45 <= (rooms == "A").mean() <= 0.55


def test_stats_are_dev_only_and_room_weighted():
    table, seqs = _toy()
    stats = fit_dev_stats(table, seqs, CFG)
    dev = table[table["split"] == "dev"]
    per_room = []
    for _, g in dev.groupby("room"):
        rows = np.concatenate([prepare(seqs[c], stats.data_sc) for c in g["capture_id"]]).astype(np.float64)
        per_room.append((rows.mean(0), (rows ** 2).mean(0)))
    m1 = np.mean([p[0] for p in per_room], 0)
    std = np.sqrt(np.mean([p[1] for p in per_room], 0) - m1 ** 2)
    np.testing.assert_allclose(stats.mean, m1, atol=1e-5)
    np.testing.assert_allclose(stats.std, std, atol=1e-5)


def test_sampler_is_deterministic():
    table, seqs = _toy()
    a, b, c = _sampler(table, seqs, 1), _sampler(table, seqs, 1), _sampler(table, seqs, 2)
    xa, xb, xc = a.batch()[0], b.batch()[0], c.batch()[0]
    np.testing.assert_array_equal(xa, xb)
    assert xa.shape != xc.shape or not np.array_equal(xa, xc)


def test_row_normalization_removes_row_gain(tmp_path):
    table, seqs = _toy()
    stats = fit_dev_stats(table, seqs, CFG)
    x = np.asarray(seqs["A/c0"], np.float32)
    gain = np.random.default_rng(3).uniform(0.5, 2, size=(len(x), 1)).astype(np.float32)
    np.testing.assert_allclose(apply_norm(x, stats), apply_norm(x * gain, stats), atol=1e-3)
    stats.save(tmp_path / "s.npz")
    back = NormStats.load(tmp_path / "s.npz")
    np.testing.assert_array_equal(back.mean, stats.mean)
    assert (back.row, back.log, back.standardize) == (stats.row, stats.log, stats.standardize)


def test_fold_selection_and_no_activity_column_needed():
    table, seqs = _toy()
    hold = dev_capture_ids(table, seqs, fold="holdout")
    assert sorted(hold) == ["A/c5", "B/c5"]
    _sampler(table.drop(columns="activity"), seqs).batch()  # label-free: works without activities


def test_real_data_no_leakage_and_balance():
    """Gate T1 on the real cache: 200 room-balanced batches from dev-train, no validation id."""
    from xroom.dataset import load_sequences, sequences_path

    if not sequences_path(CFG).exists():
        pytest.skip("cache not built (run scripts/build_windows.py)")
    table, seqs = load_sequences(CFG)
    ids = dev_capture_ids(table, seqs, fold="train")
    stats = fit_dev_stats(table, seqs, CFG)
    s = CropSampler(ids[::4], table, seqs, stats, batch_size=64, crop_min=D["crop_min"],
                    crop_max=D["crop_max"], room_balance=True, seed=0)
    seen, rooms = set(), []
    for _ in range(200):
        x, b = s.batch()
        seen |= set(b)
        rooms.append(s.batch_rooms(b))
    assert not seen & set(table.loc[table["split"] == "val", "capture_id"])
    assert 0.45 <= (np.concatenate(rooms) == "A").mean() <= 0.55
    assert x.shape[2] == 242 and np.isfinite(x).all()
