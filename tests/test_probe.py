"""Tests for the linear probe and baseline features (Task 2, gate T6)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import f1_score

from xroom.probe import (bootstrap_macro_f1, flat_windows, handcrafted_features, linear_probe, macro_f1,
                         pca_features, probe_suite, shuffle_by_capture)
from xroom.ssl_data import NormStats


def test_macro_f1_matches_sklearn():
    rng = np.random.default_rng(0)
    y = rng.choice(list("abcd"), 300)
    p = np.where(rng.random(300) < 0.6, y, rng.choice(list("abcd"), 300))
    assert abs(macro_f1(y, p) - f1_score(y, p, average="macro")) < 1e-12
    labels = list("abcd")
    assert abs(macro_f1(y, p, labels) - f1_score(y, p, labels=labels, average="macro")) < 1e-12


def _toy(n_caps=40, per=3, seed=0, sep=3.0):
    """Two rooms, three activities, per-capture windows; features separate activities."""
    rng = np.random.default_rng(seed)
    rows, feats = [], []
    acts = ["sit", "stand", "walk"]
    for room in "AB":
        for i in range(n_caps):
            a = acts[i % 3]
            split = "val" if i % 5 == 0 else "dev"
            fold = "holdout" if i % 5 == 1 else "train"
            for k in range(per):
                rows.append({"capture_id": f"{room}/{i}", "room": room, "activity": a, "split": split,
                             "dev_fold": fold if split == "dev" else np.nan})
                feats.append(rng.normal(size=8) + sep * np.eye(8)[acts.index(a)])
    return np.array(feats, np.float32), pd.DataFrame(rows)


def test_probe_scores_and_guards():
    f, meta = _toy()
    dev, val = (meta["split"] == "dev").to_numpy(), (meta["split"] == "val").to_numpy()
    r = linear_probe(f, meta["activity"], meta, dev, val, [0.1, 1.0], reps=100)
    assert r["macro_f1"] > 0.9 and r["ci95"][0] <= r["macro_f1"] <= r["ci95"][1]
    assert set(r["per_class_f1"]) == {"sit", "stand", "walk"}
    with pytest.raises(AssertionError):  # validation rows in training are refused
        linear_probe(f, meta["activity"], meta, dev | val, val, [1.0], reps=10)
    with pytest.raises(AssertionError):  # development rows as test are refused
        linear_probe(f, meta["activity"], meta, dev, dev, [1.0], reps=10)


def test_tuning_never_sees_validation(monkeypatch):
    import xroom.probe as P

    f, meta = _toy()
    seen = []
    real = P.tune_c

    def spy(ff, y, fold, grid, seed=0):
        seen.append(len(ff))
        assert not pd.isna(fold).any()  # only dev rows carry a dev_fold
        return real(ff, y, fold, grid, seed)

    monkeypatch.setattr(P, "tune_c", spy)
    dev, val = (meta["split"] == "dev").to_numpy(), (meta["split"] == "val").to_numpy()
    P.linear_probe(f, meta["activity"], meta, dev, val, [1.0], reps=10)
    assert seen == [int(dev.sum())]


def test_shuffled_labels_give_chance():
    # One shuffle maps whole activity clusters to arbitrary labels (macro-F1 anywhere in
    # [0, 1]), so chance level is checked on the mean over shuffle seeds.
    f, meta = _toy(n_caps=150, seed=1)
    scores = [probe_suite(f, meta, [0.01, 1.0], reps=10, seed=s, shuffle_labels=True, sets=["joint"])
              ["joint"]["macro_f1"] for s in range(10)]
    assert abs(np.mean(scores) - 1 / 3) < 0.1
    y = meta["activity"].to_numpy()
    s = shuffle_by_capture(y, meta["capture_id"].to_numpy())
    assert (pd.Series(s).groupby(meta["capture_id"]).nunique() == 1).all()


def test_probe_suite_sets():
    f, meta = _toy()
    res = probe_suite(f, meta, [1.0], reps=10)
    assert {"room_A", "room_B", "joint", "cross_A_to_B", "cross_B_to_A"} <= set(res)
    assert res["headline_mean_room_macro_f1"] == pytest.approx((res["room_A"]["macro_f1"] + res["room_B"]["macro_f1"]) / 2)


def test_bootstrap_is_deterministic_and_brackets():
    rng = np.random.default_rng(0)
    y = np.repeat(list("abc"), 30)
    p = np.where(rng.random(90) < 0.7, y, "a")
    caps = np.repeat(np.arange(30), 3)
    assert bootstrap_macro_f1(y, p, caps, 200, 1) == bootstrap_macro_f1(y, p, caps, 200, 1)


def test_baseline_features_shapes():
    rng = np.random.default_rng(0)
    x = rng.uniform(1, 100, size=(12, 150, 256)).astype(np.float16)
    sc = np.arange(2, 244)
    assert handcrafted_features(x, sc).shape == (12, 484)
    stats = NormStats(sc, np.zeros(242, np.float32), np.ones(242, np.float32))
    flat = flat_windows(x, stats, pool=2)
    assert flat.shape == (12, 75 * 242)
    fit = np.arange(12) < 8
    assert pca_features(flat, fit, 4).shape == (12, 4)
