"""Tests for Task 3 metrics and methods (U5)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import f1_score

from xroom.config import load_task3_config
from xroom.fewshot import Protocol, sample_support
from xroom.probe import per_class_f1
from xroom.task3_methods import METHODS, predict
from xroom.task3_metrics import _f1_from_confusion, aggregate, bootstrap_draw_mean, episode_scores

CFG = load_task3_config()
P_AB = Protocol.from_config(CFG, "AtoB")


def toy_episode(seed: int = 0, n_per_class: int = 12, acc: float = 0.7):
    rng = np.random.default_rng(seed)
    labels = list(P_AB.target_labels)
    y = np.repeat(labels, n_per_class).astype(object)
    pred = y.copy()
    flip = rng.random(len(y)) > acc
    pred[flip] = rng.choice(labels, flip.sum())
    caps = np.array([f"{c}{i // 3}" for c, i in zip(y, np.tile(np.arange(n_per_class), len(labels)))])
    return y, pred, caps


# ---------- metrics ----------

def test_episode_scores_match_sklearn():
    y, pred, _ = toy_episode()
    s = episode_scores(y, pred, P_AB)
    labels = list(P_AB.target_labels)
    assert s["overall"] == pytest.approx(f1_score(y, pred, labels=labels, average="macro"))
    pc = f1_score(y, pred, labels=labels, average=None)
    by = dict(zip(labels, pc))
    assert s["exclusive"] == pytest.approx(np.mean([by[c] for c in P_AB.target_only]))
    assert s["shared"] == pytest.approx(np.mean([by[c] for c in P_AB.shared]))
    cm = np.array(s["confusion"]["matrix"])
    assert cm.sum(1).tolist() == [12] * len(labels) and cm.sum() == len(y)
    with pytest.raises(ValueError):
        episode_scores(y, np.where(pred == "sit", "fall", pred), P_AB)  # fall is not a Room B label


def test_confusion_f1_equals_per_class_f1():
    y, pred, _ = toy_episode(seed=3)
    s = episode_scores(y, pred, P_AB)
    f1 = _f1_from_confusion(np.array(s["confusion"]["matrix"])[None])[0]
    assert np.allclose(f1, list(per_class_f1(y, pred, P_AB.target_labels).values()))


def test_bootstrap_brackets_mean_and_collapses_when_perfect():
    y, _, caps = toy_episode()
    draws = np.array([toy_episode(seed=s)[1] for s in range(4)], dtype=object)
    episodes = [{"scores": episode_scores(y, d, P_AB), "predictions": list(d)} for d in draws]
    agg = aggregate(episodes, y, caps, P_AB, reps=300)
    for m in ("overall", "exclusive", "shared"):
        lo, hi = agg[m]["ci95"]
        assert lo <= agg[m]["mean"] <= hi and hi > lo
    perfect = bootstrap_draw_mean(y, np.array([y, y], dtype=object), caps, P_AB, reps=50)
    assert all(v == [1.0, 1.0] for v in perfect.values())


# ---------- methods on real data ----------

def _real():
    emb = Path(CFG["paths"]["task3_dir"]) / "emb"
    if not (emb / "default.npy").exists():
        pytest.skip("embedding cache not built (scripts/task3_embed.py)")
    from xroom.dataset import load_windows
    _, meta = load_windows(CFG)
    rooms = meta["room"].to_numpy()[:, None]
    feats = {"base": np.load(emb / "default.npy"),
             "adabn": np.where(rooms == "A", np.load(emb / "adabn_A.npy"), np.load(emb / "adabn_B.npy"))}
    return meta, feats


@pytest.mark.parametrize("direction", ["AtoB", "BtoA"])
def test_methods_cannot_see_hidden_labels(direction):
    """Sentinel test: hiding every non-support target label (incl. all validation labels)
    behind 'SENTINEL' must not change any method's predictions."""
    meta, feats = _real()
    p = Protocol.from_config(CFG, direction)
    support = sample_support(p, meta, 5, 0)
    poisoned = meta.copy()
    hidden = (poisoned["room"] == p.target) & ~poisoned["capture_id"].isin(support)
    poisoned.loc[hidden, "activity"] = "SENTINEL"
    n_val = p.masks(meta)["target_val"].sum()
    for method in METHODS:
        clean = predict(method, feats, meta, p, support, CFG, seed=0)
        dirty = predict(method, feats, poisoned, p, support, CFG, seed=0)
        assert np.array_equal(clean, dirty), method
        assert len(clean) == n_val and set(clean) <= set(p.target_labels), method


def test_methods_are_deterministic():
    meta, feats = _real()
    support = sample_support(P_AB, meta, 1, 2)
    for method in METHODS + ("UB", "CHANCE"):
        a = predict(method, feats, meta, P_AB, support, CFG, seed=2)
        b = predict(method, feats, meta, P_AB, support, CFG, seed=2)
        assert np.array_equal(a, b), method


# ---------- U7: source model for routed windows ----------

from xroom.task3_methods import predict_source  # noqa: E402


@pytest.mark.parametrize("direction", ["AtoB", "BtoA"])
def test_source_model_sees_no_target_labels(direction):
    meta, feats = _real()
    p = Protocol.from_config(CFG, direction)
    poisoned = meta.copy()
    poisoned.loc[poisoned["room"] == p.target, "activity"] = "SENTINEL"
    poisoned.loc[(poisoned["room"] == p.source) & (poisoned["split"] == "val"), "activity"] = "SENTINEL"
    clean = predict_source(feats, meta, p, CFG)
    assert np.array_equal(clean, predict_source(feats, poisoned, p, CFG))
    assert len(clean) == p.masks(meta)["source_val"].sum() and set(clean) <= set(p.source_labels)
