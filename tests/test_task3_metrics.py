"""Tests for Task 3 metrics and methods (U5)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
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
             "adabn": np.where(rooms == "A", np.load(emb / "adabn_A.npy"), np.load(emb / "adabn_B.npy")),
             "inputnorm": np.where(rooms == "A", np.load(emb / "inputnorm_A.npy"), np.load(emb / "inputnorm_B.npy"))}
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
    for method in METHODS + ("UB", "CHANCE", "SHUF"):
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


# ---------- U8: controls and M6 ----------

from xroom.adapt import room_input_stats  # noqa: E402
from xroom.task3_methods import _shuffle_support  # noqa: E402


def test_shuffled_support_permutes_capture_labels():
    meta, _ = _real()
    p = Protocol.from_config(CFG, "BtoA")
    support = sample_support(p, meta, 10, 0)
    rows = meta["capture_id"].isin(support).to_numpy()
    lab = np.where(rows, meta["activity"].to_numpy(dtype=object), None)
    out = _shuffle_support(lab, meta, rows, seed=0)
    caps = meta.loc[rows, "capture_id"]
    per_cap = pd.Series(out[rows], index=caps.index).groupby(caps).nunique()
    assert (per_cap == 1).all()  # one label per capture
    assert sorted(out[rows]) == sorted(lab[rows])  # same label multiset
    assert (out[rows] != lab[rows]).any()  # actually shuffled
    assert np.array_equal(out, _shuffle_support(lab, meta, rows, seed=0))  # deterministic
    assert (out[~rows] == None).all()  # noqa: E711  other rows untouched


def test_room_input_stats_use_room_dev_only():
    meta, _ = _real()
    from xroom.dataset import load_windows
    from xroom.encoder import load_encoder
    x, _ = load_windows(CFG)
    base = load_encoder(CFG, device="cpu").stats
    st = room_input_stats(base, x, meta, "A")
    assert st.mean.shape == base.mean.shape and np.isfinite(st.std).all() and (st.std > 0).all()
    assert (st.row, st.log, st.standardize) == (base.row, base.log, base.standardize)
    x2 = np.array(x[:])  # change every validation window and every Room B window
    change = ((meta["split"] == "val") | (meta["room"] == "B")).to_numpy()
    x2[change] = (x2[change].astype(np.float32) * 1.7 + 5).astype(x2.dtype)
    st2 = room_input_stats(base, x2, meta, "A")
    assert np.array_equal(st.mean, st2.mean) and np.array_equal(st.std, st2.std)


# ---------- post-hoc idea 2 (M7-M9) ----------

from xroom.config import PROJECT_ROOT, load_config  # noqa: E402
from xroom.task3_methods import POSTHOC  # noqa: E402

CFG_PH = load_config(PROJECT_ROOT / "configs" / "task3_posthoc.yaml")


@pytest.mark.parametrize("direction", ["AtoB", "BtoA"])
def test_posthoc_methods_cannot_see_hidden_labels(direction):
    meta, feats = _real()
    feats = {**feats, "members": [feats["base"], feats["adabn"], feats["inputnorm"]]}  # for M11
    p = Protocol.from_config(CFG_PH, direction)
    support = sample_support(p, meta, 5, 0)
    poisoned = meta.copy()
    poisoned.loc[(poisoned["room"] == p.target) & ~poisoned["capture_id"].isin(support), "activity"] = "SENTINEL"
    for method in POSTHOC:
        clean = predict(method, feats, meta, p, support, CFG_PH, seed=0)
        assert np.array_equal(clean, predict(method, feats, poisoned, p, support, CFG_PH, seed=0)), method
        assert np.array_equal(clean, predict(method, feats, meta, p, support, CFG_PH, seed=0)), method
        assert set(clean) <= set(p.target_labels), method


def test_posthoc_needs_its_config():
    meta, feats = _real()
    with pytest.raises(ValueError):
        predict("M7", feats, meta, P_AB, sample_support(P_AB, meta, 1, 0), CFG, seed=0)


def test_probability_ensemble_of_one_member_equals_m9():
    meta, feats = _real()
    support = sample_support(P_AB, meta, 5, 1)
    one = {**feats, "members": [feats["base"]]}
    assert np.array_equal(predict("M11", one, meta, P_AB, support, CFG_PH, seed=1),
                          predict("M9", feats, meta, P_AB, support, CFG_PH, seed=1))
    with pytest.raises(ValueError):
        predict("M11", feats, meta, P_AB, support, CFG_PH, seed=1)  # no members


# ---------- published post-hoc MAE encoders (models/task3_mae_s*) ----------

@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_published_mae_encoders_load_and_embed(seed):
    from xroom.encoder import FrozenEncoder, embed, load_run
    run_dir = PROJECT_ROOT / "models" / f"task3_mae_s{seed}"
    if not (run_dir / "encoder.pt").exists():
        pytest.skip("published MAE encoders not present (scripts/task3_publish_mae.py)")
    run = __import__("json").loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert run["method"] == "mae" and run["seed"] == seed and "label-free" in run["selection"]
    assert "no activity labels" in (run_dir / "PROVENANCE.md").read_text(encoding="utf-8")
    model, stats = load_run(run_dir, "cpu")
    x = np.random.default_rng(seed).uniform(10, 1000, size=(4, 150, 256)).astype(np.float32)
    z = embed(FrozenEncoder(model, stats, "cpu", run_dir), x)
    assert z.shape == (4, 128) and np.isfinite(z).all()
