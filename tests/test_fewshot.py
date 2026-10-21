"""Tests for the Task 3 protocol and the label access guard (U1)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from xroom.config import load_task3_config
from xroom.fewshot import Protocol, check_support, visible_labels

CFG = load_task3_config()


def toy_meta(seed: int = 0) -> pd.DataFrame:
    """Small metadata table with both rooms, both splits, all activities; B captures have 3 windows."""
    rng = np.random.default_rng(seed)
    rows = []
    for room, acts in CFG["task3"]["activities"].items():
        n_win = 1 if room == "A" else 3
        for act in acts:
            for i in range(6):
                split = "val" if i >= 4 else "dev"
                cid = f"{room}/{act}{i}"
                for w in range(n_win):
                    rows.append({"capture_id": cid, "room": room, "activity": act, "split": split,
                                 "dev_fold": None if split == "val" else rng.choice(["train", "holdout"]),
                                 "start": 150 * w})
    return pd.DataFrame(rows)


def target_dev_caps(protocol: Protocol, meta: pd.DataFrame, acts) -> list[str]:
    m = meta[(meta["room"] == protocol.target) & (meta["split"] == "dev") & meta["activity"].isin(acts)]
    return sorted(m["capture_id"].unique())


# ---------- protocol ----------

def test_label_sets_both_directions():
    ab, ba = Protocol.from_config(CFG, "AtoB"), Protocol.from_config(CFG, "BtoA")
    assert (ab.source, ab.target, ba.source, ba.target) == ("A", "B", "B", "A")
    assert set(ab.target_labels) == {"empty", "lying", "sit", "stand", "walk"}
    assert set(ba.target_labels) == {"fall", "sit", "sitdown", "stand", "standup", "walk"}
    assert set(ab.target_only) == {"empty", "lying"} and set(ba.target_only) == {"fall", "sitdown", "standup"}
    assert set(ab.source_only) == set(ba.target_only) and set(ba.source_only) == set(ab.target_only)
    assert set(ab.shared) == set(ba.shared) == {"sit", "stand", "walk"}
    with pytest.raises(ValueError):
        Protocol.from_config(CFG, "AtoA")


def test_masks_partition_rows():
    meta = toy_meta()
    m = Protocol.from_config(CFG, "AtoB").masks(meta)
    stacked = np.stack(list(m.values()))
    assert (stacked.sum(0) == 1).all()  # every row in exactly one of the four masks


def test_default_method_is_preregistered_m1():
    assert CFG["task3"]["default_method"] == "M1"
    assert CFG["task3"]["k_values"] == [1, 5, 10, 20]


# ---------- access guard ----------

@pytest.mark.parametrize("direction", ["AtoB", "BtoA"])
def test_guard_shows_only_source_dev_and_support(direction):
    meta = toy_meta()
    p = Protocol.from_config(CFG, direction)
    rng = np.random.default_rng(1)
    pool = target_dev_caps(p, meta, p.target_only)
    for _ in range(20):
        support = list(rng.choice(pool, size=rng.integers(0, len(pool) + 1), replace=False))
        vis = visible_labels(p, meta, support)
        seen = vis.notna().to_numpy()
        tgt = (meta["room"] == p.target).to_numpy()
        # every visible target label belongs to a support capture
        assert set(meta.loc[seen & tgt, "capture_id"]) <= set(support)
        # all windows of each support capture are visible (Room B: all 3)
        assert vis[meta["capture_id"].isin(support)].notna().all()
        # no validation label is ever visible
        assert (meta.loc[seen, "split"] == "dev").all()
        # source dev labels are all visible and correct
        sd = p.masks(meta)["source_dev"]
        assert (vis[sd] == meta.loc[sd, "activity"]).all()
        # no shared-class target label is ever visible
        assert not (seen & tgt & meta["activity"].isin(p.shared).to_numpy()).any()


def test_guard_sentinel_never_leaks():
    meta = toy_meta()
    p = Protocol.from_config(CFG, "AtoB")
    pool = target_dev_caps(p, meta, p.target_only)
    support = pool[:2]
    hidden = (meta["room"] == p.target) & (meta["split"] == "dev") & ~meta["capture_id"].isin(support)
    meta = meta.copy()
    meta.loc[hidden[hidden].index[0], "activity"] = "SENTINEL"
    meta.loc[(meta["split"] == "val").idxmax(), "activity"] = "SENTINEL"
    assert "SENTINEL" not in set(visible_labels(p, meta, support).dropna())


@pytest.mark.parametrize("bad", ["val", "shared", "source_room", "unknown"])
def test_guard_rejects_invalid_support(bad):
    meta = toy_meta()
    p = Protocol.from_config(CFG, "AtoB")
    caps = meta.drop_duplicates("capture_id")
    pick = {
        "val": caps[(caps.room == "B") & (caps.split == "val") & caps.activity.isin(p.target_only)],
        "shared": caps[(caps.room == "B") & (caps.split == "dev") & caps.activity.isin(p.shared)],
        "source_room": caps[(caps.room == "A") & (caps.split == "dev")],
    }
    support = ["B/nope"] if bad == "unknown" else [pick[bad]["capture_id"].iloc[0]]
    with pytest.raises(ValueError):
        visible_labels(p, meta, support)
    with pytest.raises(ValueError):
        check_support(p, meta, support)


# ---------- real data ----------

def _real_meta():
    try:
        from xroom.dataset import load_windows
        return load_windows(CFG)[1]
    except (FileNotFoundError, OSError):
        pytest.skip("window cache not built (scripts/build_windows.py)")


def test_config_activity_sets_match_data():
    meta = _real_meta()
    for room, acts in CFG["task3"]["activities"].items():
        assert set(meta.loc[meta["room"] == room, "activity"]) == set(acts)


@pytest.mark.parametrize("direction", ["AtoB", "BtoA"])
def test_guard_on_real_data(direction):
    meta = _real_meta()
    p = Protocol.from_config(CFG, direction)
    support = target_dev_caps(p, meta, p.target_only)[::40]
    vis = visible_labels(p, meta, support)
    seen = vis.notna().to_numpy()
    assert (meta.loc[seen, "split"] == "dev").all()
    tgt_seen = seen & (meta["room"] == p.target).to_numpy()
    assert set(meta.loc[tgt_seen, "capture_id"]) == set(support)
    assert seen.sum() == p.masks(meta)["source_dev"].sum() + meta["capture_id"].isin(support).sum()


# ---------- support sampling (U2) ----------

from xroom.fewshot import load_support, sample_support, support_pool, write_support_manifest  # noqa: E402


@pytest.mark.parametrize("direction", ["AtoB", "BtoA"])
def test_support_counts_nested_dev_only_deterministic(direction):
    meta = _real_meta()
    p = Protocol.from_config(CFG, direction)
    caps = meta.drop_duplicates("capture_id").set_index("capture_id")
    win_per_cap = 1 if p.target == "A" else 3
    for seed in (0, 1):
        prev = set()
        for k in CFG["task3"]["k_values"]:
            ids = sample_support(p, meta, k, seed)
            sel = caps.loc[ids]
            assert sel.groupby("activity").size().to_dict() == {a: k for a in p.target_only}  # K captures/class
            assert (sel["split"] == "dev").all() and (sel["room"] == p.target).all()
            assert meta["capture_id"].isin(ids).sum() == win_per_cap * k * len(p.target_only)  # windows
            assert prev <= set(ids)  # nested
            prev = set(ids)
    assert sample_support(p, meta, 5, 3) == sample_support(p, meta, 5, 3)  # deterministic
    assert sample_support(p, meta, 5, 3) != sample_support(p, meta, 5, 4)  # seeds differ


@pytest.mark.parametrize("direction", ["AtoB", "BtoA"])
def test_support_coverage_over_draws(direction):
    meta = _real_meta()
    p = Protocol.from_config(CFG, direction)
    t3 = CFG["task3"]
    kmax = max(t3["k_values"])
    caps = meta.drop_duplicates("capture_id").set_index("capture_id")
    used = {a: set() for a in p.target_only}
    for seed in range(t3["seed_base"], t3["seed_base"] + t3["draws"]):
        for cid in sample_support(p, meta, kmax, seed):
            used[caps.loc[cid, "activity"]].add(cid)
    assert all(len(u) >= 100 for u in used.values()), {a: len(u) for a, u in used.items()}


def test_support_rejects_bad_k_and_pool_is_dev_only():
    meta = toy_meta()
    p = Protocol.from_config(CFG, "AtoB")
    pool = support_pool(p, meta)
    assert all(meta.set_index("capture_id").loc[ids, "split"].eq("dev").all() for ids in pool.values())
    with pytest.raises(ValueError):
        sample_support(p, meta, 0, 0)
    with pytest.raises(ValueError):
        sample_support(p, meta, 99, 0)  # more than the 4 dev captures per class in toy data


def test_manifest_roundtrip(tmp_path):
    meta = toy_meta()
    p = Protocol.from_config(CFG, "BtoA")
    path = tmp_path / "BtoA_seed7.json"
    write_support_manifest(p, meta, [1, 2, 4], 7, path)
    for k in (1, 2, 4):
        assert sorted(load_support(path, k)) == sorted(sample_support(p, meta, k, 7))
    with pytest.raises(ValueError):
        load_support(path, 3)
