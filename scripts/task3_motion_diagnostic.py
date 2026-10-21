"""Idea 1 diagnostic (docs/ideas.md): is there a room-independent motion signal for sit/stand/walk?

Post-hoc experiment (designed after U6-U8). Settings fixed before running:
  motion features per 150-row window, on per-row normalized amplitude (data subcarriers):
    tstd   mean over subcarriers of the std over time
    mad    mean over subcarriers of the mean absolute row-to-row change
    lfr    share of temporal variance in the slowest 25% of frequencies (mean over subcarriers)
  room-relative transform, fitted per room on that room's UNLABELLED development windows:
    z      z-score per feature
    rank   empirical percentile within the room's dev windows
  classifier: StandardScaler + class-balanced logistic regression (C = 1), trained on SOURCE
  dev windows of sit/stand/walk (source labels), scored on TARGET validation windows of
  sit/stand/walk (labels for scoring only).
Tasks: 3-class sit/stand/walk (chance 1/3) and 2-class walk vs still (sit+stand, chance 1/2).
Reference: the same 3-class transfer with M1 embeddings (nearest centroid) from
scripts/task3_diagnostics.py.

Second pass (added AFTER seeing the first pass, which found the motion ordering reversed in
Room B): the same features computed after a moving average over time (width 5 and 15 rows,
``smooth_time`` from scripts/task1_ablations.py) applied to both rooms, to test whether Room B's
row noise hides the real motion signal.

Usage:  python scripts/task3_motion_diagnostic.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from task1_ablations import smooth_time  # noqa: E402
from xroom.config import load_task3_config  # noqa: E402
from xroom.dataset import load_windows  # noqa: E402
from xroom.features import data_subcarriers  # noqa: E402
from xroom.fewshot import Protocol  # noqa: E402
from xroom.probe import macro_f1, per_class_f1  # noqa: E402
from xroom.room_classifier import make_linear_model  # noqa: E402
from xroom.ssl_data import prepare  # noqa: E402

FEATURES = ("tstd", "mad", "lfr")
SMOOTHING = (0, 5, 15)  # 0 = first pass (pre-specified); 5, 15 = second pass (post hoc)


def motion_features(x: np.ndarray, data_sc: np.ndarray, smooth: int = 0, chunk: int = 256) -> np.ndarray:
    out = []
    for i in range(0, len(x), chunk):
        xb = np.asarray(x[i : i + chunk], dtype=np.float32)
        if smooth > 1:
            xb = smooth_time(xb, smooth)
        r = prepare(xb, data_sc, row=True).astype(np.float64)  # (n, T, C)
        tstd = r.std(axis=1).mean(axis=1)
        mad = np.abs(np.diff(r, axis=1)).mean(axis=(1, 2))
        c = r - r.mean(axis=1, keepdims=True)
        power = np.abs(np.fft.rfft(c, axis=1)) ** 2  # (n, F, C)
        cut = max(1, power.shape[1] // 4)
        lfr = (power[:, 1 : cut + 1].sum(axis=1) / np.maximum(power[:, 1:].sum(axis=1), 1e-12)).mean(axis=1)
        out.append(np.stack([tstd, mad, lfr], axis=1))
    return np.concatenate(out)


def room_relative(f: np.ndarray, rooms: np.ndarray, dev: np.ndarray, mode: str) -> np.ndarray:
    out = np.empty_like(f)
    for room in np.unique(rooms):
        in_room, fit = rooms == room, (rooms == room) & dev
        if mode == "raw":
            out[in_room] = f[in_room]
        elif mode == "z":
            out[in_room] = (f[in_room] - f[fit].mean(0)) / np.maximum(f[fit].std(0), 1e-12)
        elif mode == "rank":
            ref = np.sort(f[fit], axis=0)
            out[in_room] = np.stack([np.searchsorted(ref[:, j], f[in_room, j]) / len(ref) for j in range(f.shape[1])], 1)
        else:
            raise ValueError(mode)
    return out


def run_transfer(cfg, meta, raw, rooms, dev, act, shared, ref) -> dict:
    results = {}
    for direction in cfg["task3"]["directions"]:
        p = Protocol.from_config(cfg, direction)
        m = p.masks(meta)
        tr = m["source_dev"] & np.isin(act, shared)
        te = m["target_val"] & np.isin(act, shared)
        res = {}
        for mode in ("raw", "z", "rank"):
            f = room_relative(raw, rooms, dev, mode)
            p3 = make_linear_model(1.0).fit(f[tr], act[tr].astype(str)).predict(f[te])
            y2tr = np.where(act[tr] == "walk", "walk", "still")
            y2te = np.where(act[te] == "walk", "walk", "still")
            p2 = make_linear_model(1.0).fit(f[tr], y2tr).predict(f[te])
            res[mode] = {"3class_macro_f1": macro_f1(act[te], p3, shared),
                         "3class_per_class": per_class_f1(act[te], p3, shared),
                         "walk_vs_still_macro_f1": macro_f1(y2te, p2, ("still", "walk"))}
        results[direction] = res
        emb = ref.get(direction, {}).get("M1", {}).get("shared_only_3class_transfer_macro_f1")
        print(f"{direction} (train {p.source} sit/stand/walk dev -> test {p.target} val); "
              f"M1-embedding 3-class reference: {emb if emb is None else round(emb, 3)}")
        for mode, r in res.items():
            pc = ", ".join(f"{c} {v:.2f}" for c, v in r["3class_per_class"].items())
            print(f"  {mode:5s} 3-class {r['3class_macro_f1']:.3f} (chance 0.333; {pc}) | "
                  f"walk vs still {r['walk_vs_still_macro_f1']:.3f} (chance 0.5)")
    return results


def main() -> None:
    cfg = load_task3_config()
    root = Path(cfg["paths"]["task3_dir"])
    x, meta = load_windows(cfg)
    data_sc = data_subcarriers(cfg["common"]["null_subcarriers"], cfg["common"]["n_subcarriers"])
    rooms, dev = meta["room"].to_numpy(), (meta["split"] == "dev").to_numpy()
    act = meta["activity"].to_numpy(dtype=object)
    shared = tuple(sorted(cfg["task3"]["shared"]))
    ref_path = root / "diagnostics.json"
    ref = json.loads(ref_path.read_text(encoding="utf-8")) if ref_path.exists() else {}

    results = {}
    for smooth in SMOOTHING:
        raw = motion_features(x, data_sc, smooth)
        tag = "unsmoothed" if smooth == 0 else f"time-smoothed w={smooth}"
        means = pd.DataFrame(raw, columns=FEATURES).assign(room=rooms, activity=act)[dev & np.isin(act, shared)]
        print(f"\n===== {tag} =====")
        print("dev feature means by room and activity:")
        print(means.groupby(["room", "activity"]).mean().round(4).to_string())
        results[tag] = {"dev_means": {f"{r}/{a}": row.to_dict() for (r, a), row in
                                      means.groupby(["room", "activity"]).mean().iterrows()},
                        "transfer": run_transfer(cfg, meta, raw, rooms, dev, act, shared, ref)}
    out = root / "motion_diagnostic.json"
    out.write_text(json.dumps(results, indent=1, sort_keys=True), encoding="utf-8")
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
