"""Task 1 window-length study (M6): room accuracy vs window length.

For each window length L (rows), windows are contiguous, non-overlapping L-row slices
of every capture (Room A: 150 resampled rows ~ 3.9 s; Room B: 500 rows, rate unknown),
so a shorter window always means less observation time. At every L the M5 protocol is
repeated: per-row normalized time-averaged spectrum, class-balanced logistic regression,
C tuned on dev holdout by log-loss, refit on all dev, scored once on validation.

Curves (validation balanced accuracy, 95% capture-level bootstrap interval):
  all activities | shared activities (sit/stand/walk) | still vs moving activities |
  format-only control | train sit+stand -> test walk

Usage:  python scripts/task1_window_length.py [--lengths 5 10 25 50 100 150]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from functools import partial
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from xroom.config import PROJECT_ROOT, load_config  # noqa: E402
from xroom.dataset import load_sequences, windows_from_sequences  # noqa: E402
from xroom.eval import SHARED_ACTIVITIES, bootstrap_balanced_accuracy, room_metrics  # noqa: E402
from xroom.features import data_subcarriers, format_stats, in_chunks, spectrum_features  # noqa: E402
from xroom.room_classifier import make_linear_model  # noqa: E402

STILL_ACTIVITIES = ("empty", "lying", "sit", "stand")
MOVING_ACTIVITIES = ("walk", "fall", "sitdown", "standup")


def fit_predict(train_x, train_y, test_x, C):
    model = make_linear_model(C).fit(train_x, train_y)
    return model.predict_proba(test_x)[:, list(model.classes_).index("B")]


def scored(meta: pd.DataFrame, mask: np.ndarray, prob_b: np.ndarray, boot: dict) -> dict:
    """Balanced accuracy + bootstrap interval on the windows selected by ``mask``."""
    sub = meta[mask]
    y = sub["room"].to_numpy()
    m = room_metrics(y, prob_b)
    lo, hi = bootstrap_balanced_accuracy(sub, np.where(prob_b >= 0.5, "B", "A") == y, **boot)
    return {"balanced_accuracy": m["balanced_accuracy"], "ci_low": lo, "ci_high": hi,
            "recall_A": m["recall_A"], "recall_B": m["recall_B"], "n_windows": int(len(sub))}


def run_length(L: int, table, seqs, cfg, data_sc, boot) -> dict:
    x, meta, _ = windows_from_sequences(table, seqs, L)
    f = in_chunks(partial(spectrum_features, data_sc=data_sc, variant=cfg["room_classifier"]["feature_variant"]), x)
    fmt = in_chunks(partial(format_stats, data_sc=data_sc), x).to_numpy()
    del x
    y = meta["room"].to_numpy()
    act = meta["activity"].to_numpy()
    dev, val = (meta["split"] == "dev").to_numpy(), (meta["split"] == "val").to_numpy()
    tr, ho = (meta["dev_fold"] == "train").to_numpy(), (meta["dev_fold"] == "holdout").to_numpy()

    # tune C on dev holdout (same rule as M5)
    losses = {C: room_metrics(y[ho], fit_predict(f[tr], y[tr], f[ho], C))["log_loss"]
              for C in cfg["room_classifier"]["c_grid"]}
    C = min(losses, key=lambda c: (losses[c], c))

    p_val = fit_predict(f[dev], y[dev], f[val], C)
    val_meta = meta[val]
    in_val = lambda acts: np.isin(act[val], acts)  # noqa: E731
    res = {
        "length": L, "C": C,
        "n_windows_dev": {r: int((dev & (y == r)).sum()) for r in ("A", "B")},
        "n_windows_val": {r: int((val & (y == r)).sum()) for r in ("A", "B")},
        "all": scored(val_meta, np.ones(val.sum(), bool), p_val, boot),
        "shared": scored(val_meta, in_val(SHARED_ACTIVITIES), p_val[in_val(SHARED_ACTIVITIES)], boot),
        "still": scored(val_meta, in_val(STILL_ACTIVITIES), p_val[in_val(STILL_ACTIVITIES)], boot),
        "moving": scored(val_meta, in_val(MOVING_ACTIVITIES), p_val[in_val(MOVING_ACTIVITIES)], boot),
    }
    p_fmt = fit_predict(fmt[dev], y[dev], fmt[val], 1.0)
    res["format_only"] = scored(val_meta, np.ones(val.sum(), bool), p_fmt, boot)
    still_tr = dev & np.isin(act, ("sit", "stand"))
    walk_val = val & (act == "walk")
    p_walk = fit_predict(f[still_tr], y[still_tr], f[walk_val], C)
    res["sit_stand_to_walk"] = scored(meta, walk_val, p_walk, boot)
    return res


CURVES = [
    ("all", "all activities", "o-"),
    ("shared", "shared activities (sit/stand/walk)", "s-"),
    ("format_only", "format-only control", "^--"),
    ("sit_stand_to_walk", "trained sit+stand, tested walk", "d-"),
]


def plot(results: list[dict], path: Path) -> None:
    Ls = [r["length"] for r in results]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, curves, title in [
        (axes[0], CURVES, "Room accuracy vs window length"),
        (axes[1], [("still", "still activities", "o-"), ("moving", "moving activities", "s-")], "Still vs moving windows"),
    ]:
        for key, label, style in curves:
            mid = [r[key]["balanced_accuracy"] for r in results]
            ax.plot(Ls, mid, style, label=label, ms=4)
            ax.fill_between(Ls, [r[key]["ci_low"] for r in results], [r[key]["ci_high"] for r in results], alpha=0.15)
        ax.set_ylim(0.75, 1.005)  # chance (0.5) is far below every curve
        ax.set_xscale("log")
        ax.set_xticks(Ls, [str(L) for L in Ls])
        ax.set_xlabel("window length (rows)")
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("val balanced accuracy (95% CI)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    cfg = load_config()
    st = cfg["window_length_study"]
    ap = argparse.ArgumentParser()
    ap.add_argument("--lengths", type=int, nargs="+", default=st["lengths"])
    ap.add_argument("--plot-only", action="store_true", help="redraw the figure from saved results.json")
    args = ap.parse_args()
    boot = {"reps": st["bootstrap_reps"], "seed": st["bootstrap_seed"]}
    out_dir = PROJECT_ROOT / "outputs" / "task1" / "window_length"
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.plot_only:
        plot(json.loads((out_dir / "results.json").read_text(encoding="utf-8")), out_dir / "room_acc_vs_length.png")
        return

    table, seqs = load_sequences(cfg)
    data_sc = data_subcarriers(cfg["common"]["null_subcarriers"], cfg["common"]["n_subcarriers"])
    results = []
    for L in sorted(args.lengths):
        t0 = time.time()
        r = run_length(L, table, seqs, cfg, data_sc, boot)
        results.append(r)
        print(f"L={L:>3}  C={r['C']:<7} all {r['all']['balanced_accuracy']:.3f}  shared {r['shared']['balanced_accuracy']:.3f}  "
              f"still {r['still']['balanced_accuracy']:.3f}  moving {r['moving']['balanced_accuracy']:.3f}  "
              f"format-only {r['format_only']['balanced_accuracy']:.3f}  sit+stand->walk {r['sit_stand_to_walk']['balanced_accuracy']:.3f}  "
              f"(val windows A/B {r['n_windows_val']['A']}/{r['n_windows_val']['B']}, {time.time() - t0:.0f}s)", flush=True)

    (out_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    rows = [{"length": r["length"], "C": r["C"], "curve": k, **{m: r[k][m] for m in
             ("balanced_accuracy", "ci_low", "ci_high", "recall_A", "recall_B", "n_windows")}}
            for r in results for k in ("all", "shared", "still", "moving", "format_only", "sit_stand_to_walk")]
    pd.DataFrame(rows).to_csv(out_dir / "results.csv", index=False)
    plot(results, out_dir / "room_acc_vs_length.png")
    print(f"\nsaved to {out_dir}")


if __name__ == "__main__":
    main()
