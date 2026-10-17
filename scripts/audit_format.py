"""Format-leak audit: can a trivial, non-physical statistic tell the rooms apart?

For each window we compute simple scalar statistics and report how well each one
alone separates Room A from Room B (ROC AUC, folded so 0.5 = useless, 1.0 = perfect).
Statistics with AUC near 1 are format differences that a room classifier could exploit.
Uses development windows only.

Usage:  python scripts/audit_format.py [--length 150]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from xroom.config import PROJECT_ROOT, load_config  # noqa: E402
from xroom.dataset import load_windows  # noqa: E402


def _lag1_corr(a: np.ndarray, axis: int) -> np.ndarray:
    """Mean lag-1 correlation along ``axis`` (1 = time, 2 = frequency) for each window."""
    a = a - a.mean(axis=axis, keepdims=True)
    x = np.take(a, range(a.shape[axis] - 1), axis=axis)
    y = np.take(a, range(1, a.shape[axis]), axis=axis)
    num = (x * y).sum(axis)
    den = np.sqrt((x * x).sum(axis) * (y * y).sum(axis)) + 1e-9
    return (num / den).mean(axis=1)


def window_stats(x: np.ndarray, data_sc: np.ndarray) -> pd.DataFrame:
    """x: (n, T, 256) float32 amplitude windows."""
    d = x[:, :, data_sc]
    mean = d.mean(axis=(1, 2))
    return pd.DataFrame(
        {
            "mean amplitude": mean,
            "99th percentile": np.percentile(d, 99, axis=(1, 2)),
            "zero fraction": (d == 0).mean(axis=(1, 2)),
            "integer fraction": (d == np.round(d)).mean(axis=(1, 2)),
            "frequency smoothness (lag-1 corr)": _lag1_corr(d, axis=2),
            "time smoothness (lag-1 corr)": _lag1_corr(d, axis=1),
            "coefficient of variation over time": (d.std(axis=1) / (d.mean(axis=1) + 1e-9)).mean(axis=1),
            "peak / mean (per window)": d.max(axis=(1, 2)) / (mean + 1e-9),
        }
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--length", type=int, default=None)
    args = ap.parse_args()
    cfg = load_config()
    x, meta = load_windows(cfg, args.length)
    dev = meta[meta["split"] == "dev"]
    null = set(cfg["common"]["null_subcarriers"])
    data_sc = np.array([i for i in range(x.shape[2]) if i not in null])

    stats = pd.concat(
        [window_stats(np.asarray(x[i : i + 500], dtype=np.float32), data_sc)
         for i in range(0, len(x), 500)],
        ignore_index=True,
    ).iloc[dev.index]
    is_b = (dev["room"] == "B").to_numpy()

    rows = []
    for col in stats.columns:
        v = stats[col].to_numpy()
        auc = roc_auc_score(is_b, v)
        rows.append({"statistic": col, "Room A median": np.median(v[~is_b]),
                     "Room B median": np.median(v[is_b]), "separability (AUC)": max(auc, 1 - auc)})
    table = pd.DataFrame(rows)
    print(table.to_string(index=False, float_format=lambda f: f"{f:.3f}"))
    out = PROJECT_ROOT / "outputs" / f"format_audit_L{x.shape[1]}.csv"
    out.parent.mkdir(exist_ok=True)
    table.to_csv(out, index=False)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
