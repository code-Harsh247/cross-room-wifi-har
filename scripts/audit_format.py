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
from functools import partial
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from xroom.config import PROJECT_ROOT, load_config  # noqa: E402
from xroom.dataset import load_windows  # noqa: E402
from xroom.features import data_subcarriers, format_stats, in_chunks  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--length", type=int, default=None)
    args = ap.parse_args()
    cfg = load_config()
    x, meta = load_windows(cfg, args.length)
    dev = meta[meta["split"] == "dev"]
    data_sc = data_subcarriers(cfg["common"]["null_subcarriers"], x.shape[2])

    stats = in_chunks(partial(format_stats, data_sc=data_sc), x).iloc[dev.index]
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
