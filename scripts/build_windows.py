"""Decode both rooms and cache fixed-size CSI windows.

Usage:
    python scripts/build_windows.py                 # length from config (150)
    python scripts/build_windows.py --length 50     # e.g. for the window-length study
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from xroom.config import load_config  # noqa: E402
from xroom.dataset import build_windows, cache_paths  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--length", type=int, default=None, help="rows per window")
    ap.add_argument("--stride-b", type=int, default=None, help="Room B stride (default: = config)")
    args = ap.parse_args()

    cfg = load_config(args.config) if args.config else load_config()
    length = args.length or cfg["windows"]["length"]
    t0 = time.time()
    build_windows(cfg, length, args.stride_b)

    caps = pd.read_csv(cache_paths(cfg, length)[2])
    print(f"done in {time.time() - t0:.0f}s\n")
    print("captures / windows per room, split, activity:")
    print(
        caps.groupby(["room", "split", "activity"])
        .agg(captures=("capture_id", "size"), windows=("n_windows", "sum"))
        .to_string()
    )
    excluded = caps[caps["n_windows"] == 0]
    print(f"\nexcluded captures: {len(excluded)}")
    if len(excluded):
        print(excluded[["capture_id", "split", "excluded_reason"]].to_string(index=False))


if __name__ == "__main__":
    main()
