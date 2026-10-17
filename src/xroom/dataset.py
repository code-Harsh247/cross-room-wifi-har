"""Build and load the cached window dataset."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .catalog import build_catalog
from .splits import assign_splits
from .windows import room_a_windows, room_b_windows


def cache_paths(cfg: dict, length: int) -> tuple[Path, Path, Path]:
    d = Path(cfg["paths"]["cache_dir"])
    return d / f"windows_L{length}.npy", d / f"windows_L{length}_meta.csv", d / f"captures_L{length}.csv"


def build_windows(cfg: dict, length: int | None = None, stride_b: int | None = None, log=print) -> None:
    """Decode every capture, cut windows, and write them to the cache.

    Outputs:
        windows_L{length}.npy        (N, length, 256) float16 amplitude windows
        windows_L{length}_meta.csv   one row per window (capture_id, room, activity, split, ...)
        captures_L{length}.csv       one row per capture with packet counts / exclusion reasons
    """
    length = length or cfg["windows"]["length"]
    stride_b = stride_b or cfg["windows"]["room_b_stride"]
    catalog = assign_splits(build_catalog(cfg), cfg)

    windows, meta, captures = [], [], []
    for i, row in enumerate(catalog.itertuples(index=False)):
        if row.room == "A":
            cw = room_a_windows(row.path, cfg, length)
        else:
            cw = room_b_windows(row.path, cfg, length, stride_b)
        base = {
            "capture_id": row.capture_id,
            "room": row.room,
            "activity": row.activity,
            "split": row.split,
            "dev_fold": row.dev_fold,
        }
        captures.append({**base, "n_windows": len(cw.starts), **cw.info})
        for s, w in zip(cw.starts, cw.windows):
            windows.append(w.astype(np.float16))
            meta.append({**base, "start": s})
        if (i + 1) % 250 == 0:
            log(f"  {i + 1}/{len(catalog)} captures")

    x_path, meta_path, cap_path = cache_paths(cfg, length)
    x_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(x_path, np.stack(windows))
    pd.DataFrame(meta).to_csv(meta_path, index_label="window_idx")
    pd.DataFrame(captures).to_csv(cap_path, index=False)
    log(f"wrote {len(windows)} windows -> {x_path}")


def load_windows(cfg: dict, length: int | None = None, mmap: bool = True) -> tuple[np.ndarray, pd.DataFrame]:
    """Return (windows, meta). Windows are float16 on disk; cast slices to float32 before use."""
    length = length or cfg["windows"]["length"]
    x_path, meta_path, _ = cache_paths(cfg, length)
    x = np.load(x_path, mmap_mode="r" if mmap else None)
    meta = pd.read_csv(meta_path, index_col="window_idx")
    if len(meta) != len(x):
        raise RuntimeError("cache is inconsistent; rebuild with scripts/build_windows.py")
    return x, meta
