"""Build and load the cached datasets.

* sequences: every capture as one (rows, 256) float16 sequence (Room A resampled to
  150 rows, Room B 500 rows), plus the capture table. Decoding happens only here.
* windows: fixed-length windows cut from the sequences, for one window length.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .catalog import build_catalog
from .splits import assign_splits
from .windows import capture_sequence, slice_windows

CAPTURE_COLUMNS = ["capture_id", "room", "activity", "split", "dev_fold"]


def cache_paths(cfg: dict, length: int) -> tuple[Path, Path, Path]:
    d = Path(cfg["paths"]["cache_dir"])
    return d / f"windows_L{length}.npy", d / f"windows_L{length}_meta.csv", d / f"captures_L{length}.csv"


def sequences_path(cfg: dict) -> Path:
    return Path(cfg["paths"]["cache_dir"]) / "sequences.joblib"


def build_sequences(cfg: dict, log=print) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Decode every capture once. Returns (captures table, {capture_id: sequence}).
    Excluded captures appear in the table with ``excluded_reason`` and no sequence."""
    catalog = assign_splits(build_catalog(cfg), cfg)
    captures, seqs = [], {}
    for i, row in enumerate(catalog.itertuples(index=False)):
        seq, info = capture_sequence(row.room, row.path, cfg)
        captures.append({c: getattr(row, c) for c in CAPTURE_COLUMNS} | info)
        if seq is not None:
            seqs[row.capture_id] = seq.astype(np.float16)
        if (i + 1) % 250 == 0:
            log(f"  decoded {i + 1}/{len(catalog)} captures")
    table = pd.DataFrame(captures)
    path = sequences_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump((table, seqs), path)
    return table, seqs


def load_sequences(cfg: dict, rebuild: bool = False, log=print) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    path = sequences_path(cfg)
    if rebuild or not path.exists():
        return build_sequences(cfg, log)
    return joblib.load(path)


def windows_from_sequences(table: pd.DataFrame, seqs: dict[str, np.ndarray], length: int,
                           stride: int | None = None) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    """Cut every sequence into windows. Returns (windows float16, window meta, capture table + n_windows)."""
    windows, meta, n_windows = [], [], []
    for row in table.itertuples(index=False):
        w, starts = slice_windows(seqs.get(row.capture_id), length, stride)
        n_windows.append(len(starts))
        windows.append(w.astype(np.float16))
        base = {c: getattr(row, c) for c in CAPTURE_COLUMNS}
        meta += [{**base, "start": s} for s in starts]
    caps = table.assign(n_windows=n_windows)
    return np.concatenate(windows), pd.DataFrame(meta), caps


def build_windows(cfg: dict, length: int | None = None, stride: int | None = None,
                  rebuild_sequences: bool = False, log=print) -> None:
    """Cut windows of ``length`` rows (non-overlapping unless ``stride`` is given) and cache them.

    Outputs:
        windows_L{length}.npy        (N, length, 256) float16 amplitude windows
        windows_L{length}_meta.csv   one row per window (capture_id, room, activity, split, ...)
        captures_L{length}.csv       one row per capture with packet counts / exclusion reasons
    """
    length = length or cfg["windows"]["length"]
    stride = stride or cfg["windows"].get("stride")
    table, seqs = load_sequences(cfg, rebuild_sequences, log)
    x, meta, caps = windows_from_sequences(table, seqs, length, stride)
    x_path, meta_path, cap_path = cache_paths(cfg, length)
    x_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(x_path, x)
    meta.to_csv(meta_path, index_label="window_idx")
    caps.to_csv(cap_path, index=False)
    log(f"wrote {len(x)} windows -> {x_path}")


def load_windows(cfg: dict, length: int | None = None, mmap: bool = True) -> tuple[np.ndarray, pd.DataFrame]:
    """Return (windows, meta). Windows are float16 on disk; cast slices to float32 before use."""
    length = length or cfg["windows"]["length"]
    x_path, meta_path, _ = cache_paths(cfg, length)
    x = np.load(x_path, mmap_mode="r" if mmap else None)
    meta = pd.read_csv(meta_path, index_col="window_idx")
    if len(meta) != len(x):
        raise RuntimeError("cache is inconsistent; rebuild with scripts/build_windows.py")
    return x, meta
