"""Window -> feature vector.

* ``spectrum_features``: the room fingerprint, a time-averaged amplitude spectrum over the
  data subcarriers, with optional per-row gain removal.
* ``format_stats``: scalar statistics that describe the data *format* rather than the room
  (scale, smoothness, integer-ness, ...). Used for the leak audit and the format-only control.

All functions are stateless (nothing is fitted), so they can be applied to any split.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

SPECTRUM_VARIANTS = ("none", "row", "log")
_EPS = 1e-6


def data_subcarriers(null_idx: list[int], n_subcarriers: int = 256) -> np.ndarray:
    """Indices of subcarriers that carry channel information."""
    null = set(null_idx)
    return np.array([i for i in range(n_subcarriers) if i not in null])


def spectrum_features(x: np.ndarray, data_sc: np.ndarray, variant: str = "row") -> np.ndarray:
    """Time-averaged spectrum of each window.

    Args:
        x: (n, T, 256) amplitude windows.
        data_sc: subcarrier indices to keep.
        variant:
            "none": mean amplitude over time (keeps absolute scale).
            "row":  divide every row (packet) by its own mean first, then average.
                    Removes per-packet gain (Nexmon autoscaling / AGC), keeps spectral shape.
            "log":  log(amplitude + 1) minus the row's mean, then average. Approximately
                    gain invariant (exactly so for amplitudes >> 1; the +1 keeps zeros finite)
                    and compresses deep fades.
    Returns:
        (n, len(data_sc)) float32.
    """
    d = np.asarray(x[:, :, data_sc], dtype=np.float32)
    if variant == "none":
        f = d.mean(axis=1)
    elif variant == "row":
        f = (d / (d.mean(axis=2, keepdims=True) + _EPS)).mean(axis=1)
    elif variant == "log":
        ld = np.log(d + 1.0)
        f = (ld - ld.mean(axis=2, keepdims=True)).mean(axis=1)
    else:
        raise ValueError(f"unknown variant {variant!r}; choose from {SPECTRUM_VARIANTS}")
    return f.astype(np.float32)


def _lag1_corr(a: np.ndarray, axis: int) -> np.ndarray:
    """Mean lag-1 correlation along ``axis`` (1 = time, 2 = frequency) for each window."""
    a = a - a.mean(axis=axis, keepdims=True)
    x = np.take(a, range(a.shape[axis] - 1), axis=axis)
    y = np.take(a, range(1, a.shape[axis]), axis=axis)
    num = (x * y).sum(axis)
    den = np.sqrt((x * x).sum(axis) * (y * y).sum(axis)) + 1e-9
    return (num / den).mean(axis=1)


def format_stats(x: np.ndarray, data_sc: np.ndarray) -> pd.DataFrame:
    """Per-window scalar statistics of the data format (not of the room)."""
    d = np.asarray(x[:, :, data_sc], dtype=np.float32)
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


def in_chunks(fn: Callable[[np.ndarray], np.ndarray | pd.DataFrame], x: np.ndarray, chunk: int = 500):
    """Apply ``fn`` to ``x`` (e.g. a float16 memmap) in chunks to bound memory use."""
    parts = [fn(x[i : i + chunk]) for i in range(0, len(x), chunk)]
    if isinstance(parts[0], pd.DataFrame):
        return pd.concat(parts, ignore_index=True)
    return np.concatenate(parts)
