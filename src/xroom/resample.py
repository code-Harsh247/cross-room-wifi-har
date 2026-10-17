"""Resample an irregularly sampled capture onto a fixed number of rows."""
from __future__ import annotations

import numpy as np


def resample_in_time(t: np.ndarray, x: np.ndarray, length: int) -> np.ndarray:
    """Linearly interpolate rows of ``x`` (sampled at times ``t``) onto ``length``
    evenly spaced times spanning the first to the last sample.

    Packets that share a timestamp are averaged first so interpolation is well defined.
    """
    if len(t) != len(x):
        raise ValueError("t and x must have the same number of rows")
    if len(t) < 2:
        raise ValueError("need at least 2 samples to resample")
    order = np.argsort(t, kind="stable")
    t, x = t[order], x[order].astype(np.float64)
    uniq, inverse, counts = np.unique(t, return_inverse=True, return_counts=True)
    if len(uniq) < len(t):
        summed = np.zeros((len(uniq), x.shape[1]))
        np.add.at(summed, inverse, x)
        x = summed / counts[:, None]
        t = uniq
    grid = np.linspace(t[0], t[-1], length)
    idx = np.clip(np.searchsorted(t, grid, side="right") - 1, 0, len(t) - 2)
    w = ((grid - t[idx]) / (t[idx + 1] - t[idx]))[:, None]
    return ((1 - w) * x[idx] + w * x[idx + 1]).astype(np.float32)
