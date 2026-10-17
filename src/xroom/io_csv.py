"""Room B: read amplitude CSI from the Experiment 3 .csv files."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def load_amplitude_csv(path: str | Path, n_subcarriers: int = 256) -> np.ndarray:
    """Return a (rows, n_subcarriers) float32 amplitude array.

    Files have one header row (``Subcarrier_1..256`` or ``0..255``) and no index
    column; columns are kept in file order, which is the same FFT order as Room A.
    """
    x = pd.read_csv(path).to_numpy(dtype=np.float32)
    if x.ndim != 2 or x.shape[1] != n_subcarriers:
        raise ValueError(f"{path}: expected {n_subcarriers} columns, got shape {x.shape}")
    if not np.isfinite(x).all():
        raise ValueError(f"{path}: contains NaN/inf")
    return x
