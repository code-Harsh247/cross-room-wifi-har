"""Validity checks and the common format shared by both rooms."""
from __future__ import annotations

import numpy as np


def primary_band_energy_fraction(amp: np.ndarray, band: tuple[int, int]) -> np.ndarray:
    """Share of each row's energy that falls inside subcarriers ``band[0]:band[1]``."""
    energy = amp.astype(np.float64) ** 2
    total = energy.sum(axis=1)
    inside = energy[:, band[0] : band[1]].sum(axis=1)
    return np.divide(inside, total, out=np.ones_like(total), where=total > 0)


def fullband_mask(amp: np.ndarray, band: tuple[int, int], narrow_frac: float) -> np.ndarray:
    """True for 80 MHz packets; False for narrow 20 MHz frames and all-zero packets."""
    return primary_band_energy_fraction(amp, band) < narrow_frac


def zero_null_subcarriers(x: np.ndarray, null_idx: list[int]) -> np.ndarray:
    """Return a copy with DC/guard subcarriers set to 0 (last axis = subcarriers)."""
    out = x.copy()
    out[..., null_idx] = 0
    return out
