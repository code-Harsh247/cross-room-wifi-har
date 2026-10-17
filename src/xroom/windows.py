"""Turn one capture into fixed-size (length x 256) amplitude windows.

Two stages, the same for both rooms:
1. ``*_sequence``: the whole capture as a (rows, 256) amplitude sequence.
   Room A: keep 80 MHz packets only, then resample the capture (~3.9 s) onto
   ``room_a.resample_rows`` (150) evenly spaced rows.
   Room B: the 500-row file as is.
2. ``slice_windows``: contiguous ``length``-row windows with ``stride``
   (non-overlapping by default). A shorter window therefore always means
   less observation time, in both rooms. Windows never cross a capture boundary.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .clean import fullband_mask, zero_null_subcarriers
from .io_csv import load_amplitude_csv
from .io_pcap import load_nexmon_pcap
from .resample import resample_in_time


@dataclass
class CaptureWindows:
    windows: np.ndarray  # (k, length, 256) float32
    starts: list[int]  # start row of each window within the capture sequence
    info: dict  # per-capture bookkeeping for the data report


def room_a_sequence(path: str, cfg: dict) -> tuple[np.ndarray | None, dict]:
    """Full-band packets of one Room A capture, resampled to ``resample_rows`` rows.
    Returns (None, info) if the capture has too few full-band packets."""
    ra = cfg["room_a"]
    cap = load_nexmon_pcap(path, nfft=cfg["common"]["n_subcarriers"])
    amp = np.abs(cap.csi)
    keep = fullband_mask(amp, tuple(ra["primary_band"]), ra["narrow_energy_frac"])
    info = {
        "n_packets": len(amp),
        "n_fullband": int(keep.sum()),
        "n_dropped_malformed": cap.n_dropped,
        "duration_s": float(cap.timestamps[-1] - cap.timestamps[0]) if len(amp) > 1 else 0.0,
        # recording time, for session (day) analyses only; never used as a feature
        "t_start_unix": float(cap.timestamps[0]) if len(amp) else float("nan"),
    }
    if keep.sum() < ra["min_fullband_packets"]:
        info["excluded_reason"] = f"only {int(keep.sum())} full-band packets"
        return None, info
    seq = resample_in_time(cap.timestamps[keep], amp[keep], ra["resample_rows"])
    return zero_null_subcarriers(seq, cfg["common"]["null_subcarriers"]), info


def room_b_sequence(path: str, cfg: dict) -> tuple[np.ndarray, dict]:
    """The whole Room B file as a (rows, 256) amplitude sequence."""
    amp = load_amplitude_csv(path, cfg["common"]["n_subcarriers"])
    return zero_null_subcarriers(amp, cfg["common"]["null_subcarriers"]), {"n_packets": len(amp)}


def slice_windows(seq: np.ndarray | None, length: int, stride: int | None = None) -> tuple[np.ndarray, list[int]]:
    """Contiguous windows of ``length`` rows every ``stride`` rows (default: non-overlapping)."""
    stride = stride or length
    n_sub = 256 if seq is None else seq.shape[1]
    if seq is None or len(seq) < length:
        return np.zeros((0, length, n_sub), np.float32), []
    starts = list(range(0, len(seq) - length + 1, stride))
    return np.stack([seq[s : s + length] for s in starts]), starts


def capture_sequence(room: str, path: str, cfg: dict) -> tuple[np.ndarray | None, dict]:
    return room_a_sequence(path, cfg) if room == "A" else room_b_sequence(path, cfg)


def capture_windows(room: str, path: str, cfg: dict, length: int, stride: int | None = None) -> CaptureWindows:
    seq, info = capture_sequence(room, path, cfg)
    windows, starts = slice_windows(seq, length, stride)
    if seq is not None and not starts:
        info["excluded_reason"] = f"only {len(seq)} rows"
    return CaptureWindows(windows, starts, info)
