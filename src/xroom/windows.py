"""Turn one capture into fixed-size (length x 256) amplitude windows.

Room A: keep 80 MHz packets only, then resample the whole capture (~3.9 s) onto
``length`` rows -> exactly one window per capture.
Room B: cut the 500-row file into windows of ``length`` rows with ``stride``;
windows never cross a capture boundary.
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
    starts: list[int]  # start row of each window (Room A: always 0)
    info: dict  # per-capture bookkeeping for the data report


def room_a_windows(path: str, cfg: dict, length: int) -> CaptureWindows:
    ra = cfg["room_a"]
    cap = load_nexmon_pcap(path, nfft=cfg["common"]["n_subcarriers"])
    amp = np.abs(cap.csi)
    keep = fullband_mask(amp, tuple(ra["primary_band"]), ra["narrow_energy_frac"])
    info = {
        "n_packets": len(amp),
        "n_fullband": int(keep.sum()),
        "n_dropped_malformed": cap.n_dropped,
        "duration_s": float(cap.timestamps[-1] - cap.timestamps[0]) if len(amp) > 1 else 0.0,
    }
    if keep.sum() < ra["min_fullband_packets"]:
        info["excluded_reason"] = f"only {int(keep.sum())} full-band packets"
        return CaptureWindows(np.zeros((0, length, amp.shape[1]), np.float32), [], info)
    x = resample_in_time(cap.timestamps[keep], amp[keep], length)
    x = zero_null_subcarriers(x, cfg["common"]["null_subcarriers"])
    return CaptureWindows(x[None], [0], info)


def room_b_windows(path: str, cfg: dict, length: int, stride: int) -> CaptureWindows:
    amp = load_amplitude_csv(path, cfg["common"]["n_subcarriers"])
    amp = zero_null_subcarriers(amp, cfg["common"]["null_subcarriers"])
    starts = list(range(0, len(amp) - length + 1, stride))
    info = {"n_packets": len(amp)}
    if not starts:
        info["excluded_reason"] = f"only {len(amp)} rows"
        return CaptureWindows(np.zeros((0, length, amp.shape[1]), np.float32), [], info)
    return CaptureWindows(np.stack([amp[s : s + length] for s in starts]), starts, info)
