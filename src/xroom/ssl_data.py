"""Self-supervised training data (Task 2, PRD FR-1).

* Input normalization: keep the 242 data subcarriers, divide each row by its mean
  (per-packet gain / format scale), optionally log(1 + x), then a per-subcarrier z-score
  whose statistics are fitted on development rows only and saved with each encoder.
* ``CropSampler``: random contiguous crops from development capture sequences, never
  crossing a capture, in room-balanced batches. Validation captures are refused outright.

Nothing here reads the ``activity`` column: SSL training is label-free.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .features import data_subcarriers, in_chunks

_EPS = 1e-6


@dataclass
class NormStats:
    """Everything needed to turn raw (T, 256) amplitudes into encoder input (T, 242)."""

    data_sc: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    row: bool = True
    log: bool = False
    standardize: bool = True

    def save(self, path: str | Path) -> None:
        np.savez(path, data_sc=self.data_sc, mean=self.mean, std=self.std,
                 flags=np.array([self.row, self.log, self.standardize]))

    @classmethod
    def load(cls, path: str | Path) -> "NormStats":
        z = np.load(path)
        row, log, std = (bool(v) for v in z["flags"])
        return cls(z["data_sc"], z["mean"], z["std"], row, log, std)


def prepare(x: np.ndarray, data_sc: np.ndarray, row: bool = True, log: bool = False) -> np.ndarray:
    """Stateless part of the normalization: (..., T, 256) -> (..., T, len(data_sc)) float32."""
    d = np.asarray(x[..., data_sc], dtype=np.float32)
    if row:
        d = d / (d.mean(axis=-1, keepdims=True) + _EPS)
    if log:
        d = np.log1p(np.maximum(d, 0.0))
    return d


def fit_norm_stats(prepared: list[np.ndarray], rooms: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Per-subcarrier mean/std over the rows of ``prepared`` (each (T, C)), weighting the two
    rooms equally so Room B's longer sequences do not dominate. Callers pass dev rows only."""
    moments = []
    for room in sorted(set(rooms)):
        rows = np.concatenate([p for p, r in zip(prepared, rooms) if r == room]).astype(np.float64)
        moments.append((rows.mean(0), (rows ** 2).mean(0)))
    m1 = np.mean([m[0] for m in moments], axis=0)
    m2 = np.mean([m[1] for m in moments], axis=0)
    std = np.sqrt(np.maximum(m2 - m1 ** 2, 0.0))
    return m1.astype(np.float32), np.maximum(std, _EPS).astype(np.float32)


def apply_norm(x: np.ndarray, stats: NormStats) -> np.ndarray:
    """Raw (..., T, 256) amplitudes -> normalized (..., T, 242) float32 encoder input."""
    d = prepare(x, stats.data_sc, stats.row, stats.log)
    if stats.standardize:
        d = (d - stats.mean) / stats.std
    return d.astype(np.float32)


def normalize_windows(x: np.ndarray, stats: NormStats, chunk: int = 256) -> np.ndarray:
    """``apply_norm`` over a (possibly memory-mapped float16) window array, in chunks."""
    return in_chunks(lambda c: apply_norm(c, stats), x, chunk)


def assert_dev_only(ids, table: pd.DataFrame) -> None:
    """Hard guard: every capture id must exist and belong to the development split."""
    split = table.set_index("capture_id")["split"]
    ids = list(ids)
    missing = set(ids) - set(split.index)
    if missing:
        raise ValueError(f"unknown capture ids: {sorted(missing)[:5]}")
    bad = [c for c in ids if split[c] != "dev"]
    if bad:
        raise AssertionError(f"validation captures must never enter SSL training: {bad[:5]}")


def dev_capture_ids(table: pd.DataFrame, seqs: dict, fold: str | None = None,
                    exclude: set[str] | None = None) -> list[str]:
    """Development captures that have a sequence, optionally one ``dev_fold`` only."""
    t = table[(table["split"] == "dev") & table["capture_id"].isin(list(seqs))]
    if fold is not None:
        t = t[t["dev_fold"] == fold]
    ids = [c for c in t["capture_id"] if not exclude or c not in exclude]
    assert_dev_only(ids, table)
    return ids


def fit_dev_stats(table: pd.DataFrame, seqs: dict, cfg: dict) -> NormStats:
    """Fit normalization statistics on every development capture (never validation)."""
    n = cfg["normalization"]
    data_sc = data_subcarriers(cfg["common"]["null_subcarriers"], cfg["common"]["n_subcarriers"])
    ids = dev_capture_ids(table, seqs)
    rooms = table.set_index("capture_id").loc[ids, "room"].tolist()
    if n["standardize"]:
        mean, std = fit_norm_stats([prepare(seqs[c], data_sc, n["row"], n["log"]) for c in ids], rooms)
    else:
        mean, std = np.zeros(len(data_sc), np.float32), np.ones(len(data_sc), np.float32)
    return NormStats(data_sc, mean, std, n["row"], n["log"], n["standardize"])


class CropSampler:
    """Room-balanced batches of random contiguous crops from development sequences.

    Each batch has one crop length T drawn from [crop_min, crop_max] (so it stacks), and each
    crop starts at a random row of its capture so it never crosses a capture boundary.
    Sequences are normalized once at construction and kept as float16 to save memory.
    """

    def __init__(self, ids: list[str], table: pd.DataFrame, seqs: dict, stats: NormStats,
                 batch_size: int, crop_min: int, crop_max: int, room_balance: bool = True, seed: int = 0):
        assert_dev_only(ids, table)
        if not ids:
            raise ValueError("no captures to sample from")
        room = table.set_index("capture_id")["room"]
        self.ids = list(ids)
        self.data = [apply_norm(seqs[c], stats).astype(np.float16) for c in self.ids]
        self.rooms = np.array([room[c] for c in self.ids])
        shortest = min(len(d) for d in self.data)
        if crop_max > shortest:
            raise ValueError(f"crop_max={crop_max} exceeds the shortest sequence ({shortest} rows)")
        self.batch_size, self.crop_min, self.crop_max = batch_size, crop_min, crop_max
        self.room_balance = room_balance
        self.by_room = {r: np.flatnonzero(self.rooms == r) for r in sorted(set(self.rooms))}
        self.rng = np.random.default_rng(seed)

    def _pick(self) -> np.ndarray:
        if not self.room_balance or len(self.by_room) == 1:
            return self.rng.integers(0, len(self.ids), self.batch_size)
        groups = list(self.by_room.values())
        sizes = np.full(len(groups), self.batch_size // len(groups))
        sizes[self.rng.permutation(len(groups))[: self.batch_size - sizes.sum()]] += 1
        return self.rng.permutation(np.concatenate([g[self.rng.integers(0, len(g), k)] for g, k in zip(groups, sizes)]))

    def batch(self) -> tuple[np.ndarray, list[str]]:
        """Returns (crops (B, T, C) float32, capture ids of the crops)."""
        idx = self._pick()
        t = int(self.rng.integers(self.crop_min, self.crop_max + 1))
        out = np.empty((len(idx), t, self.data[0].shape[1]), np.float32)
        for k, i in enumerate(idx):
            start = int(self.rng.integers(0, len(self.data[i]) - t + 1))
            out[k] = self.data[i][start : start + t]
        return out, [self.ids[i] for i in idx]

    def batch_rooms(self, ids: list[str]) -> np.ndarray:
        lookup = dict(zip(self.ids, self.rooms))
        return np.array([lookup[c] for c in ids])
