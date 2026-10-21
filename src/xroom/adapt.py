"""Task 3 domain adaptation building blocks (PRD §6, FR-4).

Everything here uses unlabelled development data only. No function takes activity labels,
except prototype refinement, which takes the support labels the access guard allows.
"""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import torch
from torch import nn

from .encoder import FrozenEncoder
from .ssl_data import NormStats, apply_norm, prepare


def adabn(encoder: FrozenEncoder, x: np.ndarray, meta: pd.DataFrame, room: str, batch: int = 256) -> FrozenEncoder:
    """AdaBN (M3): a copy of ``encoder`` whose BatchNorm running statistics are re-estimated on
    the unlabelled **development** windows of one room. Weights are unchanged, and the input
    normalization (``encoder.stats``) is kept. The original encoder is not modified.

    Running stats are a cumulative average over batches (``momentum=None``) of the room's dev
    windows, in a fixed order, so the result is deterministic.
    """
    rows = np.where(((meta["room"] == room) & (meta["split"] == "dev")).to_numpy())[0]
    if len(rows) == 0:
        raise ValueError(f"no development windows for room {room!r}")
    if (meta["split"].to_numpy()[rows] != "dev").any():  # defensive: never validation
        raise AssertionError("AdaBN must only see development windows")
    model = copy.deepcopy(encoder.model)
    bns = [m for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    if not bns:
        raise ValueError("encoder has no BatchNorm layers")
    for bn in bns:
        bn.reset_running_stats()
        bn.momentum = None  # cumulative moving average over all batches
    model.train()  # BatchNorm updates running stats in train mode; the encoder has no dropout
    with torch.no_grad():
        for i in range(0, len(rows), batch):
            xb = torch.from_numpy(apply_norm(np.asarray(x[rows[i : i + batch]]), encoder.stats))
            model(xb.to(encoder.device))
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return FrozenEncoder(model, encoder.stats, encoder.device, encoder.run_dir)


# ---------- embedding-level alignment ----------

def per_room_standardize(f: np.ndarray, rooms: np.ndarray, fit_mask: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """M1: standardize each room's rows with that room's mean and std, estimated on the
    ``fit_mask`` rows of the room (unlabelled development rows). Every row of the room,
    including validation rows, is transformed with those statistics."""
    f = np.asarray(f, dtype=np.float64)
    rooms, fit_mask = np.asarray(rooms), np.asarray(fit_mask, dtype=bool)
    out = np.empty_like(f)
    for room in np.unique(rooms):
        in_room = rooms == room
        fit = in_room & fit_mask
        if not fit.any():
            raise ValueError(f"no fit rows for room {room!r}")
        mu, sd = f[fit].mean(0), f[fit].std(0)
        out[in_room] = (f[in_room] - mu) / np.maximum(sd, eps)
    return out.astype(np.float32)


def _sym_power(c: np.ndarray, power: float) -> np.ndarray:
    w, v = np.linalg.eigh(c)
    return (v * np.maximum(w, 1e-12) ** power) @ v.T


def coral_fit(source: np.ndarray, target: np.ndarray, eps: float = 1.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """M2: CORAL (Sun et al., 2016). Fit on unlabelled source and target rows. Returns
    (A, mu_s, mu_t) such that ``(x - mu_s) @ A + mu_t`` whitens source rows with the source
    covariance and re-colours them with the target covariance (both + eps * I)."""
    source, target = np.asarray(source, np.float64), np.asarray(target, np.float64)
    d = source.shape[1]
    cs = np.cov(source, rowvar=False) + eps * np.eye(d)
    ct = np.cov(target, rowvar=False) + eps * np.eye(d)
    return _sym_power(cs, -0.5) @ _sym_power(ct, 0.5), source.mean(0), target.mean(0)


def coral_apply(x: np.ndarray, fit: tuple[np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
    a, mu_s, mu_t = fit
    return ((np.asarray(x, np.float64) - mu_s) @ a + mu_t).astype(np.float32)


# ---------- transductive prototype refinement ----------

def _unit(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def refine_prototypes(protos: np.ndarray, unlabeled: np.ndarray, support: np.ndarray | None = None,
                      support_idx: np.ndarray | None = None, iters: int = 10, temperature: float = 0.1) -> np.ndarray:
    """M4: soft k-means on the unit sphere, starting from ``protos`` (k, d).

    Each iteration assigns unlabelled rows softly (softmax of cosine similarity / temperature)
    and recomputes every prototype as the weighted mean of its unlabelled rows plus its pinned
    support rows (``support`` (m, d) with class indices ``support_idx``; weight 1 each, hard).
    No label other than the support's is used. Deterministic."""
    p = _unit(np.asarray(protos, np.float64))
    u = _unit(np.asarray(unlabeled, np.float64))
    k = len(p)
    s_sum = np.zeros_like(p)
    s_cnt = np.zeros(k)
    if support is not None and len(support):
        s = _unit(np.asarray(support, np.float64))
        np.add.at(s_sum, np.asarray(support_idx), s)
        s_cnt = np.bincount(np.asarray(support_idx), minlength=k).astype(float)
    for _ in range(iters):
        logits = (u @ p.T) / temperature
        logits -= logits.max(1, keepdims=True)
        r = np.exp(logits)
        r /= r.sum(1, keepdims=True)
        num = r.T @ u + s_sum
        den = r.sum(0) + s_cnt
        p = _unit(num / np.maximum(den, 1e-12)[:, None])
    return p.astype(np.float32)


# ---------- input-level per-room normalization (M6) ----------

def room_input_stats(stats: NormStats, x: np.ndarray, meta: pd.DataFrame, room: str, chunk: int = 256) -> NormStats:
    """M6: the encoder's input normalization re-fitted on one room's unlabelled **development**
    windows: per-subcarrier mean and std of the (row-normalized) input over all rows of that
    room's dev windows. Same flags and subcarriers as ``stats``."""
    rows = np.where(((meta["room"] == room) & (meta["split"] == "dev")).to_numpy())[0]
    if len(rows) == 0:
        raise ValueError(f"no development windows for room {room!r}")
    total = np.zeros(len(stats.data_sc))
    total_sq = np.zeros(len(stats.data_sc))
    n = 0
    for i in range(0, len(rows), chunk):
        d = prepare(np.asarray(x[rows[i : i + chunk]]), stats.data_sc, stats.row, stats.log).astype(np.float64)
        d = d.reshape(-1, d.shape[-1])
        total += d.sum(0)
        total_sq += (d ** 2).sum(0)
        n += len(d)
    mean = total / n
    std = np.sqrt(np.maximum(total_sq / n - mean ** 2, 0.0))
    return NormStats(stats.data_sc, mean.astype(np.float32), np.maximum(std, 1e-6).astype(np.float32),
                     stats.row, stats.log, stats.standardize)


# ---------- balanced assignment (post-hoc idea 2) ----------

def sinkhorn(scores: np.ndarray, iters: int = 100, row_marginal: np.ndarray | None = None,
             col_marginal: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Entropic optimal transport in the log domain. ``scores`` (n, k) are log-kernel values
    (e.g. similarity / temperature). Marginals default to uniform rows (1/n) and uniform
    columns (1/k). Returns (plan (n, k), row potentials f (n,), column potentials g (k,)), with
    plan = exp(scores + f[:, None] + g[None, :]). The column potentials act as per-class score
    offsets that balance the class assignment."""
    from scipy.special import logsumexp
    s = np.asarray(scores, dtype=np.float64)
    n, k = s.shape
    log_r = np.log(np.full(n, 1.0 / n) if row_marginal is None else np.asarray(row_marginal, np.float64))
    log_c = np.log(np.full(k, 1.0 / k) if col_marginal is None else np.asarray(col_marginal, np.float64))
    f, g = np.zeros(n), np.zeros(k)
    for _ in range(iters):
        f = log_r - logsumexp(s + g[None, :], axis=1)
        g = log_c - logsumexp(s + f[:, None], axis=0)
    return np.exp(s + f[:, None] + g[None, :]), f, g
