"""Task 3 scores (PRD §4, FR-5).

Per episode, on target validation windows:
  overall    macro-F1 over the target label space (shared + target-only); the headline
  exclusive  mean F1 over the target-only classes, from the same predictions
  shared     mean F1 over sit / stand / walk
plus per-class F1 and the confusion matrix. Over the draws of one (direction, method, K):
mean ± std, and a 95% interval of the draw-mean that resamples target validation captures
within each class (windows of a capture are not independent).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .fewshot import Protocol
from .probe import per_class_f1

METRICS = ("overall", "exclusive", "shared")


def episode_scores(y_true, y_pred, protocol: Protocol) -> dict:
    """Scores of one episode. ``y_true``/``y_pred`` are activity names of target val windows."""
    labels = list(protocol.target_labels)
    y_true, y_pred = np.asarray(y_true, dtype=object), np.asarray(y_pred, dtype=object)
    stray = set(y_pred) - set(labels)
    if stray:
        raise ValueError(f"predictions outside the target label space: {sorted(stray)}")
    pc = per_class_f1(y_true, y_pred, labels)
    idx = {c: i for i, c in enumerate(labels)}
    cm = np.zeros((len(labels), len(labels)), dtype=int)
    np.add.at(cm, ([idx[t] for t in y_true], [idx[p] for p in y_pred]), 1)
    return {
        "overall": float(np.mean([pc[c] for c in labels])),
        "exclusive": float(np.mean([pc[c] for c in protocol.target_only])),
        "shared": float(np.mean([pc[c] for c in protocol.shared])),
        "per_class": pc,
        "confusion": {"labels": labels, "matrix": cm.tolist()},
        "n_windows": int(len(y_true)),
    }


def _f1_from_confusion(cm: np.ndarray) -> np.ndarray:
    """Per-class F1 from (..., k, k) confusion counts (rows = true, cols = predicted)."""
    tp = np.diagonal(cm, axis1=-2, axis2=-1)
    denom = cm.sum(-1) + cm.sum(-2)
    return np.divide(2 * tp, denom, out=np.zeros(tp.shape), where=denom > 0)


def bootstrap_draw_mean(y_true, preds: np.ndarray, captures, protocol: Protocol,
                        reps: int = 1000, seed: int = 0, level: float = 0.95) -> dict:
    """95% intervals of the draw-mean of each metric. ``preds`` is (draws, n) activity names for
    the same n target val windows. Captures are resampled with replacement within each true
    class, the same resample for every draw."""
    labels = list(protocol.target_labels)
    idx = {c: i for i, c in enumerate(labels)}
    k = len(labels)
    t = np.array([idx[v] for v in y_true])
    p = np.vectorize(idx.get)(np.asarray(preds, dtype=object))
    caps = pd.Series(np.asarray(captures))
    cap_codes, cap_ids = pd.factorize(caps)
    n_caps = len(cap_ids)
    # per-capture confusion counts for every draw: (draws, captures, k, k)
    per_cap = np.zeros((p.shape[0], n_caps, k, k))
    for d in range(p.shape[0]):
        np.add.at(per_cap[d], (cap_codes, t, p[d]), 1)
    cap_class = np.zeros(n_caps, dtype=int)
    cap_class[cap_codes] = t
    excl = [idx[c] for c in protocol.target_only]
    shared = [idx[c] for c in protocol.shared]
    rng = np.random.default_rng(seed)
    stats = np.zeros((reps, 3))
    groups = [np.where(cap_class == c)[0] for c in range(k)]
    for r in range(reps):
        w = np.zeros(n_caps)
        for g in groups:
            if len(g):
                np.add.at(w, rng.choice(g, size=len(g), replace=True), 1)
        f1 = _f1_from_confusion(np.einsum("c,dcij->dij", w, per_cap))  # (draws, k)
        stats[r] = [f1.mean(1).mean(), f1[:, excl].mean(1).mean(), f1[:, shared].mean(1).mean()]
    a = (1 - level) / 2
    lo, hi = np.quantile(stats, a, axis=0), np.quantile(stats, 1 - a, axis=0)
    return {m: [float(lo[i]), float(hi[i])] for i, m in enumerate(METRICS)}


def aggregate(episodes: list[dict], y_true, captures, protocol: Protocol, reps: int = 1000, seed: int = 0) -> dict:
    """Mean ± std over draws of one (direction, method, source, K), the bootstrap interval of
    the mean, and the mean per-class F1. Each episode needs ``scores`` and ``predictions``."""
    out = {"n_draws": len(episodes)}
    for m in METRICS:
        v = np.array([e["scores"][m] for e in episodes])
        out[m] = {"mean": float(v.mean()), "std": float(v.std(ddof=0)), "draws": v.tolist()}
    ci = bootstrap_draw_mean(y_true, np.array([e["predictions"] for e in episodes], dtype=object),
                             captures, protocol, reps, seed)
    for m in METRICS:
        out[m]["ci95"] = ci[m]
    labels = protocol.target_labels
    out["per_class_mean"] = {c: float(np.mean([e["scores"]["per_class"][c] for e in episodes])) for c in labels}
    return out
