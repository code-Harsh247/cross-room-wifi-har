"""Metrics for the room classifier."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, log_loss

SHARED_ACTIVITIES = ("sit", "stand", "walk")


def room_metrics(true_room: np.ndarray, prob_b: np.ndarray) -> dict:
    """Window-level metrics; ``prob_b`` = predicted probability of Room B."""
    true_room = np.asarray(true_room)
    pred = np.where(np.asarray(prob_b) >= 0.5, "B", "A")
    out = {"n_windows": int(len(true_room)), "accuracy": float(accuracy_score(true_room, pred))}
    rooms = sorted(set(true_room))
    for r in rooms:
        out[f"recall_{r}"] = float((pred[true_room == r] == r).mean())
    if len(rooms) == 2:
        out["balanced_accuracy"] = float(balanced_accuracy_score(true_room, pred))
        out["log_loss"] = float(log_loss(true_room == "B", np.clip(prob_b, 1e-7, 1 - 1e-7), labels=[False, True]))
        out["confusion_matrix(rows=true A,B; cols=pred A,B)"] = confusion_matrix(
            true_room, pred, labels=["A", "B"]
        ).tolist()
    return out


def capture_level(meta: pd.DataFrame, prob_b: np.ndarray) -> dict:
    """Average window probabilities within each capture, then score per capture."""
    df = meta[["capture_id", "room"]].assign(prob_b=prob_b)
    cap = df.groupby("capture_id").agg(room=("room", "first"), prob_b=("prob_b", "mean"))
    return room_metrics(cap["room"].to_numpy(), cap["prob_b"].to_numpy())


def bootstrap_balanced_accuracy(meta: pd.DataFrame, correct: np.ndarray, reps: int = 1000,
                                seed: int = 0, level: float = 0.95) -> tuple[float, float]:
    """Confidence interval for window-level balanced accuracy, resampling *captures*
    with replacement within each room (windows of one capture are not independent)."""
    df = meta[["capture_id", "room"]].assign(correct=np.asarray(correct, dtype=float))
    cap = df.groupby("capture_id").agg(room=("room", "first"), n=("correct", "size"), k=("correct", "sum"))
    rng = np.random.default_rng(seed)
    per_room = []
    for _, g in cap.groupby("room"):
        idx = rng.integers(0, len(g), size=(reps, len(g)))
        per_room.append(g["k"].to_numpy()[idx].sum(1) / g["n"].to_numpy()[idx].sum(1))
    stats = np.mean(per_room, axis=0)
    alpha = (1 - level) / 2
    return float(np.quantile(stats, alpha)), float(np.quantile(stats, 1 - alpha))


def full_report(meta: pd.DataFrame, prob_b: np.ndarray) -> dict:
    """All windows, shared activities only, capture level, and per activity."""
    prob_b = np.asarray(prob_b)
    shared = meta["activity"].isin(SHARED_ACTIVITIES).to_numpy()
    report = {
        "all_activities": room_metrics(meta["room"].to_numpy(), prob_b),
        "shared_activities_sit_stand_walk": room_metrics(meta["room"].to_numpy()[shared], prob_b[shared]),
        "capture_level": capture_level(meta, prob_b),
        "per_room_activity_accuracy": {},
    }
    pred = np.where(prob_b >= 0.5, "B", "A")
    for (room, act), idx in meta.groupby(["room", "activity"]).indices.items():
        report["per_room_activity_accuracy"][f"{room}/{act}"] = float((pred[idx] == room).mean())
    return report
