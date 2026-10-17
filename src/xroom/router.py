"""Room routing for Task 3: decide which room each window belongs to.

Two kinds of routing, so Task 3 can report "oracle vs predicted room":
  oracle     the true room from the metadata (upper bound)
  predicted  the Task 1 room router

Two saved routers (both trained on all development windows of both rooms with room labels
only, never activity labels; made by scripts/task1_router_variants.py):
  strict          settings chosen with room labels only -> rule-safe default for Task 3
  activity_tuned  settings chosen for transfer to unseen activities, using dev activity
                  labels of both rooms -> ablation only (see reports/task1_router_variants.md)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .room_classifier import RoomClassifier

ROUTER_FILES = {"strict": "room_router.joblib", "activity_tuned": "room_router_activity_tuned.joblib"}


def load_router(cfg: dict, kind: str = "strict") -> RoomClassifier:
    if kind not in ROUTER_FILES:
        raise ValueError(f"unknown router {kind!r}; choose from {sorted(ROUTER_FILES)}")
    path = Path(cfg["paths"]["router_dir"]) / ROUTER_FILES[kind]
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run scripts/task1_router_variants.py first")
    return RoomClassifier.load(path)


def route(windows: np.ndarray, meta: pd.DataFrame, mode: str = "predicted",
          router: RoomClassifier | None = None) -> np.ndarray:
    """Room ('A'/'B') per window. ``mode='oracle'`` reads ``meta['room']``;
    ``mode='predicted'`` asks ``router`` (required)."""
    if mode == "oracle":
        return meta["room"].to_numpy()
    if mode == "predicted":
        if router is None:
            raise ValueError("predicted routing needs a router (see load_router)")
        return router.predict(windows)
    raise ValueError(f"unknown mode {mode!r}; use 'oracle' or 'predicted'")
