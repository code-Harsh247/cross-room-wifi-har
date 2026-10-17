"""Room classifier (Task 1), also the room router for Task 3.

Raw windows go in; the classifier computes its own features, so callers never
have to remember which feature variant a saved model expects.

Optional slice averaging (``slice_rows``): every window is cut into contiguous
``slice_rows``-row slices, the model is trained on slices, and a window's probability
is the mean of its slices' probabilities. Shorter slices transfer better to
activities the router never saw (reports/task1_router_variants.md).
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import in_chunks, spectrum_features

ROOMS = np.array(["A", "B"])


def make_linear_model(C: float = 1.0, seed: int = 0) -> Pipeline:
    """Standardize (fitted on training data only) + class-balanced logistic regression."""
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=C, class_weight="balanced", max_iter=5000, random_state=seed),
    )


class RoomClassifier:
    """Time-averaged spectrum fingerprint + logistic regression, with optional slice averaging."""

    def __init__(self, data_sc: np.ndarray, variant: str = "row", C: float = 1.0, seed: int = 0,
                 slice_rows: int | None = None):
        self.data_sc = np.asarray(data_sc)
        self.variant = variant
        self.C = C
        self.seed = seed
        self.slice_rows = slice_rows
        self.model = make_linear_model(C, seed)

    def _slice_features(self, chunk: np.ndarray) -> np.ndarray:
        """(n, T, 256) windows -> (n, k, F) features of k contiguous slices (k = 1 without slicing)."""
        s = getattr(self, "slice_rows", None)  # models saved before slicing existed have no attribute
        n, t = chunk.shape[:2]
        if not s or s >= t:
            return spectrum_features(chunk, self.data_sc, self.variant)[:, None, :]
        k = t // s
        sliced = np.asarray(chunk[:, : k * s]).reshape(n * k, s, chunk.shape[2])
        return spectrum_features(sliced, self.data_sc, self.variant).reshape(n, k, -1)

    def features(self, windows: np.ndarray) -> np.ndarray:
        """(n, k, F) per-slice features; k = 1 when slice averaging is off."""
        return in_chunks(self._slice_features, windows)

    def fit(self, windows: np.ndarray, rooms: np.ndarray) -> "RoomClassifier":
        return self.fit_features(self.features(windows), rooms)

    def fit_features(self, feats: np.ndarray, rooms: np.ndarray) -> "RoomClassifier":
        """Fit on precomputed features: (n, F), or (n, k, F) from ``features`` (each slice
        inherits its window's room)."""
        feats = feats if feats.ndim == 3 else feats[:, None, :]
        n, k, f = feats.shape
        self.model.fit(feats.reshape(n * k, f), np.repeat(np.asarray(rooms), k))
        return self

    def proba_from_features(self, feats: np.ndarray) -> np.ndarray:
        """Probability of Room B per window (mean over its slices)."""
        feats = feats if feats.ndim == 3 else feats[:, None, :]
        n, k, f = feats.shape
        classes = list(self.model.classes_)
        p = self.model.predict_proba(feats.reshape(n * k, f))[:, classes.index("B")]
        return p.reshape(n, k).mean(axis=1)

    def predict_proba(self, windows: np.ndarray) -> np.ndarray:
        """Probability of Room B for each window."""
        return self.proba_from_features(self.features(windows))

    def predict(self, windows: np.ndarray) -> np.ndarray:
        """'A' or 'B' for each window."""
        return ROOMS[(self.predict_proba(windows) >= 0.5).astype(int)]

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: str | Path) -> "RoomClassifier":
        return joblib.load(path)
