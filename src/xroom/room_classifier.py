"""Room classifier (Task 1), also the room router for Task 3.

Raw windows go in; the classifier computes its own features, so callers never
have to remember which feature variant a saved model expects.
"""
from __future__ import annotations

from functools import partial
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
    """Time-averaged spectrum fingerprint + logistic regression."""

    def __init__(self, data_sc: np.ndarray, variant: str = "row", C: float = 1.0, seed: int = 0):
        self.data_sc = np.asarray(data_sc)
        self.variant = variant
        self.C = C
        self.seed = seed
        self.model = make_linear_model(C, seed)

    def features(self, windows: np.ndarray) -> np.ndarray:
        return in_chunks(partial(spectrum_features, data_sc=self.data_sc, variant=self.variant), windows)

    def fit(self, windows: np.ndarray, rooms: np.ndarray) -> "RoomClassifier":
        return self.fit_features(self.features(windows), rooms)

    def fit_features(self, feats: np.ndarray, rooms: np.ndarray) -> "RoomClassifier":
        """Fit on precomputed ``features(...)`` output (saves recomputing in sweeps)."""
        self.model.fit(feats, np.asarray(rooms))
        return self

    def predict_proba(self, windows: np.ndarray) -> np.ndarray:
        """Probability of Room B for each window."""
        return self.proba_from_features(self.features(windows))

    def proba_from_features(self, feats: np.ndarray) -> np.ndarray:
        classes = list(self.model.classes_)
        return self.model.predict_proba(feats)[:, classes.index("B")]

    def predict(self, windows: np.ndarray) -> np.ndarray:
        """'A' or 'B' for each window."""
        return ROOMS[(self.predict_proba(windows) >= 0.5).astype(int)]

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: str | Path) -> "RoomClassifier":
        return joblib.load(path)
