"""Tests for features, the room classifier and metrics."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from xroom.eval import capture_level, full_report, room_metrics
from xroom.features import data_subcarriers, format_stats, in_chunks, spectrum_features
from xroom.room_classifier import RoomClassifier

SC = data_subcarriers([0, 1, 255], 256)


def _windows(n=6, T=20, seed=0):
    rng = np.random.default_rng(seed)
    return rng.uniform(1, 100, size=(n, T, 256)).astype(np.float32)


def test_row_spectrum_is_invariant_to_per_row_gain():
    x = _windows()
    gains = np.random.default_rng(1).uniform(0.2, 5, size=(6, 20, 1)).astype(np.float32)
    np.testing.assert_allclose(spectrum_features(x, SC, "row"), spectrum_features(x * gains, SC, "row"),
                               rtol=1e-4, atol=1e-5)


def test_log_spectrum_is_nearly_gain_invariant_for_large_amplitudes():
    x = 100 * _windows()  # realistic CSI amplitudes are in the hundreds
    gains = np.random.default_rng(1).uniform(0.5, 2, size=(6, 20, 1)).astype(np.float32)
    np.testing.assert_allclose(spectrum_features(x, SC, "log"), spectrum_features(x * gains, SC, "log"),
                               atol=0.01)


def test_spectrum_none_keeps_scale_and_shapes():
    x = _windows()
    f = spectrum_features(x, SC, "none")
    assert f.shape == (6, len(SC))
    np.testing.assert_allclose(spectrum_features(2 * x, SC, "none"), 2 * f, rtol=1e-5)
    with pytest.raises(ValueError):
        spectrum_features(x, SC, "bogus")


def test_in_chunks_matches_single_call():
    x = _windows(n=7)
    np.testing.assert_allclose(in_chunks(lambda c: spectrum_features(c, SC, "row"), x, chunk=3),
                               spectrum_features(x, SC, "row"))
    stats = in_chunks(lambda c: format_stats(c, SC), x, chunk=3)
    assert len(stats) == 7 and stats.notna().all().all()


def test_classifier_separates_rooms_and_roundtrips(tmp_path):
    rng = np.random.default_rng(0)
    shape_a = 1 + np.sin(np.linspace(0, 6, 256))
    shape_b = 1 + np.cos(np.linspace(0, 6, 256))
    xa = shape_a * rng.uniform(0.5, 1.5, size=(30, 20, 256))
    xb = shape_b * rng.uniform(0.5, 1.5, size=(30, 20, 256))
    x = np.concatenate([xa, xb]).astype(np.float32)
    y = np.array(["A"] * 30 + ["B"] * 30)
    clf = RoomClassifier(SC, variant="row", C=1.0).fit(x, y)
    assert (clf.predict(x) == y).mean() == 1.0
    path = tmp_path / "m.joblib"
    clf.save(path)
    np.testing.assert_allclose(RoomClassifier.load(path).predict_proba(x), clf.predict_proba(x))


def test_room_metrics_and_capture_level():
    meta = pd.DataFrame({
        "capture_id": ["a1", "b1", "b1", "b1"],
        "room": ["A", "B", "B", "B"],
        "activity": ["sit", "sit", "lying", "lying"],
    })
    prob_b = np.array([0.1, 0.9, 0.4, 0.8])  # one B window wrong
    m = room_metrics(meta["room"].to_numpy(), prob_b)
    assert m["recall_A"] == 1.0 and m["recall_B"] == pytest.approx(2 / 3)
    assert m["balanced_accuracy"] == pytest.approx((1 + 2 / 3) / 2)
    assert capture_level(meta, prob_b)["accuracy"] == 1.0  # mean prob of b1 = 0.7
    rep = full_report(meta, prob_b)
    assert rep["shared_activities_sit_stand_walk"]["n_windows"] == 2
    assert rep["per_room_activity_accuracy"]["B/lying"] == 0.5
