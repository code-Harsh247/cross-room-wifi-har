"""Tests for Task 3 room routing."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from xroom.config import load_config
from xroom.router import load_router, route

CFG = load_config()


def test_oracle_routing_returns_true_rooms():
    meta = pd.DataFrame({"room": ["A", "B", "B"]})
    assert route(np.zeros((3, 150, 256)), meta, mode="oracle").tolist() == ["A", "B", "B"]


def test_predicted_routing_requires_router_and_valid_mode():
    meta = pd.DataFrame({"room": ["A"]})
    with pytest.raises(ValueError):
        route(np.zeros((1, 150, 256)), meta, mode="predicted")
    with pytest.raises(ValueError):
        route(np.zeros((1, 150, 256)), meta, mode="guess")
    with pytest.raises(ValueError):
        load_router(CFG, "bogus")


@pytest.mark.parametrize("kind", ["strict", "activity_tuned"])
def test_saved_routers_route_validation_windows(kind):
    try:
        router = load_router(CFG, kind)
    except FileNotFoundError:
        pytest.skip("routers not built (run scripts/task1_router_variants.py)")
    from xroom.dataset import load_windows
    x, meta = load_windows(CFG)
    val = meta[meta["split"] == "val"].groupby("room").head(5)  # 5 windows per room
    pred = route(x[val.index.to_numpy()], val, mode="predicted", router=router)
    assert (pred == val["room"].to_numpy()).all()
