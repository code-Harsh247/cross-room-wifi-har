"""Capture-level splits.

* ``split``: 'val' (scoring only, never trained on) or 'dev'.
  Room A uses the supplied validation/ folder. Room B uses 25 captures per class
  drawn with seed 42, saved once to a manifest and reused forever after.
* ``dev_fold``: 'train' or 'holdout' inside dev, for model selection. Stratified by
  room and activity, again by capture.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .catalog import natural_key


def draw_room_b_validation(catalog_b: pd.DataFrame, per_class: int, seed: int) -> list[str]:
    """Pick ``per_class`` capture_ids per activity, deterministically."""
    rng = np.random.default_rng(seed)
    chosen = []
    for activity in sorted(catalog_b["activity"].unique()):
        ids = sorted(catalog_b.loc[catalog_b["activity"] == activity, "capture_id"], key=natural_key)
        picked = rng.choice(len(ids), size=per_class, replace=False)
        chosen += [ids[i] for i in sorted(picked)]
    return chosen


def room_b_validation(catalog_b: pd.DataFrame, manifest: Path, per_class: int, seed: int) -> list[str]:
    """Load the committed manifest, or create it on first use. Never silently redraws."""
    if manifest.exists():
        data = json.loads(manifest.read_text(encoding="utf-8"))
        if (data["per_class"], data["seed"]) != (per_class, seed):
            raise ValueError(f"{manifest} was made with different settings; delete it deliberately to redraw")
        missing = set(data["capture_ids"]) - set(catalog_b["capture_id"])
        if missing:
            raise ValueError(f"{manifest} lists captures that do not exist: {sorted(missing)[:5]}")
        return data["capture_ids"]
    ids = draw_room_b_validation(catalog_b, per_class, seed)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        json.dumps(
            {
                "description": "Room B validation captures (scoring only). Drawn per activity "
                "from naturally sorted capture_ids with numpy default_rng(seed).choice.",
                "per_class": per_class,
                "seed": seed,
                "capture_ids": ids,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return ids


def assign_splits(catalog: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Add ``split`` and ``dev_fold`` columns to the catalog."""
    sp = cfg["splits"]
    out = catalog.copy()
    b_val = set(
        room_b_validation(
            out[out["room"] == "B"],
            Path(cfg["paths"]["splits_dir"]) / "room_b_validation.json",
            sp["room_b_val_per_class"],
            sp["room_b_val_seed"],
        )
    )
    is_val = (out["provided_split"] == "val") | out["capture_id"].isin(b_val)
    out["split"] = np.where(is_val, "val", "dev")

    rng = np.random.default_rng(sp["dev_holdout_seed"])
    out["dev_fold"] = None
    for _, group in out[out["split"] == "dev"].groupby(["room", "activity"], sort=True):
        ids = sorted(group["capture_id"], key=natural_key)
        n_hold = int(round(len(ids) * sp["dev_holdout_frac"]))
        hold = {ids[i] for i in rng.choice(len(ids), size=n_hold, replace=False)}
        out.loc[group.index, "dev_fold"] = ["holdout" if c in hold else "train" for c in group["capture_id"]]
    return out
