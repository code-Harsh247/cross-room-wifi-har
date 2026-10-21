"""Task 3 episode protocol and the label access guard (PRD §4, FR-1).

A direction S -> T moves an activity recogniser from source room S (all development labels)
to target room T, where:
  * the shared activities (sit, stand, walk) have NO labels,
  * each target-only activity has labels for K support captures (drawn from target dev),
  * every other target development window may be used, but only unlabelled,
  * target validation windows are scored and nothing else.

Every method must obtain activity labels through ``visible_labels``, which hides everything
the Task 3 rules do not allow. Masks for the unlabelled data come from ``Protocol.masks``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

DIRECTIONS = {"AtoB": ("A", "B"), "BtoA": ("B", "A")}


@dataclass(frozen=True)
class Protocol:
    """Rooms and label sets of one transfer direction."""

    direction: str
    source: str
    target: str
    shared: tuple[str, ...]
    source_only: tuple[str, ...]
    target_only: tuple[str, ...]

    @classmethod
    def from_config(cls, cfg: dict, direction: str) -> "Protocol":
        if direction not in DIRECTIONS:
            raise ValueError(f"unknown direction {direction!r}; choose from {sorted(DIRECTIONS)}")
        source, target = DIRECTIONS[direction]
        t3 = cfg["task3"]
        shared = tuple(sorted(t3["shared"]))
        acts = {room: set(a) for room, a in t3["activities"].items()}
        for room in (source, target):
            if not set(shared) <= acts[room]:
                raise ValueError(f"room {room} is missing shared activities {set(shared) - acts[room]}")
        return cls(
            direction=direction,
            source=source,
            target=target,
            shared=shared,
            source_only=tuple(sorted(acts[source] - set(shared))),
            target_only=tuple(sorted(acts[target] - set(shared))),
        )

    @property
    def target_labels(self) -> tuple[str, ...]:
        """The target room's label space: shared + target-only (source-only classes excluded, D5)."""
        return tuple(sorted(self.shared + self.target_only))

    @property
    def source_labels(self) -> tuple[str, ...]:
        return tuple(sorted(self.shared + self.source_only))

    def masks(self, meta: pd.DataFrame) -> dict[str, np.ndarray]:
        """Boolean row masks: source/target x dev/val."""
        room, split = meta["room"].to_numpy(), meta["split"].to_numpy()
        return {
            "source_dev": (room == self.source) & (split == "dev"),
            "source_val": (room == self.source) & (split == "val"),
            "target_dev": (room == self.target) & (split == "dev"),
            "target_val": (room == self.target) & (split == "val"),
        }


def check_support(protocol: Protocol, meta: pd.DataFrame, support_ids) -> set[str]:
    """Validate support capture IDs: they must be target-room development captures of a
    target-only activity. Raises ``ValueError`` otherwise; returns them as a set."""
    ids = set(support_ids)
    caps = meta.drop_duplicates("capture_id").set_index("capture_id")
    unknown = ids - set(caps.index)
    if unknown:
        raise ValueError(f"unknown support captures: {sorted(unknown)[:5]}")
    sel = caps.loc[sorted(ids)]
    bad_room = sel.index[sel["room"] != protocol.target]
    bad_split = sel.index[sel["split"] != "dev"]
    bad_act = sel.index[~sel["activity"].isin(protocol.target_only)]
    if len(bad_room):
        raise ValueError(f"support captures not in target room {protocol.target}: {list(bad_room)[:5]}")
    if len(bad_split):
        raise ValueError(f"support captures from validation: {list(bad_split)[:5]}")
    if len(bad_act):
        raise ValueError(f"support captures of shared/non-target-only activities: {list(bad_act)[:5]}")
    return ids


def visible_labels(protocol: Protocol, meta: pd.DataFrame, support_ids=()) -> pd.Series:
    """The access guard: the activity labels a Task 3 method may see, aligned with ``meta``.

    Visible: every source-room development window, and the windows of the K support captures.
    Everything else (target shared classes, other target dev windows, all validation) is NA.
    """
    ids = check_support(protocol, meta, support_ids)
    m = protocol.masks(meta)
    allowed = m["source_dev"] | meta["capture_id"].isin(ids).to_numpy()
    out = pd.Series(pd.NA, index=meta.index, dtype="object", name="visible_activity")
    out[allowed] = meta.loc[allowed, "activity"]
    return out
