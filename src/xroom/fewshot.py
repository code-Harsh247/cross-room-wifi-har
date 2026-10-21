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

import json
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .catalog import natural_key

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


# ---------- support sampling (FR-2, D3) ----------

def support_pool(protocol: Protocol, meta: pd.DataFrame) -> dict[str, list[str]]:
    """Target-room development capture IDs per target-only activity, naturally sorted."""
    tgt_dev = protocol.masks(meta)["target_dev"]
    caps = meta[tgt_dev].drop_duplicates("capture_id")
    return {act: sorted(caps.loc[caps["activity"] == act, "capture_id"], key=natural_key)
            for act in protocol.target_only}


def support_order(protocol: Protocol, meta: pd.DataFrame, seed: int) -> dict[str, list[str]]:
    """A seeded random order of each target-only class's pool. The support for K is the first K
    captures of each order, so supports are nested across K for a fixed seed. Each class has its
    own generator (seed, stable hash of direction + class), so adding a class cannot reshuffle
    the others."""
    out = {}
    for act, pool in support_pool(protocol, meta).items():
        key = zlib.crc32(f"{protocol.direction}/{act}".encode())
        rng = np.random.default_rng([seed, key])
        out[act] = [pool[i] for i in rng.permutation(len(pool))]
    return out


def sample_support(protocol: Protocol, meta: pd.DataFrame, k: int, seed: int) -> list[str]:
    """K support captures (not windows) per target-only class, from target dev only."""
    if k < 1:
        raise ValueError("k must be >= 1")
    order = support_order(protocol, meta, seed)
    short = {act: len(o) for act, o in order.items() if len(o) < k}
    if short:
        raise ValueError(f"not enough target dev captures for K={k}: {short}")
    ids = [cid for act in protocol.target_only for cid in order[act][:k]]
    check_support(protocol, meta, ids)
    return ids


def write_support_manifest(protocol: Protocol, meta: pd.DataFrame, k_values, seed: int, path: str | Path) -> dict:
    """Save the nested supports of one (direction, seed) as JSON; returns the manifest."""
    order = support_order(protocol, meta, seed)
    kmax = max(k_values)
    manifest = {
        "direction": protocol.direction,
        "seed": seed,
        "k_values": list(k_values),
        "description": "K support captures per target-only class (target dev only); nested: "
                       "K=k uses the first k captures of each class list.",
        "support": {act: order[act][:kmax] for act in protocol.target_only},
    }
    for k in k_values:  # validates every K before writing
        sample_support(protocol, meta, k, seed)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def load_support(path: str | Path, k: int) -> list[str]:
    """Support capture IDs for K from a saved manifest."""
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    if k not in manifest["k_values"]:
        raise ValueError(f"K={k} not in manifest {path} ({manifest['k_values']})")
    return [cid for ids in manifest["support"].values() for cid in ids[:k]]


def manifest_path(cfg: dict, direction: str, seed: int) -> Path:
    return Path(cfg["paths"]["task3_dir"]) / "support" / f"{direction}_seed{seed}.json"


# ---------- classifiers (FR-4) ----------

def prototypes(f: np.ndarray, labels, classes=None) -> tuple[tuple[str, ...], np.ndarray]:
    """Class means. ``labels`` aligned with ``f``; NA / None rows are ignored. Returns
    (classes, (k, d) prototypes) in sorted class order (or the given ``classes``)."""
    lab = pd.Series(labels).to_numpy(dtype=object)
    known = pd.notna(lab)
    classes = tuple(sorted(set(lab[known]))) if classes is None else tuple(classes)
    missing = [c for c in classes if not (lab[known] == c).any()]
    if missing:
        raise ValueError(f"no rows for classes {missing}")
    f = np.asarray(f, dtype=np.float64)
    return classes, np.stack([f[known & (lab == c)].mean(0) for c in classes]).astype(np.float32)


def nearest_centroid(f: np.ndarray, classes, protos: np.ndarray, metric: str = "cosine") -> np.ndarray:
    """Assign each row to the closest prototype (cosine similarity or euclidean distance)."""
    f, protos = np.asarray(f, np.float64), np.asarray(protos, np.float64)
    if metric == "cosine":
        fn = f / np.maximum(np.linalg.norm(f, axis=1, keepdims=True), 1e-12)
        pn = protos / np.maximum(np.linalg.norm(protos, axis=1, keepdims=True), 1e-12)
        idx = np.argmax(fn @ pn.T, axis=1)
    elif metric == "euclidean":
        idx = np.argmin(((f[:, None, :] - protos[None]) ** 2).sum(-1), axis=1)
    else:
        raise ValueError(f"unknown metric {metric!r}")
    return np.asarray(classes, dtype=object)[idx]


def logreg_classifier(f_train: np.ndarray, y_train, c: float = 1.0, seed: int = 0):
    """M5: StandardScaler + class-balanced logistic regression (same model as Task 1)."""
    from .room_classifier import make_linear_model  # local import keeps fewshot light
    y = pd.Series(y_train).to_numpy(dtype=object)
    keep = pd.notna(y)
    return make_linear_model(c, seed).fit(np.asarray(f_train)[keep], y[keep].astype(str))
