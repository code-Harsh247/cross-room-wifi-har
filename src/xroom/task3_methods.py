"""Task 3 methods (PRD §6): predictions for the target validation windows of one episode.

Every method gets activity labels ONLY through ``visible_labels`` (the access guard): the
source room's development labels and the K support captures. The reference rows are the
only exceptions and are clearly separate:
  UB      target-supervised upper bound: prototypes from ALL target dev labels (not a valid result)
  CHANCE  uniform random guessing over the target label space
  SHUF    control: M1 with the support captures' labels permuted across captures
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .adapt import coral_apply, coral_fit, per_room_standardize, refine_prototypes
from .fewshot import Protocol, logreg_classifier, nearest_centroid, prototypes, visible_labels

METHODS = ("M0", "M1", "M2", "M3", "M4", "M5", "M6")
REFERENCES = ("UB", "CHANCE", "SHUF")


def _features(method: str, feats: dict[str, np.ndarray], meta: pd.DataFrame) -> np.ndarray:
    rooms = meta["room"].to_numpy()
    dev = (meta["split"] == "dev").to_numpy()  # unlabelled development rows of each room
    if method == "M0":
        return np.asarray(feats["base"], dtype=np.float32)
    if method == "M3":
        if "adabn" not in feats:
            raise ValueError("M3 needs room-matched AdaBN features (feats['adabn'])")
        return per_room_standardize(feats["adabn"], rooms, dev)
    if method == "M6":
        if "inputnorm" not in feats:
            raise ValueError("M6 needs room-matched input-normalized features (feats['inputnorm'])")
        return per_room_standardize(feats["inputnorm"], rooms, dev)
    return per_room_standardize(feats["base"], rooms, dev)


def predict(method: str, feats: dict[str, np.ndarray], meta: pd.DataFrame, protocol: Protocol,
            support_ids, cfg: dict, seed: int = 0) -> np.ndarray:
    """Activity predictions for ``protocol.masks(meta)["target_val"]`` rows, in row order."""
    t3 = cfg["task3"]
    m = protocol.masks(meta)
    tv = m["target_val"]
    if method == "CHANCE":
        rng = np.random.default_rng([seed, 99])
        return np.asarray(protocol.target_labels, dtype=object)[rng.integers(0, len(protocol.target_labels), tv.sum())]
    if method == "UB":
        return _upper_bound(feats, meta, protocol, t3["distance"])
    shuffle = method == "SHUF"
    if shuffle:
        method = "M1"
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}")

    vis = visible_labels(protocol, meta, support_ids)  # the only label source
    known = vis.notna().to_numpy()
    lab = np.where(known, vis.to_numpy(dtype=object), None)  # hidden labels -> None (not pd.NA)
    src_rows = m["source_dev"] & known & np.isin(lab, protocol.shared)  # shared classes from source
    sup_rows = m["target_dev"] & known  # the K support captures (target-only classes)
    if shuffle:
        lab = _shuffle_support(lab, meta, sup_rows, seed)
    proto_lab = np.where(src_rows | sup_rows, lab, None)

    f = _features(method, feats, meta)
    if method == "M2":
        fit = coral_fit(f[m["source_dev"]], f[m["target_dev"]], eps=t3["coral_eps"])
        f = f.copy()
        f[m["source_dev"]] = coral_apply(f[m["source_dev"]], fit)
    if method == "M5":
        train = src_rows | sup_rows
        clf = logreg_classifier(f[train], proto_lab[train], c=t3["logreg_c"], seed=seed)
        return np.asarray(clf.predict(f[tv]), dtype=object)

    classes, protos = prototypes(f, proto_lab, classes=protocol.target_labels)
    if method == "M4":
        index = {c: i for i, c in enumerate(classes)}
        unlabeled = m["target_dev"] & ~sup_rows
        protos = refine_prototypes(protos, f[unlabeled], f[sup_rows],
                                   np.array([index[c] for c in lab[sup_rows]]),
                                   iters=t3["refine"]["iters"], temperature=t3["refine"]["temperature"])
    return nearest_centroid(f[tv], classes, protos, t3["distance"])


def _shuffle_support(lab: np.ndarray, meta: pd.DataFrame, sup_rows: np.ndarray, seed: int) -> np.ndarray:
    """SHUF control: permute the support labels across support captures (all windows of a
    capture keep one label), seeded. Breaks the link between support signal and label."""
    caps = meta.loc[sup_rows, "capture_id"]
    cap_lab = pd.Series(lab[sup_rows], index=caps.index).groupby(caps).first()
    perm = np.random.default_rng([seed, 7]).permutation(cap_lab.to_numpy(dtype=object))
    new = pd.Series(perm, index=cap_lab.index)
    out = lab.copy()
    out[sup_rows] = caps.map(new).to_numpy(dtype=object)
    return out


def _upper_bound(feats: dict[str, np.ndarray], meta: pd.DataFrame, protocol: Protocol, metric: str) -> np.ndarray:
    """Reference only (NOT a valid Task 3 result): M1 features with prototypes from every
    target development label, which the Task 3 rules forbid."""
    m = protocol.masks(meta)
    f = _features("M1", feats, meta)
    lab = np.where(m["target_dev"], meta["activity"].to_numpy(dtype=object), None)
    classes, protos = prototypes(f, lab, classes=protocol.target_labels)
    return nearest_centroid(f[m["target_val"]], classes, protos, metric)


def predict_source(feats: dict[str, np.ndarray], meta: pd.DataFrame, protocol: Protocol, cfg: dict) -> np.ndarray:
    """Source-room model for windows routed to the source room (FR-7): M1 features, nearest
    centroid with prototypes of every source class from source development labels (seen only
    through the access guard). Predicts ``protocol.masks(meta)["source_val"]`` rows."""
    m = protocol.masks(meta)
    vis = visible_labels(protocol, meta, ())  # source dev labels only
    lab = np.where(vis.notna().to_numpy() & m["source_dev"], vis.to_numpy(dtype=object), None)
    f = _features("M1", feats, meta)
    classes, protos = prototypes(f, lab, classes=protocol.source_labels)
    return nearest_centroid(f[m["source_val"]], classes, protos, cfg["task3"]["distance"])
