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

from .adapt import coral_apply, coral_fit, per_room_standardize, refine_prototypes, sinkhorn
from .fewshot import Protocol, logreg_classifier, nearest_centroid, prototypes, visible_labels

METHODS = ("M0", "M1", "M2", "M3", "M4", "M5", "M6")
REFERENCES = ("UB", "CHANCE", "SHUF")
POSTHOC = ("M7", "M8", "M9", "M10", "M11")  # ideas 2 + batch 2, configs/task3_posthoc.yaml; post-hoc ablations only


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
    if method in POSTHOC:
        return _predict_balanced(method, feats, meta, protocol, support_ids, cfg, seed)
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


def _cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)
    b = b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-12)
    return a @ b.T


def _predict_balanced(method: str, feats: dict[str, np.ndarray], meta: pd.DataFrame, protocol: Protocol,
                      support_ids, cfg: dict, seed: int) -> np.ndarray:
    """Post-hoc idea 2: balance the class assignment on unlabelled target dev windows with
    Sinkhorn (uniform class marginals), then predict target val."""
    if "posthoc" not in cfg:
        raise ValueError("post-hoc methods need configs/task3_posthoc.yaml (cfg['posthoc'])")
    bal = cfg["posthoc"]["balance"]
    if bal["marginal"] != "uniform":
        raise ValueError("only uniform class marginals are pre-registered")
    t3 = cfg["task3"]
    m = protocol.masks(meta)
    tv, td = m["target_val"], m["target_dev"]
    vis = visible_labels(protocol, meta, support_ids)  # the only label source
    known = vis.notna().to_numpy()
    lab = np.where(known, vis.to_numpy(dtype=object), None)
    src_rows = m["source_dev"] & known & np.isin(lab, protocol.shared)
    sup_rows = td & known
    proto_lab = np.where(src_rows | sup_rows, lab, None)
    f = _features("M1", feats, meta)

    if method == "M9":
        train = src_rows | sup_rows
        clf = logreg_classifier(f[train], proto_lab[train], c=t3["logreg_c"], seed=seed)
        classes = np.asarray(clf.classes_, dtype=object)
        _, _, g = sinkhorn(clf.predict_log_proba(f[td]), iters=bal["iters"])
        return classes[np.argmax(clf.predict_log_proba(f[tv]) + g[None, :], axis=1)]
    if method == "M10":
        return _balanced_self_training(f, proto_lab, src_rows | sup_rows, td & ~sup_rows, td, tv, cfg, seed)
    if method == "M11":
        return _probability_ensemble(feats, meta, proto_lab, src_rows | sup_rows, td, tv, cfg, seed)

    classes, protos = prototypes(f, proto_lab, classes=protocol.target_labels)
    classes = np.asarray(classes, dtype=object)
    tau = bal["temperature"]
    if method == "M7":
        _, _, g = sinkhorn(_cosine(f[td], protos) / tau, iters=bal["iters"])
        return classes[np.argmax(_cosine(f[tv], protos) / tau + g[None, :], axis=1)]

    # M8: transport-weighted means of unlabelled target dev + pinned support windows
    unl = td & ~sup_rows
    plan, _, _ = sinkhorn(_cosine(f[unl], protos) / tau, iters=bal["iters"])
    soft = plan * unl.sum()  # each unlabelled row's assignment sums to 1
    index = {c: i for i, c in enumerate(classes)}
    s_idx = np.array([index[c] for c in lab[sup_rows]])
    num = soft.T @ f[unl].astype(np.float64)
    den = soft.sum(0)
    np.add.at(num, s_idx, f[sup_rows].astype(np.float64))
    den = den + np.bincount(s_idx, minlength=len(classes))
    new = (num / np.maximum(den, 1e-12)[:, None]).astype(np.float32)
    return nearest_centroid(f[tv], classes, new, t3["distance"])


def _probability_ensemble(feats, meta, proto_lab, train, td, tv, cfg, seed) -> np.ndarray:
    """M11: one class-balanced logistic regression per ensemble member (M1 features of that
    member), mean of the members' probabilities, then the M9 balanced class offsets estimated on
    the unlabelled target dev windows. Every member has equal weight."""
    if "members" not in feats:
        raise ValueError("M11 needs an ensemble source (feats['members']; posthoc.ensembles)")
    t3, bal = cfg["task3"], cfg["posthoc"]["balance"]
    rooms, dev = meta["room"].to_numpy(), (meta["split"] == "dev").to_numpy()
    p_td, p_tv, classes = 0.0, 0.0, None
    for member in feats["members"]:
        f = per_room_standardize(member, rooms, dev)
        clf = logreg_classifier(f[train], proto_lab[train], c=t3["logreg_c"], seed=seed)
        if classes is None:
            classes = np.asarray(clf.classes_, dtype=object)
        elif list(clf.classes_) != list(classes):
            raise AssertionError("ensemble members disagree on the class order")
        p_td = p_td + clf.predict_proba(f[td])
        p_tv = p_tv + clf.predict_proba(f[tv])
    n = len(feats["members"])
    _, _, g = sinkhorn(np.log(p_td / n + 1e-12), iters=bal["iters"])
    return classes[np.argmax(np.log(p_tv / n + 1e-12) + g[None, :], axis=1)]


def _balanced_self_training(f, proto_lab, train, unl, td, tv, cfg, seed) -> np.ndarray:
    """M10: M9 + balanced self-training. Each round: fit the logistic regression, assign the
    unlabelled target dev windows with Sinkhorn (uniform class marginals) on its log-probs, keep
    the most confident ``keep_frac`` of each class's assigned windows as pseudo-labels, and refit
    on source shared + support + pseudo-labelled windows. Finally predict target val with the
    balanced class offsets of the final model. Deterministic."""
    t3, bal, st = cfg["task3"], cfg["posthoc"]["balance"], cfg["posthoc"]["selftrain"]
    x_tr, y_tr = f[train], proto_lab[train]
    x_cur, y_cur = x_tr, y_tr
    fu = f[unl]
    for _ in range(st["rounds"]):
        clf = logreg_classifier(x_cur, y_cur, c=t3["logreg_c"], seed=seed)
        classes = np.asarray(clf.classes_, dtype=object)
        plan, _, _ = sinkhorn(clf.predict_log_proba(fu), iters=bal["iters"])
        assign = plan.argmax(1)
        conf = plan.max(1) / plan.sum(1)
        keep = np.zeros(len(fu), dtype=bool)
        for c in range(len(classes)):
            idx = np.where(assign == c)[0]
            if len(idx):
                n_keep = int(np.ceil(st["keep_frac"] * len(idx)))
                keep[idx[np.argsort(-conf[idx], kind="stable")[:n_keep]]] = True
        x_cur = np.concatenate([x_tr, fu[keep]])
        y_cur = np.concatenate([y_tr, classes[assign[keep]]])
    clf = logreg_classifier(x_cur, y_cur, c=t3["logreg_c"], seed=seed)
    classes = np.asarray(clf.classes_, dtype=object)
    _, _, g = sinkhorn(clf.predict_log_proba(f[td]), iters=bal["iters"])
    return classes[np.argmax(clf.predict_log_proba(f[tv]) + g[None, :], axis=1)]


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
