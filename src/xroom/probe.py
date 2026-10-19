"""Linear-probe evaluation of frozen features (Task 2, PRD FR-5/FR-6).

Protocol for one probe:
  1. C is tuned on development data only: fit on ``dev_fold == "train"``, macro-F1 on
     ``dev_fold == "holdout"``.
  2. The probe (StandardScaler + class-balanced logistic regression) is refit on all
     training rows with that C.
  3. Validation rows are scored once: window macro-F1, per-class F1, confusion matrix,
     capture-level macro-F1 (mean of a capture's window probabilities) and a 95%
     bootstrap interval that resamples validation captures within each class.

Also: the hand-crafted (per-row normalized mean + std) and PCA baseline features.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from threadpoolctl import threadpool_limits

from .room_classifier import make_linear_model
from .ssl_data import NormStats, apply_norm, prepare

SHARED = ("sit", "stand", "walk")


def macro_f1(y_true, y_pred, labels=None) -> float:
    """Unweighted mean of per-class F1 over ``labels`` (default: classes in y_true or y_pred)."""
    return float(np.mean(list(per_class_f1(y_true, y_pred, labels).values())))


def per_class_f1(y_true, y_pred, labels=None) -> dict:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    labels = sorted(set(y_true) | set(y_pred)) if labels is None else list(labels)
    out = {}
    for c in labels:
        tp = np.sum((y_true == c) & (y_pred == c))
        denom = np.sum(y_true == c) + np.sum(y_pred == c)
        out[str(c)] = float(2 * tp / denom) if denom else 0.0
    return out


def shuffle_by_capture(y: np.ndarray, captures: np.ndarray, seed: int = 0) -> np.ndarray:
    """Permute labels between captures (all windows of a capture keep one shared label)."""
    caps, first = np.unique(captures, return_index=True)
    perm = np.random.default_rng(seed).permutation(len(caps))
    mapping = dict(zip(caps, np.asarray(y)[first][perm]))
    return np.array([mapping[c] for c in captures])


def bootstrap_macro_f1(y: np.ndarray, pred: np.ndarray, captures: np.ndarray, reps: int = 1000,
                       seed: int = 0, level: float = 0.95) -> tuple[float, float]:
    """CI for window-level macro-F1, resampling captures with replacement within each class."""
    y, pred, captures = np.asarray(y), np.asarray(pred), np.asarray(captures)
    labels = sorted(set(y))
    cap_rows = pd.Series(np.arange(len(y))).groupby(captures).apply(list)
    cap_class = pd.Series(y).groupby(captures).first()
    by_class = [cap_class.index[cap_class == c].to_numpy() for c in labels]
    rng = np.random.default_rng(seed)
    stats = np.empty(reps)
    for r in range(reps):
        caps = np.concatenate([g[rng.integers(0, len(g), len(g))] for g in by_class])
        rows = np.concatenate([cap_rows[c] for c in caps])
        stats[r] = macro_f1(y[rows], pred[rows], labels)
    a = (1 - level) / 2
    return float(np.quantile(stats, a)), float(np.quantile(stats, 1 - a))


def tune_c(f: np.ndarray, y: np.ndarray, fold: np.ndarray, c_grid, seed: int = 0) -> tuple[float, dict]:
    """Pick C by holdout macro-F1. ``f``/``y``/``fold`` are development rows only."""
    tr, ho = fold == "train", fold == "holdout"
    if not tr.any() or not ho.any():
        raise ValueError("tuning needs both dev_fold train and holdout rows")
    labels = sorted(set(y[ho]))
    scores = {}
    for c in c_grid:
        m = make_linear_model(c, seed).fit(f[tr], y[tr])
        scores[float(c)] = macro_f1(y[ho], m.predict(f[ho]), labels)
    best = max(scores, key=lambda c: (scores[c], -c))  # ties -> smaller C (stronger regularization)
    return best, scores


def linear_probe(*args, **kwargs) -> dict:
    """See ``_linear_probe``. Runs with one BLAS thread: these small logistic regressions are
    ~60x slower when BLAS spreads them over all cores."""
    with threadpool_limits(limits=1, user_api="blas"):
        return _linear_probe(*args, **kwargs)


def _linear_probe(f: np.ndarray, y: np.ndarray, meta: pd.DataFrame, train: np.ndarray, test: np.ndarray,
                 c_grid, split: pd.Series | np.ndarray | None = None, reps: int = 1000, seed: int = 0,
                 shuffle_labels: bool = False) -> dict:
    """Tune on dev folds of ``train`` rows, refit on all ``train`` rows, score ``test`` rows once.

    ``split`` (default ``meta.split``) must be "dev" on every training row and "val" on every
    test row; this is checked, so validation can never leak into tuning or fitting.
    """
    split = np.asarray(meta["split"] if split is None else split)
    train, test = np.asarray(train, bool), np.asarray(test, bool)
    if not (split[train] == "dev").all():
        raise AssertionError("probe training rows must all be development rows")
    if not (split[test] == "val").all():
        raise AssertionError("probe test rows must all be validation rows")
    if (train & test).any():
        raise AssertionError("a row is both training and test")
    y = np.asarray(y).astype(str)
    captures = meta["capture_id"].to_numpy()
    y_train = y[train]
    if shuffle_labels:
        y_train = shuffle_by_capture(y_train, captures[train], seed)
    best_c, scores = tune_c(f[train], y_train, meta["dev_fold"].to_numpy()[train], c_grid, seed)
    model = make_linear_model(best_c, seed).fit(f[train], y_train)
    prob = model.predict_proba(f[test])
    classes = model.classes_
    pred = classes[prob.argmax(1)]
    y_test = y[test]
    labels = sorted(set(y_test))
    cap = pd.DataFrame(prob, columns=classes).assign(cap=captures[test], y=y_test).groupby("cap")
    cap_prob, cap_y = cap[list(classes)].mean(), cap["y"].first()
    cap_pred = classes[cap_prob.to_numpy().argmax(1)]
    lo, hi = bootstrap_macro_f1(y_test, pred, captures[test], reps, seed)
    return {
        "macro_f1": macro_f1(y_test, pred, labels),
        "ci95": [lo, hi],
        "capture_macro_f1": macro_f1(cap_y.to_numpy(), cap_pred, labels),
        "accuracy": float((pred == y_test).mean()),
        "per_class_f1": per_class_f1(y_test, pred, labels),
        "confusion": {"labels": labels,
                      "matrix(rows=true, cols=pred)": pd.crosstab(pd.Categorical(y_test, labels),
                                                                  pd.Categorical(pred, labels),
                                                                  dropna=False).to_numpy().tolist()},
        "C": best_c, "tuning_holdout_macro_f1": scores,
        "n_train": int(train.sum()), "n_test": int(test.sum()), "n_test_captures": int(len(cap_y)),
        "chance_macro_f1_uniform": 1.0 / len(labels),
    }


def dev_holdout_scores(f: np.ndarray, meta: pd.DataFrame, c_grid, seed: int = 0) -> dict:
    """Per-room activity macro-F1 on ``dev_fold == "holdout"`` (probe fit on dev-train, best C).
    Uses development rows only, so it may drive choices between encoders; validation may not."""
    dev = (meta["split"] == "dev").to_numpy()
    out = {}
    with threadpool_limits(limits=1, user_api="blas"):
        for room in ("A", "B"):
            m = dev & (meta["room"] == room).to_numpy()
            best, scores = tune_c(f[m], meta["activity"].to_numpy()[m].astype(str),
                                  meta["dev_fold"].to_numpy()[m], c_grid, seed)
            out[f"room_{room}"] = scores[best]
    out["mean"] = (out["room_A"] + out["room_B"]) / 2
    return out


def probe_suite(f: np.ndarray, meta: pd.DataFrame, c_grid, reps: int = 1000, seed: int = 0,
                shuffle_labels: bool = False, split=None, sets=None) -> dict:
    """The standard probe sets: per-room A and B (headline = their mean), joint, cross-room."""
    split = np.asarray(meta["split"] if split is None else split)
    room, act = meta["room"].to_numpy(), meta["activity"].to_numpy()
    dev, val = split == "dev", split == "val"
    shared = np.isin(act, SHARED)
    defs = {
        "room_A": (dev & (room == "A"), val & (room == "A")),
        "room_B": (dev & (room == "B"), val & (room == "B")),
        "joint": (dev, val),
        "cross_A_to_B": (dev & (room == "A") & shared, val & (room == "B") & shared),
        "cross_B_to_A": (dev & (room == "B") & shared, val & (room == "A") & shared),
    }
    out = {}
    for name, (tr, te) in defs.items():
        if sets is not None and name not in sets:
            continue
        if tr.any() and te.any():
            out[name] = linear_probe(f, act, meta, tr, te, c_grid, split, reps, seed, shuffle_labels)
    if "room_A" in out and "room_B" in out:
        out["headline_mean_room_macro_f1"] = (out["room_A"]["macro_f1"] + out["room_B"]["macro_f1"]) / 2
    return out


# ---------- baseline features ----------


def handcrafted_features(x: np.ndarray, data_sc: np.ndarray, chunk: int = 256) -> np.ndarray:
    """Per-row normalized amplitude, mean and std over time per subcarrier (2 x 242)."""
    parts = []
    for i in range(0, len(x), chunk):
        r = prepare(np.asarray(x[i : i + chunk]), data_sc, row=True)
        parts.append(np.concatenate([r.mean(1), r.std(1)], axis=1))
    return np.concatenate(parts).astype(np.float32)


def flat_windows(x: np.ndarray, stats: NormStats, pool: int = 1, chunk: int = 256) -> np.ndarray:
    """Normalized windows, averaged over ``pool`` rows, flattened to vectors."""
    parts = []
    for i in range(0, len(x), chunk):
        d = apply_norm(np.asarray(x[i : i + chunk]), stats)
        if pool > 1:
            t = d.shape[1] // pool * pool
            d = d[:, :t].reshape(len(d), t // pool, pool, -1).mean(2)
        parts.append(d.reshape(len(d), -1))
    return np.concatenate(parts).astype(np.float32)


def pca_features(flat: np.ndarray, fit_rows: np.ndarray, dim: int, seed: int = 0) -> np.ndarray:
    """PCA fitted on ``fit_rows`` (development only), applied to every row."""
    pca = PCA(n_components=dim, svd_solver="randomized", random_state=seed).fit(flat[fit_rows])
    return pca.transform(flat).astype(np.float32)
