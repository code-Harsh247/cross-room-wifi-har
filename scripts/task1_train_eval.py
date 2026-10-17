"""Task 1 baseline: train, select, evaluate and stress-test the room classifier.

Protocol
1. Model selection on development data only: the feature variant is fixed by design
   (config: per-row normalized spectrum); C is picked on dev_fold=holdout by log-loss.
2. Refit the chosen model on all development windows; score once on validation.
3. Controls (all trained on dev, scored on validation):
   - format-only: same linear model on the 8 format statistics, no spectrum
   - shuffled labels: room labels permuted across dev captures (expect ~50%)
   - leave-activity-out: train on some activities, test on others, to check the
     model recognises rooms rather than activities
4. Ablation: every feature variant at its holdout-best C, scored on validation
   (reported only; validation is never used to choose anything).
5. Robustness checks (is the 100% overfitting or a shortcut?):
   - train vs validation accuracy
   - learning curve: train on only k captures per room
   - nearest centroid: no learning at all, just "closer to the A or B average?"
   - edge subcarriers: weight share near band edges / guard bands, and accuracy without them
   - leave-one-day-out (Room A): hold out a whole recording day of Room A

Usage:  python scripts/task1_train_eval.py [--length 150]
"""
from __future__ import annotations

import argparse
import json
import sys
from functools import partial
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from xroom.config import PROJECT_ROOT, load_config  # noqa: E402
from xroom.dataset import cache_paths, load_windows  # noqa: E402
from xroom.eval import SHARED_ACTIVITIES, full_report, room_metrics  # noqa: E402
from xroom.features import SPECTRUM_VARIANTS, data_subcarriers, format_stats, in_chunks, spectrum_features  # noqa: E402
from xroom.room_classifier import RoomClassifier, make_linear_model  # noqa: E402

N_SHUFFLES = 20
EXCLUSIVE_ACTIVITIES = ("empty", "lying", "fall", "sitdown", "standup")
LEARNING_CURVE_K = (1, 2, 5, 20, 50)
LEARNING_CURVE_REPEATS = 10
# 8 data subcarriers next to each band edge / guard band (FFT order): low SNR, most format-sensitive
EDGE_SUBCARRIERS = (*range(2, 10), *range(115, 123), *range(134, 142), *range(247, 255))


def fit_score(train_x, train_y, test_x, test_y, C=1.0, seed=0) -> tuple[dict, np.ndarray]:
    model = make_linear_model(C, seed).fit(train_x, train_y)
    prob_b = model.predict_proba(test_x)[:, list(model.classes_).index("B")]
    return room_metrics(test_y, prob_b), prob_b


def select_model(feats: dict, meta: pd.DataFrame, c_grid: list[float]) -> pd.DataFrame:
    tr = (meta["dev_fold"] == "train").to_numpy()
    ho = (meta["dev_fold"] == "holdout").to_numpy()
    y = meta["room"].to_numpy()
    shared_ho = ho & meta["activity"].isin(SHARED_ACTIVITIES).to_numpy()
    rows = []
    for variant in SPECTRUM_VARIANTS:
        for C in c_grid:
            m, prob = fit_score(feats[variant][tr], y[tr], feats[variant][ho], y[ho], C)
            shared = room_metrics(y[shared_ho], prob[shared_ho[ho]])
            rows.append({"variant": variant, "C": C, "holdout_log_loss": m["log_loss"],
                         "holdout_balanced_acc": m["balanced_accuracy"],
                         "holdout_shared_balanced_acc": shared["balanced_accuracy"]})
    return pd.DataFrame(rows)


def shuffled_control(f: np.ndarray, meta: pd.DataFrame, dev: np.ndarray, val: np.ndarray, C: float) -> dict:
    """Permute room labels across dev *captures* (windows of a capture keep one label)."""
    accs = []
    dev_meta = meta[dev]
    caps = dev_meta.groupby("capture_id")["room"].first()
    for seed in range(N_SHUFFLES):
        rng = np.random.default_rng(seed)
        fake = pd.Series(rng.permutation(caps.to_numpy()), index=caps.index)
        y_fake = dev_meta["capture_id"].map(fake).to_numpy()
        m, _ = fit_score(f[dev], y_fake, f[val], meta["room"].to_numpy()[val], C, seed)
        accs.append(m["balanced_accuracy"])
    return {"val_balanced_accuracy_mean": float(np.mean(accs)), "val_balanced_accuracy_std": float(np.std(accs)),
            "val_balanced_accuracy_runs": accs}


def leave_activity_out(f: np.ndarray, meta: pd.DataFrame, dev: np.ndarray, val: np.ndarray, C: float) -> dict:
    act = meta["activity"].to_numpy()
    y = meta["room"].to_numpy()
    experiments = {
        "train sit+stand -> test walk": (("sit", "stand"), ("walk",)),
        "train sit+stand+walk -> test room-only activities": (SHARED_ACTIVITIES, EXCLUSIVE_ACTIVITIES),
        "train room-only activities -> test sit+stand+walk": (EXCLUSIVE_ACTIVITIES, SHARED_ACTIVITIES),
    }
    out = {}
    for name, (train_acts, test_acts) in experiments.items():
        tr = dev & np.isin(act, train_acts)
        te = val & np.isin(act, test_acts)
        m, _ = fit_score(f[tr], y[tr], f[te], y[te], C)
        out[name] = {k: m[k] for k in ("n_windows", "accuracy", "balanced_accuracy", "recall_A", "recall_B") if k in m}
    return out


def learning_curve(f: np.ndarray, meta: pd.DataFrame, dev: np.ndarray, val: np.ndarray, C: float) -> dict:
    """Train on k random dev captures per room (all their windows), score on validation."""
    y = meta["room"].to_numpy()
    caps = meta[dev].groupby("capture_id")["room"].first()
    rng = np.random.default_rng(0)
    out = {}
    for k in LEARNING_CURVE_K:
        accs = []
        for _ in range(LEARNING_CURVE_REPEATS):
            pick = [*rng.choice(caps[caps == "A"].index, k, replace=False),
                    *rng.choice(caps[caps == "B"].index, k, replace=False)]
            tr = meta["capture_id"].isin(pick).to_numpy()
            m, _ = fit_score(f[tr], y[tr], f[val], y[val], C)
            accs.append(m["balanced_accuracy"])
        out[str(k)] = {"mean": float(np.mean(accs)), "min": float(np.min(accs)), "max": float(np.max(accs))}
    return out


def nearest_centroid(f: np.ndarray, meta: pd.DataFrame, dev: np.ndarray, val: np.ndarray) -> dict:
    """No learning: assign each validation window to the closer room-average fingerprint."""
    y = meta["room"].to_numpy()
    ca, cb = f[dev & (y == "A")].mean(0), f[dev & (y == "B")].mean(0)
    da, db = np.linalg.norm(f[val] - ca, axis=1), np.linalg.norm(f[val] - cb, axis=1)
    m = room_metrics(y[val], (da > db).astype(float))
    return {"val_balanced_accuracy": m["balanced_accuracy"], "centroid_distance": float(np.linalg.norm(ca - cb)),
            "median_distance_to_own_centroid_A": float(np.median(da[y[val] == "A"])),
            "median_distance_to_own_centroid_B": float(np.median(db[y[val] == "B"]))}


def edge_check(clf: RoomClassifier, f: np.ndarray, y: np.ndarray, dev: np.ndarray, val: np.ndarray,
               data_sc: np.ndarray, C: float) -> dict:
    """Does the model lean on edge / near-guard subcarriers, where format differences are largest?"""
    edge = np.isin(data_sc, EDGE_SUBCARRIERS)
    w = np.abs(clf.model[-1].coef_[0])  # standardized features -> weights are comparable
    m, _ = fit_score(f[dev][:, ~edge], y[dev], f[val][:, ~edge], y[val], C)
    return {"n_edge_subcarriers": int(edge.sum()), "edge_feature_share": float(edge.mean()),
            "edge_weight_share": float(w[edge].sum() / w.sum()),
            "top15_subcarriers": sorted(int(s) for s in data_sc[np.argsort(w)[::-1][:15]]),
            "val_balanced_accuracy_without_edges": m["balanced_accuracy"]}


def leave_one_day_out(f: np.ndarray, meta: pd.DataFrame, dev: np.ndarray, val: np.ndarray, C: float,
                      captures_csv: Path) -> dict:
    """Hold out every Room A capture recorded on one day (dev and val); train on the remaining
    Room A dev days + all Room B dev. Tests whether the fingerprint survives a new session.
    Room B has no timestamps, so only Room A can be checked this way."""
    caps = pd.read_csv(captures_csv)
    t0 = caps.set_index("capture_id")["t_start_unix"]
    day = pd.to_datetime(meta["capture_id"].map(t0), unit="s").dt.date.astype(str).to_numpy()
    y = meta["room"].to_numpy()
    is_a = y == "A"
    rows = {}
    for d in sorted(set(day[is_a])):
        tr = dev & (~is_a | (day != d))
        te = is_a & (day == d)
        model = make_linear_model(C).fit(f[tr], y[tr])
        rows[d] = {"n_windows": int(te.sum()), "recall_A": float((model.predict(f[te]) == "A").mean()),
                   "val_recall_B": float((model.predict(f[val & ~is_a]) == "B").mean())}
    n = np.array([r["n_windows"] for r in rows.values()])
    rec = np.array([r["recall_A"] for r in rows.values()])
    return {"per_day": rows, "weighted_recall_A": float((n * rec).sum() / n.sum()), "worst_day_recall_A": float(rec.min())}


def plot_learning_curve(lc: dict, full_acc: float, n_full: int, path: Path) -> None:
    ks = [int(k) for k in lc]
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    ax.plot(ks, [lc[str(k)]["mean"] for k in ks], "o-", label="mean of 10 draws")
    ax.fill_between(ks, [lc[str(k)]["min"] for k in ks], [lc[str(k)]["max"] for k in ks], alpha=0.2, label="min-max")
    ax.axhline(full_acc, ls="--", c="gray", lw=1, label=f"all dev (~{n_full}/room)")
    ax.axhline(0.5, ls=":", c="gray", lw=1)
    ax.set_xscale("log")
    ax.set_xticks(ks, [str(k) for k in ks])
    ax.set_xlabel("training captures per room")
    ax.set_ylabel("val balanced accuracy")
    ax.set_ylim(0.3, 1.02)
    ax.set_title("Room classifier learning curve", fontsize=10)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_confusion(cm: list, path: Path, title: str) -> None:
    cm = np.array(cm)
    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    ax.imshow(cm / cm.sum(1, keepdims=True), cmap="Blues", vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm[i].sum() / 2 else "black", fontsize=12)
    ax.set_xticks([0, 1], ["A", "B"])
    ax.set_yticks([0, 1], ["A", "B"])
    ax.set_xlabel("predicted room")
    ax.set_ylabel("true room")
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--length", type=int, default=None)
    args = ap.parse_args()
    cfg = load_config()
    x, meta = load_windows(cfg, args.length)
    length = x.shape[1]
    out_dir = PROJECT_ROOT / "outputs" / "task1" / f"L{length}"
    out_dir.mkdir(parents=True, exist_ok=True)
    data_sc = data_subcarriers(cfg["common"]["null_subcarriers"], x.shape[2])

    print(f"computing features for {len(x)} windows of length {length} ...")
    feats = {v: in_chunks(partial(spectrum_features, data_sc=data_sc, variant=v), x) for v in SPECTRUM_VARIANTS}
    fmt = in_chunks(partial(format_stats, data_sc=data_sc), x).to_numpy()

    dev = (meta["split"] == "dev").to_numpy()
    val = (meta["split"] == "val").to_numpy()
    y = meta["room"].to_numpy()

    # 1. model selection on dev only
    rc = cfg["room_classifier"]
    sel = select_model(feats, meta, rc["c_grid"])
    sel.to_csv(out_dir / "model_selection.csv", index=False)
    variant = rc["feature_variant"]
    best = sel[sel["variant"] == variant].sort_values(["holdout_log_loss", "C"]).iloc[0]
    C = float(best["C"])
    print(f"selected on dev holdout: variant={variant}, C={C} (log-loss {best['holdout_log_loss']:.4f})")

    # 2. final model: all dev -> validation
    clf = RoomClassifier(data_sc, variant=variant, C=C).fit_features(feats[variant][dev], y[dev])
    prob_val = clf.proba_from_features(feats[variant][val])
    val_meta = meta[val]
    main_report = full_report(val_meta, prob_val)
    clf.save(out_dir / "room_classifier.joblib")
    pd.DataFrame({
        "capture_id": val_meta["capture_id"], "start": val_meta["start"], "activity": val_meta["activity"],
        "true_room": val_meta["room"], "pred_room": np.where(prob_val >= 0.5, "B", "A"), "prob_B": prob_val,
    }).to_csv(out_dir / "predictions_val.csv", index=False)
    plot_confusion(main_report["all_activities"]["confusion_matrix(rows=true A,B; cols=pred A,B)"],
                   out_dir / "confusion_matrix_val.png", f"Room classifier (val), variant={variant}")

    # 3. controls
    fmt_all, _ = fit_score(fmt[dev], y[dev], fmt[val], y[val])
    shared_val = val & meta["activity"].isin(SHARED_ACTIVITIES).to_numpy()
    fmt_shared, _ = fit_score(fmt[dev], y[dev], fmt[shared_val], y[shared_val])
    controls = {
        "format_only": {"all_activities": fmt_all, "shared_activities_sit_stand_walk": fmt_shared},
        "shuffled_labels": shuffled_control(feats[variant], meta, dev, val, C),
        "leave_activity_out": leave_activity_out(feats[variant], meta, dev, val, C),
    }

    # 4. ablation over feature variants (reported, not used for selection)
    ablation = {}
    for v in SPECTRUM_VARIANTS:
        c_v = float(sel[sel["variant"] == v].sort_values(["holdout_log_loss", "C"]).iloc[0]["C"])
        m_all, p = fit_score(feats[v][dev], y[dev], feats[v][val], y[val], c_v)
        m_shared = room_metrics(y[val][shared_val[val]], p[shared_val[val]])
        ablation[v] = {"C": c_v, "val_balanced_accuracy": m_all["balanced_accuracy"],
                       "val_shared_balanced_accuracy": m_shared["balanced_accuracy"],
                       "val_log_loss": m_all["log_loss"]}

    # 5. robustness checks
    f_main = feats[variant]
    lc = learning_curve(f_main, meta, dev, val, C)
    robustness = {
        "train_balanced_accuracy": room_metrics(y[dev], clf.proba_from_features(f_main[dev]))["balanced_accuracy"],
        "learning_curve_captures_per_room": lc,
        "nearest_centroid": nearest_centroid(f_main, meta, dev, val),
        "edge_subcarriers": edge_check(clf, f_main, y, dev, val, data_sc, C),
        "leave_one_day_out_room_A": leave_one_day_out(f_main, meta, dev, val, C, cache_paths(cfg, length)[2]),
    }
    n_dev_caps = meta[dev].groupby("room")["capture_id"].nunique().min()
    plot_learning_curve(lc, main_report["all_activities"]["balanced_accuracy"], n_dev_caps,
                        out_dir / "learning_curve_val.png")

    results = {"window_length": length, "selected": {"variant": variant, "C": C},
               "validation": main_report, "controls": controls, "ablation_feature_variant": ablation,
               "robustness": robustness}
    (out_dir / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    # summary
    a, s, c = main_report["all_activities"], main_report["shared_activities_sit_stand_walk"], main_report["capture_level"]
    print("\n=== Validation (final model) ===")
    print(f"all activities     acc {a['accuracy']:.4f}  balanced {a['balanced_accuracy']:.4f}  "
          f"recall A {a['recall_A']:.4f}  recall B {a['recall_B']:.4f}  (n={a['n_windows']})")
    print(f"shared sit/stand/walk  balanced {s['balanced_accuracy']:.4f}  (n={s['n_windows']})")
    print(f"capture level      balanced {c['balanced_accuracy']:.4f}  (n={c['n_windows']} captures)")
    print("per room/activity accuracy:", {k: round(v, 3) for k, v in main_report["per_room_activity_accuracy"].items()})
    print("\n=== Controls ===")
    print(f"format-only        balanced {fmt_all['balanced_accuracy']:.4f}  shared {fmt_shared['balanced_accuracy']:.4f}")
    sh = controls["shuffled_labels"]
    print(f"shuffled labels    balanced {sh['val_balanced_accuracy_mean']:.4f} +/- {sh['val_balanced_accuracy_std']:.4f} "
          f"(mean +/- std of {N_SHUFFLES})")
    for name, m in controls["leave_activity_out"].items():
        extra = f"balanced {m['balanced_accuracy']:.4f}" if "balanced_accuracy" in m else f"acc {m['accuracy']:.4f}"
        print(f"{name:52s} {extra}  recall A {m.get('recall_A', float('nan')):.3f}  recall B {m.get('recall_B', float('nan')):.3f}")
    print("\n=== Ablation (feature variant) ===")
    for v, m in ablation.items():
        print(f"{v:5s} C={m['C']:<6} balanced {m['val_balanced_accuracy']:.4f}  shared {m['val_shared_balanced_accuracy']:.4f}  log-loss {m['val_log_loss']:.4f}")
    r = robustness
    print("\n=== Robustness ===")
    print(f"train balanced acc {r['train_balanced_accuracy']:.4f}  (val {a['balanced_accuracy']:.4f})")
    print("learning curve (captures/room -> val balanced acc mean [min, max]):")
    for k, v in r["learning_curve_captures_per_room"].items():
        print(f"   {k:>3}: {v['mean']:.3f} [{v['min']:.3f}, {v['max']:.3f}]")
    nc = r["nearest_centroid"]
    print(f"nearest centroid   balanced {nc['val_balanced_accuracy']:.4f}  centroid gap {nc['centroid_distance']:.2f} vs "
          f"median spread A {nc['median_distance_to_own_centroid_A']:.2f} B {nc['median_distance_to_own_centroid_B']:.2f}")
    e = r["edge_subcarriers"]
    print(f"edge subcarriers   {e['edge_feature_share']:.0%} of features carry {e['edge_weight_share']:.0%} of weight; "
          f"without them balanced {e['val_balanced_accuracy_without_edges']:.4f}")
    d = r["leave_one_day_out_room_A"]
    print(f"leave-one-day-out  Room A recall on unseen days {d['weighted_recall_A']:.3f} (worst day {d['worst_day_recall_A']:.3f})")
    for day, v in d["per_day"].items():
        print(f"   {day}: n={v['n_windows']:>3}  recall A {v['recall_A']:.3f}  val recall B {v['val_recall_B']:.3f}")
    print(f"\nsaved to {out_dir}")


if __name__ == "__main__":
    main()
