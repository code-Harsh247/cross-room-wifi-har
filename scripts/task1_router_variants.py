"""Router variants: can the room classifier cope better with activities it never saw?

M6 suggested two cheap changes for the Task 3 room router:
  1. stronger regularization (smaller C)
  2. slice averaging: cut each 150-row window into s-row slices, score every slice
     with a model trained on s-row slices, and average the slice probabilities

Grid: slice rows s in {150 (= no averaging), 75, 50, 25, 15, 10} x C in {0.01 ... 1000}.
Scenarios (train activities -> test activities):
  standard            all -> all (must stay ~100%)
  sit+stand -> walk   still-only training, unseen walking
  shared -> room-only sit/stand/walk training, unseen room-specific activities
  room-only -> shared room-specific training, unseen sit/stand/walk

Selection uses development data only. Models are trained on dev_fold=train. The standard
scenario is scored on dev_fold=holdout. Transfer scenarios are scored on *all* dev captures
of the test activities: those activities are never in the training set, so every dev capture
of them is unseen, and the holdout fold alone was too small (its walking test scored 1.000 for
27 of 30 variants). Rule: best mean balanced accuracy over the three transfer scenarios among
variants whose standard score is >= 0.995. Validation is reported for every variant.
The test unit is always a 150-row window (what Task 3 will route).

Usage:  python scripts/task1_router_variants.py
"""
from __future__ import annotations

import json
import sys
import time
from functools import partial
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from xroom.config import PROJECT_ROOT, load_config  # noqa: E402
from xroom.dataset import load_sequences, windows_from_sequences  # noqa: E402
from xroom.eval import SHARED_ACTIVITIES, room_metrics  # noqa: E402
from xroom.features import data_subcarriers, in_chunks, spectrum_features  # noqa: E402
from xroom.dataset import load_windows  # noqa: E402
from xroom.room_classifier import RoomClassifier, make_linear_model  # noqa: E402

WINDOW = 150
SLICE_ROWS = (150, 75, 50, 25, 15, 10)
C_GRID = (0.01, 0.1, 1.0, 10.0, 1000.0)
EXCLUSIVE = ("empty", "lying", "fall", "sitdown", "standup")
SCENARIOS = {
    "standard": (None, None),
    "sit+stand -> walk": (("sit", "stand"), ("walk",)),
    "shared -> room-only": (SHARED_ACTIVITIES, EXCLUSIVE),
    "room-only -> shared": (EXCLUSIVE, SHARED_ACTIVITIES),
}
TRANSFER = [s for s in SCENARIOS if s != "standard"]
MIN_STANDARD = 0.995


def slices_for(table, seqs, s: int, data_sc) -> tuple[np.ndarray, pd.DataFrame]:
    """Features of every s-row slice that lies inside a 150-row parent window."""
    x, meta, _ = windows_from_sequences(table, seqs, s)
    seq_len = meta["capture_id"].map({k: len(v) for k, v in seqs.items()}).to_numpy()
    parent = (meta["start"].to_numpy() // WINDOW) * WINDOW
    keep = parent + WINDOW <= seq_len  # parents are the 150-row windows used everywhere else
    f = in_chunks(partial(spectrum_features, data_sc=data_sc, variant="row"), x)[keep]
    meta = meta[keep].reset_index(drop=True)
    meta["parent"] = meta["capture_id"] + "@" + pd.Series(parent[keep]).astype(str)
    return f, meta


def window_predictions(meta: pd.DataFrame, test: np.ndarray, prob_b: np.ndarray) -> pd.DataFrame:
    """Average slice probabilities per 150-row parent window."""
    df = meta.loc[test, ["parent", "capture_id", "room"]].assign(p=prob_b)
    win = df.groupby("parent").agg(capture_id=("capture_id", "first"), room=("room", "first"), p=("p", "mean"))
    return win.assign(correct=np.where(win["p"] >= 0.5, "B", "A") == win["room"])


def window_scores(win: pd.DataFrame) -> dict:
    m = room_metrics(win["room"].to_numpy(), win["p"].to_numpy())
    return {"balanced_accuracy": m.get("balanced_accuracy", m["accuracy"]), "recall_A": m.get("recall_A"),
            "recall_B": m.get("recall_B"), "n_windows": int(len(win))}


def paired_bootstrap(base: pd.DataFrame, other: pd.DataFrame, reps: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """Difference in balanced accuracy (other - base) on the same windows, with a 95% interval
    from resampling captures within each room (same resample for both models)."""
    df = base[["capture_id", "room", "correct"]].join(other["correct"].rename("correct_o"))
    cap = df.groupby("capture_id").agg(room=("room", "first"), n=("correct", "size"),
                                       ka=("correct", "sum"), kb=("correct_o", "sum"))
    rng = np.random.default_rng(seed)
    diffs, point = [], []
    for _, g in cap.groupby("room"):
        n, ka, kb = (g[c].to_numpy(dtype=float) for c in ("n", "ka", "kb"))
        idx = rng.integers(0, len(g), size=(reps, len(g)))
        diffs.append((kb[idx].sum(1) - ka[idx].sum(1)) / n[idx].sum(1))
        point.append((kb.sum() - ka.sum()) / n.sum())
    d = np.mean(diffs, axis=0)
    return float(np.mean(point)), float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))


def main() -> None:
    cfg = load_config()
    out_dir = PROJECT_ROOT / "outputs" / "task1" / "router_variants"
    out_dir.mkdir(parents=True, exist_ok=True)
    table, seqs = load_sequences(cfg)
    data_sc = data_subcarriers(cfg["common"]["null_subcarriers"], cfg["common"]["n_subcarriers"])

    rows, val_windows = [], {}
    for s in SLICE_ROWS:
        t0 = time.time()
        f, meta = slices_for(table, seqs, s, data_sc)
        y, act = meta["room"].to_numpy(), meta["activity"].to_numpy()
        dev, dev_train = (meta["split"] == "dev").to_numpy(), (meta["dev_fold"] == "train").to_numpy()
        pools = {
            "dev": (dev_train, (meta["dev_fold"] == "holdout").to_numpy(), dev),
            "val": (dev, (meta["split"] == "val").to_numpy(), (meta["split"] == "val").to_numpy()),
        }
        for phase, (train_pool, standard_test, transfer_test) in pools.items():
            for scen, (tr_acts, te_acts) in SCENARIOS.items():
                test_pool = standard_test if scen == "standard" else transfer_test
                tr = train_pool & (np.isin(act, tr_acts) if tr_acts else True)
                te = test_pool & (np.isin(act, te_acts) if te_acts else True)
                for C in C_GRID:
                    model = make_linear_model(C).fit(f[tr], y[tr])
                    p = model.predict_proba(f[te])[:, list(model.classes_).index("B")]
                    win = window_predictions(meta, te, p)
                    if phase == "val":
                        val_windows[(s, C, scen)] = win
                    rows.append({"slice_rows": s, "C": C, "phase": phase, "scenario": scen, **window_scores(win)})
        print(f"slice {s:>3} rows done in {time.time() - t0:.0f}s", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "results.csv", index=False)

    # selection on development data only
    wide = df.pivot_table(index=["phase", "slice_rows", "C"], columns="scenario", values="balanced_accuracy")
    wide["transfer_mean"] = wide[TRANSFER].mean(axis=1)
    ho = wide.loc["dev"]
    eligible = ho[ho["standard"] >= MIN_STANDARD]
    best = eligible.sort_values(["transfer_mean", "slice_rows", "C"], ascending=[False, False, True]).index[0]
    baseline = (150, 1000.0)
    summary = {
        "selected_on_dev": {"slice_rows": int(best[0]), "C": float(best[1])},
        "baseline": {"slice_rows": baseline[0], "C": baseline[1]},
        "dev": {"selected": ho.loc[best].to_dict(), "baseline": ho.loc[baseline].to_dict()},
        "val": {"selected": wide.loc["val"].loc[best].to_dict(), "baseline": wide.loc["val"].loc[baseline].to_dict()},
    }
    paired = pd.DataFrame([
        {"slice_rows": s, "C": C, "scenario": scen,
         **dict(zip(("diff_vs_baseline", "ci_low", "ci_high"),
                    paired_bootstrap(val_windows[(*baseline, scen)], val_windows[(s, C, scen)])))}
        for s in SLICE_ROWS for C in C_GRID for scen in TRANSFER if (s, C) != baseline
    ])
    paired.to_csv(out_dir / "paired_vs_baseline_val.csv", index=False)
    summary["val_paired_vs_baseline_selected"] = paired[
        (paired["slice_rows"] == best[0]) & (paired["C"] == best[1])].drop(columns=["slice_rows", "C"]).to_dict("records")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    pd.set_option("display.width", 200)
    for phase in ("dev", "val"):
        print(f"\n=== {phase}: balanced accuracy on 150-row windows ===")
        print(wide.loc[phase][["standard", *TRANSFER, "transfer_mean"]].round(3).to_string())
    print(f"\nselected on dev: slice_rows={best[0]}, C={best[1]}  (baseline: 150 rows, C=1000)")
    for phase in ("dev", "val"):
        b, s_ = summary[phase]["baseline"], summary[phase]["selected"]
        print(f"{phase:8s} baseline: standard {b['standard']:.3f} transfer mean {b['transfer_mean']:.3f} | "
              f"selected: standard {s_['standard']:.3f} transfer mean {s_['transfer_mean']:.3f}")
    print("\nval, paired bootstrap vs baseline (diff [95% CI]):")
    show = paired[paired["C"].isin([0.1, 1.0])].copy()
    show["v"] = show.apply(lambda r: f"{r.diff_vs_baseline:+.3f} [{r.ci_low:+.3f},{r.ci_high:+.3f}]", axis=1)
    print(show.pivot_table(index=["slice_rows", "C"], columns="scenario", values="v", aggfunc="first")[TRANSFER].to_string())

    # final router: selected variant, trained on all dev 150-row windows (all activities)
    x150, meta150 = load_windows(cfg, WINDOW)
    dev150, val150 = (meta150["split"] == "dev").to_numpy(), (meta150["split"] == "val").to_numpy()
    router = RoomClassifier(data_sc, variant="row", C=float(best[1]), slice_rows=int(best[0]))
    router.fit(x150[dev150], meta150["room"].to_numpy()[dev150])
    p = router.predict_proba(x150[val150])
    vm = meta150[val150]
    router_dir = PROJECT_ROOT / "outputs" / "task1" / "router"
    router.save(router_dir / "room_router.joblib")
    pd.DataFrame({"capture_id": vm["capture_id"], "start": vm["start"], "activity": vm["activity"],
                  "true_room": vm["room"], "pred_room": np.where(p >= 0.5, "B", "A"), "prob_B": p}
                 ).to_csv(router_dir / "predictions_val.csv", index=False)
    std = room_metrics(vm["room"].to_numpy(), p)["balanced_accuracy"]
    print(f"\nsaved router (slice_rows={best[0]}, C={best[1]}) -> {router_dir}; val standard balanced acc {std:.4f}")

    # figure: validation, one panel per scenario
    fig, axes = plt.subplots(1, len(SCENARIOS), figsize=(16, 3.6), sharey=True)
    axes[0].set_ylim(0.83, 1.005)  # one scale for all panels so differences are comparable
    v = df[df["phase"] == "val"]
    for ax, scen in zip(axes, SCENARIOS):
        for C in C_GRID:
            d = v[(v["scenario"] == scen) & (v["C"] == C)].sort_values("slice_rows")
            ax.plot(d["slice_rows"], d["balanced_accuracy"], "o-", ms=3, label=f"C={C:g}")
        ax.set_xscale("log")
        ax.set_xticks(SLICE_ROWS, [str(s) for s in SLICE_ROWS])
        ax.set_xlabel("slice rows (150 = no averaging)")
        ax.set_title(scen, fontsize=10)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("val balanced accuracy (150-row windows)")
    axes[-1].legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_dir / "router_variants_val.png", dpi=150)
    plt.close(fig)
    print(f"\nsaved to {out_dir}")


if __name__ == "__main__":
    main()
