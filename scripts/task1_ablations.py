"""Task 1 stretch models and ablations (M8).

The M5 baseline already scores 1.000, so richer models cannot win on accuracy. These
experiments instead ask which models lean on *format* shortcuts and whether the room
signal survives when the format gaps between the rooms are reduced.

Part 1  model x feature comparison (all trained on dev, scored on validation)
        features (per-row normalized): mean | mean+std | percentiles (10/50/90) over time
        models: logistic regression (C tuned on dev holdout), gradient boosting, 1-D CNN
        metrics: standard; three unseen-activity scenarios; shortcut probe = the same model
        scored on validation windows smoothed in frequency and time (Room B's roughness is a
        format cue; a model that relies on it starts calling smoothed B windows "A")
Part 2  harmonization: smooth BOTH rooms (frequency + time), rerun the format audit,
        retrain baseline + format-only control
Part 3  contiguous Room B split: validation = a contiguous block of 25 files per class,
        with 2 neighbouring files on each side dropped, to test the consecutive-chunks problem

Usage:  python scripts/task1_ablations.py [--skip-cnn]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from functools import partial
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.ndimage import uniform_filter1d  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from xroom.config import PROJECT_ROOT, load_config  # noqa: E402
from xroom.dataset import load_windows  # noqa: E402
from xroom.eval import SHARED_ACTIVITIES, room_metrics  # noqa: E402
from xroom.features import data_subcarriers, format_stats, in_chunks  # noqa: E402
from xroom.room_classifier import make_linear_model  # noqa: E402

EPS = 1e-6
EXCLUSIVE = ("empty", "lying", "fall", "sitdown", "standup")
SCENARIOS = {
    "standard": (None, None),
    "sit+stand -> walk": (("sit", "stand"), ("walk",)),
    "shared -> room-only": (SHARED_ACTIVITIES, EXCLUSIVE),
    "room-only -> shared": (EXCLUSIVE, SHARED_ACTIVITIES),
}
TRANSFER = [s for s in SCENARIOS if s != "standard"]
FEATURE_SETS = ("mean", "mean+std", "percentiles")
C_GRID = (0.01, 1.0, 100.0, 1000.0)
SMOOTH = 5  # moving-average width (subcarriers / rows) for the probe and harmonization
DATA_SEGMENTS = ((2, 123), (134, 255))  # contiguous data-subcarrier runs between DC/guard bands


# ---------- transforms ----------

def smooth_freq(x: np.ndarray, w: int) -> np.ndarray:
    """Moving average across neighbouring subcarriers, inside each data segment."""
    out = np.asarray(x, dtype=np.float32).copy()
    for a, b in DATA_SEGMENTS:
        out[..., a:b] = uniform_filter1d(out[..., a:b], size=w, axis=-1, mode="nearest")
    return out


def smooth_time(x: np.ndarray, w: int) -> np.ndarray:
    return uniform_filter1d(np.asarray(x, dtype=np.float32), size=w, axis=1, mode="nearest")


def harmonize(x: np.ndarray, w: int = SMOOTH) -> np.ndarray:
    return smooth_time(smooth_freq(x, w), w)


# ---------- features ----------

def row_normalized(chunk: np.ndarray, data_sc: np.ndarray) -> np.ndarray:
    d = np.asarray(chunk[:, :, data_sc], dtype=np.float32)
    return d / (d.mean(axis=2, keepdims=True) + EPS)


def feature_set(chunk: np.ndarray, data_sc: np.ndarray, kind: str) -> np.ndarray:
    r = row_normalized(chunk, data_sc)
    if kind == "mean":
        return r.mean(1)
    if kind == "mean+std":
        return np.concatenate([r.mean(1), r.std(1)], axis=1)
    if kind == "percentiles":
        return np.percentile(r, [10, 50, 90], axis=1).transpose(1, 0, 2).reshape(len(r), -1)
    raise ValueError(kind)


# ---------- models ----------

class LinearModel:
    name = "logreg"

    def __init__(self, C):
        self.m = make_linear_model(C)

    def fit(self, f, y):
        self.m.fit(f, y)
        return self

    def proba(self, f):
        return self.m.predict_proba(f)[:, list(self.m.classes_).index("B")]


class BoostedTrees:
    name = "grad-boost"

    def __init__(self):
        self.m = HistGradientBoostingClassifier(class_weight="balanced", random_state=0)

    def fit(self, f, y):
        self.m.fit(f, y)
        return self

    def proba(self, f):
        return self.m.predict_proba(f)[:, list(self.m.classes_).index("B")]


class SmallCNN:
    """1-D CNN over time; the 242 per-row normalized subcarriers are input channels."""
    name = "1d-cnn"

    def __init__(self, data_sc, epochs=10, seed=0):
        import torch
        self.torch, self.data_sc, self.epochs, self.seed = torch, data_sc, epochs, seed
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"

    def _prep(self, x):
        r = in_chunks(partial(row_normalized, data_sc=self.data_sc), x)
        return ((r - self.mu) / self.sd).transpose(0, 2, 1).astype(np.float32)  # (n, C, T)

    def fit(self, x, y):
        torch, nn = self.torch, self.torch.nn
        torch.manual_seed(self.seed)
        r = in_chunks(partial(row_normalized, data_sc=self.data_sc), x)
        self.mu, self.sd = r.mean((0, 1)), r.std((0, 1)) + EPS  # fitted on training windows only
        xt = torch.tensor(((r - self.mu) / self.sd).transpose(0, 2, 1).astype(np.float32))
        yt = torch.tensor((np.asarray(y) == "B").astype(np.float32))
        c = len(self.data_sc)
        self.net = nn.Sequential(
            nn.Conv1d(c, 64, 5, padding=2), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 64, 5, padding=2), nn.BatchNorm1d(64), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Dropout(0.2), nn.Linear(64, 1),
        ).to(self.dev)
        n_b = float(yt.sum())
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((len(yt) - n_b) / max(n_b, 1.0), device=self.dev))
        opt = torch.optim.Adam(self.net.parameters(), lr=1e-3, weight_decay=1e-4)
        gen = torch.Generator().manual_seed(self.seed)
        self.net.train()
        for _ in range(self.epochs):
            for idx in torch.randperm(len(yt), generator=gen).split(64):
                opt.zero_grad()
                loss = loss_fn(self.net(xt[idx].to(self.dev)).squeeze(1), yt[idx].to(self.dev))
                loss.backward()
                opt.step()
        return self

    def proba(self, x):
        torch = self.torch
        self.net.eval()
        xt = torch.tensor(self._prep(x))
        with torch.no_grad():
            out = [torch.sigmoid(self.net(b.to(self.dev)).squeeze(1)).cpu().numpy() for b in xt.split(256)]
        return np.concatenate(out)


# ---------- helpers ----------

def bal_acc(y, p) -> float:
    return room_metrics(np.asarray(y), np.asarray(p))["balanced_accuracy"]


def scenario_masks(meta, train_pool, test_pool):
    act = meta["activity"].to_numpy()
    for scen, (tr_a, te_a) in SCENARIOS.items():
        tr = train_pool & (np.isin(act, tr_a) if tr_a else True)
        te = test_pool & (np.isin(act, te_a) if te_a else True)
        yield scen, tr, te


def tune_c(f, meta) -> float:
    y = meta["room"].to_numpy()
    tr, ho = (meta["dev_fold"] == "train").to_numpy(), (meta["dev_fold"] == "holdout").to_numpy()
    losses = {C: room_metrics(y[ho], LinearModel(C).fit(f[tr], y[tr]).proba(f[ho]))["log_loss"] for C in C_GRID}
    return min(losses, key=lambda c: (losses[c], c))


# ---------- part 1 ----------

def part1(x, meta, data_sc, skip_cnn: bool) -> pd.DataFrame:
    y = meta["room"].to_numpy()
    dev, val = (meta["split"] == "dev").to_numpy(), (meta["split"] == "val").to_numpy()
    val_idx = np.where(val)[0]
    x_val_probe = harmonize(x[val_idx])  # shortcut probe: smoothed validation windows
    probe = np.zeros(len(meta), bool)
    rows = []

    def run(label, make, f_all, f_probe_val):
        res = {"model": label[0], "features": label[1]}
        for scen, tr, te in scenario_masks(meta, dev, val):
            m = make().fit(f_all[tr] if f_all is not None else x[np.where(tr)[0]], y[tr])
            if f_all is not None:
                p = m.proba(f_all[te])
            else:
                p = m.proba(x[np.where(te)[0]])
            res[scen] = bal_acc(y[te], p)
            if scen == "standard":  # same model, smoothed validation windows
                pp = m.proba(f_probe_val) if f_all is not None else m.proba(x_val_probe)
                res["probe: smoothed val"] = bal_acc(y[val], pp)
                res["probe: recall B"] = room_metrics(y[val], pp)["recall_B"]
        res["transfer mean"] = float(np.mean([res[s] for s in TRANSFER]))
        rows.append(res)
        print(f"  {label[0]:10s} {label[1]:12s} standard {res['standard']:.3f}  transfer {res['transfer mean']:.3f}  "
              f"probe {res['probe: smoothed val']:.3f} (recall B {res['probe: recall B']:.3f})", flush=True)

    for fs in FEATURE_SETS:
        f = in_chunks(partial(feature_set, data_sc=data_sc, kind=fs), x)
        f_probe = feature_set(x_val_probe, data_sc, fs)
        C = tune_c(f, meta)
        run(("logreg", f"{fs} (C={C:g})"), lambda C=C: LinearModel(C), f, f_probe)
        run(("grad-boost", fs), BoostedTrees, f, f_probe)
    if not skip_cnn:
        run(("1d-cnn", "full window"), lambda: SmallCNN(data_sc), None, None)
    del probe
    return pd.DataFrame(rows)


# ---------- part 2 ----------

def part2(x, meta, data_sc) -> dict:
    y = meta["room"].to_numpy()
    dev, val = (meta["split"] == "dev").to_numpy(), (meta["split"] == "val").to_numpy()
    out = {}
    for w in (1, SMOOTH, 9):
        xh = (lambda c: c) if w == 1 else partial(harmonize, w=w)
        stats = in_chunks(lambda c: format_stats(xh(c), data_sc), x)
        is_b = y[dev] == "B"
        aucs = {}
        for col in stats.columns:
            a = roc_auc_score(is_b, stats[col].to_numpy()[dev])
            aucs[col] = max(a, 1 - a)
        f = in_chunks(lambda c: feature_set(xh(c), data_sc, "mean"), x)
        C = tune_c(f, meta)
        res = {"C": C, "format_stat_auc": aucs}
        for scen, tr, te in scenario_masks(meta, dev, val):
            res[scen] = bal_acc(y[te], LinearModel(C).fit(f[tr], y[tr]).proba(f[te]))
        fmt = stats.to_numpy()
        res["format-only"] = bal_acc(y[val], LinearModel(1.0).fit(fmt[dev], y[dev]).proba(fmt[val]))
        # format-only without the scale statistics (mean amplitude, 99th percentile)
        keep = [i for i, c in enumerate(stats.columns) if c not in ("mean amplitude", "99th percentile")]
        res["format-only (no scale stats)"] = bal_acc(
            y[val], LinearModel(1.0).fit(fmt[dev][:, keep], y[dev]).proba(fmt[val][:, keep]))
        out[f"smoothing width {w}"] = res
        print(f"  width {w}: standard {res['standard']:.3f}  transfer {np.mean([res[s] for s in TRANSFER]):.3f}  "
              f"format-only {res['format-only']:.3f}  (no scale {res['format-only (no scale stats)']:.3f})  "
              f"max format AUC {max(aucs.values()):.3f}, min {min(aucs.values()):.3f}", flush=True)
    return out


# ---------- part 3 ----------

def contiguous_split(meta: pd.DataFrame, per_class=25, guard=2, seed=42) -> pd.Series:
    """Room B: per class, a random contiguous block of ``per_class`` files (by file number) is
    validation; ``guard`` files on each side are dropped. Room A keeps its official split.
    Files without the main naming scheme (e.g. Empty_11.csv) are dropped."""
    rng = np.random.default_rng(seed)
    caps = meta.drop_duplicates("capture_id").set_index("capture_id")
    split = caps["split"].copy()
    for act, g in caps[caps["room"] == "B"].groupby("activity"):
        num = {c: int(m.group(1)) for c in g.index if (m := re.search(r"_(?:Amp_)?(\d+)$", c))
               and ("_Amp_" in c or c.startswith("B/Walking_"))}
        order = sorted(num, key=num.get)
        start = int(rng.integers(0, len(order) - per_class + 1))
        for i, c in enumerate(order):
            if start <= i < start + per_class:
                split[c] = "val"
            elif start - guard <= i < start + per_class + guard:
                split[c] = "drop"
            else:
                split[c] = "dev"
        for c in set(g.index) - set(order):
            split[c] = "drop"
    return meta["capture_id"].map(split)


def part3(x, meta, data_sc) -> dict:
    y = meta["room"].to_numpy()
    f = in_chunks(partial(feature_set, data_sc=data_sc, kind="mean"), x)
    C = tune_c(f, meta)
    out = {}
    for name, split in [("official random split", meta["split"]), ("contiguous Room B split", contiguous_split(meta))]:
        dev, val = (split == "dev").to_numpy(), (split == "val").to_numpy()
        res = {"n_val_windows_B": int((val & (y == "B")).sum())}
        for scen, tr, te in scenario_masks(meta, dev, val):
            res[scen] = bal_acc(y[te], LinearModel(C).fit(f[tr], y[tr]).proba(f[te]))
        res["transfer mean"] = float(np.mean([res[s] for s in TRANSFER]))
        out[name] = res
        print(f"  {name:26s} standard {res['standard']:.3f}  " +
              "  ".join(f"{s} {res[s]:.3f}" for s in TRANSFER), flush=True)
    return out


def plot_part1(df: pd.DataFrame, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [f"{m}\n{f.split(' (')[0]}" for m, f in zip(df["model"], df["features"])]
    cols = [("standard", "standard"), ("transfer mean", "unseen activities (mean)"),
            ("probe: smoothed val", "shortcut probe (smoothed val)")]
    xs = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(11, 4.3))
    for i, (c, lab) in enumerate(cols):
        ax.bar(xs + (i - 1) * 0.27, df[c], width=0.27, label=lab)
    ax.axhline(0.5, ls=":", c="gray", lw=1)
    ax.set_xticks(xs, labels, fontsize=8)
    ax.set_ylim(0.4, 1.02)
    ax.set_ylabel("val balanced accuracy")
    ax.set_title("Room classifiers: accuracy vs reliance on format shortcuts", fontsize=10)
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=3, frameon=False)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-cnn", action="store_true")
    ap.add_argument("--plot-only", action="store_true", help="redraw the figure from part1_models.csv")
    args = ap.parse_args()
    cfg = load_config()
    out_dir = PROJECT_ROOT / "outputs" / "task1" / "ablations"
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.plot_only:
        plot_part1(pd.read_csv(out_dir / "part1_models.csv"), out_dir / "models_vs_shortcuts.png")
        return
    x, meta = load_windows(cfg)
    data_sc = data_subcarriers(cfg["common"]["null_subcarriers"], cfg["common"]["n_subcarriers"])

    t0 = time.time()
    print("Part 1: models x features")
    p1 = part1(x, meta, data_sc, args.skip_cnn)
    p1.to_csv(out_dir / "part1_models.csv", index=False)
    plot_part1(p1, out_dir / "models_vs_shortcuts.png")
    print("Part 2: harmonization (smooth both rooms)")
    p2 = part2(x, meta, data_sc)
    print("Part 3: contiguous Room B split")
    p3 = part3(x, meta, data_sc)
    (out_dir / "results.json").write_text(json.dumps(
        {"part1_models": p1.to_dict("records"), "part2_harmonization": p2, "part3_contiguous_split": p3},
        indent=2), encoding="utf-8")
    pd.set_option("display.width", 220)
    print("\n" + p1.round(3).to_string(index=False))
    print(f"\nsaved to {out_dir} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
