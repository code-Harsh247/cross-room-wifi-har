"""Task 2 figures (T8).

1. UMAP of validation embeddings from the default encoder: UMAP (and the scaler before it)
   is fitted on DEVELOPMENT embeddings only, then validation is transformed (seed 0).
   Panel (a) colour = room; panel (b) colour = activity, marker = room.
2. SSL loss curves (train and dev-holdout) for the three default seeds.
3. Per-room probe macro-F1: SSL vs baselines, error bars = std over seeds (3 seeds) or the
   95% capture-bootstrap interval for the deterministic baselines (single run).

Writes outputs/task2/figures/*.png and copies them to reports/figures/task2_*.png.

Usage:  python scripts/task2_figures.py [--encoder default] [--seeds simclr_s0 simclr_s1 simclr_s2]
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from xroom.config import load_task2_config  # noqa: E402
from xroom.dataset import load_windows  # noqa: E402
from xroom.encoder import embed, load_encoder  # noqa: E402

# Reference categorical palette, fixed order (dataviz skill, light mode).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3dd"
ACTIVITIES = ["empty", "lying", "sit", "stand", "walk", "fall", "sitdown", "standup"]
ROOM_COLOR = {"A": SERIES[0], "B": SERIES[1]}
ROOM_MARKER = {"A": "o", "B": "^"}

plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "text.color": INK, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6})


def umap_figure(cfg, encoder: str, out: Path) -> None:
    import umap

    x, meta = load_windows(cfg)
    f = embed(load_encoder(cfg, encoder), x)
    dev, val = (meta["split"] == "dev").to_numpy(), (meta["split"] == "val").to_numpy()
    scaler = StandardScaler().fit(f[dev])
    reducer = umap.UMAP(n_neighbors=30, min_dist=0.3, random_state=0).fit(scaler.transform(f[dev]))
    xy = reducer.transform(scaler.transform(f[val]))
    mv = meta[val].reset_index(drop=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6), constrained_layout=True)
    for ax in axes:
        ax.set_xticks([]), ax.set_yticks([])
        ax.grid(False)
        ax.set_xlabel("UMAP 1"), ax.set_ylabel("UMAP 2")
    for room in "AB":
        m = (mv["room"] == room).to_numpy()
        axes[0].scatter(*xy[m].T, s=22, marker=ROOM_MARKER[room], c=ROOM_COLOR[room], edgecolors="white",
                        linewidths=0.6, label=f"Room {room} (n={m.sum()})")
    axes[0].legend(frameon=False, loc="best")
    axes[0].set_title("(a) coloured by room", loc="left")
    for i, act in enumerate(ACTIVITIES):
        for room in "AB":
            m = ((mv["activity"] == act) & (mv["room"] == room)).to_numpy()
            if m.any():
                axes[1].scatter(*xy[m].T, s=24, marker=ROOM_MARKER[room], c=SERIES[i], edgecolors="white",
                                linewidths=0.6)
    handles = [Line2D([], [], ls="", marker="s", ms=8, color=SERIES[i], label=a) for i, a in enumerate(ACTIVITIES)]
    handles += [Line2D([], [], ls="", marker=ROOM_MARKER[r], ms=7, mfc="none", mec=INK, label=f"Room {r}")
                for r in "AB"]
    axes[1].legend(handles=handles, frameon=False, loc="center left", bbox_to_anchor=(1.0, 0.5))
    axes[1].set_title("(b) coloured by activity, marker = room", loc="left")
    fig.suptitle(f"Validation embeddings of the default SSL encoder ({encoder}); "
                 "UMAP fitted on development embeddings only, seed 0", fontsize=11)
    fig.savefig(out, dpi=150)
    plt.close(fig)


def loss_figure(cfg, seeds: list[str], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.4), constrained_layout=True)
    for k, run in enumerate(seeds):
        h = json.loads((Path(cfg["paths"]["encoder_dir"]) / run / "history.json").read_text())
        ep = [r["epoch"] for r in h]
        lab = k == 0
        ax.plot(ep, [r["train_loss"] for r in h], color=SERIES[0], lw=2, alpha=0.9,
                label="train (dev_fold = train)" if lab else None)
        ax.plot(ep, [r["holdout_loss"] for r in h], color=SERIES[1], lw=2, alpha=0.9,
                label="dev-holdout (label-free validation)" if lab else None)
        best = min(h, key=lambda r: r["holdout_loss"])
        ax.plot(best["epoch"], best["holdout_loss"], "o", ms=8, color=SERIES[1], mec="white", mew=2,
                label="kept checkpoint (lowest holdout loss)" if lab else None)
    ax.set_xlabel("epoch (40 steps of 256 crops)")
    ax.set_ylabel("NT-Xent loss")
    ax.set_title(f"SimCLR pre-training, default config, {len(seeds)} seeds (one line per seed)", loc="left")
    ax.legend(frameon=False)
    fig.savefig(out, dpi=150)
    plt.close(fig)


def bars_figure(cfg, out: Path) -> None:
    root = Path(cfg["paths"]["encoder_dir"]) / "probe"
    load = lambda n, f="summary.json": json.loads((root / n / f).read_text())  # noqa: E731
    methods = []
    for name, label in [("simclr_default", "SimCLR encoder (3 seeds)"), ("random_init", "random-init encoder (3 seeds)")]:
        s = load(name)["probes"]
        methods.append((label, [s[f"room_{r}"]["macro_f1"]["mean"] for r in "AB"],
                        [[s[f"room_{r}"]["macro_f1"]["std"]] * 2 for r in "AB"]))
    for name, label in [("handcrafted", "hand-crafted mean+std (1 run)"), ("pca", "PCA-128 (1 run)")]:
        p = load(name, "metrics.json")["probes"]
        vals = [p[f"room_{r}"]["macro_f1"] for r in "AB"]
        errs = [[v - p[f"room_{r}"]["ci95"][0], p[f"room_{r}"]["ci95"][1] - v] for v, r in zip(vals, "AB")]
        methods.append((label, vals, errs))
    fig, ax = plt.subplots(figsize=(8.5, 4.6), constrained_layout=True)
    width = 0.19
    centers = np.arange(2)
    for i, (label, vals, errs) in enumerate(methods):
        pos = centers + (i - 1.5) * (width + 0.02)
        e = np.array(errs).T
        ax.bar(pos, vals, width, color=SERIES[i], label=label, zorder=2)
        ax.errorbar(pos, vals, yerr=e, fmt="none", ecolor=INK, elinewidth=1.2, capsize=3, zorder=3)
        for p_, v in zip(pos, vals):
            ax.text(p_, 0.505, f"{v:.2f}", ha="center", va="bottom", fontsize=8, color="white", rotation=90)
    for r, chance in zip(range(2), (1 / 6, 1 / 5)):
        ax.text(r, 1.035, f"chance = {chance:.2f} (below axis)", ha="center", fontsize=8, color=MUTED)
    ax.set_xticks(centers, ["Room A (6 activities)", "Room B (5 activities)"])
    ax.set_ylim(0.5, 1.06)
    ax.set_ylabel("validation macro-F1 (linear probe)")
    ax.set_title("Linear probe on frozen features, per room (y axis starts at 0.5)\nerror bars: std over 3 seeds (encoders) or 95% capture bootstrap (1-run baselines)", loc="left", fontsize=10)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2)
    ax.grid(axis="x", visible=False)
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--encoder", default="default")
    ap.add_argument("--seeds", nargs="*", default=["simclr_s0", "simclr_s1", "simclr_s2"])
    args = ap.parse_args()
    cfg = load_task2_config()
    out = Path(cfg["paths"]["encoder_dir"]) / "figures"
    out.mkdir(parents=True, exist_ok=True)
    files = {"task2_umap.png": lambda p: umap_figure(cfg, args.encoder, p),
             "task2_loss_curves.png": lambda p: loss_figure(cfg, args.seeds, p),
             "task2_probe_bars.png": lambda p: bars_figure(cfg, p)}
    dest = ROOT / "reports" / "figures"
    for name, make in files.items():
        make(out / name)
        shutil.copy(out / name, dest / name)
        print(f"wrote {dest / name}")


if __name__ == "__main__":
    main()
