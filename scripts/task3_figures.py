"""Task 3 figures (U9, PRD FR-8). Style matches scripts/task2_figures.py.

  task3_label_efficiency.png  overall and target-exclusive macro-F1 vs K, both directions:
                              default M1 vs M0, the best ablation (M5, dashed), upper bound and
                              chance as reference lines; bands = ± std over the 10 draws
  task3_confusion_K5.png      default M1 at K = 5, both directions: confusion matrix summed over
                              the 10 draws, row-normalized, with counts
  task3_routing.png           oracle vs predicted room: target macro-F1 per K, both directions
Also writes outputs/task3/per_class_K5.csv (precision / recall / F1 per class, mean ± std over
draws) and copies the PNGs to reports/figures/.

Needs task3_run.py (M0 M1 M5 UB CHANCE) and task3_routing.py outputs.
Usage:  python scripts/task3_figures.py
"""
from __future__ import annotations

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
import pandas as pd  # noqa: E402

from xroom.config import load_task3_config  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3dd"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "text.color": INK, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6})
DIR_TITLE = {"AtoB": "A → B (target: Room B)", "BtoA": "B → A (target: Room A)"}
LINES = [("M1", "M1 default (per-room norm + prototypes)", SERIES[0], "-"),
         ("M0", "M0 no adaptation", SERIES[1], "-"),
         ("M5", "M5 logistic regression (ablation)", SERIES[2], "--")]


def summary(root: Path, d: str, m: str) -> dict:
    return json.loads((root / "runs" / d / "default" / m / "summary.json").read_text(encoding="utf-8"))["by_k"]


def label_efficiency(root: Path, ks: list[int], out: Path) -> dict:
    """Returns the plotted values (for the gate spot-check)."""
    plotted = {}
    pos = np.arange(len(ks))
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True, sharey="row")
    for col, d in enumerate(("AtoB", "BtoA")):
        for row, metric in enumerate(("overall", "exclusive")):
            ax = axes[row, col]
            for m, label, color, ls in LINES:
                s = summary(root, d, m)
                mean = np.array([s[str(k)][metric]["mean"] for k in ks])
                std = np.array([s[str(k)][metric]["std"] for k in ks])
                ax.plot(pos, mean, ls, color=color, marker="o", ms=4, lw=1.8, label=label)
                ax.fill_between(pos, mean - std, mean + std, color=color, alpha=0.15, lw=0)
                plotted[(d, m, metric)] = mean.tolist()
            for m, label, color in (("UB", "upper bound (all target labels; not valid)", INK),
                                    ("CHANCE", "chance", MUTED)):
                v = summary(root, d, m)[str(ks[0])][metric]["mean"]
                ax.axhline(v, color=color, ls=":", lw=1.2, label=label)
                plotted[(d, m, metric)] = v
            ax.set_ylim(0, 1)
            ax.set_xticks(pos, [str(k) for k in ks])
            if row == 0:
                ax.set_title(DIR_TITLE[d], fontsize=10)
            if row == 1:
                ax.set_xlabel("K (labelled support captures per room-only activity)")
            if col == 0:
                ax.set_ylabel({"overall": "overall macro-F1", "exclusive": "room-only (exclusive) macro-F1"}[metric])
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=9)
    fig.suptitle("Task 3 label efficiency (mean ± std over 10 support draws)", fontsize=11)
    fig.tight_layout(rect=(0, 0.07, 1, 0.96))
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return plotted


def confusion_and_per_class(root: Path, out_png: Path, out_csv: Path) -> pd.DataFrame:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    rows = []
    for ax, d in zip(axes, ("AtoB", "BtoA")):
        eps = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((root / "runs" / d / "default" / "M1").glob("K5_seed*.json"))]
        labels = eps[0]["scores"]["confusion"]["labels"]
        cms = np.array([e["scores"]["confusion"]["matrix"] for e in eps], dtype=float)
        total = cms.sum(0)
        norm = total / total.sum(1, keepdims=True)
        ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j, i, f"{norm[i, j]:.2f}\n({int(total[i, j])})", ha="center", va="center", fontsize=7,
                        color="white" if norm[i, j] > 0.55 else INK)
        ax.set_xticks(range(len(labels)), labels, rotation=35, ha="right")
        ax.set_yticks(range(len(labels)), labels)
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        ax.grid(False)
        ax.set_title(f"{DIR_TITLE[d]}, M1, K = 5 (10 draws summed)", fontsize=10)
        tp = np.diagonal(cms, axis1=1, axis2=2)
        prec = np.divide(tp, cms.sum(1), out=np.zeros_like(tp), where=cms.sum(1) > 0)
        rec = np.divide(tp, cms.sum(2), out=np.zeros_like(tp), where=cms.sum(2) > 0)
        f1 = np.divide(2 * tp, cms.sum(1) + cms.sum(2), out=np.zeros_like(tp), where=(cms.sum(1) + cms.sum(2)) > 0)
        shared = set(json.loads((root / "runs" / d / "default" / "M1" / "summary.json").read_text(encoding="utf-8"))
                     ["target_labels"]) & {"sit", "stand", "walk"}
        for i, c in enumerate(labels):
            rows.append({"direction": d, "class": c, "type": "shared" if c in shared else "target-only",
                         "precision_mean": prec[:, i].mean(), "precision_std": prec[:, i].std(),
                         "recall_mean": rec[:, i].mean(), "recall_std": rec[:, i].std(),
                         "f1_mean": f1[:, i].mean(), "f1_std": f1[:, i].std(), "n_val_windows_per_draw": int(total[i].sum()) // len(eps)})
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    return df


def routing(root: Path, ks: list[int], out: Path) -> dict:
    r = json.loads((root / "routing" / "summary.json").read_text(encoding="utf-8"))
    plotted = {}
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    pos = np.arange(len(ks))
    for ax, d in zip(axes, ("AtoB", "BtoA")):
        for j, (mode, color) in enumerate((("oracle", SERIES[0]), ("predicted", SERIES[1]))):
            mean = np.array([r["by_direction"][d][str(k)][mode]["target"]["mean"] for k in ks])
            std = np.array([r["by_direction"][d][str(k)][mode]["target"]["std"] for k in ks])
            ax.bar(pos + (j - 0.5) * 0.36, mean, width=0.36, yerr=std, color=color, capsize=3,
                   label=f"{mode} room" + (" (Task 1 strict router)" if mode == "predicted" else " (true room)"))
            plotted[(d, mode)] = mean.tolist()
        ax.set_xticks(pos, [str(k) for k in ks])
        ax.set_xlabel("K")
        ax.set_title(DIR_TITLE[d], fontsize=10)
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("target macro-F1 (M1)")
    acc = r["router"]["accuracy"]
    fig.suptitle(f"Oracle vs predicted room (router accuracy on pooled val: {acc:.3f})", fontsize=11)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.08, 1, 0.93))
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return plotted


def main() -> None:
    cfg = load_task3_config()
    root = Path(cfg["paths"]["task3_dir"])
    ks = cfg["task3"]["k_values"]
    fig_dir = root / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    plotted = {"label_efficiency": label_efficiency(root, ks, fig_dir / "task3_label_efficiency.png")}
    pc = confusion_and_per_class(root, fig_dir / "task3_confusion_K5.png", root / "per_class_K5.csv")
    plotted["routing"] = routing(root, ks, fig_dir / "task3_routing.png")
    (root / "figures" / "plotted_values.json").write_text(
        json.dumps({k: {"|".join(map(str, kk)): vv for kk, vv in v.items()} for k, v in plotted.items()}, indent=1),
        encoding="utf-8")
    dest = ROOT / "reports" / "figures"
    for png in sorted(fig_dir.glob("task3_*.png")):
        shutil.copy2(png, dest / png.name)
        print(f"copied {png.name} -> reports/figures/")
    pd.set_option("display.width", 200)
    print(pc[["direction", "class", "type", "precision_mean", "recall_mean", "f1_mean", "n_val_windows_per_draw"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
