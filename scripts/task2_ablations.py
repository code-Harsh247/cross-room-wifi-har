"""Task 2 ablations and the contiguous Room B check (T7 debugging + T9).

Part 1  ablation grid: each variant = config overrides on top of configs/task2.yaml, pre-trained
        with >= 2 seeds and probed. Two scores per run:
          * dev-holdout per-room macro-F1 (probe fit on dev-train): the ONLY score used to
            compare or choose between variants (it touches development rows only);
          * validation per-room macro-F1 + cross-room preview: reported, never used to choose.
        The "default" row reuses the main runs simclr_s0 / simclr_s1.
Part 2  contiguous Room B split (Task 1 M8 `contiguous_split`): an encoder pre-trained without
        any capture of the contiguous validation block (or its guard files), then the per-room B
        probe trained on dev rows outside the block and scored on the block.

Runs that already exist under outputs/task2/ are reused (delete a folder to retrain it).
Writes outputs/task2/ablations/results.csv (one row per run) and summary.csv (mean +- std).

Usage:  python scripts/task2_ablations.py [--only VARIANT ...] [--seeds 0 1] [--skip-contiguous]
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from task1_ablations import contiguous_split  # noqa: E402
from xroom.config import apply_overrides, load_task2_config  # noqa: E402
from xroom.dataset import load_sequences, load_windows  # noqa: E402
from xroom.encoder import embed, load_encoder  # noqa: E402
from xroom.probe import dev_holdout_scores, handcrafted_features, linear_probe, probe_suite  # noqa: E402
from xroom.ssl import train_ssl  # noqa: E402
from xroom.ssl_data import fit_dev_stats  # noqa: E402

AUG_OFF = lambda *names: [f"augment.{n}=null" for n in names]  # noqa: E731

# Priority order from the PRD (FR-9), then the T7 debugging knobs (temperature).
VARIANTS: dict[str, list[str]] = {
    "default": [],
    "aug_no_format": AUG_OFF("format"),
    "aug_no_subcarrier_mask": AUG_OFF("subcarrier_mask"),
    "aug_no_crop_resize": AUG_OFF("crop_resize"),
    "aug_crop_only": AUG_OFF("format", "noise", "gain", "subcarrier_mask", "time_mask"),
    "norm_standardize_only": ["normalization.row=false"],
    "norm_row_log": ["normalization.log=true"],
    "method_mae": ["ssl.method=mae"],
    "embed_64": ["encoder.embed_dim=64"],
    "embed_256": ["encoder.embed_dim=256"],
    "crop_150_fixed": ["ssl_data.crop_min=150"],
    "crop_64_150": ["ssl_data.crop_min=64"],
    "tau_0.1": ["ssl.temperature=0.1"],
    "tau_0.5": ["ssl.temperature=0.5"],
}


def run_name(variant: str, seed: int) -> str:
    return f"simclr_s{seed}" if variant == "default" else f"abl_{variant}_s{seed}"


def ensure_trained(cfg: dict, name: str, table, seqs, exclude=None) -> dict:
    out = Path(cfg["paths"]["encoder_dir"]) / name
    if (out / "run.json").exists():
        return json.loads((out / "run.json").read_text())
    print(f"--- training {name}")
    run = train_ssl(cfg, table, seqs, out, exclude=exclude, log=lambda m: None)
    print(f"    {run['runtime_s']:.0f}s, best epoch {run['best_epoch']}")
    gc.collect()
    return run


def part1(base: dict, variants: list[str], seeds: list[int], x, meta, table, seqs) -> list[dict]:
    pc = base["probe"]
    rows = []
    for v in variants:
        for s in seeds:
            cfg = apply_overrides(base, VARIANTS[v] + [f"ssl.seed={s}"])
            name = run_name(v, s)
            run = ensure_trained(cfg, name, table, seqs)
            f = embed(load_encoder(cfg, name), x)
            hold = dev_holdout_scores(f, meta, pc["c_grid"])
            val = probe_suite(f, meta, pc["c_grid"], reps=200, seed=pc["bootstrap_seed"],
                              sets=["room_A", "room_B", "cross_A_to_B", "cross_B_to_A"])
            hist = json.loads((Path(cfg["paths"]["encoder_dir"]) / name / "history.json").read_text())
            row = {"variant": v, "seed": s, "run": name,
                   "devholdout_A": hold["room_A"], "devholdout_B": hold["room_B"], "devholdout_mean": hold["mean"],
                   "val_A": val["room_A"]["macro_f1"], "val_B": val["room_B"]["macro_f1"],
                   "val_headline": val["headline_mean_room_macro_f1"],
                   "val_cross_A_to_B": val["cross_A_to_B"]["macro_f1"],
                   "val_cross_B_to_A": val["cross_B_to_A"]["macro_f1"],
                   "best_epoch": run["best_epoch"], "best_ssl_holdout_loss": min(h["holdout_loss"] for h in hist),
                   "eff_rank_last": hist[-1]["eff_rank"], "runtime_s": run["runtime_s"]}
            rows.append(row)
            print(f"{name:34s} dev-holdout A {row['devholdout_A']:.3f} B {row['devholdout_B']:.3f} | "
                  f"val A {row['val_A']:.3f} B {row['val_B']:.3f} A→B {row['val_cross_A_to_B']:.3f} "
                  f"B→A {row['val_cross_B_to_A']:.3f}")
    return rows


def part2(base: dict, seeds: list[int], x, meta, table, seqs) -> list[dict]:
    """Contiguous Room B block as validation; SSL never sees the block or its guard files."""
    pc = base["probe"]
    contig = contiguous_split(meta).to_numpy()
    official = meta["split"].to_numpy()
    # probe split: train on rows that are development under BOTH splits, test on the block
    split = np.where((official == "dev") & (contig == "dev"), "dev", np.where(contig == "val", "val", "drop"))
    keep_out = sorted(set(meta.loc[np.isin(contig, ["val", "drop"]), "capture_id"]))
    room_b = (meta["room"] == "B").to_numpy()
    train, test = (split == "dev") & room_b, (split == "val") & room_b
    print(f"contiguous split: {test.sum()} Room B test windows, {train.sum()} train windows; "
          f"{len(keep_out)} captures kept out of SSL")

    def score(f):
        r = linear_probe(f, meta["activity"], meta, train, test, pc["c_grid"], split=split, reps=200)
        return r["macro_f1"], r["ci95"]

    rows = []
    stats = fit_dev_stats(table, seqs, base)
    f_hc = handcrafted_features(x, stats.data_sc)
    m, ci = score(f_hc)
    rows.append({"variant": "contiguous_B: handcrafted", "seed": -1, "run": "handcrafted", "val_B": m,
                 "val_B_ci_lo": ci[0], "val_B_ci_hi": ci[1]})
    for s in seeds:
        cfg = apply_overrides(base, [f"ssl.seed={s}"])
        name = f"contig_s{s}"
        ensure_trained(cfg, name, table, seqs, exclude=set(keep_out))
        m, ci = score(embed(load_encoder(cfg, name), x))
        rows.append({"variant": "contiguous_B: SSL (block held out of SSL)", "seed": s, "run": name, "val_B": m,
                     "val_B_ci_lo": ci[0], "val_B_ci_hi": ci[1]})
        m, ci = score(embed(load_encoder(cfg, f"simclr_s{s}"), x))
        rows.append({"variant": "contiguous_B: SSL default (block seen unlabelled)", "seed": s,
                     "run": f"simclr_s{s}", "val_B": m, "val_B_ci_lo": ci[0], "val_B_ci_hi": ci[1]})
    for r in rows:
        print(f"{r['variant']:52s} seed {r['seed']:2d}  Room B macro-F1 {r['val_B']:.3f} "
              f"[{r['val_B_ci_lo']:.3f}, {r['val_B_ci_hi']:.3f}]")
    return rows


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in df.columns if c.startswith(("devholdout_", "val_")) and not c.endswith(("_lo", "_hi"))]
    g = df.groupby("variant", sort=False)[cols]
    out = g.mean().round(4).add_suffix("_mean").join(g.std(ddof=1).round(4).add_suffix("_std"))
    out.insert(0, "n_seeds", df.groupby("variant", sort=False).size())
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="*", default=None, choices=list(VARIANTS))
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1])
    ap.add_argument("--skip-contiguous", action="store_true")
    args = ap.parse_args()

    base = load_task2_config()
    x, meta = load_windows(base)
    table, seqs = load_sequences(base)
    out = Path(base["paths"]["encoder_dir"]) / "ablations"
    out.mkdir(parents=True, exist_ok=True)

    rows = part1(base, args.only or list(VARIANTS), args.seeds, x, meta, table, seqs)
    df = pd.DataFrame(rows)
    if not args.skip_contiguous:
        df = pd.concat([df, pd.DataFrame(part2(base, args.seeds, x, meta, table, seqs))], ignore_index=True)
    suffix = "" if args.only is None and not args.skip_contiguous else "_partial"
    df.to_csv(out / f"results{suffix}.csv", index=False)
    summ = summarize(df)
    summ.to_csv(out / f"summary{suffix}.csv")
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(summ)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
