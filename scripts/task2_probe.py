"""Linear-probe evaluation of frozen encoders and baselines (Task 2, T6/T7/T9 controls).

For each feature source, runs the probe suite on the fixed 150-row windows
(per-room A and B = headline, joint 8-class, cross-room A->B and B->A on sit/stand/walk),
plus two controls: a room probe (A vs B) and a format-shift probe (activity probe trained on
clean dev windows, scored on validation windows smoothed in time and frequency as in Task 1 M8).
Writes outputs/task2/probe/<name>/metrics.json, and summary.json (mean +- std over seeds)
when several runs are aggregated.

Usage:
    python scripts/task2_probe.py --baselines-only
    python scripts/task2_probe.py --encoder simclr_s0 simclr_s1 simclr_s2 --summary simclr_default
    python scripts/task2_probe.py --baselines-only --shuffle-labels
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from task1_ablations import harmonize  # noqa: E402
from xroom.config import apply_overrides, load_task2_config  # noqa: E402
from xroom.dataset import load_sequences, load_windows  # noqa: E402
from xroom.encoder import FrozenEncoder, build_encoder, default_device, embed, load_encoder  # noqa: E402
from xroom.probe import flat_windows, handcrafted_features, linear_probe, pca_features, probe_suite  # noqa: E402
from xroom.ssl_data import fit_dev_stats  # noqa: E402

PROBE_SETS = ("room_A", "room_B", "joint", "cross_A_to_B", "cross_B_to_A")


def controls(feat_fn, f: np.ndarray, x: np.ndarray, meta: pd.DataFrame, pc: dict) -> dict:
    """Room probe on the features + format-shift probe (smoothed validation windows)."""
    split = meta["split"].to_numpy()
    dev, val = split == "dev", split == "val"
    room = linear_probe(f, meta["room"], meta, dev, val, pc["c_grid"], reps=pc["bootstrap_reps"],
                        seed=pc["bootstrap_seed"])
    val_idx = np.flatnonzero(val)
    shifted = f.copy()
    smoothed = harmonize(np.asarray(x[val_idx], np.float32), pc["smooth_width"])
    shifted[val_idx] = feat_fn(smoothed)
    shift = probe_suite(shifted, meta, pc["c_grid"], pc["bootstrap_reps"], pc["bootstrap_seed"],
                        sets=["room_A", "room_B"])
    return {"room_probe": {k: room[k] for k in ("accuracy", "macro_f1", "ci95", "confusion")},
            "format_shift": {k: (v["macro_f1"] if isinstance(v, dict) else v) for k, v in shift.items()}}


def evaluate(name: str, feat_fn, f: np.ndarray, x, meta, cfg, out_root: Path, shuffle: bool,
             with_controls: bool = True) -> dict:
    pc = cfg["probe"]
    res = {"name": name, "dim": int(f.shape[1]),
           "probes": probe_suite(f, meta, pc["c_grid"], pc["bootstrap_reps"], pc["bootstrap_seed"],
                                 shuffle_labels=shuffle)}
    if with_controls and not shuffle and feat_fn is not None:
        res["controls"] = controls(feat_fn, f, x, meta, pc)
    out = out_root / (name + ("_shuffled" if shuffle else ""))
    out.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(out / "metrics.json", "w"), indent=1)
    p = res["probes"]
    line = "  ".join(f"{k} {p[k]['macro_f1']:.3f}" for k in PROBE_SETS if k in p)
    extra = ""
    if "controls" in res:
        c = res["controls"]
        extra = (f"  | room acc {c['room_probe']['accuracy']:.3f}  shift A/B "
                 f"{c['format_shift']['room_A']:.3f}/{c['format_shift']['room_B']:.3f}")
    print(f"{out.name:28s} {line}{extra}")
    return res


def summarize(results: list[dict], name: str, out_root: Path) -> dict:
    """mean / std / min / max over runs of every probe's macro-F1 (and capture-level macro-F1)."""
    def agg(vals):
        v = np.asarray(vals, float)
        return {"mean": float(v.mean()), "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
                "min": float(v.min()), "max": float(v.max()), "values": v.tolist()}

    summ = {"runs": [r["name"] for r in results], "probes": {}}
    for k in PROBE_SETS:
        if all(k in r["probes"] for r in results):
            summ["probes"][k] = {m: agg([r["probes"][k][m] for r in results])
                                 for m in ("macro_f1", "capture_macro_f1")}
    summ["probes"]["headline_mean_room_macro_f1"] = agg([r["probes"]["headline_mean_room_macro_f1"]
                                                         for r in results])
    if all("controls" in r for r in results):
        summ["controls"] = {
            "room_probe_accuracy": agg([r["controls"]["room_probe"]["accuracy"] for r in results]),
            **{f"format_shift_{k}": agg([r["controls"]["format_shift"][k] for r in results])
               for k in ("room_A", "room_B")}}
    out = out_root / name
    out.mkdir(parents=True, exist_ok=True)
    json.dump(summ, open(out / "summary.json", "w"), indent=1)
    s = summ["probes"]
    print(f"{name} (n={len(results)}): A {s['room_A']['macro_f1']['mean']:.3f}±{s['room_A']['macro_f1']['std']:.3f}  "
          f"B {s['room_B']['macro_f1']['mean']:.3f}±{s['room_B']['macro_f1']['std']:.3f}  "
          f"joint {s['joint']['macro_f1']['mean']:.3f}  A→B {s['cross_A_to_B']['macro_f1']['mean']:.3f}  "
          f"B→A {s['cross_B_to_A']['macro_f1']['mean']:.3f}")
    return summ


def encoder_fn(fe: FrozenEncoder):
    return lambda w: embed(fe, w)


def run_baselines(cfg, x, meta, out_root: Path, shuffle: bool) -> None:
    pc = cfg["probe"]
    table, seqs = load_sequences(cfg)
    stats = fit_dev_stats(table, seqs, cfg)
    del seqs
    hc_fn = lambda w: handcrafted_features(w, stats.data_sc)  # noqa: E731
    evaluate("handcrafted", hc_fn, hc_fn(x), x, meta, cfg, out_root, shuffle)

    flat = flat_windows(x, stats, pool=pc.get("pca_time_pool", 1))
    dev = (meta["split"] == "dev").to_numpy()
    f_pca = pca_features(flat, dev, pc["pca_dim"], seed=0)
    del flat
    evaluate("pca", None, f_pca, x, meta, cfg, out_root, shuffle)

    results = []
    device = default_device()
    for s in pc["random_init_seeds"]:
        torch.manual_seed(s)
        model = build_encoder(cfg["encoder"], len(stats.data_sc)).to(device).eval()
        fe = FrozenEncoder(model, stats, device)
        results.append(evaluate(f"random_init_s{s}", encoder_fn(fe), embed(fe, x), x, meta, cfg, out_root, shuffle))
    if not shuffle:
        summarize(results, "random_init", out_root)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--encoder", nargs="*", default=[], help="SSL run names under outputs/task2/")
    ap.add_argument("--summary", default=None, help="aggregate the --encoder runs under this name")
    ap.add_argument("--baselines-only", action="store_true")
    ap.add_argument("--shuffle-labels", action="store_true", help="control: permute training labels by capture")
    ap.add_argument("--no-controls", action="store_true")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = ap.parse_args()

    cfg = apply_overrides(load_task2_config(), args.set)
    x, meta = load_windows(cfg)
    out_root = Path(cfg["paths"]["encoder_dir"]) / "probe"
    if args.baselines_only:
        run_baselines(cfg, x, meta, out_root, args.shuffle_labels)
        return
    results = []
    for run in args.encoder:
        fe = load_encoder(cfg, run)
        results.append(evaluate(run, encoder_fn(fe), embed(fe, x), x, meta, cfg, out_root,
                                args.shuffle_labels, not args.no_controls))
    if args.summary and results and not args.shuffle_labels:
        summarize(results, args.summary, out_root)


if __name__ == "__main__":
    main()
