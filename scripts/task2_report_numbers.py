"""Collect every Task 2 number the report cites, from the saved outputs (T11).

Reads outputs/task2/probe/*/{summary,metrics}.json, outputs/task2/ablations/{results,summary}.csv
and the run histories, and prints markdown tables. The report section copies its numbers from
this output (saved to outputs/task2/report_numbers.md); nothing is typed in by hand.

Usage:  python scripts/task2_report_numbers.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from xroom.config import load_task2_config  # noqa: E402

SETS = [("room_A", "Room A (6 cl.)"), ("room_B", "Room B (5 cl.)"), ("joint", "Joint (8 cl.)"),
        ("cross_A_to_B", "Cross A→B (3 cl.)"), ("cross_B_to_A", "Cross B→A (3 cl.)")]


def fmt(m, s=None) -> str:
    return f"{m:.3f}" if s is None else f"{m:.3f} ± {s:.3f}"


def main() -> None:
    cfg = load_task2_config()
    root = Path(cfg["paths"]["encoder_dir"])
    probe = root / "probe"
    lines: list[str] = []
    out = lines.append

    out("## Main probe results (validation macro-F1)\n")
    out("| Features | " + " | ".join(n for _, n in SETS) + " | Headline (mean of rooms) |")
    out("|---|" + "---|" * (len(SETS) + 1))
    for name, label in [("simclr_default", "SimCLR (3 seeds)"), ("random_init", "Random-init encoder (3 seeds)")]:
        s = json.loads((probe / name / "summary.json").read_text())["probes"]
        cells = [fmt(s[k]["macro_f1"]["mean"], s[k]["macro_f1"]["std"]) for k, _ in SETS]
        h = s["headline_mean_room_macro_f1"]
        out(f"| {label} | " + " | ".join(cells) + f" | {fmt(h['mean'], h['std'])} |")
    for name, label in [("handcrafted", "Hand-crafted mean+std (484-d)"), ("pca", "PCA-128")]:
        p = json.loads((probe / name / "metrics.json").read_text())["probes"]
        cells = [f"{p[k]['macro_f1']:.3f} [{p[k]['ci95'][0]:.2f}, {p[k]['ci95'][1]:.2f}]" for k, _ in SETS]
        out(f"| {label} | " + " | ".join(cells) + f" | {p['headline_mean_room_macro_f1']:.3f} |")

    out("\n## SimCLR per seed (validation macro-F1 [95% capture bootstrap])\n")
    out("| Run | Room A | Room B | Joint | Capture-level A / B | Cross A→B / B→A | Room probe acc. | Format shift A / B |")
    out("|---|---|---|---|---|---|---|---|")
    for run in ("simclr_s0", "simclr_s1", "simclr_s2"):
        m = json.loads((probe / run / "metrics.json").read_text())
        p, c = m["probes"], m["controls"]
        ci = lambda k: f"{p[k]['macro_f1']:.3f} [{p[k]['ci95'][0]:.2f}, {p[k]['ci95'][1]:.2f}]"  # noqa: E731
        out(f"| {run} | {ci('room_A')} | {ci('room_B')} | {ci('joint')} | "
            f"{p['room_A']['capture_macro_f1']:.3f} / {p['room_B']['capture_macro_f1']:.3f} | "
            f"{p['cross_A_to_B']['macro_f1']:.3f} / {p['cross_B_to_A']['macro_f1']:.3f} | "
            f"{c['room_probe']['accuracy']:.3f} | {c['format_shift']['room_A']:.3f} / {c['format_shift']['room_B']:.3f} |")

    out("\n## Dev-holdout per-room macro-F1 per run (probe fit on dev-train; development data only)\n")
    for name in ("simclr_s0", "simclr_s1", "simclr_s2", "random_init_s0", "random_init_s1", "random_init_s2",
                 "handcrafted", "pca"):
        d = json.loads((probe / name / "metrics.json").read_text()).get("dev_holdout")
        if d:
            out(f"- {name}: A {d['room_A']:.3f}, B {d['room_B']:.3f}")

    out("\n## Per-class F1 and Room A confusion, SimCLR seed 0 (validation)\n")
    p = json.loads((probe / "simclr_s0" / "metrics.json").read_text())["probes"]
    for k in ("room_A", "room_B"):
        out(f"- {k}: " + ", ".join(f"{a} {v:.2f}" for a, v in p[k]["per_class_f1"].items()))
    conf = p["room_A"]["confusion"]
    out(f"- room_A confusion, labels {conf['labels']} (rows true, cols predicted): {conf['matrix(rows=true, cols=pred)']}")

    out("\n## Controls: room probe and format shift (validation)\n")
    out("| Features | Room probe acc. | Activity A clean → shifted | Activity B clean → shifted |")
    out("|---|---|---|---|")
    s = json.loads((probe / "simclr_default" / "summary.json").read_text())
    sp, sc = s["probes"], s["controls"]
    out(f"| SimCLR (3 seeds) | {fmt(sc['room_probe_accuracy']['mean'], sc['room_probe_accuracy']['std'])} | "
        f"{sp['room_A']['macro_f1']['mean']:.3f} → {sc['format_shift_room_A']['mean']:.3f} | "
        f"{sp['room_B']['macro_f1']['mean']:.3f} → {sc['format_shift_room_B']['mean']:.3f} |")
    r = json.loads((probe / "random_init" / "summary.json").read_text())
    rp, rc = r["probes"], r["controls"]
    out(f"| Random-init (3 seeds) | {fmt(rc['room_probe_accuracy']['mean'], rc['room_probe_accuracy']['std'])} | "
        f"{rp['room_A']['macro_f1']['mean']:.3f} → {rc['format_shift_room_A']['mean']:.3f} | "
        f"{rp['room_B']['macro_f1']['mean']:.3f} → {rc['format_shift_room_B']['mean']:.3f} |")
    m = json.loads((probe / "handcrafted" / "metrics.json").read_text())
    out(f"| Hand-crafted | {m['controls']['room_probe']['accuracy']:.3f} | "
        f"{m['probes']['room_A']['macro_f1']:.3f} → {m['controls']['format_shift']['room_A']:.3f} | "
        f"{m['probes']['room_B']['macro_f1']:.3f} → {m['controls']['format_shift']['room_B']:.3f} |")

    out("\n## Shuffled-label control (validation macro-F1)\n")
    for name in ("handcrafted", "pca", "random_init_s0", "random_init_s1", "random_init_s2"):
        f = probe / f"{name}_shuffled" / "metrics.json"
        if f.exists():
            p = json.loads(f.read_text())["probes"]
            out(f"- {name}: A {p['room_A']['macro_f1']:.3f}, B {p['room_B']['macro_f1']:.3f}, "
                f"joint {p['joint']['macro_f1']:.3f}")

    out("\n## SSL training (default config)\n")
    for run in ("simclr_s0", "simclr_s1", "simclr_s2"):
        h = json.loads((root / run / "history.json").read_text())
        info = json.loads((root / run / "run.json").read_text())
        best = min(h, key=lambda x: x["holdout_loss"])
        out(f"- {run}: train loss {h[0]['train_loss']:.3f} → {h[-1]['train_loss']:.3f}; holdout "
            f"{h[0]['holdout_loss']:.3f} → best {best['holdout_loss']:.3f} (epoch {best['epoch']}); "
            f"eff. rank {h[-1]['eff_rank']:.1f}; emb std {h[-1]['emb_std']:.3f}; {info['runtime_s']:.0f} s; "
            f"{info['n_train_captures']} train / {info['n_holdout_captures']} holdout captures")

    abl = root / "ablations" / "summary.csv"
    if abl.exists():
        out("\n## Ablations (mean ± std over seeds; dev-holdout = selection score, val = reported)\n")
        d = pd.read_csv(abl, index_col="variant")
        out("| Variant | n | Dev-holdout A | Dev-holdout B | Val A | Val B | Val cross A→B | Val cross B→A |")
        out("|---|---|---|---|---|---|---|---|")
        for v, row in d.iterrows():
            if str(v).startswith("contiguous"):
                continue
            c = lambda k: fmt(row[f"{k}_mean"], row[f"{k}_std"])  # noqa: E731
            out(f"| {v} | {int(row['n_seeds'])} | {c('devholdout_A')} | {c('devholdout_B')} | {c('val_A')} | "
                f"{c('val_B')} | {c('val_cross_A_to_B')} | {c('val_cross_B_to_A')} |")
        res = pd.read_csv(root / "ablations" / "results.csv")
        partial = root / "ablations" / "results_partial.csv"
        allruns = pd.concat([res, pd.read_csv(partial)]) if partial.exists() else res
        out("\n## Top variants with 3 seeds (dev-holdout chooses; validation reports)\n")
        out("| Variant | Seeds | Dev-holdout A | Dev-holdout B | Dev-holdout mean | Val A | Val B | "
            "Val cross A→B / B→A | Format shift A / B | Room probe |")
        out("|---|---|---|---|---|---|---|---|---|---|")
        for v, summ_name in [("default", "simclr_default"), ("tau_0.5", "tau_0.5"), ("method_mae", "method_mae"),
                             ("aug_no_format", "aug_no_format")]:
            f = probe / summ_name / "summary.json"
            if not f.exists():
                continue
            r = allruns[allruns["variant"] == v]
            s = json.loads(f.read_text())
            sp, sc = s["probes"], s["controls"]
            g = lambda k: fmt(sp[k]["macro_f1"]["mean"], sp[k]["macro_f1"]["std"])  # noqa: E731
            out(f"| {v} | {len(s['runs'])} (dev-holdout: {len(r)}) | {r['devholdout_A'].mean():.3f} | "
                f"{r['devholdout_B'].mean():.3f} | {r['devholdout_mean'].mean():.4f} | {g('room_A')} | {g('room_B')} | "
                f"{sp['cross_A_to_B']['macro_f1']['mean']:.3f} / {sp['cross_B_to_A']['macro_f1']['mean']:.3f} | "
                f"{sc['format_shift_room_A']['mean']:.3f} / {sc['format_shift_room_B']['mean']:.3f} | "
                f"{sc['room_probe_accuracy']['mean']:.3f} |")
            vals = sp["room_A"]["macro_f1"]["values"]
            out(f"|  | val A per seed: {', '.join(f'{x:.3f}' for x in vals)} | | | | | | | | |")
        con = res[res["variant"].str.startswith("contiguous")]
        out("\n## Contiguous Room B split (Room B macro-F1 [95% capture bootstrap])\n")
        for _, row in con.iterrows():
            out(f"- {row['variant']} (run {row['run']}): {row['val_B']:.3f} "
                f"[{row['val_B_ci_lo']:.2f}, {row['val_B_ci_hi']:.2f}]")

    text = "\n".join(lines)
    print(text)
    (root / "report_numbers.md").write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
