"""Task 3 ablation tables and the contiguous Room B control (U8, PRD FR-6, FR-9).

1. Contiguous Room B check (A -> B, default method M1): Room B validation is replaced by a random
   contiguous block of 25 files per class (2 neighbouring files dropped on each side;
   ``contiguous_split`` from scripts/task1_ablations.py). Supports are drawn from the remaining
   Room B dev captures (same seeds, same nested sampling); per-room statistics use those dev rows.
   Note: the frozen Task 2 encoder saw some of the block's captures during SSL pre-training
   (unlabelled), so this tests support/validation adjacency, not the encoder.
2. Tables from every summary under outputs/task3/runs: ablations_all.csv (every direction x
   source x method x K) and ablations_K5.csv.

Run the episodes first:
    python scripts/task3_run.py --methods M0 M1 M2 M3 M4 M5 M6 UB CHANCE SHUF --sources default
    python scripts/task3_run.py --methods M1 --sources handcrafted random_init_s0
Usage:  python scripts/task3_ablations.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from threadpoolctl import threadpool_limits  # noqa: E402

from task1_ablations import contiguous_split  # noqa: E402
from task3_run import load_feats  # noqa: E402
from xroom.config import load_task3_config  # noqa: E402
from xroom.dataset import load_windows  # noqa: E402
from xroom.fewshot import Protocol, sample_support  # noqa: E402
from xroom.task3_methods import predict  # noqa: E402
from xroom.task3_metrics import METRICS, episode_scores  # noqa: E402


def contiguous_check(cfg: dict, meta: pd.DataFrame, root: Path) -> dict:
    t3 = cfg["task3"]
    split = contiguous_split(meta)
    keep = (split != "drop").to_numpy()
    meta_c = meta.assign(split=split)[keep].reset_index(drop=True)
    feats = {k: v[keep] for k, v in load_feats(root / "emb", t3["default_source"], meta).items()}
    p = Protocol.from_config(cfg, "AtoB")
    tv = p.masks(meta_c)["target_val"]
    y = meta_c.loc[tv, "activity"].to_numpy(dtype=object)  # scoring only
    method = t3["default_method"]
    out = {"direction": "AtoB", "method": method, "n_target_val_windows": int(tv.sum()), "by_k": {}}
    official = json.loads((root / "runs" / "AtoB" / "default" / method / "summary.json").read_text(encoding="utf-8"))
    seeds = range(t3["seed_base"], t3["seed_base"] + t3["draws"])
    for k in t3["k_values"]:
        scores = [episode_scores(y, predict(method, feats, meta_c, p, sample_support(p, meta_c, k, s), cfg, seed=s), p)
                  for s in seeds]
        entry = {}
        for m in METRICS:
            v = np.array([sc[m] for sc in scores])
            off = official["by_k"][str(k)][m]["mean"]
            entry[m] = {"contiguous_mean": float(v.mean()), "contiguous_std": float(v.std()),
                        "official_mean": off, "contiguous_minus_official": float(v.mean() - off)}
        out["by_k"][str(k)] = entry
        print(f"contiguous A->B {method} K={k:<2d} exclusive {entry['exclusive']['contiguous_mean']:.3f} "
              f"(official {entry['exclusive']['official_mean']:.3f}) overall {entry['overall']['contiguous_mean']:.3f} "
              f"(official {entry['overall']['official_mean']:.3f})", flush=True)
    return out


def tables(root: Path) -> pd.DataFrame:
    rows = []
    for path in sorted((root / "runs").glob("*/*/*/summary.json")):
        s = json.loads(path.read_text(encoding="utf-8"))
        for k, v in s["by_k"].items():
            row = {"direction": s["direction"], "source": s["source"], "method": s["method"], "K": int(k)}
            for m in METRICS:
                row[f"{m}_mean"], row[f"{m}_std"] = v[m]["mean"], v[m]["std"]
                row[f"{m}_ci_low"], row[f"{m}_ci_high"] = v[m]["ci95"]
            rows.append(row)
    return pd.DataFrame(rows).sort_values(["direction", "source", "method", "K"]).reset_index(drop=True)


def main() -> None:
    cfg = load_task3_config()
    root = Path(cfg["paths"]["task3_dir"])
    _, meta = load_windows(cfg)
    with threadpool_limits(limits=1, user_api="blas"):
        contig = contiguous_check(cfg, meta, root)
    (root / "contiguous_check.json").write_text(json.dumps(contig, indent=1, sort_keys=True), encoding="utf-8")
    df = tables(root)
    df.to_csv(root / "ablations_all.csv", index=False)
    k5 = df[df["K"] == 5]
    k5.to_csv(root / "ablations_K5.csv", index=False)
    pd.set_option("display.width", 200)
    cols = ["direction", "source", "method", "overall_mean", "overall_std", "exclusive_mean", "shared_mean"]
    print("\nK = 5:")
    print(k5[cols].round(3).to_string(index=False))
    print(f"\nsaved {root / 'ablations_all.csv'}, {root / 'ablations_K5.csv'}, {root / 'contiguous_check.json'}")


if __name__ == "__main__":
    main()
