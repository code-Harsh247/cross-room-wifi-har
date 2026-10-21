"""Run Task 3 episodes (U5/U6/U8, PRD FR-5).

For every direction x feature source x method x K x draw seed: load the saved support manifest,
predict the target validation windows (labels only through the access guard), score them,
and write outputs/task3/runs/<direction>/<source>/<method>/K<k>_seed<r>.json. Then write
summary.json per (direction, source, method): mean ± std over draws and a 95% capture
bootstrap interval of the mean, for overall / exclusive / shared macro-F1.

Needs: scripts/task3_support.py (manifests) and scripts/task3_embed.py (feature caches).

Usage:
    python scripts/task3_run.py --methods M0 M1                       # all K, all draws, both directions
    python scripts/task3_run.py --methods M0 M1 UB CHANCE --sources default
    python scripts/task3_run.py --methods M0 M1 --k 5 --draws 2       # smoke run
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
from threadpoolctl import threadpool_limits  # noqa: E402

from xroom.config import load_task3_config  # noqa: E402
from xroom.dataset import load_windows  # noqa: E402
from xroom.fewshot import Protocol, load_support, manifest_path  # noqa: E402
from xroom.task3_methods import METHODS, REFERENCES, predict  # noqa: E402
from xroom.task3_metrics import aggregate, episode_scores  # noqa: E402


def dump(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, sort_keys=True), encoding="utf-8")


def load_feats(emb_dir: Path, source: str, meta) -> dict[str, np.ndarray]:
    path = emb_dir / f"{source}.npy"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; run scripts/task3_embed.py")
    feats = {"base": np.load(path)}
    if source == "default":
        rooms = meta["room"].to_numpy()[:, None]
        feats["adabn"] = np.where(rooms == "A", np.load(emb_dir / "adabn_A.npy"), np.load(emb_dir / "adabn_B.npy"))
    return feats


def main() -> None:
    cfg = load_task3_config()
    t3 = cfg["task3"]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--methods", nargs="+", default=[t3["default_method"]], choices=list(METHODS + REFERENCES))
    ap.add_argument("--sources", nargs="+", default=[t3["default_source"]])
    ap.add_argument("--directions", nargs="+", default=t3["directions"])
    ap.add_argument("--k", nargs="+", type=int, default=t3["k_values"])
    ap.add_argument("--draws", type=int, default=t3["draws"])
    args = ap.parse_args()

    root = Path(cfg["paths"]["task3_dir"])
    _, meta = load_windows(cfg)
    seeds = list(range(t3["seed_base"], t3["seed_base"] + args.draws))
    t0 = time.time()
    with threadpool_limits(limits=1, user_api="blas"):
        for direction in args.directions:
            p = Protocol.from_config(cfg, direction)
            tv = p.masks(meta)["target_val"]
            y_true = meta.loc[tv, "activity"].to_numpy(dtype=object)  # scoring only
            captures = meta.loc[tv, "capture_id"].to_numpy()
            for source in args.sources:
                feats = load_feats(root / "emb", source, meta)
                for method in args.methods:
                    if method == "M3" and source != "default":
                        continue  # AdaBN caches exist for the default encoder only
                    summary = {"direction": direction, "source": source, "method": method,
                               "target_labels": list(p.target_labels), "by_k": {}}
                    for k in args.k:
                        episodes = []
                        for seed in seeds:
                            support = load_support(manifest_path(cfg, direction, seed), k)
                            pred = predict(method, feats, meta, p, support, cfg, seed=seed)
                            ep = {"direction": direction, "source": source, "method": method, "k": k,
                                  "seed": seed, "n_support_captures": len(support),
                                  "scores": episode_scores(y_true, pred, p), "predictions": pred.tolist()}
                            dump(root / "runs" / direction / source / method / f"K{k}_seed{seed}.json", ep)
                            episodes.append(ep)
                        summary["by_k"][str(k)] = aggregate(episodes, y_true, captures, p,
                                                            reps=t3["bootstrap_reps"], seed=0)
                        s = summary["by_k"][str(k)]
                        print(f"{direction} {source:15s} {method:6s} K={k:<2d} overall {s['overall']['mean']:.3f}"
                              f"+/-{s['overall']['std']:.3f}  excl {s['exclusive']['mean']:.3f}"
                              f"+/-{s['exclusive']['std']:.3f}  shared {s['shared']['mean']:.3f}"
                              f"+/-{s['shared']['std']:.3f}", flush=True)
                    dump(root / "runs" / direction / source / method / "summary.json", summary)
    print(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
