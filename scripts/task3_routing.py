"""Task 3 routing: oracle vs predicted room (U7, PRD FR-7, D6).

The validation windows of BOTH rooms are pooled. Each window is routed to a room either by its
true room ("oracle") or by the Task 1 strict router ("predicted"). It is then classified by that
room's model:
  routed to the target room -> the episode's default few-shot model (M1, target label space)
  routed to the source room -> a source model (M1 features, nearest centroid over all source
                               classes from source dev labels)
Per-room standardization uses the routed room for validation windows (deployment semantics).
Scores per (direction, K, draw):
  target   macro-F1 over the target labels, on windows whose TRUE room is the target
           (equals the U6 M1 overall score under oracle routing)
  pooled   macro-F1 over all 8 activities, on every validation window of both rooms
Writes outputs/task3/routing/summary.json.

Needs: task3_support.py, task3_embed.py and the strict router (scripts/task1_router_variants.py).
Usage:  python scripts/task3_routing.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
from threadpoolctl import threadpool_limits  # noqa: E402

from task3_run import load_feats  # noqa: E402
from xroom.config import load_task3_config  # noqa: E402
from xroom.dataset import load_windows  # noqa: E402
from xroom.fewshot import Protocol, load_support, manifest_path  # noqa: E402
from xroom.probe import per_class_f1  # noqa: E402
from xroom.router import load_router, route  # noqa: E402
from xroom.task3_methods import predict, predict_source  # noqa: E402

MODES = ("oracle", "predicted")


def routed_predictions(feats, meta, val_idx, rooms, protocol, support, cfg, seed) -> np.ndarray:
    """Predicted activity for every validation row (in ``val_idx`` order) under one routing."""
    meta_r = meta.copy()
    meta_r.loc[val_idx, "room"] = rooms
    m = protocol.masks(meta_r)
    pred = {}
    pred.update(zip(meta_r.index[m["target_val"]], predict("M1", feats, meta_r, protocol, support, cfg, seed=seed)))
    if m["source_val"].any():
        pred.update(zip(meta_r.index[m["source_val"]], predict_source(feats, meta_r, protocol, cfg)))
    return np.array([pred[i] for i in val_idx], dtype=object)


def macro(y, p, labels) -> float:
    return float(np.mean(list(per_class_f1(y, p, labels).values())))


def main() -> None:
    cfg = load_task3_config()
    t3 = cfg["task3"]
    root = Path(cfg["paths"]["task3_dir"])
    x, meta = load_windows(cfg)
    val_idx = meta.index[(meta["split"] == "val").to_numpy()]
    vmeta = meta.loc[val_idx]
    true_room = vmeta["room"].to_numpy()
    y = vmeta["activity"].to_numpy(dtype=object)  # scoring only
    all_labels = sorted(set(a for acts in t3["activities"].values() for a in acts))

    t0 = time.time()
    router = load_router(cfg, "strict")
    routes = {"oracle": route(None, vmeta, mode="oracle"),
              "predicted": route(np.asarray(x[val_idx.to_numpy()]), vmeta, mode="predicted", router=router)}
    mis = np.where(routes["predicted"] != true_room)[0]
    summary = {"router": {"kind": "strict", "n_val_windows": int(len(val_idx)),
                          "accuracy": float((routes["predicted"] == true_room).mean()),
                          "misrouted": vmeta.iloc[mis][["capture_id", "room", "activity"]].to_dict("records")},
               "by_direction": {}}
    print(f"strict router on pooled val: accuracy {summary['router']['accuracy']:.4f} "
          f"({len(mis)} misrouted of {len(val_idx)})")

    seeds = range(t3["seed_base"], t3["seed_base"] + t3["draws"])
    with threadpool_limits(limits=1, user_api="blas"):
        for direction in t3["directions"]:
            p = Protocol.from_config(cfg, direction)
            feats = load_feats(root / "emb", t3["default_source"], meta)
            tgt = true_room == p.target
            by_k = {}
            for k in t3["k_values"]:
                vals = {mode: {"target": [], "pooled": []} for mode in MODES}
                for seed in seeds:
                    support = load_support(manifest_path(cfg, direction, seed), k)
                    for mode in MODES:
                        pred = routed_predictions(feats, meta, val_idx, routes[mode], p, support, cfg, seed)
                        vals[mode]["target"].append(macro(y[tgt], pred[tgt], p.target_labels))
                        vals[mode]["pooled"].append(macro(y, pred, all_labels))
                entry = {mode: {s: {"mean": float(np.mean(v)), "std": float(np.std(v)), "draws": v}
                                for s, v in vals[mode].items()} for mode in MODES}
                entry["predicted_minus_oracle"] = {
                    s: float(np.mean(vals["predicted"][s]) - np.mean(vals["oracle"][s])) for s in ("target", "pooled")}
                by_k[str(k)] = entry
                print(f"{direction} K={k:<2d} target oracle {entry['oracle']['target']['mean']:.3f} "
                      f"predicted {entry['predicted']['target']['mean']:.3f} | pooled oracle "
                      f"{entry['oracle']['pooled']['mean']:.3f} predicted {entry['predicted']['pooled']['mean']:.3f}",
                      flush=True)
            summary["by_direction"][direction] = by_k
    out = root / "routing" / "summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=1, sort_keys=True), encoding="utf-8")
    print(f"saved {out} ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
