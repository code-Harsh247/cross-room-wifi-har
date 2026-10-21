"""Draw and save the Task 3 support sets (U2, PRD FR-2).

For each direction and draw seed, K support captures per target-only activity are drawn from
the target room's development captures. Supports are nested across K for a fixed seed.
Writes outputs/task3/support/<direction>_seed<r>.json and prints a coverage summary.

Usage:  python scripts/task3_support.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from xroom.config import load_task3_config  # noqa: E402
from xroom.dataset import load_windows  # noqa: E402
from xroom.fewshot import Protocol, manifest_path, support_pool, write_support_manifest  # noqa: E402


def main() -> None:
    cfg = load_task3_config()
    t3 = cfg["task3"]
    _, meta = load_windows(cfg)
    seeds = range(t3["seed_base"], t3["seed_base"] + t3["draws"])
    rows = []
    for direction in t3["directions"]:
        p = Protocol.from_config(cfg, direction)
        pool = support_pool(p, meta)
        used = {act: set() for act in p.target_only}
        for seed in seeds:
            path = manifest_path(cfg, direction, seed)
            manifest = write_support_manifest(p, meta, t3["k_values"], seed, path)
            for act, ids in manifest["support"].items():
                used[act] |= set(ids)
        kmax = max(t3["k_values"])
        for act in p.target_only:
            rows.append({"direction": direction, "class": act, "pool_captures": len(pool[act]),
                         f"distinct_used_K{kmax}_{len(seeds)}draws": len(used[act])})
        print(f"{direction}: wrote {len(seeds)} manifests to {manifest_path(cfg, direction, 0).parent}")
    print()
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
