"""Task 3 diagnostics: why shared-class transfer is low (U6 gate check 4).

For each direction, on the default encoder's embeddings:
  (a) prototype sanity: source prototypes (all source classes, source dev labels) classify the
      SOURCE room's validation windows (expected well above chance; otherwise labels are broken)
  (b) like-for-like with the Task 2 cross-room preview: 3-class sit/stand/walk only, source
      prototypes -> target validation windows of those classes (chance 1/3)
  (c) failure mode in the real episodes (K = 5, all draws): the share of target sit/stand/walk
      validation windows predicted as a target-only class
(a) and (b) use M0 (raw) and M1 (per-room standardized) features. Validation labels are used
for scoring only. Writes outputs/task3/diagnostics.json.

Needs scripts/task3_embed.py and scripts/task3_run.py --methods M0 M1 (for (c)).
Usage:  python scripts/task3_diagnostics.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from xroom.adapt import per_room_standardize  # noqa: E402
from xroom.config import load_task3_config  # noqa: E402
from xroom.dataset import load_windows  # noqa: E402
from xroom.fewshot import Protocol, nearest_centroid, prototypes  # noqa: E402
from xroom.probe import macro_f1  # noqa: E402


def main() -> None:
    cfg = load_task3_config()
    root = Path(cfg["paths"]["task3_dir"])
    _, meta = load_windows(cfg)
    f0 = np.load(root / "emb" / "default.npy")
    rooms, dev = meta["room"].to_numpy(), (meta["split"] == "dev").to_numpy()
    feats = {"M0": f0, "M1": per_room_standardize(f0, rooms, dev)}
    act = meta["activity"].to_numpy(dtype=object)
    out = {}
    for d in cfg["task3"]["directions"]:
        p = Protocol.from_config(cfg, d)
        m = p.masks(meta)
        res = {}
        for name, f in feats.items():
            cl, protos = prototypes(f, np.where(m["source_dev"], act, None), classes=p.source_labels)
            sv = m["source_val"]
            a = macro_f1(act[sv], nearest_centroid(f[sv], cl, protos), p.source_labels)
            cl3, p3 = prototypes(f, np.where(m["source_dev"] & np.isin(act, p.shared), act, None), classes=p.shared)
            tvs = m["target_val"] & np.isin(act, p.shared)
            b = macro_f1(act[tvs], nearest_centroid(f[tvs], cl3, p3), p.shared)
            res[name] = {"source_val_prototype_macro_f1": a, "shared_only_3class_transfer_macro_f1": b}
        y = act[m["target_val"]]
        shared_rows = np.isin(y, p.shared)
        for name in feats:
            frac = []
            for ep_path in sorted((root / "runs" / d / "default" / name).glob("K5_seed*.json")):
                pred = np.array(json.loads(ep_path.read_text(encoding="utf-8"))["predictions"], dtype=object)
                frac.append(float(np.isin(pred[shared_rows], p.target_only).mean()))
            res[name]["K5_share_of_shared_windows_predicted_target_only"] = float(np.mean(frac)) if frac else None
        out[d] = res
        for name, r in res.items():
            print(f"{d} {name}: (a) source-val prototypes {r['source_val_prototype_macro_f1']:.3f} | "
                  f"(b) 3-class shared transfer {r['shared_only_3class_transfer_macro_f1']:.3f} (chance 0.333) | "
                  f"(c) K=5 shared windows -> target-only {r['K5_share_of_shared_windows_predicted_target_only']:.2f}")
    (root / "diagnostics.json").write_text(json.dumps(out, indent=1, sort_keys=True), encoding="utf-8")
    print(f"saved {root / 'diagnostics.json'}")


if __name__ == "__main__":
    main()
