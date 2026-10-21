"""Cache Task 3 feature arrays for every 150-row window (U3, PRD FR-3).

Writes outputs/task3/emb/<name>.npy, each (4680, d) float32, rows aligned with
windows_L150_meta.csv:
  default.npy          Task 2 label-clean default encoder (models/task2_default)
  handcrafted.npy      per-row normalized mean + std over time (484-d)
  random_init_s0.npy   untrained encoder (seed 0), built as in scripts/task2_probe.py
  adabn_A.npy          default encoder with BatchNorm stats re-estimated on Room A dev windows
  adabn_B.npy          ... on Room B dev windows
  inputnorm_A.npy      default encoder with its input normalization re-fitted on Room A dev windows (M6)
  inputnorm_B.npy      ... on Room B dev windows
M3 uses adabn_<room>.npy rows for that room's windows. Also writes emb/manifest.json.

Usage:  python scripts/task3_embed.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from xroom.adapt import adabn, room_input_stats  # noqa: E402
from xroom.config import load_task3_config  # noqa: E402
from xroom.dataset import load_sequences, load_windows  # noqa: E402
from xroom.encoder import FrozenEncoder, build_encoder, default_device, embed, load_encoder  # noqa: E402
from xroom.probe import handcrafted_features  # noqa: E402
from xroom.ssl_data import fit_dev_stats  # noqa: E402


def save(out: Path, name: str, f: np.ndarray, info: dict) -> None:
    f = np.ascontiguousarray(f, dtype=np.float32)
    if not np.isfinite(f).all():
        raise ValueError(f"{name}: non-finite values")
    np.save(out / f"{name}.npy", f)
    info[name] = {"shape": list(f.shape)}
    print(f"  {name:16s} {f.shape}", flush=True)


def main() -> None:
    cfg = load_task3_config()
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    out = Path(cfg["paths"]["task3_dir"]) / "emb"
    out.mkdir(parents=True, exist_ok=True)
    x, meta = load_windows(cfg)
    info: dict = {"rows": len(meta), "meta": "data_cache/windows_L150_meta.csv"}
    t0 = time.time()

    enc = load_encoder(cfg)  # label-clean default (committed)
    info["default_encoder"] = str(enc.run_dir.relative_to(ROOT)) if enc.run_dir else None
    save(out, "default", embed(enc, x), info)

    table, seqs = load_sequences(cfg)
    stats = fit_dev_stats(table, seqs, cfg)  # same dev-only stats as the Task 2 baselines
    del seqs
    save(out, "handcrafted", handcrafted_features(x, stats.data_sc), info)

    torch.manual_seed(0)
    model = build_encoder(cfg["encoder"], len(stats.data_sc)).to(default_device()).eval()
    save(out, "random_init_s0", embed(FrozenEncoder(model, stats, default_device()), x), info)

    batch = cfg["task3"]["adabn_batch"]
    for room in ("A", "B"):
        save(out, f"adabn_{room}", embed(adabn(enc, x, meta, room, batch), x), info)

    for room in ("A", "B"):  # M6: input normalization re-fitted per room on its dev windows
        fe = FrozenEncoder(enc.model, room_input_stats(enc.stats, x, meta, room), enc.device, enc.run_dir)
        save(out, f"inputnorm_{room}", embed(fe, x), info)

    info["seconds"] = round(time.time() - t0, 1)
    (out / "manifest.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(f"wrote {out} in {info['seconds']} s")


if __name__ == "__main__":
    main()
