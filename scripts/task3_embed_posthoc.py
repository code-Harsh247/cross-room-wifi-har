"""Feature caches for post-hoc batch 2 (docs/ideas.md ideas 6 and 7; configs/task3_posthoc.yaml).

Writes outputs/task3/emb/<name>.npy, (4680, 128) float32, rows aligned with windows_L150_meta.csv:
  slice25, slice50   default encoder on contiguous 25- / 50-row slices of each 150-row window,
                     slice embeddings averaged per window (25 rows is below the encoder's
                     documented 32-row minimum; it runs and gives finite embeddings)
  mae_s<seed>        label-free MAE encoders outputs/task2/t3_mae_s<seed> for posthoc.mae_seeds (train first with
                     scripts/task2_pretrain.py --run t3_mae_s<seed> --seed <seed> --method mae);
                     skipped with a message if the run does not exist
The concatenated source ``default+random_init_s0`` is built on the fly by scripts/task3_run.py.

Usage:  python scripts/task3_embed_posthoc.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from xroom.config import load_config  # noqa: E402
from xroom.dataset import load_windows  # noqa: E402
from xroom.encoder import embed, load_encoder  # noqa: E402


def slice_embeddings(enc, x: np.ndarray, rows: int, chunk: int = 512) -> np.ndarray:
    """Mean of the embeddings of the contiguous ``rows``-row slices of every window."""
    t = x.shape[1]
    k = t // rows
    out = []
    for i in range(0, len(x), chunk):
        xb = np.asarray(x[i : i + chunk])
        parts = [embed(enc, xb[:, j * rows : (j + 1) * rows]) for j in range(k)]
        out.append(np.mean(parts, axis=0))
    return np.concatenate(out).astype(np.float32)


def main() -> None:
    cfg = load_config(ROOT / "configs" / "task3_posthoc.yaml")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    out = Path(cfg["paths"]["task3_dir"]) / "emb"
    out.mkdir(parents=True, exist_ok=True)
    x, _ = load_windows(cfg)
    info_path = out / "manifest_posthoc.json"
    info = json.loads(info_path.read_text(encoding="utf-8")) if info_path.exists() else {}
    enc = load_encoder(cfg)
    for rows in cfg["posthoc"]["slices"]:
        f = slice_embeddings(enc, x, rows)
        assert np.isfinite(f).all()
        np.save(out / f"slice{rows}.npy", f)
        info[f"slice{rows}"] = {"shape": list(f.shape), "slice_rows": rows}
        print(f"  slice{rows:<3d} {f.shape}", flush=True)
    for seed in cfg["posthoc"].get("mae_seeds", [cfg["posthoc"]["mae"]["seed"]]):
        run = f"t3_mae_s{seed}"
        run_dir = Path(cfg["paths"]["encoder_dir"]) / run
        if (run_dir / "run.json").exists():  # run.json is written only when training has finished
            f = embed(load_encoder(cfg, run=run), x)
            assert np.isfinite(f).all()
            np.save(out / f"mae_s{seed}.npy", f)
            info[f"mae_s{seed}"] = {"shape": list(f.shape), "run": run}
            print(f"  mae_s{seed}    {f.shape} (from {run_dir})", flush=True)
        else:
            print(f"  mae_s{seed}    skipped: {run_dir} is missing or still training")
    info_path.write_text(json.dumps(info, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
