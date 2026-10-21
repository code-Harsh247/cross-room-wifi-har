"""Publish the label-free MAE encoders behind the best post-hoc Task 3 system (docs/ideas.md batch 4).

Copies outputs/task2/t3_mae_s<seed>/ (seeds from configs/task3_posthoc.yaml: mae_seeds +
mae_seeds_batch4) to models/task3_mae_s<seed>/ with the same files as models/task2_default
(encoder.pt, norm_stats.npz, config.yaml, history.json, run.json) and writes PROVENANCE.md.
Refuses a run whose training settings differ from the Task 2 defaults in anything but
ssl.method = mae and the seed. Committing these (~3.3 MB each) is the documented exception in
CLAUDE.md, so the 5-seed ensemble (ens_mae5, M11) can be rebuilt exactly on any machine.

Usage:  python scripts/task3_publish_mae.py
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import yaml  # noqa: E402

from xroom.config import load_config, load_task2_config  # noqa: E402
from xroom.ssl import _plain, git_hash  # noqa: E402

FILES = ("encoder.pt", "norm_stats.npz", "config.yaml", "history.json", "run.json")


def training_settings(cfg: dict) -> dict:
    c = {k: v for k, v in cfg.items() if k not in ("probe", "paths")}
    c["ssl"] = {k: v for k, v in c["ssl"].items() if k not in ("seed", "method")}
    return c


def main() -> None:
    cfg = load_config(ROOT / "configs" / "task3_posthoc.yaml")
    ph = cfg["posthoc"]
    seeds = list(ph["mae_seeds"]) + list(ph.get("mae_seeds_batch4", []))
    expected = training_settings(_plain(load_task2_config()))
    src_root = Path(cfg["paths"]["encoder_dir"])
    for seed in seeds:
        src = src_root / f"t3_mae_s{seed}"
        dst = ROOT / "models" / f"task3_mae_s{seed}"
        saved = yaml.safe_load((src / "config.yaml").read_text(encoding="utf-8"))
        run = json.loads((src / "run.json").read_text(encoding="utf-8"))
        if saved["ssl"]["method"] != "mae" or run["seed"] != seed:
            raise SystemExit(f"{src}: not an MAE run with seed {seed}")
        if training_settings(saved) != expected:
            raise SystemExit(f"{src}: trained with non-default Task 2 settings; refusing to publish")
        dst.mkdir(parents=True, exist_ok=True)
        for f in FILES:
            shutil.copy(src / f, dst / f)
        (dst / "PROVENANCE.md").write_text(f"""# Task 3 post-hoc MAE encoder, seed {seed} (label-free)

- Source run: `t3_mae_s{seed}` (mae, seed {seed}), trained at git `{run['git']}` with
  `python scripts/task2_pretrain.py --run t3_mae_s{seed} --seed {seed} --method mae`
  ("-dirty" = uncommitted config edits at the time; the training code was committed).
- Published at git `{git_hash()}` by `scripts/task3_publish_mae.py`.
- Settings: the Task 2 defaults (`configs/task2.yaml`) with `ssl.method = mae`; the seed list was
  fixed in `configs/task3_posthoc.yaml` before training, and no seed was chosen by a score.
- Training data: development captures of both rooms with `dev_fold == "train"`
  ({run['n_train_captures']} captures); no validation capture; no activity labels.
- Normalization statistics: fitted on development rows only (`norm_stats.npz`).
- Checkpoint selection: {run['selection']}; epoch {run['best_epoch']}.
- Use: one member of the post-hoc ensemble `ens_mae5` (M11). Not the Task 3 default, which stays
  the pre-registered M1 on `models/task2_default`.
""", encoding="utf-8")
        print(f"published {src.name} -> {dst.relative_to(ROOT)} (epoch {run['best_epoch']})")


if __name__ == "__main__":
    main()
