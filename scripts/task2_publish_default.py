"""Publish the label-clean default encoder for Task 3 (T10).

Copies an SSL run trained with the unchanged default config (configs/task2.yaml, fixed before
any probe was run) to models/task2_default/ (``paths.default_encoder``, committed to the repo so
everyone uses the exact encoder behind the Task 2 numbers) and writes PROVENANCE.md. Within the run, the
checkpoint was chosen by the lowest SSL loss on dev-holdout crops (no labels). Refuses a run
whose saved config differs from the default config in anything but the seed.

Usage:  python scripts/task2_publish_default.py [--run simclr_s0]
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import yaml  # noqa: E402

from xroom.config import load_task2_config  # noqa: E402
from xroom.ssl import _plain, git_hash  # noqa: E402

FILES = ("encoder.pt", "norm_stats.npz", "config.yaml", "history.json", "run.json")  # no encoder_last.pt


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="simclr_s0")
    args = ap.parse_args()
    cfg = load_task2_config()
    root = Path(cfg["paths"]["encoder_dir"])
    src, dst = root / args.run, Path(cfg["paths"]["default_encoder"])
    saved = yaml.safe_load((src / "config.yaml").read_text())
    expected = _plain(cfg)
    for c in (saved, expected):  # compare what shapes training; probe settings and paths do not
        c["ssl"] = {k: v for k, v in c["ssl"].items() if k != "seed"}
        for k in ("probe", "paths"):
            c.pop(k, None)
    if saved != expected:
        raise SystemExit(f"{args.run} was not trained with the default config; refusing to publish it as default")
    dst.mkdir(parents=True, exist_ok=True)
    for f in FILES:
        shutil.copy(src / f, dst / f)
    run = json.loads((src / "run.json").read_text())
    trained_at = run["git"].removesuffix("-dirty")
    on_main = subprocess.run(["git", "merge-base", "--is-ancestor", trained_at, "HEAD"], cwd=ROOT,
                             capture_output=True).returncode == 0
    note = "" if on_main else (" This ID is not in the published history: commits made before the first push"
                               " were rewritten (author e-mail only), so the same code has a different ID;"
                               " match it by commit subject and milestone in `docs/tasks_task2.md`.")
    (dst / "PROVENANCE.md").write_text(f"""# Default Task 2 encoder (label-clean)

- Source run: `{args.run}` ({run['method']}, seed {run['seed']}), trained at git `{run['git']}`.{note}
- Published at git `{git_hash()}` by `scripts/task2_publish_default.py`.
- Settings: the default `configs/task2.yaml`, fixed before any probe was run and not changed
  afterwards. The seed was fixed in advance (0) and not chosen by any score.
- Training data: development captures of both rooms with `dev_fold == "train"`
  ({run['n_train_captures']} captures); no validation capture; no activity labels.
  Room identity was used only to balance batches 50/50.
- Normalization statistics: fitted on development rows only (`norm_stats.npz`).
- Checkpoint selection: {run['selection']}; epoch {run['best_epoch']}.
- **No activity labels were used for training or for selecting this encoder or its settings.**
  Variants compared with activity-labelled probes (`outputs/task2/abl_*`, not committed) are ablations and
  must not be used as the Task 3 default.
""", encoding="utf-8")
    print(f"published {src} -> {dst}")


if __name__ == "__main__":
    main()
