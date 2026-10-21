"""Copy the git-ignored outputs/ into the committed results/ folder, without model checkpoints.

outputs/ holds everything the scripts write (metrics, tables, histories, figures, checkpoints).
results/ is its committed snapshot so the saved numbers behind every report can be inspected
without rerunning anything. Checkpoints (*.pt) are left out; the one encoder Task 3 needs is
committed separately in models/task2_default/.

Run it after rerunning any script whose numbers a report cites, and commit results/ together
with the report change.

Usage:  python scripts/snapshot_results.py
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from xroom.config import load_task2_config  # noqa: E402

EXCLUDE_SUFFIXES = {".pt"}


def main() -> None:
    cfg = load_task2_config()
    src = Path(cfg["paths"]["encoder_dir"]).parent  # outputs/
    dst = ROOT / "results"
    keep = {dst / "README.md"}
    for old in sorted(dst.glob("*")) if dst.exists() else []:
        if old not in keep:
            shutil.rmtree(old) if old.is_dir() else old.unlink()
    n = 0
    for f in sorted(src.rglob("*")):
        if f.is_file() and f.suffix not in EXCLUDE_SUFFIXES:
            out = dst / f.relative_to(src)
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(f, out)
            n += 1
    print(f"copied {n} files from {src} to {dst} (checkpoints excluded)")


if __name__ == "__main__":
    main()
