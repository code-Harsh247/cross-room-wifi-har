"""Pre-train one self-supervised CSI encoder (Task 2, T5).

Trains on development crops of both rooms (dev_fold == "train"), monitors the same SSL loss
on dev-holdout crops (label-free), and keeps the lowest-holdout-loss epoch.
Writes outputs/task2/<run>/ (see xroom.ssl.train_ssl).

Usage:
    python scripts/task2_pretrain.py --run simclr_s0 --seed 0
    python scripts/task2_pretrain.py --run debug --max-windows 256 --epochs 50
    python scripts/task2_pretrain.py --run mae_s0 --method mae --set ssl.lr=5e-4
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from xroom.config import apply_overrides, load_task2_config  # noqa: E402
from xroom.dataset import load_sequences  # noqa: E402
from xroom.ssl import train_ssl  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="run name -> outputs/task2/<run>/")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--method", choices=["simclr", "mae"], default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--max-windows", type=int, default=None,
                    help="train on at most this many dev-train captures (debug / overfit check)")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                    help="config overrides, e.g. ssl.temperature=0.5 augment.format=null")
    ap.add_argument("--exclude", type=Path, default=None,
                    help="JSON list of capture ids to keep out of SSL training")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    overrides = list(args.set)
    if args.seed is not None:
        overrides += [f"ssl.seed={args.seed}"]
    if args.method:
        overrides += [f"ssl.method={args.method}"]
    if args.epochs:
        overrides += [f"ssl.epochs={args.epochs}"]
    cfg = apply_overrides(load_task2_config(), overrides)
    exclude = set(json.loads(args.exclude.read_text())) if args.exclude else None
    table, seqs = load_sequences(cfg)
    out = Path(cfg["paths"]["encoder_dir"]) / args.run
    train_ssl(cfg, table, seqs, out, max_captures=args.max_windows, exclude=exclude, device=args.device)


if __name__ == "__main__":
    main()
