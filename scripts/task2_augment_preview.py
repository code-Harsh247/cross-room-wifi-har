"""Visual sanity check of the SSL augmentations (Task 2, gate T2).

One development window per room: the normalized original plus 4 augmented views
(time x subcarrier heatmaps). Views of the same window should look alike but not identical.
Writes outputs/task2/augment_preview.png.

Usage:  python scripts/task2_augment_preview.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from xroom.augment import augment  # noqa: E402
from xroom.config import load_task2_config  # noqa: E402
from xroom.dataset import load_sequences, load_windows  # noqa: E402
from xroom.ssl_data import apply_norm, fit_dev_stats  # noqa: E402


def main() -> None:
    cfg = load_task2_config()
    x, meta = load_windows(cfg)
    table, seqs = load_sequences(cfg)
    stats = fit_dev_stats(table, seqs, cfg)
    del seqs
    rng = np.random.default_rng(0)
    g = torch.Generator().manual_seed(0)
    fig, axes = plt.subplots(2, 5, figsize=(16, 6.2), constrained_layout=True)
    for r, room in enumerate("AB"):
        idx = rng.choice(np.flatnonzero(((meta["room"] == room) & (meta["split"] == "dev")).to_numpy()))
        w = torch.from_numpy(apply_norm(np.asarray(x[idx : idx + 1]), stats))
        views = [w] + [augment(w, g, cfg["augment"]) for _ in range(4)]
        for c, v in enumerate(views):
            ax = axes[r, c]
            im = ax.imshow(v[0].numpy().T, aspect="auto", origin="lower", cmap="viridis", vmin=-3, vmax=3,
                           interpolation="nearest")
            ax.set_title(f"Room {room}: " + ("original" if c == 0 else f"view {c}"), fontsize=10)
            ax.set_xlabel("row (time)")
            if c == 0:
                ax.set_ylabel("data subcarrier")
    fig.colorbar(im, ax=axes, shrink=0.7, label="normalized amplitude (z)")
    fig.suptitle("SSL augmentations: normalized window and 4 random views (configs/task2.yaml)")
    out = Path(cfg["paths"]["encoder_dir"]) / "augment_preview.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
