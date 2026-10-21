"""Task 3 domain adaptation building blocks (PRD §6, FR-4).

Everything here uses unlabelled development data only. No function takes activity labels,
except prototype refinement, which takes the support labels the access guard allows.
"""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import torch
from torch import nn

from .encoder import FrozenEncoder
from .ssl_data import apply_norm


def adabn(encoder: FrozenEncoder, x: np.ndarray, meta: pd.DataFrame, room: str, batch: int = 256) -> FrozenEncoder:
    """AdaBN (M3): a copy of ``encoder`` whose BatchNorm running statistics are re-estimated on
    the unlabelled **development** windows of one room. Weights are unchanged, and the input
    normalization (``encoder.stats``) is kept. The original encoder is not modified.

    Running stats are a cumulative average over batches (``momentum=None``) of the room's dev
    windows, in a fixed order, so the result is deterministic.
    """
    rows = np.where(((meta["room"] == room) & (meta["split"] == "dev")).to_numpy())[0]
    if len(rows) == 0:
        raise ValueError(f"no development windows for room {room!r}")
    if (meta["split"].to_numpy()[rows] != "dev").any():  # defensive: never validation
        raise AssertionError("AdaBN must only see development windows")
    model = copy.deepcopy(encoder.model)
    bns = [m for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    if not bns:
        raise ValueError("encoder has no BatchNorm layers")
    for bn in bns:
        bn.reset_running_stats()
        bn.momentum = None  # cumulative moving average over all batches
    model.train()  # BatchNorm updates running stats in train mode; the encoder has no dropout
    with torch.no_grad():
        for i in range(0, len(rows), batch):
            xb = torch.from_numpy(apply_norm(np.asarray(x[rows[i : i + batch]]), encoder.stats))
            model(xb.to(encoder.device))
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return FrozenEncoder(model, encoder.stats, encoder.device, encoder.run_dir)
