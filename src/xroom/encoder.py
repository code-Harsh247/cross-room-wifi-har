"""CSI encoder and SSL heads (Task 2, PRD FR-3).

* ``CSIEncoder``: 1-D CNN over time with the 242 data subcarriers as input channels,
  Conv1d -> BatchNorm -> ReLU blocks, global average pooling over time, linear to d.
  Global pooling makes it accept any length >= 32 rows.
* ``ProjectionHead``: SimCLR MLP d -> hidden -> out (training only, discarded afterwards).
* ``MAEDecoder``: maps the encoder's per-time features back to (T, 242) rows for masked
  reconstruction (training only).
* Task 3 hand-off (FR-10): ``load_encoder(cfg, run)`` -> ``FrozenEncoder`` and
  ``embed(encoder, windows)`` -> (n, d) embeddings of raw amplitude windows.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .ssl_data import NormStats, apply_norm


class CSIEncoder(nn.Module):
    def __init__(self, in_channels: int = 242, channels=(128, 128, 256, 256), kernels=(7, 5, 5, 5),
                 strides=(1, 2, 2, 2), embed_dim: int = 128):
        super().__init__()
        if not len(channels) == len(kernels) == len(strides):
            raise ValueError("channels, kernels and strides must have the same length")
        blocks, prev = [], in_channels
        for ch, k, s in zip(channels, kernels, strides):
            blocks += [nn.Conv1d(prev, ch, k, stride=s, padding=k // 2, bias=False), nn.BatchNorm1d(ch),
                       nn.ReLU(inplace=True)]
            prev = ch
        self.blocks = nn.Sequential(*blocks)
        self.out = nn.Linear(prev, embed_dim)
        self.in_channels, self.feature_dim, self.embed_dim = in_channels, prev, embed_dim

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, C) normalized crops -> (B, F, T') per-time features."""
        return self.blocks(x.transpose(1, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, C) -> (B, embed_dim) embedding."""
        return self.out(self.features(x).mean(dim=2))


class ProjectionHead(nn.Module):
    def __init__(self, embed_dim: int = 128, hidden: int = 128, out_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(embed_dim, hidden), nn.BatchNorm1d(hidden), nn.ReLU(inplace=True),
                                 nn.Linear(hidden, out_dim))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class MAEDecoder(nn.Module):
    """Per-time encoder features (B, F, T') -> reconstructed rows (B, T, C)."""

    def __init__(self, feature_dim: int, out_channels: int = 242, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(nn.Conv1d(feature_dim, hidden, 3, padding=1), nn.ReLU(inplace=True),
                                 nn.Conv1d(hidden, out_channels, 1))

    def forward(self, feats: torch.Tensor, length: int) -> torch.Tensor:
        up = F.interpolate(feats, size=length, mode="linear", align_corners=False)
        return self.net(up).transpose(1, 2)


def build_encoder(enc_cfg: dict, in_channels: int = 242) -> CSIEncoder:
    """Encoder from the ``encoder`` config section."""
    return CSIEncoder(in_channels, tuple(enc_cfg["channels"]), tuple(enc_cfg["kernels"]),
                      tuple(enc_cfg["strides"]), enc_cfg["embed_dim"])


def n_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


@dataclass
class FrozenEncoder:
    """A trained encoder plus the dev normalization it expects; what Task 3 consumes."""

    model: CSIEncoder
    stats: NormStats
    device: str
    run_dir: Path | None = None

    @property
    def dim(self) -> int:
        return self.model.embed_dim


def default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_encoder(cfg: dict, run: str = "default", device: str | None = None) -> FrozenEncoder:
    """Load the label-clean default encoder (``run="default"``: ``paths.default_encoder``, committed
    to the repo) or any SSL run under ``<paths.encoder_dir>/<run>``."""
    paths = cfg["paths"]
    run_dir = Path(paths["default_encoder"]) if run == "default" and "default_encoder" in paths else \
        Path(paths["encoder_dir"]) / run
    if not (run_dir / "encoder.pt").exists():
        raise FileNotFoundError(f"no encoder at {run_dir}; train it with scripts/task2_pretrain.py")
    device = device or default_device()
    model, stats = load_run(run_dir, device)
    return FrozenEncoder(model, stats, device, run_dir)


def embed(encoder: FrozenEncoder, windows: np.ndarray, batch: int = 256) -> np.ndarray:
    """Raw amplitude windows (n, T, 256), any float dtype / memmap, T >= 32 ->
    (n, d) float32 embeddings. Applies the saved dev normalization; eval mode, no grad."""
    model = encoder.model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(windows), batch):
            xb = torch.from_numpy(apply_norm(np.asarray(windows[i : i + batch]), encoder.stats))
            out.append(model(xb.to(encoder.device)).float().cpu().numpy())
    if not out:
        return np.zeros((0, encoder.dim), np.float32)
    return np.concatenate(out).astype(np.float32)


def load_run(run_dir: str | Path, device: str = "cpu", which: str = "encoder.pt") -> tuple[CSIEncoder, NormStats]:
    """Frozen encoder (eval mode) + its dev normalization statistics from an SSL run folder."""
    run_dir = Path(run_dir)
    ck = torch.load(run_dir / which, map_location=device, weights_only=True)
    enc = build_encoder(ck["encoder_cfg"], ck["in_channels"])
    enc.load_state_dict(ck["encoder"])
    enc.to(device).eval()
    for p in enc.parameters():
        p.requires_grad_(False)
    return enc, NormStats.load(run_dir / "norm_stats.npz")
