"""Augmentations for contrastive SSL (Task 2, PRD FR-2).

Every function takes a batch of normalized crops ``x`` (B, T, C), a ``torch.Generator`` and
its settings, and applies the augmentation to each example independently with probability
``p`` (examples not drawn are returned bit-for-bit unchanged). Shape and dtype are kept.
All randomness comes from the generator, so a seed reproduces the output exactly.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

Tensor = torch.Tensor


def _rand(g: torch.Generator, *shape, device) -> Tensor:
    return torch.rand(*shape, generator=g, device=device)


def _randint(g: torch.Generator, lo: int, hi: int, n: int, device) -> Tensor:
    """n integers uniform in [lo, hi] (inclusive)."""
    return torch.randint(lo, hi + 1, (n,), generator=g, device=device)


def _chosen(g: torch.Generator, x: Tensor, p: float) -> Tensor:
    """(B, 1, 1) boolean: which examples get the augmentation."""
    return (_rand(g, x.shape[0], device=x.device) < p).view(-1, 1, 1)


def crop_resize(x: Tensor, g: torch.Generator, p: float, min_frac: float) -> Tensor:
    """Random contiguous sub-crop covering a fraction in [min_frac, 1] of the rows,
    linearly resized back to T rows."""
    b, t, c = x.shape
    frac = min_frac + (1 - min_frac) * _rand(g, b, device=x.device)
    span = frac * (t - 1)
    start = _rand(g, b, device=x.device) * (t - 1 - span)
    pos = start[:, None] + torch.linspace(0, 1, t, device=x.device)[None] * span[:, None]
    i0 = pos.floor().long().clamp(0, t - 1)
    i1 = (i0 + 1).clamp(max=t - 1)
    w = (pos - i0).unsqueeze(-1).to(x.dtype)
    gather = lambda i: torch.gather(x, 1, i.unsqueeze(-1).expand(b, t, c))  # noqa: E731
    out = gather(i0) * (1 - w) + gather(i1) * w
    return torch.where(_chosen(g, x, p), out, x)


def gaussian_noise(x: Tensor, g: torch.Generator, p: float, sigma: float) -> Tensor:
    """Additive noise with standard deviation ``sigma`` times the example's own std."""
    sd = x.std(dim=(1, 2), keepdim=True)
    noise = torch.randn(x.shape, generator=g, device=x.device, dtype=x.dtype)
    return torch.where(_chosen(g, x, p), x + sigma * sd * noise, x)


def _block_mask(g: torch.Generator, b: int, n: int, blocks: tuple[int, int], width: tuple[int, int],
                device) -> Tensor:
    """(b, n) boolean mask with ``blocks`` contiguous runs of ``width`` positions each."""
    n_blocks = _randint(g, blocks[0], blocks[1], b, device)
    ar = torch.arange(n, device=device)[None]
    mask = torch.zeros(b, n, dtype=torch.bool, device=device)
    for k in range(blocks[1]):
        w = _randint(g, width[0], min(width[1], n), b, device)
        start = (_rand(g, b, device=device) * (n - w + 1)).long()
        run = (ar >= start[:, None]) & (ar < (start + w)[:, None])
        mask |= run & (k < n_blocks)[:, None]
    return mask


def subcarrier_mask(x: Tensor, g: torch.Generator, p: float, blocks, width) -> Tensor:
    """Zero 1-3 contiguous blocks of subcarriers (0 = the dev mean after standardization)."""
    mask = _block_mask(g, x.shape[0], x.shape[2], tuple(blocks), tuple(width), x.device)[:, None, :]
    return torch.where(_chosen(g, x, p) & mask, torch.zeros((), dtype=x.dtype, device=x.device), x)


def time_mask(x: Tensor, g: torch.Generator, p: float, blocks, width) -> Tensor:
    """Zero 1-2 contiguous blocks of rows."""
    mask = _block_mask(g, x.shape[0], x.shape[1], tuple(blocks), tuple(width), x.device)[:, :, None]
    return torch.where(_chosen(g, x, p) & mask, torch.zeros((), dtype=x.dtype, device=x.device), x)


def row_gain(x: Tensor, g: torch.Generator, p: float, sigma: float) -> Tensor:
    """Multiply each row by 1 + eps, eps ~ N(0, sigma^2): no per-packet gain cue survives."""
    eps = torch.randn((x.shape[0], x.shape[1], 1), generator=g, device=x.device, dtype=x.dtype)
    return torch.where(_chosen(g, x, p), x * (1 + sigma * eps), x)


def smooth_rows(x: Tensor, width: int) -> Tensor:
    """Moving average over rows (odd ``width``), edge rows repeated."""
    if width <= 1:
        return x
    xt = F.pad(x.transpose(1, 2), (width // 2, width // 2), mode="replicate")
    return F.avg_pool1d(xt, width, stride=1).transpose(1, 2)


def format_jitter(x: Tensor, g: torch.Generator, p: float, smooth_width, noise_sigma: float) -> Tensor:
    """Format robustness: half of the chosen examples are smoothed over time (Room A looks
    smooth), the other half get independent row noise (Room B looks noisy). This breaks the
    "smooth = A, noisy = B" shortcut instead of letting the encoder learn it."""
    b = x.shape[0]
    widths = list(range(smooth_width[0] | 1, smooth_width[1] + 1, 2))
    pick = _randint(g, 0, len(widths) - 1, b, x.device)
    smoothed = x
    for k, w in enumerate(widths):
        smoothed = torch.where((pick == k).view(-1, 1, 1), smooth_rows(x, w), smoothed)
    noise = torch.randn(x.shape, generator=g, device=x.device, dtype=x.dtype)
    noisy = x + noise_sigma * x.std(dim=(1, 2), keepdim=True) * noise
    use_smooth = (_rand(g, b, device=x.device) < 0.5).view(-1, 1, 1)
    out = torch.where(use_smooth, smoothed, noisy)
    return torch.where(_chosen(g, x, p), out, x)


def augment(x: Tensor, g: torch.Generator, cfg: dict) -> Tensor:
    """One random view. ``cfg`` is the ``augment`` config section; a missing key = off."""
    if "crop_resize" in cfg:
        x = crop_resize(x, g, **cfg["crop_resize"])
    if "format" in cfg:
        x = format_jitter(x, g, **cfg["format"])
    if "gain" in cfg:
        x = row_gain(x, g, **cfg["gain"])
    if "noise" in cfg:
        x = gaussian_noise(x, g, **cfg["noise"])
    if "subcarrier_mask" in cfg:
        x = subcarrier_mask(x, g, **cfg["subcarrier_mask"])
    if "time_mask" in cfg:
        x = time_mask(x, g, **cfg["time_mask"])
    return x


def make_views(x: Tensor, g: torch.Generator, cfg: dict) -> tuple[Tensor, Tensor]:
    """Two independent random views of the same crops (SimCLR positives)."""
    return augment(x, g, cfg), augment(x, g, cfg)
