"""Self-supervised objectives and training loop (Task 2, PRD FR-4).

* ``nt_xent``: SimCLR contrastive loss; two views of a crop are positives, every other
  example in the batch is a negative.
* ``mae_loss``: masked reconstruction (MSE on masked rows only), with ``time_patch_mask``.
* ``train_ssl``: AdamW + warm-up/cosine schedule on room-balanced dev crops. The label-free
  validation signal is the same SSL loss on fixed dev-holdout crops; the saved encoder is the
  epoch with the lowest holdout loss. No activity label is read anywhere in this module.
"""
from __future__ import annotations

import json
import math
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml

from .augment import make_views
from .config import PROJECT_ROOT
from .encoder import MAEDecoder, ProjectionHead, build_encoder, n_parameters
from .ssl_data import CropSampler, dev_capture_ids, fit_dev_stats

# ---------- objectives ----------


def nt_xent(z1: torch.Tensor, z2: torch.Tensor, temperature: float) -> torch.Tensor:
    """Normalized-temperature cross-entropy over 2B embeddings (computed in float32)."""
    b = z1.shape[0]
    z = F.normalize(torch.cat([z1, z2]).float(), dim=1)
    sim = z @ z.T / temperature
    sim.fill_diagonal_(float("-inf"))
    targets = torch.cat([torch.arange(b, 2 * b), torch.arange(0, b)]).to(z.device)
    return F.cross_entropy(sim, targets)


def time_patch_mask(b: int, t: int, patch: int, ratio: float, g: torch.Generator, device="cpu") -> torch.Tensor:
    """(B, T) boolean mask of masked rows: whole ``patch``-row blocks, about ``ratio`` of them.
    The number of masked patches is rounded randomly (up or down) so the expected share of
    masked patches equals ``ratio`` exactly."""
    n_p = math.ceil(t / patch)
    k = torch.floor(ratio * n_p + torch.rand(b, generator=g, device=device)).long()
    order = torch.rand(b, n_p, generator=g, device=device).argsort(dim=1)
    ranks = order.argsort(dim=1)
    patch_mask = ranks < k[:, None]
    return patch_mask.repeat_interleave(patch, dim=1)[:, :t]


def mae_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean squared error over the masked rows only; ``mask`` is (B, T)."""
    per_row = (pred.float() - target.float()).pow(2).mean(dim=2)
    return per_row[mask].mean()


# ---------- collapse monitoring ----------


def effective_rank(z: torch.Tensor) -> float:
    """exp(entropy) of the normalized singular values of centred embeddings (1 .. d)."""
    s = torch.linalg.svdvals((z - z.mean(0)).double())
    p = s / s.sum()
    p = p[p > 0]
    return float(torch.exp(-(p * p.log()).sum()))


# ---------- training ----------


def git_hash() -> str:
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, capture_output=True,
                             text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=PROJECT_ROOT,
                               capture_output=True, text=True).stdout.strip()
        return sha + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _plain(obj):
    """Config -> YAML-safe (paths as strings)."""
    if isinstance(obj, dict):
        return {k: _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    return str(obj) if isinstance(obj, Path) else obj


def _lr_lambda(total: int, warmup: int):
    def f(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        return 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total - warmup)))
    return f


class SSLModel(torch.nn.Module):
    """Encoder + the head of the chosen objective; ``loss`` returns one scalar."""

    def __init__(self, cfg: dict, in_channels: int):
        super().__init__()
        self.method = cfg["ssl"]["method"]
        self.encoder = build_encoder(cfg["encoder"], in_channels)
        e = cfg["encoder"]
        if self.method == "simclr":
            self.head = ProjectionHead(e["embed_dim"], e["proj_hidden"], e["proj_dim"])
        elif self.method == "mae":
            self.head = MAEDecoder(self.encoder.feature_dim, in_channels)
        else:
            raise ValueError(f"unknown SSL method {self.method!r}")
        self.cfg = cfg

    def loss(self, x: torch.Tensor, g: torch.Generator) -> torch.Tensor:
        s = self.cfg["ssl"]
        if self.method == "simclr":
            v1, v2 = make_views(x, g, self.cfg["augment"])
            return nt_xent(self.head(self.encoder(v1)), self.head(self.encoder(v2)), s["temperature"])
        mask = time_patch_mask(x.shape[0], x.shape[1], s["mae_patch"], s["mae_mask_ratio"], g, x.device)
        pred = self.head(self.encoder.features(x.masked_fill(mask[..., None], 0.0)), x.shape[1])
        return mae_loss(pred, x, mask)


def train_ssl(cfg: dict, table: pd.DataFrame, seqs: dict, out_dir: str | Path, *, max_captures: int | None = None,
              exclude: set[str] | None = None, device: str | None = None, log=print) -> dict:
    """Pre-train one encoder and write the run folder:

    encoder.pt (lowest holdout-loss epoch), encoder_last.pt, norm_stats.npz, config.yaml,
    history.json (per-epoch losses and collapse metrics), run.json (seed, git hash, runtime).
    ``exclude``: capture ids kept out of SSL training (e.g. for the contiguous-split check).
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    d, s = cfg["ssl_data"], cfg["ssl"]
    seed = int(s["seed"])
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    stats = fit_dev_stats(table, seqs, cfg)
    train_ids = dev_capture_ids(table, seqs, fold="train", exclude=exclude)
    hold_ids = dev_capture_ids(table, seqs, fold="holdout", exclude=exclude)
    if max_captures:
        pick = np.random.default_rng(seed).choice(len(train_ids), min(max_captures, len(train_ids)), replace=False)
        train_ids = [train_ids[i] for i in sorted(pick)]
    kw = dict(table=table, seqs=seqs, stats=stats, batch_size=d["batch_size"], crop_min=d["crop_min"],
              crop_max=d["crop_max"], room_balance=d["room_balance"])
    train = CropSampler(train_ids, seed=seed, **kw)
    hold = CropSampler(hold_ids, seed=10_000 + seed, **kw)
    hold_batches = [hold.batch()[0].astype(np.float16) for _ in range(d["holdout_batches"])]
    del hold

    model = SSLModel(cfg, len(stats.data_sc)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=s["lr"], weight_decay=s["weight_decay"])
    steps = d["steps_per_epoch"]
    sched = torch.optim.lr_scheduler.LambdaLR(opt, _lr_lambda(s["epochs"] * steps, s["warmup_epochs"] * steps))
    amp = bool(s.get("amp")) and device.startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    g = torch.Generator(device=device).manual_seed(seed)

    def to_dev(a: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(np.asarray(a, np.float32)).to(device)

    def holdout_eval() -> dict:
        model.eval()
        gh = torch.Generator(device=device).manual_seed(12_345)  # same views every epoch
        losses, embs = [], []
        with torch.no_grad(), torch.autocast("cuda", enabled=amp):
            for hb in hold_batches:
                xb = to_dev(hb)
                losses.append(float(model.loss(xb, gh)))
                embs.append(model.encoder(xb).float())
        z = torch.cat(embs)
        model.train()
        return {"holdout_loss": float(np.mean(losses)), "emb_std": float(z.std(0).mean()),
                "eff_rank": effective_rank(z)}

    yaml.safe_dump(_plain(cfg), open(out / "config.yaml", "w", encoding="utf-8"), sort_keys=False)
    stats.save(out / "norm_stats.npz")
    log(f"SSL {s['method']} seed={seed} device={device}: {len(train_ids)} train / {len(hold_ids)} holdout "
        f"captures, encoder {n_parameters(model.encoder):,} params")

    history, best, t0 = [], math.inf, time.time()
    for epoch in range(1, s["epochs"] + 1):
        model.train()
        tot = 0.0
        for _ in range(steps):
            xb = to_dev(train.batch()[0])
            with torch.autocast("cuda", enabled=amp):
                loss = model.loss(xb, g)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += float(loss.detach())
        row = {"epoch": epoch, "train_loss": tot / steps, **holdout_eval(), "seconds": round(time.time() - t0, 1)}
        history.append(row)
        if row["holdout_loss"] < best:
            best = row["holdout_loss"]
            torch.save(_checkpoint(model, cfg, epoch), out / "encoder.pt")
        if epoch == 1 or epoch % 5 == 0 or epoch == s["epochs"]:
            log(f"  epoch {epoch:3d}  train {row['train_loss']:.4f}  holdout {row['holdout_loss']:.4f}  "
                f"emb_std {row['emb_std']:.3f}  eff_rank {row['eff_rank']:.1f}  ({row['seconds']:.0f}s)")
    torch.save(_checkpoint(model, cfg, s["epochs"]), out / "encoder_last.pt")
    best_epoch = min(history, key=lambda r: r["holdout_loss"])["epoch"]
    run = {"method": s["method"], "seed": seed, "git": git_hash(), "device": device,
           "runtime_s": round(time.time() - t0, 1), "best_epoch": best_epoch,
           "selection": "lowest SSL loss on dev-holdout crops (label-free)",
           "n_train_captures": len(train_ids), "n_holdout_captures": len(hold_ids),
           "n_excluded_captures": len(exclude or ())}
    json.dump(history, open(out / "history.json", "w"), indent=1)
    json.dump(run, open(out / "run.json", "w"), indent=1)
    log(f"done in {run['runtime_s']:.0f}s; best holdout epoch {best_epoch}; saved {out}")
    return run


def _checkpoint(model: SSLModel, cfg: dict, epoch: int) -> dict:
    return {"encoder": model.encoder.state_dict(), "encoder_cfg": _plain(cfg["encoder"]),
            "in_channels": model.encoder.in_channels, "epoch": epoch, "method": model.method}
