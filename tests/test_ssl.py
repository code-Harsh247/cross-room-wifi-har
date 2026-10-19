"""Tests for the SSL objectives and training loop (Task 2, gates T4-T5)."""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from xroom.config import PROJECT_ROOT, apply_overrides, load_task2_config
from xroom.ssl import effective_rank, mae_loss, nt_xent, time_patch_mask, train_ssl

CFG = load_task2_config()


def test_nt_xent_identical_views_beat_random_pairs():
    g = torch.Generator().manual_seed(0)
    z = torch.randn(32, 16, generator=g)
    assert nt_xent(z, z, 0.2) < nt_xent(z, torch.randn(32, 16, generator=g), 0.2)


def test_nt_xent_analytic_value():
    # B = 2, views identical, examples orthogonal: each anchor sees its positive at
    # similarity 1 and two negatives at 0  ->  loss = -log(e^(1/t) / (e^(1/t) + 2)).
    t = 0.5
    z = torch.eye(2)
    expected = -math.log(math.exp(1 / t) / (math.exp(1 / t) + 2))
    assert abs(float(nt_xent(z, z.clone(), t)) - expected) < 1e-5


def test_nt_xent_symmetric():
    g = torch.Generator().manual_seed(1)
    a, b = torch.randn(8, 4, generator=g), torch.randn(8, 4, generator=g)
    assert abs(float(nt_xent(a, b, 0.3)) - float(nt_xent(b, a, 0.3))) < 1e-6


def test_mae_loss_masked_rows_only_and_mask_ratio():
    g = torch.Generator().manual_seed(0)
    target = torch.randn(512, 150, 242, generator=g)
    mask = time_patch_mask(512, 150, 10, 0.5, g)
    assert abs(float(mask.float().mean()) - 0.5) <= 0.02
    assert float(mae_loss(target.clone(), target, mask)) == 0.0
    pred = target.clone()
    pred[~mask] += 100.0  # errors on unmasked rows are ignored
    assert float(mae_loss(pred, target, mask)) == 0.0
    pred[mask] += 1.0
    assert abs(float(mae_loss(pred, target, mask)) - 1.0) < 1e-5


def test_effective_rank_bounds():
    g = torch.Generator().manual_seed(0)
    assert effective_rank(torch.randn(500, 1, generator=g) @ torch.ones(1, 32)) < 1.5
    assert effective_rank(torch.randn(2000, 32, generator=g)) > 25


def _toy_table():
    rng = np.random.default_rng(0)
    rows, seqs = [], {}
    for room, length in (("A", 150), ("B", 500)):
        for i in range(10):
            cid = f"{room}/c{i}"
            rows.append({"capture_id": cid, "room": room, "split": "val" if i == 9 else "dev",
                         "dev_fold": "holdout" if i == 8 else "train"})
            seqs[cid] = rng.uniform(1, 100, size=(length, 256)).astype(np.float16)
    return pd.DataFrame(rows), seqs


@pytest.mark.parametrize("method", ["simclr", "mae"])
def test_training_runs_saves_and_reloads_exactly(tmp_path, method):
    from xroom.encoder import load_run

    cfg = apply_overrides(CFG, [f"ssl.method={method}", "ssl.epochs=2", "ssl.warmup_epochs=1",
                                "ssl_data.steps_per_epoch=2", "ssl_data.batch_size=8",
                                "ssl_data.holdout_batches=1", "ssl.amp=false"])
    table, seqs = _toy_table()
    run = train_ssl(cfg, table, seqs, tmp_path, device="cpu", log=lambda *_: None)
    assert run["n_train_captures"] == 16 and run["method"] == method
    hist = json.loads((tmp_path / "history.json").read_text())
    assert len(hist) == 2 and all(np.isfinite(h["train_loss"]) for h in hist)
    enc, stats = load_run(tmp_path, device="cpu")
    enc2, _ = load_run(tmp_path, device="cpu")
    x = torch.randn(3, 150, len(stats.data_sc))
    with torch.no_grad():
        assert (enc(x) - enc2(x)).abs().max() < 1e-5


def test_ssl_code_never_reads_activity_labels():
    files = ["src/xroom/ssl_data.py", "src/xroom/ssl.py", "src/xroom/augment.py", "scripts/task2_pretrain.py"]
    for f in files:
        path = PROJECT_ROOT / f
        if path.exists():
            code = "\n".join(line for line in path.read_text().splitlines() if not line.strip().startswith("#"))
            code = re.sub(r'""".*?"""', "", code, flags=re.S)  # docstrings may mention the rule
            assert "activity" not in code, f
