"""Tests for the CSI encoder and heads (Task 2, gate T3)."""
from __future__ import annotations

import pytest
import torch

from xroom.config import load_task2_config
from xroom.encoder import MAEDecoder, ProjectionHead, build_encoder, n_parameters

CFG = load_task2_config()["encoder"]


def _enc(seed=0):
    torch.manual_seed(seed)
    return build_encoder(CFG)


@pytest.mark.parametrize("t", [32, 96, 150])
def test_variable_length(t):
    enc = _enc().eval()
    with torch.no_grad():
        z = enc(torch.randn(4, t, 242))
    assert z.shape == (4, CFG["embed_dim"])


def test_every_parameter_gets_a_gradient():
    enc = _enc()
    head = ProjectionHead(CFG["embed_dim"], CFG["proj_hidden"], CFG["proj_dim"])
    dec = MAEDecoder(enc.feature_dim)
    x = torch.randn(8, 100, 242)
    loss = head(enc(x)).pow(2).mean() + dec(enc.features(x), 100).pow(2).mean()
    loss.backward()
    for name, p in [*enc.named_parameters(), *head.named_parameters(), *dec.named_parameters()]:
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0, name


def test_size_and_heads():
    enc = _enc()
    head = ProjectionHead(CFG["embed_dim"], CFG["proj_hidden"], CFG["proj_dim"])
    assert n_parameters(enc) < 1_500_000
    assert head(torch.randn(4, CFG["embed_dim"])).shape == (4, CFG["proj_dim"])
    assert MAEDecoder(enc.feature_dim)(enc.features(torch.randn(2, 96, 242)), 96).shape == (2, 96, 242)


def test_eval_is_deterministic():
    enc = _enc().eval()
    x = torch.randn(4, 150, 242)
    with torch.no_grad():
        assert torch.equal(enc(x), enc(x))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
def test_gpu_matches_cpu():
    enc = _enc().eval()
    x = torch.randn(4, 150, 242)
    with torch.no_grad():
        cpu = enc(x)
        gpu = enc.cuda()(x.cuda()).cpu()
    assert torch.allclose(cpu, gpu, atol=1e-4)
