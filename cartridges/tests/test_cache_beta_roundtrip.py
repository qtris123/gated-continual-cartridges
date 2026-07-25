"""Tests for TrainableCache beta checkpoint round-trip."""

from __future__ import annotations

import tempfile
from pathlib import Path

import torch

from cartridges.cache import AttnConfig, TrainableCache


def _make_cache(n_layers=2, n_heads=2, n_tokens=8, head_dim=16) -> TrainableCache:
    attn = AttnConfig(n_layers=n_layers, n_heads=n_heads, head_dim=head_dim)
    init_keys = [
        torch.randn(1, n_heads, n_tokens, head_dim) for _ in range(n_layers)
    ]
    init_values = [
        torch.randn(1, n_heads, n_tokens, head_dim) for _ in range(n_layers)
    ]
    return TrainableCache(
        config=attn,
        init_keys=init_keys,
        init_values=init_values,
        num_frozen_tokens=0,
    )


def test_v1_checkpoint_loads_without_beta():
    cache = _make_cache()
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "cache.pt"
        cache.save(str(path))
        payload = torch.load(path, weights_only=False)
        del payload["trainable_beta"]
        payload["cache_format_version"] = 1
        torch.save(payload, path)
        loaded = TrainableCache.from_pretrained(str(path))
        assert loaded.trainable_beta is not None
        assert torch.all(loaded.trainable_beta[0] == 0)


def test_v2_beta_roundtrip():
    cache = _make_cache()
    with torch.no_grad():
        cache.trainable_beta[0].fill_(0.5)
        cache.trainable_beta[1].fill_(-0.25)
    cache.enable_attention_bias(True)

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "cache.pt"
        cache.save(str(path))
        loaded = TrainableCache.from_pretrained(str(path))
        assert torch.allclose(loaded.trainable_beta[0], cache.trainable_beta[0])
        assert torch.allclose(loaded.trainable_beta[1], cache.trainable_beta[1])
        assert loaded._attention_bias_enabled
