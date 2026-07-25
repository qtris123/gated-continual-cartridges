"""Tests for OMP key selection."""

from __future__ import annotations

import torch

from cartridges.attention_matching import select_keys_highest_attention, select_keys_omp

HEAD_DIM = 16
N_KEYS = 20
N_QUERIES = 10


def test_omp_selects_requested_count():
    torch.manual_seed(4)
    keys = torch.randn(N_KEYS, HEAD_DIM)
    queries = torch.randn(N_QUERIES, HEAD_DIM)
    t = 5
    sel_k, beta, indices = select_keys_omp(keys, queries, t, HEAD_DIM)
    assert sel_k.shape == (t, HEAD_DIM)
    assert beta.shape == (t,)
    assert len(indices) == t
    assert len(set(indices)) == t


def test_omp_and_highest_attention_return_valid_indices():
    torch.manual_seed(5)
    keys = torch.randn(N_KEYS, HEAD_DIM)
    keys[0] = keys[0] * 5.0
    queries = torch.randn(N_QUERIES, HEAD_DIM)
    t = 4

    _, _, idx_omp = select_keys_omp(keys, queries, t, HEAD_DIM)
    _, _, idx_ha = select_keys_highest_attention(
        keys, queries, t, HEAD_DIM, score_method="rms",
    )
    assert len(idx_omp) == t and len(set(idx_omp)) == t
    assert len(idx_ha) == t and len(set(idx_ha)) == t
