"""Tests for teacher KV concat and target computation."""

from __future__ import annotations

import torch

from cartridges.am.components.teacher import (
    compute_teacher_mass,
    compute_teacher_targets,
    concat_teacher_kv,
)
from cartridges.am.core import compute_attention_output

HEAD_DIM = 16


def test_concat_teacher_kv_shapes():
    k_c = torch.randn(4, HEAD_DIM)
    v_c = torch.randn(4, HEAD_DIM)
    k_d = torch.randn(6, HEAD_DIM)
    v_d = torch.randn(6, HEAD_DIM)
    k_t, v_t = concat_teacher_kv(k_c, v_c, k_d, v_d)
    assert k_t.shape == (10, HEAD_DIM)
    assert v_t.shape == (10, HEAD_DIM)


def test_teacher_targets_match_manual_attention():
    torch.manual_seed(0)
    k_c = torch.randn(3, HEAD_DIM)
    v_c = torch.randn(3, HEAD_DIM)
    k_d = torch.randn(2, HEAD_DIM)
    v_d = torch.randn(2, HEAD_DIM)
    queries = torch.randn(5, HEAD_DIM)
    k_t, v_t = concat_teacher_kv(k_c, v_c, k_d, v_d)

    y1 = compute_teacher_targets(queries, k_t, v_t, HEAD_DIM)
    y2 = compute_attention_output(queries, k_t, v_t, HEAD_DIM)
    assert torch.allclose(y1, y2, atol=1e-5)


def test_teacher_doc_rope_offset_changes_scores():
    torch.manual_seed(2)
    head_dim = 16
    k_c = torch.randn(3, head_dim)
    k_d = torch.randn(2, head_dim)
    queries = torch.randn(4, head_dim)
    k_t = torch.cat([k_c, k_d], dim=0)

    from cartridges.am.core import _attention_scores

    naive = (queries @ k_t.T).to(torch.float32) * (1.0 / head_dim) ** 0.5
    corrected = _attention_scores(
        queries,
        k_t,
        head_dim,
        doc_key_start=k_c.shape[0],
        doc_rope_offset=k_d.shape[0],
    )
    assert not torch.allclose(naive, corrected, atol=1e-5)


def test_teacher_mass_positive():
    torch.manual_seed(1)
    keys = torch.randn(8, HEAD_DIM)
    queries = torch.randn(4, HEAD_DIM)
    mass = compute_teacher_mass(queries, keys, HEAD_DIM)
    assert mass.shape == (4,)
    assert (mass > 0).all()
