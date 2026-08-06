"""Tests for NNLS beta refitting."""

from __future__ import annotations

import torch

from cartridges.am.components.beta import refit_beta_nnls
from cartridges.am.components.teacher import compute_teacher_log_mass

HEAD_DIM = 16
N_KEYS = 8
N_QUERIES = 12


def test_refit_beta_reduces_mass_error():
    torch.manual_seed(2)
    keys = torch.randn(N_KEYS, HEAD_DIM)
    queries = torch.randn(N_QUERIES, HEAD_DIM)
    source_log_mass = compute_teacher_log_mass(queries, keys, HEAD_DIM)
    target_log_mass = source_log_mass + torch.log(torch.tensor(1.5))

    beta = refit_beta_nnls(keys, queries, target_log_mass, HEAD_DIM)
    inv_sqrt_d = (1.0 / HEAD_DIM) ** 0.5
    scores = (queries @ keys.T).float() * inv_sqrt_d + beta
    approx_log_mass = torch.logsumexp(scores, dim=1)

    err_before = (target_log_mass - source_log_mass).pow(2).mean()
    err_after = (target_log_mass - approx_log_mass).pow(2).mean()
    assert err_after < err_before


def test_refit_beta_selected_scope():
    torch.manual_seed(3)
    keys = torch.randn(N_KEYS, HEAD_DIM)
    queries = torch.randn(N_QUERIES, HEAD_DIM)
    target_log_mass = compute_teacher_log_mass(queries, keys, HEAD_DIM)
    selected = torch.tensor([1, 3, 5])
    base_beta = torch.linspace(-0.2, 0.2, N_KEYS)

    beta = refit_beta_nnls(
        keys,
        queries,
        target_log_mass,
        HEAD_DIM,
        selected_indices=selected,
        base_beta=base_beta,
    )
    assert beta.shape == (N_KEYS,)
    zero_mask = torch.ones(N_KEYS, dtype=torch.bool)
    zero_mask[selected] = False
    assert torch.allclose(beta[zero_mask], base_beta[zero_mask])
