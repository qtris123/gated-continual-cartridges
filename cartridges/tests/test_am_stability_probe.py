"""Unit tests for AM stability probe helpers and ridge floor."""

from __future__ import annotations

import torch

from cartridges.am_stability_probe import (
    probe_tfidf_vs_absolute_mass,
    summarize_slot_mass_probes,
)
from cartridges.attention_matching import effective_ridge_lambda, sparse_am_value_update


def test_effective_ridge_lambda_spectral_floor():
    X = torch.ones(8, 4) * 1e-3
    raw = effective_ridge_lambda(X, ridge_lambda=1e-4, ridge_scale="spectral")
    floored = effective_ridge_lambda(
        X, ridge_lambda=1e-4, ridge_scale="spectral", ridge_lambda_min=1e-4
    )
    assert raw < 1e-4
    assert floored == 1e-4


def test_fixed_ridge_larger_than_spectral_on_tiny_X():
    X = torch.rand(16, 8).abs() * 1e-3
    spectral = effective_ridge_lambda(X, ridge_lambda=1e-4, ridge_scale="spectral")
    fixed = effective_ridge_lambda(X, ridge_lambda=1e-4, ridge_scale="fixed")
    assert fixed > spectral


def test_delta_weight_limits_value_growth():
    torch.manual_seed(0)
    T, d, n, t = 32, 8, 20, 4
    keys = torch.randn(T, d)
    values = torch.randn(T, d) * 0.1
    queries = torch.randn(n, d)
    selected = torch.arange(t)
    # Force a target far from current output so unregularized solve wants large V.
    targets = torch.randn(n, d) * 5.0

    _, stats_spec = sparse_am_value_update(
        keys, values, queries, selected, targets=targets, head_dim=d,
        ridge_lambda=1e-4, ridge_scale="spectral", delta_weight=0.0,
    )
    _, stats_delta = sparse_am_value_update(
        keys, values, queries, selected, targets=targets, head_dim=d,
        ridge_lambda=1e-4, ridge_scale="spectral", delta_weight=1.0,
    )
    assert stats_delta["v_selected_absmax_after"] <= stats_spec["v_selected_absmax_after"] + 1e-5


def test_probe_tfidf_vs_absolute_mass_flags_low_abs_high_tfidf():
    n, T = 10, 16
    alpha = torch.full((n, T), 1.0 / T)
    # Selected slots have high TF-IDF but below-median absolute access.
    abs_access = torch.ones(T)
    abs_access[:4] = 0.1  # selected
    abs_access[4:] = 1.0
    tfidf = torch.ones(T)
    tfidf[:4] = 5.0
    selected = torch.arange(4)
    probe = probe_tfidf_vs_absolute_mass(
        layer_idx=0,
        head_idx=0,
        selected_indices=selected,
        alpha=alpha,
        abs_access=abs_access,
        tfidf=tfidf,
        n_frozen=0,
    )
    assert probe.tfidf_on_S_over_global > 1.0
    assert probe.abs_access_on_S_over_global < 1.0
    assert probe.frac_S_below_global_median_access == 1.0
    summary = summarize_slot_mass_probes([probe])
    assert summary["mean_tfidf_on_S_over_global"] > 1.0
