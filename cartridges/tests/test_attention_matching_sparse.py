"""Unit tests for sparse attention matching value updates."""

from __future__ import annotations

import pytest
import torch

from cartridges.am.components.objective import (
    guarded_sparse_am_value_update,
    sparse_am_value_update,
)
from cartridges.am.components.queries import AMTargetAccumulator
from cartridges.am.core import compute_attention_output
from cartridges.am.initial.compaction import naive_compaction_c2_update


HEAD_DIM = 32
N_QUERIES = 16
N_TOKENS = 64
TOP_T = 8


def _random_kv_queries(seed: int = 0, dtype=torch.float32):
    torch.manual_seed(seed)
    keys = torch.randn(N_TOKENS, HEAD_DIM, dtype=dtype)
    values = torch.randn(N_TOKENS, HEAD_DIM, dtype=dtype)
    queries = torch.randn(N_QUERIES, HEAD_DIM, dtype=dtype)
    return keys, values, queries


class TestSparseAMValueUpdate:
    def test_self_match_near_zero_mse(self):
        keys, values, queries = _random_kv_queries()
        selected = torch.arange(TOP_T)

        new_values, stats = sparse_am_value_update(
            keys, values, queries, selected, head_dim=HEAD_DIM, ridge_lambda=1e-6,
        )

        target = compute_attention_output(queries, keys, values, HEAD_DIM)
        output = compute_attention_output(queries, keys, new_values, HEAD_DIM)
        mse = torch.mean((output - target) ** 2).item()

        assert mse < 1e-5, f"Self-match MSE too high: {mse}"
        assert stats["mse"] < 1e-5

    def test_diagnostic_free_update_matches_full_stats_update(self):
        keys, values, queries = _random_kv_queries(seed=7)
        selected = torch.arange(TOP_T)

        new_values_full, stats_full = sparse_am_value_update(
            keys, values, queries, selected,
            head_dim=HEAD_DIM, ridge_lambda=1e-6, compute_stats=True,
        )
        new_values_fast, stats_fast = sparse_am_value_update(
            keys, values, queries, selected,
            head_dim=HEAD_DIM, ridge_lambda=1e-6, compute_stats=False,
        )

        assert torch.allclose(new_values_fast, new_values_full, atol=1e-6)
        assert stats_full["mse"] < 1e-5
        assert stats_fast["mse"] is None
        assert stats_fast["residual_norm"] is None
        assert stats_fast["n_selected"] == TOP_T

    def test_naive_compaction_c2_fails_sparse_case(self):
        """Naive compaction C2 (attention over subset only) should NOT self-match."""
        keys, values, queries = _random_kv_queries(seed=42)
        selected = torch.arange(TOP_T)

        _, stats_correct = sparse_am_value_update(
            keys, values, queries, selected, head_dim=HEAD_DIM, ridge_lambda=1e-6,
        )
        _, stats_naive = naive_compaction_c2_update(
            keys, values, queries, selected, head_dim=HEAD_DIM, ridge_lambda=1e-6,
        )

        # Correct method should achieve near-zero MSE
        assert stats_correct["mse"] < 1e-4
        # Naive method should have significantly higher MSE
        assert stats_naive["mse"] > stats_correct["mse"] * 10

    def test_teacher_match(self):
        keys, values, queries = _random_kv_queries(seed=1)
        teacher_values = values + 0.5 * torch.randn_like(values)
        selected = torch.arange(TOP_T)

        targets = compute_attention_output(queries, keys, teacher_values, HEAD_DIM)
        new_values, stats = sparse_am_value_update(
            keys, values, queries, selected,
            targets=targets, head_dim=HEAD_DIM, ridge_lambda=1e-4,
        )

        output = compute_attention_output(queries, keys, new_values, HEAD_DIM)
        mse = torch.mean((output - targets) ** 2).item()
        assert mse < 0.1, f"Teacher-match MSE too high: {mse}"

    def test_single_slot(self):
        keys, values, queries = _random_kv_queries(seed=2)
        selected = torch.tensor([5])

        new_values, stats = sparse_am_value_update(
            keys, values, queries, selected, head_dim=HEAD_DIM,
        )
        assert not torch.isnan(new_values).any()
        assert stats["n_selected"] == 1

    def test_empty_selection(self):
        keys, values, queries = _random_kv_queries(seed=3)
        selected = torch.tensor([], dtype=torch.long)

        new_values, stats = sparse_am_value_update(
            keys, values, queries, selected, head_dim=HEAD_DIM,
        )
        assert torch.allclose(new_values, values)
        assert stats["n_selected"] == 0

    def test_only_nonselected_unchanged(self):
        keys, values, queries = _random_kv_queries(seed=4)
        selected = torch.tensor([0, 3, 7])

        new_values, _ = sparse_am_value_update(
            keys, values, queries, selected, head_dim=HEAD_DIM,
        )

        non_sel = torch.ones(N_TOKENS, dtype=torch.bool)
        non_sel[selected] = False
        assert torch.allclose(new_values[non_sel], values[non_sel])

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    def test_dtype_policy(self, dtype):
        if dtype == torch.bfloat16 and not torch.cuda.is_available():
            pytest.skip("bfloat16 test needs CUDA")
        keys, values, queries = _random_kv_queries(seed=5)
        keys = keys.to(dtype)
        values = values.to(dtype)
        queries = queries.to(dtype)
        selected = torch.arange(TOP_T)

        new_values, stats = sparse_am_value_update(
            keys, values, queries, selected, head_dim=HEAD_DIM, ridge_lambda=1e-4,
        )
        assert new_values.dtype == dtype
        assert not torch.isnan(new_values).any()
        assert stats["mse"] < 1.0

    def test_partial_selection_improves_teacher_match(self):
        keys, values, queries = _random_kv_queries(seed=6)
        teacher_values = values + torch.randn_like(values) * 0.3
        targets = compute_attention_output(queries, keys, teacher_values, HEAD_DIM)
        selected = torch.arange(TOP_T)

        new_values, stats = sparse_am_value_update(
            keys, values, queries, selected,
            targets=targets, head_dim=HEAD_DIM,
        )

        baseline_mse = torch.mean(
            (compute_attention_output(queries, keys, values, HEAD_DIM) - targets) ** 2
        ).item()
        assert stats["mse"] < baseline_mse

    def test_sparse_update_with_external_teacher_targets(self):
        keys, values, queries = _random_kv_queries(seed=3)
        selected = torch.arange(TOP_T)
        teacher_values = torch.randn(N_TOKENS, HEAD_DIM)
        targets = compute_attention_output(queries, keys, teacher_values, HEAD_DIM)

        new_values, stats = sparse_am_value_update(
            keys, values, queries, selected,
            targets=targets,
            head_dim=HEAD_DIM,
            ridge_lambda=1e-6,
        )
        assert stats["mse"] is not None

    def test_guarded_update_with_teacher_targets_and_old_ref(self):
        keys, values, queries = _random_kv_queries(seed=11)
        old_queries = queries[:4]
        selected = torch.arange(TOP_T)
        targets = compute_attention_output(queries, keys, values, HEAD_DIM) + 0.1

        guarded_values, stats = guarded_sparse_am_value_update(
            keys, values, queries, selected,
            old_queries=old_queries,
            targets_new=targets,
            old_reference_weight=1.0,
            head_dim=HEAD_DIM,
            ridge_lambda=1e-6,
        )
        assert stats["mse_new"] is not None
        assert stats["n_queries_old"] == 4
        assert not torch.allclose(guarded_values, values)

    def test_guarded_update_trades_new_fit_for_old_preservation(self):
        keys, values, new_queries = _random_kv_queries(seed=8)
        old_queries = torch.randn_like(new_queries)
        selected = torch.arange(TOP_T)

        teacher_values = values.clone()
        teacher_values[selected] = teacher_values[selected] + 2.0
        targets_new = compute_attention_output(
            new_queries, keys, teacher_values, HEAD_DIM,
        )
        target_old = compute_attention_output(old_queries, keys, values, HEAD_DIM)

        unguarded_values, _ = guarded_sparse_am_value_update(
            keys, values, new_queries, selected,
            old_queries=old_queries,
            targets_new=targets_new,
            head_dim=HEAD_DIM,
            old_reference_weight=0.0,
            delta_weight=0.0,
        )
        guarded_values, _ = guarded_sparse_am_value_update(
            keys, values, new_queries, selected,
            old_queries=old_queries,
            targets_new=targets_new,
            head_dim=HEAD_DIM,
            old_reference_weight=10.0,
            delta_weight=0.0,
        )

        old_mse_unguarded = torch.mean(
            (compute_attention_output(old_queries, keys, unguarded_values, HEAD_DIM) - target_old) ** 2
        ).item()
        old_mse_guarded = torch.mean(
            (compute_attention_output(old_queries, keys, guarded_values, HEAD_DIM) - target_old) ** 2
        ).item()

        assert old_mse_guarded < old_mse_unguarded

    def test_delta_weight_keeps_solution_closer_to_original_values(self):
        keys, values, new_queries = _random_kv_queries(seed=9)
        selected = torch.arange(TOP_T)
        teacher_values = values.clone()
        teacher_values[selected] = teacher_values[selected] + 3.0
        targets_new = compute_attention_output(
            new_queries, keys, teacher_values, HEAD_DIM,
        )

        free_values, _ = guarded_sparse_am_value_update(
            keys, values, new_queries, selected,
            targets_new=targets_new,
            head_dim=HEAD_DIM,
            delta_weight=0.0,
        )
        regularized_values, _ = guarded_sparse_am_value_update(
            keys, values, new_queries, selected,
            targets_new=targets_new,
            head_dim=HEAD_DIM,
            delta_weight=100.0,
        )

        free_delta = (free_values[selected] - values[selected]).norm().item()
        reg_delta = (regularized_values[selected] - values[selected]).norm().item()
        assert reg_delta < free_delta


class TestAMTargetAccumulator:
    def test_layer_head_targets_match_query_flatten_order(self):
        accumulator = AMTargetAccumulator(n_layers=1, n_kv_heads=2)
        target = torch.arange(1 * 4 * 3 * 5).view(1, 4, 3, 5)

        accumulator.accumulate_from_hooks({0: target})

        head_1 = accumulator.get_layer_head_targets(
            layer_idx=0,
            head_idx=1,
            n_q_heads=4,
            n_kv_heads=2,
        )

        expected = target[0, 2:4].reshape(-1, 5)
        assert torch.equal(head_1, expected)
