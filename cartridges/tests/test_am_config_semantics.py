"""Config semantics for Phase 2 AM finetuning."""

from __future__ import annotations

import torch

from cartridges.attention_matching_finetuning import (
    AMQueryAccumulator,
    AttentionMatchingFinetuningConfig,
    _should_fit_beta,
)


def test_should_fit_beta_auto_for_key_rewrite():
    config = AttentionMatchingFinetuningConfig(key_mode="omp", enable_beta=None)
    assert _should_fit_beta(config) is True


def test_should_fit_beta_respects_explicit_disable():
    config = AttentionMatchingFinetuningConfig(key_mode="omp", enable_beta=False)
    assert _should_fit_beta(config) is False


def test_should_fit_beta_freeze_requires_explicit_enable():
    config = AttentionMatchingFinetuningConfig(key_mode="freeze", enable_beta=None)
    assert _should_fit_beta(config) is False

    config = AttentionMatchingFinetuningConfig(key_mode="freeze", enable_beta=True)
    assert _should_fit_beta(config) is True


def test_per_head_query_access_scores_preserve_kv_heads():
    class Cache:
        trainable_keys = [torch.zeros(1, 2, 3, 2)]

    accumulator = AMQueryAccumulator(
        granularity="per_head",
        queries_per_batch="all_tokens",
        n_layers=1,
        n_kv_heads=2,
        device=torch.device("cpu"),
    )
    queries = torch.tensor(
        [[
            [[4.0, 0.0]], [[4.0, 0.0]],
            [[0.0, 4.0]], [[0.0, 4.0]],
        ]]
    )
    Cache.trainable_keys[0][0, 0] = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]
    )
    Cache.trainable_keys[0][0, 1] = torch.tensor(
        [[0.0, 1.0], [1.0, 0.0], [0.0, -1.0]]
    )

    accumulator.accumulate_from_hooks(
        {0: queries},
        Cache(),
        scaling=1.0,
        seq_ids=torch.tensor([0]),
    )
    scores = accumulator.get_access_scores()

    assert scores.shape == (1, 2, 3)
    assert scores[0, 0, 0] > scores[0, 0, 1]
    assert scores[0, 1, 0] > scores[0, 1, 1]
