"""Config semantics for the Phase 2 AM write."""

from __future__ import annotations

import torch

from cartridges.am.components.beta import BetaFitter
from cartridges.am.components.queries import AMQueryAccumulator


def _fitter(enabled) -> BetaFitter:
    return BetaFitter.Config(enabled=enabled).instantiate()


def test_should_fit_beta_auto_is_off_for_key_rewrite():
    # B-SOLVE: beta is an INDEPENDENT axis from `key_mode`. The historical
    # fallback (`key_mode != "freeze"`) silently switched the NNLS path on for
    # every key experiment and made "beta with frozen keys" -- what B-ROUTE
    # needs -- inexpressible. Unset now means OFF whatever the keys are doing.
    assert _fitter(None).should_fit(key_mode="omp") is False


def test_should_fit_beta_respects_explicit_disable():
    assert _fitter(False).should_fit(key_mode="omp") is False


def test_should_fit_beta_freeze_requires_explicit_enable():
    assert _fitter(None).should_fit(key_mode="freeze") is False
    assert _fitter(True).should_fit(key_mode="freeze") is True


def test_beta_enabled_is_independent_of_key_mode():
    for key_mode in ("freeze", "highest_attention", "omp"):
        assert _fitter(True).should_fit(key_mode=key_mode) is True
        assert _fitter(False).should_fit(key_mode=key_mode) is False


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
