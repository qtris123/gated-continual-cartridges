"""Backward-compatible shim.

AM finetuning now lives in the :mod:`cartridges.am` package (``finetune``,
``query_accum``, ``ranking``, ``phase1``). This module re-exports the historical
public API so existing imports keep working, e.g.::

    from cartridges.attention_matching_finetuning import (
        AttentionMatchingFinetuningConfig, refine_cache_am_phase1,
    )
"""

from __future__ import annotations

# Primary AM finetuning surface.
from cartridges.am.finetune import (
    AMUpdateStats,
    AttentionMatchingFinetuningConfig,
    _collect_reference_queries,
    _should_fit_beta,
    apply_am_update_to_cache,
    apply_document_am_write_to_cache,
    run_decoupled_tfidf_am_update,
)
from cartridges.am.query_accum import (
    AMQueryAccumulator,
    AMTargetAccumulator,
    install_teacher_attention_capture_hooks,
)
from cartridges.am.ranking import (
    _rank_attention_mass_per_layer,
    _rank_residual_budget_per_layer,
    rank_am_slots,
)
from cartridges.am.phase1 import compact_cache_am_phase1, refine_cache_am_phase1

# Historical re-exports (names that used to be importable from this module).
from cartridges.am.value_solve import (
    guarded_sparse_am_value_update,
    refine_kv_head_values,
    sparse_am_value_update,
)
from cartridges.am.key_select import refit_beta_nnls, rewrite_keys_on_support
from cartridges.am.teacher import (
    compute_teacher_log_mass,
    compute_teacher_targets,
    concat_teacher_kv,
)
from cartridges.models.attention import flex_attention_forward
from cartridges.sparse_cache_finetuning import (
    BackgroundAccessTracker,
    CacheAccessTracker,
    CacheTFIDFRanker,
    GradientMask,
    TFIDFRankingInfo,
    _apply_rotary_pos_emb,
    install_query_capture_hooks,
)

__all__ = [
    "AMUpdateStats",
    "AttentionMatchingFinetuningConfig",
    "AMQueryAccumulator",
    "AMTargetAccumulator",
    "install_teacher_attention_capture_hooks",
    "apply_am_update_to_cache",
    "apply_document_am_write_to_cache",
    "run_decoupled_tfidf_am_update",
    "_collect_reference_queries",
    "_should_fit_beta",
    "_rank_attention_mass_per_layer",
    "_rank_residual_budget_per_layer",
    "rank_am_slots",
    "compact_cache_am_phase1",
    "refine_cache_am_phase1",
    "guarded_sparse_am_value_update",
    "refine_kv_head_values",
    "sparse_am_value_update",
    "refit_beta_nnls",
    "rewrite_keys_on_support",
    "compute_teacher_log_mass",
    "compute_teacher_targets",
    "concat_teacher_kv",
    "flex_attention_forward",
    "BackgroundAccessTracker",
    "CacheAccessTracker",
    "CacheTFIDFRanker",
    "GradientMask",
    "TFIDFRankingInfo",
    "_apply_rotary_pos_emb",
    "install_query_capture_hooks",
]
