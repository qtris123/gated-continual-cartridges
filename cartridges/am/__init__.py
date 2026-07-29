"""Modular Attention Matching package.

Re-exports the full public API previously spread across
``cartridges.attention_matching``, ``cartridges.attention_matching_finetuning``,
``cartridges.am_teacher``, ``cartridges.am_continual`` and
``cartridges.am_reference_data``. Those top-level modules remain importable as
thin backward-compatible shims.
"""

from __future__ import annotations

# --- core attention math + ridge solver ---------------------------------
from cartridges.am.core import (
    _apply_rope_offset_to_queries,
    _attention_scores,
    _inv_sqrt_d,
    _ridge_lstsq,
    compute_attention_output,
    compute_attention_weights,
    effective_ridge_lambda,
)

# --- value-only solves ---------------------------------------------------
from cartridges.am.value_solve import (
    evaluate_attention_reconstruction,
    guarded_sparse_am_value_update,
    refine_kv_head_values,
    sparse_am_value_update,
)

# --- key selection / beta / key rewrite ----------------------------------
from cartridges.am.key_select import (
    nnls_projected_gradient,
    refit_beta_nnls,
    rewrite_keys_on_support,
    select_keys_highest_attention,
    select_keys_omp,
)

# --- compaction building blocks ------------------------------------------
from cartridges.am.compaction import (
    compact_kv_head,
    compute_compaction_c2,
    naive_compaction_c2_update,
)

# --- query/target accumulators + capture hooks ---------------------------
from cartridges.am.query_accum import (
    AMQueryAccumulator,
    AMTargetAccumulator,
    install_query_capture_hooks,
    install_teacher_attention_capture_hooks,
)

# --- slot ranking --------------------------------------------------------
from cartridges.am.ranking import (
    SLOT_PRIOR_SELECTIONS,
    _rank_attention_mass_per_layer,
    _rank_residual_budget_per_layer,
    _rank_slot_prior_per_layer,
    compute_slot_redundancy,
    load_slot_fisher_scores,
    rank_am_slots,
)

# --- finetune config + cache updates -------------------------------------
from cartridges.am.finetune import (
    AMUpdateStats,
    AttentionMatchingFinetuningConfig,
    _collect_reference_queries,
    _should_fit_beta,
    apply_am_update_to_cache,
    apply_document_am_write_to_cache,
    run_decoupled_tfidf_am_update,
)

# --- teacher KV ----------------------------------------------------------
from cartridges.am.teacher import (
    compute_teacher_log_mass,
    compute_teacher_mass,
    compute_teacher_targets,
    concat_teacher_kv,
    prefill_document_kv_cache,
)

# --- reference data ------------------------------------------------------
from cartridges.am.reference_data import (
    build_reference_dataloader,
    canonical_document_prompt,
    cleanup_reference_parquet,
    document_key,
    group_conversations_by_document,
    group_conversations_by_system_prompt,
    limit_conversations,
    load_conversations,
    load_old_reference_bank,
)

# --- phase 1 -------------------------------------------------------------
from cartridges.am.phase1 import compact_cache_am_phase1, refine_cache_am_phase1

# --- phase 2 orchestration ----------------------------------------------
from cartridges.am.continual import run_per_document_am_phase2

__all__ = [
    # core
    "_apply_rope_offset_to_queries",
    "_attention_scores",
    "_inv_sqrt_d",
    "_ridge_lstsq",
    "compute_attention_output",
    "compute_attention_weights",
    "effective_ridge_lambda",
    # value solves
    "evaluate_attention_reconstruction",
    "guarded_sparse_am_value_update",
    "refine_kv_head_values",
    "sparse_am_value_update",
    # key select
    "nnls_projected_gradient",
    "refit_beta_nnls",
    "rewrite_keys_on_support",
    "select_keys_highest_attention",
    "select_keys_omp",
    # compaction
    "compact_kv_head",
    "compute_compaction_c2",
    "naive_compaction_c2_update",
    # query accum
    "AMQueryAccumulator",
    "AMTargetAccumulator",
    "install_query_capture_hooks",
    "install_teacher_attention_capture_hooks",
    # ranking
    "SLOT_PRIOR_SELECTIONS",
    "_rank_attention_mass_per_layer",
    "_rank_residual_budget_per_layer",
    "_rank_slot_prior_per_layer",
    "compute_slot_redundancy",
    "load_slot_fisher_scores",
    "rank_am_slots",
    # finetune
    "AMUpdateStats",
    "AttentionMatchingFinetuningConfig",
    "_collect_reference_queries",
    "_should_fit_beta",
    "apply_am_update_to_cache",
    "apply_document_am_write_to_cache",
    "run_decoupled_tfidf_am_update",
    # teacher
    "compute_teacher_log_mass",
    "compute_teacher_mass",
    "compute_teacher_targets",
    "concat_teacher_kv",
    "prefill_document_kv_cache",
    # reference data
    "build_reference_dataloader",
    "canonical_document_prompt",
    "cleanup_reference_parquet",
    "document_key",
    "group_conversations_by_document",
    "group_conversations_by_system_prompt",
    "limit_conversations",
    "load_conversations",
    "load_old_reference_bank",
    # phase 1
    "compact_cache_am_phase1",
    "refine_cache_am_phase1",
    # phase 2
    "run_per_document_am_phase2",
]
