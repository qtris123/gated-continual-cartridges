"""Backward-compatible shim.

The Attention Matching solvers now live in the :mod:`cartridges.am` package.
This module re-exports the historical public API so existing imports keep working:

    from cartridges.attention_matching import guarded_sparse_am_value_update  # still works
"""

from __future__ import annotations

from cartridges.am.core import (
    _apply_rope_offset_to_queries,
    _attention_scores,
    _inv_sqrt_d,
    _ridge_lstsq,
    compute_attention_output,
    compute_attention_weights,
    effective_ridge_lambda,
)
from cartridges.am.value_solve import (
    evaluate_attention_reconstruction,
    guarded_sparse_am_value_update,
    refine_kv_head_values,
    sparse_am_value_update,
)
from cartridges.am.key_select import (
    nnls_projected_gradient,
    refit_beta_nnls,
    rewrite_keys_on_support,
    select_keys_highest_attention,
    select_keys_omp,
)
from cartridges.am.compaction import (
    compact_kv_head,
    compute_compaction_c2,
    naive_compaction_c2_update,
)

__all__ = [
    "_apply_rope_offset_to_queries",
    "_attention_scores",
    "_inv_sqrt_d",
    "_ridge_lstsq",
    "compute_attention_output",
    "compute_attention_weights",
    "effective_ridge_lambda",
    "evaluate_attention_reconstruction",
    "guarded_sparse_am_value_update",
    "refine_kv_head_values",
    "sparse_am_value_update",
    "nnls_projected_gradient",
    "refit_beta_nnls",
    "rewrite_keys_on_support",
    "select_keys_highest_attention",
    "select_keys_omp",
    "compact_kv_head",
    "compute_compaction_c2",
    "naive_compaction_c2_update",
]
