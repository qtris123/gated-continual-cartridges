"""Backward-compatible shim; implementation lives in cartridges.am.teacher."""

from __future__ import annotations

from cartridges.am.teacher import (
    _tokenize_system_prompt,
    compute_teacher_log_mass,
    compute_teacher_mass,
    compute_teacher_targets,
    concat_teacher_kv,
    prefill_document_kv_cache,
)

__all__ = [
    "_tokenize_system_prompt",
    "compute_teacher_log_mass",
    "compute_teacher_mass",
    "compute_teacher_targets",
    "concat_teacher_kv",
    "prefill_document_kv_cache",
]
