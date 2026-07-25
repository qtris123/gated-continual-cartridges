"""Backward-compatible shim; implementation lives in cartridges.am.reference_data."""

from __future__ import annotations

from cartridges.am.reference_data import (
    _extract_tag,
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

__all__ = [
    "_extract_tag",
    "build_reference_dataloader",
    "canonical_document_prompt",
    "cleanup_reference_parquet",
    "document_key",
    "group_conversations_by_document",
    "group_conversations_by_system_prompt",
    "limit_conversations",
    "load_conversations",
    "load_old_reference_bank",
]
