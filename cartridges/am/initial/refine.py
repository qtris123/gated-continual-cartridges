"""Deprecated Phase-1 variant: KVFromText self-match refine.

Do not call. Classic AM construction lives in ``initial.compaction``.
"""

from __future__ import annotations


def refine_cache_am_phase1(*args, **kwargs):
    raise RuntimeError(
        "refine_cache_am_phase1 is deprecated. "
        "Use compact_cache_am_phase1 from cartridges.am.initial.compaction."
    )
