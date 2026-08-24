"""Phase 1: building a cartridge from scratch with Attention Matching.

``compaction`` is the only Phase-1 construction: select teacher keys, ridge-fit
values. The older KVFromText self-match refine path is deprecated and is not
exported.

Sibling of ``am.continual`` with no edge between them: both reach down into
``am.core`` and ``am.components`` and never across.
"""

from __future__ import annotations

from cartridges.am.initial.compaction import compact_cache_am_phase1

__all__ = ["compact_cache_am_phase1"]
