"""Phase 1: building a cartridge from scratch with Attention Matching.

``compaction`` is the classic construction (select teacher keys, ridge-fit
values); ``refine`` is the older self-match pass over an existing cartridge.
Sibling of ``am.continual`` with no edge between them: both reach down into
``am.core`` and ``am.components`` and never across.
"""

from __future__ import annotations

from cartridges.am.initial.compaction import compact_cache_am_phase1
from cartridges.am.initial.refine import refine_cache_am_phase1

__all__ = ["compact_cache_am_phase1", "refine_cache_am_phase1"]
