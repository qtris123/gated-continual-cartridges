"""Phase 2: writing new documents into an existing cartridge, one at a time.

``config`` declares the run, ``run`` drives the document loop, ``write`` applies
one document's closed-form update. Sibling of ``am.initial`` with no edge
between them: both reach down into ``am.core`` and ``am.components``.
"""

from __future__ import annotations

from cartridges.am.continual.config import AMContinualConfig, AMStages
from cartridges.am.continual.run import run_am_continual
from cartridges.am.continual.write import (
    AMUpdateStats,
    apply_document_am_write_to_cache,
)

__all__ = [
    "AMContinualConfig",
    "AMStages",
    "AMUpdateStats",
    "apply_document_am_write_to_cache",
    "run_am_continual",
]
