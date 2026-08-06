"""Attention Matching: closed-form, backprop-free cartridge writes.

Layout::

    core          attention math, RoPE re-basing, the ridge solve
    components/   the six stages of a write (slots, queries, teacher,
                  keys, beta, objective), each a stateless class with an
                  inner ObjectConfig
    initial/      Phase 1 -- build a cartridge from scratch
    continual/    Phase 2 -- write new documents into an existing one
    diagnostics   read-only probes used by sweeps

``initial`` and ``continual`` are siblings with no edge between them: both
depend on ``core`` and ``components``, neither on the other.

This module exports the public surface only. Reach into the submodules for the
leaf math -- e.g. ``from cartridges.am.core import compute_attention_weights``.
"""

from __future__ import annotations

from cartridges.am.components import (
    BetaFitter,
    KeyWriter,
    ReferenceQueries,
    SlotSelector,
    TeacherTarget,
    ValueObjective,
)
from cartridges.am.continual import AMContinualConfig, AMStages, run_am_continual
from cartridges.am.initial import compact_cache_am_phase1, refine_cache_am_phase1

__all__ = [
    # the six stages
    "BetaFitter",
    "KeyWriter",
    "ReferenceQueries",
    "SlotSelector",
    "TeacherTarget",
    "ValueObjective",
    # phase 2
    "AMContinualConfig",
    "AMStages",
    "run_am_continual",
    # phase 1
    "compact_cache_am_phase1",
    "refine_cache_am_phase1",
]
