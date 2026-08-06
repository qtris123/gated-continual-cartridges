"""The six stages of an Attention Matching write.

Each stage is a stateless class with an inner ``ObjectConfig``: constructed once
per run from the config, holding nothing but ``self.config``, and exposing one
primary verb. The leaf math each verb forwards to lives in the same module, so a
call site can be read end to end without leaving the file.

    slots      -- which cartridge slots may be overwritten
    queries    -- the reference queries Q the write is fitted against
    teacher    -- the target the write is fitted TO
    keys       -- which key vectors occupy the written slots
    beta       -- the per-key attention-mass bias
    objective  -- the closed-form value solve
"""

from __future__ import annotations

from cartridges.am.components.beta import BetaFitter
from cartridges.am.components.keys import KeyWriter
from cartridges.am.components.objective import ValueObjective
from cartridges.am.components.queries import ReferenceQueries
from cartridges.am.components.slots import SlotSelector
from cartridges.am.components.teacher import TeacherTarget

__all__ = [
    "BetaFitter",
    "KeyWriter",
    "ReferenceQueries",
    "SlotSelector",
    "TeacherTarget",
    "ValueObjective",
]
