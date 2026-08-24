"""Deprecated. Do not use.

Phase 1 AM is classic teacher-key compaction
(``examples/shared/am/initial_compaction.py``), not KVFromText + self-match
refine.
"""

raise SystemExit(
    "examples/shared/am/initial_fit.py is deprecated. "
    "Use examples/shared/am/initial_compaction.py "
    "(or examples/qasper/pipelines/train_initial_am_compaction.sh / "
    "examples/quality/pipelines/run_5phase_am.sh)."
)
