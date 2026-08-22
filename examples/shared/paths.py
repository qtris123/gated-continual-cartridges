"""Stable repository paths for example workflow drivers."""

from __future__ import annotations

import os
from pathlib import Path


def repository_root() -> Path:
    """Return the project root without depending on this file's nesting depth."""
    configured = os.environ.get("CARTRIDGES_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    return Path(__file__).resolve().parents[2]


ROOT = repository_root()
OUTPUTS = Path(os.environ.get("CARTRIDGES_OUTPUT_DIR", ROOT / "outputs")).resolve()
