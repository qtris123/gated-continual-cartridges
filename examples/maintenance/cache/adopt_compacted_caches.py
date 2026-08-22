#!/usr/bin/env python3
"""Move original wandb run dirs into outputs/caches/<dataset>/<stage>/<technique>/run/."""

from __future__ import annotations

import json
from pathlib import Path

from examples.shared.evaluate.cache_layout import (  # noqa: E402
    CACHES_ROOT,
    relocate_originals,
)


def main() -> None:
    already_moved: dict[str, str] = {}
    sources = sorted(
        CACHES_ROOT.glob("*/*/*/source.json"),
        key=lambda path: (
            0 if path.parent.name == "selfdistill-qwen512" else 1,
            str(path),
        ),
    )
    for source_path in sources:
        source = json.loads(source_path.read_text())
        run_dir = source_path.parent / "run"
        if run_dir.exists():
            already_moved.setdefault(source["original_run_dir"], str(run_dir.resolve()))
    for source_path in sources:
        source = json.loads(source_path.read_text())
        dest_dir = source_path.parent
        original_cache = Path(source["original_cache_path"])
        original_run = Path(source["original_run_dir"])
        relocated = relocate_originals(
            dest_dir,
            original_cache,
            original_run,
            already_moved=already_moved,
        )
        source["run_dir"] = relocated["run_dir"]
        source_path.write_text(json.dumps(source, indent=2) + "\n")
        print(f"{dest_dir.relative_to(CACHES_ROOT)} -> {relocated['run_dir']}")
    print(f"relocated {len(sources)} cache folders")


if __name__ == "__main__":
    main()
