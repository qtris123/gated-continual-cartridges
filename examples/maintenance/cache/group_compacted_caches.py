#!/usr/bin/env python3
"""Group existing compacted KV caches under outputs/caches/<dataset>/<stage>/<technique>/."""

from __future__ import annotations

import json

from examples.shared.evaluate.cache_layout import (  # noqa: E402
    CACHES_ROOT,
    publish_cache,
    rewrite_state_cache_path,
)
from examples.shared.paths import ROOT as REPO

QUALITY_TECHNIQUE = "am-canonical-512"
QUALITY_STATE = REPO / "outputs" / "quality_5phase_state"
QASPER_STATE = REPO / "outputs" / "qasper_asr_kg_state"
QASPER_P1 = REPO / "outputs" / "phase1_selfdistill_qwen512" / "cache_last.pt"
QASPER_P1_TECHNIQUE = "selfdistill-qwen512"


def _index_entry(dataset: str, stage: str, technique: str, cache: Path) -> dict:
    return {
        "dataset": dataset,
        "stage": stage,
        "technique": technique,
        "cache_path": str(cache),
    }


def group_quality() -> list[dict]:
    entries = []
    for phase in range(1, 6):
        state_path = QUALITY_STATE / f"p{phase}.json"
        state = json.loads(state_path.read_text())
        dest = publish_cache(
            state["cache_path"],
            "quality",
            phase,
            QUALITY_TECHNIQUE,
            metadata={"run_name": state.get("run_name"), "state_path": str(state_path)},
        )
        rewrite_state_cache_path(state_path, dest)
        entries.append(
            _index_entry("quality", f"p{phase:02d}", QUALITY_TECHNIQUE, dest)
        )
    return entries


def group_qasper() -> list[dict]:
    entries = []
    inventory = json.loads((QASPER_STATE / "inventory.json").read_text())
    p1_dest = publish_cache(
        QASPER_P1,
        "qasper",
        1,
        QASPER_P1_TECHNIQUE,
        metadata={"state_path": str(QASPER_STATE / "baseline_p1_new.json")},
    )
    rewrite_state_cache_path(QASPER_STATE / "baseline_p1_new.json", p1_dest)
    entries.append(_index_entry("qasper", "p01", QASPER_P1_TECHNIQUE, p1_dest))

    tags = sorted({row["tag"] for row in inventory})
    by_tag = {row["tag"]: row for row in inventory}
    for tag in tags:
        dest = publish_cache(
            QASPER_P1,
            "qasper",
            1,
            tag,
            metadata={"shared_with": QASPER_P1_TECHNIQUE},
        )
        entries.append(_index_entry("qasper", "p01", tag, dest))

        row = by_tag[tag]
        for phase, key in ((2, "p2_cache"), (3, "p3_cache")):
            dest = publish_cache(
                row[key],
                "qasper",
                phase,
                tag,
                metadata={"inventory_key": key},
            )
            entries.append(_index_entry("qasper", f"p{phase:02d}", tag, dest))

        for phase in (4, 5):
            state_path = QASPER_STATE / tag / f"p{phase}.json"
            if not state_path.is_file():
                continue
            state = json.loads(state_path.read_text())
            dest = publish_cache(
                state["cache_path"],
                "qasper",
                phase,
                tag,
                metadata={"run_name": state.get("run_name"), "state_path": str(state_path)},
            )
            rewrite_state_cache_path(state_path, dest)
            entries.append(_index_entry("qasper", f"p{phase:02d}", tag, dest))
    return entries


def write_index(entries: list[dict]) -> Path:
    CACHES_ROOT.mkdir(parents=True, exist_ok=True)
    index_path = CACHES_ROOT / "index.json"
    payload = {
        "root": str(CACHES_ROOT),
        "layout": "dataset/stage/technique/cache.pt",
        "n_caches": len(entries),
        "caches": sorted(
            entries, key=lambda entry: (
                entry["dataset"],
                entry["stage"],
                entry["technique"],
            )
        ),
    }
    index_path.write_text(json.dumps(payload, indent=2) + "\n")
    readme = CACHES_ROOT / "README.md"
    readme.write_text(
        "# Compacted KV caches\n\n"
        "Layout: `outputs/caches/<dataset>/<stage>/<technique>/cache.pt`\n\n"
        "- `cache.pt` is a hardlink (or copy) of the training run checkpoint.\n"
        "- `source.json` records the original wandb-style run directory.\n"
        "- Training run dirs under `outputs/` are not deleted.\n"
        "- See `index.json` for the full inventory.\n"
    )
    return index_path


def main() -> None:
    entries = group_quality() + group_qasper()
    index_path = write_index(entries)
    print(f"published {len(entries)} caches")
    print(index_path)


if __name__ == "__main__":
    main()
