#!/usr/bin/env python3
"""Collect the QASPER continual-AM stage-by-eval matrix from existing run state.

No inference is run. Every number is read from the artifact the producing run
wrote, and each cell records the file it came from so the matrix stays audit-
able. The chain is the self-distilled p01 cartridge followed by continual AM
writes p02-p05.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from examples.shared.paths import ROOT

PROTOCOL = "teacher-forced-logppl-v1"
TASKS = ["qa", "mt", "sa", "asr", "kg"]
STAGE_LABELS = {
    "p01": "after QA (self-distilled)",
    "p02": "after MT",
    "p03": "after SA",
    "p04": "after ASR",
    "p05": "after KG",
}
ACQUIRED_AT = {"qa": "p01", "mt": "p02", "sa": "p03", "asr": "p04", "kg": "p05"}

SWEEP = ROOT / "outputs/ablations/sweep_eval_p1_p2_sa_refs_20260820_033638/summary.json"
STATE = ROOT / "outputs/qasper_asr_kg_state"
CACHES = ROOT / "outputs/caches/qasper"

# Producing runs used different metric names for the same eval parquet.
SUMMARY_ALIASES = {
    "qa_forgetting": "qa",
    "mt_acquisition": "mt",
    "sa_acquisition": "sa",
}


def _cell(value: float, source: Path) -> dict[str, Any]:
    return {"loss": value, "source": str(source.relative_to(ROOT))}


def _from_state(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text())
    return {
        SUMMARY_ALIASES.get(name, name): _cell(metric["loss"], path)
        for name, metric in payload["eval_metrics"].items()
    }


def _from_sweep(tag: str) -> dict[str, dict[str, Any]]:
    for entry in json.loads(SWEEP.read_text()):
        if entry["tag"] == tag:
            return {
                task: _cell(entry[task], SWEEP)
                for task in TASKS
                if entry.get(task) is not None
            }
    raise KeyError(f"{tag} not found in {SWEEP}")


def collect(technique: str) -> dict[str, dict[str, dict[str, Any]]]:
    lineage = STATE / technique
    rows: dict[str, dict[str, dict[str, Any]]] = {}

    # p01 is shared by every lineage: the self-distilled cartridge.
    rows["p01"] = {
        **_from_sweep("P1_phase1_selfdistill_qwen512"),
        **_from_state(STATE / "baseline_p1_new.json"),
    }
    rows["p02"] = {
        **_from_sweep(f"P2_{technique}"),
        **_from_state(lineage / "baseline_p2_new.json"),
    }
    rows["p03"] = {
        **_from_state(CACHES / "p03" / technique / "run/phase2_summary.json"),
        **_from_state(lineage / "baseline_p3_new.json"),
    }
    rows["p04"] = _from_state(lineage / "p4.json")
    rows["p05"] = _from_state(lineage / "p5.json")
    return rows


def continual_metrics(rows: dict[str, dict[str, dict[str, Any]]]) -> dict[str, Any]:
    def loss(stage: str, task: str) -> float | None:
        cell = rows.get(stage, {}).get(task)
        return cell["loss"] if cell else None

    stages = list(STAGE_LABELS)
    seen_average = {}
    for index, stage in enumerate(stages):
        seen = [task for task in TASKS if stages.index(ACQUIRED_AT[task]) <= index]
        values = [loss(stage, task) for task in seen]
        seen_average[stage] = sum(values) / len(values) if all(values) else None

    final = stages[-1]
    forgetting = {}
    for task in TASKS:
        learned_at = ACQUIRED_AT[task]
        if learned_at == final:
            continue
        acquisition = loss(learned_at, task)
        retained = loss(final, task)
        if acquisition is not None and retained is not None:
            forgetting[task] = retained - acquisition

    return {
        "average_loss_on_seen_tasks": seen_average,
        "forgetting_at_final_stage": forgetting,
        "mean_forgetting": (
            sum(forgetting.values()) / len(forgetting) if forgetting else None
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--technique", default="both_ha_b0_idf0")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    rows = collect(args.technique)
    output_dir = Path(
        args.output_dir
        or ROOT / "outputs/evaluations/qasper" / f"am-{args.technique}" / PROTOCOL
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "dataset": "qasper",
        "technique": args.technique,
        "chain": "selfdistill-qwen512 p01 -> continual AM p02-p05",
        "protocol": PROTOCOL,
        "metric": "teacher-forced mean token loss (natural log)",
        "note": "collected from producing-run artifacts; no inference re-run",
        "task_order": TASKS,
        "acquired_at": ACQUIRED_AT,
        "stage_labels": STAGE_LABELS,
        "cells": rows,
        "continual_metrics": continual_metrics(rows),
    }
    (output_dir / "matrix.json").write_text(json.dumps(payload, indent=2) + "\n")

    with (output_dir / "matrix.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["stage", *TASKS])
        for stage in STAGE_LABELS:
            writer.writerow(
                [stage]
                + [
                    rows.get(stage, {}).get(task, {}).get("loss", "")
                    for task in TASKS
                ]
            )

    header = f"{'stage':<6}" + "".join(f"{task.upper():>8}" for task in TASKS)
    print(header)
    for stage in STAGE_LABELS:
        line = f"{stage:<6}"
        for task in TASKS:
            cell = rows.get(stage, {}).get(task)
            line += f"{cell['loss']:>8.3f}" if cell else f"{'-':>8}"
        print(line)
    print()
    print(json.dumps(payload["continual_metrics"], indent=2))
    print(f"\nwrote {output_dir}")


if __name__ == "__main__":
    main()
