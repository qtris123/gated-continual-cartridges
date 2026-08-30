#!/usr/bin/env python3
"""Ask whether a continual-AM stage write is task-specific at all.

A stage-by-eval grid that descends everywhere is easy to misread as continual
learning with no forgetting. But if a stage improves the tasks it did *not* train
on by as much as the one it did, the write is not acquiring that stage's
knowledge -- it is making the cartridge generically better (or drifting toward
the teacher), and the diagonal is not measuring acquisition.

For each stage this reports the change on the target task against the mean change
on the other four. `specificity` is the difference: strongly negative means the
write really did buy something task-specific, near zero means it did not.

Usage:
    python examples/shared/evaluate/continual_specificity.py \\
        --matrix outputs/evaluations/quality/<tag>/teacher-forced-logppl-v1/matrix.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def report(matrix_path: Path) -> None:
    payload = json.loads(matrix_path.read_text())
    cells = payload["cells"]
    tasks = payload["task_order"]
    stages = sorted(cells)

    def loss(stage: str, task: str) -> float:
        return cells[stage][task]["loss"]

    print(f"=== {payload['dataset']} / {payload['technique']} ===")
    print(
        f"{'stage':<7}{'target':>8}{'target d':>10}"
        f"{'non-target mean d':>20}{'specificity':>13}"
    )
    rows = []
    for index, stage in enumerate(stages[1:], start=1):
        previous, target = stages[index - 1], tasks[index]
        target_delta = loss(stage, target) - loss(previous, target)
        others = [
            loss(stage, task) - loss(previous, task)
            for position, task in enumerate(tasks)
            if position != index
        ]
        other_delta = sum(others) / len(others)
        rows.append((stage, target, target_delta, other_delta))
        print(
            f"{stage:<7}{target.upper():>8}{target_delta:>+10.3f}"
            f"{other_delta:>+20.3f}{target_delta - other_delta:>+13.3f}"
        )

    mean_specificity = sum(t - o for _, _, t, o in rows) / len(rows)
    print(f"\nmean specificity: {mean_specificity:+.3f}")
    print(
        "near zero => the stage write is not task-specific; the grid is "
        "descending for reasons other than acquiring that stage's documents"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", required=True, nargs="+", help="matrix.json path(s)")
    args = parser.parse_args()
    for index, path in enumerate(args.matrix):
        if index:
            print()
        report(Path(path))


if __name__ == "__main__":
    main()
