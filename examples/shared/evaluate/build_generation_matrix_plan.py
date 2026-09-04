#!/usr/bin/env python3
"""Build a phase-matrix evaluation plan from stage state JSON files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--technique", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--state-dir", required=True)
    parser.add_argument(
        "--state-template",
        default="p{phase}.json",
        help="Filename under --state-dir; must contain cache_path",
    )
    parser.add_argument("--eval-template", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--plan-out", required=True)
    parser.add_argument("--phases", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--answer-prime",
        default="",
        help=(
            "Optional text pre-filled as the start of the assistant answer (e.g. "
            "'Answer:'). Appended to every prompt after the generation prompt and "
            "prepended to the generated text before scoring. Empty = free generation."
        ),
    )
    args = parser.parse_args()

    state_dir = Path(args.state_dir).resolve()
    stages = []
    eval_sets = []
    for phase in args.phases:
        state_path = state_dir / args.state_template.format(phase=phase)
        state = json.loads(state_path.read_text())
        cache_path = Path(state["cache_path"]).resolve()
        if not cache_path.is_file():
            raise FileNotFoundError(cache_path)
        eval_path = Path(args.eval_template.format(phase=phase)).resolve()
        if not eval_path.is_file():
            raise FileNotFoundError(eval_path)
        phase_id = f"p{phase:02d}"
        stages.append(
            {
                "id": phase_id,
                "index": phase,
                "label": f"after phase {phase}",
                "cache_path": str(cache_path),
                "state_path": str(state_path.resolve()),
            }
        )
        eval_sets.append(
            {
                "id": phase_id,
                "phase": phase,
                "label": f"phase {phase}",
                "path": str(eval_path),
            }
        )

    plan = {
        "schema_version": 1,
        "dataset": args.dataset,
        "technique": args.technique,
        "protocol": "freeform-mc-options-v1",
        "model": args.model,
        "output_dir": str(Path(args.output_dir).resolve()),
        "decode": {
            "backend": "flex",
            "temperature": args.temperature,
            "max_new_tokens": args.max_new_tokens,
            "batch_size": args.batch_size,
            "num_shots": 0,
            "prompt_variant": "phase-parquet-verbatim",
            "thinking": False,
            "answer_prime": args.answer_prime,
        },
        "stages": stages,
        "eval_sets": eval_sets,
    }
    plan_out = Path(args.plan_out)
    plan_out.parent.mkdir(parents=True, exist_ok=True)
    plan_out.write_text(json.dumps(plan, indent=2) + "\n")
    print(plan_out.resolve())


if __name__ == "__main__":
    main()
