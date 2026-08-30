#!/usr/bin/env python3
"""Chain one cartridge through phases 2-5 and report the 5x5 stage-by-eval grid.

`qasper_five_phase.py` resumes 24 pre-existing p03 QASPER lineages through ASR
and KG only, from a hard-wired self-distilled p01. This driver instead takes
*any* p01 cartridge on *any* supported dataset and runs the whole continual tail,
evaluating on all five held-out phase sets after every stage. Five stages by five
evals is the grid: row `pNN` is the cartridge after stage NN, so the diagonal is
acquisition and everything left of it is retention.

The write recipe is read off a saved `config.yaml` rather than transcribed, and
the environment is built by the same `continual_env.stage_write_env` the QASPER
driver uses, so a cross-dataset comparison differs only in the data. Each stage
is guarded by a marker holding the cache path and all five metrics, so a rerun
resumes at the first incomplete stage.

Usage:
    python examples/shared/am/continual_chain.py \\
        --dataset quality --p01-cache /path/to/cache.pt --tag my-lineage
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import time
from pathlib import Path

import yaml

from examples.shared.am.continual_env import (
    base_env,
    phase_eval_env,
    spec,
    stage_write_env,
)
from examples.shared.am.qasper_five_phase import CONTINUAL, PYTHON, valid_marker
from examples.shared.evaluate.collect_qasper_continual_matrix import (
    PROTOCOL,
    continual_metrics,
)
from examples.shared.paths import ROOT

EVALUATOR = ROOT / "examples/shared/evaluate/cartridge_perplexity.py"
# Defaults match the directories each dataset's existing runs already use, so a
# rerun resumes from work already on disk instead of starting over.
RUN_DIRS = {
    "qasper": (ROOT / "outputs/qasper_asr_kg_runs", ROOT / "outputs/qasper_asr_kg_state"),
    "quality": (
        ROOT / "outputs/quality_5phase_runs",
        ROOT / "outputs/quality_5phase_state",
    ),
}
# The ranked-best of the 24 completed QASPER p03 lineages (mean p01-p03 loss
# 2.238): delta objective, highest-attention keys with MECH-005 repositioning,
# beta off. Reused across datasets so the write rule is held constant.
DEFAULT_RECIPE = ROOT / "outputs/caches/qasper/p03/delta_ha_b0_idf0/run/config.yaml"


def stage_id(phase: int) -> str:
    return f"p{phase:02d}"


def find_completed_run(runs_dir: Path, run_name: str, started: float) -> Path:
    """The summary this launch produced, matched by run name and mtime."""
    candidates = [
        summary
        for summary in runs_dir.glob(f"*-{run_name}/*/phase2_summary.json")
        if summary.stat().st_mtime >= started - 2
    ]
    if not candidates:
        raise RuntimeError(f"No completed summary found for {run_name} in {runs_dir}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def evaluate_cache(
    cache: str, marker: Path, gpu: str, dataset: str, metrics, runs_dir: Path
) -> dict:
    """Score one cartridge on all five phase evals, writing an audit marker."""
    if valid_marker(marker, metrics):
        return json.loads(marker.read_text())
    env = phase_eval_env(base_env(gpu, runs_dir), dataset)
    env["CACHE_PATH"] = str(cache)
    env["OUT_JSON"] = str(marker)
    marker.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([str(PYTHON), str(EVALUATOR)], env=env, cwd=ROOT, check=True)
    payload = json.loads(marker.read_text())
    if not metrics <= set(payload.get("eval_metrics", {})):
        marker.unlink(missing_ok=True)
        raise RuntimeError(f"Incomplete evaluation marker {marker}")
    return payload


def run_stage(
    cfg: dict,
    tag: str,
    dataset: str,
    phase: int,
    input_cache: str,
    gpu: str,
    metrics,
    runs_dir: Path,
    state_dir: Path,
) -> tuple[str, dict]:
    """Apply the stage-`phase` AM write, then score all five evals."""
    marker = state_dir / tag / f"{stage_id(phase)}.json"
    if valid_marker(marker, metrics):
        payload = json.loads(marker.read_text())
        return payload["cache_path"], payload

    run_name = f"{dataset.upper()}_P{phase}_{tag}"
    env = stage_write_env(
        cfg,
        dataset=dataset,
        phase=phase,
        input_cache=input_cache,
        run_name=run_name,
        gpu=gpu,
        runs_dir=runs_dir,
    )
    # The write is the artifact; per-document snapshots cost ~3.5 GiB a stage.
    env["SAVE_AFTER_EACH_DOCUMENT"] = "0"
    started = time.time()
    subprocess.run([str(PYTHON), str(CONTINUAL)], env=env, cwd=ROOT, check=True)

    summary_path = find_completed_run(runs_dir, run_name, started)
    payload = json.loads(summary_path.read_text())
    cache = summary_path.parent / "cache_last.pt"
    if not cache.exists():
        raise RuntimeError(f"P{phase} run produced no cache_last.pt: {summary_path}")
    if not metrics <= set(payload.get("eval_metrics", {})):
        raise RuntimeError(
            f"P{phase} run missing evals: got {sorted(payload.get('eval_metrics', {}))}"
        )
    payload["cache_path"] = str(cache.resolve())
    payload["method_tag"] = tag
    payload["input_cache"] = str(input_cache)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps(payload, indent=2))
    return payload["cache_path"], payload


def _row(payload: dict, marker: Path, metrics) -> dict[str, dict]:
    return {
        task: {"loss": metric["loss"], "source": str(marker)}
        for task, metric in payload["eval_metrics"].items()
        if task in metrics
    }


def render(rows: dict[str, dict[str, dict]], tasks: list[str], stages: list[str]) -> str:
    lines = [f"{'stage':<7}" + "".join(f"{task.upper():>9}" for task in tasks)]
    for stage in stages:
        line = f"{stage:<7}"
        for task in tasks:
            cell = rows.get(stage, {}).get(task)
            line += f"{cell['loss']:>9.3f}" if cell else f"{'-':>9}"
        lines.append(line)
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="qasper", choices=("qasper", "quality"))
    parser.add_argument("--p01-cache", required=True, help="starting cartridge")
    parser.add_argument("--tag", required=True, help="lineage name for state/runs")
    parser.add_argument("--recipe-config", default=str(DEFAULT_RECIPE))
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--rope-theta", default=None, help="override recipe rope_theta")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--runs-dir", default=None)
    parser.add_argument("--state-dir", default=None)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()

    default_runs, default_state = RUN_DIRS[args.dataset]
    runs_dir = Path(args.runs_dir) if args.runs_dir else default_runs
    state_dir = Path(args.state_dir) if args.state_dir else default_state

    dataset_spec = spec(args.dataset)
    tasks = [dataset_spec.task_names[phase] for phase in range(1, 6)]
    metrics = set(tasks)
    stages = [stage_id(phase) for phase in range(1, 6)]
    stage_labels = {
        stage_id(phase): f"after {dataset_spec.task_names[phase]}"
        for phase in range(1, 6)
    }
    acquired_at = {
        dataset_spec.task_names[phase]: stage_id(phase) for phase in range(1, 6)
    }

    p01_cache = Path(args.p01_cache).resolve()
    recipe_path = Path(args.recipe_config).resolve()
    required = [
        PYTHON,
        CONTINUAL,
        EVALUATOR,
        p01_cache,
        recipe_path,
        *(dataset_spec.eval_path(phase) for phase in range(1, 6)),
        *(dataset_spec.synth_path(phase) for phase in range(2, 6)),
    ]
    missing = [str(path) for path in required if not Path(path).exists()]
    if missing:
        raise FileNotFoundError(f"Missing required inputs: {missing}")

    cfg = yaml.safe_load(recipe_path.read_text())
    if args.rope_theta is not None:
        cfg["rope_theta"] = float(args.rope_theta)

    print(f"dataset      {args.dataset}")
    print(f"lineage      {args.tag}")
    print(f"p01 cache    {p01_cache}")
    print(f"recipe       {recipe_path}")
    print(
        "recipe knobs "
        f"key_mode={cfg['keys']['key_mode']} "
        f"reposition={cfg['keys']['key_reposition']} "
        f"beta={cfg['beta']['enabled']} top_t={cfg['slots']['top_t']} "
        f"ridge={cfg['objective']['ridge_lambda']} "
        f"delta={cfg['objective']['delta_weight']} "
        f"rope_theta={cfg['rope_theta']:g}"
    )
    print(f"runs/state   {runs_dir} | {state_dir}")
    if args.preflight:
        print("Preflight OK")
        return

    runs_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / args.tag).mkdir(parents=True, exist_ok=True)

    rows: dict[str, dict[str, dict]] = {}
    caches = {"p01": str(p01_cache)}

    marker = state_dir / args.tag / "p01.json"
    print("\n=== p01: evaluating the starting cartridge on all 5 evals ===", flush=True)
    payload = evaluate_cache(
        str(p01_cache), marker, args.gpu, args.dataset, metrics, runs_dir
    )
    rows["p01"] = _row(payload, marker, metrics)

    cache = str(p01_cache)
    for phase in (2, 3, 4, 5):
        stage = stage_id(phase)
        print(f"\n=== {stage}: AM write + 5 evals ===", flush=True)
        cache, payload = run_stage(
            cfg,
            args.tag,
            args.dataset,
            phase,
            cache,
            args.gpu,
            metrics,
            runs_dir,
            state_dir,
        )
        rows[stage] = _row(payload, state_dir / args.tag / f"{stage}.json", metrics)
        caches[stage] = cache
        print(render(rows, tasks, stages), flush=True)

    output_dir = Path(
        args.output_dir
        or ROOT / "outputs/evaluations" / args.dataset / args.tag / PROTOCOL
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "dataset": args.dataset,
        "technique": args.tag,
        "chain": f"{p01_cache.name} p01 -> continual AM p02-p05",
        "protocol": PROTOCOL,
        "metric": "teacher-forced mean token loss (natural log)",
        "p01_cache": str(p01_cache),
        "recipe_config": str(recipe_path),
        "recipe": {
            "key_mode": cfg["keys"]["key_mode"],
            "key_reposition": cfg["keys"]["key_reposition"],
            "beta_enabled": cfg["beta"]["enabled"],
            "top_t": cfg["slots"]["top_t"],
            "ridge_lambda": cfg["objective"]["ridge_lambda"],
            "delta_weight": cfg["objective"]["delta_weight"],
            "rope_theta": cfg["rope_theta"],
        },
        "stage_caches": caches,
        "task_order": tasks,
        "acquired_at": acquired_at,
        "stage_labels": stage_labels,
        "cells": rows,
        "continual_metrics": continual_metrics(rows, tasks, acquired_at, stage_labels),
    }
    (output_dir / "matrix.json").write_text(json.dumps(result, indent=2) + "\n")

    with (output_dir / "matrix.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["stage", *tasks])
        for stage in stages:
            writer.writerow(
                [stage]
                + [rows.get(stage, {}).get(task, {}).get("loss", "") for task in tasks]
            )

    print("\n" + render(rows, tasks, stages))
    print()
    print(json.dumps(result["continual_metrics"], indent=2))
    print(f"\nwrote {output_dir}")


if __name__ == "__main__":
    main()
