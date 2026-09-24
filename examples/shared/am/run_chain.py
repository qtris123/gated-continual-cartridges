#!/usr/bin/env python3
"""Chain one cartridge through phases 2-5 and report the 5x5 stage-by-eval grid.

`run_chain` is the generic continual runner: it takes *any* p01 cartridge on
*any* supported dataset and runs the whole continual tail, evaluating on all five
held-out phase sets after every stage. Five stages by five evals is the grid: row
`pNN` is the cartridge after stage NN, so the diagonal is acquisition and
everything left of it is retention.

One runner owns the shared parts -- the stage loop, the per-stage 5x5 eval, and
the `matrix.json`/`matrix.csv` emission -- and dispatches only the per-stage
*step* by `--method`:

  - `am`       -> examples/shared/am/continual_write.py (closed-form AM write)
  - `gradient` -> examples/shared/am/grad_step.py (optimizer; density is a knob)

Density (dense/sparse) and top_t are recipe knobs inside whichever engine, not
separate methods. The write recipe is read off a saved `config.yaml` rather than
transcribed, so a lineage provably uses the same write rule as the run that
produced it. Each stage is guarded by a marker holding the cache path and all
five metrics, so a rerun resumes at the first incomplete stage.

Usage:
    python examples/shared/am/run_chain.py \\
        --dataset quality --method am --p01-cache /path/to/cache.pt --tag my-lineage
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml

from examples.shared.am.continual_env import (
    base_env,
    phase_eval_env,
    spec,
    stage_write_env,
)
from examples.shared.paths import ROOT

PROTOCOL = "teacher-forced-logppl-v1"

# The per-stage step engines dispatched on --method. Previously imported from
# qasper_five_phase.py; inlined here so run_chain owns its own dependencies.
PYTHON = Path(os.environ.get("CARTRIDGES_PYTHON") or sys.executable)
AM_STEP = ROOT / "examples/shared/am/continual_write.py"
GRAD_STEP = ROOT / "examples/shared/am/grad_step.py"
EVALUATOR = ROOT / "examples/shared/evaluate/cartridge_perplexity.py"

# Defaults match the directories each dataset's existing runs already use, so a
# rerun resumes from work already on disk instead of starting over.
RUN_DIRS = {
    "qasper": (ROOT / "outputs/qasper_asr_kg_runs", ROOT / "outputs/qasper_asr_kg_state"),
    "quality": (
        ROOT / "outputs/quality_5phase_runs",
        ROOT / "outputs/quality_5phase_state",
    ),
    "finqa": (
        ROOT / "outputs/finqa_5phase_runs",
        ROOT / "outputs/finqa_5phase_state",
    ),
    "techqa": (
        ROOT / "outputs/techqa_5phase_runs",
        ROOT / "outputs/techqa_5phase_state",
    ),
    "longhealth": (
        ROOT / "outputs/longhealth_5phase_runs",
        ROOT / "outputs/longhealth_5phase_state",
    ),
}
# The ranked-best of the 24 completed QASPER p03 lineages (mean p01-p03 loss
# 2.238): delta objective, highest-attention keys with MECH-005 repositioning,
# beta off. Reused across datasets so the write rule is held constant.
DEFAULT_RECIPE = ROOT / "outputs/caches/qasper/p03/delta_ha_b0_idf0/run/config.yaml"


def valid_marker(path: Path, expected_metrics: set[str]) -> str | None:
    """A marker is valid iff its cache exists and all expected metrics are present.

    Inlined from qasper_five_phase.py so run_chain has no dependency on it.
    """
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    cache = data.get("cache_path")
    metrics = set(data.get("eval_metrics", {}))
    if cache and Path(cache).exists() and expected_metrics <= metrics:
        return cache
    return None


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


def _single_phase_eval(
    cache: str, task: str, eval_path: Path, gpu: str, runs_dir: Path, out_json: Path
) -> dict:
    """Evaluate one cartridge on ONE phase eval set, pinned to a single GPU."""
    env = base_env(gpu, runs_dir)  # sets CUDA_VISIBLE_DEVICES and clears EVAL_*
    env["CACHE_PATH"] = str(cache)
    env["EVAL_P1_PATH"] = str(eval_path)
    env["EVAL_P1_NAME"] = task
    env["OUT_JSON"] = str(out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([str(PYTHON), str(EVALUATOR)], env=env, cwd=ROOT, check=True)
    payload = json.loads(out_json.read_text())
    metric = payload.get("eval_metrics", {}).get(task)
    if metric is None:
        raise RuntimeError(f"phase eval {task!r} produced no metric ({out_json})")
    return metric


def evaluate_cache(
    cache: str,
    marker: Path,
    eval_gpus: list[str],
    dataset_spec,
    metrics,
    runs_dir: Path,
) -> dict:
    """Score one cartridge on all five phase evals, fanned across ``eval_gpus``."""
    if valid_marker(marker, metrics):
        return json.loads(marker.read_text())
    marker.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(f"{marker.with_suffix('')}_evals")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tasks_paths = [
        (dataset_spec.task_names[phase], dataset_spec.eval_path(phase))
        for phase in range(1, 6)
    ]
    eval_metrics: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=len(eval_gpus)) as pool:
        futures = {}
        for idx, (task, eval_path) in enumerate(tasks_paths):
            gpu = eval_gpus[idx % len(eval_gpus)]
            futures[
                pool.submit(
                    _single_phase_eval,
                    cache,
                    task,
                    eval_path,
                    gpu,
                    runs_dir,
                    tmp_dir / f"{task}.json",
                )
            ] = task
        for future in as_completed(futures):
            eval_metrics[futures[future]] = future.result()
    if not metrics <= set(eval_metrics):
        raise RuntimeError(
            f"Incomplete fan-out evaluation for {marker}: {sorted(eval_metrics)}"
        )
    payload = {"cache_path": str(Path(cache).resolve()), "eval_metrics": eval_metrics}
    marker.write_text(json.dumps(payload, indent=2))
    return payload


def _grad_stage_env(
    recipe_path: str,
    *,
    dataset: str,
    phase: int,
    input_cache: str,
    run_name: str,
    gpu: str,
    runs_dir: Path,
) -> dict[str, str]:
    """Runtime environment for one gradient continual step (consumed by grad_step.py).

    Mirrors ``continual_env.stage_write_env`` but for the gradient engine: the
    write rule is passed by pointer (``RECIPE_CONFIG``); grad_step reads the
    ``gradient:`` section (density / lr / epochs / num_gpus) from that file.
    """
    env = phase_eval_env(base_env(gpu, runs_dir), dataset)
    env.update(spec(dataset).teacher_env(phase))
    env["RECIPE_CONFIG"] = str(recipe_path)
    env["PHASE1_CACHE_PATH"] = str(input_cache)
    env["SYNTH_DATA_PATH"] = str(spec(dataset).synth_path(phase))
    env["RUN_NAME"] = run_name
    return env


def run_stage(
    recipe_path: str,
    tag: str,
    dataset: str,
    phase: int,
    input_cache: str,
    gpu: str,
    eval_gpus: list[str],
    dataset_spec,
    metrics,
    runs_dir: Path,
    state_dir: Path,
    method: str,
) -> tuple[str, dict]:
    """Apply the stage-`phase` write (AM or gradient) on one GPU, then fan the 5 evals out."""
    marker = state_dir / tag / f"{stage_id(phase)}.json"
    if valid_marker(marker, metrics):
        payload = json.loads(marker.read_text())
        return payload["cache_path"], payload

    run_name = f"{dataset.upper()}_P{phase}_{tag}"
    if method == "am":
        env = stage_write_env(
            recipe_path,
            dataset=dataset,
            phase=phase,
            input_cache=input_cache,
            run_name=run_name,
            gpu=gpu,
            runs_dir=runs_dir,
        )
        step = AM_STEP
    else:  # gradient
        env = _grad_stage_env(
            recipe_path,
            dataset=dataset,
            phase=phase,
            input_cache=input_cache,
            run_name=run_name,
            gpu=gpu,
            runs_dir=runs_dir,
        )
        step = GRAD_STEP

    # The write is the artifact; per-document snapshots cost ~3.5 GiB a stage.
    env["SAVE_AFTER_EACH_DOCUMENT"] = "0"
    # Train-only: drop the phase evals so the write GPU does not score inline;
    # the five evals are fanned across eval_gpus once the cache exists.
    for key in [name for name in env if name.startswith("EVAL_P")]:
        env.pop(key)

    # For the gradient engine, grad_step reports its cache via a pointer file
    # (its run-dir naming differs from the AM glob contract).
    step_result = state_dir / tag / f"{stage_id(phase)}_step.json"
    if method == "gradient":
        env["STEP_RESULT_JSON"] = str(step_result)

    started = time.time()
    subprocess.run([str(PYTHON), str(step)], env=env, cwd=ROOT, check=True)

    if method == "am":
        summary_path = find_completed_run(runs_dir, run_name, started)
        cache = summary_path.parent / "cache_last.pt"
        if not cache.exists():
            raise RuntimeError(f"P{phase} run produced no cache_last.pt: {summary_path}")
    else:
        summary = json.loads(step_result.read_text())
        cache = Path(summary["cache_path"])
        summary_path = Path(summary.get("run_dir", cache.parent)) / "phase2_summary.json"
        if not cache.exists():
            raise RuntimeError(f"P{phase} gradient step produced no cache: {cache}")

    payload = evaluate_cache(
        str(cache.resolve()), marker, eval_gpus, dataset_spec, metrics, runs_dir
    )
    payload["method_tag"] = tag
    payload["method"] = method
    payload["input_cache"] = str(input_cache)
    payload["write_summary"] = str(summary_path)
    marker.write_text(json.dumps(payload, indent=2))
    return payload["cache_path"], payload


def continual_metrics(
    rows: dict[str, dict[str, dict]],
    tasks: list[str],
    acquired_at: dict[str, str],
    stage_labels: dict[str, str],
) -> dict:
    """Average-on-seen-tasks and end-of-stream forgetting for one 5x5 matrix.

    Inlined from the (removed) collect_qasper_continual_matrix.py; parameterized
    over the task stream so it scores any dataset's five-phase matrix.
    """
    def loss(stage: str, task: str):
        cell = rows.get(stage, {}).get(task)
        return cell["loss"] if cell else None

    stages = list(stage_labels)
    seen_average = {}
    for index, stage in enumerate(stages):
        seen = [task for task in tasks if stages.index(acquired_at[task]) <= index]
        values = [loss(stage, task) for task in seen]
        seen_average[stage] = sum(values) / len(values) if values and all(values) else None

    final = stages[-1]
    forgetting = {}
    for task in tasks:
        learned_at = acquired_at[task]
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
    parser.add_argument(
        "--dataset",
        default="qasper",
        choices=("qasper", "quality", "finqa", "techqa", "longhealth"),
    )
    parser.add_argument(
        "--method",
        default="am",
        choices=("am", "gradient"),
        help="per-stage step engine; density (dense/sparse) is a recipe knob",
    )
    parser.add_argument("--p01-cache", required=True, help="starting cartridge")
    parser.add_argument("--tag", required=True, help="lineage name for state/runs")
    parser.add_argument("--recipe-config", default=str(DEFAULT_RECIPE))
    parser.add_argument("--gpu", default="0", help="single GPU for the writes")
    parser.add_argument(
        "--eval-gpus",
        default="0,1,2,3",
        help="comma-separated GPUs to fan the per-stage evals across",
    )
    parser.add_argument("--rope-theta", default=None, help="override recipe rope_theta")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--runs-dir", default=None)
    parser.add_argument("--state-dir", default=None)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()

    default_runs, default_state = RUN_DIRS[args.dataset]
    runs_dir = Path(args.runs_dir) if args.runs_dir else default_runs
    state_dir = Path(args.state_dir) if args.state_dir else default_state
    eval_gpus = [gpu.strip() for gpu in args.eval_gpus.split(",") if gpu.strip()]
    if not eval_gpus:
        raise SystemExit("--eval-gpus must name at least one GPU")

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
    step = AM_STEP if args.method == "am" else GRAD_STEP
    required = [
        PYTHON,
        step,
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
    print(f"method       {args.method}")
    print(f"lineage      {args.tag}")
    print(f"p01 cache    {p01_cache}")
    print(f"recipe       {recipe_path}")
    if args.method == "am":
        print(
            "recipe knobs "
            f"key_mode={cfg['keys']['key_mode']} "
            f"reposition={cfg['keys']['key_reposition']} "
            f"beta={cfg['beta']['enabled']} top_t={cfg['slots']['top_t']} "
            f"ridge={cfg['objective']['ridge_lambda']} "
            f"delta={cfg['objective']['delta_weight']} "
            f"rope_theta={cfg['rope_theta']}"
        )
    else:
        grad = cfg.get("gradient", {}) if isinstance(cfg, dict) else {}
        print(
            "recipe knobs "
            f"density={grad.get('density', 'dense')} "
            f"lr={grad.get('lr', 'default')} epochs={grad.get('epochs', 'default')}"
        )
    print(f"runs/state   {runs_dir} | {state_dir}")
    print(f"write gpu    {args.gpu}   eval fan-out gpus {eval_gpus}")
    if args.preflight:
        print("Preflight OK")
        return

    runs_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / args.tag).mkdir(parents=True, exist_ok=True)

    # Freeze the resolved recipe (with any --rope-theta override) to a file and
    # pass it by pointer: the step engines read RECIPE_CONFIG, not env knobs.
    resolved_recipe = state_dir / args.tag / "resolved_recipe.yaml"
    resolved_recipe.write_text(yaml.safe_dump(cfg, sort_keys=True))
    recipe_pointer = str(resolved_recipe)

    rows: dict[str, dict[str, dict]] = {}
    caches = {"p01": str(p01_cache)}

    marker = state_dir / args.tag / "p01.json"
    print(
        f"\n=== p01: evaluating the starting cartridge on all 5 evals "
        f"(fan-out gpus {eval_gpus}) ===",
        flush=True,
    )
    payload = evaluate_cache(
        str(p01_cache), marker, eval_gpus, dataset_spec, metrics, runs_dir
    )
    rows["p01"] = _row(payload, marker, metrics)

    cache = str(p01_cache)
    for phase in (2, 3, 4, 5):
        stage = stage_id(phase)
        print(
            f"\n=== {stage}: {args.method} write on gpu {args.gpu} + 5 evals fanned "
            f"across {eval_gpus} ===",
            flush=True,
        )
        cache, payload = run_stage(
            recipe_pointer,
            args.tag,
            args.dataset,
            phase,
            cache,
            args.gpu,
            eval_gpus,
            dataset_spec,
            metrics,
            runs_dir,
            state_dir,
            args.method,
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
        "method": args.method,
        "technique": args.tag,
        "chain": f"{p01_cache.name} p01 -> continual {args.method} p02-p05",
        "protocol": PROTOCOL,
        "metric": "teacher-forced mean token loss (natural log)",
        "p01_cache": str(p01_cache),
        "recipe_config": str(recipe_path),
        "stage_caches": caches,
        "task_order": tasks,
        "acquired_at": acquired_at,
        "stage_labels": stage_labels,
        "cells": rows,
        "continual_metrics": continual_metrics(rows, tasks, acquired_at, stage_labels),
    }
    if args.method == "am":
        result["recipe"] = {
            "key_mode": cfg["keys"]["key_mode"],
            "key_reposition": cfg["keys"]["key_reposition"],
            "beta_enabled": cfg["beta"]["enabled"],
            "top_t": cfg["slots"]["top_t"],
            "ridge_lambda": cfg["objective"]["ridge_lambda"],
            "delta_weight": cfg["objective"]["delta_weight"],
            "rope_theta": cfg["rope_theta"],
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
