#!/usr/bin/env python3
"""Resume all completed QASPER P3 AM lineages through ASR and KG.

The inventory is derived from each saved P3 config rather than reconstructed
from folder names.  Each lineage keeps its exact write recipe.  Stage markers
are written only after a cache and all requested evaluation metrics exist.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml

from examples.shared.am.continual_env import (
    base_env as _base_env,
    phase_eval_env,
    spec,
    stage_write_env,
)
from examples.shared.evaluate.cache_layout import publish_cache
from examples.shared.paths import ROOT

OUTPUTS = ROOT / "outputs"
RUNS = OUTPUTS / "qasper_asr_kg_runs"
STATE = OUTPUTS / "qasper_asr_kg_state"
PYTHON = Path(os.environ.get("CARTRIDGES_PYTHON") or sys.executable)
CONTINUAL = ROOT / "examples/shared/am/continual_write.py"
EVALUATOR = ROOT / "examples/shared/evaluate/cartridge_perplexity.py"
P1_CACHE = (
    OUTPUTS / "caches" / "qasper" / "p01" / "selfdistill-qwen512" / "cache.pt"
)

SPEC = spec("qasper")
EVALS = {phase: SPEC.eval_path(phase) for phase in range(1, 6)}
SYNTH = {phase: SPEC.synth_path(phase) for phase in range(1, 6)}
TOPIC = dict(SPEC.topics)


def _method_tag(cfg: dict) -> str:
    objective = cfg["objective"]
    ridge = float(objective["ridge_lambda"])
    delta = float(objective["delta_weight"])
    mode = "both" if ridge and delta else "ridge" if ridge else "delta"
    key = cfg["keys"]["key_mode"].replace("highest_attention", "ha")
    beta = int(bool(cfg["beta"]["enabled"]))
    idf = int(bool(cfg["slots"]["use_idf"]))
    return f"{mode}_{key}_b{beta}_idf{idf}"


def discover_lineages() -> list[dict]:
    by_tag: dict[str, dict] = {}
    inventory_path = STATE / "inventory.json"
    if inventory_path.is_file():
        inventory = json.loads(inventory_path.read_text())
        candidates = [
            (
                Path(item["p3_config"]),
                Path(item["p2_cache"]),
                Path(item["p3_cache"]),
            )
            for item in inventory
        ]
    else:
        sa3_cfgs = list(OUTPUTS.glob("*-SA3_*/*/config.yaml"))
        sa3_cfgs += list((OUTPUTS / "ablations").glob("*-SA3_*/*/config.yaml"))
        candidates = [
            (cfg_path, None, cfg_path.parent / "cache_last.pt")
            for cfg_path in sa3_cfgs
            if (cfg_path.parent / "phase2_summary.json").is_file()
        ]

    for cfg_path, inventory_p2, cache in sorted(candidates):
        if not cfg_path.is_file() or not cache.is_file():
            continue
        cfg = yaml.safe_load(cfg_path.read_text())
        if cfg.get("teacher", {}).get("qasper_topic") != "SA":
            continue
        tag = _method_tag(cfg)
        p2 = inventory_p2
        if p2 is None:
            p2 = Path(cfg["kv_cache_initializer"]["path"])
            if not p2.is_absolute():
                p2 = ROOT / p2
        record = {
            "tag": tag,
            "p2_cache": str(p2),
            "p3_cache": str(cache),
            "p3_config": str(cfg_path),
            "config": cfg,
        }
        previous = by_tag.get(tag)
        if previous is None or str(cfg_path) > previous["p3_config"]:
            by_tag[tag] = record
    records = sorted(by_tag.values(), key=lambda item: item["tag"])
    if len(records) != 24:
        raise RuntimeError(
            f"Expected 24 unique completed P3 method lineages, found {len(records)}: "
            f"{[r['tag'] for r in records]}"
        )
    for record in records:
        for key in ("p2_cache", "p3_cache", "p3_config"):
            if not Path(record[key]).exists():
                raise FileNotFoundError(f"{record['tag']} missing {key}: {record[key]}")
    return records


def base_env(gpu: str) -> dict[str, str]:
    return _base_env(gpu, RUNS)


def eval_env(env: dict[str, str], phases=range(1, 6)) -> dict[str, str]:
    return phase_eval_env(env, "qasper", phases)


def valid_marker(path: Path, expected_metrics: set[str]) -> str | None:
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    cache = data.get("cache_path")
    metrics = set(data.get("eval_metrics", {}))
    if cache and Path(cache).exists() and expected_metrics <= metrics:
        return cache
    return None


def evaluate_cache(cache: str, marker: Path, gpu: str, phases) -> None:
    expected = {SPEC.task_names[phase] for phase in phases}
    if valid_marker(marker, expected):
        return
    env = eval_env(base_env(gpu), phases)
    env["CACHE_PATH"] = cache
    env["OUT_JSON"] = str(marker)
    marker.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([str(PYTHON), str(EVALUATOR)], env=env, cwd=ROOT, check=True)
    data = json.loads(marker.read_text())
    if not expected <= set(data.get("eval_metrics", {})):
        marker.unlink(missing_ok=True)
        raise RuntimeError(f"Incomplete evaluation marker {marker}")


def write_env(record: dict, input_cache: str, phase: int, gpu: str) -> dict[str, str]:
    return stage_write_env(
        record["config"],
        dataset="qasper",
        phase=phase,
        input_cache=input_cache,
        run_name=f"QASPER_P{phase}_{record['tag']}",
        gpu=gpu,
        runs_dir=RUNS,
    )


def find_completed_run(run_name: str, started: float) -> Path:
    candidates = []
    for summary in RUNS.glob(f"*-{run_name}/*/phase2_summary.json"):
        if summary.stat().st_mtime >= started - 2:
            candidates.append(summary)
    if not candidates:
        raise RuntimeError(f"No completed summary found for {run_name}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def run_write(record: dict, input_cache: str, phase: int, gpu: str) -> str:
    marker = STATE / record["tag"] / f"p{phase}.json"
    cached = valid_marker(marker, {"qa", "mt", "sa", "asr", "kg"})
    if cached:
        return cached
    env = write_env(record, input_cache, phase, gpu)
    run_name = env["RUN_NAME"]
    started = time.time()
    subprocess.run([str(PYTHON), str(CONTINUAL)], env=env, cwd=ROOT, check=True)
    summary_path = find_completed_run(run_name, started)
    summary = json.loads(summary_path.read_text())
    cache = str((summary_path.parent / "cache_last.pt").resolve())
    summary["cache_path"] = cache
    summary["source_p3_config"] = record["p3_config"]
    summary["method_tag"] = record["tag"]
    expected = {"qa", "mt", "sa", "asr", "kg"}
    if not Path(cache).exists() or not expected <= set(summary.get("eval_metrics", {})):
        raise RuntimeError(f"Incomplete P{phase} run: {summary_path}")
    published = publish_cache(
        cache,
        "qasper",
        phase,
        record["tag"],
        metadata={"run_name": run_name, "state_path": str(marker)},
    )
    summary["original_cache_path"] = cache
    summary["cache_path"] = str(published)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps(summary, indent=2))
    return str(published)


def run_lineage(record: dict, gpu: str) -> str:
    tag = record["tag"]
    log = STATE / tag / "worker.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a") as stream:
        original_stdout, original_stderr = sys.stdout, sys.stderr
        sys.stdout = sys.stderr = stream
        try:
            print(f"\n=== {time.ctime()} GPU {gpu} {tag} ===", flush=True)
            evaluate_cache(
                record["p2_cache"], STATE / tag / "baseline_p2_new.json", gpu, (4, 5)
            )
            evaluate_cache(
                record["p3_cache"], STATE / tag / "baseline_p3_new.json", gpu, (4, 5)
            )
            p4_cache = run_write(record, record["p3_cache"], 4, gpu)
            run_write(record, p4_cache, 5, gpu)
            print(f"=== completed {tag} ===", flush=True)
            return tag
        finally:
            sys.stdout, sys.stderr = original_stdout, original_stderr


def run_gpu_queue(gpu: str, records: list[dict]) -> dict[str, list]:
    completed = []
    failed = []
    for record in records:
        try:
            completed.append(run_lineage(record, gpu))
        except Exception as exc:
            failed.append((record["tag"], repr(exc)))
    return {"completed": completed, "failed": failed}


def preflight(records: list[dict]) -> None:
    required = [PYTHON, CONTINUAL, EVALUATOR, P1_CACHE, *EVALS.values(), *SYNTH.values()]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required inputs: {missing}")
    print(f"Preflight OK: {len(records)} P3 lineages, all synth/eval inputs present")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", default="0,1,2")
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    gpus = [gpu.strip() for gpu in args.gpus.split(",") if gpu.strip()]
    if not gpus:
        raise SystemExit("--gpus must name at least one GPU")
    RUNS.mkdir(parents=True, exist_ok=True)
    STATE.mkdir(parents=True, exist_ok=True)
    records = discover_lineages()
    preflight(records)
    inventory = [{k: v for k, v in record.items() if k != "config"} for record in records]
    (STATE / "inventory.json").write_text(json.dumps(inventory, indent=2))
    if args.preflight:
        return

    # P1 is shared by every lineage; evaluate its two new future domains once.
    evaluate_cache(P1_CACHE, STATE / "baseline_p1_new.json", gpus[0], (4, 5))

    failures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        queues = [records[index::len(gpus)] for index in range(len(gpus))]
        futures = {
            pool.submit(run_gpu_queue, gpu, queue): gpu
            for gpu, queue in zip(gpus, queues)
        }
        for future in concurrent.futures.as_completed(futures):
            gpu = futures[future]
            try:
                result = future.result()
                failures.extend(result["failed"])
                print(
                    f"GPU {gpu} completed {len(result['completed'])}, "
                    f"failed {len(result['failed'])}",
                    flush=True,
                )
            except Exception as exc:
                failures.append((f"gpu-{gpu}", repr(exc)))
                print(f"FAILED GPU {gpu} queue: {exc}", file=sys.stderr, flush=True)
    (STATE / "failures.json").write_text(json.dumps(failures, indent=2))
    if failures:
        raise SystemExit(f"{len(failures)} lineages failed; rerun is stage-resumable")
    print("All 24 QASPER lineages completed ASR and KG")


if __name__ == "__main__":
    main()
