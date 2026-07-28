#!/usr/bin/env python3
"""Free-GPU launcher for Phase-2 AM regularization e2e A/B.

Polls nvidia-smi for idle GPUs and launches one Phase-2 continual run per free
GPU. After all training finishes, aggregates phase2_summary.json (QA forgetting
+ MT acquisition + value norms) into SUMMARY.md.

Example:
  PHASE1_CACHE_PATH=.../cache_last.pt \\
  BG_STATS_PATH=.../bg_stats.pt \\
  python examples/qasper2/scripts/sweeps/launch_am_reg_e2e.py
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


REPO = Path(__file__).resolve().parents[3]
TRAIN_SH = REPO / "examples" / "qasper2" / "scripts" / "core" / "train_continual_am_sparse.sh"

DEFAULT_REGIMES = [
    {
        "name": "spectral_baseline",
        "RIDGE_SCALE": "spectral",
        "RIDGE_LAMBDA": "1e-4",
        "RIDGE_LAMBDA_MIN": "0.0",
        "DELTA_WEIGHT": "0.0",
    },
    {
        "name": "spectral_floor_1e-4",
        "RIDGE_SCALE": "spectral",
        "RIDGE_LAMBDA": "1e-4",
        "RIDGE_LAMBDA_MIN": "1e-4",
        "DELTA_WEIGHT": "0.0",
    },
    {
        "name": "delta_1e-2",
        "RIDGE_SCALE": "spectral",
        "RIDGE_LAMBDA": "1e-4",
        "RIDGE_LAMBDA_MIN": "0.0",
        "DELTA_WEIGHT": "1e-2",
    },
    {
        "name": "floor_delta",
        "RIDGE_SCALE": "spectral",
        "RIDGE_LAMBDA": "1e-4",
        "RIDGE_LAMBDA_MIN": "1e-4",
        "DELTA_WEIGHT": "1e-2",
    },
]


@dataclass
class Job:
    regime: dict
    gpu: int
    proc: subprocess.Popen
    log_path: Path
    run_name: str
    started_at: float


_SMI_OK: bool | None = None


def nvidia_gpu_free_mib(min_free_mib: int, timeout_s: float = 5.0) -> list[int] | None:
    """Return free GPU indices, or None if nvidia-smi is unavailable/hung."""
    global _SMI_OK
    if _SMI_OK is False:
        return None
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.free,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=timeout_s,
        )
        _SMI_OK = True
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        _SMI_OK = False
        return None
    free = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        idx, mem_free, util = int(parts[0]), int(float(parts[1])), int(float(parts[2]))
        if mem_free >= min_free_mib and util <= 5:
            free.append(idx)
    return free


def discover_gpus_from_proc() -> list[int]:
    root = Path("/proc/driver/nvidia/gpus")
    if not root.is_dir():
        return []
    # Directory names are PCI IDs; enumerate in sorted order → indices 0..N-1
    return list(range(len(list(root.iterdir()))))


def resolve_candidate_gpus(
    explicit: list[int] | None,
    min_free_mib: int,
    reserved: set[int],
) -> list[int]:
    """Pick launchable GPUs. Prefer nvidia-smi; fall back to explicit/proc."""
    smi = nvidia_gpu_free_mib(min_free_mib)
    if smi is not None:
        return [g for g in smi if g not in reserved]
    pool = explicit if explicit is not None else discover_gpus_from_proc()
    return [g for g in pool if g not in reserved]


def build_env(base: dict, regime: dict, gpu: int, run_name: str, out_dir: Path) -> dict:
    env = os.environ.copy()
    env.update(base)
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "RUN_NAME": run_name,
            "RIDGE_SCALE": regime["RIDGE_SCALE"],
            "RIDGE_LAMBDA": regime["RIDGE_LAMBDA"],
            "RIDGE_LAMBDA_MIN": regime["RIDGE_LAMBDA_MIN"],
            "DELTA_WEIGHT": regime["DELTA_WEIGHT"],
            "CARTRIDGES_OUTPUT_DIR": str(out_dir),
            "WANDB_NOTES": (
                f"EXP-009 e2e reg A/B: {regime['name']} "
                f"(scale={regime['RIDGE_SCALE']}, "
                f"lam_min={regime['RIDGE_LAMBDA_MIN']}, "
                f"delta={regime['DELTA_WEIGHT']})"
            ),
        }
    )
    return env


def find_newest_summary(out_dir: Path, run_name: str, after_ts: float) -> Path | None:
    candidates = []
    for p in out_dir.glob("**/phase2_summary.json"):
        try:
            data = json.loads(p.read_text())
        except Exception:
            continue
        if data.get("run_name") != run_name:
            continue
        if p.stat().st_mtime < after_ts - 5:
            continue
        candidates.append(p)
    if not candidates:
        # Fallback: newest cache under continual_am_sparse with matching RUN via config.yaml
        for cfg in out_dir.glob("**/continual_am_sparse/*/config.yaml"):
            text = cfg.read_text()
            if f"name: {run_name}" in text or f'name: "{run_name}"' in text:
                summary = cfg.parent / "phase2_summary.json"
                if summary.exists() and summary.stat().st_mtime >= after_ts - 5:
                    candidates.append(summary)
    return max(candidates, key=lambda p: p.stat().st_mtime) if candidates else None


def write_summary(results_dir: Path, rows: list[dict]) -> Path:
    md = results_dir / "SUMMARY.md"
    lines = [
        "# Phase 2 Regularization E2E A/B",
        "",
        f"- Generated: {datetime.utcnow().isoformat()}Z",
        f"- Results dir: `{results_dir}`",
        "",
        "| regime | QA loss (forget↓) | QA ppl | MT loss (acq↓) | MT ppl | max\\|V\\| | wall_s | status |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in rows:
        em = r.get("eval_metrics") or {}
        qa = em.get("qa_forgetting") or {}
        mt = em.get("mt_acquisition") or {}
        vn = r.get("value_norms") or {}
        lines.append(
            "| {name} | {qa_loss} | {qa_ppl} | {mt_loss} | {mt_ppl} | {vmax} | {wall} | {status} |".format(
                name=r.get("regime", "?"),
                qa_loss=_fmt(qa.get("loss")),
                qa_ppl=_fmt(qa.get("perplexity")),
                mt_loss=_fmt(mt.get("loss")),
                mt_ppl=_fmt(mt.get("perplexity")),
                vmax=_fmt(vn.get("global_max_abs")),
                wall=_fmt(r.get("wall_clock_s"), digits=1),
                status=r.get("status", "?"),
            )
        )
    lines.append("")
    lines.append("Lower QA loss = less forgetting; lower MT loss = better acquisition.")
    lines.append("")
    md.write_text("\n".join(lines))
    (results_dir / "summary.json").write_text(json.dumps(rows, indent=2))
    return md


def _fmt(x, digits: int = 4) -> str:
    if x is None:
        return "—"
    try:
        return f"{float(x):.{digits}g}"
    except Exception:
        return str(x)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--phase1-cache",
        default=os.environ.get("PHASE1_CACHE_PATH"),
        help="Phase1 cache_last.pt",
    )
    ap.add_argument(
        "--bg-stats",
        default=os.environ.get("BG_STATS_PATH"),
        help="Phase1 bg_stats.pt",
    )
    ap.add_argument(
        "--output-root",
        default=os.environ.get(
            "CARTRIDGES_OUTPUT_DIR",
            str(REPO / "outputs" / "e2e_am_reg_ab"),
        ),
    )
    ap.add_argument("--min-free-mib", type=int, default=40000)
    ap.add_argument("--poll-s", type=float, default=20.0)
    ap.add_argument(
        "--gpus",
        default=os.environ.get("LAUNCH_GPUS", ""),
        help="Comma-separated GPU indices (fallback when nvidia-smi hangs)",
    )
    ap.add_argument(
        "--skip-baseline",
        action="store_true",
        help="Skip spectral_baseline (known unstable)",
    )
    ap.add_argument(
        "--regimes",
        default="",
        help="Comma-separated regime names (default: all)",
    )
    args = ap.parse_args()

    if not args.phase1_cache or not Path(args.phase1_cache).is_file():
        print("ERROR: --phase1-cache / PHASE1_CACHE_PATH required", file=sys.stderr)
        return 2
    if not args.bg_stats or not Path(args.bg_stats).is_file():
        print("ERROR: --bg-stats / BG_STATS_PATH required", file=sys.stderr)
        return 2

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(args.output_root) / f"am_reg_e2e_{stamp}"
    results_dir.mkdir(parents=True, exist_ok=True)
    runs_out = results_dir / "runs"
    runs_out.mkdir(exist_ok=True)

    regimes = list(DEFAULT_REGIMES)
    if args.skip_baseline:
        regimes = [r for r in regimes if r["name"] != "spectral_baseline"]
    if args.regimes.strip():
        want = {x.strip() for x in args.regimes.split(",") if x.strip()}
        regimes = [r for r in regimes if r["name"] in want]
    if not regimes:
        print("ERROR: no regimes selected", file=sys.stderr)
        return 2

    base_env = {
        "CARTRIDGES_DIR": str(REPO),
        "PHASE1_CACHE_PATH": str(Path(args.phase1_cache).resolve()),
        "BG_STATS_PATH": str(Path(args.bg_stats).resolve()),
        "SYNTH_DATA_PATH": str(
            REPO / "data" / "qasper" / "train" / "qwen_qasper_MT_task_8192.parquet"
        ),
        "EVAL_QA_PATH": str(
            REPO / "data" / "qasper" / "eval" / "qasper_eval_QA.parquet"
        ),
        "EVAL_MT_PATH": str(
            REPO / "data" / "qasper" / "eval" / "qasper_eval_MT.parquet"
        ),
        "OLD_REF_DATA_PATH": str(
            REPO / "data" / "qasper" / "train" / "qwen_qasper_QA_task_8192.parquet"
        ),
        "AM_EXECUTION_MODE": "per_document",
        "TARGET_MODE": "cartridge_plus_doc",
        "KEY_MODE": "freeze",
        "ENABLE_BETA": "0",
        "ENABLE_OLD_REFERENCE_GUARD": "1",
        "OLD_REFERENCE_WEIGHT": "1.0",
        "OLD_REF_MAX_EXAMPLES": "64",
        "MAX_REF_EXAMPLES_PER_DOC": "32",
        "TOP_T": "64",
        "GRANULARITY": "per_layer",
        "USE_IDF": "1",
        "SAVE_AFTER_EACH_DOCUMENT": "1",
        "AM_COMPUTE_STATS": "1",
        "WANDB_DISABLED": os.environ.get("WANDB_DISABLED", "0"),
        "WANDB_GROUP": os.environ.get(
            "WANDB_GROUP", f"qasper-am-reg-e2e-{stamp}"
        ),
        "TORCH_CUDA_ARCH_LIST": os.environ.get("TORCH_CUDA_ARCH_LIST", "9.0"),
        "MODEL_NAME": os.environ.get("MODEL_NAME", "Qwen/Qwen3-4B-Instruct-2507"),
    }

    queue = list(regimes)
    active: list[Job] = []
    finished: list[dict] = []
    reserved_gpus: set[int] = set()
    explicit_gpus = (
        [int(x) for x in args.gpus.split(",") if x.strip()]
        if args.gpus.strip()
        else None
    )
    if explicit_gpus is None:
        # Prefer explicit fallback list from /proc so we can launch even if
        # nvidia-smi hangs (observed on this host).
        explicit_gpus = discover_gpus_from_proc() or [0, 1]
    print(f"GPU pool (fallback): {explicit_gpus}")

    manifest = {
        "results_dir": str(results_dir),
        "phase1_cache": base_env["PHASE1_CACHE_PATH"],
        "bg_stats": base_env["BG_STATS_PATH"],
        "regimes": [r["name"] for r in regimes],
        "wandb_group": base_env["WANDB_GROUP"],
        "gpus": explicit_gpus,
    }
    (results_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Results → {results_dir}")
    print(f"Queue: {[r['name'] for r in queue]}")

    smi_warned = False
    while queue or active:
        # Reap finished
        still = []
        for job in active:
            rc = job.proc.poll()
            if rc is None:
                still.append(job)
                continue
            reserved_gpus.discard(job.gpu)
            summary = find_newest_summary(runs_out, job.run_name, job.started_at)
            row = {
                "regime": job.regime["name"],
                "run_name": job.run_name,
                "gpu": job.gpu,
                "exit_code": rc,
                "status": "ok" if rc == 0 else f"fail:{rc}",
                "log": str(job.log_path),
                "summary_path": str(summary) if summary else None,
            }
            if summary and summary.exists():
                data = json.loads(summary.read_text())
                row.update(
                    {
                        "run_dir": data.get("run_dir"),
                        "wall_clock_s": data.get("wall_clock_s"),
                        "eval_metrics": data.get("eval_metrics"),
                        "value_norms": data.get("value_norms"),
                        "mean_mse_last_doc": data.get("mean_mse_last_doc"),
                        "reg": data.get("reg"),
                    }
                )
            finished.append(row)
            print(
                f"[done] {job.regime['name']} gpu={job.gpu} rc={rc} "
                f"summary={summary}"
            )
            write_summary(results_dir, finished)
        active = still

        # Launch onto free GPUs
        if args.gpus.strip():
            # Explicit pool: track occupancy ourselves (nvidia-smi often hangs here).
            free = [g for g in explicit_gpus if g not in reserved_gpus]
        else:
            smi = nvidia_gpu_free_mib(args.min_free_mib)
            if smi is None and not smi_warned:
                print(
                    "[warn] nvidia-smi unavailable/timeout — "
                    "using GPU pool occupancy tracking only",
                    flush=True,
                )
                smi_warned = True
            free = resolve_candidate_gpus(
                explicit_gpus, args.min_free_mib, reserved_gpus
            )
        busy = {j.gpu for j in active}
        free = [g for g in free if g not in busy]

        while queue and free:
            regime = queue.pop(0)
            gpu = free.pop(0)
            run_name = f"phase2_reg_{regime['name']}_{stamp}"
            log_path = results_dir / "logs" / f"{regime['name']}.log"
            log_path.parent.mkdir(exist_ok=True)
            env = build_env(base_env, regime, gpu, run_name, runs_out)
            print(f"[launch] {regime['name']} → GPU {gpu}  log={log_path}", flush=True)
            log_f = open(log_path, "w")
            proc = subprocess.Popen(
                ["bash", str(TRAIN_SH)],
                cwd=str(REPO),
                env=env,
                stdout=log_f,
                stderr=subprocess.STDOUT,
            )
            reserved_gpus.add(gpu)
            active.append(
                Job(
                    regime=regime,
                    gpu=gpu,
                    proc=proc,
                    log_path=log_path,
                    run_name=run_name,
                    started_at=time.time(),
                )
            )

        if queue or active:
            time.sleep(args.poll_s)

    md = write_summary(results_dir, finished)
    print(f"\nAll done. Summary: {md}")
    fails = [r for r in finished if r.get("exit_code", 1) != 0]
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
