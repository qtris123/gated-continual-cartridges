#!/usr/bin/env python3
"""Free-GPU sweep launcher for Phase-2 AM over granularity x sparsity.

Fixes DELTA_WEIGHT=1e-2 (new Phase-2 default) and sweeps:
  granularity in {per_layer, per_head, global} x top_t in {32,64,128,256}

One Phase-2 run per GPU. Matching IDF stats are required per granularity.
After all runs finish, aggregates QA forgetting + MT acquisition + wall-clock
+ value norms into SUMMARY.md.

Example:
  PHASE1_CACHE_PATH=.../cache_last.pt \\
  BG_STATS_PER_LAYER=.../bg_stats.pt \\
  BG_STATS_PER_HEAD=.../bg_stats_per_head.pt \\
  BG_STATS_GLOBAL=.../bg_stats_global.pt \\
  python examples/shared/am/sweeps/launch_am_sweep_e2e.py --gpus 0,1
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


from examples.shared.paths import ROOT as REPO
TRAIN_SH = REPO / "examples" / "qasper" / "pipelines" / "train_continual_am_sparse.sh"

GRANULARITIES = ["per_layer", "per_head", "global"]
TOP_TS = [32, 64, 128, 256]


@dataclass
class Job:
    cfg: dict
    gpu: int
    proc: subprocess.Popen
    log_path: Path
    run_name: str
    started_at: float


_SMI_OK: bool | None = None


def nvidia_gpu_free_mib(min_free_mib: int, timeout_s: float = 5.0) -> list[int] | None:
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
    return list(range(len(list(root.iterdir()))))


def build_configs(
    bg_per_layer: str | None,
    bg_per_head: str | None,
    bg_global: str | None,
    granularities: list[str],
) -> list[dict]:
    bg_by_gran = {
        "per_layer": bg_per_layer,
        "per_head": bg_per_head,
        "global": bg_global,
    }
    cfgs = []
    for gran in granularities:
        bg = bg_by_gran[gran]
        if not bg:
            raise ValueError(f"Missing bg_stats path for granularity={gran}")
        for top_t in TOP_TS:
            cfgs.append(
                {
                    "name": f"{gran}_top{top_t}",
                    "GRANULARITY": gran,
                    "TOP_T": str(top_t),
                    "BG_STATS_PATH": bg,
                }
            )
    return cfgs


def build_env(base: dict, cfg: dict, gpu: int, run_name: str, out_dir: Path) -> dict:
    env = os.environ.copy()
    env.update(base)
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "RUN_NAME": run_name,
            "GRANULARITY": cfg["GRANULARITY"],
            "TOP_T": cfg["TOP_T"],
            "BG_STATS_PATH": cfg["BG_STATS_PATH"],
            "CARTRIDGES_OUTPUT_DIR": str(out_dir),
            "WANDB_NOTES": (
                f"EXP-010 AM sweep: {cfg['name']} delta=1e-2 "
                f"(gran={cfg['GRANULARITY']}, top_t={cfg['TOP_T']})"
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
    return max(candidates, key=lambda p: p.stat().st_mtime) if candidates else None


def _fmt(x, digits: int = 4) -> str:
    if x is None:
        return "—"
    try:
        return f"{float(x):.{digits}g}"
    except Exception:
        return str(x)


def write_summary(results_dir: Path, rows: list[dict], phase1_qa: float | None) -> Path:
    md = results_dir / "SUMMARY.md"
    lines = [
        "# Phase 2 AM Sweep: granularity x sparsity (delta=1e-2)",
        "",
        f"- Generated: {datetime.now(timezone.utc).isoformat()}",
        f"- Results dir: `{results_dir}`",
    ]
    if phase1_qa is not None:
        lines.append(f"- Phase1 QA loss baseline: {phase1_qa:.4f}")
    lines += [
        "",
        "| config | gran | top_t | QA loss (forget↓) | MT loss (acq↓) | QA ppl | MT ppl | max\\|V\\| | wall_s | status |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    # Sort by granularity then top_t for readable pattern.
    def _key(r):
        return (r.get("GRANULARITY", ""), int(r.get("TOP_T", 0)))

    for r in sorted(rows, key=_key):
        em = r.get("eval_metrics") or {}
        qa = em.get("qa_forgetting") or {}
        mt = em.get("mt_acquisition") or {}
        vn = r.get("value_norms") or {}
        lines.append(
            "| {name} | {gran} | {top_t} | {qa_loss} | {mt_loss} | {qa_ppl} | {mt_ppl} | {vmax} | {wall} | {status} |".format(
                name=r.get("config", "?"),
                gran=r.get("GRANULARITY", "?"),
                top_t=r.get("TOP_T", "?"),
                qa_loss=_fmt(qa.get("loss")),
                mt_loss=_fmt(mt.get("loss")),
                qa_ppl=_fmt(qa.get("perplexity")),
                mt_ppl=_fmt(mt.get("perplexity")),
                vmax=_fmt(vn.get("global_max_abs")),
                wall=_fmt(r.get("wall_clock_s"), digits=1),
                status=r.get("status", "?"),
            )
        )
    lines += [
        "",
        "Lower QA loss = less forgetting; lower MT loss = better acquisition.",
        "",
    ]
    md.write_text("\n".join(lines))
    (results_dir / "summary.json").write_text(json.dumps(rows, indent=2))
    return md


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--phase1-cache", default=os.environ.get("PHASE1_CACHE_PATH"))
    ap.add_argument("--bg-per-layer", default=os.environ.get("BG_STATS_PER_LAYER"))
    ap.add_argument("--bg-per-head", default=os.environ.get("BG_STATS_PER_HEAD"))
    ap.add_argument("--bg-global", default=os.environ.get("BG_STATS_GLOBAL"))
    ap.add_argument(
        "--output-root",
        default=os.environ.get(
            "CARTRIDGES_OUTPUT_DIR", str(REPO / "outputs" / "e2e_am_sweep")
        ),
    )
    ap.add_argument("--gpus", default=os.environ.get("LAUNCH_GPUS", ""))
    ap.add_argument("--min-free-mib", type=int, default=40000)
    ap.add_argument("--poll-s", type=float, default=20.0)
    ap.add_argument(
        "--phase1-qa-loss",
        type=float,
        default=float(os.environ.get("PHASE1_QA_LOSS", "3.821")),
    )
    ap.add_argument(
        "--configs",
        default="",
        help="Comma-separated config names (default: all)",
    )
    ap.add_argument(
        "--granularities",
        default="",
        help="Comma-separated subset of per_layer,per_head,global (default: all)",
    )
    args = ap.parse_args()

    if not args.phase1_cache or not Path(args.phase1_cache).is_file():
        print("ERROR: --phase1-cache required", file=sys.stderr)
        return 2

    granularities = (
        [x.strip() for x in args.granularities.split(",") if x.strip()]
        if args.granularities.strip()
        else list(GRANULARITIES)
    )
    unknown = [g for g in granularities if g not in GRANULARITIES]
    if unknown:
        print(f"ERROR: unknown granularities: {unknown}", file=sys.stderr)
        return 2

    bg_map = {
        "per_layer": args.bg_per_layer,
        "per_head": args.bg_per_head,
        "global": args.bg_global,
    }
    for gran in granularities:
        path = bg_map[gran]
        if not path or not Path(path).is_file():
            print(f"ERROR: missing bg_stats for {gran}: {path}", file=sys.stderr)
            return 2

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(args.output_root) / f"am_sweep_{stamp}"
    results_dir.mkdir(parents=True, exist_ok=True)
    runs_out = results_dir / "runs"
    runs_out.mkdir(exist_ok=True)
    (results_dir / "logs").mkdir(exist_ok=True)

    cfgs = build_configs(
        str(Path(args.bg_per_layer).resolve()) if args.bg_per_layer else None,
        str(Path(args.bg_per_head).resolve()) if args.bg_per_head else None,
        str(Path(args.bg_global).resolve()) if args.bg_global else None,
        granularities,
    )
    if args.configs.strip():
        want = {x.strip() for x in args.configs.split(",") if x.strip()}
        cfgs = [c for c in cfgs if c["name"] in want]
    if not cfgs:
        print("ERROR: no configs selected", file=sys.stderr)
        return 2

    base_env = {
        "CARTRIDGES_DIR": str(REPO),
        "PHASE1_CACHE_PATH": str(Path(args.phase1_cache).resolve()),
        "SYNTH_DATA_PATH": str(
            REPO / "data" / "qasper" / "train" / "qwen_qasper_MT_task_8192.parquet"
        ),
        "EVAL_QA_PATH": str(REPO / "data" / "qasper" / "eval" / "qasper_eval_QA.parquet"),
        "EVAL_MT_PATH": str(REPO / "data" / "qasper" / "eval" / "qasper_eval_MT.parquet"),
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
        "USE_IDF": "1",
        "SAVE_AFTER_EACH_DOCUMENT": "0",
        "AM_COMPUTE_STATS": "1",
        # New default stabilizer.
        "DELTA_WEIGHT": "1e-2",
        "RIDGE_SCALE": "spectral",
        "RIDGE_LAMBDA": "1e-4",
        "RIDGE_LAMBDA_MIN": "0.0",
        "WANDB_DISABLED": os.environ.get("WANDB_DISABLED", "0"),
        "WANDB_GROUP": os.environ.get("WANDB_GROUP", f"qasper-am-sweep-{stamp}"),
        "TORCH_CUDA_ARCH_LIST": os.environ.get("TORCH_CUDA_ARCH_LIST", "9.0"),
        "MODEL_NAME": os.environ.get("MODEL_NAME", "Qwen/Qwen3-4B-Instruct-2507"),
    }

    explicit_gpus = (
        [int(x) for x in args.gpus.split(",") if x.strip()]
        if args.gpus.strip()
        else (discover_gpus_from_proc() or [0, 1])
    )

    queue = list(cfgs)
    active: list[Job] = []
    finished: list[dict] = []
    reserved: set[int] = set()

    manifest = {
        "results_dir": str(results_dir),
        "phase1_cache": base_env["PHASE1_CACHE_PATH"],
        "bg_per_layer": args.bg_per_layer,
        "bg_per_head": args.bg_per_head,
        "bg_global": args.bg_global,
        "granularities": granularities,
        "configs": [c["name"] for c in cfgs],
        "gpus": explicit_gpus,
        "wandb_group": base_env["WANDB_GROUP"],
        "delta_weight": "1e-2",
    }
    (results_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Results → {results_dir}")
    print(f"GPU pool: {explicit_gpus}")
    print(f"Queue ({len(queue)}): {[c['name'] for c in queue]}", flush=True)

    while queue or active:
        still = []
        for job in active:
            rc = job.proc.poll()
            if rc is None:
                still.append(job)
                continue
            reserved.discard(job.gpu)
            summary = find_newest_summary(runs_out, job.run_name, job.started_at)
            row = {
                "config": job.cfg["name"],
                "GRANULARITY": job.cfg["GRANULARITY"],
                "TOP_T": job.cfg["TOP_T"],
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
                    }
                )
            finished.append(row)
            print(f"[done] {job.cfg['name']} gpu={job.gpu} rc={rc}", flush=True)
            write_summary(results_dir, finished, args.phase1_qa_loss)
        active = still

        if args.gpus.strip():
            free = [g for g in explicit_gpus if g not in reserved]
        else:
            smi = nvidia_gpu_free_mib(args.min_free_mib)
            free = (
                [g for g in smi if g not in reserved]
                if smi is not None
                else [g for g in explicit_gpus if g not in reserved]
            )
        busy = {j.gpu for j in active}
        free = [g for g in free if g not in busy]

        while queue and free:
            cfg = queue.pop(0)
            gpu = free.pop(0)
            run_name = f"phase2_sweep_{cfg['name']}_{stamp}"
            log_path = results_dir / "logs" / f"{cfg['name']}.log"
            env = build_env(base_env, cfg, gpu, run_name, runs_out)
            print(f"[launch] {cfg['name']} → GPU {gpu}", flush=True)
            log_f = open(log_path, "w")
            proc = subprocess.Popen(
                ["bash", str(TRAIN_SH)],
                cwd=str(REPO),
                env=env,
                stdout=log_f,
                stderr=subprocess.STDOUT,
            )
            reserved.add(gpu)
            active.append(
                Job(
                    cfg=cfg,
                    gpu=gpu,
                    proc=proc,
                    log_path=log_path,
                    run_name=run_name,
                    started_at=time.time(),
                )
            )

        if queue or active:
            time.sleep(args.poll_s)

    md = write_summary(results_dir, finished, args.phase1_qa_loss)
    print(f"\nAll done. Summary: {md}", flush=True)
    fails = [r for r in finished if r.get("exit_code", 1) != 0]
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
