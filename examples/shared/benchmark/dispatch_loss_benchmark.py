#!/usr/bin/env python3
"""Unattended GPU-queue dispatcher for the Qasper QA -> MT loss benchmark.

ICL jobs (full topic-panel context) run one at a time on a single GPU via
chunked generate-mode prefill.  Cartridge-only jobs run in parallel on the
remaining GPU(s).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

BENCHMARK_SCRIPT = Path(__file__).resolve().parent / "qasper_loss_benchmark.py"

METHODS = [
    "icl_QA_raw",
    "icl_MT_raw",
    "icl_QA_plus_MT",
    "icl_MT_plus_QA",
    "icl_MT_plus_QA_cartridge",
]

ICL_METHODS = {
    "icl_QA_raw",
    "icl_MT_raw",
    "icl_QA_plus_MT",
    "icl_MT_plus_QA",
    "icl_MT_plus_QA_cartridge",
}


def detect_gpus(explicit: Optional[list[int]]) -> list[int]:
    if explicit:
        return explicit
    env = os.environ.get("CUDA_VISIBLE_DEVICES")
    if env:
        return [int(x) for x in env.split(",") if x.strip() != ""]
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            text=True,
            timeout=15,
        )
        gpus = [int(line.strip()) for line in out.splitlines() if line.strip()]
        return gpus or [0]
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        print(
            "WARNING: nvidia-smi unavailable; defaulting to GPU [0]. "
            "Pass --gpus explicitly to override.",
            flush=True,
        )
        return [0]


class Job:
    def __init__(self, family: str, method: str, run_root: Path):
        self.family = family
        self.method = method
        self.name = f"{family}__{method}"
        self.out_dir = run_root / self.name
        self.log_path = self.out_dir / "job.log"
        self.proc: Optional[subprocess.Popen] = None
        self.gpus: list[int] = []
        self.log_fh = None
        self.returncode: Optional[int] = None

    @property
    def is_icl(self) -> bool:
        return self.method in ICL_METHODS

    def launch(self, gpus: list[int], common_args: list[str]) -> None:
        self.gpus = gpus
        self.out_dir.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpus)
        cmd = [
            sys.executable,
            str(BENCHMARK_SCRIPT),
            "--models", self.family,
            "--only-methods", self.method,
            "--output-dir", str(self.out_dir),
            "--device", "cuda",
            *common_args,
        ]
        self.log_fh = open(self.log_path, "w", encoding="utf-8")
        self.log_fh.write(
            f"# {self.name} on GPU(s) {gpus}\n# cmd: {' '.join(cmd)}\n\n"
        )
        self.log_fh.flush()
        self.proc = subprocess.Popen(cmd, stdout=self.log_fh, stderr=subprocess.STDOUT, env=env)

    def poll(self) -> bool:
        if self.proc is None:
            return False
        rc = self.proc.poll()
        if rc is None:
            return False
        self.returncode = rc
        if self.log_fh is not None:
            self.log_fh.flush()
            self.log_fh.close()
            self.log_fh = None
        return True


def merge_results(run_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rows_file in sorted(run_root.glob("*/*_loss_rows.json")):
        try:
            rows.extend(json.loads(rows_file.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError) as exc:
            print(f"WARNING: could not read {rows_file}: {exc}", flush=True)
    return rows


def print_table(rows: list[dict[str, Any]]) -> None:
    if not rows:
        print("(no results)")
        return
    families = sorted({r["model_family"] for r in rows})
    print("\n" + "=" * 72)
    print("Qasper QA -> MT perplexity benchmark")
    print("=" * 72)
    for family in families:
        print(f"\n[{family}]")
        print(f"{'method':<28} {'QA eval ppl':>14} {'MT eval ppl':>14}")
        print("-" * 58)
        for method in METHODS:
            qa = next(
                (r for r in rows if r["model_family"] == family and r["method"] == method and r["eval"] == "QA"),
                None,
            )
            mt = next(
                (r for r in rows if r["model_family"] == family and r["method"] == method and r["eval"] == "MT"),
                None,
            )
            qa_s = f"{qa['perplexity']:.4f}" if qa and qa.get("perplexity") is not None else "N/A"
            mt_s = f"{mt['perplexity']:.4f}" if mt and mt.get("perplexity") is not None else "N/A"
            print(f"{method:<28} {qa_s:>14} {mt_s:>14}")


def main() -> None:
    parser = argparse.ArgumentParser(description="GPU-queue dispatcher for Qasper loss benchmark.")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--gpus", type=int, nargs="+", default=None, help="GPU indices (default: auto-detect).")
    parser.add_argument("--models", nargs="+", choices=["llama", "qwen"], default=["llama", "qwen"])
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=METHODS)
    parser.add_argument("--qa-eval", type=Path, default=Path("data/qasper/eval/qasper_eval_QA.parquet"))
    parser.add_argument("--mt-eval", type=Path, default=Path("data/qasper/eval/qasper_eval_MT.parquet"))
    parser.add_argument("--max-context-tokens", type=int, default=None)
    parser.add_argument("--prefill-chunk-size", type=int, default=2048)
    parser.add_argument("--poll-interval", type=float, default=5.0)
    args = parser.parse_args()

    run_root = args.output_dir or Path(
        os.environ.get("CARTRIDGES_OUTPUT_DIR", "outputs")
    ) / f"qasper_loss_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_root.mkdir(parents=True, exist_ok=True)

    gpus = detect_gpus(args.gpus)
    qa_eval = args.qa_eval.expanduser().resolve()
    mt_eval = args.mt_eval.expanduser().resolve()

    print("=" * 72)
    print("Qasper loss benchmark dispatcher")
    print("=" * 72)
    print(f"Run root : {run_root}")
    print(f"GPUs     : {gpus}")
    print(f"Models   : {args.models}")
    print(f"Methods  : {args.methods}")
    print(f"QA eval  : {qa_eval}")
    print(f"MT eval  : {mt_eval}")
    print("ICL jobs : full topic panel, single GPU, one at a time")
    print("Cart jobs: remaining GPU(s), parallel with ICL when available")

    print("\nEnsuring eval parquets are present...", flush=True)
    subprocess.run(
        [
            sys.executable, str(BENCHMARK_SCRIPT),
            "--download-only",
            "--qa-eval", str(qa_eval),
            "--mt-eval", str(mt_eval),
        ],
        check=True,
    )

    common_args = [
        "--qa-eval", str(qa_eval),
        "--mt-eval", str(mt_eval),
        "--prefill-chunk-size", str(args.prefill_chunk_size),
    ]
    if args.max_context_tokens is not None:
        common_args += ["--max-context-tokens", str(args.max_context_tokens)]

    icl_queue: list[Job] = []
    cart_queue: list[Job] = []
    for family in args.models:
        for method in args.methods:
            job = Job(family, method, run_root)
            (icl_queue if job.is_icl else cart_queue).append(job)

    total = len(icl_queue) + len(cart_queue)
    print(f"\nQueued {total} jobs ({len(icl_queue)} ICL, {len(cart_queue)} cartridge).\n", flush=True)

    icl_running: Optional[tuple[int, Job]] = None
    single_running: dict[int, Job] = {}
    completed: list[Job] = []

    def busy_gpus() -> set[int]:
        busy = set(single_running)
        if icl_running is not None:
            busy.add(icl_running[0])
        return busy

    while icl_queue or cart_queue or icl_running or single_running:
        if icl_running is None and icl_queue:
            free = [gpu for gpu in gpus if gpu not in busy_gpus()]
            if free:
                gpu = free[0]
                job = icl_queue.pop(0)
                job.launch([gpu], common_args)
                icl_running = (gpu, job)
                print(
                    f"[{len(completed)}/{total}] launched {job.name} on GPU {gpu} (ICL) "
                    f"(log: {job.log_path})",
                    flush=True,
                )

        for gpu in gpus:
            if gpu not in busy_gpus() and cart_queue:
                job = cart_queue.pop(0)
                job.launch([gpu], common_args)
                single_running[gpu] = job
                print(
                    f"[{len(completed)}/{total}] launched {job.name} on GPU {gpu} "
                    f"(log: {job.log_path})",
                    flush=True,
                )

        if icl_running is not None:
            gpu, job = icl_running
            if job.poll():
                status = "ok" if job.returncode == 0 else f"FAILED(rc={job.returncode})"
                print(
                    f"[{len(completed) + 1}/{total}] finished {job.name} on GPU {gpu}: {status}",
                    flush=True,
                )
                completed.append(job)
                icl_running = None

        for gpu, job in list(single_running.items()):
            if job.poll():
                status = "ok" if job.returncode == 0 else f"FAILED(rc={job.returncode})"
                print(
                    f"[{len(completed) + 1}/{total}] finished {job.name} on GPU {gpu}: {status}",
                    flush=True,
                )
                completed.append(job)
                del single_running[gpu]

        time.sleep(args.poll_interval)

    rows = merge_results(run_root)
    summary_path = run_root / "qasper_loss_benchmark_summary.json"
    summary_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    print_table(rows)

    failures = [j for j in completed if j.returncode != 0]
    print("\n" + "=" * 72)
    print(f"All {total} jobs done. {len(failures)} failed.")
    if failures:
        for j in failures:
            print(f"  FAILED: {j.name}  (see {j.log_path})")
    print(f"Merged summary: {summary_path}")
    print("=" * 72)

    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
