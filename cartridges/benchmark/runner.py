from __future__ import annotations

import asyncio
import math
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from pydantic import Field
from pydrantic import BaseConfig, RunConfig

from cartridges.clients.base import ClientConfig
from cartridges.benchmark.datasets import BenchmarkItem, load_dataset_items
from cartridges.benchmark.scorers import SCORER_REGISTRY, ScorerFn
from cartridges.utils import get_logger

logger = get_logger(__name__)


class BenchmarkConfig(RunConfig):
    # ---- dataset ----
    dataset_name: str                          # "mmlu", "qasper", "hellaswag", "truthfulqa", "local"
    dataset_path: Optional[str] = None         # HF id override or local file path
    subset: Optional[str] = None               # e.g. "abstract_algebra" for MMLU
    num_few_shot: int = 0
    prompt_template: Optional[str] = None      # override the default per-dataset prompt
    max_samples: Optional[int] = None          # limit for quick runs

    # ---- inference ----
    client: ClientConfig
    temperature: float = 0.0
    max_completion_tokens: int = 256
    batch_size: int = 16
    max_concurrent_batches: int = 4

    # ---- scoring ----
    scorer: str = "exact_match"                # key into SCORER_REGISTRY

    # ---- output ----
    name: Optional[str] = "benchmark"
    output_dir: str = Field(default=os.environ.get("CARTRIDGES_OUTPUT_DIR", "."))
    seed: int = 42

    def run(self):
        asyncio.run(_run_benchmark(self))


async def _run_benchmark(config: BenchmarkConfig):
    t_start = time.time()

    # --- load data ---
    items = load_dataset_items(
        config.dataset_name,
        dataset_path=config.dataset_path,
        subset=config.subset,
        num_few_shot=config.num_few_shot,
        prompt_template=config.prompt_template,
        max_samples=config.max_samples,
        seed=config.seed,
    )

    if not items:
        logger.error("No benchmark items loaded — aborting.")
        return

    # --- resolve scorer ---
    scorer_fn: ScorerFn = SCORER_REGISTRY.get(config.scorer)
    if scorer_fn is None:
        raise ValueError(
            f"Unknown scorer '{config.scorer}'. Available: {list(SCORER_REGISTRY.keys())}"
        )

    # --- instantiate client ---
    client = config.client.instantiate()
    logger.info(
        f"Running benchmark '{config.dataset_name}' "
        f"({len(items)} items, batch_size={config.batch_size}, "
        f"max_concurrent={config.max_concurrent_batches})"
    )

    # --- run batched inference ---
    total_batches = math.ceil(len(items) / config.batch_size)
    results: list[dict] = [None] * len(items)  # type: ignore[list-item]
    semaphore = asyncio.Semaphore(config.max_concurrent_batches)

    async def _process_batch(batch_idx: int):
        start = batch_idx * config.batch_size
        end = min(start + config.batch_size, len(items))
        batch_items = items[start:end]

        chats = [
            [{"role": "user", "content": item.prompt}]
            for item in batch_items
        ]

        async with semaphore:
            t0 = time.time()
            response = await client.chat(
                chats=chats,
                temperature=config.temperature,
                max_completion_tokens=config.max_completion_tokens,
            )
            elapsed = time.time() - t0

        logger.info(
            f"Batch {batch_idx + 1}/{total_batches} completed in {elapsed:.1f}s"
        )

        for local_idx, (item, sample) in enumerate(
            zip(batch_items, response.samples)
        ):
            prediction = sample.text or ""
            score = scorer_fn(prediction, item.ground_truth, metadata=item.metadata)
            row = {
                "prompt": item.prompt,
                "ground_truth": item.ground_truth,
                "prediction": prediction,
                "score": score,
                **item.metadata,
            }
            if sample.chosen_logprobs is not None and len(sample.chosen_logprobs) > 0:
                mean_nll = -float(np.mean(sample.chosen_logprobs))
                row["perplexity"] = float(np.exp(mean_nll))
                row["mean_log_prob"] = float(np.mean(sample.chosen_logprobs))
                row["num_generated_tokens"] = len(sample.chosen_logprobs)
            results[start + local_idx] = row

    tasks = [_process_batch(i) for i in range(total_batches)]
    await asyncio.gather(*tasks)

    # --- build output dataframe ---
    df = pd.DataFrame(results)

    # --- save locally ---
    run_dir = Path(config.output_dir) / (config.name or "benchmark")
    run_dir.mkdir(parents=True, exist_ok=True)

    parquet_path = run_dir / "results.parquet"
    csv_path = run_dir / "results.csv"
    df.to_parquet(parquet_path, index=False)
    df.to_csv(csv_path, index=False)

    # --- summary ---
    elapsed_total = time.time() - t_start
    avg_score = df["score"].mean()
    num_correct = (df["score"] == 1.0).sum()
    total = len(df)

    ppl_line = ""
    if "perplexity" in df.columns:
        avg_ppl = df["perplexity"].mean()
        median_ppl = df["perplexity"].median()
        avg_logp = df["mean_log_prob"].mean()
        ppl_line = (
            f"  Perplexity (mean): {avg_ppl:.2f}\n"
            f"  Perplexity (median): {median_ppl:.2f}\n"
            f"  Mean log-prob:     {avg_logp:.4f}\n"
        )

    summary = (
        f"\n{'=' * 60}\n"
        f"  Benchmark: {config.dataset_name}"
        f"{(' / ' + config.subset) if config.subset else ''}\n"
        f"  Scorer:    {config.scorer}\n"
        f"  Samples:   {total}\n"
        f"  Score:     {avg_score:.4f}  ({num_correct}/{total} correct)\n"
        f"{ppl_line}"
        f"  Time:      {elapsed_total:.1f}s\n"
        f"  Output:    {run_dir.absolute()}\n"
        f"{'=' * 60}"
    )
    logger.info(summary)
    print(summary)

    # --- per-subset breakdown (e.g. MMLU subjects) ---
    if "subject" in df.columns:
        breakdown = (
            df.groupby("subject")["score"]
            .agg(["mean", "count"])
            .rename(columns={"mean": "accuracy", "count": "n"})
            .sort_values("accuracy")
        )
        breakdown.to_csv(run_dir / "breakdown_by_subject.csv")
        logger.info(f"Per-subject breakdown saved to {run_dir / 'breakdown_by_subject.csv'}")
        print(f"\nPer-subject breakdown (bottom 10):\n{breakdown.head(10).to_string()}")

    return df


# ---------------------------------------------------------------------------
# BenchmarkWithServerConfig — auto-launch a Tokasaurus server, then benchmark
# ---------------------------------------------------------------------------

class BenchmarkWithServerConfig(RunConfig):
    """Launches a Tokasaurus server, patches the client URL, runs the
    benchmark, then shuts the server down.  Same lifecycle pattern as
    ``EvaluateTokaConfig`` in ``infra/tuning/tune_toka.py`` but for
    ``BenchmarkConfig`` instead of ``SynthesizeConfig``.
    """
    benchmark: BenchmarkConfig
    tokasaurus: "TokaServerConfig"
    output_dir: str = Field(default=os.environ.get("CARTRIDGES_OUTPUT_DIR", "."))
    conda_env: Optional[str] = None

    def run(self):
        _run_benchmark_with_server(self)


class TokaServerConfig(BaseConfig):
    """Mirrors ``TokaConfig`` from ``infra/tuning/tune_toka.py`` so the
    benchmark module is self-contained.  All fields have the same names
    as the ``toka`` CLI arguments."""

    model: str
    tokenizer: Optional[str] = None

    trust_remote_code: bool = False
    dtype: str = "bfloat16"
    rope_scaling: Optional[str] = None

    use_hydragen: bool = False
    hydragen_min_group_size: int = 32
    hydragen_min_prefix_len: int = 256

    enable_chosen_logprobs: bool = True
    max_topk_logprobs: Optional[int] = None

    port: int = 10210
    local_proc_name: str = "server"

    log_level: str = "INFO"
    log_procs: Optional[list[str]] = None
    uvicorn_log_level: str = "info"

    stats_report_seconds: float = 5.0
    statsd_server_url: Optional[str] = None

    page_size: int = 16
    kv_cache_num_tokens: int = 1024 * 128

    torch_compile: bool = False
    async_tp_threshold: Optional[int] = None

    max_tokens_per_forward: int = 8192
    max_seqs_per_forward: int = 1024
    prefill_round_up_multiple: int = 16

    scheduling_steps_ahead: int = 8
    stop_string_num_token_lookback: int = 5

    dp_size: int = 1
    pp_size: int = 1
    tp_size: int = 1
    pp_num_buffer_stages: int = 1

    track_early_stopping: bool = True
    early_stopping_buffer_size: int = 2048
    early_stopping_num_prediction_buckets: int = 1024
    early_stopping_initial_wait: int = 16
    early_stopping_init_mean: Optional[float] = None
    early_stopping_init_std: Optional[float] = None
    max_num_tokens_per_request: Optional[int] = None

    enable_precise_onboard: bool = True
    precise_onboard_batch_size: int = 128
    greedy_prefill: bool = True

    use_spec_allocation: bool = True
    spec_allocation_std_buffer_scale: float = 0.25
    spec_allocation_target_kv_cache_utilization: float = 1.0

    use_cudagraphs: bool = True
    cudagraph_max_size: int = 128
    cudagraph_step: int = 16
    cudagraph_max_kv_indices_per_seq: int = 32768

    allocator_sanity_checks: bool = False


def _run_benchmark_with_server(config: BenchmarkWithServerConfig):
    import socket
    import subprocess
    import requests as http_requests

    _EXCLUDED_CLI_FIELDS = {
        "wandb_enabled", "wandb_entity", "wandb_project", "wandb_run_name",
        "run_dir", "output_dir", "run_id", "launch_id", "script_id",
    }

    def _is_port_in_use(port: int) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("localhost", port))
                return False
            except OSError:
                return True

    def _find_available_port(start: int) -> int:
        port = start
        while _is_port_in_use(port):
            logger.info(f"Port {port} in use, trying {port + 1}")
            port += 1
        return port

    toka = config.tokasaurus
    port = _find_available_port(toka.port)

    # Build CLI command
    toka_cmd: list[str] = ["toka"]
    for field_name, field_value in toka.__dict__.items():
        if field_name in _EXCLUDED_CLI_FIELDS or field_value is None:
            continue
        if field_name == "port":
            field_value = port
        if field_name == "log_procs" and isinstance(field_value, list):
            field_value = ",".join(field_value)
        toka_cmd.append(f"{field_name}={field_value}")

    if config.conda_env:
        cmd = ["conda", "run", "--no-capture-output", "-n", config.conda_env] + toka_cmd
    else:
        cmd = toka_cmd

    logger.info(f"Starting Tokasaurus server: {' '.join(cmd)}")
    process = subprocess.Popen(cmd)

    try:
        # Wait for server readiness
        logger.info(f"Waiting for server on port {port}...")
        start_time = time.time()
        timeout = 300
        while time.time() - start_time < timeout:
            try:
                r = http_requests.get(f"http://localhost:{port}/ping", timeout=1.0)
                if r.json().get("message") == "pong":
                    logger.info("Tokasaurus server is ready!")
                    break
            except http_requests.RequestException:
                pass
            time.sleep(2)
        else:
            raise TimeoutError(f"Tokasaurus server failed to start within {timeout}s")

        # Patch the benchmark client URL + model name to point at the launched server
        from cartridges.clients.tokasaurus import TokasaurusClient

        bench = config.benchmark.model_copy(deep=True)
        if isinstance(bench.client, TokasaurusClient.Config):
            bench.client.url = f"http://localhost:{port}"
            bench.client.model_name = toka.model
        bench.run_dir = config.run_dir
        bench.run()
    finally:
        logger.info("Shutting down Tokasaurus server...")
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        logger.info("Tokasaurus server shut down")
