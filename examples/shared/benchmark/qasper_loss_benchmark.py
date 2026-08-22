#!/usr/bin/env python3
"""Full-context ICL loss benchmark for Qasper (QA -> MT).

Evaluates the open-ended Qasper parquet evals (not the MCQ/yes-no CSVs). Each
eval row is a conversation; the assistant message is retokenized and used as the
loss target, using the SAME soft-CE-vs-teacher-top-k scorer as `eval_forgetting.py`
(the shared implementation now lives in `examples/shared/evaluate/icl.py`).

Methods (all carry full-paper ICL context):

  - icl_QA_raw: QA full-paper context + raw model
  - icl_MT_raw: MT full-paper context + raw model
  - icl_QA_plus_MT: QA papers followed by MT papers + raw model
  - icl_MT_plus_QA: MT papers followed by QA papers + raw model
  - icl_MT_plus_QA_cartridge: MT full-paper context + Phase-1 QA cartridge

Pure-cartridge eval (no ICL context) now lives in `eval_forgetting.py`
(EVAL_MODE=cartridge) — the single source of truth for cartridge scoring — so
the old `cartridge_p1` / `cartridge_p2` methods were removed here.

Each condition is evaluated on both `qasper_eval_QA.parquet` and
`qasper_eval_MT.parquet`, for both Llama and Qwen by default.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from transformers import AutoTokenizer

try:
    from huggingface_hub import hf_hub_download
except ImportError:
    hf_hub_download = None  # type: ignore[assignment]

from cartridges.cache import AttnConfig, TrainableCache
from cartridges.data.qasper.resources import QASPERResource
from cartridges.datasets import MODEL_TO_MESSAGE_CONVERTER
from cartridges.models import FlexLlamaForCausalLM, FlexQwen3ForCausalLM, HFModelConfig
from cartridges.structs import Conversation, read_conversations
from cartridges.train import CacheAndModel  # noqa: F401 (kept for API compatibility)

# Single source of truth for the full-context ICL scorer + context builder.
import sys as _sys
_sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "train"))
from eval_icl import (  # noqa: E402
    build_qasper_system_prompt,
    evaluate_loss_chunked,
    load_model_and_tokenizer,
    _model_device,
)


QA_EVAL_URL = (
    "https://raw.githubusercontent.com/faridlazuarda/gated-continual-cartridges/"
    "tri_work_placeholder/data/qasper/eval/qasper_eval_QA.parquet"
)
MT_EVAL_URL = (
    "https://raw.githubusercontent.com/faridlazuarda/gated-continual-cartridges/"
    "tri_work_placeholder/data/qasper/eval/qasper_eval_MT.parquet"
)



@dataclass(frozen=True)
class ModelSpec:
    family: str
    model_name: str
    phase1_cartridge: str
    phase2_cartridge: str


DEFAULT_MODELS = [
    ModelSpec(
        family="llama",
        model_name="meta-llama/Llama-3.2-3B-Instruct",
        phase1_cartridge="qtris123/llama_qasper-QA-task_8192_512_no-cartridge_10-epochs",
        phase2_cartridge="qtris123/llama_qasper-MT-task_8192_10-epochs_with-cartridge_qasper-QA-task_8192_512",
    ),
    ModelSpec(
        family="qwen",
        model_name="Qwen/Qwen3-4B-Instruct-2507",
        phase1_cartridge="qtris123/qwen_qasper-QA-task_8192_512_no-cartridge_10-epochs",
        phase2_cartridge="qtris123/qwen_qasper-MT-task_8192_10-epochs_with-cartridge_qasper-QA-task_8192_512",
    ),
]


def ensure_eval_file(path: Path, url: str, *, download: bool) -> Path:
    path = path.expanduser()
    if path.exists():
        return path
    if not download:
        raise FileNotFoundError(
            f"{path} does not exist. Re-run with --download-evals or download {url}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url} -> {path}", flush=True)
    # Download to a temp file then atomically rename so concurrent workers never
    # observe a half-written parquet.
    tmp_path = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    urllib.request.urlretrieve(url, tmp_path)
    os.replace(tmp_path, path)
    return path




def resolve_cartridge_path(spec: str, *, hf_filename: str = "cache_last.pt") -> str:
    local = Path(spec).expanduser()
    if local.is_file():
        return str(local.resolve())

    if hf_hub_download is None:
        raise ImportError("pip install huggingface_hub")

    parts = [p for p in spec.strip("/").split("/") if p]
    if len(parts) == 2 and not parts[-1].endswith(".pt"):
        return hf_hub_download(repo_id=f"{parts[0]}/{parts[1]}", filename=hf_filename)
    if len(parts) >= 3 and parts[-1].endswith(".pt"):
        return hf_hub_download(repo_id=f"{parts[0]}/{parts[1]}", filename="/".join(parts[2:]))
    raise FileNotFoundError(
        f"Cartridge not found: {spec!r}. Use a local .pt path or HF repo id."
    )


def load_cache(spec: Optional[str], device: str) -> Optional[TrainableCache]:
    if spec is None:
        return None
    path = resolve_cartridge_path(spec)
    return TrainableCache.from_pretrained(path, device=device).to(device).to(torch.bfloat16)














def benchmark_model(
    spec: ModelSpec,
    *,
    qa_eval: Path,
    mt_eval: Path,
    out_dir: Path,
    device: str,
    max_context_tokens: Optional[int],
    only_methods: Optional[set[str]] = None,
    multi_gpu: bool = False,
    prefill_chunk_size: int = 2048,
) -> list[dict[str, Any]]:
    # Method -> (context_topic, cartridge). context_topic=None means no ICL context.
    # ICL / hybrid-context methods only. Pure-cartridge eval now lives in
    # eval_forgetting.py (EVAL_MODE=cartridge) — the single source of truth for
    # cartridge scoring — so the old cartridge_p1/cartridge_p2 methods (which
    # duplicated it) are removed here.
    condition_specs = [
        ("icl_QA_raw", "QA", None),
        ("icl_MT_raw", "MT", None),
        ("icl_QA_plus_MT", "QA+MT", None),
        ("icl_MT_plus_QA", "MT+QA", None),
        ("icl_MT_plus_QA_cartridge", "MT", spec.phase1_cartridge),
    ]
    if only_methods is not None:
        condition_specs = [c for c in condition_specs if c[0] in only_methods]
    if not condition_specs:
        print(f"No conditions selected for {spec.family}; skipping.", flush=True)
        return []

    print(f"\n=== Loading {spec.family}: {spec.model_name} (multi_gpu={multi_gpu}) ===", flush=True)
    model, tokenizer = load_model_and_tokenizer(spec.model_name, device, multi_gpu=multi_gpu)
    model_dev = _model_device(model)

    # Only build the full-paper ICL system prompts that are actually needed.
    needed_topics = {topic for _, topic, _ in condition_specs if topic is not None}
    system_prompts = {
        topic: build_qasper_system_prompt(
            topic, tokenizer=tokenizer, max_context_tokens=max_context_tokens
        )
        for topic in needed_topics
    }

    conditions = [
        {
            "method": method,
            "context_topic": topic,
            "system_prompt": system_prompts.get(topic),
            "cartridge": cartridge,
        }
        for method, topic, cartridge in condition_specs
    ]

    rows: list[dict[str, Any]] = []
    for condition in conditions:
        cache = load_cache(condition["cartridge"], str(model_dev))
        for eval_name, eval_path in [("QA", qa_eval), ("MT", mt_eval)]:
            desc = f"{spec.family}|{condition['method']}|{eval_name}"
            print(f"Evaluating {desc}", flush=True)
            # All remaining methods carry a full-context system_prompt → always chunked.
            metrics = evaluate_loss_chunked(
                model=model,
                cartridge=cache,
                tokenizer=tokenizer,
                eval_path=eval_path,
                system_prompt=condition["system_prompt"],
                device=str(model_dev),
                desc=desc,
                prefill_chunk_size=prefill_chunk_size,
            )
            rows.append(
                {
                    "model_family": spec.family,
                    "model_name": spec.model_name,
                    "method": condition["method"],
                    "eval": eval_name,
                    "eval_path": str(eval_path),
                    "context_topic": condition["context_topic"],
                    "cartridge": condition["cartridge"],
                    **metrics,
                }
            )
            print(
                f"  {desc}: loss={metrics['loss']:.4f} ppl={metrics['perplexity']:.4f}",
                flush=True,
            )
        if cache is not None:
            del cache

    out_dir.mkdir(parents=True, exist_ok=True)
    if only_methods is not None and len(only_methods) == 1:
        tag = f"{spec.family}__{next(iter(only_methods))}"
    else:
        tag = spec.family
    (out_dir / f"{tag}_loss_rows.json").write_text(
        json.dumps(rows, indent=2),
        encoding="utf-8",
    )
    return rows


def print_wide_table(rows: list[dict[str, Any]]) -> None:
    keys = sorted({(r["model_family"], r["method"]) for r in rows})
    print("\nPerplexity table")
    print(f"{'model':<8} {'method':<28} {'QA eval':>12} {'MT eval':>12}")
    print("-" * 64)
    for model_family, method in keys:
        qa = next((r for r in rows if r["model_family"] == model_family and r["method"] == method and r["eval"] == "QA"), None)
        mt = next((r for r in rows if r["model_family"] == model_family and r["method"] == method and r["eval"] == "MT"), None)
        qa_ppl = f"{qa['perplexity']:.4f}" if qa else "N/A"
        mt_ppl = f"{mt['perplexity']:.4f}" if mt else "N/A"
        print(f"{model_family:<8} {method:<28} {qa_ppl:>12} {mt_ppl:>12}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Qasper QA -> MT loss benchmark.")
    parser.add_argument(
        "--qa-eval",
        type=Path,
        default=Path("data/qasper/eval/qasper_eval_QA.parquet"),
    )
    parser.add_argument(
        "--mt-eval",
        type=Path,
        default=Path("data/qasper/eval/qasper_eval_MT.parquet"),
    )
    parser.add_argument("--download-evals", action="store_true")
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="Only ensure eval parquets are present, then exit (no model load).",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/qasper_loss_benchmark"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--multi-gpu",
        action="store_true",
        help="Shard model weights across visible GPUs (device_map=auto).",
    )
    parser.add_argument(
        "--prefill-chunk-size",
        type=int,
        default=2048,
        help="Chunk size for full-panel ICL prefill.",
    )
    parser.add_argument("--max-context-tokens", type=int, default=None)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["llama", "qwen"],
        default=["llama", "qwen"],
        help="Model families to evaluate.",
    )
    parser.add_argument(
        "--only-methods",
        nargs="+",
        choices=[
            "icl_QA_raw",
            "icl_MT_raw",
            "icl_QA_plus_MT",
            "icl_MT_plus_QA",
            "icl_MT_plus_QA_cartridge",
        ],
        default=None,
        help="Restrict to specific ICL methods (default: all five). "
        "Cartridge-only eval moved to eval_forgetting.py (EVAL_MODE=cartridge).",
    )
    args = parser.parse_args()

    download = args.download_evals or args.download_only
    qa_eval = ensure_eval_file(args.qa_eval, QA_EVAL_URL, download=download)
    mt_eval = ensure_eval_file(args.mt_eval, MT_EVAL_URL, download=download)

    if args.download_only:
        print(f"Eval files ready:\n  QA: {qa_eval}\n  MT: {mt_eval}")
        return

    only_methods = set(args.only_methods) if args.only_methods else None
    specs = [spec for spec in DEFAULT_MODELS if spec.family in set(args.models)]
    all_rows: list[dict[str, Any]] = []
    for spec in specs:
        all_rows.extend(
            benchmark_model(
                spec,
                qa_eval=qa_eval,
                mt_eval=mt_eval,
                out_dir=args.output_dir,
                device=args.device,
                max_context_tokens=args.max_context_tokens,
                only_methods=only_methods,
                multi_gpu=args.multi_gpu,
                prefill_chunk_size=args.prefill_chunk_size,
            )
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "qasper_loss_benchmark_summary.json"
    summary_path.write_text(json.dumps(all_rows, indent=2), encoding="utf-8")
    print_wide_table(all_rows)
    print(f"\nWrote summary to {summary_path}")


if __name__ == "__main__":
    main()
