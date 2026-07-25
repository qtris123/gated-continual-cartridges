#!/usr/bin/env python3
"""Loss/perplexity benchmark for Qasper QA -> MT cartridges and ICL.

This script evaluates the open-ended Qasper parquet evals, not the MCQ/yes-no
CSVs.  Each eval row is a conversation; the assistant message is retokenized and
used as the loss target, matching the existing `eval_forgetting.py` path.

Requested benchmark table:

  - icl_QA_raw: QA full-paper context + raw model
  - icl_MT_raw: MT full-paper context + raw model
  - icl_QA_plus_MT: QA papers followed by MT papers + raw model
  - icl_MT_plus_QA: MT papers followed by QA papers + raw model
  - icl_MT_plus_QA_cartridge: MT full-paper context + Phase-1 QA cartridge
  - cartridge_p1: Phase-1 QA cartridge only
  - cartridge_p2: Phase-2 MT continual cartridge only

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
from cartridges.train import CacheAndModel


QA_EVAL_URL = (
    "https://raw.githubusercontent.com/faridlazuarda/gated-continual-cartridges/"
    "tri_work_placeholder/examples/qasper2/qasper_eval_QA.parquet"
)
MT_EVAL_URL = (
    "https://raw.githubusercontent.com/faridlazuarda/gated-continual-cartridges/"
    "tri_work_placeholder/examples/qasper2/qasper_eval_MT.parquet"
)

QASPER_ICL_SYSTEM_TEMPLATE = """\
Please reference the scientific papers included below to answer questions about them.

{content}
"""


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


def build_qasper_system_prompt(
    topic: str,
    *,
    tokenizer,
    max_context_tokens: Optional[int] = None,
) -> str:
    topics = topic.split("+")
    panels = [
        QASPERResource(QASPERResource.Config(topic=panel_topic)).to_string()
        for panel_topic in topics
    ]
    if len(topics) == 1:
        ctx_text = panels[0]
    else:
        ctx_text = "\n\n".join(
            f'<topic-panel topic="{panel_topic}">\n{panel}\n</topic-panel>'
            for panel_topic, panel in zip(topics, panels)
        )
    if max_context_tokens is not None:
        ctx_text = tokenizer.decode(
            tokenizer.encode(ctx_text)[:max_context_tokens],
            add_special_tokens=False,
            max_length=999_999_999,
            truncation=True,
        )
    return QASPER_ICL_SYSTEM_TEMPLATE.format(content=ctx_text)


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


def _model_head_dim(model) -> int:
    if hasattr(model.config, "head_dim"):
        return int(model.config.head_dim)
    return model.config.hidden_size // model.config.num_attention_heads


def _model_device(model) -> torch.device:
    return next(model.parameters()).device


def _make_prompt_cache(
    model,
    cartridge: Optional[TrainableCache] = None,
    device: Optional[torch.device] = None,
) -> TrainableCache:
    if cartridge is not None:
        return cartridge
    dev = device or _model_device(model)
    return TrainableCache(
        config=AttnConfig(
            n_layers=model.config.num_hidden_layers,
            n_heads=model.config.num_key_value_heads,
            head_dim=_model_head_dim(model),
        ),
    ).to(dev)


def load_model_and_tokenizer(model_name: str, device: str, *, multi_gpu: bool = False):
    model_cls = FlexQwen3ForCausalLM if "qwen" in model_name.lower() else FlexLlamaForCausalLM
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    load_kwargs: dict[str, Any] = {}
    if multi_gpu:
        load_kwargs["device_map"] = "auto"
    model = HFModelConfig(
        pretrained_model_name_or_path=model_name,
        model_cls=model_cls,
        load_kwargs=load_kwargs,
    ).instantiate()
    if multi_gpu:
        model = model.to(torch.bfloat16)
    else:
        model = model.to(device).to(torch.bfloat16)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model, tokenizer


def evaluate_loss_chunked(
    *,
    model,
    cartridge: Optional[TrainableCache],
    tokenizer,
    eval_path: Path,
    system_prompt: str,
    device: str,
    desc: str,
    prefill_chunk_size: int = 2048,
) -> dict[str, Any]:
    """Score assistant tokens with full ICL context via chunked generate-mode prefill."""
    conversations = read_conversations(str(eval_path))
    converter = MODEL_TO_MESSAGE_CONVERTER[tokenizer.name_or_path.lower()]
    model_dev = _model_device(model)

    total_loss = torch.tensor(0.0, device=model_dev)
    total_tokens = torch.tensor(0, device=model_dev)
    max_sequence_tokens = 0

    with torch.inference_mode():
        for convo in tqdm(conversations, total=len(conversations), desc=desc, leave=False):
            messages = [
                Conversation.Message(
                    role="system",
                    content=system_prompt,
                    token_ids=None,
                    top_logprobs=None,
                ),
                *convo.messages,
            ]
            element = converter(messages, retokenize=True, tokenizer=tokenizer)
            max_sequence_tokens = max(max_sequence_tokens, len(element.input_ids))

            kv_cache = _make_prompt_cache(model, cartridge, device=model_dev)
            if cartridge is not None:
                kv_cache.clear()

            target_idxs = element.topk_token_idxs
            target_ids = element.topk_token_ids
            target_logprobs = element.topk_logprobs
            assistant_start = int(target_idxs.min().item())

            last_logits: Optional[torch.Tensor] = None
            pos = 0
            while pos < assistant_start:
                end = min(pos + prefill_chunk_size, assistant_start)
                chunk_ids = element.input_ids[pos:end]
                chunk_len = len(chunk_ids)
                chunk_seq_ids = torch.zeros(chunk_len, dtype=torch.long)
                chunk_pos_ids = torch.arange(pos, pos + chunk_len, dtype=torch.long)

                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                    outputs = model(
                        input_ids=chunk_ids.to(model_dev),
                        seq_ids=chunk_seq_ids.to(model_dev),
                        position_ids=chunk_pos_ids.to(model_dev),
                        past_key_values=kv_cache,
                        use_cache=True,
                        mode="generate",
                    )
                last_logits = outputs.logits[0, -1, :].contiguous()
                pos = end

            for tidx, tid, hard_logprob in zip(target_idxs, target_ids, target_logprobs):
                tidx = int(tidx.item())
                tid = int(tid.item())

                if tidx == assistant_start:
                    logits = last_logits
                else:
                    prev_id = element.input_ids[tidx - 1 : tidx]
                    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                        outputs = model(
                            input_ids=prev_id.to(model_dev),
                            seq_ids=torch.zeros(1, dtype=torch.long, device=model_dev),
                            position_ids=torch.tensor([tidx - 1], dtype=torch.long, device=model_dev),
                            past_key_values=kv_cache,
                            use_cache=True,
                            mode="generate",
                        )
                    logits = outputs.logits[0, -1, :].contiguous()

                pred_logp = F.log_softmax(logits.float(), dim=-1)[tid]
                ce = -hard_logprob.to(model_dev).exp() * pred_logp
                total_loss += ce
                total_tokens += 1

            if cartridge is not None:
                kv_cache.clear()
            else:
                kv_cache.clear()

    loss = float((total_loss / total_tokens).item())
    return {
        "loss": loss,
        "perplexity": math.exp(loss),
        "num_target_tokens": int(total_tokens.item()),
        "num_examples": len(conversations),
        "max_sequence_tokens": max_sequence_tokens,
        "system_prompt_tokens": len(tokenizer.encode(system_prompt, add_special_tokens=False)),
        "context_mode": "full_topic_panel_chunked",
        "prefill_chunk_size": prefill_chunk_size,
    }


def evaluate_loss(
    *,
    model,
    cache: Optional[TrainableCache],
    tokenizer,
    eval_path: Path,
    system_prompt: Optional[str],
    device: str,
    desc: str,
) -> dict[str, Any]:
    conversations = read_conversations(str(eval_path))
    converter = MODEL_TO_MESSAGE_CONVERTER[tokenizer.name_or_path.lower()]
    wrapped_model = CacheAndModel(cache, model)

    total_loss = torch.tensor(0.0, device=device)
    total_tokens = torch.tensor(0, device=device)
    max_sequence_tokens = 0

    with torch.inference_mode():
        for convo in tqdm(conversations, total=len(conversations), desc=desc, leave=False):
            messages = convo.messages
            if system_prompt is not None:
                messages = [
                    Conversation.Message(
                        role="system",
                        content=system_prompt,
                        token_ids=None,
                        top_logprobs=None,
                    ),
                    *messages,
                ]
            element = converter(messages, retokenize=True, tokenizer=tokenizer)
            max_sequence_tokens = max(max_sequence_tokens, len(element.input_ids))
            element_ids = torch.zeros_like(element.input_ids)
            position_ids = torch.arange(len(element.input_ids), dtype=torch.long)

            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = wrapped_model(
                    input_ids=element.input_ids.to(device),
                    seq_ids=element_ids.to(device),
                    position_ids=position_ids.to(device),
                )
                target_idxs = element.topk_token_idxs.to(device) - 1
                target_ids = element.topk_token_ids.to(device)
                pred_logprobs = F.log_softmax(outputs.logits, dim=-1)[
                    0,
                    target_idxs,
                    target_ids,
                ]
                ce_by_token = -element.topk_logprobs.to(device).exp() * pred_logprobs
                total_loss += ce_by_token.sum()
                total_tokens += ce_by_token.shape[0]

            if cache is not None:
                cache.clear()

    loss = float((total_loss / total_tokens).item())
    return {
        "loss": loss,
        "perplexity": math.exp(loss),
        "num_target_tokens": int(total_tokens.item()),
        "num_examples": len(conversations),
        "max_sequence_tokens": max_sequence_tokens,
        "system_prompt_tokens": 0,
        "context_mode": "cartridge_only",
    }


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
    condition_specs = [
        ("icl_QA_raw", "QA", None),
        ("icl_MT_raw", "MT", None),
        ("icl_QA_plus_MT", "QA+MT", None),
        ("icl_MT_plus_QA", "MT+QA", None),
        ("icl_MT_plus_QA_cartridge", "MT", spec.phase1_cartridge),
        ("cartridge_p1", None, spec.phase1_cartridge),
        ("cartridge_p2", None, spec.phase2_cartridge),
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
            if condition["system_prompt"] is not None:
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
            else:
                metrics = evaluate_loss(
                    model=model,
                    cache=cache,
                    tokenizer=tokenizer,
                    eval_path=eval_path,
                    system_prompt=None,
                    device=str(model_dev),
                    desc=desc,
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
        default=Path("examples/qasper2/qasper_eval_QA.parquet"),
    )
    parser.add_argument(
        "--mt-eval",
        type=Path,
        default=Path("examples/qasper2/qasper_eval_MT.parquet"),
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
            "cartridge_p1",
            "cartridge_p2",
        ],
        default=None,
        help="Restrict to specific benchmark methods (default: all five).",
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
