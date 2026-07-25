#!/usr/bin/env python3
"""
ICL (in-context learning) evaluation on Qasper QA / MT CSV datasets.

Evaluates the raw model with full paper context in the system prompt, optionally
with a cartridge attached.  Scoring matches ``cartridge_eval.py``.

Benchmark conditions (for QA -> MT continual learning):

1. **stage1_icl** — QA-topic full context in system prompt, raw model (no cartridge)
2. **stage2_icl_plus_p1_cartridge** — MT-topic full context + Phase-1 cartridge

Compare these against ``cartridge_eval.py`` (compressed context via cartridge only).

Usage::

    # Stage-1 ICL baseline (QA context, raw model)
    python icl_eval.py \\
        --context-topic QA \\
        --datasets examples/qasper2/qasper_QA-mcq.csv:mcq \\
                     examples/qasper2/qasper_MT-mcq.csv:mcq \\
        --model meta-llama/Llama-3.2-3B-Instruct \\
        --output-dir ./eval_out/stage1_icl

    # Stage-2 ICL + Phase-1 cartridge (MT context + cartridge)
    python icl_eval.py \\
        --context-topic MT \\
        --cartridges qtris123/llama_qasper-QA-task_8192_512_no-cartridge_10-epochs \\
        --datasets examples/qasper2/qasper_QA-mcq.csv:mcq \\
                     examples/qasper2/qasper_MT-mcq.csv:mcq \\
        --model meta-llama/Llama-3.2-3B-Instruct \\
        --output-dir ./eval_out/stage2_icl_plus_p1
"""

from __future__ import annotations

import os

os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")

import argparse
import asyncio
import json
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from cartridges.clients.tokasaurus import TokasaurusClient
from cartridges.clients.base import CartridgeConfig
from cartridges.data.qasper.resources import QASPERResource

from cartridge_eval import (
    DIST_TOPK,
    _autocast_context,
    _build_unanswered_row,
    _is_thinking_model,
    _resolve_chat_template,
    _score_and_build_row,
    build_answer_token_maps,
    build_user_message,
    compute_global_stats,
    extract_answer_logprobs_from_topk,
    find_answer_position,
    infer_question_type,
    load_cache,
    load_model_and_tokenizer,
    option_logprobs_from_logits,
    parse_dataset_arg,
    resolve_cartridge_source,
    topk_distribution_from_server,
    topk_logprobs_non_whitespace,
)
from helpers import load_eval_records


QASPER_ICL_SYSTEM_TEMPLATE = """\
Please reference the scientific papers included below to answer questions about them.

{content}
"""


def build_qasper_context(topic: str, *, max_context_tokens: Optional[int] = None, tokenizer=None) -> str:
    """Load all papers for a Qasper topic (QA or MT) as a single context string."""
    resource = QASPERResource(QASPERResource.Config(topic=topic))
    ctx_text = resource.to_string()
    if max_context_tokens is not None and tokenizer is not None:
        ctx_text = tokenizer.decode(
            tokenizer.encode(ctx_text)[:max_context_tokens],
            add_special_tokens=False,
            max_length=999_999_999,
            truncation=True,
        )
    return ctx_text


def build_system_prompt(topic: str, *, max_context_tokens: Optional[int] = None, tokenizer=None) -> str:
    content = build_qasper_context(topic, max_context_tokens=max_context_tokens, tokenizer=tokenizer)
    return QASPER_ICL_SYSTEM_TEMPLATE.format(content=content)


def _encode_chat_prompt(
    tokenizer,
    messages: List[Dict[str, str]],
    device: str,
    model_name: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    kwargs: Dict[str, object] = {}
    if _is_thinking_model(model_name):
        kwargs["enable_thinking"] = False

    template = _resolve_chat_template(tokenizer)
    if template is not None:
        kwargs["chat_template"] = template

    result = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        **kwargs,
    )
    if isinstance(result, torch.Tensor):
        input_ids = result
    else:
        input_ids = result["input_ids"]
    flat = input_ids.to(device).flatten()
    seq_ids = torch.zeros_like(flat)
    position_ids = torch.arange(flat.shape[0], device=device)
    return flat, seq_ids, position_ids


def find_answer_token_step_icl(
    model,
    tokenizer,
    system_content: str,
    user_content: str,
    model_name: str,
    cache,
    device: str,
    record: dict,
    *,
    cot: bool,
    max_new_tokens: int = 256,
    token_maps: Optional[Tuple[Dict[str, Set[int]], Set[int]]] = None,
    debug: bool = False,
) -> Tuple[Optional[torch.Tensor], Dict[str, object]]:
    """Greedily decode with system+user messages and optional cartridge cache."""
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]
    flat, seq_ids, position_ids = _encode_chat_prompt(tokenizer, messages, device, model_name)

    next_input_ids = flat
    next_seq_ids = seq_ids
    next_position_ids = position_ids
    last_position = int(position_ids[-1].item())

    generated_ids: List[int] = []
    generated_text = ""
    answer_zone_active = not cot

    if token_maps is not None:
        label_to_ids, valid_ids = token_maps
    else:
        label_to_ids, valid_ids = build_answer_token_maps(
            tokenizer, infer_question_type(record)
        )

    found_logits: Optional[torch.Tensor] = None
    found_token_id: Optional[int] = None
    found_label: Optional[str] = None
    found_step: Optional[int] = None

    with torch.inference_mode():
        for step in range(max_new_tokens):
            with _autocast_context(device):
                outputs = model(
                    input_ids=next_input_ids,
                    seq_ids=next_seq_ids,
                    position_ids=next_position_ids,
                    past_key_values=cache,
                    use_cache=True,
                    mode="generate",
                )

            logits = outputs.logits[0, -1, :].contiguous()
            next_token_id = int(torch.argmax(logits).item())
            next_token_text = tokenizer.decode(
                [next_token_id], skip_special_tokens=False
            )

            if answer_zone_active and next_token_id in valid_ids:
                found_logits = logits
                found_token_id = next_token_id
                found_step = step
                for label, ids in label_to_ids.items():
                    if next_token_id in ids:
                        found_label = label
                        break
                generated_ids.append(next_token_id)
                generated_text += next_token_text
                break

            generated_ids.append(next_token_id)
            generated_text += next_token_text

            if cot and not answer_zone_active and "Answer:" in generated_text:
                answer_zone_active = True

            next_input_ids = torch.tensor(
                [next_token_id], device=device, dtype=flat.dtype
            )
            next_seq_ids = torch.zeros_like(next_input_ids)
            last_position += 1
            next_position_ids = torch.tensor(
                [last_position], device=device, dtype=position_ids.dtype
            )

    if cache is not None:
        cache.clear()

    if found_logits is None:
        if debug:
            print(
                f"[unanswered] id={record.get('id', '?')!r} cot={cot} "
                f"steps={len(generated_ids)}/{max_new_tokens}\n"
                f"  text: {generated_text!r}",
                file=sys.stderr,
                flush=True,
            )
        return None, {
            "unanswered": True,
            "cot": cot,
            "max_new_tokens": max_new_tokens,
            "answer_zone_activated": answer_zone_active,
            "generated_text_scan": generated_text,
            "generated_token_ids_scan": generated_ids,
            "mode": "icl",
        }

    answer_token_text = tokenizer.decode(
        [found_token_id], skip_special_tokens=False
    )
    prefix = generated_text
    if generated_text.endswith(answer_token_text) and answer_token_text:
        prefix = generated_text[: -len(answer_token_text)]

    return found_logits, {
        "unanswered": False,
        "cot": cot,
        "max_new_tokens": max_new_tokens,
        "answer_zone_activated": answer_zone_active,
        "generated_prefix_before_answer": prefix,
        "generated_text_through_answer_token": generated_text,
        "generated_token_ids_through_answer_token": generated_ids,
        "answer_token_id": found_token_id,
        "answer_token_text": answer_token_text,
        "answer_label_by_generated_token": found_label,
        "answer_generation_step": found_step,
        "mode": "icl",
    }


def evaluate_icl_local(
    *,
    model,
    tokenizer,
    model_name: str,
    system_prompt: str,
    cartridge_path: Optional[str],
    eval_records: List[dict],
    device: str,
    cot: bool = False,
    max_answer_scan_tokens: int = 256,
    debug: bool = False,
    desc: str = "icl_eval",
) -> List[dict]:
    token_maps_yn = build_answer_token_maps(tokenizer, "yes_no")
    token_maps_mcq = build_answer_token_maps(tokenizer, "mcq")
    cache = load_cache(cartridge_path, device) if cartridge_path else None

    rows: List[dict] = []
    n_correct = 0
    n_answered = 0

    pbar = tqdm(
        enumerate(eval_records),
        total=len(eval_records),
        desc=desc,
        unit="ex",
        dynamic_ncols=True,
    )

    for idx, record in pbar:
        user_msg = build_user_message(record, cot=cot)
        qtype = infer_question_type(record)
        token_maps = token_maps_yn if qtype == "yes_no" else token_maps_mcq

        logits, answer_meta = find_answer_token_step_icl(
            model=model,
            tokenizer=tokenizer,
            system_content=system_prompt,
            user_content=user_msg,
            model_name=model_name,
            cache=cache,
            device=device,
            record=record,
            cot=cot,
            max_new_tokens=max_answer_scan_tokens,
            token_maps=token_maps,
            debug=debug,
        )

        if logits is None:
            rows.append(_build_unanswered_row(idx, record, answer_meta))
            pbar.set_postfix(acc=f"{n_correct}/{n_answered}", miss=len(rows) - n_answered)
            continue

        n_answered += 1
        choice_logprobs = option_logprobs_from_logits(logits, tokenizer, record)
        dist_topk, _ = topk_logprobs_non_whitespace(logits, tokenizer, k=DIST_TOPK)
        row, correct = _score_and_build_row(idx, record, choice_logprobs, dist_topk, answer_meta)
        n_correct += int(bool(correct))
        rows.append(row)
        pbar.set_postfix(acc=f"{n_correct}/{n_answered}", ok=correct, pred=row["generated_answer"])

    return rows


def evaluate_icl_tokasaurus(
    *,
    client: TokasaurusClient,
    tokenizer,
    system_prompt: str,
    cartridge_id: Optional[str],
    eval_records: List[dict],
    cot: bool = False,
    max_completion_tokens: int = 256,
    top_logprobs: int = 20,
    batch_size: int = 16,
    debug: bool = False,
    desc: str = "icl_eval",
) -> List[dict]:
    token_maps_yn = build_answer_token_maps(tokenizer, "yes_no")
    token_maps_mcq = build_answer_token_maps(tokenizer, "mcq")
    decode_fn = lambda ids: tokenizer.decode(ids, skip_special_tokens=False)

    cartridge_cfg = None
    if cartridge_id:
        cartridge_cfg = [CartridgeConfig(id=cartridge_id, source="huggingface").model_dump()]

    chats_and_records: List[Tuple[List[Dict[str, str]], dict, int]] = []
    for idx, record in enumerate(eval_records):
        user_msg = build_user_message(record, cot=cot)
        chat = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ]
        chats_and_records.append((chat, record, idx))

    rows: List[dict] = [None] * len(eval_records)  # type: ignore[list-item]
    n_correct = 0
    n_answered = 0
    n_unanswered = 0

    batches = [
        chats_and_records[i : i + batch_size]
        for i in range(0, len(chats_and_records), batch_size)
    ]

    pbar = tqdm(batches, desc=desc, unit="batch", dynamic_ncols=True)

    for batch in pbar:
        batch_chats = [item[0] for item in batch]
        batch_records = [item[1] for item in batch]
        batch_indices = [item[2] for item in batch]

        chat_kwargs: Dict[str, Any] = {
            "chats": batch_chats,
            "max_completion_tokens": max_completion_tokens,
            "temperature": 0.0,
            "top_logprobs": top_logprobs,
        }
        if cartridge_cfg is not None:
            chat_kwargs["cartridges"] = cartridge_cfg

        response = asyncio.run(client.chat(**chat_kwargs))

        for i, sample in enumerate(response.samples):
            record = batch_records[i]
            idx = batch_indices[i]
            user_msg = batch_chats[i][1]["content"]
            qtype = infer_question_type(record)
            label_to_ids, valid_ids = (
                token_maps_yn if qtype == "yes_no" else token_maps_mcq
            )

            if sample.token_ids is None or sample.top_logprobs is None:
                n_unanswered += 1
                rows[idx] = _build_unanswered_row(idx, record, {
                    "unanswered": True,
                    "failure_reason": "no_token_ids_or_logprobs_from_server",
                    "generated_text": sample.text or "",
                    "mode": "icl",
                })
                rows[idx]["user_message"] = user_msg
                continue

            ans_pos = find_answer_position(
                sample.token_ids, valid_ids, decode_fn, cot=cot
            )

            if ans_pos is None:
                n_unanswered += 1
                generated_text = decode_fn(sample.token_ids) if sample.token_ids else ""
                if debug:
                    print(
                        f"[unanswered] id={record.get('id', '?')!r} "
                        f"tokens={len(sample.token_ids)} text={generated_text!r}",
                        file=sys.stderr, flush=True,
                    )
                rows[idx] = _build_unanswered_row(idx, record, {
                    "unanswered": True,
                    "failure_reason": "no_answer_token_in_completion",
                    "cot": cot,
                    "generated_text_scan": generated_text,
                    "generated_token_ids_scan": sample.token_ids,
                    "mode": "icl",
                })
                rows[idx]["user_message"] = user_msg
                continue

            n_answered += 1
            choice_logprobs = extract_answer_logprobs_from_topk(
                sample.top_logprobs, ans_pos, label_to_ids
            )
            dist_topk = topk_distribution_from_server(
                sample.top_logprobs, ans_pos, tokenizer
            )

            ans_token_id = sample.token_ids[ans_pos]
            ans_token_text = decode_fn([ans_token_id])
            ans_label = None
            for label, ids in label_to_ids.items():
                if ans_token_id in ids:
                    ans_label = label
                    break

            answer_meta = {
                "unanswered": False,
                "cot": cot,
                "answer_token_id": ans_token_id,
                "answer_token_text": ans_token_text,
                "answer_label_by_generated_token": ans_label,
                "answer_generation_step": ans_pos,
                "generated_text": sample.text or "",
                "backend": "tokasaurus",
                "mode": "icl",
            }

            row, correct = _score_and_build_row(
                idx, record, choice_logprobs, dist_topk, answer_meta
            )
            n_correct += int(bool(correct))
            row["user_message"] = user_msg
            rows[idx] = row

        pbar.set_postfix(
            acc=f"{n_correct}/{n_answered}",
            miss=n_unanswered,
        )

    return rows


def run_all_icl_evaluations(
    *,
    context_topic: str,
    cartridge_specs: Optional[List[str]],
    dataset_configs: List[Tuple[str, str]],
    model_name: str,
    backend: str = "local",
    device: str = "cuda",
    cot: bool = False,
    max_answer_scan_tokens: int = 256,
    max_context_tokens: Optional[int] = None,
    num_eval_questions: Optional[int] = None,
    hf_cartridge_filename: str = "cache_last.pt",
    hf_username: Optional[str] = None,
    debug: bool = False,
    url: Optional[str] = None,
    top_logprobs: int = 20,
    batch_size: int = 16,
) -> List[Dict[str, Any]]:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    system_prompt = build_system_prompt(
        context_topic,
        max_context_tokens=max_context_tokens,
        tokenizer=tokenizer,
    )

    if backend == "local":
        model, tokenizer = load_model_and_tokenizer(model_name, device)
        client = None
    elif backend == "tokasaurus":
        if url is None:
            raise ValueError("--url is required for the tokasaurus backend")
        client = TokasaurusClient(TokasaurusClient.Config(
            url=url,
            model_name=model_name,
        ))
        model = None
    else:
        raise ValueError(f"Unknown backend: {backend!r}")

    # No cartridge → one ICL condition; with cartridges → one result per cartridge
    cart_specs = cartridge_specs or [None]

    results: List[Dict[str, Any]] = []

    for cart_spec in cart_specs:
        if cart_spec is None:
            cart_label = "none"
            cart_local_path = None
        elif backend == "local":
            _ckpt, cart_local_path, cart_label = resolve_cartridge_source(
                cart_spec,
                default_hf_filename=hf_cartridge_filename,
                hf_username=hf_username,
            )
            del _ckpt
        else:
            cart_label = cart_spec.strip()
            cart_local_path = None

        condition = f"icl_{context_topic}"
        if cart_spec is not None:
            condition += f"+cartridge"

        for ds_path_str, ds_type in dataset_configs:
            ds_path = Path(ds_path_str)
            records = load_eval_records(ds_path, ds_type)
            if num_eval_questions is not None:
                records = records[:num_eval_questions]

            desc = f"{condition}|{Path(cart_label).name}|{ds_path.stem}"
            print(
                f"\n{'='*72}\n"
                f"Backend       : {backend}\n"
                f"Condition     : {condition}\n"
                f"Context topic : {context_topic}\n"
                f"Cartridge     : {cart_label}\n"
                f"Dataset       : {ds_path} ({ds_type}, {len(records)} questions)\n"
                f"{'='*72}",
                flush=True,
            )

            if backend == "local":
                eval_rows = evaluate_icl_local(
                    model=model,
                    tokenizer=tokenizer,
                    model_name=model_name,
                    system_prompt=system_prompt,
                    cartridge_path=cart_local_path,
                    eval_records=records,
                    device=device,
                    cot=cot,
                    max_answer_scan_tokens=max_answer_scan_tokens,
                    debug=debug,
                    desc=desc,
                )
            else:
                eval_rows = evaluate_icl_tokasaurus(
                    client=client,
                    tokenizer=tokenizer,
                    system_prompt=system_prompt,
                    cartridge_id=cart_label if cart_spec else None,
                    eval_records=records,
                    cot=cot,
                    max_completion_tokens=max_answer_scan_tokens,
                    top_logprobs=top_logprobs,
                    batch_size=batch_size,
                    debug=debug,
                    desc=desc,
                )

            stats = compute_global_stats(eval_rows)

            print(
                f"  -> accuracy={stats['accuracy']:.4f}  "
                f"avg_gt_ppl={stats['avg_gt_answer_perplexity']}  "
                f"gt_missing_topk={stats['num_gt_missing_from_topk']}"
                if stats["accuracy"] is not None
                else "  -> no answered questions",
                flush=True,
            )

            results.append({
                "condition": condition,
                "context_topic": context_topic,
                "cartridge": cart_label,
                "dataset": str(ds_path),
                "dataset_type": ds_type,
                "global_stats": stats,
                "eval_results": eval_rows,
            })

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ICL evaluation on Qasper (full context in system prompt).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--context-topic", type=str, required=True, choices=["QA", "MT"],
        help="Qasper topic whose papers are placed in the system prompt.",
    )
    parser.add_argument(
        "--backend", type=str, default="local", choices=["local", "tokasaurus"],
        help="Inference backend.",
    )
    parser.add_argument(
        "--cartridges", nargs="*", default=None,
        help="Optional cartridge specs (HF repo or local .pt). Omit for raw-model ICL.",
    )
    parser.add_argument(
        "--datasets", nargs="+", required=True,
        help="Dataset specs as path:type (mcq, yes_no, or auto).",
    )
    parser.add_argument("--model", type=str, required=True, help="HuggingFace model id.")
    parser.add_argument("--output-dir", type=str, default=".", help="Output directory.")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--cot", action="store_true", help="Use chain-of-thought prompts.")
    parser.add_argument(
        "--max-answer-scan-tokens", type=int, default=256,
        help="Max tokens to generate when scanning for an answer token.",
    )
    parser.add_argument(
        "--max-context-tokens", type=int, default=None,
        help="Truncate system-prompt context to this many tokens (default: no truncation).",
    )
    parser.add_argument(
        "--num-eval-questions", type=int, default=None, metavar="N",
        help="Evaluate only the first N records per dataset.",
    )
    parser.add_argument(
        "--hf-cartridge-filename", type=str, default="cache_last.pt",
        help="Default filename when downloading from a two-segment HF repo id.",
    )
    parser.add_argument("--hf-username", type=str, default=None)
    parser.add_argument("--debug", action="store_true")

    toka = parser.add_argument_group("Tokasaurus backend options")
    toka.add_argument("--url", type=str, default=None)
    toka.add_argument("--top-logprobs", type=int, default=20)
    toka.add_argument("--batch-size", type=int, default=16)

    args = parser.parse_args()

    if args.num_eval_questions is not None and args.num_eval_questions < 1:
        parser.error("--num-eval-questions must be >= 1")
    if args.backend == "tokasaurus" and args.url is None:
        parser.error("--url is required when using --backend tokasaurus")

    dataset_configs = [parse_dataset_arg(s) for s in args.datasets]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = run_all_icl_evaluations(
        context_topic=args.context_topic,
        cartridge_specs=args.cartridges,
        dataset_configs=dataset_configs,
        model_name=args.model,
        backend=args.backend,
        device=args.device,
        cot=args.cot,
        max_answer_scan_tokens=args.max_answer_scan_tokens,
        max_context_tokens=args.max_context_tokens,
        num_eval_questions=args.num_eval_questions,
        hf_cartridge_filename=args.hf_cartridge_filename,
        hf_username=args.hf_username,
        debug=args.debug,
        url=args.url,
        top_logprobs=args.top_logprobs,
        batch_size=args.batch_size,
    )

    out_file = out_dir / "icl_eval_results.json"
    serializable = []
    for entry in results:
        serializable.append({
            "condition": entry["condition"],
            "context_topic": entry["context_topic"],
            "cartridge": entry["cartridge"],
            "dataset": entry["dataset"],
            "dataset_type": entry["dataset_type"],
            "global_stats": entry["global_stats"],
            "eval_results": entry["eval_results"],
        })
    out_file.write_text(json.dumps(serializable, indent=2), encoding="utf-8")

    print(f"\n{'='*72}")
    print(f"Wrote {len(results)} evaluation results to {out_file}")

    summary_file = out_dir / "icl_eval_summary.json"
    summary = []
    for entry in results:
        summary.append({
            "condition": entry["condition"],
            "context_topic": entry["context_topic"],
            "cartridge": entry["cartridge"],
            "dataset": entry["dataset"],
            "dataset_type": entry["dataset_type"],
            **entry["global_stats"],
        })
    summary_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote summary to {summary_file}")

    for s in summary:
        acc = f"{s['accuracy']:.4f}" if s["accuracy"] is not None else "N/A"
        ppl = f"{s['avg_gt_answer_perplexity']:.4f}" if s["avg_gt_answer_perplexity"] is not None else "N/A"
        gt_miss = s.get("num_gt_missing_from_topk", 0)
        print(
            f"  {s['condition']} | {s['cartridge']} x {Path(s['dataset']).name}: "
            f"acc={acc}  gt_ppl={ppl}  gt_missing_topk={gt_miss}"
        )


if __name__ == "__main__":
    main()
