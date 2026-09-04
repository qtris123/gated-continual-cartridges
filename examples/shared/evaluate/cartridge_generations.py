#!/usr/bin/env python3
"""Greedy-decode a phase's eval questions against one or more cartridges.

Dataset-agnostic sibling of `qasper_cartridge_generations.py`: perplexity can
improve while decoding stays degenerate, so this prints a degeneracy panel (how
many generations are distinct / empty) plus verbatim samples next to the gold
answer, for any dataset whose eval parquet has `messages` (+ optional
`metadata`). No assumptions about <question>/<answer> tags or a `title` field.

    python examples/shared/evaluate/cartridge_generations.py \
        --cache p01=/path/cache_last.pt --cache p05=/path/cache_last.pt \
        --eval-path data/finqa/phases/phase3_eval.parquet --n 12
"""

from __future__ import annotations

import argparse
import json
import re
import string
from collections import Counter
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoTokenizer

import cartridges.models.attention as cartridges_attention
from cartridges.cache import TrainableCache
from cartridges.generation import flex_generate
from cartridges.models import FlexLlamaForCausalLM, FlexQwen3ForCausalLM, HFModelConfig

# Same opt-out as the QASPER script: the compiled decode kernel fails to lower on
# a cache-backed dynamic block mask; decode is one token vs ~600 keys, so the
# uncompiled path is cheap.
cartridges_attention.flex_attention_generate = cartridges_attention.flex_attention
# Decode grows the KV by one token each step, so a *compiled* block mask recompiles
# every step (a fresh KV_LEN) -- tens of ~5s inductor compiles per question, which
# reads as a hang. Eager `create_block_mask` is bit-identical and ~free at these
# lengths (the 0.75 GiB concern is only for multi-k-token prefills), so use it here.
cartridges_attention.create_block_mask_compiled = cartridges_attention.create_block_mask

_GROUP_KEYS = ("company", "doc_source", "filenames", "title", "question_id")


def normalize(text: str) -> str:
    text = text.lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def token_f1(pred: str, gold: str) -> float:
    p, g = normalize(pred).split(), normalize(gold).split()
    if not p or not g:
        return float(p == g)
    overlap = sum((Counter(p) & Counter(g)).values())
    if overlap == 0:
        return 0.0
    precision, recall = overlap / len(p), overlap / len(g)
    return 2 * precision * recall / (precision + recall)


def group_key(md) -> str:
    if isinstance(md, dict):
        for key in _GROUP_KEYS:
            if key in md and md[key] is not None:
                val = md[key]
                return str(val[0] if isinstance(val, (list, tuple)) and val else val)
    return ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", action="append", required=True,
                    metavar="LABEL=PATH", help="repeatable")
    ap.add_argument("--eval-path", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--max-new-tokens", type=int, default=96)
    ap.add_argument("--show", type=int, default=8)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    arms = []
    for spec in args.cache:
        label, _, path = spec.partition("=")
        if not path:
            ap.error(f"--cache expects LABEL=PATH, got {spec!r}")
        arms.append((label, path))

    df = pd.read_parquet(args.eval_path)
    # Spread the sample across the file so we hit multiple documents, not one.
    idx = [round(i * (len(df) - 1) / max(args.n - 1, 1)) for i in range(args.n)]
    df = df.iloc[sorted(set(idx))]
    groups = {group_key(m) for m in df["metadata"]} if "metadata" in df.columns else set()
    print(f"{len(df)} questions spanning {len(groups)} distinct documents")

    questions, golds = [], []
    for row in df.itertuples():
        msgs = list(row.messages)
        questions.append(next(m["content"] for m in msgs if m["role"] == "user"))
        golds.append(next((m["content"] for m in msgs if m["role"] == "assistant"), ""))

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model_cls = FlexQwen3ForCausalLM if "qwen" in args.model.lower() else FlexLlamaForCausalLM
    model = HFModelConfig(
        pretrained_model_name_or_path=args.model,
        model_cls=model_cls,
        load_kwargs={"torch_dtype": "bfloat16"},
    ).instantiate().to(torch.bfloat16).to("cuda")
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    results: dict[str, list[dict]] = {}
    for label, path in arms:
        cache = TrainableCache.from_pretrained(path, device="cuda").to("cuda").to(torch.bfloat16)
        rows = []
        for question in questions:
            ids = tokenizer.apply_chat_template(
                [{"role": "user", "content": question}],
                tokenize=True, add_generation_prompt=True,
                return_tensors="pt", enable_thinking=False,
            ).flatten().to("cuda")
            out = flex_generate(
                model=model, tokenizer=tokenizer, input_ids=ids,
                seq_ids=torch.zeros(ids.shape[0], dtype=torch.long, device="cuda"),
                position_ids=torch.arange(ids.shape[0], device="cuda"),
                cache=cache, max_new_tokens=args.max_new_tokens, temperature=0.0,
            )
            text = tokenizer.decode(out.get(0, []), skip_special_tokens=True).strip()
            rows.append({"answer": text})
            cache.clear()
        results[label] = rows
        del cache
        torch.cuda.empty_cache()

    print(f"\n{'arm':<10} {'F1':>7} {'distinct':>9} {'empty':>7} {'mean_len':>9}")
    print("-" * 46)
    for label, rows in results.items():
        f1 = sum(token_f1(r["answer"], g) for r, g in zip(rows, golds)) / len(rows)
        distinct = len({r["answer"] for r in rows}) / len(rows)
        empty = sum(not r["answer"] for r in rows) / len(rows)
        mean_len = sum(len(r["answer"].split()) for r in rows) / len(rows)
        print(f"{label:<10} {f1:>7.4f} {distinct:>9.2f} {empty:>7.2f} {mean_len:>9.1f}")

    for i in range(min(args.show, len(questions))):
        print(f"\n--- Q{i}: {questions[i][:160].replace(chr(10),' ')}")
        print(f"    gold: {golds[i][:160].replace(chr(10),' ')!r}")
        for label, rows in results.items():
            print(f"    {label:<8}: {rows[i]['answer'][:200].replace(chr(10),' ')!r}")

    if args.out:
        Path(args.out).write_text(json.dumps(
            {"eval_path": args.eval_path, "questions": questions,
             "golds": golds, "results": results}, indent=2))
        print(f"\nfull generations -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
