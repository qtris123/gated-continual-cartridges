#!/usr/bin/env python3
"""Greedy-decode QASPER eval questions against one or more cartridges.

Perplexity can improve while decoding stays degenerate, so this reports the two
separately: answer-F1 against the gold span, and a degeneracy panel (how many of
the generations are distinct, empty, or malformed). A cartridge that emits the
same trivial string for every question scores `distinct` near 1/n however good
its loss is.

Usage:
    python examples/shared/evaluate/qasper_cartridge_generations.py \
        --cache shipped=/path/to/cache_last.pt \
        --cache fixed=/path/to/cache_last.pt \
        --n 16
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

# The compiled decode kernel (`dynamic=True`) fails to lower on torch 2.13 once the
# block mask closes over a cache-backed `kv_seq_ids` of dynamic size:
# "NoValidChoicesError: No choices to select" out of the flex_attention HOP. Decode
# here is one query token against ~600 keys, where the uncompiled path costs a few
# MiB, so this script opts out rather than tuning inductor.
cartridges_attention.flex_attention_generate = (
    cartridges_attention.flex_attention
)


def normalize(text: str) -> str:
    """QASPER/SQuAD-style normalisation before token overlap."""
    text = text.lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def answer_f1(pred: str, gold: str) -> float:
    p, g = normalize(pred).split(), normalize(gold).split()
    if not p or not g:
        return float(p == g)
    common = Counter(p) & Counter(g)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision, recall = overlap / len(p), overlap / len(g)
    return 2 * precision * recall / (precision + recall)


def strip_answer_tag(text: str) -> tuple[str, bool]:
    match = re.search(r"<answer>(.*?)</answer>", text, flags=re.DOTALL)
    if match:
        return match.group(1).strip(), True
    return text.strip(), False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", action="append", required=True,
                    metavar="LABEL=PATH", help="repeatable")
    ap.add_argument("--eval-path", default="data/qasper/eval/qasper_eval_QA.parquet")
    ap.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--show", type=int, default=6, help="how many generations to print verbatim")
    ap.add_argument("--out", default=None, help="write full generations to this JSON")
    args = ap.parse_args()

    arms = []
    for spec in args.cache:
        label, _, path = spec.partition("=")
        if not path:
            ap.error(f"--cache expects LABEL=PATH, got {spec!r}")
        arms.append((label, path))

    df = pd.read_parquet(args.eval_path)
    # The eval rows are grouped by paper, so `head(n)` samples one paper's questions
    # and cannot distinguish "answers this paper" from "answers anything".
    df = df.iloc[[round(i * (len(df) - 1) / max(args.n - 1, 1)) for i in range(args.n)]]
    papers = {m["title"] for m in df["metadata"]}
    print(f"{len(df)} questions spanning {len(papers)} of the phase's papers")

    questions, golds = [], []
    for row in df.itertuples():
        msgs = list(row.messages)
        user = next(m["content"] for m in msgs if m["role"] == "user")
        assistant = next((m["content"] for m in msgs if m["role"] == "assistant"), "")
        questions.append(user)
        golds.append(strip_answer_tag(assistant)[0])

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
            text = tokenizer.decode(out.get(0, []), skip_special_tokens=True)
            answer, well_formed = strip_answer_tag(text)
            rows.append({"raw": text, "answer": answer, "well_formed": well_formed})
            # Drop the KV this question appended; the cartridge slots survive.
            cache.clear()
        results[label] = rows
        del cache
        torch.cuda.empty_cache()

    print(f"\n{'arm':<28} {'F1':>7} {'distinct':>9} {'well-formed':>12} {'empty':>7} {'mean len':>9}")
    print("-" * 76)
    for label, rows in results.items():
        f1 = sum(answer_f1(r["answer"], g) for r, g in zip(rows, golds)) / len(rows)
        distinct = len({r["answer"] for r in rows}) / len(rows)
        formed = sum(r["well_formed"] for r in rows) / len(rows)
        empty = sum(not r["answer"] for r in rows) / len(rows)
        mean_len = sum(len(r["answer"].split()) for r in rows) / len(rows)
        print(f"{label:<28} {f1:>7.4f} {distinct:>9.2f} {formed:>12.2f} "
              f"{empty:>7.2f} {mean_len:>9.1f}")

    for i in range(min(args.show, len(questions))):
        q = re.search(r"<question>(.*?)</question>", questions[i], flags=re.DOTALL)
        print(f"\n--- Q{i}: {(q.group(1).strip() if q else questions[i])[:180]}")
        print(f"    gold: {golds[i][:180]}")
        for label, rows in results.items():
            print(f"    {label:<26}: {rows[i]['answer'][:180]!r}")

    if args.out:
        Path(args.out).write_text(json.dumps(
            {"questions": questions, "golds": golds, "results": results}, indent=2))
        print(f"\nfull generations -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
