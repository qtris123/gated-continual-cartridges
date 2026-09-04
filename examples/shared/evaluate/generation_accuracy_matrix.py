#!/usr/bin/env python3
"""Evaluate staged cartridges as a resumable stage-by-eval accuracy matrix.

The evaluator is technique-agnostic. A JSON plan names the model, technique,
protocol, stage cache artifacts, and phase eval parquets. Each matrix cell is
an independent directory, so workers can safely split stages across GPUs.

Current supported protocol:
  freeform-mc-options-v1
    Greedy free-form generation scored per the eval row's ``category`` via
    ``CATEGORY_SCORERS``: ``mc_options`` for the QuALITY/LongHealth MCQ streams
    and ``numeric_match`` for FinQA's numeric answers.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from examples.shared.paths import ROOT as REPO_ROOT

os.environ.setdefault("CARTRIDGES_DIR", str(REPO_ROOT))
os.environ.setdefault("CARTRIDGES_OUTPUT_DIR", str(REPO_ROOT / "outputs"))
# Decode eagerly: the compiled generate-path flex-attention kernels raise a Triton
# illegal-memory-access on the short shapes decoding produces. Must be set before the
# (deferred) `cartridges.models` import so attention.py reads it at module load.
os.environ.setdefault("CARTRIDGES_EAGER_GENERATE", "1")

from cartridges.benchmark.scorers import CATEGORY_SCORERS, resolve_mc_option
from cartridges.cache import TrainableCache
from cartridges.generation import flex_generate
from cartridges.initialization.tokenization_utils import (
    MODELS_WITH_THINKING,
    MODEL_TO_CHAT_TEMPLATE,
)
from cartridges.structs import read_conversations

SCHEMA_VERSION = 1
PROTOCOL = "freeform-mc-options-v1"
SCORER_SOURCE = "faridlazuarda/gated-continual-cartridges@7c11bf1:mc_options"


@dataclass(frozen=True)
class EvalExample:
    question_id: str
    prompt_messages: list[dict[str, str]]
    prompt_text: str
    reference_answer: str
    metadata: dict[str, Any]


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "tolist"):
        return _jsonable(value.tolist())
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        json.dump(_jsonable(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def _plan_hash(plan: dict[str, Any]) -> str:
    encoded = json.dumps(plan, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _validate_plan(plan: dict[str, Any]) -> None:
    if plan.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"plan schema_version must be {SCHEMA_VERSION}, "
            f"got {plan.get('schema_version')!r}"
        )
    if plan.get("protocol") != PROTOCOL:
        raise ValueError(f"unsupported protocol {plan.get('protocol')!r}")
    for field in ("dataset", "technique", "model", "output_dir", "stages", "eval_sets"):
        if not plan.get(field):
            raise ValueError(f"plan field {field!r} is required")

    stage_ids = [stage["id"] for stage in plan["stages"]]
    eval_ids = [eval_set["id"] for eval_set in plan["eval_sets"]]
    if len(stage_ids) != len(set(stage_ids)):
        raise ValueError("stage ids must be unique")
    if len(eval_ids) != len(set(eval_ids)):
        raise ValueError("eval ids must be unique")
    for stage in plan["stages"]:
        if not Path(stage["cache_path"]).is_file():
            raise FileNotFoundError(stage["cache_path"])
    for eval_set in plan["eval_sets"]:
        if not Path(eval_set["path"]).is_file():
            raise FileNotFoundError(eval_set["path"])


def _load_examples(path: str) -> list[EvalExample]:
    examples = []
    for index, conversation in enumerate(read_conversations(path)):
        if len(conversation.messages) < 2:
            raise ValueError(f"{path}: row {index} has fewer than two messages")
        if conversation.messages[-1].role != "assistant":
            raise ValueError(f"{path}: row {index} does not end with an assistant gold")
        prompt_messages = [
            {"role": message.role, "content": message.content}
            for message in conversation.messages[:-1]
        ]
        metadata = _jsonable(dict(conversation.metadata))
        question_id = str(metadata.get("question_id", index))
        prompt_text = prompt_messages[-1]["content"] if prompt_messages else ""
        examples.append(
            EvalExample(
                question_id=question_id,
                prompt_messages=prompt_messages,
                prompt_text=prompt_text,
                reference_answer=conversation.messages[-1].content,
                metadata=metadata,
            )
        )
    if len({example.question_id for example in examples}) != len(examples):
        raise ValueError(f"{path}: question_id values must be unique")
    return examples


def _tokenize_batch(
    tokenizer: AutoTokenizer,
    examples: list[EvalExample],
    device: torch.device,
    prime_ids: list[int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    input_ids, sequence_ids, position_ids = [], [], []
    kwargs = {}
    if tokenizer.name_or_path in MODELS_WITH_THINKING:
        kwargs["enable_thinking"] = False
    for sequence_id, example in enumerate(examples):
        ids = tokenizer.apply_chat_template(
            example.prompt_messages,
            add_generation_prompt=True,
            return_tensors="pt",
            chat_template=MODEL_TO_CHAT_TEMPLATE.get(tokenizer.name_or_path),
            **kwargs,
        ).flatten().to(device)
        if prime_ids:
            # Pre-fill the start of the assistant answer (e.g. "Answer:") so decoding
            # continues from it instead of choosing its own opening token.
            ids = torch.cat(
                [ids, torch.tensor(prime_ids, device=device, dtype=ids.dtype)]
            )
        input_ids.append(ids)
        sequence_ids.append(torch.full_like(ids, sequence_id))
        position_ids.append(torch.arange(ids.numel(), device=device))
    return (
        torch.cat(input_ids),
        torch.cat(sequence_ids),
        torch.cat(position_ids),
    )


def _read_records(
    path: Path,
    plan_hash: str,
    stage_id: str,
    eval_id: str,
) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return records
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            expected = (plan_hash, stage_id, eval_id)
            actual = (
                record.get("plan_hash"),
                record.get("stage_id"),
                record.get("eval_id"),
            )
            if actual != expected:
                raise ValueError(
                    f"{path}:{line_number} belongs to {actual}, expected {expected}"
                )
            records[str(record["question_id"])] = record
    return records


def _cell_metrics(
    *,
    plan: dict[str, Any],
    plan_hash: str,
    stage: dict[str, Any],
    eval_set: dict[str, Any],
    records: list[dict[str, Any]],
    expected_questions: int,
    wall_clock_s: float,
) -> dict[str, Any]:
    # Records with score=None are "record-only" (no scorer for the category); they
    # contribute generations but no metric, so accuracy is computed over scored rows.
    scored = [record for record in records if record.get("score") is not None]
    correct = sum(float(record["score"]) for record in scored)
    difficult = [record for record in scored if int(record.get("difficult", 0)) == 1]
    difficult_correct = sum(float(record["score"]) for record in difficult)
    # `resolved_option` / `unparsed` are MCQ-only diagnostics: a numeric task
    # (FinQA) has no option list to resolve against, so counting its records as
    # "unparsed" would report a meaningless 100% rate. Restrict the accounting
    # to records that actually carry options.
    mcq_records = [
        record
        for record in records
        if (record.get("metadata") or {}).get("options")
    ]
    unparsed = sum(record.get("resolved_option") is None for record in mcq_records)
    scorer_name = records[0].get("scorer", "mc_options") if records else "mc_options"
    complete = len(records) == expected_questions
    return {
        "schema_version": SCHEMA_VERSION,
        "plan_hash": plan_hash,
        "dataset": plan["dataset"],
        "technique": plan["technique"],
        "protocol": plan["protocol"],
        "scorer": scorer_name,
        "scorer_source": SCORER_SOURCE,
        "model": plan["model"],
        "stage_id": stage["id"],
        "stage_index": stage["index"],
        "stage_label": stage.get("label", stage["id"]),
        "eval_id": eval_set["id"],
        "eval_phase": eval_set["phase"],
        "eval_label": eval_set.get("label", eval_set["id"]),
        "cache_path": stage["cache_path"],
        "eval_path": eval_set["path"],
        "decode": plan["decode"],
        "complete": complete,
        "num_expected": expected_questions,
        "num_recorded": len(records),
        "num_scored": len(scored),
        "num_correct": int(correct),
        "accuracy": correct / len(scored) if scored else None,
        "num_unparsed": unparsed,
        "unparsed_rate": unparsed / len(mcq_records) if mcq_records else None,
        "subgroups": {
            "difficult_1": {
                "num_scored": len(difficult),
                "num_correct": int(difficult_correct),
                "accuracy": difficult_correct / len(difficult) if difficult else None,
            }
        },
        "wall_clock_s": wall_clock_s,
    }


def _run_cell(
    *,
    plan: dict[str, Any],
    plan_hash: str,
    stage: dict[str, Any],
    eval_set: dict[str, Any],
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    cache: TrainableCache,
    device: torch.device,
    force: bool,
    max_samples: int | None,
) -> dict[str, Any]:
    cell_id = f"stage-{stage['id']}__eval-{eval_set['id']}"
    cell_dir = Path(plan["output_dir"]) / "cells" / cell_id
    metrics_path = cell_dir / "metrics.json"
    generations_path = cell_dir / "generations.jsonl"
    if metrics_path.exists() and not force:
        metrics = json.loads(metrics_path.read_text())
        if metrics.get("complete") and metrics.get("plan_hash") == plan_hash:
            print(f"[skip] {cell_id}: complete")
            return metrics
    cell_dir.mkdir(parents=True, exist_ok=True)
    if force:
        generations_path.unlink(missing_ok=True)

    examples = _load_examples(eval_set["path"])
    if max_samples is not None:
        examples = examples[:max_samples]
    existing = _read_records(generations_path, plan_hash, stage["id"], eval_set["id"])
    pending = [example for example in examples if example.question_id not in existing]
    batch_size = int(plan["decode"]["batch_size"])
    answer_prime = str(plan["decode"].get("answer_prime", "") or "")
    prime_ids = (
        tokenizer(answer_prime, add_special_tokens=False)["input_ids"]
        if answer_prime
        else None
    )
    started = time.time()

    with generations_path.open("a") as output:
        for offset in tqdm(
            range(0, len(pending), batch_size),
            desc=cell_id,
            leave=False,
        ):
            batch = pending[offset : offset + batch_size]
            input_ids, sequence_ids, position_ids = _tokenize_batch(
                tokenizer, batch, device, prime_ids=prime_ids
            )
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                generated = flex_generate(
                    model=model,
                    tokenizer=tokenizer,
                    input_ids=input_ids,
                    seq_ids=sequence_ids,
                    position_ids=position_ids,
                    cache=cache,
                    max_new_tokens=int(plan["decode"]["max_new_tokens"]),
                    temperature=float(plan["decode"]["temperature"]),
                )
            for sequence_id, example in enumerate(batch):
                generated_text = tokenizer.decode(
                    generated.get(sequence_id, []), skip_special_tokens=True
                )
                if answer_prime:
                    # The prime is part of the assistant's answer; include it so the
                    # scorer and the stored generation reflect the full response.
                    generated_text = answer_prime + generated_text
                category = str(example.metadata.get("category", ""))
                scorer = CATEGORY_SCORERS.get(category)
                if scorer is not None:
                    score = float(
                        scorer(
                            generated_text,
                            example.reference_answer,
                            metadata=example.metadata,
                        )
                    )
                    scorer_name = scorer.__name__
                else:
                    # Record-only: no scorer for this category (e.g. the free-form
                    # techqa_freeform / qasper_freeform streams). Save the generation
                    # with a null score so a metric (F1, etc.) can be computed later
                    # from generated_answer+reference_answer; logppl stays the headline
                    # number. This makes generation recording metric-agnostic.
                    score = None
                    scorer_name = "record_only"
                options = [str(option) for option in example.metadata.get("options") or []]
                resolved = resolve_mc_option(generated_text, options)
                record = {
                    "schema_version": SCHEMA_VERSION,
                    "plan_hash": plan_hash,
                    "dataset": plan["dataset"],
                    "technique": plan["technique"],
                    "protocol": plan["protocol"],
                    "stage_id": stage["id"],
                    "eval_id": eval_set["id"],
                    "question_id": example.question_id,
                    "question": example.prompt_text,
                    "generated_answer": generated_text,
                    "reference_answer": example.reference_answer,
                    "resolved_option": resolved,
                    "score": score,
                    "correct": (score == 1.0) if score is not None else None,
                    "scorer": scorer_name,
                    "category": category,
                    "difficult": int(example.metadata.get("difficult", 0)),
                    "metadata": example.metadata,
                }
                output.write(json.dumps(_jsonable(record), sort_keys=True) + "\n")
            output.flush()

    records_by_id = _read_records(
        generations_path, plan_hash, stage["id"], eval_set["id"]
    )
    records = [
        records_by_id[example.question_id]
        for example in examples
        if example.question_id in records_by_id
    ]
    metrics = _cell_metrics(
        plan=plan,
        plan_hash=plan_hash,
        stage=stage,
        eval_set=eval_set,
        records=records,
        expected_questions=len(examples),
        wall_clock_s=time.time() - started,
    )
    _atomic_json(metrics_path, metrics)
    acc = metrics["accuracy"]
    acc_str = f"{acc:.2%}" if acc is not None else "n/a (record-only)"
    print(
        f"[cell] {cell_id}: accuracy={acc_str} "
        f"recorded={metrics['num_recorded']}/{metrics['num_expected']}"
    )
    return metrics


def summarize(plan: dict[str, Any], plan_hash: str) -> dict[str, Any]:
    output_dir = Path(plan["output_dir"])
    cells: dict[tuple[str, str], dict[str, Any]] = {}
    for metrics_path in sorted((output_dir / "cells").glob("*/metrics.json")):
        metrics = json.loads(metrics_path.read_text())
        if metrics.get("plan_hash") == plan_hash:
            cells[(metrics["stage_id"], metrics["eval_id"])] = metrics

    stage_ids = [stage["id"] for stage in plan["stages"]]
    eval_ids = [eval_set["id"] for eval_set in plan["eval_sets"]]
    matrix = [
        [cells.get((stage_id, eval_id), {}).get("accuracy") for eval_id in eval_ids]
        for stage_id in stage_ids
    ]
    summary = {
        "schema_version": SCHEMA_VERSION,
        "plan_hash": plan_hash,
        "dataset": plan["dataset"],
        "technique": plan["technique"],
        "protocol": plan["protocol"],
        "model": plan["model"],
        "stage_ids": stage_ids,
        "eval_ids": eval_ids,
        "accuracy": matrix,
        "num_complete_cells": sum(
            bool(cell.get("complete")) for cell in cells.values()
        ),
        "num_expected_cells": len(stage_ids) * len(eval_ids),
    }
    _atomic_json(output_dir / "matrix.json", summary)
    with tempfile.NamedTemporaryFile(
        "w", dir=output_dir, prefix=".matrix.", delete=False, newline=""
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["stage_id", *eval_ids])
        for stage_id, row in zip(stage_ids, matrix):
            writer.writerow([stage_id, *row])
        temporary = Path(handle.name)
    temporary.replace(output_dir / "matrix.csv")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--stages", nargs="+", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()

    plan = json.loads(Path(args.plan).read_text())
    _validate_plan(plan)
    plan_hash = _plan_hash(plan)
    output_dir = Path(plan["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(output_dir / "plan.json", plan)

    if args.preflight:
        for eval_set in plan["eval_sets"]:
            examples = _load_examples(eval_set["path"])
            print(f"{eval_set['id']}: {len(examples)} questions")
        print(f"plan_hash={plan_hash}")
        return
    if args.summarize_only:
        print(json.dumps(summarize(plan, plan_hash), indent=2))
        return

    # Model imports initialize the flex-attention stack and are intentionally
    # deferred so preflight/summarization remain lightweight CLI operations.
    from cartridges.models import FlexLlamaForCausalLM, FlexQwen3ForCausalLM
    from cartridges.models.config import HFModelConfig

    requested = set(args.stages or [stage["id"] for stage in plan["stages"]])
    stages = [stage for stage in plan["stages"] if stage["id"] in requested]
    missing = requested - {stage["id"] for stage in stages}
    if missing:
        raise ValueError(f"unknown stage ids: {sorted(missing)}")

    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(plan["model"])
    model_class = (
        FlexQwen3ForCausalLM
        if "qwen" in plan["model"].lower()
        else FlexLlamaForCausalLM
    )
    model = HFModelConfig(
        pretrained_model_name_or_path=plan["model"],
        model_cls=model_class,
    ).instantiate().to(device).to(torch.bfloat16)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False

    for stage in stages:
        cache = TrainableCache.from_pretrained(stage["cache_path"], device="cpu")
        cache = cache.to(device)
        for eval_set in plan["eval_sets"]:
            _run_cell(
                plan=plan,
                plan_hash=plan_hash,
                stage=stage,
                eval_set=eval_set,
                model=model,
                tokenizer=tokenizer,
                cache=cache,
                device=device,
                force=args.force,
                max_samples=args.max_samples,
            )
        del cache
        torch.cuda.empty_cache()

    summary = summarize(plan, plan_hash)
    print(
        f"complete cells: {summary['num_complete_cells']}/"
        f"{summary['num_expected_cells']}"
    )


if __name__ == "__main__":
    main()
