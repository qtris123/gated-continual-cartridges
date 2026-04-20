#!/usr/bin/env python3
"""
Evaluate cartridge checkpoints on Qasper-style CSV datasets (Yes/No and MCQ).

Supports two inference backends:

- **local** (default): loads the model onto GPU, greedy-decodes one question at
  a time.  Full vocabulary logits are available at the answer step.
- **tokasaurus**: sends batched requests to a running Tokasaurus server.
  Cartridges are loaded server-side from HuggingFace.  Top-K logprobs
  (not full logits) are returned per generated token; answer-option logprobs
  are extracted from the top-K at the answer position.

Usage examples::

    # Local backend (loads model locally)
    python cartridge_eval.py \\
        --cartridges  qtris123/cartridge-A  qtris123/cartridge-B \\
        --datasets    /path/to/mcq.csv:mcq  /path/to/yesno.csv:yes_no \\
        --model       Qwen/Qwen3-4B \\
        --output-dir  ./eval_outputs

    # Tokasaurus backend (requires a running server)
    python cartridge_eval.py \\
        --backend tokasaurus \\
        --url http://localhost:10210 \\
        --cartridges  qtris123/cartridge-A  qtris123/cartridge-B \\
        --datasets    /path/to/mcq.csv:mcq  /path/to/yesno.csv:yes_no \\
        --model       meta-llama/Llama-3.2-3B-Instruct \\
        --output-dir  ./eval_outputs \\
        --top-logprobs 20 \\
        --batch-size 16
"""

from __future__ import annotations

import os

os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")

import argparse
import asyncio
import json
import math
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from transformers import AutoTokenizer

try:
    from huggingface_hub import hf_hub_download
except ImportError:
    hf_hub_download = None  # type: ignore[misc, assignment]

from cartridges.clients.tokasaurus import TokasaurusClient
from cartridges.clients.base import CartridgeConfig, ClientSample, TopLogprobs

from cartridges.cache import KVCacheFactory, TrainableCache
from cartridges.models import FlexLlamaForCausalLM, FlexQwen3ForCausalLM, HFModelConfig
from cartridges.initialization.tokenization_utils import MODEL_TO_CHAT_TEMPLATE, MODELS_WITH_THINKING

from helpers import load_eval_records, infer_question_type, option_letters_and_texts, reference_answer_letter


# ---------------------------------------------------------------------------
# Constants & prompt templates
# ---------------------------------------------------------------------------

MCQ_INSTRUCTION = (
    "Answer with A, B, C, or D only, no other text"
)
YESNO_INSTRUCTION = (
    "Answer with Yes or No only, no other text"
)
COT_MCQ_INSTRUCTION = (
    "Explain your reasoning, then end your response with 'Answer: X' "
    "where X is the letter (A, B, C, or D)"
)
COT_YESNO_INSTRUCTION = (
    "Explain your reasoning, then end your response with 'Answer: X' "
    "where X is Yes or No"
)

DIST_TOPK = 32
ANSWER_ANCHOR = "Answer:"


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------

def _mcq_options_block(record: dict) -> str:
    lines = [str(record.get(f"option_{L}") or "").strip() for L in ("a", "b", "c", "d")]
    return "\n".join(line for line in lines if line)


def _build_user_prompt(record: dict, qtype: str, *, cot: bool) -> str:
    pid = str(record.get("id", "")).strip()
    intro = f"Please answer the question below about the following patient: {pid or 'Unknown ID'}"

    if qtype == "yes_no":
        qtext = str(record.get("yes_no_question") or "").strip()
        tail = COT_YESNO_INSTRUCTION if cot else YESNO_INSTRUCTION
        return f"{intro}\nQuestion: {qtext}\n{tail}".rstrip() + " "

    mq = str(record.get("mcq_question") or "").strip()
    options_block = _mcq_options_block(record)
    tail = COT_MCQ_INSTRUCTION if cot else MCQ_INSTRUCTION
    return (
        f"{intro}"
        f"\n Question: {mq}"
        f"\n Options: {options_block}"
        f"\n {tail}"
    ).rstrip() + " "


def build_user_message(record: dict, *, cot: bool = False) -> str:
    qtype = infer_question_type(record)
    return _build_user_prompt(record, qtype, cot=cot)


# ---------------------------------------------------------------------------
# Token & logprob helpers (shared by both backends)
# ---------------------------------------------------------------------------

def _answer_option_variants(qtype: str) -> Dict[str, List[str]]:
    if qtype == "yes_no":
        return {
            "Yes": ["Yes", " Yes", "\nYes", "\n Yes", "YES", "yes", " yes", "\nyes", "\n yes"],
            "No":  ["No",  " No",  "\nNo",  "\n No",  "NO",  "no",  " no",  "\nno",  "\n no"],
        }
    if qtype == "mcq":
        return {
            L: [
                L, f" {L}", f"\n{L}", f"\n {L}", f"({L})", f"{L})", f" {L})", f"[{L}]",
                L.lower(), f" {L.lower()}", f"\n{L.lower()}", f"\n {L.lower()}",
                f"({L.lower()})", f"{L.lower()})", f" {L.lower()})", f"[{L.lower()}]",
            ]
            for L in ("A", "B", "C", "D")
        }
    raise ValueError(f"Unknown qtype: {qtype!r}")


def _first_token_ids(tokenizer, variants: List[str]) -> Set[int]:
    out: Set[int] = set()
    for s in variants:
        ids = tokenizer.encode(s, add_special_tokens=False)
        if ids:
            out.add(int(ids[0]))
    return out


def build_answer_token_maps(
    tokenizer, qtype: str
) -> Tuple[Dict[str, Set[int]], Set[int]]:
    """Return ``(label_to_token_ids, all_valid_token_ids)`` for a question type."""
    variants = _answer_option_variants(qtype)
    label_to_ids = {
        label: _first_token_ids(tokenizer, strs) for label, strs in variants.items()
    }
    valid: Set[int] = set().union(*label_to_ids.values())
    return label_to_ids, valid


def normalized_answer_distribution(choice_logprobs: Dict[str, float]) -> Dict[str, float]:
    if not choice_logprobs:
        return {}
    labels = list(choice_logprobs.keys())
    vals = np.array([choice_logprobs[k] for k in labels], dtype=np.float32)
    probs = np.exp(vals - vals.max())
    probs /= probs.sum()
    return {label: float(p) for label, p in zip(labels, probs)}


def find_answer_position(
    token_ids: List[int],
    valid_answer_ids: Set[int],
    tokenizer_decode,
    *,
    cot: bool = False,
) -> Optional[int]:
    """Return the index in ``token_ids`` of the first answer token."""
    answer_zone = not cot
    generated_text = ""

    for pos, tid in enumerate(token_ids):
        tok_text = tokenizer_decode([tid])
        generated_text += tok_text

        if cot and not answer_zone:
            if ANSWER_ANCHOR in generated_text:
                answer_zone = True
            continue

        if answer_zone and tid in valid_answer_ids:
            return pos
    # fallback to last valid answer token for cot cases 
    if cot:
        for pos in range(len(token_ids) - 1, len(token_ids) - 10, -1):
            tid = token_ids[pos]
            if tid in valid_answer_ids:
                return pos
    return None


# ---------------------------------------------------------------------------
# Local backend: full-logit helpers
# ---------------------------------------------------------------------------

def _max_logprob_first_token(tokenizer, logp: torch.Tensor, variants: List[str]) -> float:
    best = float("-inf")
    for s in variants:
        ids = tokenizer.encode(s, add_special_tokens=False)
        if ids:
            best = max(best, float(logp[ids[0]].item()))
    return best


def option_logprobs_from_logits(
    logits: torch.Tensor, tokenizer, record: dict
) -> Dict[str, float]:
    logp = F.log_softmax(logits.float(), dim=-1)
    variants = _answer_option_variants(infer_question_type(record))
    return {
        label: _max_logprob_first_token(tokenizer, logp, strs)
        for label, strs in variants.items()
    }


def _is_whitespace_only_token(tokenizer, token_id: int) -> bool:
    return len(tokenizer.decode([token_id], skip_special_tokens=False).strip()) == 0


def topk_logprobs_non_whitespace(
    logits: torch.Tensor, tokenizer, *, k: int
) -> Tuple[List[Dict[str, Any]], Optional[int]]:
    logp = F.log_softmax(logits.float(), dim=-1)
    order = torch.argsort(logp, descending=True)
    picked: List[Dict[str, Any]] = []
    first_non_ws: Optional[int] = None
    for idx in order.tolist():
        tid = int(idx)
        if _is_whitespace_only_token(tokenizer, tid):
            continue
        if first_non_ws is None:
            first_non_ws = tid
        picked.append({
            "token_id": tid,
            "token_text": tokenizer.decode([tid], skip_special_tokens=False),
            "logprob": float(logp[tid].item()),
        })
        if len(picked) >= k:
            break
    return picked, first_non_ws


# ---------------------------------------------------------------------------
# Tokasaurus backend: top-K logprob helpers
# ---------------------------------------------------------------------------

def extract_answer_logprobs_from_topk(
    top_logprobs: TopLogprobs,
    answer_pos: int,
    label_to_ids: Dict[str, Set[int]],
) -> Dict[str, float]:
    """At ``answer_pos``, extract logprob for each answer label from the top-K."""
    topk_ids = top_logprobs.token_ids[answer_pos]
    topk_logp = top_logprobs.logprobs[answer_pos]

    id_to_logp = {
        int(topk_ids[j]): float(topk_logp[j])
        for j in range(len(topk_ids))
    }

    result: Dict[str, float] = {}
    for label, ids in label_to_ids.items():
        best = None
        for tid in ids:
            if tid in id_to_logp:
                lp = id_to_logp[tid]
                if best is None or lp > best:
                    best = lp
        if best is not None:
            result[label] = best
    return result


def topk_distribution_from_server(
    top_logprobs: TopLogprobs,
    answer_pos: int,
    tokenizer,
) -> List[Dict[str, Any]]:
    """Build the top-K token distribution list at the answer position."""
    topk_ids = top_logprobs.token_ids[answer_pos]
    topk_logp = top_logprobs.logprobs[answer_pos]
    out: List[Dict[str, Any]] = []
    for j in range(len(topk_ids)):
        tid = int(topk_ids[j])
        if tid < 0:
            continue
        out.append({
            "token_id": tid,
            "token_text": tokenizer.decode([tid], skip_special_tokens=False),
            "logprob": float(topk_logp[j]),
        })
    return out


# ---------------------------------------------------------------------------
# Local backend: model & cartridge loading
# ---------------------------------------------------------------------------

class KVFromLocal(KVCacheFactory):
    class Config(KVCacheFactory.Config):
        path: str

    def __init__(self, config: "KVFromLocal.Config", load_device: str = "cuda"):
        super().__init__(config)
        self._load_device = load_device

    def initialize_kv_cache(self, tokenizer=None, model=None, attn_config=None):
        return TrainableCache.from_pretrained(self.config.path, device=self._load_device)


def resolve_cartridge_source(
    spec: str,
    *,
    default_hf_filename: str = "cache_last.pt",
    hf_username: Optional[str] = None,
) -> Tuple[dict, str, str]:
    """Load a cartridge from a local .pt path or HF repo.

    Returns ``(checkpoint_dict, local_path, label)``.
    """
    raw = spec.strip()

    p = Path(raw).expanduser()
    if p.is_file():
        ap = str(p.resolve())
        return torch.load(ap, map_location="cpu", weights_only=False), ap, ap

    if hf_hub_download is None:
        raise ImportError("pip install huggingface_hub")

    s = raw.lstrip("/")
    parts = [x for x in s.split("/") if x]

    if hf_username and len(parts) == 2 and parts[-1].endswith(".pt"):
        repo_id = hf_username.strip().strip("/") + "/" + parts[0]
        filename = parts[1]
    elif len(parts) == 2 and not parts[-1].endswith(".pt"):
        repo_id = f"{parts[0]}/{parts[1]}"
        filename = default_hf_filename
    elif len(parts) >= 3 and parts[-1].endswith(".pt"):
        repo_id = f"{parts[0]}/{parts[1]}"
        filename = "/".join(parts[2:])
    else:
        raise FileNotFoundError(
            f"Cartridge not found: {spec!r}. Use a local .pt path, "
            "a Hub repo id like namespace/model, or namespace/model/path/to/file.pt."
        )

    local_path = hf_hub_download(repo_id=repo_id, filename=filename)
    ckpt = torch.load(local_path, map_location="cpu", weights_only=False)
    return ckpt, local_path, f"{repo_id}/{filename}"


def load_model_and_tokenizer(model_name: str, device: str):
    model_cls = FlexQwen3ForCausalLM if "qwen" in model_name.lower() else FlexLlamaForCausalLM
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = HFModelConfig(
        pretrained_model_name_or_path=model_name,
        model_cls=model_cls,
    ).instantiate()
    model = model.to(device).to(torch.bfloat16)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model, tokenizer


def load_cache(cartridge_path: str, device: str) -> TrainableCache:
    factory = KVFromLocal(KVFromLocal.Config(path=cartridge_path), load_device=device)
    cache = factory.initialize_kv_cache()
    return cache.to(device)


# ---------------------------------------------------------------------------
# Local backend: greedy answer-token extraction
# ---------------------------------------------------------------------------

_QWEN3_CHAT_TEMPLATE = (
    "{%- for message in messages %}"
    "{%- if message['role'] == 'system' %}"
    "<|im_start|>system\n{{ message['content'] }}<|im_end|>\n"
    "{%- elif message['role'] == 'user' %}"
    "<|im_start|>user\n{{ message['content'] }}<|im_end|>\n"
    "{%- elif message['role'] == 'assistant' %}"
    "<|im_start|>assistant\n{{ message['content'] }}<|im_end|>\n"
    "{%- endif %}"
    "{%- endfor %}"
    "{%- if add_generation_prompt %}"
    "<|im_start|>assistant\n"
    "{%- endif %}"
)


def _is_thinking_model(model_name: str) -> bool:
    name = model_name.lower()
    if any(name == m.lower() for m in MODELS_WITH_THINKING):
        return True
    return "qwen3" in name


def _resolve_chat_template(tokenizer) -> Optional[str]:
    name = getattr(tokenizer, "name_or_path", "") or ""
    template = MODEL_TO_CHAT_TEMPLATE.get(name)
    if template is not None:
        return template
    if getattr(tokenizer, "chat_template", None) is not None:
        return None
    if "qwen" in name.lower():
        return _QWEN3_CHAT_TEMPLATE
    return None


def _encode_user_prompt(
    tokenizer, user_content: str, device: str
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    name = (getattr(tokenizer, "name_or_path", None) or "").strip()
    kwargs: Dict[str, object] = {}
    if _is_thinking_model(name):
        kwargs["enable_thinking"] = False

    template = _resolve_chat_template(tokenizer)
    if template is not None:
        kwargs["chat_template"] = template

    result = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_content}],
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


def _autocast_context(device: str):
    if str(device).startswith("cuda"):
        return torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def find_answer_token_step(
    model,
    tokenizer,
    user_content: str,
    cache: TrainableCache,
    device: str,
    record: dict,
    *,
    cot: bool,
    max_new_tokens: int = 256,
    token_maps: Optional[Tuple[Dict[str, Set[int]], Set[int]]] = None,
    debug: bool = False,
) -> Tuple[Optional[torch.Tensor], Dict[str, object]]:
    """Greedily decode until the model generates a target answer token.

    Returns ``(logits_at_answer_step | None, metadata_dict)``.
    """
    flat, seq_ids, position_ids = _encode_user_prompt(tokenizer, user_content, device)

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

            if cot and not answer_zone_active and ANSWER_ANCHOR in generated_text:
                answer_zone_active = True

            next_input_ids = torch.tensor(
                [next_token_id], device=device, dtype=flat.dtype
            )
            next_seq_ids = torch.zeros_like(next_input_ids)
            last_position += 1
            next_position_ids = torch.tensor(
                [last_position], device=device, dtype=position_ids.dtype
            )

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
    }


# ---------------------------------------------------------------------------
# Scoring & statistics (shared by both backends)
# ---------------------------------------------------------------------------

def score_mcq(pred_letter: str, record: dict) -> bool:
    ref = reference_answer_letter(record)
    if ref is None:
        return False
    return pred_letter.strip().upper() == ref.strip().upper()


def score_yes_no(predicted: str, record: dict) -> bool:
    return (
        str(record.get("correct", "")).strip().lower()
        == str(predicted).strip().lower()
    )


def gt_answer_logprob(choice_logprobs: Dict[str, float], record: dict) -> Optional[float]:
    qtype = infer_question_type(record)
    if qtype == "yes_no":
        gt = str(record.get("correct", "")).strip().capitalize()
    else:
        gt = reference_answer_letter(record)
    if gt is None or gt not in choice_logprobs:
        return None
    return choice_logprobs[gt]


def logprob_to_perplexity(logprob: float) -> float:
    return math.exp(-logprob)


def _score_and_build_row(
    idx: int,
    record: dict,
    choice_logprobs: Dict[str, float],
    token_dist_topk: List[Dict[str, Any]],
    answer_meta: Dict[str, object],
) -> Tuple[dict, bool]:
    """Build one result row from answer-option logprobs. Returns ``(row, correct)``."""
    qtype = infer_question_type(record)
    q_text = record.get("yes_no_question") or record.get("mcq_question") or ""
    options = {L: t for L, t in option_letters_and_texts(record)}
    choice_probs = normalized_answer_distribution(choice_logprobs)
    predicted = max(choice_logprobs, key=choice_logprobs.get) if choice_logprobs else ""

    if qtype == "yes_no":
        correct = score_yes_no(predicted, record)
    else:
        correct = score_mcq(predicted, record)

    gt_lp = gt_answer_logprob(choice_logprobs, record)
    gt_ppl = logprob_to_perplexity(gt_lp) if gt_lp is not None else None
    gt_in_topk = gt_lp is not None

    row = {
        "idx": idx,
        "question": q_text,
        "question_type": qtype,
        "answered": True,
        "correct": correct,
        "generated_answer": str(predicted),
        "reference_answer": record.get("correct"),
        "reference_answer_letter": reference_answer_letter(record),
        "options": options,
        "choice_logprobs": choice_logprobs,
        "choice_probs_normalized": choice_probs,
        "gt_answer_logprob": gt_lp,
        "gt_answer_perplexity": gt_ppl,
        "gt_in_topk": gt_in_topk,
        "token_distribution_topk": token_dist_topk,
        "answer_extraction_meta": answer_meta,
    }
    return row, correct


def _build_unanswered_row(
    idx: int,
    record: dict,
    answer_meta: Dict[str, object],
) -> dict:
    qtype = infer_question_type(record)
    q_text = record.get("yes_no_question") or record.get("mcq_question") or ""
    options = {L: t for L, t in option_letters_and_texts(record)}
    return {
        "idx": idx,
        "question": q_text,
        "question_type": qtype,
        "answered": False,
        "correct": None,
        "generated_answer": "",
        "reference_answer": record.get("correct"),
        "reference_answer_letter": reference_answer_letter(record),
        "options": options,
        "choice_logprobs": {},
        "choice_probs_normalized": {},
        "gt_answer_logprob": None,
        "gt_answer_perplexity": None,
        "gt_in_topk": None,
        "token_distribution_topk": [],
        "answer_extraction_meta": answer_meta,
    }


def compute_global_stats(rows: List[dict]) -> Dict[str, Any]:
    answered = [r for r in rows if r.get("answered")]
    n_answered = len(answered)
    n_unanswered = len(rows) - n_answered
    n_correct = sum(1 for r in answered if r.get("correct") is True)
    accuracy = (n_correct / n_answered) if n_answered else None

    gt_perplexities = [r["gt_answer_perplexity"] for r in answered if r.get("gt_answer_perplexity") is not None]
    avg_gt_perplexity = (
        sum(gt_perplexities) / len(gt_perplexities) if gt_perplexities else None
    )

    gt_logprobs = [r["gt_answer_logprob"] for r in answered if r.get("gt_answer_logprob") is not None]
    avg_gt_logprob = (
        sum(gt_logprobs) / len(gt_logprobs) if gt_logprobs else None
    )

    gt_missing_from_topk = [
        {"idx": r["idx"], "question": r.get("question", ""), "question_type": r.get("question_type", ""),
         "reference_answer": r.get("reference_answer"), "reference_answer_letter": r.get("reference_answer_letter"),
         "generated_answer": r.get("generated_answer", "")}
        for r in answered if r.get("gt_in_topk") is False
    ]

    return {
        "num_total": len(rows),
        "num_answered": n_answered,
        "num_unanswered": n_unanswered,
        "num_correct": n_correct,
        "accuracy": accuracy,
        "avg_gt_answer_logprob": avg_gt_logprob,
        "avg_gt_answer_perplexity": avg_gt_perplexity,
        "num_gt_missing_from_topk": len(gt_missing_from_topk),
        "gt_missing_from_topk": gt_missing_from_topk,
    }


# ---------------------------------------------------------------------------
# Local backend: evaluate one (cartridge, dataset)
# ---------------------------------------------------------------------------

def evaluate_local(
    *,
    model,
    tokenizer,
    cartridge_path: str,
    eval_records: List[dict],
    device: str,
    cot: bool = False,
    max_answer_scan_tokens: int = 256,
    debug: bool = False,
    desc: str = "eval",
) -> List[dict]:
    token_maps_yn = build_answer_token_maps(tokenizer, "yes_no")
    token_maps_mcq = build_answer_token_maps(tokenizer, "mcq")
    cache = load_cache(cartridge_path, device)

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

        logits, answer_meta = find_answer_token_step(
            model=model,
            tokenizer=tokenizer,
            user_content=user_msg,
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


# ---------------------------------------------------------------------------
# Tokasaurus backend: evaluate one (cartridge, dataset)
# ---------------------------------------------------------------------------

def evaluate_tokasaurus(
    *,
    client: TokasaurusClient,
    tokenizer,
    cartridge_id: str,
    eval_records: List[dict],
    cot: bool = False,
    max_completion_tokens: int = 256,
    top_logprobs: int = 20,
    batch_size: int = 16,
    debug: bool = False,
    desc: str = "eval",
) -> List[dict]:
    token_maps_yn = build_answer_token_maps(tokenizer, "yes_no")
    token_maps_mcq = build_answer_token_maps(tokenizer, "mcq")
    decode_fn = lambda ids: tokenizer.decode(ids, skip_special_tokens=False)

    cartridge_cfg = [CartridgeConfig(id=cartridge_id, source="huggingface").model_dump()]

    chats_and_records: List[Tuple[List[Dict[str, str]], dict, int]] = []
    for idx, record in enumerate(eval_records):
        user_msg = build_user_message(record, cot=cot)
        chat = [{"role": "user", "content": user_msg}]
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

        response = asyncio.run(client.chat(
            chats=batch_chats,
            max_completion_tokens=max_completion_tokens,
            temperature=0.0,
            top_logprobs=top_logprobs,
            cartridges=cartridge_cfg,
        ))

        for i, sample in enumerate(response.samples):
            record = batch_records[i]
            idx = batch_indices[i]
            user_msg = batch_chats[i][0]["content"]
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


# ---------------------------------------------------------------------------
# Full evaluation across all (cartridge, dataset) pairs
# ---------------------------------------------------------------------------

def run_all_evaluations(
    *,
    cartridge_specs: List[str],
    dataset_configs: List[Tuple[str, str]],
    model_name: str,
    backend: str = "local",
    device: str = "cuda",
    cot: bool = False,
    max_answer_scan_tokens: int = 256,
    num_eval_questions: Optional[int] = None,
    hf_cartridge_filename: str = "cache_last.pt",
    hf_username: Optional[str] = None,
    debug: bool = False,
    # Tokasaurus-specific
    url: Optional[str] = None,
    top_logprobs: int = 20,
    batch_size: int = 16,
) -> List[Dict[str, Any]]:

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if backend == "local":
        model, tokenizer = load_model_and_tokenizer(model_name, device)
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

    results: List[Dict[str, Any]] = []

    for cart_spec in cartridge_specs:
        if backend == "local":
            _ckpt, cart_local_path, cart_label = resolve_cartridge_source(
                cart_spec,
                default_hf_filename=hf_cartridge_filename,
                hf_username=hf_username,
            )
            del _ckpt
        else:
            cart_label = cart_spec.strip()

        for ds_path_str, ds_type in dataset_configs:
            ds_path = Path(ds_path_str)
            records = load_eval_records(ds_path, ds_type)
            if num_eval_questions is not None:
                records = records[:num_eval_questions]

            desc = f"{Path(cart_label).name}|{ds_path.stem}"
            print(
                f"\n{'='*72}\n"
                f"Backend   : {backend}\n"
                f"Cartridge : {cart_label}\n"
                f"Dataset   : {ds_path} ({ds_type}, {len(records)} questions)\n"
                f"{'='*72}",
                flush=True,
            )

            if backend == "local":
                eval_rows = evaluate_local(
                    model=model,
                    tokenizer=tokenizer,
                    cartridge_path=cart_local_path,
                    eval_records=records,
                    device=device,
                    cot=cot,
                    max_answer_scan_tokens=max_answer_scan_tokens,
                    debug=debug,
                    desc=desc,
                )
            else:
                eval_rows = evaluate_tokasaurus(
                    client=client,
                    tokenizer=tokenizer,
                    cartridge_id=cart_label,
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
                "cartridge": cart_label,
                "dataset": str(ds_path),
                "dataset_type": ds_type,
                "global_stats": stats,
                "eval_results": eval_rows,
            })

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_dataset_arg(s: str) -> Tuple[str, str]:
    if ":" in s:
        path, dtype = s.rsplit(":", 1)
        dtype = dtype.strip().lower()
        if dtype not in ("mcq", "yes_no", "auto"):
            raise argparse.ArgumentTypeError(
                f"Dataset type must be mcq, yes_no, or auto; got {dtype!r}"
            )
        return path.strip(), dtype
    return s.strip(), "auto"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate cartridges on Qasper-style CSV datasets (MCQ & Yes/No).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--backend", type=str, default="local", choices=["local", "tokasaurus"],
        help="Inference backend: 'local' (load model on GPU) or 'tokasaurus' (remote server).",
    )
    parser.add_argument(
        "--cartridges", nargs="+", required=True,
        help="Cartridge specs: local .pt paths or HF repo ids.",
    )
    parser.add_argument(
        "--datasets", nargs="+", required=True,
        help="Dataset specs as path:type (mcq, yes_no, or auto). Example: /data/mcq.csv:mcq",
    )
    parser.add_argument("--model", type=str, required=True, help="HuggingFace model id.")
    parser.add_argument("--output-dir", type=str, default=".", help="Directory for output JSON files.")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--cot", action="store_true", help="Use chain-of-thought prompts.")
    parser.add_argument(
        "--max-answer-scan-tokens", type=int, default=256,
        help="Max tokens to generate when scanning for an answer token.",
    )
    parser.add_argument(
        "--num-eval-questions", type=int, default=None, metavar="N",
        help="Evaluate only the first N records per dataset.",
    )
    parser.add_argument(
        "--hf-cartridge-filename", type=str, default="cache_last.pt",
        help="Default filename when downloading from a two-segment HF repo id (local backend).",
    )
    parser.add_argument("--hf-username", type=str, default=None, help="Legacy HF username prefix.")
    parser.add_argument("--debug", action="store_true", help="Log unanswered examples to stderr.")

    toka = parser.add_argument_group("Tokasaurus backend options")
    toka.add_argument("--url", type=str, default=None, help="Tokasaurus server URL (required for tokasaurus backend).")
    toka.add_argument("--top-logprobs", type=int, default=20, help="Number of top logprobs to request per token.")
    toka.add_argument("--batch-size", type=int, default=16, help="Number of questions per Tokasaurus batch request.")

    args = parser.parse_args()

    if args.num_eval_questions is not None and args.num_eval_questions < 1:
        parser.error("--num-eval-questions must be >= 1")
    if args.backend == "tokasaurus" and args.url is None:
        parser.error("--url is required when using --backend tokasaurus")

    dataset_configs = [parse_dataset_arg(s) for s in args.datasets]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = run_all_evaluations(
        cartridge_specs=args.cartridges,
        dataset_configs=dataset_configs,
        model_name=args.model,
        backend=args.backend,
        device=args.device,
        cot=args.cot,
        max_answer_scan_tokens=args.max_answer_scan_tokens,
        num_eval_questions=args.num_eval_questions,
        hf_cartridge_filename=args.hf_cartridge_filename,
        hf_username=args.hf_username,
        debug=args.debug,
        url=args.url,
        top_logprobs=args.top_logprobs,
        batch_size=args.batch_size,
    )

    out_file = out_dir / "cartridge_eval_results.json"
    serializable = []
    for entry in results:
        serializable.append({
            "cartridge": entry["cartridge"],
            "dataset": entry["dataset"],
            "dataset_type": entry["dataset_type"],
            "global_stats": entry["global_stats"],
            "eval_results": entry["eval_results"],
        })
    out_file.write_text(json.dumps(serializable, indent=2), encoding="utf-8")

    print(f"\n{'='*72}")
    print(f"Wrote {len(results)} evaluation results to {out_file}")

    summary_file = out_dir / "cartridge_eval_summary.json"
    summary = []
    for entry in results:
        summary.append({
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
        print(f"  {s['cartridge']} x {Path(s['dataset']).name}: acc={acc}  gt_ppl={ppl}  gt_missing_topk={gt_miss}")


if __name__ == "__main__":
    main()
