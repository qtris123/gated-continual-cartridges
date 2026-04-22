"""Evaluation loops for cartridge intervention experiments.

Two evaluators:
  run_openended_eval — teacher-forced log-perplexity over the gold answer span
                       plus optional free-form generation (scored with
                       exact / contains / F1).
  run_choice_eval    — one forward pass; the step-0 logits over the label
                       tokens give the score.

Each returns a list of per-question result dicts. Use aggregate_openended /
aggregate_choice to fold them into summary stats + the per-question list.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from cartridges.benchmark.scorers import contains_match, exact_match, f1_score
from cartridges.generation import flex_generate


def _extract_question(element) -> str:
    if isinstance(element.prompt, list):
        for m in reversed(element.prompt):
            if m.get("role") == "user":
                return m.get("content", "")
        return ""
    return element.prompt or ""


def run_openended_eval(
    model,
    tokenizer,
    cache,
    dataset,
    n: int,
    device: str,
    batch_size: int,
    max_new_tokens: int = 32,
    do_generate: bool = True,
):
    """Open-ended QA evaluation.

    Log-perplexity: one teacher-forced forward pass per batch; slice logits for
    each sequence's answer span and compute per-token cross-entropy. Returns
    per-question mean NLL (natural log) and the total-NLL / total-tokens
    aggregates.

    Generation (optional): separate call to flex_generate from the question-only
    prompt; decode, compare to gold with contains_match / exact_match / f1.
    """
    results = []
    for batch_start in range(0, n, batch_size):
        batch_end = min(batch_start + batch_size, n)
        batch = [dataset[i] for i in range(batch_start, batch_end)]

        # --- log-perplexity forward pass (packed, teacher-forced) ---
        all_ids, all_seq, all_pos = [], [], []
        seq_offsets = []  # start of each sequence within the packed tensor
        for i, element in enumerate(batch):
            full_ids = torch.tensor(
                element.metadata["full_input_ids"], dtype=torch.long, device=device
            )
            seq_offsets.append(sum(len(x) for x in all_ids))
            all_ids.append(full_ids)
            all_seq.append(torch.full_like(full_ids, i))
            all_pos.append(torch.arange(len(full_ids), device=device))

        packed_ids = torch.cat(all_ids)
        packed_seq = torch.cat(all_seq)
        packed_pos = torch.cat(all_pos)

        with torch.no_grad():
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = model(
                    input_ids=packed_ids,
                    seq_ids=packed_seq,
                    position_ids=packed_pos,
                    past_key_values=cache,
                    use_cache=True,
                    mode="generate",
                )
        logits = outputs.logits[0]
        # Clear the cache so question tokens don't pollute the next config.
        cache.clear()

        batch_nll = []
        for i, element in enumerate(batch):
            s = seq_offsets[i]
            a_start = element.metadata["answer_start"]
            a_end = element.metadata["answer_end"]
            # Predict target token t at (s + t) using logits at (s + t - 1).
            pred_positions = torch.arange(s + a_start - 1, s + a_end - 1, device=device)
            target_positions = torch.arange(s + a_start, s + a_end, device=device)
            pred_logits = logits[pred_positions].float()
            target_ids = packed_ids[target_positions]
            token_nll = F.cross_entropy(pred_logits, target_ids, reduction="none")
            mean_nll = token_nll.mean().item()
            total_nll = token_nll.sum().item()
            n_tokens = token_nll.shape[0]
            batch_nll.append((mean_nll, total_nll, n_tokens))

        # --- generation pass (question-only prompt -> free-form answer) ---
        generated_texts = [None] * len(batch)
        if do_generate:
            gen_ids_list, gen_seq_list, gen_pos_list = [], [], []
            for i, element in enumerate(batch):
                ids = element.input_ids.flatten().to(device)
                gen_ids_list.append(ids)
                gen_seq_list.append(torch.full_like(ids, i))
                gen_pos_list.append(torch.arange(len(ids), device=device))
            batched_input_ids = torch.cat(gen_ids_list)
            batched_seq_ids = torch.cat(gen_seq_list)
            batched_position_ids = torch.cat(gen_pos_list)

            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                gen_tokens, _ = flex_generate(
                    model=model,
                    tokenizer=tokenizer,
                    input_ids=batched_input_ids,
                    seq_ids=batched_seq_ids,
                    position_ids=batched_position_ids,
                    cache=cache,
                    max_new_tokens=max_new_tokens,
                    collect_token_ids=[],
                )
            for i in range(len(batch)):
                gen_ids = gen_tokens.get(i, [])
                generated_texts[i] = tokenizer.decode(gen_ids, skip_special_tokens=True)

        for i, element in enumerate(batch):
            mean_nll, total_nll, n_tokens = batch_nll[i]
            gen_text = generated_texts[i] if do_generate else ""
            gold = element.answer or ""
            em = float(exact_match(gen_text, gold)) if do_generate else None
            cm = float(contains_match(gen_text, gold)) if do_generate else None
            f1 = float(f1_score(gen_text, gold)) if do_generate else None

            results.append({
                "question_id": element.convo_id,
                "doc_source": element.metadata.get("doc_source"),
                "question": _extract_question(element),
                "reference_answer": gold,
                "generated_text": gen_text,
                "mean_nll": mean_nll,
                "perplexity": math.exp(mean_nll) if mean_nll < 50 else float("inf"),
                "total_nll": total_nll,
                "num_answer_tokens": n_tokens,
                "exact_match": em,
                "contains_match": cm,
                "f1": f1,
            })
    return results


def run_choice_eval(
    model,
    tokenizer,
    cache,
    dataset,
    n: int,
    device: str,
    choice_tids: dict,
    batch_size: int,
    max_new_tokens: int,
):
    """Evaluate a dataset whose answers are one of the labels in `choice_tids`.

    `choice_tids` maps label string (e.g. 'Yes', 'A') -> token id. The
    predicted label is argmax over their step-0 logprobs.
    """
    token_ids = list(choice_tids.values())
    labels = list(choice_tids.keys())
    results = []
    for batch_start in range(0, n, batch_size):
        batch_end = min(batch_start + batch_size, n)
        batch = [dataset[i] for i in range(batch_start, batch_end)]

        all_input_ids, all_seq_ids, all_position_ids = [], [], []
        for i, element in enumerate(batch):
            ids = element.input_ids.flatten().to(device)
            all_input_ids.append(ids)
            all_seq_ids.append(torch.full_like(ids, i))
            all_position_ids.append(torch.arange(len(ids), device=device))
        batched_input_ids = torch.cat(all_input_ids)
        batched_seq_ids = torch.cat(all_seq_ids)
        batched_position_ids = torch.cat(all_position_ids)

        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            generated_tokens, collected = flex_generate(
                model=model,
                tokenizer=tokenizer,
                input_ids=batched_input_ids,
                seq_ids=batched_seq_ids,
                position_ids=batched_position_ids,
                cache=cache,
                max_new_tokens=max_new_tokens,
                collect_token_ids=token_ids,
            )

        for i, element in enumerate(batch):
            entry = collected[i][0]
            choice_logprobs = {lbl: entry[choice_tids[lbl]] for lbl in labels}
            predicted = max(choice_logprobs, key=choice_logprobs.get)
            ref = (element.answer or "").strip().lower()
            correct = predicted.strip().lower() == ref

            gen_ids = generated_tokens.get(i, [])
            generated_text = tokenizer.decode(gen_ids, skip_special_tokens=True)

            results.append({
                "question_id": element.convo_id,
                "doc_source": element.metadata.get("doc_source"),
                "question": _extract_question(element),
                "reference_answer": element.answer,
                "predicted": predicted,
                "generated_text": generated_text,
                "correct": bool(correct),
                "choice_logprobs": choice_logprobs,
            })
    return results


def aggregate_openended(results: list) -> dict:
    """Fold per-question open-ended results into summary stats + the result list."""
    total_nll = sum(r["total_nll"] for r in results)
    total_tok = sum(r["num_answer_tokens"] for r in results)
    token_nll = total_nll / max(total_tok, 1)
    token_ppl = math.exp(token_nll) if token_nll < 50 else float("inf")
    have_gen = results and results[0]["exact_match"] is not None
    em = (sum(r["exact_match"] for r in results) / max(len(results), 1)
          if have_gen else None)
    cm = (sum(r["contains_match"] for r in results) / max(len(results), 1)
          if have_gen else None)
    f1 = (sum(r["f1"] for r in results) / max(len(results), 1)
          if have_gen else None)
    return {
        "token_nll": token_nll,
        "token_perplexity": token_ppl,
        "exact_match": em,
        "contains_match": cm,
        "f1": f1,
        "num_total": len(results),
        "results": results,
    }


def aggregate_choice(results: list) -> dict:
    """Fold per-question choice results into summary stats + the result list."""
    num_correct = sum(r["correct"] for r in results)
    return {
        "accuracy": num_correct / max(len(results), 1),
        "num_correct": num_correct,
        "num_total": len(results),
        "results": results,
    }
