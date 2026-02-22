"""ROME-style causal tracing — where are facts stored in CARTRIDGE KV caches?

Corrupts the cartridge with Gaussian noise, then restores one layer at a time
to measure which layers store factual knowledge.

Usage:
    python experiments/continual_learning/causal_tracing.py \
        --cache outputs/cartridge_amd_2021_10k_512.pt \
        --eval-parquet data/eval/eval_phase1.parquet \
        --output-dir ./outputs
"""

import argparse
import json
import os
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from cartridges.cache import TrainableCache
from cartridges.generation import flex_generate
from cartridges.models import FlexLlamaForCausalLM, HFModelConfig

SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"


# ---------------------------------------------------------------------------
# Cache manipulation utilities
# ---------------------------------------------------------------------------

def _get_full_kv(cache: TrainableCache):
    """Reconstruct full keys/values by concatenating frozen + trainable."""
    n_layers = cache.config.n_layers
    keys, values = [], []
    for l in range(n_layers):
        parts_k, parts_v = [], []
        if cache._num_frozen_tokens > 0:
            parts_k.append(cache.frozen_keys[l].data)
            parts_v.append(cache.frozen_values[l].data)
        if cache._num_trainable_tokens > 0:
            parts_k.append(cache.trainable_keys[l].data)
            parts_v.append(cache.trainable_values[l].data)
        keys.append(torch.cat(parts_k, dim=2))
        values.append(torch.cat(parts_v, dim=2))
    return keys, values


def corrupt_cache_values(
    cart: TrainableCache, noise_std: float | None = None,
) -> TrainableCache:
    """Create a corrupted copy of the cache by adding Gaussian noise to all value vectors.

    Keys are left untouched. If noise_std is None, it is auto-computed as
    3 * empirical_std across all value vectors.
    """
    keys, values = _get_full_kv(cart)

    if noise_std is None:
        all_values = torch.cat([v.reshape(-1) for v in values])
        noise_std = 3.0 * all_values.std().item()
        print(f"  Auto noise_std = {noise_std:.4f}")

    corrupted_values = [v.clone() + torch.randn_like(v) * noise_std for v in values]
    corrupted_keys = [k.clone() for k in keys]

    return TrainableCache(
        config=cart.config,
        init_keys=corrupted_keys,
        init_values=corrupted_values,
        num_frozen_tokens=cart._num_frozen_tokens,
    )


def restore_layer_values(
    corrupted_cart: TrainableCache,
    clean_cart: TrainableCache,
    layer_idx: int,
) -> TrainableCache:
    """Clone corrupted cache but restore one layer's values from the clean cache."""
    corr_keys, corr_values = _get_full_kv(corrupted_cart)
    _, clean_values = _get_full_kv(clean_cart)

    restored_keys = [k.clone() for k in corr_keys]
    restored_values = [v.clone() for v in corr_values]
    restored_values[layer_idx] = clean_values[layer_idx].clone()

    return TrainableCache(
        config=corrupted_cart.config,
        init_keys=restored_keys,
        init_values=restored_values,
        num_frozen_tokens=corrupted_cart._num_frozen_tokens,
    )


# ---------------------------------------------------------------------------
# Scoring utilities
# ---------------------------------------------------------------------------

def compute_answer_logprob(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    cache: TrainableCache,
    question_input_ids: torch.Tensor,
    answer_text: str,
    device: str,
) -> dict:
    """Compute log-probability of an answer given a question and cache.

    Force-decodes the answer tokens autoregressively (capped at 32 tokens)
    and returns the average and total log-probabilities.
    """
    answer_tokens = tokenizer.encode(answer_text, add_special_tokens=False)
    score_len = min(len(answer_tokens), 128)
    if score_len == 0:
        return {"avg_logprob": 0.0, "total_logprob": 0.0, "num_tokens": 0}

    answer_tokens = answer_tokens[:score_len]

    input_ids = question_input_ids.to(device)
    seq_ids = torch.zeros_like(input_ids)
    position_ids = torch.arange(len(input_ids), device=device)

    cache = cache.to(device)

    with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        outputs = model(
            input_ids=input_ids,
            seq_ids=seq_ids,
            position_ids=position_ids,
            past_key_values=cache,
            use_cache=True,
            mode="generate",
        )

        total_logprob = 0.0
        curr_input = None
        for i, token_id in enumerate(answer_tokens):
            if i == 0:
                logits = outputs.logits[0, -1, :]
            else:
                curr_ids = torch.tensor([curr_input], device=device)
                curr_seq = torch.zeros(1, device=device, dtype=torch.long)
                curr_pos = torch.tensor([position_ids[-1] + i], device=device)
                step_out = model(
                    input_ids=curr_ids,
                    seq_ids=curr_seq,
                    position_ids=curr_pos,
                    past_key_values=cache,
                    use_cache=True,
                    mode="generate",
                )
                logits = step_out.logits[0, -1, :]

            log_probs = F.log_softmax(logits.float(), dim=-1)
            total_logprob += log_probs[token_id].item()
            curr_input = token_id

    cache.clear()

    return {
        "avg_logprob": total_logprob / score_len,
        "total_logprob": total_logprob,
        "num_tokens": score_len,
    }


def generate_answer(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    cache: TrainableCache,
    question_input_ids: torch.Tensor,
    device: str,
    max_new_tokens: int = 256,
) -> str:
    """Generate an answer for a single question with the given cache."""
    input_ids = question_input_ids.to(device)
    seq_ids = torch.zeros_like(input_ids)
    position_ids = torch.arange(len(input_ids), device=device)

    cache = cache.to(device)

    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        generated_tokens = flex_generate(
            model=model,
            tokenizer=tokenizer,
            input_ids=input_ids,
            seq_ids=seq_ids,
            position_ids=position_ids,
            cache=cache,
            max_new_tokens=max_new_tokens,
        )

    gen_token_ids = generated_tokens.get(0, [])
    return tokenizer.decode(gen_token_ids, skip_special_tokens=True)


# ---------------------------------------------------------------------------
# Experiment: Layer-level causal tracing (ROME-style)
# ---------------------------------------------------------------------------

def run_layer_tracing(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    cart: TrainableCache,
    questions: list[dict],
    device: str,
    noise_std: float | None = None,
    max_new_tokens: int = 256,
    metadata: dict | None = None,
) -> list[dict]:
    """ROME-style causal tracing: corrupt all values, restore one layer at a time.

    For each question:
      1. Clean run — score correct answer with clean cartridge
      2. Corrupted run — score with all values noised
      3. Restored runs — for each layer, restore that layer's values to clean

    Recovery = (restored - corrupted) / (clean - corrupted).
    0 means no help, 1 means full recovery.

    Returns one result dict per (question, layer) pair.
    """
    n_layers = cart.config.n_layers
    metadata = metadata or {}

    # Create corrupted cache once (noise applied to all value vectors)
    print("Creating corrupted cache...")
    cart_corrupted = corrupt_cache_values(cart, noise_std)
    cart_corrupted = cart_corrupted.to(device)

    results = []

    for q in tqdm(questions, desc="Layer tracing (questions)"):
        q_input_ids = q["input_ids"]
        q_id = q["question_id"]
        answer_ref = q["answer"]

        # 1. Clean run
        clean_scores = compute_answer_logprob(
            model, tokenizer, cart, q_input_ids, answer_ref, device,
        )
        clean_logprob = clean_scores["avg_logprob"]
        print(f"  Q{q_id} clean avg log-prob: {clean_logprob:.4f}")

        clean_answer = generate_answer(
            model, tokenizer, cart, q_input_ids, device, max_new_tokens,
        )
        print(f"  Q{q_id} clean generated answer: {clean_answer}")
        cart.clear()

        # 2. Corrupted run
        corrupted_scores = compute_answer_logprob(
            model, tokenizer, cart_corrupted, q_input_ids, answer_ref, device,
        )
        corrupted_logprob = corrupted_scores["avg_logprob"]
        print(f"  Q{q_id} corrupted avg log-prob: {corrupted_logprob:.4f}")

        corrupted_answer = generate_answer(
            model, tokenizer, cart_corrupted, q_input_ids, device, max_new_tokens,
        )
        print(f"  Q{q_id} corrupted generated answer: {corrupted_answer}")
        cart_corrupted.clear()

        gap = clean_logprob - corrupted_logprob

        # 3. Restored runs — one per layer
        for layer_idx in tqdm(range(n_layers), desc=f"  Q{q_id} layers", leave=False):
            cart_restored = restore_layer_values(cart_corrupted, cart, layer_idx)
            cart_restored = cart_restored.to(device)

            restored_scores = compute_answer_logprob(
                model, tokenizer, cart_restored, q_input_ids, answer_ref, device,
            )
            restored_logprob = restored_scores["avg_logprob"]
            print(f"Q{q_id} restoed_logprob for layer_idx {layer_idx}: {restored_logprob:.4f}") 

            recovery = (restored_logprob - corrupted_logprob) / gap if gap > 0 else 0.0

            results.append({
                **metadata,
                "question_id": q_id,
                "question": q.get("question", ""),
                "answer_ref": answer_ref,
                "clean_logprob": clean_logprob,
                "corrupted_logprob": corrupted_logprob,
                "layer": layer_idx,
                "restored_logprob": restored_logprob,
                "recovery": recovery,
                "clean_answer": clean_answer,
                "corrupted_answer": corrupted_answer,
            })

            del cart_restored

    return results


# ---------------------------------------------------------------------------
# Helpers for loading questions
# ---------------------------------------------------------------------------

def load_questions(
    parquet_path: str,
    tokenizer: AutoTokenizer,
    question_ids: list[str] | None = None,
) -> list[dict]:
    """Load questions directly from an eval parquet and tokenize prompts.

    Each parquet row must have ``messages`` (list of role/content dicts) and
    ``metadata`` (dict with at least ``question_id``).  The prompt is built
    from all messages except the last assistant answer, which becomes the
    reference answer.
    """
    df = pd.read_parquet(parquet_path)

    questions = []
    for _, row in df.iterrows():
        messages = row["messages"]
        metadata = row["metadata"]
        qid = metadata["question_id"]

        if question_ids is not None and qid not in question_ids:
            continue

        # Prompt = all messages except the last assistant answer
        prompt_messages = messages[:-1]
        answer = messages[-1]["content"]

        input_ids = tokenizer.apply_chat_template(
            prompt_messages, add_generation_prompt=True, return_tensors="pt",
        ).flatten()

        question_text = next(
            (m["content"] for m in reversed(prompt_messages) if m["role"] == "user"),
            "",
        )

        questions.append({
            "question_id": qid,
            "question": question_text,
            "input_ids": input_ids,
            "answer": answer,
        })

    return questions


def load_cache(path: str, device: str) -> TrainableCache:
    """Load a cartridge checkpoint."""
    print(f"  Loading cache: {path}")
    cache = TrainableCache.from_pretrained(path, device=device)
    cache = cache.to(device)
    n_total = cache._num_frozen_tokens + cache._num_trainable_tokens
    print(f"    {cache._num_frozen_tokens} frozen + {cache._num_trainable_tokens} trainable = {n_total} tokens")
    return cache


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="ROME-style causal tracing on CARTRIDGE KV caches"
    )
    parser.add_argument("--cache", type=str, required=True,
                        help="Path to cartridge checkpoint")
    parser.add_argument("--eval-parquet", type=str, required=True,
                        help="Path to eval parquet (e.g. data/eval/eval_phase1.parquet)")
    parser.add_argument("--question-ids", type=str, default=None,
                        help="Comma-separated question IDs to filter (e.g. S01,S02,U01)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output-dir", type=str,
                        default=os.environ.get("CARTRIDGES_OUTPUT_DIR", "./outputs"))
    parser.add_argument("--noise-std", type=float, default=None,
                        help="Gaussian noise std for corruption (default: 3x empirical std)")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()

    device = args.device
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    parsed_ids = [qid.strip() for qid in args.question_ids.split(",")] if args.question_ids else None

    # Load model
    print(f"Loading model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = HFModelConfig(
        pretrained_model_name_or_path=MODEL_NAME,
        model_cls=FlexLlamaForCausalLM,
    ).instantiate()
    model = model.to(device).to(torch.bfloat16)
    for param in model.parameters():
        param.requires_grad = False

    # Load cache
    cart = load_cache(args.cache, device)

    # Build run metadata
    run_metadata = {
        "cache_path": args.cache,
        "eval_parquet": args.eval_parquet,
        "num_frozen_tokens": cart._num_frozen_tokens,
        "num_trainable_tokens": cart._num_trainable_tokens,
        "num_total_tokens": cart._num_frozen_tokens + cart._num_trainable_tokens,
        "n_layers": cart.config.n_layers,
        "noise_std": args.noise_std,
    }

    # Load questions from parquet
    questions = load_questions(args.eval_parquet, tokenizer, question_ids=parsed_ids)
    print(f"Loaded {len(questions)} questions")

    # Run ROME-style layer tracing
    print("\n" + "=" * 60)
    print("ROME-style layer-level causal tracing")
    print("=" * 60)
    results = run_layer_tracing(
        model, tokenizer, cart, questions, device,
        noise_std=args.noise_std,
        max_new_tokens=args.max_new_tokens,
        metadata=run_metadata,
    )

    # Update noise_std in metadata if it was auto-computed
    if args.noise_std is None and results:
        run_metadata["noise_std"] = "auto (3x empirical std)"

    cache_stem = Path(args.cache).stem
    out_path = output_dir / f"causal_tracing_{cache_stem}.json"
    with open(out_path, "w") as f:
        json.dump({"metadata": run_metadata, "results": results}, f, indent=2)
    print(f"Saved {len(results)} layer tracing results to {out_path}")


if __name__ == "__main__":
    main()
