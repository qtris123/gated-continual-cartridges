"""Evaluate sequential cross-document cartridges via perplexity + generation.

Evaluation matrix (4 conditions):
  Phase 1 cache + eval_amd       → AMD baseline
  Phase 2 cache + eval_amd       → AMD forgetting
  Phase 2 cache + eval_pepsi     → PepsiCo learning
  Phase 2 cache + eval_cross_doc → Cross-document reasoning

Usage:
    python experiments/sequential_documents/evaluate_sequential.py \
        --phase1-cache path/to/phase1/cache_last.pt \
        --phase2-cache path/to/phase2/cache_last.pt

    python experiments/sequential_documents/evaluate_sequential.py \
        --phase1-cache path/to/phase1/cache_last.pt \
        --phase2-cache path/to/phase2/cache_last.pt \
        --output-dir ./outputs
"""

import argparse
import json
import os
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from cartridges.cache import AttnConfig, TrainableCache
from cartridges.datasets import DataSource, GenerateEvalDataset, LossEvalDataset
from cartridges.generation import flex_generate
from cartridges.models import FlexLlamaForCausalLM, HFModelConfig
from cartridges.train import CacheAndModel

SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"


def load_cache(path: str, device: str) -> TrainableCache:
    """Load a cartridge checkpoint and move to device."""
    print(f"  Loading cache: {path}")
    cache = TrainableCache.from_pretrained(path, device=device)
    cache = cache.to(device)
    n_total = cache._num_frozen_tokens + cache._num_trainable_tokens
    print(f"    {cache._num_frozen_tokens} frozen + {cache._num_trainable_tokens} trainable = {n_total} tokens")
    return cache


def evaluate_loss(
    model: torch.nn.Module,
    cache: Optional[TrainableCache],
    eval_dataset: LossEvalDataset,
    device: str = "cuda",
) -> dict:
    """Compute mean cross-entropy loss on an evaluation dataset.

    Returns dict with 'loss', 'log_perplexity', 'num_tokens', 'num_batches'.
    """
    dataloader = DataLoader(
        eval_dataset,
        batch_size=1,
        collate_fn=lambda x: x[0],
        num_workers=0,
    )

    if cache is not None:
        wrapped = CacheAndModel(cache, model)
    else:
        empty_cache = TrainableCache(config=AttnConfig(
            n_layers=model.config.num_hidden_layers,
            n_heads=model.config.num_key_value_heads,
            head_dim=(
                model.config.head_dim
                if hasattr(model.config, "head_dim")
                else model.config.hidden_size // model.config.num_attention_heads
            ),
        ))
        empty_cache = empty_cache.to(device)
        wrapped = CacheAndModel(empty_cache, model)

    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = wrapped(
                    input_ids=batch.input_ids.to(device),
                    seq_ids=batch.element_ids.to(device),
                    position_ids=batch.position_ids.to(device),
                )

                topk_pred_logprobs = F.log_softmax(outputs.logits, dim=-1)[
                    0,
                    batch.topk_token_idxs.to(device) - 1,
                    batch.topk_token_ids.to(device),
                ]

                ce_by_token = (
                    -batch.topk_logprobs.to(device).exp() * topk_pred_logprobs
                )

                total_loss += ce_by_token.sum().item()
                total_tokens += ce_by_token.shape[0]

            if cache is not None:
                cache.clear()
            else:
                empty_cache.clear()

    mean_loss = total_loss / max(total_tokens, 1)
    return {
        "loss": mean_loss,
        "log_perplexity": mean_loss,
        "num_tokens": total_tokens,
        "num_batches": len(dataloader),
    }


def generate_for_dataset(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    cache: TrainableCache,
    eval_dataset: GenerateEvalDataset,
    condition_label: str,
    device: str,
    max_new_tokens: int = 256,
) -> list[dict]:
    """Generate responses for all questions in an eval dataset.

    Returns a list of dicts with question, generated answer, reference, etc.
    """
    results = []

    for idx in tqdm(range(len(eval_dataset)), desc=f"Generating ({condition_label})"):
        element = eval_dataset[idx]

        input_ids = element.input_ids.flatten().to(device)
        seq_ids = torch.zeros_like(input_ids)
        position_ids = torch.arange(len(input_ids), device=device)

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

        # Decode generated tokens (sequence id 0)
        gen_token_ids = generated_tokens.get(0, [])
        generated_text = tokenizer.decode(gen_token_ids, skip_special_tokens=True)

        # Build prompt string from messages
        if isinstance(element.prompt, list):
            question_text = element.prompt[-1].get("content", "") if element.prompt else ""
        else:
            question_text = element.prompt

        results.append({
            "question_id": element.convo_id,
            "question": question_text,
            "generated_answer": generated_text,
            "reference_answer": element.answer,
            "condition": condition_label,
            **element.metadata,
        })

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate sequential cross-document cartridges via perplexity + generation"
    )
    parser.add_argument(
        "--phase1-cache", type=str, required=True,
        help="Path to Phase 1 (AMD) cartridge checkpoint",
    )
    parser.add_argument(
        "--phase2-cache", type=str, required=True,
        help="Path to Phase 2 (AMD+PepsiCo) cartridge checkpoint",
    )
    parser.add_argument(
        "--eval-amd", type=str,
        default=str(SCRIPT_DIR / "data" / "eval" / "eval_amd.parquet"),
        help="Path to AMD eval parquet",
    )
    parser.add_argument(
        "--eval-pepsi", type=str,
        default=str(SCRIPT_DIR / "data" / "eval" / "eval_pepsi.parquet"),
        help="Path to PepsiCo eval parquet",
    )
    parser.add_argument(
        "--eval-cross-doc", type=str,
        default=str(SCRIPT_DIR / "data" / "eval" / "eval_cross_doc.parquet"),
        help="Path to cross-document eval parquet",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--output-dir", type=str,
        default=os.environ.get("CARTRIDGES_OUTPUT_DIR", "."),
        help="Directory to save evaluation results",
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=256,
        help="Maximum number of tokens to generate per question",
    )
    args = parser.parse_args()

    device = args.device

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

    # Build evaluation conditions: (label, cache_path, eval_path)
    eval_conditions = [
        ("phase1_amd", args.phase1_cache, args.eval_amd),
        ("phase2_amd", args.phase2_cache, args.eval_amd),
        ("phase2_pepsi", args.phase2_cache, args.eval_pepsi),
        ("phase2_cross_doc", args.phase2_cache, args.eval_cross_doc),
    ]

    all_perplexity = {}
    all_generations = []

    for condition_label, cache_path, eval_path in eval_conditions:
        if not os.path.exists(eval_path):
            print(f"WARNING: eval data not found at {eval_path}, skipping {condition_label}")
            continue

        print(f"\n{'='*60}")
        print(f"Evaluating: {condition_label}")
        print(f"  cache: {cache_path}")
        print(f"  eval:  {eval_path}")
        print(f"{'='*60}")

        # Load cache for this condition
        cache = load_cache(cache_path, device)

        # --- Log-perplexity ---
        print(f"\n[{condition_label}] Computing log-perplexity...")
        loss_dataset = LossEvalDataset.Config(
            data_source=DataSource(path=eval_path, type="local"),
        ).instantiate(tokenizer=tokenizer, seed=42)
        print(f"  {len(loss_dataset)} loss-eval samples")

        ppl_result = evaluate_loss(model, cache, loss_dataset, device=device)
        ppl_result["cache"] = cache_path
        ppl_result["eval_set"] = eval_path
        all_perplexity[condition_label] = ppl_result
        print(f"  log-perplexity: {ppl_result['log_perplexity']:.4f} ({ppl_result['num_tokens']} tokens)")

        # --- Generation ---
        print(f"\n[{condition_label}] Generating responses...")
        gen_dataset = GenerateEvalDataset.Config(
            data_source=DataSource(path=eval_path, type="local"),
        ).instantiate(tokenizer=tokenizer, seed=42)
        print(f"  {len(gen_dataset)} generation questions")

        generations = generate_for_dataset(
            model=model,
            tokenizer=tokenizer,
            cache=cache,
            eval_dataset=gen_dataset,
            condition_label=condition_label,
            device=device,
            max_new_tokens=args.max_new_tokens,
        )
        all_generations.extend(generations)

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ppl_path = output_dir / "sequential_eval_perplexity.json"
    with open(ppl_path, "w") as f:
        json.dump(all_perplexity, f, indent=2)
    print(f"\nSaved perplexity results to {ppl_path}")

    gen_path = output_dir / "sequential_eval_generations.json"
    with open(gen_path, "w") as f:
        json.dump(all_generations, f, indent=2)
    print(f"Saved {len(all_generations)} generations to {gen_path}")

    # Print summary
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    for label, result in all_perplexity.items():
        print(f"  {label:20s}  log-ppl={result['log_perplexity']:.4f}  tokens={result['num_tokens']}")


if __name__ == "__main__":
    main()
