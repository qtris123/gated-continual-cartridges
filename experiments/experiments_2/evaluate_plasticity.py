"""Evaluate a single cartridge on a single eval dataset.

Usage:
    python experiments/experiments_2/evaluate_plasticity.py \\
        --cache path/to/cache_last.pt \\
        --eval-data data/eval/eval.parquet
"""

import argparse
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from cartridges.cache import AttnConfig, TrainableCache
from cartridges.datasets import DataSource, GenerateEvalDataset, LossEvalDataset
from cartridges.generation import flex_generate
from cartridges.models import FlexLlamaForCausalLM, FlexQwen3ForCausalLM, HFModelConfig
from cartridges.train import CacheAndModel

SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"


def load_cache(path: str, device: str) -> TrainableCache:
    print(f"  Loading cache: {path}")
    cache = TrainableCache.from_pretrained(path, device=device)
    cache = cache.to(device)
    n_total = cache.num_cartridge_tokens()
    print(f"    {cache._num_frozen_tokens} frozen + {cache._num_trainable_tokens} trainable = {n_total} tokens")
    return cache


def evaluate_loss(
    model: torch.nn.Module,
    cache: TrainableCache,
    eval_dataset: LossEvalDataset,
    device: str = "cuda",
) -> dict:
    dataloader = DataLoader(
        eval_dataset,
        batch_size=1,
        collate_fn=lambda x: x[0],
        num_workers=0,
    )

    wrapped = CacheAndModel(cache, model)
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

            cache.clear()

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
    device: str,
    max_new_tokens: int = 256,
) -> list[dict]:
    results = []

    for idx in tqdm(range(len(eval_dataset)), desc="Generating"):
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

        gen_token_ids = generated_tokens.get(0, [])
        generated_text = tokenizer.decode(gen_token_ids, skip_special_tokens=True)

        if isinstance(element.prompt, list):
            question_text = element.prompt[-1].get("content", "") if element.prompt else ""
        else:
            question_text = element.prompt

        results.append({
            "question_id": element.convo_id,
            "question": question_text,
            "generated_answer": generated_text,
            "reference_answer": element.answer,
            **element.metadata,
        })

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate a single cartridge on a single eval dataset")
    parser.add_argument("--cache", type=str, required=True, help="Path to cartridge checkpoint")
    parser.add_argument(
        "--eval-data", type=str,
        default=str(SCRIPT_DIR / "data" / "eval" / "eval.parquet"),
        help="Path to eval parquet",
    )
    parser.add_argument("--model", type=str, default=MODEL_NAME, help="HuggingFace model ID")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--output-dir", type=str,
        default=os.environ.get("CARTRIDGES_OUTPUT_DIR", "."),
        help="Directory to save evaluation results",
    )
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()

    device = args.device

    model_name = args.model
    model_cls = FlexQwen3ForCausalLM if "qwen" in model_name.lower() else FlexLlamaForCausalLM

    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = HFModelConfig(
        pretrained_model_name_or_path=model_name,
        model_cls=model_cls,
    ).instantiate()
    model = model.to(device).to(torch.bfloat16)
    for param in model.parameters():
        param.requires_grad = False

    cache = load_cache(args.cache, device)
    cartridge_size = cache.num_cartridge_tokens()

    # --- Log-perplexity ---
    print("\nComputing log-perplexity...")
    loss_dataset = LossEvalDataset.Config(
        data_source=DataSource(path=args.eval_data, type="local"),
    ).instantiate(tokenizer=tokenizer, seed=42)
    print(f"  {len(loss_dataset)} loss-eval samples")

    ppl_result = evaluate_loss(model, cache, loss_dataset, device=device)
    ppl_result["cache"] = args.cache
    ppl_result["eval_set"] = args.eval_data
    print(f"  log-perplexity: {ppl_result['log_perplexity']:.4f} ({ppl_result['num_tokens']} tokens)")

    # --- Generation ---
    print("\nGenerating responses...")
    gen_dataset = GenerateEvalDataset.Config(
        data_source=DataSource(path=args.eval_data, type="local"),
    ).instantiate(tokenizer=tokenizer, seed=42)
    print(f"  {len(gen_dataset)} generation questions")

    generations = generate_for_dataset(
        model=model,
        tokenizer=tokenizer,
        cache=cache,
        eval_dataset=gen_dataset,
        device=device,
        max_new_tokens=args.max_new_tokens,
    )

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ppl_path = output_dir / f"eval_perplexity_{cartridge_size}tok.json"
    with open(ppl_path, "w") as f:
        json.dump(ppl_result, f, indent=2)
    print(f"\nSaved perplexity results to {ppl_path}")

    gen_path = output_dir / f"eval_generations_{cartridge_size}tok.json"
    with open(gen_path, "w") as f:
        json.dump(generations, f, indent=2)
    print(f"Saved {len(generations)} generations to {gen_path}")


if __name__ == "__main__":
    main()
