"""Evaluate plasticity by comparing continual and fresh cartridges on the same phase 2 eval set.

Evaluation matrix:
  Phase 1 cache  + eval_phase1.parquet → Baseline (pre-update)
  Phase 2 cache  + eval_phase2.parquet → Continual cartridge (sequential training)
  Fresh cache    + eval_phase2.parquet → Fresh cartridge (trained directly on A+B)

Comparing phase2 vs fresh on the same eval set measures plasticity: if fresh > phase2,
the continual cartridge has reduced ability to absorb new information.

Usage:
    python experiments/experiments_2/evaluate_plasticity.py \\
        --phase1-cache path/to/phase1/cache_last.pt \\
        --phase2-cache path/to/continual/cache_last.pt \\
        --fresh-cache  path/to/fresh/cache_last.pt \\
        --eval-phase2  data/eval/eval_phase2.parquet
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
    n_total = cache.num_cartridge_tokens()
    print(f"    {cache._num_frozen_tokens} frozen + {cache._num_trainable_tokens} trainable = {n_total} tokens")
    return cache


def evaluate_loss(
    model: torch.nn.Module,
    cache: Optional[TrainableCache],
    eval_dataset: LossEvalDataset,
    device: str = "cuda",
) -> dict:
    """Compute mean cross-entropy loss on an evaluation dataset."""
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
    phase_label: str,
    device: str,
    max_new_tokens: int = 256,
) -> list[dict]:
    """Generate responses for all questions in an eval dataset."""
    results = []

    for idx in tqdm(range(len(eval_dataset)), desc=f"Generating ({phase_label})"):
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
            "phase": phase_label,
            **element.metadata,
        })

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate plasticity: compare continual vs fresh cartridges on phase 2 eval"
    )
    parser.add_argument(
        "--phase1-cache", type=str, required=True,
        help="Path to phase 1 cartridge checkpoint (evaluated on eval_phase1.parquet)",
    )
    parser.add_argument(
        "--phase2-cache", type=str, default=None,
        help="Path to continual phase 2 cartridge checkpoint (evaluated on eval_phase2.parquet)",
    )
    parser.add_argument(
        "--fresh-cache", type=str, default=None,
        help="Path to fresh cartridge checkpoint trained directly on A+B (evaluated on eval_phase2.parquet)",
    )
    parser.add_argument(
        "--eval-phase1", type=str,
        default=str(SCRIPT_DIR / "data" / "eval" / "eval_phase1.parquet"),
        help="Path to phase 1 eval parquet",
    )
    parser.add_argument(
        "--eval-phase2", type=str,
        default=str(SCRIPT_DIR / "data" / "eval" / "eval_phase2.parquet"),
        help="Path to phase 2 eval parquet",
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

    # Build evaluation pairs: (label, cache_path, eval_path)
    eval_pairs = [("phase1", args.phase1_cache, args.eval_phase1)]
    if args.phase2_cache is not None:
        eval_pairs.append(("phase2", args.phase2_cache, args.eval_phase2))
    if args.fresh_cache is not None:
        eval_pairs.append(("fresh", args.fresh_cache, args.eval_phase2))

    all_perplexity = {}
    all_generations = []
    cartridge_size = None

    for phase_label, cache_path, eval_path in eval_pairs:
        if not os.path.exists(eval_path):
            print(f"WARNING: {phase_label} eval data not found at {eval_path}, skipping")
            continue

        print(f"\n{'='*60}")
        print(f"Evaluating: {phase_label} cache on {Path(eval_path).name}")
        print(f"{'='*60}")

        cache = load_cache(cache_path, device)

        if cartridge_size is None:
            cartridge_size = cache.num_cartridge_tokens()

        # --- Log-perplexity ---
        print(f"\n[{phase_label}] Computing log-perplexity...")
        loss_dataset = LossEvalDataset.Config(
            data_source=DataSource(path=eval_path, type="local"),
        ).instantiate(tokenizer=tokenizer, seed=42)
        print(f"  {len(loss_dataset)} loss-eval samples")

        ppl_result = evaluate_loss(model, cache, loss_dataset, device=device)
        ppl_result["cache"] = cache_path
        ppl_result["eval_set"] = eval_path
        all_perplexity[phase_label] = ppl_result
        print(f"  log-perplexity: {ppl_result['log_perplexity']:.4f} ({ppl_result['num_tokens']} tokens)")

        # --- Generation ---
        print(f"\n[{phase_label}] Generating responses...")
        gen_dataset = GenerateEvalDataset.Config(
            data_source=DataSource(path=eval_path, type="local"),
        ).instantiate(tokenizer=tokenizer, seed=42)
        print(f"  {len(gen_dataset)} generation questions")

        generations = generate_for_dataset(
            model=model,
            tokenizer=tokenizer,
            cache=cache,
            eval_dataset=gen_dataset,
            phase_label=phase_label,
            device=device,
            max_new_tokens=args.max_new_tokens,
        )
        all_generations.extend(generations)

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ppl_path = output_dir / f"plasticity_eval_perplexity_{cartridge_size}tok.json"
    with open(ppl_path, "w") as f:
        json.dump(all_perplexity, f, indent=2)
    print(f"\nSaved perplexity results to {ppl_path}")

    gen_path = output_dir / f"plasticity_eval_generations_{cartridge_size}tok.json"
    with open(gen_path, "w") as f:
        json.dump(all_generations, f, indent=2)
    print(f"Saved {len(all_generations)} generations to {gen_path}")


if __name__ == "__main__":
    main()
