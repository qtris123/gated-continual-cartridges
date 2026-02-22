"""Phase 0: Evaluate the base model (no cartridge) on all eval questions.

Runs all questions from eval_phase1 and eval_phase2 through the base Llama 3.2-3B-Instruct
model with an empty cache, then saves generations for LLM-as-a-judge scoring.

Usage:
    python experiments/continual_learning/evaluate_base_model.py

    python experiments/continual_learning/evaluate_base_model.py \
        --output-dir ./outputs \
        --device cuda
"""

import argparse
import json
import os
from pathlib import Path

import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from cartridges.cache import AttnConfig, TrainableCache
from cartridges.datasets import DataSource, GenerateEvalDataset
from cartridges.generation import flex_generate
from cartridges.models import FlexLlamaForCausalLM, HFModelConfig

SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"


def generate_base_model(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    eval_dataset: GenerateEvalDataset,
    phase_label: str,
    device: str,
    max_new_tokens: int = 256,
) -> list[dict]:
    """Generate responses for all questions using the base model (empty cache)."""
    results = []

    for idx in tqdm(range(len(eval_dataset)), desc=f"Generating base model ({phase_label})"):
        element = eval_dataset[idx]

        input_ids = element.input_ids.flatten().to(device)
        seq_ids = torch.zeros_like(input_ids)
        position_ids = torch.arange(len(input_ids), device=device)

        # Create a fresh empty cache for each question
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

        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            generated_tokens = flex_generate(
                model=model,
                tokenizer=tokenizer,
                input_ids=input_ids,
                seq_ids=seq_ids,
                position_ids=position_ids,
                cache=empty_cache,
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
        description="Phase 0: Evaluate base model (no cartridge) on all eval questions"
    )
    parser.add_argument(
        "--eval-phase1", type=str,
        default=str(SCRIPT_DIR / "data" / "eval_calendar_year" / "eval_phase1.parquet"),
        help="Path to phase 1 eval parquet",
    )
    parser.add_argument(
        "--eval-phase2", type=str,
        default=str(SCRIPT_DIR / "data" / "eval_calendar_year" / "eval_phase2.parquet"),
        help="Path to phase 2 eval parquet",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--output-dir", type=str,
        default=os.environ.get("CARTRIDGES_OUTPUT_DIR", "./outputs"),
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

    all_generations = []

    # Evaluate on each phase's eval set
    eval_sets = [
        ("phase1", args.eval_phase1),
        # ("phase2", args.eval_phase2),
    ]

    for phase_label, eval_path in eval_sets:
        if not os.path.exists(eval_path):
            print(f"WARNING: {phase_label} eval data not found at {eval_path}, skipping")
            continue

        print(f"\n{'='*60}")
        print(f"Base model evaluation: {phase_label} ({Path(eval_path).name})")
        print(f"{'='*60}")

        gen_dataset = GenerateEvalDataset.Config(
            data_source=DataSource(path=eval_path, type="local"),
        ).instantiate(tokenizer=tokenizer, seed=42)
        print(f"  {len(gen_dataset)} generation questions")

        generations = generate_base_model(
            model=model,
            tokenizer=tokenizer,
            eval_dataset=gen_dataset,
            phase_label=phase_label,
            device=device,
            max_new_tokens=args.max_new_tokens,
        )
        all_generations.extend(generations)

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    gen_path = output_dir / "base_model_generations.json"
    with open(gen_path, "w") as f:
        json.dump(all_generations, f, indent=2)
    print(f"\nSaved {len(all_generations)} base model generations to {gen_path}")


if __name__ == "__main__":
    main()
