#!/usr/bin/env python3
"""Per-layer (and multi-layer group) V-swap experiment.

For each layer L, construct a cache where V at layer L in the base cartridge
is replaced by V at the same layer in the donor cartridge, run evaluation,
and save one JSON per config.

Two swap directions:
  B_into_A: base = A (initial), graft B's V at layer L
  A_into_B: base = B (continual), graft A's V at layer L

Three eval modes:
  openended — teacher-forced log-ppl (+ optional free-form generation)
  yesno     — step-0 logits over {'Yes', 'No'}
  mcq       — step-0 logits over {'A', 'B', 'C', 'D'}

Multi-layer groups via --layer-groups-json (JSON list of
  {"name": <str>, "layers": [L1, L2, ...]}).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

from cartridges.cache import TrainableCache
from experiments.intervention import (
    DEFAULT_MODEL_KEY,
    MODE_CONFIG,
    MODEL_REGISTRY,
    aggregate_choice,
    aggregate_openended,
    build_datasets,
    build_swapped_cache,
    load_cache,
    load_model,
    parse_eval_args,
    run_choice_eval,
    run_openended_eval,
)


def _build_configs(
    cache_a: str,
    cache_b: str,
    directions: list[str],
    layers: list[int],
    layer_groups: list[dict],
    include_all_layers_swap: bool,
    skip_baselines: bool,
    num_layers: int,
    device: str,
    cache_a_label: str = "baseline_A",
    cache_b_label: str = "baseline_B",
    skip_baseline_a: bool = False,
    skip_baseline_b: bool = False,
):
    """Return a list of (label, cache_factory) pairs for the run loop."""
    configs = []

    if not skip_baselines:
        if not skip_baseline_a:
            configs.append(
                (cache_a_label, lambda: load_cache(cache_a, device))
            )
        if not skip_baseline_b:
            configs.append(
                (cache_b_label, lambda: load_cache(cache_b, device))
            )

    # (direction_label, base, donor)
    dir_specs = []
    if "B_into_A" in directions:
        dir_specs.append(("B_into_A", cache_a, cache_b))
    if "A_into_B" in directions:
        dir_specs.append(("A_into_B", cache_b, cache_a))

    for dir_label, base, donor in dir_specs:
        if include_all_layers_swap:
            all_layers = list(range(num_layers))
            configs.append((
                f"{dir_label}_Lall",
                lambda base=base, donor=donor, all_layers=all_layers:
                    build_swapped_cache(base, donor, all_layers, device),
            ))
        for L in layers:
            configs.append((
                f"{dir_label}_L{L:02d}",
                lambda base=base, donor=donor, L=L:
                    build_swapped_cache(base, donor, L, device),
            ))
        for group in layer_groups:
            name = group["name"]
            gl = list(group["layers"])
            configs.append((
                f"{dir_label}_G_{name}",
                lambda base=base, donor=donor, gl=gl:
                    build_swapped_cache(base, donor, gl, device),
            ))

    return configs


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default=DEFAULT_MODEL_KEY,
                        choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument("--cache-a", required=True,
                        help="Path to cartridge A (base / initial).")
    parser.add_argument("--cache-b", required=True,
                        help="Path to cartridge B (continual / delta).")
    parser.add_argument("--eval", action="append", required=True,
                        help="Eval parquet as name=/path/to.parquet (repeatable).")
    parser.add_argument("--mode", default="openended",
                        choices=list(MODE_CONFIG.keys()))
    parser.add_argument("--layers", default="all",
                        help="Comma-separated layer indices or 'all'.")
    parser.add_argument("--directions", default="A_into_B",
                        choices=["both", "B_into_A", "A_into_B"])
    parser.add_argument("--layer-groups-json", default=None,
                        help='JSON list of {"name": str, "layers": [L, ...]} entries.')
    parser.add_argument("--include-all-layers-swap", action="store_true")
    parser.add_argument("--skip-baselines", action="store_true")
    parser.add_argument("--skip-baseline-a", action="store_true",
                        help="Skip only the cache-A baseline eval.")
    parser.add_argument("--skip-baseline-b", action="store_true",
                        help="Skip only the cache-B baseline eval.")
    parser.add_argument("--cache-a-label", default="baseline_A",
                        help="Output filename stem for cache A baseline.")
    parser.add_argument("--cache-b-label", default="baseline_B",
                        help="Output filename stem for cache B baseline.")
    parser.add_argument("--skip-generation", action="store_true",
                        help="Open-ended only: skip free-form generation, keep only log-ppl.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--max-questions", type=int, default=None)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    device = "cuda"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer, num_layers = load_model(args.model, device)
    print(f"  num_hidden_layers={num_layers}")

    is_openended = args.mode == "openended"
    choice_tids = None
    if not is_openended:
        choice_tids = {
            lbl: tokenizer.encode(lbl, add_special_tokens=False)[0]
            for lbl in MODE_CONFIG[args.mode]["labels"]
        }
        print(f"  mode={args.mode}  choice_tids={choice_tids}")
    else:
        print(f"  mode={args.mode}  (teacher-forced log-perplexity + "
              f"{'generation' if not args.skip_generation else 'log-ppl only'})")

    eval_files = parse_eval_args(args.eval)
    datasets = build_datasets(eval_files, tokenizer, args.mode)
    for name, ds in datasets.items():
        print(f"  {name}: {len(ds)} questions")

    if args.layers == "all":
        layers = list(range(num_layers))
    else:
        layers = [int(x) for x in args.layers.split(",") if x.strip()]
    print(f"  layers: {layers}")

    directions = (["B_into_A", "A_into_B"] if args.directions == "both"
                  else [args.directions])

    layer_groups = json.loads(args.layer_groups_json) if args.layer_groups_json else []
    assert isinstance(layer_groups, list)

    configs = _build_configs(
        cache_a=args.cache_a,
        cache_b=args.cache_b,
        directions=directions,
        layers=layers,
        layer_groups=layer_groups,
        include_all_layers_swap=args.include_all_layers_swap,
        skip_baselines=args.skip_baselines,
        num_layers=num_layers,
        device=device,
        cache_a_label=args.cache_a_label,
        cache_b_label=args.cache_b_label,
        skip_baseline_a=args.skip_baseline_a,
        skip_baseline_b=args.skip_baseline_b,
    )

    summary = {}
    for label, make_cache in configs:
        print(f"\n=== {label} ===")
        cache = make_cache()

        per_config = {}
        for eval_name, ds in datasets.items():
            n = len(ds) if args.max_questions is None else min(len(ds), args.max_questions)
            if is_openended:
                raw = run_openended_eval(
                    model, tokenizer, cache, ds, n, device,
                    batch_size=args.batch_size,
                    max_new_tokens=args.max_new_tokens,
                    do_generate=not args.skip_generation,
                )
                agg = aggregate_openended(raw)
                per_config[eval_name] = agg
                extra = (f"  EM={agg['exact_match']:.2%}  contains={agg['contains_match']:.2%}  "
                         f"F1={agg['f1']:.3f}") if agg["exact_match"] is not None else ""
                print(f"  {eval_name}: nll/tok={agg['token_nll']:.3f}  "
                      f"ppl={agg['token_perplexity']:.2f}{extra}  "
                      f"({agg['num_total']} qs)")
            else:
                raw = run_choice_eval(
                    model, tokenizer, cache, ds, n, device,
                    choice_tids,
                    batch_size=args.batch_size,
                    max_new_tokens=args.max_new_tokens,
                )
                agg = aggregate_choice(raw)
                per_config[eval_name] = agg
                print(f"  {eval_name}: {agg['accuracy']:.2%} "
                      f"({agg['num_correct']}/{agg['num_total']})")

        with open(output_dir / f"{label}.json", "w") as f:
            json.dump(per_config, f, indent=2)

        if is_openended:
            keys = ("token_nll", "token_perplexity", "exact_match",
                    "contains_match", "f1", "num_total")
        else:
            keys = ("accuracy", "num_correct", "num_total")
        summary[label] = {
            name: {k: v[k] for k in keys}
            for name, v in per_config.items()
        }
        with open(output_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        del cache
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
