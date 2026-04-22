#!/usr/bin/env python3
"""Per-slot A→B V-swap sweep.

For each slot position p (one of n_slots 'virtual tokens' in the cartridge),
swap V at p from cartridge A into cartridge B *across all layers
simultaneously*. Measure aggregate log-perplexity on an open-ended eval.

Saves `[n_slots]` tensor of post-swap log-perplexity + baselines.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from experiments.intervention import (
    DEFAULT_MODEL_KEY,
    MODEL_REGISTRY,
    build_datasets,
    load_cache,
    load_model,
    parse_eval_args,
    per_token_slot_swap_trace,
    per_token_slot_swap_trace_per_q,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default=DEFAULT_MODEL_KEY,
                        choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument("--cache-a", required=True, help="Donor (phase-1 / A).")
    parser.add_argument("--cache-b", required=True, help="Base (phase-2 / B).")
    parser.add_argument("--eval", required=True,
                        help="name=/path/to.parquet (one open-ended eval).")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--per-question", action="store_true",
                        help="Also record per-question NLL grid (n_q, n_slots) "
                             "and question IDs. Slight metadata overhead.")
    args = parser.parse_args()

    device = "cuda"
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer, num_layers = load_model(args.model, device)
    print(f"  num_hidden_layers={num_layers}")

    eval_files = parse_eval_args([args.eval])
    datasets = build_datasets(eval_files, tokenizer, "openended")
    (eval_name, ds), = datasets.items()
    print(f"  eval={eval_name}  size={len(ds)}")

    base_cache = load_cache(args.cache_b, device)
    donor_cache = load_cache(args.cache_a, device)
    n_slots = base_cache.trainable_values[0].shape[2]
    print(f"  n_slots={n_slots}  (each slot swapped across all {num_layers} layers)")

    t0 = time.time()
    if args.per_question:
        base_per_q, donor_per_q, grid = per_token_slot_swap_trace_per_q(
            model, base_cache, donor_cache, ds, device,
            batch_size=args.batch_size,
        )
        base_nll = float(base_per_q.mean())
        donor_nll = float(donor_per_q.mean())
        nlls = grid.mean(dim=0)
        question_ids = [ds[i].convo_id for i in range(len(ds))]
    else:
        base_nll, donor_nll, nlls = per_token_slot_swap_trace(
            model, base_cache, donor_cache, ds, device,
            batch_size=args.batch_size,
        )
        grid = None
        base_per_q = None
        donor_per_q = None
        question_ids = None
    dt = time.time() - t0
    print(f"\nDone in {dt:.1f}s ({dt / n_slots * 1000:.1f}ms/slot)")
    print(f"log_ppl_B (base) = {base_nll:.4f}   log_ppl_A (donor) = {donor_nll:.4f}")
    gap = base_nll - donor_nll
    R = (base_nll - nlls) / gap if gap != 0 else torch.zeros_like(nlls)
    print(f"R range: min={R.min().item():+.3f}  max={R.max().item():+.3f}")

    bundle = {"log_ppl_per_slot": nlls, "R": R,
              "base_nll": base_nll, "donor_nll": donor_nll}
    if args.per_question:
        bundle["grid_per_q"] = grid
        bundle["base_per_q"] = base_per_q
        bundle["donor_per_q"] = donor_per_q
        bundle["question_ids"] = question_ids
    torch.save(bundle, out_dir / "per_slot_swap.pt")
    meta = {
        "model": args.model,
        "cache_a_donor": args.cache_a,
        "cache_b_base": args.cache_b,
        "eval_name": eval_name,
        "eval_path": eval_files[eval_name],
        "base_nll": base_nll,
        "donor_nll": donor_nll,
        "n_slots": int(n_slots),
        "num_layers": int(num_layers),
        "elapsed_sec": dt,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Saved {out_dir}/per_slot_swap.pt and meta.json")


if __name__ == "__main__":
    main()
