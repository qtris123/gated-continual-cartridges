"""Shared library for cartridge intervention experiments.

Public API re-exports so scripts can do:

    from experiments.intervention import (
        MODEL_REGISTRY, DEFAULT_MODEL_KEY, load_model,
        MODE_CONFIG, OpenEndedEvalDataset, ChoiceGenerateDataset,
        parse_eval_args, build_datasets,
        run_openended_eval, run_choice_eval,
        aggregate_openended, aggregate_choice,
        load_cache, build_swapped_cache, build_zeroed_cache,
    )
"""
from .caches import build_swapped_cache, build_zeroed_cache, load_cache
from .datasets import (
    MODE_CONFIG,
    ChoiceGenerateDataset,
    OpenEndedEvalDataset,
    build_datasets,
    parse_eval_args,
)
from .evals import (
    aggregate_choice,
    aggregate_openended,
    run_choice_eval,
    run_openended_eval,
)
from .models import DEFAULT_MODEL_KEY, MODEL_REGISTRY, load_model
from .per_slot import per_slot_trace
from .per_slot_swap import (
    per_layer_swap_trace_per_q,
    per_slot_swap_trace,
    per_token_slot_swap_trace,
    per_token_slot_swap_trace_per_q,
)

__all__ = [
    "per_slot_trace",
    "per_slot_swap_trace",
    "per_token_slot_swap_trace",
    "per_token_slot_swap_trace_per_q",
    "per_layer_swap_trace_per_q",
    "MODEL_REGISTRY",
    "DEFAULT_MODEL_KEY",
    "load_model",
    "MODE_CONFIG",
    "OpenEndedEvalDataset",
    "ChoiceGenerateDataset",
    "parse_eval_args",
    "build_datasets",
    "run_openended_eval",
    "run_choice_eval",
    "aggregate_openended",
    "aggregate_choice",
    "load_cache",
    "build_swapped_cache",
    "build_zeroed_cache",
]
