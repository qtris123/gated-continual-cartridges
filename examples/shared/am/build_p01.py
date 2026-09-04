#!/usr/bin/env python3
"""build_p01: the stage-1 (p01) endpoint for the continual-AM matrix.

Every dataset's p01 cartridge is built the same way, so the four per-dataset
`run_*_5x5_continual.sh` stage-1 blocks (which were byte-identical arm-D
compaction) collapse into this one Python endpoint. The method is a per-dataset
*fact* read off `DatasetSpec.p01_method`:

  - "compaction"  -> backprop-free AM compaction (arm-D recipe defaults here),
                     wrapping `examples/shared/am/initial_compaction.py`.
  - "selfdistill" -> KVFromText + gradient self-distill, wrapping
                     `examples/quality/pipelines/train_initial_selfdistill.py`.

The arm-D compaction constants (NUM_TOKENS=512, KEY_SELECT=highest_attention,
ENABLE_BETA=0, REBAKE_KEY_POSITIONS=1, AM_ROPE_THETA=model,
GLOBAL_TEACHER_POSITIONS=1, RIDGE_LAMBDA=1e-4, RIDGE_SCALE=spectral, ...) are a
shared constant and live here as defaults. A `--recipe-config` YAML may carry a
top-level `p01:` section whose keys override those defaults (this is what the
`p01_rope` / `slots_per_doc` sweep manifests vary).

The build is idempotent: if a `cache_last.pt` already exists under the p1 root it
is reused (pass `--force` to rebuild).

Usage:
    python examples/shared/am/build_p01.py --dataset quality --gpu 0
    python examples/shared/am/build_p01.py --dataset techqa \\
        --recipe-config outputs/recipes/p01_rope_5e6.yaml
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import yaml

from examples.shared.am.continual_env import base_env, phase_eval_env, spec
from examples.shared.paths import ROOT

PYTHON = Path(os.environ.get("CARTRIDGES_PYTHON") or sys.executable)
COMPACTION = ROOT / "examples/shared/am/initial_compaction.py"
SELFDISTILL = ROOT / "examples/quality/pipelines/train_initial_selfdistill.py"

# The arm-D compaction recipe: a shared constant, byte-identical across datasets.
# Keys are the knob names a `--recipe-config` `p01:` section may override.
ARM_D_COMPACTION: dict[str, object] = {
    "num_tokens": 512,
    "key_select": "highest_attention",
    "granularity": "per_head",
    "enable_beta": 0,
    "rebake_key_positions": 1,
    "rope_theta": "model",
    "global_teacher_positions": 1,
    "ridge_lambda": "1e-4",
    "ridge_scale": "spectral",
    "max_ref_batches": 50,
    "max_queries_per_head": 64,
    "max_ref_examples": 256,
    "queries_per_batch": "all_tokens",
    "strip_ref_system_prompt": 1,
    "num_bg_batches": 64,
    "seed": 42,
}
COMPACTION_KNOB_ENV = {
    "num_tokens": "NUM_TOKENS",
    "key_select": "KEY_SELECT",
    "granularity": "GRANULARITY",
    "enable_beta": "ENABLE_BETA",
    "rebake_key_positions": "REBAKE_KEY_POSITIONS",
    "rope_theta": "AM_ROPE_THETA",
    "global_teacher_positions": "GLOBAL_TEACHER_POSITIONS",
    "ridge_lambda": "RIDGE_LAMBDA",
    "ridge_scale": "RIDGE_SCALE",
    "max_ref_batches": "MAX_REF_BATCHES",
    "max_queries_per_head": "MAX_QUERIES_PER_HEAD",
    "max_ref_examples": "MAX_REF_EXAMPLES",
    "queries_per_batch": "QUERIES_PER_BATCH",
    "strip_ref_system_prompt": "STRIP_REF_SYSTEM_PROMPT",
    "num_bg_batches": "NUM_BG_BATCHES",
    "seed": "SEED",
}

# Self-distill knob defaults (match the published QuALITY p01 recipe).
SELFDISTILL_DEFAULTS: dict[str, object] = {
    "num_tokens": 512,
    "num_frozen_tokens": 1,
    "lr": "2e-2",
    "epochs": 10,
    "global_batch_size": 32,
}
SELFDISTILL_KNOB_ENV = {
    "num_tokens": "NUM_TOKENS",
    "num_frozen_tokens": "NUM_FROZEN_TOKENS",
    "lr": "LR",
    "epochs": "EPOCHS",
    "global_batch_size": "GLOBAL_BATCH_SIZE",
}


def _p01_overrides(recipe_config: str | None) -> dict:
    """The `p01:` section of a recipe file, or an empty dict."""
    if not recipe_config:
        return {}
    cfg = yaml.safe_load(Path(recipe_config).read_text()) or {}
    return dict(cfg.get("p01", {}))


def _find_cache(p1_root: Path) -> Path | None:
    caches = sorted(p1_root.glob("*/*/cache_last.pt"))
    return caches[0] if caches else None


def _synth_for_compaction(dataset_spec, knobs: dict, p1_root: Path) -> Path:
    """The phase-1 synth parquet, optionally truncated to `p01.num_docs` rows.

    The slots_per_doc sweep varies the number of documents compacted into a fixed
    slot budget (fewer docs -> more slots/doc). Each row of the self-study parquet
    is one document, so a first-N-rows subset re-expresses that axis without a
    bespoke subset builder.
    """
    src = dataset_spec.synth_path(1)
    num_docs = knobs.get("num_docs")
    if not num_docs:
        return src
    import pandas as pd

    df = pd.read_parquet(src)
    out = p1_root / f"synth_p1_first{int(num_docs)}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.iloc[: int(num_docs)].to_parquet(out)
    return out


def _build_compaction_env(dataset: str, gpu: str, p1_root: Path, knobs: dict) -> dict:
    dataset_spec = spec(dataset)
    env = phase_eval_env(base_env(gpu, p1_root), dataset)
    env.update(dataset_spec.teacher_env(1))  # AM_DATASET + phase-1 teacher key
    env["QA_DATA_PATH"] = str(_synth_for_compaction(dataset_spec, knobs, p1_root))
    env["RUN_NAME"] = f"{dataset}_p1_ropefix_armD"
    for knob, value in {**ARM_D_COMPACTION, **knobs}.items():
        if knob in COMPACTION_KNOB_ENV and value is not None:
            env[COMPACTION_KNOB_ENV[knob]] = str(value)
    return env


def _build_selfdistill_env(dataset: str, gpu: str, p1_root: Path, knobs: dict) -> dict:
    dataset_spec = spec(dataset)
    env = base_env(gpu, p1_root)
    env["TEXT_PATH"] = str(ROOT / f"data/{dataset}/init_text/{dataset}_p1.txt")
    env["SYNTH_DATA_PATH"] = str(dataset_spec.synth_path(1))
    env["EVAL_DATA_PATH"] = str(dataset_spec.eval_path(1))
    env["RUN_NAME"] = f"{dataset}_p1_selfdistill"
    env["WANDB_DISABLED"] = env.get("WANDB_DISABLED", "1")
    for knob, value in {**SELFDISTILL_DEFAULTS, **knobs}.items():
        if knob in SELFDISTILL_KNOB_ENV and value is not None:
            env[SELFDISTILL_KNOB_ENV[knob]] = str(value)
    return env


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=sorted(_DATASETS()))
    parser.add_argument("--recipe-config", default=None)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--p1-root", default=None, help="where the p01 run is written")
    parser.add_argument("--force", action="store_true", help="rebuild even if a cache exists")
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()

    dataset_spec = spec(args.dataset)
    method = dataset_spec.p01_method
    p1_root = Path(args.p1_root) if args.p1_root else ROOT / f"outputs/{args.dataset}_p1_ropefix"

    existing = _find_cache(p1_root)
    if existing and not args.force:
        print(f"=== p01 already present ({method}): {existing}")
        print(existing)
        return

    knobs = _p01_overrides(args.recipe_config)
    if method == "compaction":
        step, env = COMPACTION, _build_compaction_env(args.dataset, args.gpu, p1_root, knobs)
    elif method == "selfdistill":
        step, env = SELFDISTILL, _build_selfdistill_env(args.dataset, args.gpu, p1_root, knobs)
    else:
        raise SystemExit(f"unknown p01_method {method!r} for dataset {args.dataset!r}")

    required = [PYTHON, step, dataset_spec.synth_path(1)]
    missing = [str(p) for p in required if not Path(p).exists()]
    if missing:
        raise FileNotFoundError(f"Missing required inputs: {missing}")

    print(f"dataset   {args.dataset}")
    print(f"method    {method}")
    print(f"p1 root   {p1_root}")
    print(f"recipe    {args.recipe_config or '(arm-D defaults)'}")
    if args.preflight:
        print("Preflight OK")
        return

    p1_root.mkdir(parents=True, exist_ok=True)
    subprocess.run([str(PYTHON), str(step)], env=env, cwd=ROOT, check=True)

    cache = _find_cache(p1_root)
    if cache is None:
        raise RuntimeError(f"p01 build produced no cache_last.pt under {p1_root}")
    print(f"=== p01 cache: {cache}")
    print(cache)


def _DATASETS():
    from examples.shared.am.continual_env import DATASETS

    return DATASETS


if __name__ == "__main__":
    main()
