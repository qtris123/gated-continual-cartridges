#!/usr/bin/env python3
"""grad_step: the gradient/sparse continual step engine for run_chain.

`run_chain` owns the shared stage loop, per-stage 5x5 eval, and matrix emission,
and dispatches only the per-stage *step* to one of two engines that each own
their execution model:

  - AM     -> examples/shared/am/continual_write.py (closed-form, single process)
  - gradient -> this module (optimizer; optionally torchrun data-parallel)

Density is a recipe knob, not a separate method: the recipe's ``gradient:``
section (read from ``$RECIPE_CONFIG``) selects ``density: dense`` ->
train/continual_perplexity.py (every position updated) or ``density: sparse`` ->
train/continual_sparse_perplexity.py (TF-IDF masked), plus ``lr`` / ``epochs`` /
``num_gpus``. On success this wrapper writes an AM-compatible
`phase2_summary.json` beside the produced cache and, when `STEP_RESULT_JSON` is
set, a pointer file so run_chain can locate `cache_last.pt` uniformly across both
methods.

Runtime env (set by run_chain): PHASE1_CACHE_PATH, SYNTH_DATA_PATH, RUN_NAME,
CARTRIDGES_OUTPUT_DIR (required); RECIPE_CONFIG, STEP_RESULT_JSON (optional).
The config knobs come from RECIPE_CONFIG, not the environment.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import uuid
from pathlib import Path

import yaml

from examples.shared.paths import ROOT

PYTHON = Path(os.environ.get("CARTRIDGES_PYTHON") or sys.executable)
DENSE = ROOT / "examples/shared/train/continual_perplexity.py"
SPARSE = ROOT / "examples/shared/train/continual_sparse_perplexity.py"


def _gradient_recipe() -> dict:
    """The recipe's `gradient:` section, read from RECIPE_CONFIG (no env config)."""
    recipe_path = os.environ.get("RECIPE_CONFIG")
    if not recipe_path:
        return {}
    cfg = yaml.safe_load(Path(recipe_path).read_text()) or {}
    return dict(cfg.get("gradient", {}) or {})


def _trainer(density: str) -> Path:
    density = str(density).strip().lower()
    if density == "sparse":
        return SPARSE
    if density == "dense":
        return DENSE
    raise SystemExit(f"unknown gradient.density {density!r} (expected dense|sparse)")


def main() -> None:
    if any(a in ("-h", "--help") for a in sys.argv[1:]):
        print(__doc__)
        return
    for req in ("PHASE1_CACHE_PATH", "SYNTH_DATA_PATH", "RUN_NAME"):
        if not os.environ.get(req):
            raise SystemExit(f"grad_step requires ${req}")
    grad = _gradient_recipe()
    trainer = _trainer(grad.get("density", "dense"))

    # Isolate this stage's output so its cache_last.pt is unambiguous.
    base_out = Path(os.environ.get("CARTRIDGES_OUTPUT_DIR", str(ROOT / "outputs")))
    stage_out = base_out / f"{os.environ['RUN_NAME']}-{uuid.uuid4().hex[:8]}"
    stage_out.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["CARTRIDGES_OUTPUT_DIR"] = str(stage_out)
    # Translate the (file-sourced) gradient knobs into the downstream trainer's
    # runtime env. The trainers live outside the AM path and remain env-driven.
    if grad.get("lr") is not None:
        env["LR"] = str(grad["lr"])
    if grad.get("epochs") is not None:
        env["EPOCHS"] = str(grad["epochs"])

    ngpu = int(grad.get("num_gpus", 1))
    if ngpu > 1:
        cmd = ["torchrun", f"--nproc_per_node={ngpu}", str(trainer)]
    else:
        cmd = [str(PYTHON), str(trainer)]
    subprocess.run(cmd, env=env, cwd=ROOT, check=True)

    caches = sorted(stage_out.glob("**/cache_last.pt"), key=lambda p: p.stat().st_mtime)
    if not caches:
        raise RuntimeError(f"grad_step produced no cache_last.pt under {stage_out}")
    cache = caches[-1]
    summary = {
        "run_name": os.environ["RUN_NAME"],
        "run_dir": str(cache.parent),
        "cache_path": str(cache.resolve()),
        "method": "gradient",
        "density": grad.get("density", "dense"),
        "eval_metrics": {},
    }
    (cache.parent / "phase2_summary.json").write_text(json.dumps(summary, indent=2))
    result_json = os.environ.get("STEP_RESULT_JSON")
    if result_json:
        Path(result_json).parent.mkdir(parents=True, exist_ok=True)
        Path(result_json).write_text(json.dumps(summary, indent=2))
    print(f"grad_step cache: {cache}")


if __name__ == "__main__":
    main()
