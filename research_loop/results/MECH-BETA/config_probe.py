"""MECH-BETA: prove the driver config CONSTRUCTS for all three arms (no GPU).

Runs `continual_am_sparse.py`'s module-level config construction under each arm's
env and dumps `attention_matching_finetuning` so the three configs can be diffed.
"""

from __future__ import annotations

import json
import os
import runpy
import sys

REPO = "/localhome/local-triv/gated-continual-cartridges_explore"
ARMS = {
    "control": {"AM_ROPE_THETA": "10000.0"},
    "rope": {"AM_ROPE_THETA": "5000000.0"},
    "beta": {"AM_ROPE_THETA": "5000000.0", "ENABLE_BETA": "1"},
    "stock": {},  # nothing set at all -> must match the historical config exactly
}

BASE = {
    "PHASE1_CACHE_PATH": f"{REPO}/outputs/phase1_selfdistill_qwen512/cache_last.pt",
    "SYNTH_DATA_PATH": f"{REPO}/data/qasper/train/qwen_qasper_MT_task_8192.parquet",
    "MODEL_NAME": "Qwen/Qwen3-4B-Instruct-2507",
    "TOP_T": "32",
    "GRANULARITY": "per_layer",
    "SLOT_SELECTION": "tfidf",
    "USE_IDF": "0",
    "KEY_MODE": "freeze",
    "RIDGE_LAMBDA": "1e-4",
    "RIDGE_SCALE": "spectral",
    "DELTA_WEIGHT": "1e-2",
    "MAX_QUERIES_PER_HEAD": "64",
    "WANDB_DISABLED": "1",
    "CARTRIDGES_DIR": REPO,
    "CARTRIDGES_OUTPUT_DIR": f"{REPO}/outputs",
}

if __name__ == "__main__":
    arm = sys.argv[1]
    for k in list(os.environ):
        if k.startswith(("AM_", "ENABLE_BETA", "TOP_T", "KEY_MODE")):
            os.environ.pop(k, None)
    os.environ.update(BASE)
    os.environ.update(ARMS[arm])
    mod = runpy.run_path(
        f"{REPO}/examples/qasper2/train/continual_am_sparse.py",
        run_name="__config_probe__",
    )
    cfg = mod["config"]
    out = cfg.attention_matching_finetuning.model_dump()
    print("ARM", arm)
    print(json.dumps(out, indent=2, default=str))
    print("TAGS", json.dumps(cfg.wandb.tags if cfg.wandb else None))
