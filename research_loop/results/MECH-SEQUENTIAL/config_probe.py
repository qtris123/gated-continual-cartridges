"""MECH-SEQUENTIAL: prove the driver config CONSTRUCTS for every arm (no GPU).

Adapted from `results/MECH-KEYS/config_probe.py`. Runs `continual_am_sparse.py`'s
module-level config construction under each arm's env (in its own interpreter) and
dumps `attention_matching_finetuning`, so the three Part-B arms can be diffed and
the fail-loud guards + the sibling-`cartridges` foot-gun (RUNBOOK 6.10) can be
exercised.

Usage:  python config_probe.py            # driver: spawns one child per arm
        python config_probe.py <arm>      # child
"""

from __future__ import annotations

import json
import os
import runpy
import subprocess
import sys

REPO = "/localhome/local-triv/gated-continual-cartridges_explore"

ARMS = {
    # the three Part-B arms
    "control": {"KEY_MODE": "freeze", "AM_ROPE_THETA": "5000000"},
    "onpolicy": {
        "KEY_MODE": "freeze", "AM_ROPE_THETA": "5000000",
        "AM_ONPOLICY_LAYERS": "4",
    },
    "onpolicy_keys": {
        "KEY_MODE": "highest_attention", "AM_ROPE_THETA": "5000000",
        "AM_KEY_REPOSITION": "1", "AM_ONPOLICY_LAYERS": "4",
    },
    # the second (untested) axis
    "onpolicy_dockv": {
        "KEY_MODE": "freeze", "AM_ROPE_THETA": "5000000",
        "AM_ONPOLICY_LAYERS": "4", "AM_ONPOLICY_DOCKV": "1",
    },
    # nothing set at all -> must match the historical config exactly
    "stock": {},
    # fail-loud guards
    "GUARD_dockv_without_layers": {
        "KEY_MODE": "freeze", "AM_ROPE_THETA": "5000000", "AM_ONPOLICY_DOCKV": "1",
    },
    "GUARD_negative_layers": {
        "KEY_MODE": "freeze", "AM_ROPE_THETA": "5000000",
        "AM_ONPOLICY_LAYERS": "-1",
    },
    "GUARD_wrong_exec_mode": {
        "KEY_MODE": "freeze", "AM_ROPE_THETA": "5000000",
        "AM_ONPOLICY_LAYERS": "4", "AM_EXECUTION_MODE": "legacy_decoupled",
    },
}

BASE = {
    "PHASE1_CACHE_PATH": f"{REPO}/outputs/phase1_selfdistill_qwen512/cache_last.pt",
    "SYNTH_DATA_PATH": f"{REPO}/data/qasper/train/qwen_qasper_MT_task_8192.parquet",
    "MODEL_NAME": "Qwen/Qwen3-4B-Instruct-2507",
    "NUM_TOKENS": "512",
    "TOP_T": "32",
    "GRANULARITY": "per_layer",
    "SLOT_SELECTION": "tfidf",
    "USE_IDF": "0",
    "ENABLE_BETA": "0",
    "RIDGE_LAMBDA": "1e-4",
    "RIDGE_SCALE": "spectral",
    "DELTA_WEIGHT": "1e-2",
    "MAX_QUERIES_PER_HEAD": "64",
    "WANDB_DISABLED": "1",
    "CARTRIDGES_DIR": REPO,
    "CARTRIDGES_OUTPUT_DIR": f"{REPO}/outputs",
}

FIELDS = (
    "onpolicy_layers", "onpolicy_refresh_doc_kv", "key_mode", "key_reposition",
    "rope_theta", "enable_beta", "freeze_keys", "top_t", "granularity", "use_idf",
    "slot_selection", "delta_weight", "ridge_lambda", "ridge_scale",
    "max_queries_per_head", "oracle_write", "execution_mode",
)


def child(arm: str) -> None:
    import cartridges  # noqa: F401  (resolved BEFORE the driver runs)

    for k in list(os.environ):
        if k.startswith(("AM_", "KEY_MODE", "ENABLE_BETA", "TOP_T")):
            os.environ.pop(k, None)
    os.environ.update(BASE)
    os.environ.update(ARMS[arm])
    if arm == "stock":
        for k in ("USE_IDF", "TOP_T", "ENABLE_BETA", "DELTA_WEIGHT",
                  "MAX_QUERIES_PER_HEAD", "SLOT_SELECTION"):
            os.environ.pop(k, None)
    mod = runpy.run_path(
        f"{REPO}/examples/qasper2/train/continual_am_sparse.py",
        run_name="__config_probe__",
    )
    cfg = mod["config"]
    dump = cfg.attention_matching_finetuning.model_dump()
    print("JSON" + json.dumps({
        "import": os.path.dirname(cartridges.__file__),
        "fields": {f: dump.get(f, "<missing>") for f in FIELDS},
        "n_fields": len(dump),
        "tags": cfg.wandb.tags if cfg.wandb else None,
    }, default=str))


def main() -> None:
    out = {}
    for arm in ARMS:
        env = dict(os.environ)
        env["PYTHONPATH"] = REPO
        env["CARTRIDGES_DIR"] = REPO
        r = subprocess.run(
            [sys.executable, os.path.abspath(__file__), arm],
            env=env, capture_output=True, text=True, cwd="/tmp",
        )
        line = [l for l in r.stdout.splitlines() if l.startswith("JSON")]
        out[arm] = (
            json.loads(line[0][4:]) if line
            else {"rc": r.returncode,
                  "err": (r.stderr.strip().splitlines() or [""])[-1]}
        )
    # RUNBOOK 6.10: unpinned, `import cartridges` resolves to the SIBLING repo,
    # which has no `onpolicy_layers` field -> the knob must fail LOUDLY, not no-op.
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env["CARTRIDGES_DIR"] = REPO
    r = subprocess.run(
        [sys.executable, os.path.abspath(__file__), "onpolicy"],
        env=env, capture_output=True, text=True, cwd="/tmp",
    )
    line = [l for l in r.stdout.splitlines() if l.startswith("JSON")]
    out["UNPINNED_must_fail_loudly"] = {
        "rc": r.returncode,
        "constructed": bool(line),
        "tail": (r.stderr.strip().splitlines() or [""])[-1],
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    if len(sys.argv) > 1:
        child(sys.argv[1])
    else:
        main()
