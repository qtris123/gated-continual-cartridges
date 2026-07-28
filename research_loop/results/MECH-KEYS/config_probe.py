"""MECH-KEYS: does the driver still CONSTRUCT its config under every arm's env?

RUNBOOK 6.10: a new kwarg passed unconditionally through the sibling `cartridges`
package crashes every run at pydantic construction. Probe each arm's env in its own
interpreter, printing the resolved `cartridges` path and the resulting AM config
fields. Also probes the two fail-loud guards.
"""
import json, os, subprocess, sys

REPO = "/localhome/local-triv/gated-continual-cartridges_explore"
PY = f"{REPO}/.venv/bin/python"
BASE = dict(
    PHASE1_CACHE_PATH=f"{REPO}/outputs/phase1_selfdistill_qwen512/cache_last.pt",
    SYNTH_DATA_PATH=f"{REPO}/data/qasper/train/qwen_qasper_MT_task_8192.parquet",
    MODEL_NAME="Qwen/Qwen3-4B-Instruct-2507", NUM_TOKENS="512", TOP_T="32",
    GRANULARITY="per_layer", SLOT_SELECTION="tfidf", USE_IDF="0", ENABLE_BETA="0",
    RIDGE_LAMBDA="1e-4", RIDGE_SCALE="spectral", DELTA_WEIGHT="1e-2",
    MAX_QUERIES_PER_HEAD="64", WANDB_DISABLED="1",
    CARTRIDGES_OUTPUT_DIR=f"{REPO}/outputs",
)
SNIP = (
    "import os,cartridges;"
    "import importlib.util as iu;"
    "spec=iu.spec_from_file_location('drv', os.environ['DRV']);"
    "m=iu.module_from_spec(spec);spec.loader.exec_module(m);"
    "c=m.config.attention_matching_finetuning;"
    "print('JSON'+__import__('json').dumps({"
    "'import':os.path.dirname(cartridges.__file__),"
    "'key_mode':c.key_mode,'key_reposition':getattr(c,'key_reposition',None),"
    "'rope_theta':getattr(c,'rope_theta',None),'enable_beta':c.enable_beta,"
    "'freeze_keys':c.freeze_keys,'top_t':c.top_t,'use_idf':c.use_idf,"
    "'delta_weight':c.delta_weight,'max_queries_per_head':c.max_queries_per_head}))"
)

ARMS = {
  "control":        dict(KEY_MODE="freeze",            AM_ROPE_THETA="5000000"),
  "keys_norepos":   dict(KEY_MODE="highest_attention", AM_ROPE_THETA="5000000", AM_KEY_REPOSITION="0"),
  "keys_repos":     dict(KEY_MODE="highest_attention", AM_ROPE_THETA="5000000", AM_KEY_REPOSITION="1"),
  "omp_repos":      dict(KEY_MODE="omp",               AM_ROPE_THETA="5000000", AM_KEY_REPOSITION="1"),
  "GUARD_repos_freeze":   dict(KEY_MODE="freeze",            AM_ROPE_THETA="5000000", AM_KEY_REPOSITION="1"),
  "GUARD_repos_notheta":  dict(KEY_MODE="highest_attention", AM_KEY_REPOSITION="1"),
}

out = {}
for label, extra in ARMS.items():
    env = dict(os.environ); env.update(BASE); env.update(extra)
    env["PYTHONPATH"] = REPO
    env["CARTRIDGES_DIR"] = REPO
    env["DRV"] = f"{REPO}/examples/qasper2/train/continual_am_sparse.py"
    r = subprocess.run([PY, "-c", SNIP], env=env, capture_output=True, text=True, cwd="/tmp")
    line = [l for l in r.stdout.splitlines() if l.startswith("JSON")]
    out[label] = json.loads(line[0][4:]) if line else {"rc": r.returncode, "err": r.stderr.strip().splitlines()[-1] if r.stderr.strip() else ""}

# unpinned (sibling) probe: the conditional kwarg must be the reason nothing breaks
env = dict(os.environ); env.update(BASE)
env.update(dict(KEY_MODE="highest_attention", AM_ROPE_THETA="5000000", AM_KEY_REPOSITION="1"))
env.pop("PYTHONPATH", None); env["CARTRIDGES_DIR"] = REPO
env["DRV"] = f"{REPO}/examples/qasper2/train/continual_am_sparse.py"
r = subprocess.run([PY, "-c", SNIP], env=env, capture_output=True, text=True, cwd="/tmp")
out["UNPINNED_must_fail_loudly"] = {
    "rc": r.returncode,
    "tail": r.stderr.strip().splitlines()[-1] if r.stderr.strip() else r.stdout.strip()[-300:],
}
print(json.dumps(out, indent=2))
