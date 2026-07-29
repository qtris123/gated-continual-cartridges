"""DIAG-KEYCURVE Part 2 pre-check (CPU, no GPU, no source edit).

RUNBOOK 9c says: "`continual_am_sparse.py` ... is a `pydrantic.main` script, so a CLI
override (`seed=<n>`) is the first thing to try".  This probe TESTS that claim instead of
relying on it: it sets sys.argv to carry `seed=7`, imports the driver module with the exact
canonical MECH-KEYS environment, and reads back `config.seed` plus the execution mode that
decides whether `pydrantic.main` is ever reached.

Nothing here loads a model or touches CUDA; module import only builds the config object.
"""
import json, os, sys

SNAP = "/tmp/amsnap_kcurve"
REPO = "/localhome/local-triv/gated-continual-cartridges_explore"

os.environ.setdefault("CARTRIDGES_DIR", SNAP)
os.environ.setdefault("CARTRIDGES_OUTPUT_DIR", f"{REPO}/outputs")
env = dict(
    PHASE1_CACHE_PATH=f"{REPO}/outputs/phase1_selfdistill_qwen512/cache_last.pt",
    SYNTH_DATA_PATH=f"{REPO}/data/qasper/train/qwen_qasper_MT_task_8192.parquet",
    MODEL_NAME="Qwen/Qwen3-4B-Instruct-2507", NUM_TOKENS="512",
    EVAL_QA_PATH=f"{REPO}/data/qasper/eval/qasper_eval_QA.parquet",
    EVAL_MT_PATH=f"{REPO}/data/qasper/eval/qasper_eval_MT.parquet",
    USE_IDF="0", GRANULARITY="per_layer", TOP_T="32", TARGET_MODE="cartridge_plus_doc",
    KEY_MODE="highest_attention", ENABLE_BETA="0", BETA_FIT_SCOPE="selected",
    RIDGE_LAMBDA="1e-4", RIDGE_SCALE="spectral", RIDGE_LAMBDA_MIN="0.0",
    DELTA_WEIGHT="1e-2", AM_EXECUTION_MODE="per_document", SAVE_AFTER_EACH_DOCUMENT="1",
    SLOT_SELECTION="tfidf", MAX_QUERIES_PER_HEAD="64",
    AM_ROPE_THETA="5000000", AM_KEY_REPOSITION="1", RUN_NAME="DIAG-KEYCURVE_seedprobe",
    WANDB_DISABLED="0", WANDB_GROUP="B-ROUTE",
)
os.environ.update(env)

sys.path.insert(0, SNAP)
sys.path.insert(0, os.path.join(SNAP, "examples/qasper2/train"))
# the CLI override RUNBOOK 9c proposes:
sys.argv = ["continual_am_sparse.py", "seed=7"]

import cartridges  # noqa: E402
import continual_am_sparse as drv  # noqa: E402

mode = drv.config.attention_matching_finetuning.execution_mode
out = {
    "cartridges_import_path": os.path.dirname(cartridges.__file__),
    "driver_import_path": drv.__file__,
    "sys_argv_used": sys.argv,
    "config_seed_after_cli_override": drv.config.seed,
    "config_seed_expected_if_override_worked": 7,
    "cli_override_reached_config": drv.config.seed == 7,
    "execution_mode": mode,
    "pydrantic_main_is_reached_for_this_mode": mode == "train_loop",
    "SEED_env_knob_exists": any(
        k in open(drv.__file__).read() for k in ('os.environ.get("SEED"', "os.environ['SEED']")
    ),
    "per_doc_reference_sampling_seed": (
        "cartridges/am/continual.py:150 -> limit_conversations(..., seed=doc_idx) and "
        "build_reference_dataloader(..., seed=doc_idx): the reference draw is seeded by the "
        "DOCUMENT INDEX, a constant, not by config.seed"
    ),
}
print(json.dumps(out, indent=1))
json.dump(out, open(f"{REPO}/research_loop/results/DIAG-KEYCURVE/seed_probe.json", "w"), indent=1)
