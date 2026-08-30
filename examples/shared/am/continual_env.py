"""Shared environment mapping for continual-AM stage runs.

`continual_write.py` is configured entirely through environment variables. Two
drivers need to build that environment — the QASPER ASR/KG resume and the
general p01->p05 chain — and a replication across datasets is only meaningful if
both build it the same way. So the recipe -> env mapping lives here once, and the
only dataset-specific parts are isolated in `DATASETS`.

A stage's *recipe* is read from a saved `config.yaml` rather than transcribed,
so a lineage provably uses the same write rule as the run that produced it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from examples.shared.paths import ROOT

MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"


@dataclass(frozen=True)
class DatasetSpec:
    """Where one dataset's five phases live, and how its teacher is selected."""

    name: str
    task_names: dict[int, str]
    synth_template: str
    # QASPER resolves documents by topic name; QuALITY by phase number.
    topics: dict[int, str] | None = None

    def eval_path(self, phase: int) -> Path:
        return ROOT / f"data/{self.name}/phases/phase{phase}_eval.parquet"

    def synth_path(self, phase: int) -> Path:
        key = self.topics[phase] if self.topics else phase
        return ROOT / self.synth_template.format(key=key)

    def teacher_env(self, phase: int) -> dict[str, str]:
        env = {"AM_DATASET": self.name}
        if self.topics:
            env["AM_QASPER_TOPIC"] = self.topics[phase]
        else:
            env["AM_QUALITY_PHASE"] = str(phase)
        return env


DATASETS = {
    "qasper": DatasetSpec(
        name="qasper",
        task_names={1: "qa", 2: "mt", 3: "sa", 4: "asr", 5: "kg"},
        synth_template="data/qasper/train/qwen_qasper_{key}_task_8192.parquet",
        topics={1: "QA", 2: "MT", 3: "SA", 4: "ASR", 5: "KG"},
    ),
    "quality": DatasetSpec(
        name="quality",
        task_names={phase: f"p{phase}" for phase in range(1, 6)},
        synth_template="data/quality/train/qwen_quality_p{key}_task_8192.parquet",
    ),
}


def spec(dataset: str) -> DatasetSpec:
    if dataset not in DATASETS:
        raise ValueError(
            f"Unsupported dataset {dataset!r}; expected one of {sorted(DATASETS)}"
        )
    return DATASETS[dataset]


def base_env(gpu: str, runs_dir: Path) -> dict[str, str]:
    """A clean child environment. Inherited EVAL_* vars are dropped so a stale
    shell export cannot silently redirect an evaluation."""
    env = os.environ.copy()
    for name in list(env):
        if name.startswith("EVAL_P") or name in {
            "EVAL_DATA_PATH",
            "EVAL_QA_PATH",
            "EVAL_MT_PATH",
            "EVAL_SA_PATH",
        }:
            env.pop(name)
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": gpu,
            "CARTRIDGES_DIR": str(ROOT),
            "CARTRIDGES_OUTPUT_DIR": str(runs_dir),
            "PYTHONPATH": str(ROOT),
            "HF_HUB_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "WANDB_DISABLED": "1",
            "MODEL_NAME": MODEL_NAME,
        }
    )
    return env


def phase_eval_env(
    env: dict[str, str], dataset: str, phases=range(1, 6)
) -> dict[str, str]:
    """Point the run at one held-out eval set per phase."""
    dataset_spec = spec(dataset)
    env = env.copy()
    for phase in phases:
        env[f"EVAL_P{phase}_PATH"] = str(dataset_spec.eval_path(phase))
        env[f"EVAL_P{phase}_NAME"] = dataset_spec.task_names[phase]
    return env


def _truth(value) -> str:
    return "1" if bool(value) else "0"


def _path(value) -> str:
    if not value:
        return ""
    path = Path(value)
    return str(path if path.is_absolute() else ROOT / path)


def recipe_env(cfg: dict) -> dict[str, str]:
    """Map a saved AMContinualConfig to the env `continual_write.py` reads."""
    slots, queries = cfg["slots"], cfg["queries"]
    keys, beta, objective = cfg["keys"], cfg["beta"], cfg["objective"]
    return {
        "TOP_T": str(slots["top_t"]),
        "GRANULARITY": str(slots["granularity"]),
        "SLOT_SELECTION": str(slots["slot_selection"]),
        "USE_IDF": _truth(slots["use_idf"]),
        "BG_STATS_PATH": _path(slots.get("background_indices_path")),
        "IDF_TOP_K": str(slots["background_top_k_per_batch"]),
        "IDF_SMOOTHING": str(slots["idf_smoothing"]),
        "IDF_PRIOR_WEIGHT": str(slots.get("idf_prior_weight", 0.0)),
        "MIN_TOP_T_PER_LAYER": str(slots.get("min_top_t_per_layer", 1)),
        "AM_REDUNDANCY_RIDGE_REL": str(slots.get("redundancy_ridge_rel", 1e-6)),
        "AM_MASS_REDUNDANCY_ALPHA": str(slots.get("mass_redundancy_alpha", 0.5)),
        "MAX_REF_EXAMPLES_PER_DOC": str(queries["max_ref_examples_per_doc"]),
        "QUERIES_PER_BATCH": str(queries["queries_per_batch"]),
        "MAX_QUERIES_PER_HEAD": str(queries["max_queries_per_head"]),
        "AM_SEED_OFFSET": str(queries.get("seed_offset", 0)),
        "AM_ONPOLICY_LAYERS": str(queries.get("onpolicy_layers", 0)),
        "AM_ONPOLICY_DOCKV": _truth(queries.get("onpolicy_refresh_doc_kv", False)),
        "AM_REF_BATCH_LIMIT": str(queries.get("ref_batch_limit", 5)),
        "KEY_MODE": str(keys["key_mode"]),
        "AM_KEY_REPOSITION": _truth(keys.get("key_reposition", False)),
        "ENABLE_BETA": _truth(beta["enabled"]),
        "BETA_FIT_SCOPE": str(beta.get("fit_scope", "selected")),
        "AM_BETA_BOX": str(beta.get("beta_box", 3.0)),
        "AM_NNLS_ITERS": str(beta.get("nnls_iters", 2)),
        "AM_NNLS_DRIVER": str(beta.get("nnls_driver", "gelsd")),
        "AM_BETA_TARGET": str(beta.get("target_mode", "residual")),
        "RIDGE_LAMBDA": str(objective["ridge_lambda"]),
        "RIDGE_SCALE": str(objective.get("ridge_scale", "spectral")),
        "RIDGE_LAMBDA_MIN": str(objective.get("ridge_lambda_min", 0.0)),
        "DELTA_WEIGHT": str(objective["delta_weight"]),
        "ENABLE_OLD_REFERENCE_GUARD": _truth(
            objective.get("enable_old_reference_guard", False)
        ),
        "OLD_REF_DATA_PATH": _path(objective.get("old_ref_data_path")),
        "OLD_REF_MAX_EXAMPLES": str(objective.get("old_ref_max_examples", 64)),
        "OLD_REFERENCE_WEIGHT": str(objective.get("old_reference_weight", 0.0)),
        "AM_ROPE_THETA": str(cfg["rope_theta"]),
        "AM_COMPUTE_STATS": _truth(cfg.get("compute_update_stats", True)),
    }


def stage_write_env(
    cfg: dict,
    *,
    dataset: str,
    phase: int,
    input_cache: str,
    run_name: str,
    gpu: str,
    runs_dir: Path,
) -> dict[str, str]:
    """The complete environment for one continual-AM stage write."""
    env = phase_eval_env(base_env(gpu, runs_dir), dataset)
    env.update(recipe_env(cfg))
    env.update(spec(dataset).teacher_env(phase))
    env.update(
        {
            "PHASE1_CACHE_PATH": str(input_cache),
            "SYNTH_DATA_PATH": str(spec(dataset).synth_path(phase)),
            "RUN_NAME": run_name,
            "SAVE_AFTER_EACH_DOCUMENT": "1",
        }
    )
    return env
