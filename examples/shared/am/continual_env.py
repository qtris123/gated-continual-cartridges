"""Shared environment for continual-AM stage runs (recipes-as-config).

`continual_write.py` reads its *write rule* from a resolved recipe YAML
(``$RECIPE_CONFIG``); this module only assembles the *runtime* environment a
stage needs -- clean CUDA/HF/model vars, the per-phase eval paths, the teacher
routing, and the input cache / synth data / recipe pointer. The dataset-specific
parts are isolated in `DATASETS`.

The recipe is passed by pointer (a file), not transcribed into per-knob env
vars, so a lineage provably uses the same write rule as the run that produced it
and there is no env<->recipe bridge to drift.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from examples.shared.paths import ROOT

MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"


@dataclass(frozen=True)
class DatasetSpec:
    """Where one dataset's five phases live, and how its teacher is selected.

    ``p01_method`` and ``scorer`` are per-dataset *facts* (Axis-A knowledge): how
    this dataset's stage-1 cartridge is built (backprop-free ``compaction`` vs
    gradient ``selfdistill``) and how its answers are scored (``logppl`` teacher-
    forced perplexity vs ``mcq`` generation accuracy). They let the shared
    endpoints (``build_p01``, ``sweep check``) dispatch without per-dataset code.
    """

    name: str
    task_names: dict[int, str]
    synth_template: str
    # QASPER resolves documents by topic name; QuALITY by phase number.
    topics: dict[int, str] | None = None
    # How stage-1 (p01) is built: "compaction" (arm-D, backprop-free) or
    # "selfdistill" (KVFromText + gradient). All current 5x5 grids use compaction.
    p01_method: str = "compaction"
    # How this dataset's answers are scored: "logppl" or "mcq".
    scorer: str = "logppl"

    def eval_path(self, phase: int) -> Path:
        return ROOT / f"data/{self.name}/phases/phase{phase}_eval.parquet"

    def synth_path(self, phase: int) -> Path:
        key = self.topics[phase] if self.topics else phase
        return ROOT / self.synth_template.format(key=key)

    def teacher_env(self, phase: int) -> dict[str, str]:
        env = {"AM_DATASET": self.name}
        if self.topics:
            env["AM_QASPER_TOPIC"] = self.topics[phase]
        elif self.name == "quality":
            env["AM_QUALITY_PHASE"] = str(phase)
        else:
            env["AM_PHASE"] = str(phase)
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
    "finqa": DatasetSpec(
        name="finqa",
        task_names={phase: f"p{phase}" for phase in range(1, 6)},
        synth_template="data/finqa/train/qwen_finqa_p{key}_task_8192.parquet",
    ),
    "techqa": DatasetSpec(
        name="techqa",
        task_names={phase: f"p{phase}" for phase in range(1, 6)},
        synth_template="data/techqa/train/qwen_techqa_p{key}_task_8192.parquet",
    ),
    # First-class 5-phase, phase-keyed dataset (patients 01-20, 4 per phase).
    # Infra lives in cartridges/data/longhealth + data/longhealth/phases; only
    # the synth train parquet + p01 kvcache remain to generate (a data task).
    "longhealth": DatasetSpec(
        name="longhealth",
        task_names={phase: f"p{phase}" for phase in range(1, 6)},
        synth_template="data/longhealth/train/qwen_longhealth_p{key}_task_8192.parquet",
        p01_method="compaction",
        scorer="mcq",
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


def stage_write_env(
    recipe_path: str,
    *,
    dataset: str,
    phase: int,
    input_cache: str,
    run_name: str,
    gpu: str,
    runs_dir: Path,
) -> dict[str, str]:
    """The complete runtime environment for one continual-AM stage write.

    The write rule is passed by pointer: ``RECIPE_CONFIG`` names the resolved
    recipe YAML that `continual_write.py` reads. Everything else here is a
    per-invocation runtime input, not a config knob.
    """
    env = phase_eval_env(base_env(gpu, runs_dir), dataset)
    env.update(spec(dataset).teacher_env(phase))
    env.update(
        {
            "RECIPE_CONFIG": str(recipe_path),
            "PHASE1_CACHE_PATH": str(input_cache),
            "SYNTH_DATA_PATH": str(spec(dataset).synth_path(phase)),
            "RUN_NAME": run_name,
            "SAVE_AFTER_EACH_DOCUMENT": "1",
        }
    )
    return env
