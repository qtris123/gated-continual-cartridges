"""Phase 2: continual cartridge update by closed-form Attention Matching.

Loads a Phase 1 cartridge and applies one closed-form AM write per unique
document in the MT parquet, with the writable slots chosen by TF-IDF (or one of
the information-theoretic priors). No optimizer, no gradients.

Recipes-as-config: the *write rule* (slots / queries / keys / beta / objective /
rope_theta) is read from a resolved recipe YAML pointed at by ``$RECIPE_CONFIG``
-- the serialized ``AMContinualConfig`` component tree. There is no env<->recipe
bridge and no slot-knob registry: sweeping is a dotted-path override on that tree
(see ``sweep._set_dotted``). Only *runtime inputs* (which cartridge, which data,
which dataset/phase, where to write, which evals) come from the environment.

Usage:
    RECIPE_CONFIG=/path/to/recipe.yaml \\
    AM_DATASET=qasper AM_QASPER_TOPIC=MT \\
    PHASE1_CACHE_PATH=/path/to/cache_last.pt \\
    SYNTH_DATA_PATH=data/qasper/train/qwen_qasper_MT_task_8192.parquet \\
    python examples/shared/am/continual_write.py

Field-by-field config reference: research_loop/AM_CONFIG.md.
"""

import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")

import pydrantic
import yaml

from cartridges.am import (
    AMContinualConfig,
    BetaFitter,
    KeyWriter,
    ReferenceQueries,
    SlotSelector,
    TeacherTarget,
    ValueObjective,
)
from cartridges.cache import KVCacheFactory, TrainableCache
from cartridges.datasets import DataSource, LossEvalDataset
from cartridges.models import FlexLlamaForCausalLM, FlexQwen3ForCausalLM, HFModelConfig
from cartridges.train import LossEvalConfig
from cartridges.utils import get_logger
from examples.shared.paths import ROOT

logger = get_logger(__name__)


class KVFromLocal(KVCacheFactory):
    """Initialize a KV cache from a local Phase-1 checkpoint."""

    class Config(KVCacheFactory.Config):
        path: str

    def initialize_kv_cache(self, tokenizer=None, model=None, attn_config=None):
        return TrainableCache.from_pretrained(self.config.path, device="cuda")


def _flag(name: str, *, default: bool) -> bool:
    """Parse a 0/1-style runtime env flag. Unset -> ``default``; else raises."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    if raw in ("1", "true", "True"):
        return True
    if raw in ("0", "false", "False"):
        return False
    raise ValueError(f"{name} must be 0/1 (or true/false), got {raw!r}")


def _opt(name: str) -> str | None:
    return os.environ.get(name) or None


def _norm_path(value) -> str | None:
    """A recipe path field, normalized ROOT-relative (mirrors the old bridge)."""
    if not value:
        return None
    path = Path(str(value))
    return str(path if path.is_absolute() else ROOT / path)


# -- Recipe (the write rule) -------------------------------------------------
RECIPE_CONFIG = os.environ.get("RECIPE_CONFIG")
if not RECIPE_CONFIG:
    raise SystemExit(
        "continual_write requires $RECIPE_CONFIG pointing at a resolved recipe "
        "YAML (the serialized AMContinualConfig write rule)."
    )
_recipe = yaml.safe_load(Path(RECIPE_CONFIG).read_text()) or {}
_slots = _recipe.get("slots", {}) or {}
_queries = _recipe.get("queries", {}) or {}
_keys = _recipe.get("keys", {}) or {}
_beta = _recipe.get("beta", {}) or {}
_objective = _recipe.get("objective", {}) or {}

# -- Slot selection (from recipe) --------------------------------------------
TOP_T = int(_slots.get("top_t", 32))
GRANULARITY = str(_slots.get("granularity", "per_layer"))
SLOT_SELECTION = str(_slots.get("slot_selection", "tfidf"))
BG_STATS_PATH = _norm_path(_slots.get("background_indices_path"))
USE_IDF = bool(_slots.get("use_idf", True))
IDF_SMOOTHING = float(_slots.get("idf_smoothing", 1.0))
IDF_TOP_K = int(_slots.get("background_top_k_per_batch", 128))
IDF_PRIOR_WEIGHT = float(_slots.get("idf_prior_weight", 0.0))
MIN_TOP_T_PER_LAYER = int(_slots.get("min_top_t_per_layer", 1))
AM_REDUNDANCY_RIDGE_REL = float(_slots.get("redundancy_ridge_rel", 1e-6))
AM_MASS_REDUNDANCY_ALPHA = float(_slots.get("mass_redundancy_alpha", 0.5))
AM_SLOT_FISHER_PATH = _norm_path(_slots.get("slot_fisher_path"))
AM_SAFE_FRACTION = float(_slots.get("safe_fraction", 1.0))
AM_SAFE_METRIC = str(_slots.get("safe_metric", "redundancy"))
AM_USAGE_PENALTY_LAMBDA = float(_slots.get("usage_penalty_lambda", 0.0))
AM_USAGE_PENALTY_MODE = str(_slots.get("usage_penalty_mode", "mult"))
AM_USAGE_DECAY = float(_slots.get("usage_decay", 1.0))

# -- Reference queries (from recipe) -----------------------------------------
MAX_REF_EXAMPLES_PER_DOC = int(_queries.get("max_ref_examples_per_doc", 32))
QUERIES_PER_BATCH = str(_queries.get("queries_per_batch", "all_tokens"))
MAX_QUERIES_PER_HEAD = int(_queries.get("max_queries_per_head", 64))
AM_SEED_OFFSET = int(_queries.get("seed_offset", 0))
AM_ONPOLICY_LAYERS = int(_queries.get("onpolicy_layers", 0))
AM_ONPOLICY_DOCKV = bool(_queries.get("onpolicy_refresh_doc_kv", False))
AM_REF_BATCH_LIMIT = int(_queries.get("ref_batch_limit", 5))

# -- Keys (from recipe) ------------------------------------------------------
KEY_MODE = str(_keys.get("key_mode", "freeze"))
AM_KEY_REPOSITION = bool(_keys.get("key_reposition", False))

# -- Beta (from recipe) ------------------------------------------------------
_beta_enabled = _beta.get("enabled")
ENABLE_BETA = None if _beta_enabled is None else bool(_beta_enabled)
BETA_FIT_SCOPE = str(_beta.get("fit_scope", "selected"))
AM_BETA_BOX = float(_beta.get("beta_box", 3.0))
AM_NNLS_ITERS = int(_beta.get("nnls_iters", 2))
AM_NNLS_DRIVER = str(_beta.get("nnls_driver", "gelsd"))
AM_BETA_TARGET = str(_beta.get("target_mode", "residual"))

# -- Value solve (from recipe) -----------------------------------------------
RIDGE_LAMBDA = float(_objective.get("ridge_lambda", 1e-4))
RIDGE_SCALE = str(_objective.get("ridge_scale", "spectral"))
RIDGE_LAMBDA_MIN = float(_objective.get("ridge_lambda_min", 0.0))
DELTA_WEIGHT = float(_objective.get("delta_weight", 1e-2))
ENABLE_OLD_REFERENCE_GUARD = bool(_objective.get("enable_old_reference_guard", False))
OLD_REF_DATA_PATH = _norm_path(_objective.get("old_ref_data_path"))
OLD_REF_MAX_EXAMPLES = int(_objective.get("old_ref_max_examples", 64))
OLD_REFERENCE_WEIGHT = float(_objective.get("old_reference_weight", 1.0))
AM_ORACLE_WRITE = bool(_objective.get("oracle_write", False))
AM_ORACLE_WRITE_ASSIGN = str(_objective.get("oracle_write_assign", "mass_ranked"))

# -- rope_theta / stats (from recipe) ----------------------------------------
_ROPE_THETA_RAW = _recipe.get("rope_theta")
AM_COMPUTE_STATS = bool(_recipe.get("compute_update_stats", True))

# -- Required runtime inputs (per-invocation, NOT part of the recipe) ---------
PHASE1_CACHE_PATH = os.environ["PHASE1_CACHE_PATH"]
SYNTH_DATA_PATH = os.environ["SYNTH_DATA_PATH"]
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen3-4B-Instruct-2507")

# -- Teacher routing (runtime: which dataset/phase this stage writes) ---------
AM_DATASET = os.environ["AM_DATASET"].strip().lower()
AM_QASPER_TOPIC = os.environ.get("AM_QASPER_TOPIC", "MT")
AM_QUALITY_PHASE_ENV = _opt("AM_QUALITY_PHASE")
AM_QUALITY_PHASE = int(AM_QUALITY_PHASE_ENV) if AM_QUALITY_PHASE_ENV else None
AM_PHASE_ENV = _opt("AM_PHASE")
AM_PHASE = int(AM_PHASE_ENV) if AM_PHASE_ENV else None

# -- Bookkeeping / eval (runtime) --------------------------------------------
SAVE_AFTER_EACH_DOCUMENT = _flag("SAVE_AFTER_EACH_DOCUMENT", default=True)
EVAL_DATA_PATH = _opt("EVAL_DATA_PATH")
EVAL_QA_PATH = _opt("EVAL_QA_PATH")
EVAL_MT_PATH = _opt("EVAL_MT_PATH")
EVAL_SA_PATH = _opt("EVAL_SA_PATH")
EVAL_PHASE_PATHS = [_opt(f"EVAL_P{i}_PATH") for i in range(1, 6)]
RUN_NAME = os.environ.get(
    "RUN_NAME",
    f"{AM_DATASET}_continual_am_top-{TOP_T}_{GRANULARITY}_{KEY_MODE}",
)


def _fs_slug(s: str) -> str:
    """Filesystem-safe launch folder tag (pydrantic uses script_id in launch_id)."""
    slug = "".join(c if (c.isalnum() or c in "-_.") else "-" for c in s.strip())
    return slug.strip("-._") or "continual_am_sparse"


def _resolve_rope_theta() -> float:
    """recipe rope_theta -> float. Accepts a number or "model"/"auto"/"config"."""
    if _ROPE_THETA_RAW is None:
        return AMContinualConfig.model_fields["rope_theta"].default
    if isinstance(_ROPE_THETA_RAW, str) and _ROPE_THETA_RAW.strip().lower() in (
        "model", "auto", "config",
    ):
        from transformers import AutoConfig

        theta = float(AutoConfig.from_pretrained(MODEL_NAME).rope_theta)
    else:
        theta = float(_ROPE_THETA_RAW)
    if not (theta > 0):
        raise ValueError(f"rope_theta must be positive, got {theta}")
    logger.info("B-ROPE: AM teacher rope_theta = %g", theta)
    return theta


AM_ROPE_THETA = _resolve_rope_theta()


def _validate() -> None:
    """Reject combinations that would run but not mean what they say."""
    if AM_KEY_REPOSITION and KEY_MODE == "freeze":
        raise ValueError(
            "keys.key_reposition=true is meaningless with keys.key_mode=freeze "
            "(no key is ever installed). Set key_mode=highest_attention or omp."
        )
    if AM_KEY_REPOSITION and AM_ROPE_THETA == 10000.0:
        # The counter-rotation must use the model's own rotary base; doing it at
        # the AM package's historical 10000.0 while the model runs 5e6 would
        # replace one frame error with another (MECH-003 / B-ROPE).
        raise ValueError(
            "keys.key_reposition=true requires rope_theta to be the model's own "
            "base (use 5000000 / 'model' for Qwen3-4B-Instruct-2507); the counter-"
            "rotation is only correct in the model's own rotary frame."
        )
    if AM_ONPOLICY_LAYERS < 0:
        raise ValueError(f"queries.onpolicy_layers must be >= 0, got {AM_ONPOLICY_LAYERS}")
    if AM_ONPOLICY_DOCKV and AM_ONPOLICY_LAYERS <= 0:
        raise ValueError(
            "queries.onpolicy_refresh_doc_kv=true is meaningless without "
            "onpolicy_layers>0 (nothing is ever re-extracted)."
        )
    if AM_SEED_OFFSET < 0:
        raise ValueError(f"queries.seed_offset must be >= 0, got {AM_SEED_OFFSET}")
    if AM_DATASET == "qasper":
        from cartridges.data.qasper.resources import TOPIC_TO_IDS

        if AM_QASPER_TOPIC not in TOPIC_TO_IDS and AM_QASPER_TOPIC != "all":
            raise ValueError(
                f"AM_QASPER_TOPIC={AM_QASPER_TOPIC!r} is not a QASPER topic; "
                f"expected one of {sorted(TOPIC_TO_IDS)} or 'all'."
            )
    elif AM_DATASET == "quality":
        if AM_QUALITY_PHASE not in range(1, 6):
            raise ValueError("AM_QUALITY_PHASE must be 1..5 when AM_DATASET=quality")
    elif AM_DATASET in {"finqa", "techqa", "longhealth"}:
        if AM_PHASE not in range(1, 6):
            raise ValueError(f"AM_PHASE must be 1..5 when AM_DATASET={AM_DATASET}")
    else:
        raise ValueError(
            f"AM_DATASET={AM_DATASET!r} is unsupported; expected "
            "qasper, quality, finqa, techqa, or longhealth"
        )
    if ENABLE_OLD_REFERENCE_GUARD and not OLD_REF_DATA_PATH:
        raise ValueError(
            "objective.enable_old_reference_guard=true requires old_ref_data_path"
        )
    needs_fisher = SLOT_SELECTION == "fisher" or (
        SLOT_SELECTION == "constrained_mass" and AM_SAFE_METRIC == "fisher"
    )
    if needs_fisher and not AM_SLOT_FISHER_PATH:
        raise ValueError(
            f"slot_selection={SLOT_SELECTION} (safe_metric={AM_SAFE_METRIC}) needs "
            "slots.slot_fisher_path pointing at a cached (n_layers, n_slots) "
            "diagonal-Fisher array."
        )
    if AM_SLOT_FISHER_PATH and not os.path.exists(AM_SLOT_FISHER_PATH):
        raise FileNotFoundError(
            f"slots.slot_fisher_path={AM_SLOT_FISHER_PATH!r} does not exist."
        )


def _build_loss_evals() -> list[LossEvalConfig]:
    """Build up to five phase evals, preserving legacy QASPER aliases."""
    if any(EVAL_PHASE_PATHS):
        pairs = [
            (path, os.environ.get(f"EVAL_P{i}_NAME", f"p{i}"))
            for i, path in enumerate(EVAL_PHASE_PATHS, start=1)
        ]
        return [
            LossEvalConfig(
                dataset=LossEvalDataset.Config(
                    data_source=DataSource(path=path, type="local"),
                    packed_seq_length=2048,
                ),
                name_for_wandb=name,
            )
            for path, name in pairs
            if path
        ]

    qa_path, mt_path, sa_path = EVAL_QA_PATH, EVAL_MT_PATH, EVAL_SA_PATH
    if EVAL_DATA_PATH and not qa_path and not mt_path:
        mt_path = EVAL_DATA_PATH
    return [
        LossEvalConfig(
            dataset=LossEvalDataset.Config(
                data_source=DataSource(path=path, type="local"),
                packed_seq_length=2048,
            ),
            name_for_wandb=name,
        )
        for path, name in (
            (qa_path, "qa_forgetting"),
            (mt_path, "mt_acquisition"),
            (sa_path, "sa_acquisition"),
        )
        if path
    ]


_validate()

config = AMContinualConfig(
    name=RUN_NAME,
    script_id=_fs_slug(RUN_NAME),
    output_dir=os.environ.get("CARTRIDGES_OUTPUT_DIR", "."),
    model=HFModelConfig(
        pretrained_model_name_or_path=MODEL_NAME,
        model_cls=(
            FlexQwen3ForCausalLM if "qwen" in MODEL_NAME.lower() else FlexLlamaForCausalLM
        ),
    ),
    kv_cache_initializer=KVFromLocal.Config(path=PHASE1_CACHE_PATH),
    document_data_path=SYNTH_DATA_PATH,
    rope_theta=AM_ROPE_THETA,
    slots=SlotSelector.Config(
        top_t=TOP_T,
        granularity=GRANULARITY,
        slot_selection=SLOT_SELECTION,
        idf_smoothing=IDF_SMOOTHING,
        background_top_k_per_batch=IDF_TOP_K,
        idf_prior_weight=IDF_PRIOR_WEIGHT,
        min_top_t_per_layer=MIN_TOP_T_PER_LAYER,
        redundancy_ridge_rel=AM_REDUNDANCY_RIDGE_REL,
        mass_redundancy_alpha=AM_MASS_REDUNDANCY_ALPHA,
        safe_fraction=AM_SAFE_FRACTION,
        safe_metric=AM_SAFE_METRIC,
        usage_penalty_lambda=AM_USAGE_PENALTY_LAMBDA,
        usage_penalty_mode=AM_USAGE_PENALTY_MODE,
        usage_decay=AM_USAGE_DECAY,
        use_idf=USE_IDF and BG_STATS_PATH is not None,
        background_indices_path=BG_STATS_PATH,
        num_background_batches=999999999,
        slot_fisher_path=AM_SLOT_FISHER_PATH,
    ),
    queries=ReferenceQueries.Config(
        max_ref_examples_per_doc=MAX_REF_EXAMPLES_PER_DOC,
        queries_per_batch=QUERIES_PER_BATCH,
        max_queries_per_head=MAX_QUERIES_PER_HEAD,
        seed_offset=AM_SEED_OFFSET,
        onpolicy_layers=AM_ONPOLICY_LAYERS,
        onpolicy_refresh_doc_kv=AM_ONPOLICY_DOCKV,
        ref_batch_limit=AM_REF_BATCH_LIMIT,
    ),
    teacher=TeacherTarget.Config(
        dataset=AM_DATASET,
        qasper_topic=AM_QASPER_TOPIC,
        quality_phase=AM_QUALITY_PHASE,
        phase=AM_PHASE,
    ),
    keys=KeyWriter.Config(
        key_mode=KEY_MODE,
        key_reposition=AM_KEY_REPOSITION,
    ),
    beta=BetaFitter.Config(
        enabled=ENABLE_BETA,
        fit_scope=BETA_FIT_SCOPE,
        beta_box=AM_BETA_BOX,
        nnls_iters=AM_NNLS_ITERS,
        nnls_driver=AM_NNLS_DRIVER,
        target_mode=AM_BETA_TARGET,
    ),
    objective=ValueObjective.Config(
        ridge_lambda=RIDGE_LAMBDA,
        ridge_scale=RIDGE_SCALE,
        ridge_lambda_min=RIDGE_LAMBDA_MIN,
        delta_weight=DELTA_WEIGHT,
        enable_old_reference_guard=ENABLE_OLD_REFERENCE_GUARD,
        old_ref_data_path=OLD_REF_DATA_PATH,
        old_ref_max_examples=OLD_REF_MAX_EXAMPLES,
        old_reference_weight=(
            OLD_REFERENCE_WEIGHT if ENABLE_OLD_REFERENCE_GUARD else 0.0
        ),
        oracle_write=AM_ORACLE_WRITE,
        oracle_write_assign=AM_ORACLE_WRITE_ASSIGN,
    ),
    loss_evals=_build_loss_evals(),
    save_after_each_document=SAVE_AFTER_EACH_DOCUMENT,
    compute_update_stats=AM_COMPUTE_STATS,
)

if __name__ == "__main__":
    pydrantic.main(config)
