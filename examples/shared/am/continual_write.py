"""Phase 2: continual cartridge update by closed-form Attention Matching.

Loads a Phase 1 cartridge and applies one closed-form AM write per unique
document in the MT parquet, with the writable slots chosen by TF-IDF (or one of
the information-theoretic priors). No optimizer, no gradients.

Usage:
    AM_DATASET=qasper AM_QASPER_TOPIC=MT \
    PHASE1_CACHE_PATH=/path/to/cache_last.pt \\
    BG_STATS_PATH=/path/to/bg_stats.pt \\
    SYNTH_DATA_PATH=data/qasper/train/qwen_qasper_MT_task_8192.parquet \\
    python examples/shared/am/continual_write.py

Field-by-field config reference: research_loop/AM_CONFIG.md.
"""

import os

import pydrantic

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

logger = get_logger(__name__)


class KVFromLocal(KVCacheFactory):
    """Initialize a KV cache from a local Phase-1 checkpoint."""

    class Config(KVCacheFactory.Config):
        path: str

    def initialize_kv_cache(self, tokenizer=None, model=None, attn_config=None):
        return TrainableCache.from_pretrained(self.config.path, device="cuda")


def _flag(name: str, *, default: bool) -> bool:
    """Parse a 0/1-style env flag. Unset -> ``default``; anything else raises.

    The old driver spelled some of these `not in ("0", "false", "False")` and
    others `in ("1", "true", "True")`, so a typo silently picked a side. It now
    fails instead.
    """
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


# -- Required inputs ---------------------------------------------------------
PHASE1_CACHE_PATH = os.environ["PHASE1_CACHE_PATH"]
SYNTH_DATA_PATH = os.environ["SYNTH_DATA_PATH"]
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen3-4B-Instruct-2507")

# -- Slot selection ----------------------------------------------------------
TOP_T = int(os.environ.get("TOP_T", "64"))
GRANULARITY = os.environ.get("GRANULARITY", "per_layer")
SLOT_SELECTION = os.environ.get("SLOT_SELECTION", "tfidf")
BG_STATS_PATH = _opt("BG_STATS_PATH")
USE_IDF = _flag("USE_IDF", default=True)
IDF_TOP_K = int(os.environ.get("IDF_TOP_K", "128"))
IDF_SMOOTHING = float(os.environ.get("IDF_SMOOTHING", "1.0"))
IDF_PRIOR_WEIGHT = float(os.environ.get("IDF_PRIOR_WEIGHT", "0.0"))
MIN_TOP_T_PER_LAYER = int(os.environ.get("MIN_TOP_T_PER_LAYER", "1"))
# MECH-008 / MECH-009 slot priors. Inert unless SLOT_SELECTION names a prior mode.
AM_REDUNDANCY_RIDGE_REL = float(os.environ.get("AM_REDUNDANCY_RIDGE_REL") or "1e-6")
AM_MASS_REDUNDANCY_ALPHA = float(os.environ.get("AM_MASS_REDUNDANCY_ALPHA") or "0.5")
AM_SLOT_FISHER_PATH = _opt("AM_SLOT_FISHER_PATH")
AM_SAFE_FRACTION = float(os.environ.get("AM_SAFE_FRACTION") or "1.0")
AM_SAFE_METRIC = os.environ.get("AM_SAFE_METRIC") or "redundancy"

# -- Reference queries -------------------------------------------------------
MAX_REF_EXAMPLES_PER_DOC = int(os.environ.get("MAX_REF_EXAMPLES_PER_DOC", "32"))
QUERIES_PER_BATCH = os.environ.get("QUERIES_PER_BATCH", "all_tokens")
# B-CASCADE: reference queries per KV head fed to the solve. `n > top_t` makes
# the system over-determined.
MAX_QUERIES_PER_HEAD = int(os.environ.get("MAX_QUERIES_PER_HEAD", "64"))
# MECH-007: shifts the per-document draw to `doc_idx + N`. This is the seed knob
# this method actually has -- `config.seed` does not reach the per-document draw.
AM_SEED_OFFSET = int(os.environ.get("AM_SEED_OFFSET") or "0")
# LIT-006 / MECH-006: re-extract the reference queries from the UPDATED cache
# every N layers (N=1 -> per layer). 0 keeps the historical single-pass write.
AM_ONPOLICY_LAYERS = int(os.environ.get("AM_ONPOLICY_LAYERS") or "0")
# Separately-testable second axis: also re-prefill the DOCUMENT KV against the
# updated cartridge. The AM paper only re-extracts queries, so this defaults OFF.
AM_ONPOLICY_DOCKV = _flag("AM_ONPOLICY_DOCKV", default=False)
AM_REF_BATCH_LIMIT = int(os.environ.get("AM_REF_BATCH_LIMIT") or "5")

# -- Teacher -----------------------------------------------------------------
# B-ROPE / MECH-003. Accepts a float, or "model"/"auto" to read `rope_theta` off
# the HF model config. Unset keeps the AM package's historical 10000.0.
AM_ROPE_THETA_ENV = _opt("AM_ROPE_THETA")
AM_DATASET = os.environ["AM_DATASET"].strip().lower()
# Which QASPER topic the document titles resolve against ("QA"/"MT"/"SA"/"all").
# The teacher prefills the full paper, so this must match SYNTH_DATA_PATH.
AM_QASPER_TOPIC = os.environ.get("AM_QASPER_TOPIC", "MT")
AM_QUALITY_PHASE_ENV = _opt("AM_QUALITY_PHASE")
AM_QUALITY_PHASE = int(AM_QUALITY_PHASE_ENV) if AM_QUALITY_PHASE_ENV else None

# -- Keys --------------------------------------------------------------------
KEY_MODE = os.environ.get("KEY_MODE", "freeze")
# B-ROPE hazard H2 / MECH-005: counter-rotate a document key back into the
# cartridge frame when it is installed into a cartridge slot.
AM_KEY_REPOSITION_ENV = _opt("AM_KEY_REPOSITION")
AM_KEY_REPOSITION = _flag("AM_KEY_REPOSITION", default=False)

# -- Beta --------------------------------------------------------------------
ENABLE_BETA_ENV = os.environ.get("ENABLE_BETA")
ENABLE_BETA = None if ENABLE_BETA_ENV is None else ENABLE_BETA_ENV in ("1", "true", "True")
BETA_FIT_SCOPE = os.environ.get("BETA_FIT_SCOPE", "selected")
# B-SOLVE / LIT-002 boxed NNLS. The paper's values, used whenever beta is on.
AM_BETA_BOX = float(os.environ.get("AM_BETA_BOX") or "3.0")
AM_NNLS_ITERS = int(os.environ.get("AM_NNLS_ITERS") or "2")
AM_NNLS_DRIVER = os.environ.get("AM_NNLS_DRIVER") or "gelsd"
AM_BETA_TARGET = os.environ.get("AM_BETA_TARGET") or "residual"

# -- Value solve -------------------------------------------------------------
RIDGE_LAMBDA = float(os.environ.get("RIDGE_LAMBDA", "1e-4"))
RIDGE_SCALE = os.environ.get("RIDGE_SCALE", "spectral")
RIDGE_LAMBDA_MIN = float(os.environ.get("RIDGE_LAMBDA_MIN", "0.0"))
DELTA_WEIGHT = float(os.environ.get("DELTA_WEIGHT", "1e-2"))
ENABLE_OLD_REFERENCE_GUARD = _flag("ENABLE_OLD_REFERENCE_GUARD", default=False)
OLD_REF_DATA_PATH = _opt("OLD_REF_DATA_PATH")
OLD_REF_MAX_EXAMPLES = int(os.environ.get("OLD_REF_MAX_EXAMPLES", "64"))
OLD_REFERENCE_WEIGHT = float(os.environ.get("OLD_REFERENCE_WEIGHT", "1.0"))
# B-ROUTE write-ceiling oracle (MECH-001), a diagnostic and not a write rule.
AM_ORACLE_WRITE = _flag("AM_ORACLE_WRITE", default=False)
AM_ORACLE_WRITE_ASSIGN = os.environ.get("AM_ORACLE_WRITE_ASSIGN", "mass_ranked")

# -- Bookkeeping / eval ------------------------------------------------------
SAVE_AFTER_EACH_DOCUMENT = _flag("SAVE_AFTER_EACH_DOCUMENT", default=True)
AM_COMPUTE_STATS = _flag("AM_COMPUTE_STATS", default=True)
EVAL_DATA_PATH = _opt("EVAL_DATA_PATH")
EVAL_QA_PATH = _opt("EVAL_QA_PATH")
EVAL_MT_PATH = _opt("EVAL_MT_PATH")
# A third stage (QA -> MT -> SA) needs all three held out at once: MT stops being
# the acquisition target and becomes a second retention probe.
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
    """AM_ROPE_THETA -> float. Unset keeps the AM package's historical default."""
    if AM_ROPE_THETA_ENV is None:
        return AMContinualConfig.model_fields["rope_theta"].default
    if AM_ROPE_THETA_ENV.strip().lower() in ("model", "auto", "config"):
        from transformers import AutoConfig

        theta = float(AutoConfig.from_pretrained(MODEL_NAME).rope_theta)
    else:
        theta = float(AM_ROPE_THETA_ENV)
    if not (theta > 0):
        raise ValueError(f"AM_ROPE_THETA must be positive, got {theta}")
    logger.info("B-ROPE: AM teacher rope_theta = %g (AM_ROPE_THETA=%s)", theta, AM_ROPE_THETA_ENV)
    return theta


def _validate() -> None:
    """Reject combinations that would run but not mean what they say."""
    if AM_KEY_REPOSITION and KEY_MODE == "freeze":
        raise ValueError(
            "AM_KEY_REPOSITION=1 is meaningless with KEY_MODE=freeze (no key is "
            "ever installed). Set KEY_MODE=highest_attention or omp."
        )
    if AM_KEY_REPOSITION and AM_ROPE_THETA_ENV is None:
        # The counter-rotation must use the model's own rotary base; doing it at
        # the AM package's historical 10000.0 while the model runs 5e6 would
        # replace one frame error with another (MECH-003 / B-ROPE).
        raise ValueError(
            "AM_KEY_REPOSITION=1 requires AM_ROPE_THETA to be set explicitly "
            "(use 5000000 / 'model' for Qwen3-4B-Instruct-2507); the counter-"
            "rotation is only correct in the model's own rotary frame."
        )
    if AM_ONPOLICY_LAYERS < 0:
        raise ValueError(f"AM_ONPOLICY_LAYERS must be >= 0, got {AM_ONPOLICY_LAYERS}")
    if AM_ONPOLICY_DOCKV and AM_ONPOLICY_LAYERS <= 0:
        raise ValueError(
            "AM_ONPOLICY_DOCKV=1 is meaningless without AM_ONPOLICY_LAYERS>0 "
            "(nothing is ever re-extracted)."
        )
    if AM_SEED_OFFSET < 0:
        raise ValueError(f"AM_SEED_OFFSET must be >= 0, got {AM_SEED_OFFSET}")
    if AM_DATASET == "qasper":
        # Local import: pulls in `datasets` and the HF cache.
        from cartridges.data.qasper.resources import TOPIC_TO_IDS

        if AM_QASPER_TOPIC not in TOPIC_TO_IDS and AM_QASPER_TOPIC != "all":
            raise ValueError(
                f"AM_QASPER_TOPIC={AM_QASPER_TOPIC!r} is not a QASPER topic; "
                f"expected one of {sorted(TOPIC_TO_IDS)} or 'all'."
            )
    elif AM_DATASET == "quality":
        if AM_QUALITY_PHASE not in range(1, 6):
            raise ValueError(
                "AM_QUALITY_PHASE must be 1..5 when AM_DATASET=quality"
            )
    else:
        raise ValueError(
            f"AM_DATASET={AM_DATASET!r} is unsupported; expected qasper or quality"
        )
    if ENABLE_OLD_REFERENCE_GUARD and not OLD_REF_DATA_PATH:
        raise ValueError(
            "ENABLE_OLD_REFERENCE_GUARD=1 requires OLD_REF_DATA_PATH (QA parquet)"
        )
    needs_fisher = SLOT_SELECTION == "fisher" or (
        SLOT_SELECTION == "constrained_mass" and AM_SAFE_METRIC == "fisher"
    )
    if needs_fisher and not AM_SLOT_FISHER_PATH:
        raise ValueError(
            f"SLOT_SELECTION={SLOT_SELECTION} (AM_SAFE_METRIC={AM_SAFE_METRIC}) "
            "needs AM_SLOT_FISHER_PATH pointing at a cached (n_layers, n_slots) "
            "diagonal-Fisher array. Generate one with "
            "research_loop/results/MECH-INFOGATE/compute_slot_fisher.py -- it is a "
            "DIAGNOSTIC backward pass (no optimizer, gradient_steps stays 0)."
        )
    if AM_SLOT_FISHER_PATH and not os.path.exists(AM_SLOT_FISHER_PATH):
        raise FileNotFoundError(
            f"AM_SLOT_FISHER_PATH={AM_SLOT_FISHER_PATH!r} does not exist."
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
    # Backward compat: a single EVAL_DATA_PATH means MT when neither is set.
    if EVAL_DATA_PATH and not qa_path and not mt_path:
        mt_path = EVAL_DATA_PATH
    # `mt_acquisition` keeps its name in three-stage runs even though MT is then a
    # retention probe, so the metric stays comparable with the two-stage runs.
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
    # pydrantic names the run folder `{timestamp}-{script_id}/…`; without this,
    # script_id defaults to the file stem (`continual_am_sparse`) and RUN_NAME
    # only appears as config.name / phase2_summary.run_name.
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
    rope_theta=_resolve_rope_theta(),
    slots=SlotSelector.Config(
        top_t=TOP_T,
        granularity=GRANULARITY,
        slot_selection=SLOT_SELECTION,
        use_idf=USE_IDF and BG_STATS_PATH is not None,
        idf_smoothing=IDF_SMOOTHING,
        background_top_k_per_batch=IDF_TOP_K,
        background_indices_path=BG_STATS_PATH,
        num_background_batches=999999999,
        redundancy_ridge_rel=AM_REDUNDANCY_RIDGE_REL,
        mass_redundancy_alpha=AM_MASS_REDUNDANCY_ALPHA,
        slot_fisher_path=AM_SLOT_FISHER_PATH,
        safe_fraction=AM_SAFE_FRACTION,
        safe_metric=AM_SAFE_METRIC,
        idf_prior_weight=IDF_PRIOR_WEIGHT,
        min_top_t_per_layer=MIN_TOP_T_PER_LAYER,
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
