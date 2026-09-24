"""Phase 1: classic Attention Matching compaction (backprop-free).

Builds one large teacher KV cache from a QASPER, QuALITY, FinQA, TechQA, or
LongHealth phase, then compacts it to a NUM_TOKENS-slot cartridge by selecting
teacher keys and ridge-fitting values to reproduce teacher attention outputs on
reference queries.

Keys are teacher-derived rather than init-text, and there are no Phase-2
stabilizers (one-shot compaction). KVFromText + self-match refine is deprecated.

Usage:
    AM_DATASET=qasper AM_QASPER_TOPIC=QA \
    QA_DATA_PATH=data/qasper/train/qwen_qasper_QA_task_8192.parquet \\
    EVAL_DATA_PATH=data/qasper/eval/qasper_eval_QA.parquet \\
    NUM_TOKENS=512 KEY_SELECT=highest_attention ENABLE_BETA=1 GRANULARITY=per_head \\
    python examples/shared/am/initial_compaction.py
"""

import json
import os
import time
import uuid
from datetime import datetime
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")

import torch

torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from cartridges.am.initial import compact_cache_am_phase1
from cartridges.cache import AttnConfig
from cartridges.datasets import DataSource, TrainDataset, LossEvalDataset
from cartridges.models import FlexLlamaForCausalLM, FlexQwen3ForCausalLM, HFModelConfig
from cartridges.sparse_cache_finetuning import (
    SparseCacheFinetuningConfig,
    collect_background_stats,
)
from cartridges.train import (
    TrainConfig,
    LossEvalConfig,
    CacheAndModel,
    evaluate_perplexity,
    save_cache,
)
from cartridges.utils import seed_everything, get_logger
from cartridges.utils.wandb import WandBConfig, prepare_wandb

logger = get_logger(__name__)

QA_DATA_PATH = os.environ.get(
    "QA_DATA_PATH",
    os.environ.get("SYNTH_DATA_PATH"),
)
EVAL_DATA_PATH = os.environ.get("EVAL_DATA_PATH", None)
AM_DATASET = os.environ["AM_DATASET"].strip().lower()
AM_QASPER_TOPIC = os.environ.get("AM_QASPER_TOPIC", "QA")
AM_QUALITY_PHASE_ENV = os.environ.get("AM_QUALITY_PHASE")
AM_QUALITY_PHASE = int(AM_QUALITY_PHASE_ENV) if AM_QUALITY_PHASE_ENV else None
AM_PHASE_ENV = os.environ.get("AM_PHASE")
AM_PHASE = int(AM_PHASE_ENV) if AM_PHASE_ENV else None
NUM_TOKENS = int(os.environ.get("NUM_TOKENS", "512"))
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen3-4B-Instruct-2507")
KEY_SELECT = os.environ.get("KEY_SELECT", "highest_attention")
ENABLE_BETA = os.environ.get("ENABLE_BETA", "1") not in ("0", "false", "False")
REBAKE_KEY_POSITIONS = os.environ.get("REBAKE_KEY_POSITIONS", "1") not in ("0", "false", "False")
STRIP_REF_SYSTEM_PROMPT = os.environ.get("STRIP_REF_SYSTEM_PROMPT", "1") not in ("0", "false", "False")
GLOBAL_TEACHER_POSITIONS = os.environ.get("GLOBAL_TEACHER_POSITIONS", "0") not in ("0", "false", "False")

def _resolve_rope_theta() -> float:
    """AM_ROPE_THETA -> float, accepting "model"/"auto" like `continual_write.py`.

    Only consumed by the REBAKE_KEY_POSITIONS rotation. The default is the
    historical hard-coded base, NOT the model's own (Qwen3-4B-Instruct-2507 =
    5e6), matching AMContinualConfig.rope_theta so old runs stay reproducible --
    but rebaking at a base the keys were not baked with rotates them off their
    slot, so a corrected run must pass 5000000 or "model" explicitly.
    """
    raw = os.environ.get("AM_ROPE_THETA")
    if raw is None:
        return 10000.0
    if raw.strip().lower() in ("model", "auto", "config"):
        from transformers import AutoConfig

        return float(AutoConfig.from_pretrained(MODEL_NAME).rope_theta)
    return float(raw)


AM_ROPE_THETA = _resolve_rope_theta()
RIDGE_LAMBDA = float(os.environ.get("RIDGE_LAMBDA", "1e-4"))
RIDGE_SCALE = os.environ.get("RIDGE_SCALE", "spectral")
MAX_REF_BATCHES = int(os.environ.get("MAX_REF_BATCHES", "50"))
MAX_QUERIES_PER_HEAD = int(os.environ.get("MAX_QUERIES_PER_HEAD", "64"))
MAX_TEACHER_TOKENS_ENV = os.environ.get("MAX_TEACHER_TOKENS")
MAX_TEACHER_TOKENS = int(MAX_TEACHER_TOKENS_ENV) if MAX_TEACHER_TOKENS_ENV else None
MAX_REF_EXAMPLES = int(os.environ.get("MAX_REF_EXAMPLES", "256"))
QUERIES_PER_BATCH = os.environ.get("QUERIES_PER_BATCH", "all_tokens")
GLOBAL_BATCH_SIZE = int(os.environ.get("GLOBAL_BATCH_SIZE", "32"))
NUM_BG_BATCHES = int(os.environ.get("NUM_BG_BATCHES", "1000"))
GRANULARITY = os.environ.get("GRANULARITY", "per_head")
DISTRIBUTED_BACKEND = os.environ.get("DISTRIBUTED_BACKEND", "nccl")
SEED = int(os.environ.get("SEED", "42"))
RUN_NAME = os.environ.get(
    "RUN_NAME",
    f"{AM_DATASET}_phase1_am_compaction_"
    f"{MODEL_NAME.split('/')[-1]}_{NUM_TOKENS}_{GRANULARITY}",
)

if not QA_DATA_PATH:
    raise ValueError("Set QA_DATA_PATH (phase-1 self-study parquet)")
if AM_DATASET not in {"qasper", "quality", "finqa", "techqa", "longhealth"}:
    raise ValueError(
        f"AM_DATASET={AM_DATASET!r} is unsupported; expected "
        "qasper, quality, finqa, techqa, or longhealth"
    )
if AM_DATASET == "quality" and AM_QUALITY_PHASE is None:
    raise ValueError("AM_QUALITY_PHASE is required when AM_DATASET=quality")
if AM_DATASET in {"finqa", "techqa", "longhealth"} and AM_PHASE not in range(1, 6):
    raise ValueError(
        f"AM_PHASE (1..5) is required when AM_DATASET={AM_DATASET!r}"
    )

_model_cls = FlexQwen3ForCausalLM if "qwen" in MODEL_NAME.lower() else FlexLlamaForCausalLM


def _collate_first(batch):
    return batch[0]


def _setup_run_dir(config: TrainConfig) -> Path:
    if config.run_dir is None:
        time_tag = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        script_id = "initial_am_compaction"
        run_id = str(uuid.uuid4())
        config.launch_id = f"{time_tag}-{script_id}"
        config.run_id = run_id
        config.run_dir = os.path.join(config.output_dir, config.launch_id, run_id)
    run_dir = Path(config.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    config.to_yaml(str(run_dir / "config.yaml"))
    return run_dir


def run_am_compaction_phase1(config: TrainConfig):
    seed_everything(config.seed)
    local_rank = "cuda" if torch.cuda.is_available() else "cpu"
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)

    run_dir = _setup_run_dir(config)

    if config.wandb is not None:
        config.wandb.name = config.name
        prepare_wandb(config.wandb, config.to_dict())
        import wandb

        wandb.log({"phase": 1, "run_dir": str(run_dir)}, step=0)

    tokenizer = AutoTokenizer.from_pretrained(config.model.pretrained_model_name_or_path)
    dataset = config.dataset.instantiate(tokenizer=tokenizer, seed=config.seed)
    dataloader = DataLoader(
        dataset, batch_size=1, collate_fn=_collate_first, num_workers=0,
    )

    # Cast before the device move: the reverse order stages a full fp32 copy of the
    # weights on the accelerator (16 GiB for Qwen3-4B) purely to throw it away.
    model = config.model.instantiate().to(torch.bfloat16).to(local_rank)
    for p in model.parameters():
        p.requires_grad = False

    attn_config = AttnConfig(
        n_layers=model.config.num_hidden_layers,
        n_heads=model.config.num_key_value_heads,
        head_dim=(
            model.config.head_dim
            if hasattr(model.config, "head_dim")
            else model.config.hidden_size // model.config.num_attention_heads
        ),
    )

    t_compact = time.time()
    cache, compaction_stats = compact_cache_am_phase1(
        model=model,
        tokenizer=tokenizer,
        qa_data_path=QA_DATA_PATH,
        attn_config=attn_config,
        dataset=AM_DATASET,
        qasper_topic=AM_QASPER_TOPIC,
        quality_phase=AM_QUALITY_PHASE,
        phase=AM_PHASE,
        num_tokens=NUM_TOKENS,
        key_select=KEY_SELECT,
        ridge_lambda=RIDGE_LAMBDA,
        ridge_scale=RIDGE_SCALE,
        enable_beta=ENABLE_BETA,
        rebake_key_positions=REBAKE_KEY_POSITIONS,
        strip_reference_system_prompt=STRIP_REF_SYSTEM_PROMPT,
        global_teacher_positions=GLOBAL_TEACHER_POSITIONS,
        rope_theta=AM_ROPE_THETA,
        max_ref_batches=MAX_REF_BATCHES,
        queries_per_batch=QUERIES_PER_BATCH,
        max_queries_per_head=MAX_QUERIES_PER_HEAD,
        max_teacher_tokens=MAX_TEACHER_TOKENS,
        max_ref_examples=MAX_REF_EXAMPLES,
        local_rank=local_rank,
    )
    compact_time = time.time() - t_compact
    compaction_stats["compact_time_s"] = compact_time
    logger.info(f"Compaction completed in {compact_time:.1f}s: {compaction_stats}")

    torch.save(compaction_stats, run_dir / "compaction_stats.pt")

    if config.wandb is not None:
        import wandb

        wandb.log(
            {
                "phase1/compact_time_s": compact_time,
                "phase1/recon_mse_mean": compaction_stats.get("recon_mse_mean") or 0.0,
                "phase1/recon_mse_max": compaction_stats.get("recon_mse_max") or 0.0,
                "phase1/T_teacher": compaction_stats.get("T_teacher", 0),
                "phase1/n_documents": compaction_stats.get("n_documents", 0),
            },
            step=0,
        )

    # Collect bg_stats (per_head to match Phase 2 best recipe).
    sparse_cfg = config.sparse_cache_finetuning
    bg_path = str(run_dir / "bg_stats.pt")
    if sparse_cfg and sparse_cfg.collect_background_stats:
        wrapped = CacheAndModel(cache, model, sparse_config=sparse_cfg)
        collect_background_stats(
            wrapped_model=wrapped,
            cache=cache,
            dataloader=dataloader,
            config=sparse_cfg,
            local_rank=local_rank,
            save_path=bg_path,
            is_ddp=False,
            is_rank_zero=True,
        )

    # Eval QA perplexity.
    eval_metrics = {}
    if config.loss_evals:
        wrapped = CacheAndModel(cache, model)
        for ds_config, eval_dataset in [
            (le, le.dataset.instantiate(tokenizer=tokenizer, seed=config.seed))
            for le in config.loss_evals
        ]:
            metrics = evaluate_perplexity(
                config=config,
                model=wrapped,
                cache=cache,
                eval_dataset=eval_dataset,
                ds_config=ds_config,
                optimizer_step=0,
                epoch=0,
                local_rank=local_rank,
                cache_tuning=True,
            )
            if metrics:
                eval_metrics[metrics["name"]] = metrics

    save_cache(config, cache, optimizer_step=0)

    summary = {
        "run_name": config.name,
        "run_dir": str(run_dir),
        "cache_path": str(run_dir / "cache_last.pt"),
        "bg_stats_path": bg_path,
        "num_tokens": NUM_TOKENS,
        "key_select": KEY_SELECT,
        "enable_beta": ENABLE_BETA,
        "rebake_key_positions": REBAKE_KEY_POSITIONS,
        "global_teacher_positions": GLOBAL_TEACHER_POSITIONS,
        "rope_theta": AM_ROPE_THETA,
        "ridge_lambda": RIDGE_LAMBDA,
        "ridge_scale": RIDGE_SCALE,
        "granularity": GRANULARITY,
        "compaction_stats": compaction_stats,
        "eval_metrics": eval_metrics,
    }
    (run_dir / "SUMMARY.md").write_text(
        "# Phase 1 AM Compaction — Summary\n\n"
        f"- run_dir: `{run_dir}`\n"
        f"- cache: `{run_dir / 'cache_last.pt'}`\n"
        f"- bg_stats: `{bg_path}`\n"
        f"- num_tokens: {NUM_TOKENS} | key_select: {KEY_SELECT} | beta: {ENABLE_BETA}\n"
        f"- ridge: {RIDGE_SCALE} lambda={RIDGE_LAMBDA}\n"
        f"- rebake_key_positions: {REBAKE_KEY_POSITIONS} | "
        f"global_teacher_positions: {GLOBAL_TEACHER_POSITIONS} | "
        f"rope_theta: {AM_ROPE_THETA:g}\n"
        f"- {compaction_stats.get('rope_note')}\n"
        f"- teacher tokens: {compaction_stats.get('T_teacher')} over "
        f"{compaction_stats.get('n_documents')} docs\n"
        f"- recon MSE mean/max: {compaction_stats.get('recon_mse_mean')} / "
        f"{compaction_stats.get('recon_mse_max')}\n"
        f"- QA eval: "
        + ", ".join(
            f"{k}: loss={v.get('loss'):.4f} ppl={v.get('perplexity'):.3f}"
            for k, v in eval_metrics.items()
        )
        + "\n\n```json\n"
        + json.dumps({k: v for k, v in summary.items() if k != "compaction_stats"}, indent=2, default=str)
        + "\n```\n"
    )
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

    logger.info(f"AM Phase 1 compaction complete. Saved to {run_dir}")

    if config.wandb is not None:
        import wandb

        payload = {
            "phase1/complete": 1,
            "phase1/cache_path": str(run_dir / "cache_last.pt"),
            "phase1/bg_stats_path": bg_path,
        }
        for name, m in eval_metrics.items():
            payload[f"phase1/eval_{name}_loss"] = m["loss"]
            payload[f"phase1/eval_{name}_ppl"] = m["perplexity"]
        wandb.log(payload, step=0)
        wandb.finish()

    return summary


def _build_loss_evals() -> list[LossEvalConfig]:
    paths = [os.environ.get(f"EVAL_P{i}_PATH") for i in range(1, 6)]
    if EVAL_DATA_PATH and not paths[0]:
        paths[0] = EVAL_DATA_PATH
    return [
        LossEvalConfig(
            dataset=LossEvalDataset.Config(
                data_source=DataSource(path=path, type="local"),
                packed_seq_length=2048,
            ),
            name_for_wandb=os.environ.get(f"EVAL_P{i}_NAME", f"p{i}"),
        )
        for i, path in enumerate(paths, start=1)
        if path
    ]


config = TrainConfig(
    model=HFModelConfig(
        pretrained_model_name_or_path=MODEL_NAME,
        model_cls=_model_cls,
        load_kwargs={"torch_dtype": "bfloat16"},
    ),
    optimizer="adam",
    lr=0.0,
    epochs=0,
    global_batch_size=GLOBAL_BATCH_SIZE,
    dataset=TrainDataset.Config(
        data_sources=[DataSource(path=QA_DATA_PATH, type="local")],
        top_k_logits=20,
        packed_seq_length=2048,
        packing_mode="truncate",
    ),
    loss_evals=_build_loss_evals(),
    sparse_cache_finetuning=SparseCacheFinetuningConfig(
        enabled=False,
        collect_background_stats=True,
        num_background_batches=NUM_BG_BATCHES,
        granularity=GRANULARITY,
    ),
    save_after_training=True,
    distributed_backend=DISTRIBUTED_BACKEND,
    wandb=WandBConfig(
        tags=["train", AM_DATASET, "phase1", "am", "compaction", GRANULARITY.replace("_", "-")],
        group=os.environ.get("WANDB_GROUP") or None,
        notes=os.environ.get("WANDB_NOTES") or None,
    ) if os.environ.get("WANDB_DISABLED", "0") not in ("1", "true", "True") else None,
    output_dir=os.environ.get("CARTRIDGES_OUTPUT_DIR", "."),
    name=RUN_NAME,
    seed=SEED,
)


if __name__ == "__main__":
    run_am_compaction_phase1(config)
