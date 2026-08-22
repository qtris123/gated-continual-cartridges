"""Diagnose the compaction Phase-1 cartridge: eval QA loss with beta on/off + norms."""

import os
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from cartridges.cache import TrainableCache
from cartridges.datasets import DataSource, LossEvalDataset
from cartridges.models import FlexQwen3ForCausalLM, HFModelConfig
from cartridges.train import TrainConfig, LossEvalConfig, CacheAndModel, evaluate_perplexity

CACHE_PATH = os.environ["CACHE_PATH"]
EVAL_QA = os.environ["EVAL_QA"]
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen3-4B-Instruct-2507")

local_rank = "cuda" if torch.cuda.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(MODEL_NAME)
model = HFModelConfig(pretrained_model_name_or_path=MODEL_NAME, model_cls=FlexQwen3ForCausalLM).instantiate().to(local_rank).to(torch.bfloat16)
for p in model.parameters():
    p.requires_grad = False

cache = TrainableCache.from_pretrained(CACHE_PATH, device=local_rank)

# Value / key / beta norms.
with torch.no_grad():
    vmax = max(float(v.detach().float().abs().max()) for v in cache.trainable_values)
    kmax = max(float(k.detach().float().abs().max()) for k in cache.trainable_keys)
    bmax = max(float(b.detach().float().abs().max()) for b in cache.trainable_beta) if cache.trainable_beta is not None else 0.0
    bmin = min(float(b.detach().float().min()) for b in cache.trainable_beta) if cache.trainable_beta is not None else 0.0
print(f"NORMS value_absmax={vmax:.3f} key_absmax={kmax:.3f} beta_absmax={bmax:.3f} beta_min={bmin:.3f} beta_enabled={cache._attention_bias_enabled}")

cfg = TrainConfig(
    model=HFModelConfig(pretrained_model_name_or_path=MODEL_NAME, model_cls=FlexQwen3ForCausalLM),
    optimizer="adam", lr=0.0, epochs=0, global_batch_size=32,
    dataset=None,
    loss_evals=[LossEvalConfig(
        dataset=LossEvalDataset.Config(
            data_source=DataSource(path=EVAL_QA, type="local"), packed_seq_length=2048),
        name_for_wandb="qa")],
    output_dir=".", name="diag",
)
eval_ds = cfg.loss_evals[0].dataset.instantiate(tokenizer=tok, seed=0)


def run_eval(tag):
    wrapped = CacheAndModel(cache, model)
    m = evaluate_perplexity(
        config=cfg, model=wrapped, cache=cache, eval_dataset=eval_ds,
        ds_config=cfg.loss_evals[0], optimizer_step=0, epoch=0,
        local_rank=local_rank, cache_tuning=True,
    )
    print(f"EVAL[{tag}] loss={m['loss']:.4f} ppl={m['perplexity']:.2f}")
    return m["loss"]


print("beta_enabled currently:", cache._attention_bias_enabled)
run_eval("beta_on" if cache._attention_bias_enabled else "beta_off_default")
cache.enable_attention_bias(False)
run_eval("beta_off")
if cache.trainable_beta is not None:
    cache.enable_attention_bias(True)
    run_eval("beta_on")
