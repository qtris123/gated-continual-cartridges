"""One per-document AM write on synthetic tensors, dumped as JSON.

Driven by `verify_am_restructure.py` check C: run once with PYTHONPATH pointed
at a pre-refactor checkout and once at the current tree, then diff the dumps.
This is the check the static harness cannot make -- the leaf bodies are provably
identical, so any difference here is a REWIRING bug at a call site.

Deliberately CPU-only and tiny, with `max_queries_per_head` above the query
supply so the `randperm` subsample never fires and the write is deterministic.

    python verify_am_write_probe.py {old|new} {keys_repos|plain|beta|oracle}
"""

import json
import sys

import torch

MODE = sys.argv[1]   # "old" | "new"
ARM = sys.argv[2]    # which write rule to exercise

# Each arm pins a different branch of the write. Between them they cover every
# path through `ValueObjective.solve` plus the key and beta stages.
ARMS = {
    # MECH-KEYS winning arm: key rewrite + reposition, delta trust region.
    "keys_repos": dict(key_mode="highest_attention", key_reposition=True,
                       enable_beta=False, delta_weight=1e-2, oracle_write=False),
    # Plain residual ridge solve: no keys, no beta, no delta.
    "plain":      dict(key_mode="freeze", key_reposition=False,
                       enable_beta=False, delta_weight=0.0, oracle_write=False),
    # Beta (NNLS mass matching) on, frozen keys -- the B-ROUTE configuration.
    "beta":       dict(key_mode="freeze", key_reposition=False,
                       enable_beta=True, delta_weight=0.0, oracle_write=False),
    # B-ROUTE write-ceiling oracle: skips the solve entirely.
    "oracle":     dict(key_mode="freeze", key_reposition=False,
                       enable_beta=False, delta_weight=1e-2, oracle_write=True),
}
arm = ARMS[ARM]

torch.manual_seed(0)

N_LAYERS, N_KV_HEADS, HEAD_DIM = 3, 2, 8
N_SLOTS, N_DOC, N_QUERIES, TOP_T = 12, 5, 6, 4
ROPE_THETA = 5000000.0
RIDGE_LAMBDA, RIDGE_SCALE, RIDGE_LAMBDA_MIN = 1e-4, "spectral", 0.0
BETA_BOX, NNLS_ITERS, NNLS_DRIVER, BETA_TARGET = 3.0, 2, "gelsd", "residual"

from cartridges.cache import AttnConfig, TrainableCache
from cartridges.sparse_cache_finetuning import GradientMask

attn = AttnConfig(n_layers=N_LAYERS, n_heads=N_KV_HEADS, head_dim=HEAD_DIM)

gen = torch.Generator().manual_seed(1234)


def rand(*shape):
    return torch.randn(*shape, generator=gen, dtype=torch.float32)


init_keys = [rand(1, N_KV_HEADS, N_SLOTS, HEAD_DIM) for _ in range(N_LAYERS)]
init_values = [rand(1, N_KV_HEADS, N_SLOTS, HEAD_DIM) for _ in range(N_LAYERS)]
cache = TrainableCache(
    config=attn,
    init_keys=[k.clone() for k in init_keys],
    init_values=[v.clone() for v in init_values],
    num_frozen_tokens=1,
)

doc_kv = {
    l: (rand(N_KV_HEADS, N_DOC, HEAD_DIM), rand(N_KV_HEADS, N_DOC, HEAD_DIM))
    for l in range(N_LAYERS)
}

mask = GradientMask(
    granularity="per_layer",
    positions_per_layer={l: torch.arange(TOP_T) + l for l in range(N_LAYERS)},
    n_tokens=N_SLOTS,
    top_t=TOP_T,
)

if MODE == "old":
    from cartridges.am.finetune import (
        AttentionMatchingFinetuningConfig,
        apply_document_am_write_to_cache,
    )
    from cartridges.am.query_accum import AMQueryAccumulator
else:
    from cartridges.am import (
        BetaFitter, KeyWriter, ReferenceQueries,
        SlotSelector, TeacherTarget, ValueObjective,
    )
    from cartridges.am.components.queries import AMQueryAccumulator
    from cartridges.am.continual.config import AMStages
    from cartridges.am.continual.write import apply_document_am_write_to_cache

# Reference queries: one batch per layer, n_q_heads == n_kv_heads (no grouping).
acc = AMQueryAccumulator(
    granularity="per_layer",
    queries_per_batch="all_tokens",
    n_layers=N_LAYERS,
    n_kv_heads=N_KV_HEADS,
    device=torch.device("cpu"),
)
for l in range(N_LAYERS):
    acc._queries[l].append(rand(1, N_KV_HEADS, N_QUERIES, HEAD_DIM))

MAX_QUERIES_PER_HEAD = 999  # above supply -> no randperm -> deterministic
KEY_MODE = arm["key_mode"]
KEY_REPOSITION = arm["key_reposition"]
ENABLE_BETA = arm["enable_beta"]
DELTA_WEIGHT = arm["delta_weight"]
ORACLE_WRITE = arm["oracle_write"]
if ENABLE_BETA:
    cache.enable_attention_bias(True)

if MODE == "old":
    config = AttentionMatchingFinetuningConfig(
        enabled=True,
        top_t=TOP_T,
        granularity="per_layer",
        queries_per_batch="all_tokens",
        max_queries_per_head=MAX_QUERIES_PER_HEAD,
        rope_theta=ROPE_THETA,
        key_mode=KEY_MODE,
        key_reposition=KEY_REPOSITION,
        enable_beta=ENABLE_BETA,
        beta_fit_scope="selected",
        beta_box=BETA_BOX,
        nnls_iters=NNLS_ITERS,
        nnls_driver=NNLS_DRIVER,
        beta_target=BETA_TARGET,
        oracle_write=ORACLE_WRITE,
        oracle_write_assign="mass_ranked",
        ridge_lambda=RIDGE_LAMBDA,
        ridge_scale=RIDGE_SCALE,
        ridge_lambda_min=RIDGE_LAMBDA_MIN,
        delta_weight=DELTA_WEIGHT,
        compute_update_stats=True,
    )
    stats = apply_document_am_write_to_cache(
        cache=cache, mask=mask, query_accumulator=acc, doc_kv=doc_kv,
        config=config, n_layers=N_LAYERS, head_dim=HEAD_DIM,
    )
else:
    stages = AMStages(
        slots=SlotSelector.Config(top_t=TOP_T, granularity="per_layer").instantiate(),
        queries=ReferenceQueries.Config(
            queries_per_batch="all_tokens",
            max_queries_per_head=MAX_QUERIES_PER_HEAD,
        ).instantiate(),
        teacher=TeacherTarget.Config().instantiate(),
        keys=KeyWriter.Config(
            key_mode=KEY_MODE, key_reposition=KEY_REPOSITION
        ).instantiate(),
        beta=BetaFitter.Config(
            enabled=ENABLE_BETA, fit_scope="selected", beta_box=BETA_BOX,
            nnls_iters=NNLS_ITERS, nnls_driver=NNLS_DRIVER, target_mode=BETA_TARGET,
        ).instantiate(),
        objective=ValueObjective.Config(
            ridge_lambda=RIDGE_LAMBDA,
            ridge_scale=RIDGE_SCALE,
            ridge_lambda_min=RIDGE_LAMBDA_MIN,
            delta_weight=DELTA_WEIGHT,
            oracle_write=ORACLE_WRITE,
            oracle_write_assign="mass_ranked",
        ).instantiate(),
        rope_theta=ROPE_THETA,
        compute_stats=True,
    )
    stats = apply_document_am_write_to_cache(
        cache=cache, mask=mask, query_accumulator=acc, doc_kv=doc_kv,
        stages=stages, n_layers=N_LAYERS, head_dim=HEAD_DIM,
    )


def digest(t):
    return [round(float(x), 10) for x in t.detach().flatten().tolist()]


out = {
    "keys": [digest(k) for k in cache.trainable_keys],
    "values": [digest(v) for v in cache.trainable_values],
    "beta": (
        [digest(b) for b in cache.trainable_beta]
        if cache.trainable_beta is not None else None
    ),
    "mean_mse": round(float(stats.mean_mse), 10),
    "n_queries": int(stats.n_queries),
    "mse_per_layer": {str(k): round(float(v), 10) for k, v in stats.mse_per_layer.items()},
    "extra": json.loads(json.dumps(stats.extra, default=str)),
}
print("###JSON###" + json.dumps(out))
