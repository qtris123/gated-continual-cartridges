"""DIAG-WIRE: does the per_document AM path consult AM config.target_mode?

CPU-only, synthetic tensors, no model. Four checks:

  A. import provenance (RUNBOOK 6.10 dual-`cartridges` foot-gun)
  B. do the three *targets* differ at all, when built explicitly?
  C. does apply_document_am_write_to_cache (the per_document write, the one
     EXP-008 exercised) produce different cache values for the three modes?
  D. control: does apply_am_update_to_cache (the legacy_decoupled write) produce
     different cache values for self vs teacher_attention?

Run:
  cd /localhome/local-triv/gated-continual-cartridges_explore
  CARTRIDGES_DIR=$PWD CARTRIDGES_OUTPUT_DIR=$PWD/outputs \
  PYTHONPATH="$PWD:$PYTHONPATH" \
  .venv/bin/python research_loop/results/DIAG-WIRE/unit_check.py
"""

from __future__ import annotations

import copy
import json
import os

import torch
import torch.nn as nn

import cartridges
import cartridges.am.finetune as amf
from cartridges.am.finetune import (
    AttentionMatchingFinetuningConfig,
    apply_am_update_to_cache,
    apply_document_am_write_to_cache,
)
from cartridges.am.query_accum import AMQueryAccumulator, AMTargetAccumulator
from cartridges.am.teacher import compute_teacher_targets, concat_teacher_kv
from cartridges.am.core import compute_attention_output
from cartridges.sparse_cache_finetuning import GradientMask

REPORT: dict = {}

# ---------------------------------------------------------------- A. provenance
pkg_dir = os.path.dirname(cartridges.__file__)
print("A. import provenance")
print("   cartridges       :", pkg_dir)
print("   am.finetune      :", amf.__file__)
print("   torch            :", torch.__version__)
assert pkg_dir.endswith("gated-continual-cartridges_explore/cartridges"), pkg_dir
REPORT["A_cartridges_pkg_dir"] = pkg_dir
REPORT["A_am_finetune_file"] = amf.__file__

# ------------------------------------------------------------- shared fixtures
torch.manual_seed(0)
N_LAYERS, N_KV_HEADS, N_Q_HEADS = 1, 2, 4
T_TRAIN, N_FROZEN, T_DOC, D = 12, 1, 5, 8
N_TOK = 9  # reference query tokens; groups = N_Q_HEADS // N_KV_HEADS = 2 -> n = 18
TOP_T = 4
HEAD_DIM = D

k_train = torch.randn(1, N_KV_HEADS, T_TRAIN, D)
v_train = torch.randn(1, N_KV_HEADS, T_TRAIN, D)
k_froz = torch.randn(1, N_KV_HEADS, N_FROZEN, D)
v_froz = torch.randn(1, N_KV_HEADS, N_FROZEN, D)
q_hook = torch.randn(1, N_Q_HEADS, N_TOK, D)          # captured post-RoPE queries
k_doc_l = torch.randn(N_KV_HEADS, T_DOC, D)
v_doc_l = torch.randn(N_KV_HEADS, T_DOC, D)
teacher_attn_out = torch.randn(1, N_Q_HEADS, N_TOK, D)  # stand-in for the
# no-cache teacher attention output that install_teacher_attention_capture_hooks
# would capture; any tensor works, the point is only whether it is *consulted*.


class FakeCache(nn.Module):
    """Minimal stand-in for TrainableCache: only the attributes the AM writes touch."""

    def __init__(self):
        super().__init__()
        self.trainable_keys = nn.ParameterList(
            [nn.Parameter(k_train.clone()) for _ in range(N_LAYERS)]
        )
        self.trainable_values = nn.ParameterList(
            [nn.Parameter(v_train.clone()) for _ in range(N_LAYERS)]
        )
        self.frozen_keys = [k_froz.clone() for _ in range(N_LAYERS)]
        self.frozen_values = [v_froz.clone() for _ in range(N_LAYERS)]
        self.trainable_beta = None
        self._num_frozen_tokens = N_FROZEN


def make_mask():
    return GradientMask(
        granularity="per_layer",
        positions_per_layer={0: torch.arange(TOP_T, dtype=torch.long)},
        n_tokens=T_TRAIN,
        top_t=TOP_T,
    )


def make_query_acc():
    acc = AMQueryAccumulator(
        granularity="per_layer",
        queries_per_batch="all_tokens",
        n_layers=N_LAYERS,
        n_kv_heads=N_KV_HEADS,
        device=torch.device("cpu"),
    )
    acc._queries[0] = [q_hook.clone()]
    return acc


def make_target_acc():
    acc = AMTargetAccumulator(n_layers=N_LAYERS, n_kv_heads=N_KV_HEADS)
    acc._targets[0] = [teacher_attn_out.clone()]
    return acc


def make_cfg(target_mode: str) -> AttentionMatchingFinetuningConfig:
    """EXP-008 canonical knobs, only target_mode varies."""
    return AttentionMatchingFinetuningConfig(
        enabled=True,
        top_t=TOP_T,
        use_idf=False,
        granularity="per_layer",
        execution_mode="per_document",
        target_mode=target_mode,
        key_mode="freeze",
        freeze_keys=True,
        enable_beta=None,
        ridge_lambda=1e-4,
        ridge_scale="spectral",
        ridge_lambda_min=0.0,
        delta_weight=1e-2,
        old_reference_weight=1.0,
        enable_old_reference_guard=False,
        queries_per_batch="all_tokens",
        max_queries_per_head=64,   # > n=18, so no randperm subsampling -> deterministic
        compute_update_stats=True,
        slot_selection="tfidf",
        min_top_t_per_layer=1,
    )


# ------------------------------------------------ B. are the 3 targets different?
print("\nB. explicit target construction (layer 0, kv head 0)")
acc = make_query_acc()
queries = acc.get_layer_head_queries(0, 0, n_q_heads=N_Q_HEADS, n_kv_heads=N_KV_HEADS)
keys_full = torch.cat([k_froz[0, 0], k_train[0, 0]], dim=0)     # (T_TRAIN+N_FROZEN, D)
values_full = torch.cat([v_froz[0, 0], v_train[0, 0]], dim=0)
k_teacher, v_teacher = concat_teacher_kv(keys_full, values_full, k_doc_l[0], v_doc_l[0])
bias = torch.zeros(k_teacher.shape[0])

tgt_cpd = compute_teacher_targets(                     # "cartridge_plus_doc"
    queries, k_teacher, v_teacher, HEAD_DIM,
    attention_bias=bias,
    n_cartridge_keys=keys_full.shape[0],
    doc_rope_offset=k_doc_l[0].shape[0],
)
tgt_self = compute_attention_output(                   # "self" (targets=None branch)
    queries, keys_full, values_full, HEAD_DIM,
)
tgt_teach = make_target_acc().get_layer_head_targets(  # "teacher_attention"
    0, 0, n_q_heads=N_Q_HEADS, n_kv_heads=N_KV_HEADS
).to(torch.float32)

d_cpd_self = (tgt_cpd - tgt_self).abs().max().item()
d_cpd_teach = (tgt_cpd - tgt_teach).abs().max().item()
d_self_teach = (tgt_self - tgt_teach).abs().max().item()
print(f"   queries shape {tuple(queries.shape)}  targets shape {tuple(tgt_cpd.shape)}")
print(f"   max|cartridge_plus_doc - self|            = {d_cpd_self:.6e}")
print(f"   max|cartridge_plus_doc - teacher_attention| = {d_cpd_teach:.6e}")
print(f"   max|self - teacher_attention|             = {d_self_teach:.6e}")
targets_differ = min(d_cpd_self, d_cpd_teach, d_self_teach) > 0
print(f"   => the three targets are GENUINELY DIFFERENT: {targets_differ}")
REPORT["B_target_maxabs_diff"] = {
    "cartridge_plus_doc_vs_self": d_cpd_self,
    "cartridge_plus_doc_vs_teacher_attention": d_cpd_teach,
    "self_vs_teacher_attention": d_self_teach,
    "targets_differ": bool(targets_differ),
}

# ------------------- C. the per_document write: does target_mode change anything?
print("\nC. apply_document_am_write_to_cache (per_document path, the EXP-008 path)")
results = {}
for mode in ("cartridge_plus_doc", "self", "teacher_attention"):
    cache = FakeCache()
    torch.manual_seed(1234)  # identical RNG state for every mode
    stats = apply_document_am_write_to_cache(
        cache=cache,
        mask=make_mask(),
        query_accumulator=make_query_acc(),
        doc_kv={0: (k_doc_l.clone(), v_doc_l.clone())},
        config=make_cfg(mode),
        n_layers=N_LAYERS,
        head_dim=HEAD_DIM,
        old_query_accumulator=None,
        old_target_bank=None,
    )
    results[mode] = (
        cache.trainable_values[0].detach().clone(),
        float(stats.mean_mse),
    )
    print(f"   {mode:20s} mean_mse={stats.mean_mse!r}")

base_v, base_mse = results["cartridge_plus_doc"]
c_diffs = {}
for mode in ("self", "teacher_attention"):
    v, mse = results[mode]
    dv = (v - base_v).abs().max().item()
    bit_eq = bool(torch.equal(v, base_v))
    c_diffs[mode] = {
        "max_abs_value_diff_vs_cartridge_plus_doc": dv,
        "bitwise_equal": bit_eq,
        "mean_mse": mse,
        "mean_mse_equal": mse == base_mse,
    }
    print(f"   {mode:20s} vs cartridge_plus_doc: max|dV|={dv:.3e} bitwise_equal={bit_eq}")
per_doc_ignores = all(c_diffs[m]["bitwise_equal"] for m in c_diffs)
print(f"   => per_document write IGNORES target_mode: {per_doc_ignores}")
REPORT["C_per_document"] = c_diffs
REPORT["C_per_document_ignores_target_mode"] = bool(per_doc_ignores)

# also: the function signature cannot even receive teacher-attention targets
import inspect

sig = list(inspect.signature(apply_document_am_write_to_cache).parameters)
src = inspect.getsource(apply_document_am_write_to_cache)
print(f"   signature params: {sig}")
print(f"   'target_mode' appears in apply_document_am_write_to_cache source: "
      f"{'target_mode' in src}")
src_cont = inspect.getsource(__import__('cartridges.am.continual', fromlist=['x']))
print(f"   'target_mode' appears in cartridges/am/continual.py source        : "
      f"{'target_mode' in src_cont}")
REPORT["C_signature_params"] = sig
REPORT["C_target_mode_in_apply_document_source"] = "target_mode" in src
REPORT["C_target_mode_in_continual_module_source"] = "target_mode" in src_cont

# ------------------------- D. control: the legacy_decoupled write DOES use it
print("\nD. control - apply_am_update_to_cache (legacy_decoupled path)")
d_results = {}
for mode in ("self", "teacher_attention"):
    cache = FakeCache()
    torch.manual_seed(1234)
    stats = apply_am_update_to_cache(
        cache=cache,
        mask=make_mask(),
        query_accumulator=make_query_acc(),
        n_layers=N_LAYERS,
        head_dim=HEAD_DIM,
        target_mode=mode,
        ridge_lambda=1e-4,
        freeze_keys=True,
        max_queries_per_head=64,
        compute_stats=True,
        target_accumulator=make_target_acc() if mode == "teacher_attention" else None,
        old_reference_weight=0.0,
        delta_weight=0.0,
        ridge_scale="spectral",
        ridge_lambda_min=0.0,
    )
    d_results[mode] = cache.trainable_values[0].detach().clone()
    print(f"   {mode:20s} mean_mse={stats.mean_mse!r}")
dv_d = (d_results["self"] - d_results["teacher_attention"]).abs().max().item()
print(f"   self vs teacher_attention: max|dV|={dv_d:.3e} "
      f"bitwise_equal={torch.equal(d_results['self'], d_results['teacher_attention'])}")
REPORT["D_legacy_decoupled_self_vs_teacher_maxabs"] = dv_d
REPORT["D_legacy_decoupled_uses_target_mode"] = bool(dv_d > 0)

# ------------------------------------------------------------------- verdict
print("\nVERDICT")
print("  targets genuinely differ                     :", REPORT["B_target_maxabs_diff"]["targets_differ"])
print("  per_document write ignores target_mode       :", REPORT["C_per_document_ignores_target_mode"])
print("  legacy_decoupled write honours target_mode   :", REPORT["D_legacy_decoupled_uses_target_mode"])

out = os.path.join(
    os.environ.get("CARTRIDGES_DIR", "."),
    "research_loop/state/diagnostics/DIAG-WIRE.json",
)
with open(out, "w") as f:
    json.dump(REPORT, f, indent=2)
print("\nwrote", out)
