#!/usr/bin/env python
"""MECH-009 unit sanity: is `constrained_mass` the selector DIAG-IMPORTANCE projected?

Runs entirely on CPU. Checks, in order:
  0. `import cartridges` resolves to the `_explore` repo (RUNBOOK §6.10/§9c-bis).
  1. the config constructs with `slot_selection='constrained_mass'` + the two new
     fields, and their defaults are the INERT ones.
  2. `compute_slot_redundancy` still reproduces DIAG-IMPORTANCE's
     `score_redundancy` array exactly (MECH-008 got max|d| = 0.0; the new mode
     re-uses that scorer, so any drift here invalidates the whole arm).
  3. DEGENERACY A: safe_fraction = 1.0 reduces EXACTLY to `attention_mass` --
     identical index tensors per layer AND a bit-identical `select_score`.
  4. DEGENERACY B: a safe_fraction small enough that the candidate set is only
     `top_t` slots reduces EXACTLY to the pure safety selector (`redundancy`
     for safe_metric=redundancy, `fisher` for safe_metric=fisher).
  5. THE REPRODUCTION THAT MATTERS: at safe_fraction = 0.25 the candidate set is
     DIAG-IMPORTANCE's "safest quartile" (127 of 511), so
     `constrained_mass(fisher, q=0.25, t=32)` must reproduce its published
     `best32_within_safest_quartile_fisher` row (4.62% MT mass / 0.28% QA
     Fisher) and `constrained_mass(redundancy, q=0.25, t=32)` its
     `best32_within_most_redundant_quartile` row (17.31% / 12.34%).
  6. the q-sweep the GPU arms will run, scored on DIAG-IMPORTANCE's own axes
     (the PREDICTION for realised bandwidth/exposure, before any GPU time).
  7. the DEFAULT path (`tfidf`) is untouched -- same selection sha as MECH-008's
     sanity run recorded (f1aa6c6173119045c73ec8e70199b7fc).
  8. every failure mode fails LOUD.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys

import numpy as np
import torch

REPO = os.environ["CARTRIDGES_DIR"]
sys.path.insert(0, REPO)

import cartridges  # noqa: E402
from cartridges.am.finetune import AttentionMatchingFinetuningConfig  # noqa: E402
from cartridges.am.ranking import (  # noqa: E402
    SLOT_PRIOR_SELECTIONS,
    _rank_attention_mass_per_layer,
    compute_slot_redundancy,
    load_slot_fisher_scores,
    rank_am_slots,
)
from cartridges.cache import TrainableCache  # noqa: E402
from cartridges.sparse_cache_finetuning import CacheTFIDFRanker  # noqa: E402

OUT: dict = {}
PKG = os.path.dirname(cartridges.__file__)
OUT["cartridges_pkg"] = PKG
print(f"[0] cartridges -> {PKG}")
assert PKG.startswith(REPO), f"WRONG PACKAGE: {PKG} (RUNBOOK §6.10)"

NPZ = os.path.join(REPO, "research_loop/state/diagnostics/DIAG-IMPORTANCE.npz")
PHASE1 = os.path.join(REPO, "outputs/phase1_selfdistill_qwen512/cache_last.pt")
FISHER = os.path.join(REPO, "research_loop/state/diagnostics/slot_fisher_qa_phase1.npz")

z = np.load(NPZ)
ref_red = z["score_redundancy"]          # (36, 511)
ref_fisher = z["score_fisher"]           # (36, 511)
ranker_tf_mt = z["score_ranker_tf_mt"]   # the incumbent's exact score
tf_mass_mt = z["score_tf_mass_mt"]       # DIAG-IMPORTANCE's routing-mass axis
n_layers, n_slots = ref_red.shape


# ---------------------------------------------------------------- 1. config
def cfg(**kw):
    base = dict(enabled=True, top_t=32, granularity="per_layer")
    base.update(kw)
    return AttentionMatchingFinetuningConfig(**base)


assert "constrained_mass" in SLOT_PRIOR_SELECTIONS
for mode in ("tfidf", "attention_mass", "residual_budget", *SLOT_PRIOR_SELECTIONS):
    assert cfg(slot_selection=mode).slot_selection == mode
d = cfg().model_dump()
OUT["default_new_fields"] = {
    k: d[k] for k in ("slot_selection", "safe_fraction", "safe_metric")
}
print(f"[1] config constructs for all 7 modes; defaults = {OUT['default_new_fields']}")
assert d["slot_selection"] == "tfidf"
assert d["safe_fraction"] == 1.0 and d["safe_metric"] == "redundancy"

# ------------------------------------------------- 2. redundancy vs diagnostic
cache = TrainableCache.from_pretrained(PHASE1, device="cpu")
red = compute_slot_redundancy(cache, ridge_rel=1e-6).numpy()
absdiff = float(np.abs(red - ref_red).max())
ovl_self = float(
    np.mean([
        len(set(np.argsort(-red[l])[:32]) & set(np.argsort(-ref_red[l])[:32])) / 32
        for l in range(n_layers)
    ])
)
OUT["redundancy_vs_DIAG_IMPORTANCE"] = {
    "max_abs_diff": absdiff,
    "top32_set_agreement": ovl_self,
    "mean": float(red.mean()),
    "diag_mean": float(ref_red.mean()),
}
print(f"[2] redundancy vs DIAG-IMPORTANCE: max|d|={absdiff:.3e} "
      f"top32 set agreement={ovl_self:.4f}")
assert absdiff == 0.0, f"redundancy no longer reproduces the diagnostic: {absdiff}"
assert ovl_self == 1.0

fsh = load_slot_fisher_scores(FISHER, n_layers, n_slots).numpy()
assert np.abs(fsh - ref_fisher).max() == 0.0
print("[2b] fisher cache byte-equal to DIAG-IMPORTANCE's score_fisher")

tfidf_ranker = CacheTFIDFRanker(granularity="per_layer", use_idf=False)


def sel(mask):
    return {l: set(mask.positions_per_layer[l].tolist()) for l in range(n_layers)}


def idx_tensor(mask):
    return torch.stack([mask.positions_per_layer[l] for l in range(n_layers)])


def overlap(a, b, k):
    return float(np.mean([len(a[l] & b[l]) / k for l in range(n_layers)]))


def axes(mask, k):
    s = sel(mask)
    idx = np.array([sorted(s[l]) for l in range(n_layers)])
    mt = float(np.mean([tf_mass_mt[l, idx[l]].sum() / tf_mass_mt[l].sum()
                        for l in range(n_layers)]))
    fi = float(np.mean([ref_fisher[l, idx[l]].sum() / ref_fisher[l].sum()
                        for l in range(n_layers)]))
    return mt, fi


def cm(access, q, metric, top_t, **kw):
    c = cfg(slot_selection="constrained_mass", safe_fraction=q, safe_metric=metric,
            slot_fisher_path=FISHER, top_t=top_t, **kw)
    return rank_am_slots(access, top_t, tfidf_ranker, c, step=1, cache=cache)


# The access scores the ranker sees in production are the incumbent's own score.
access_inc = torch.from_numpy(ranker_tf_mt.astype(np.float64)).float()
# DIAG-IMPORTANCE's tradeoff table ranks by `tf_mass_mt`, so the reproduction in
# [5] uses that array as the mass axis instead.
access_tfm = torch.from_numpy(tf_mass_mt.astype(np.float64)).float()

# ------------------------------------------------------- 3. degeneracy q = 1.0
m_q1, i_q1 = cm(access_inc, 1.0, "redundancy", 32)
m_am, i_am = _rank_attention_mass_per_layer(access_inc, 32)
same_idx = bool(torch.equal(idx_tensor(m_q1), idx_tensor(m_am)))
same_score = bool(torch.equal(i_q1.tfidf, i_am.tfidf))
# also with the fisher gate, and at a different budget
m_q1f, _ = cm(access_inc, 1.0, "fisher", 32)
m_q1_64, _ = cm(access_inc, 1.0, "redundancy", 64)
m_am64, _ = _rank_attention_mass_per_layer(access_inc, 64)
OUT["degeneracy_q1_is_attention_mass"] = {
    "identical_index_tensor": same_idx,
    "identical_select_score": same_score,
    "set_agreement": overlap(sel(m_q1), sel(m_am), 32),
    "fisher_gate_identical": bool(torch.equal(idx_tensor(m_q1f), idx_tensor(m_am))),
    "top_t64_identical": bool(torch.equal(idx_tensor(m_q1_64), idx_tensor(m_am64))),
}
print(f"[3] q=1.0 == attention_mass: idx={same_idx} score={same_score} "
      f"(fisher gate {OUT['degeneracy_q1_is_attention_mass']['fisher_gate_identical']}, "
      f"t=64 {OUT['degeneracy_q1_is_attention_mass']['top_t64_identical']})")
assert same_idx and same_score
assert OUT["degeneracy_q1_is_attention_mass"]["fisher_gate_identical"]
assert OUT["degeneracy_q1_is_attention_mass"]["top_t64_identical"]

# --------------------------------------------- 4. degeneracy q -> pure safety
# n_safe = floor(q * 511) clamped up to top_t, so any q <= 32/511 = 0.0626 gives
# a 32-slot candidate set == the pure safety selector's own top-32.
m_red, _ = rank_am_slots(access_inc, 32, tfidf_ranker,
                         cfg(slot_selection="redundancy"), step=1, cache=cache)
m_fis, _ = rank_am_slots(access_inc, 32, tfidf_ranker,
                         cfg(slot_selection="fisher", slot_fisher_path=FISHER),
                         step=1, cache=cache)
deg = {}
for q in (0.0626, 0.05, 0.001):
    a, _ = cm(access_inc, q, "redundancy", 32)
    b, _ = cm(access_inc, q, "fisher", 32)
    deg[str(q)] = {
        "vs_pure_redundancy_set": overlap(sel(a), sel(m_red), 32),
        "vs_pure_fisher_set": overlap(sel(b), sel(m_fis), 32),
    }
    assert deg[str(q)]["vs_pure_redundancy_set"] == 1.0
    assert deg[str(q)]["vs_pure_fisher_set"] == 1.0
OUT["degeneracy_small_q_is_pure_safety"] = deg
print(f"[4] small q == pure safety selector: {json.dumps(deg)}")

# ------------------------------- 5. reproduce DIAG-IMPORTANCE's projected rows
rep = {}
mF, _ = cm(access_tfm, 0.25, "fisher", 32)
mR, _ = cm(access_tfm, 0.25, "redundancy", 32)
rep["best32_within_safest_quartile_fisher"] = dict(
    zip(("frac_writable_MT_routing_mass", "frac_total_QA_Fisher_mass"), axes(mF, 32))
)
rep["best32_within_most_redundant_quartile"] = dict(
    zip(("frac_writable_MT_routing_mass", "frac_total_QA_Fisher_mass"), axes(mR, 32))
)
rep["published"] = {
    "best32_within_safest_quartile_fisher": {
        "frac_writable_MT_routing_mass": 0.0462, "frac_total_QA_Fisher_mass": 0.0028},
    "best32_within_most_redundant_quartile": {
        "frac_writable_MT_routing_mass": 0.1731, "frac_total_QA_Fisher_mass": 0.1234},
}
OUT["reproduces_DIAG_IMPORTANCE_tradeoff_rows"] = rep
for key in ("best32_within_safest_quartile_fisher", "best32_within_most_redundant_quartile"):
    got, want = rep[key], rep["published"][key]
    print(f"[5] {key}: MT {got['frac_writable_MT_routing_mass']:.4f} "
          f"(published {want['frac_writable_MT_routing_mass']}) / "
          f"QAfisher {got['frac_total_QA_Fisher_mass']:.4f} "
          f"(published {want['frac_total_QA_Fisher_mass']})")
    assert abs(got["frac_writable_MT_routing_mass"]
               - want["frac_writable_MT_routing_mass"]) < 5e-4, key
    assert abs(got["frac_total_QA_Fisher_mass"]
               - want["frac_total_QA_Fisher_mass"]) < 5e-4, key

# ------------------------------------------- 6. the q-sweep the GPU arms will run
incumbent = {l: set(np.argsort(-ranker_tf_mt[l])[:32].tolist()) for l in range(n_layers)}
sweep = {}
for q, t in ((1.0, 32), (0.75, 32), (0.50, 32), (0.25, 32), (0.50, 64)):
    m, _ = cm(access_inc, q, "redundancy", t)
    mt, fi = axes(m, t)
    n_safe = max(t, int(np.floor(q * n_slots)))
    sweep[f"q{q}_t{t}"] = {
        "n_candidate_slots_per_layer": n_safe,
        "frac_writable_MT_routing_mass": mt,
        "frac_total_QA_Fisher_mass": fi,
        "top32_overlap_vs_incumbent": overlap(sel(m), incumbent, 32) if t == 32 else None,
        "mean_redundancy_of_selected": float(np.mean([
            ref_red[l, sorted(sel(m)[l])].mean() for l in range(n_layers)])),
    }
    print(f"[6] q={q} t={t}: cand={n_safe}/511  MTmass={mt:.4f}  QAfisher={fi:.4f}  "
          f"ovl_incumbent={sweep[f'q{q}_t{t}']['top32_overlap_vs_incumbent']}")
OUT["predicted_sweep_redundancy_gate"] = sweep
# monotonicity of the prediction: tightening q can only lose MT bandwidth
mts = [sweep[k]["frac_writable_MT_routing_mass"] for k in ("q1.0_t32", "q0.75_t32",
                                                           "q0.5_t32", "q0.25_t32")]
OUT["predicted_MT_mass_monotone_decreasing_in_tightness"] = all(
    mts[i] >= mts[i + 1] for i in range(len(mts) - 1))
print(f"[6b] predicted MT-mass monotone in q: {mts}")

# ------------------------------------------------- 7. default path is untouched
mask_t, _ = rank_am_slots(access_inc, 32, tfidf_ranker, cfg(), step=3, cache=cache)
mask_t2, _ = tfidf_ranker.rank_positions(access_inc, top_t=32, step=3, return_info=True)
assert overlap(sel(mask_t), sel(mask_t2), 32) == 1.0
mask_t3, _ = rank_am_slots(access_inc, 32, tfidf_ranker, cfg(), step=3)
assert overlap(sel(mask_t), sel(mask_t3), 32) == 1.0
sha = hashlib.sha256(
    json.dumps({l: sorted(sel(mask_t)[l]) for l in range(n_layers)}).encode()
).hexdigest()[:32]
OUT["default_tfidf_selection_sha"] = sha
OUT["default_tfidf_selection_sha_matches_MECH008"] = (
    sha == "f1aa6c6173119045c73ec8e70199b7fc")
print(f"[7] default tfidf selection sha={sha} "
      f"matches MECH-008: {OUT['default_tfidf_selection_sha_matches_MECH008']}")
assert OUT["default_tfidf_selection_sha_matches_MECH008"]
# and the three MECH-008 modes are unchanged too
mech008 = {}
for mode, want in (("redundancy", 0.09574031084775925),
                   ("fisher", 0.010441286489367485),
                   ("mass_x_redundancy", 0.19762763381004333)):
    m, _ = rank_am_slots(access_inc, 32, tfidf_ranker,
                         cfg(slot_selection=mode, slot_fisher_path=FISHER),
                         step=1, cache=cache)
    mech008[mode] = axes(m, 32)[0]
    assert abs(mech008[mode] - want) < 1e-9, (mode, mech008[mode], want)
OUT["mech008_modes_unchanged"] = mech008
print(f"[7b] MECH-008 modes unchanged: {mech008}")


# ---------------------------------------------------------- 8. fails are loud
def expect(exc, fn, label):
    try:
        fn()
    except exc as e:
        print(f"[8] {label}: raised {type(e).__name__}: {str(e)[:90]}")
        OUT.setdefault("fail_loud_cases_verified", []).append(label)
        return True
    raise AssertionError(f"{label} did NOT raise {exc}")


expect(ValueError, lambda: cm(access_inc, 0.0, "redundancy", 32), "safe_fraction = 0")
expect(ValueError, lambda: cm(access_inc, -0.1, "redundancy", 32), "safe_fraction < 0")
expect(ValueError, lambda: cm(access_inc, 1.5, "redundancy", 32), "safe_fraction > 1")
expect(ValueError, lambda: rank_am_slots(
    access_inc, 32, tfidf_ranker,
    cfg(slot_selection="constrained_mass", safe_fraction=0.5, safe_metric="entropy"),
    cache=cache), "unknown safe_metric")
expect(ValueError, lambda: rank_am_slots(
    access_inc, 32, tfidf_ranker,
    cfg(slot_selection="constrained_mass", safe_fraction=0.5, safe_metric="fisher"),
    cache=cache), "fisher gate without a path")
expect(ValueError, lambda: rank_am_slots(
    access_inc, 32, tfidf_ranker,
    cfg(slot_selection="constrained_mass", safe_fraction=0.5)), "cache=None")
expect(ValueError, lambda: rank_am_slots(
    access_inc, 32, tfidf_ranker,
    cfg(slot_selection="constrained_mass", safe_fraction=0.5, granularity="per_head"),
    cache=cache), "granularity != per_layer")

dest = os.path.join(REPO, "research_loop/results/MECH-CONSTRAINED/sanity.json")
os.makedirs(os.path.dirname(dest), exist_ok=True)
with open(dest, "w") as fh:
    json.dump(OUT, fh, indent=2)
print(f"\nSANITY_OK -> {dest}")
