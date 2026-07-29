#!/usr/bin/env python
"""MECH-008 unit sanity: does the new ranker do what DIAG-IMPORTANCE measured?

Runs entirely on CPU. Checks, in order:
  0. `import cartridges` resolves to the `_explore` repo (RUNBOOK §6.10/§9c-bis).
  1. the config constructs with every new `slot_selection` value + the new fields.
  2. `compute_slot_redundancy` reproduces DIAG-IMPORTANCE's `score_redundancy`
     array on the untouched Phase-1 cartridge.
  3. all three modes return finite scores, the right shapes, and exactly `top_t`
     unique slots per layer.
  4. the selected sets reproduce DIAG-IMPORTANCE's measured top-32 overlaps with
     the incumbent ranker (redundancy 10.7%, lowest-Fisher 0.09%).
  5. `mass_x_redundancy` degenerates exactly: alpha=0 -> `attention_mass`,
     alpha=1 -> `redundancy`.
  6. the DEFAULT path (`tfidf`) is untouched and its selection is unchanged.
  7. every failure mode fails LOUD (bad alpha, missing cache, wrong shape,
     missing fisher file, non-per_layer granularity).
"""

from __future__ import annotations

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
ref_red = z["score_redundancy"]  # (36, 511)
ref_fisher = z["score_fisher"]
ranker_tf_mt = z["score_ranker_tf_mt"]  # the incumbent's exact score
n_layers, n_slots = ref_red.shape


# ---------------------------------------------------------------- 1. config
def cfg(**kw):
    base = dict(enabled=True, top_t=32, granularity="per_layer")
    base.update(kw)
    return AttentionMatchingFinetuningConfig(**base)


for mode in ("tfidf", "attention_mass", "residual_budget", *SLOT_PRIOR_SELECTIONS):
    c = cfg(slot_selection=mode)
    assert c.slot_selection == mode
d = cfg().model_dump()
OUT["default_new_fields"] = {
    k: d[k] for k in ("slot_selection", "redundancy_ridge_rel",
                      "mass_redundancy_alpha", "slot_fisher_path")
}
print(f"[1] config constructs for all 6 modes; defaults = {OUT['default_new_fields']}")
assert d["slot_selection"] == "tfidf"
assert d["slot_fisher_path"] is None

# ------------------------------------------------- 2. redundancy vs diagnostic
cache = TrainableCache.from_pretrained(PHASE1, device="cpu")
red = compute_slot_redundancy(cache, ridge_rel=1e-6).numpy()
assert red.shape == ref_red.shape, (red.shape, ref_red.shape)
absdiff = float(np.abs(red - ref_red).max())
rel = float(np.abs(red - ref_red).max() / max(np.abs(ref_red).max(), 1e-12))
from scipy.stats import spearmanr  # noqa: E402

rho = float(spearmanr(red.ravel(), ref_red.ravel()).statistic)
ovl_self = float(
    np.mean([
        len(set(np.argsort(-red[l])[:32]) & set(np.argsort(-ref_red[l])[:32])) / 32
        for l in range(n_layers)
    ])
)
OUT["redundancy_vs_DIAG_IMPORTANCE"] = {
    "max_abs_diff": absdiff,
    "max_rel_diff": rel,
    "spearman": rho,
    "top32_set_agreement": ovl_self,
    "mean": float(red.mean()),
    "diag_mean": float(ref_red.mean()),
}
print(f"[2] redundancy vs DIAG-IMPORTANCE: max|d|={absdiff:.3e} rho={rho:.6f} "
      f"top32 set agreement={ovl_self:.4f} (mean {red.mean():.4f} vs {ref_red.mean():.4f})")
assert absdiff < 1e-4, f"redundancy does not reproduce the diagnostic: {absdiff}"
assert ovl_self == 1.0, "top-32 sets differ from the diagnostic"

# ridge sensitivity, matching METRICS.md §5
for tag, key in (("1e-4", "redundancy_ridge1e4"), ("1e-8", "redundancy_ridge1e8")):
    r = compute_slot_redundancy(cache, ridge_rel=float(tag)).numpy()
    dd = float(np.abs(r - z[key]).max())
    OUT.setdefault("redundancy_ridge_sensitivity", {})[tag] = dd
    print(f"    ridge_rel={tag}: max|d| vs diagnostic = {dd:.3e}")
    assert dd < 1e-4

# ------------------------------------------------------------ fisher cache file
if not os.path.exists(FISHER):
    print(f"[!] {FISHER} missing — run compute_slot_fisher.py first", file=sys.stderr)
    sys.exit(3)
fsh = load_slot_fisher_scores(FISHER, n_layers, n_slots).numpy()
assert np.abs(fsh - ref_fisher).max() == 0.0
print("[2b] fisher cache loads and is byte-equal to DIAG-IMPORTANCE's score_fisher")

# ----------------------------------------------------- 3/4. the three selections
# Use the incumbent's own exact per-(layer, slot) score as the access_scores the
# ranker would see, so the overlaps are directly comparable to DIAG-IMPORTANCE's.
access = torch.from_numpy(ranker_tf_mt.astype(np.float64)).float()
tfidf_ranker = CacheTFIDFRanker(granularity="per_layer", use_idf=False)


def sel(mask):
    return {l: set(mask.positions_per_layer[l].tolist()) for l in range(n_layers)}


def overlap(a, b, k):
    return float(np.mean([len(a[l] & b[l]) / k for l in range(n_layers)]))


incumbent = {l: set(np.argsort(-ranker_tf_mt[l])[:32].tolist()) for l in range(n_layers)}

results = {}
for mode in SLOT_PRIOR_SELECTIONS:
    c = cfg(slot_selection=mode, slot_fisher_path=FISHER)
    mask, info = rank_am_slots(access, 32, tfidf_ranker, c, step=7, cache=cache)
    s = sel(mask)
    for l in range(n_layers):
        assert len(s[l]) == 32, (mode, l, len(s[l]))
    assert torch.isfinite(info.tfidf).all()
    assert info.tfidf.shape == (n_layers, n_slots)
    assert mask.top_t == 32 and mask.granularity == "per_layer"
    assert info.step == 7
    ov = overlap(s, incumbent, 32)
    # realised trade axes, DIAG-IMPORTANCE's `selector_tradeoff_table` definition
    idx = np.array([sorted(s[l]) for l in range(n_layers)])
    mt = z["score_tf_mass_mt"]
    mt_frac = float(np.mean([mt[l, idx[l]].sum() / mt[l].sum() for l in range(n_layers)]))
    fi_frac = float(
        np.mean([ref_fisher[l, idx[l]].sum() / ref_fisher[l].sum() for l in range(n_layers)])
    )
    results[mode] = {
        "top32_overlap_vs_incumbent": ov,
        "frac_writable_MT_routing_mass": mt_frac,
        "frac_total_QA_Fisher_mass": fi_frac,
    }
    print(f"[3] {mode:>18}: overlap_vs_incumbent={ov:.4f}  "
          f"MT-mass={mt_frac:.4f}  QA-Fisher={fi_frac:.4f}")

OUT["top32_selections"] = results
# DIAG-IMPORTANCE: most-redundant 10.7%, lowest-Fisher 0.09% overlap with incumbent
assert abs(results["redundancy"]["top32_overlap_vs_incumbent"] - 0.107) < 0.01
assert results["fisher"]["top32_overlap_vs_incumbent"] < 0.01

# --------------------------------------------------------- 5. alpha degeneracy
m0, _ = rank_am_slots(
    access, 32, tfidf_ranker,
    cfg(slot_selection="mass_x_redundancy", mass_redundancy_alpha=0.0),
    step=1, cache=cache,
)
m_am, _ = _rank_attention_mass_per_layer(access, 32)
m1, _ = rank_am_slots(
    access, 32, tfidf_ranker,
    cfg(slot_selection="mass_x_redundancy", mass_redundancy_alpha=1.0),
    step=1, cache=cache,
)
m_red, _ = rank_am_slots(access, 32, tfidf_ranker, cfg(slot_selection="redundancy"),
                         step=1, cache=cache)
a0 = overlap(sel(m0), sel(m_am), 32)
a1 = overlap(sel(m1), sel(m_red), 32)
OUT["alpha_degeneracy"] = {"alpha0_vs_attention_mass": a0, "alpha1_vs_redundancy": a1}
print(f"[5] alpha=0 vs attention_mass: {a0:.4f}   alpha=1 vs redundancy: {a1:.4f}")
assert a0 == 1.0 and a1 == 1.0

# ------------------------------------------------- 6. default path is untouched
mask_t, info_t = rank_am_slots(access, 32, tfidf_ranker, cfg(), step=3, cache=cache)
mask_t2, _ = tfidf_ranker.rank_positions(access, top_t=32, step=3, return_info=True)
assert overlap(sel(mask_t), sel(mask_t2), 32) == 1.0
# passing cache= must not change the default selection at all
mask_t3, _ = rank_am_slots(access, 32, tfidf_ranker, cfg(), step=3)
assert overlap(sel(mask_t), sel(mask_t3), 32) == 1.0
h = lambda m: json.dumps({l: sorted(sel(m)[l]) for l in range(n_layers)})  # noqa: E731
OUT["default_tfidf_unchanged"] = True
OUT["default_tfidf_selection_sha"] = __import__("hashlib").sha256(
    h(mask_t).encode()
).hexdigest()[:32]
print(f"[6] default tfidf selection unchanged; sha={OUT['default_tfidf_selection_sha']}")

# ---------------------------------------------------------- 7. fails are loud
def expect(exc, fn, label):
    try:
        fn()
    except exc as e:
        print(f"[7] {label}: raised {type(e).__name__}: {str(e)[:90]}")
        return True
    raise AssertionError(f"{label} did NOT raise {exc}")


expect(ValueError, lambda: rank_am_slots(
    access, 32, tfidf_ranker,
    cfg(slot_selection="mass_x_redundancy", mass_redundancy_alpha=1.5),
    cache=cache), "alpha out of range")
expect(ValueError, lambda: rank_am_slots(
    access, 32, tfidf_ranker, cfg(slot_selection="redundancy")), "cache=None")
expect(ValueError, lambda: rank_am_slots(
    access, 32, tfidf_ranker,
    cfg(slot_selection="redundancy", granularity="per_head"), cache=cache),
    "granularity != per_layer")
expect(ValueError, lambda: rank_am_slots(
    access, 32, tfidf_ranker, cfg(slot_selection="fisher"), cache=cache),
    "fisher without a path")
expect(FileNotFoundError, lambda: rank_am_slots(
    access, 32, tfidf_ranker,
    cfg(slot_selection="fisher", slot_fisher_path="/tmp/nope.npz"), cache=cache),
    "fisher path missing")
expect(ValueError, lambda: load_slot_fisher_scores(FISHER, 12, n_slots),
       "fisher wrong shape")
expect(ValueError, lambda: compute_slot_redundancy(cache, ridge_rel=0.0),
       "ridge_rel <= 0")

# ------------------------------------------------- top_t sweep (the reverse trade)
sweep = {}
mt = z["score_tf_mass_mt"]
for t in (32, 64, 128):
    for mode in ("redundancy", "mass_x_redundancy"):
        c = cfg(slot_selection=mode, top_t=t)
        m, _ = rank_am_slots(access, t, tfidf_ranker, c, cache=cache)
        s = sel(m)
        idx = np.array([sorted(s[l]) for l in range(n_layers)])
        sweep[f"{mode}_t{t}"] = {
            "frac_writable_MT_routing_mass": float(
                np.mean([mt[l, idx[l]].sum() / mt[l].sum() for l in range(n_layers)])),
            "frac_total_QA_Fisher_mass": float(
                np.mean([ref_fisher[l, idx[l]].sum() / ref_fisher[l].sum()
                         for l in range(n_layers)])),
        }
    # the incumbent at the same budget, for reference
    inc = {l: set(np.argsort(-ranker_tf_mt[l])[:t].tolist()) for l in range(n_layers)}
    idx = np.array([sorted(inc[l]) for l in range(n_layers)])
    sweep[f"incumbent_t{t}"] = {
        "frac_writable_MT_routing_mass": float(
            np.mean([mt[l, idx[l]].sum() / mt[l].sum() for l in range(n_layers)])),
        "frac_total_QA_Fisher_mass": float(
            np.mean([ref_fisher[l, idx[l]].sum() / ref_fisher[l].sum()
                     for l in range(n_layers)])),
    }
OUT["predicted_tradeoff_by_top_t"] = sweep
print("\n[8] predicted trade (from DIAG-IMPORTANCE arrays, before any GPU run):")
for k, v in sweep.items():
    print(f"    {k:>26}  MT-mass={v['frac_writable_MT_routing_mass']:.4f}  "
          f"QA-Fisher={v['frac_total_QA_Fisher_mass']:.4f}")

with open(os.path.join(REPO, "research_loop/results/MECH-INFOGATE/sanity.json"), "w") as fh:
    json.dump(OUT, fh, indent=2)
print("\nALL SANITY CHECKS PASSED")
