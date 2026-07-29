"""DIAG-NOISE part 2: (a) argmin-selection-aware bootstrap of DIAG-CONTROLCURVE's
'each arm at its own optimum' comparison; (b) eval-packing jitter; (c) composite resolution."""
import json
import numpy as np

C = json.load(open("/tmp/diag_noise/perexample_curves.json"))["results"]
P = json.load(open("/tmp/diag_noise/perexample_all.json"))["results"]
PAPERS = json.load(open("/tmp/diag_noise/paper_ids.json"))
rng = np.random.default_rng(4242)
B = 100_000
out = {}


def arr(res, label, split):
    r = res[label][split]
    return np.array(r["per_example_loss"], float), np.array(r["per_example_tokens"], float)


# --- gate: the 16-point curves reproduce DIAG-CONTROLCURVE / DIAG-KEYCURVE ------
kc = json.load(open("/localhome/local-triv/gated-continual-cartridges_explore/"
                    "research_loop/state/diagnostics/DIAG-KEYCURVE.json"))
cc = json.load(open("/localhome/local-triv/gated-continual-cartridges_explore/"
                    "research_loop/state/diagnostics/DIAG-CONTROLCURVE.json"))
pub_keys_mt = {e["k"]: e["mt"] for e in kc["keys_repos_curve"]}
pub_keys_qa = {e["k"]: e["qa"] for e in kc["keys_repos_curve"]}
pub_ctrl_mt = {int(k): v for k, v in cc["control_mt_by_k"].items()}
pub_ctrl_qa = {int(k): v for k, v in cc["control_qa_by_k"].items()}

curves = {}
gate = {}
for arm, pubmt, pubqa in [("ctrlK", pub_ctrl_mt, pub_ctrl_qa), ("keysK", pub_keys_mt, pub_keys_qa)]:
    for split, pub in [("MT", pubmt), ("QA", pubqa)]:
        L, T, meas = [], None, []
        for k in range(1, 17):
            l, t = arr(C, f"{arm}{k:02d}", split)
            L.append(l)
            T = t
            meas.append(float((l * t).sum() / t.sum()))
        curves[(arm, split)] = (np.array(L), T)
        err = [abs(meas[k - 1] - pub[k]) for k in range(1, 17)]
        gate[f"{arm}_{split}"] = {"max_abs_err_vs_published": float(max(err)),
                                  "measured": meas, "published": [pub[k] for k in range(1, 17)]}
out["curve_gate"] = gate

# ------ selection-aware bootstrap: each arm's argmin re-selected on each resample --
sel = {}
for split in ["MT", "QA"]:
    Lc, T = curves[("ctrlK", split)]
    Lk, _ = curves[("keysK", split)]
    n = len(T)
    Cc = Lc * T          # 16 x n
    Ck = Lk * T
    idx = rng.integers(0, n, size=(B, n))
    Tb = T[idx].sum(1)                             # B
    ctrl_b = Cc[:, idx].sum(2) / Tb                # 16 x B
    keys_b = Ck[:, idx].sum(2) / Tb
    kmin_c = ctrl_b.argmin(0)
    kmin_k = keys_b.argmin(0)
    d_sel = keys_b[kmin_k, np.arange(B)] - ctrl_b[kmin_c, np.arange(B)]
    # fixed-k version (the point estimate's own argmins), for contrast
    k_c0 = (Cc.sum(1) / T.sum()).argmin()
    k_k0 = (Ck.sum(1) / T.sum()).argmin()
    d_fix = keys_b[k_k0] - ctrl_b[k_c0]
    point_sel = float((Ck[k_k0].sum() / T.sum()) - (Cc[k_c0].sum() / T.sum()))
    lo, hi = np.percentile(d_sel, [2.5, 97.5])
    lo2, hi2 = np.percentile(d_fix, [2.5, 97.5])
    sel[split] = {
        "control_argmin_k": int(k_c0 + 1), "keys_argmin_k": int(k_k0 + 1),
        "point_delta_at_own_optima": point_sel,
        "fixed_k_bootstrap": {"se": float(d_fix.std(ddof=1)), "ci95": [float(lo2), float(hi2)],
                              "excludes_zero": bool(lo2 > 0 or hi2 < 0)},
        "selection_aware_bootstrap": {
            "se": float(d_sel.std(ddof=1)), "ci95": [float(lo), float(hi)],
            "mean": float(d_sel.mean()),
            "excludes_zero": bool(lo > 0 or hi < 0),
            "selection_bias_mean_minus_point": float(d_sel.mean() - point_sel),
        },
        "control_argmin_distribution": {int(k + 1): int((kmin_c == k).sum()) for k in range(16)
                                        if (kmin_c == k).sum() > 0},
        "keys_argmin_distribution": {int(k + 1): int((kmin_k == k).sum()) for k in range(16)
                                     if (kmin_k == k).sum() > 0},
    }
out["argmin_selection"] = sel

# --------- per-k paired deltas (keys - control) with paired SE, all 16 k ---------
perk = {}
for split in ["MT", "QA"]:
    Lc, T = curves[("ctrlK", split)]
    Lk, _ = curves[("keysK", split)]
    n = len(T)
    idx = rng.integers(0, n, size=(20_000, n))
    Tb = T[idx].sum(1)
    rows = []
    for k in range(16):
        d = Lk[k] - Lc[k]
        pt = float((T * d).sum() / T.sum())
        vals = (T[idx] * d[idx]).sum(1) / Tb
        lo, hi = np.percentile(vals, [2.5, 97.5])
        rows.append({"k": k + 1, "delta": pt, "se": float(vals.std(ddof=1)),
                     "ci95": [float(lo), float(hi)],
                     "clears_measured": bool(lo > 0 or hi < 0),
                     "clears_pm015": bool(abs(pt) > 0.15)})
    perk[split] = rows
out["per_k_keys_minus_control"] = perk

# --------------------- eval-packing jitter (subset eval vs subset of full eval) ---
pd_ref = json.load(open("/localhome/local-triv/gated-continual-cartridges_explore/"
                        "research_loop/state/diagnostics/DIAG-PERDOC.json"))["per_document"]
DOCS = {"doc000": "1910.11471", "doc009": "2002.02427", "doc011": "1903.03467",
        "doc013": "1909.01013", "doc015": "1910.10408"}
mtp = np.array(PAPERS["MT"])
pack = []
lk16, tmt = arr(P, "seq1e4_k16", "MT")
lk12, _ = arr(P, "seq1e4_k12", "MT")
lph1, _ = arr(P, "phase1_k0", "MT")
for dkey, pid in DOCS.items():
    m = (mtp == pid)
    tt = tmt[m]
    solo, _ = arr(P, f"solo_A_{dkey}", "MT")
    ours = {
        "phase1_floor": float((lph1[m] * tt).sum() / tt.sum()),
        "solo_rope1e4": float((solo[m] * tt).sum() / tt.sum()),
        "k12_rope1e4": float((lk12[m] * tt).sum() / tt.sum()),
        "k16_rope1e4": float((lk16[m] * tt).sum() / tt.sum()),
    }
    theirs = pd_ref[dkey]["own_subset_loss"]
    for key in ours:
        pack.append({"doc": dkey, "quantity": key, "subset_eval": theirs[key],
                     "subset_of_full_eval": ours[key],
                     "abs_diff": abs(theirs[key] - ours[key])})
    # token count agreement
    pack.append({"doc": dkey, "quantity": "n_scored_tokens",
                 "subset_eval": pd_ref[dkey]["n_scored_tokens"],
                 "subset_of_full_eval": int(tt.sum()),
                 "abs_diff": abs(pd_ref[dkey]["n_scored_tokens"] - int(tt.sum()))})
diffs = [p["abs_diff"] for p in pack if p["quantity"] != "n_scored_tokens"]
out["eval_packing_jitter"] = {
    "rows": pack,
    "n": len(diffs), "median_abs_diff": float(np.median(diffs)),
    "max_abs_diff": float(max(diffs)), "mean_abs_diff": float(np.mean(diffs)),
    "what_it_is": ("the SAME questions scored by (a) a standalone eval of a filtered "
                   "parquet holding only that document's questions and (b) subsetting the "
                   "full 69-question eval. Same model, same cartridge, same tokens: the only "
                   "difference is how examples are packed into 2048-token sequences, which "
                   "changes bf16 reduction order."),
}

# --------------------------------- composite resolution -------------------------
ref_draw_MT = abs(P["solo_A_doc015"]["MT"]["mean_ce"] - P["solo_A_doc015shuf"]["MT"]["mean_ce"])
ref_draw_QA = abs(P["solo_A_doc015"]["QA"]["mean_ce"] - P["solo_A_doc015shuf"]["QA"]["mean_ce"])
A1 = json.load(open("/tmp/diag_noise/analysis.json"))
med_pair_mt = float(np.median([v["paired"]["bootstrap_se"] for v in A1["claims"].values()
                               if v.get("split") == "MT"]))
med_pair_qa = float(np.median([v["paired"]["bootstrap_se"] for v in A1["claims"].values()
                               if v.get("split") == "QA"]))
# treat the single observed replicate difference as |N(0, 2 sigma^2)| -> sigma ~ d/sqrt(2)
sig_draw_mt = ref_draw_MT / np.sqrt(2)
sig_draw_qa = ref_draw_QA / np.sqrt(2)
out["composite_resolution"] = {
    "eval_sampling_paired_se_median": {"MT": med_pair_mt, "QA": med_pair_qa},
    "reference_draw_replicate_abs_diff": {"MT": ref_draw_MT, "QA": ref_draw_QA},
    "implied_reference_draw_sd_of_a_difference": {"MT": float(ref_draw_MT), "QA": float(ref_draw_QA)},
    "implied_per_arm_sd_from_one_replicate": {"MT": float(sig_draw_mt), "QA": float(sig_draw_qa)},
    "composite_se_paired_plus_draw": {
        "MT": float(np.sqrt(med_pair_mt ** 2 + ref_draw_MT ** 2)),
        "QA": float(np.sqrt(med_pair_qa ** 2 + ref_draw_QA ** 2)),
    },
    "composite_95pct_resolution": {
        "MT": float(1.96 * np.sqrt(med_pair_mt ** 2 + ref_draw_MT ** 2)),
        "QA": float(1.96 * np.sqrt(med_pair_qa ** 2 + ref_draw_QA ** 2)),
    },
    "caveat": ("the reference-draw component rests on ONE replicate pair, measured on a "
               "SOLO (1-document) write, and it is a design-choice sensitivity rather than "
               "stochastic noise (the draw is deterministic given the corpus). It is reported "
               "as an upper-bound-ish extra term, not a measured variance."),
}

json.dump(out, open("/tmp/diag_noise/analysis2.json", "w"), indent=1)

print("=== curve gate (max |measured - published| over 16 k) ===")
for k, v in gate.items():
    print(f"  {k}: {v['max_abs_err_vs_published']:.3e}")
print()
print("=== argmin-selection-aware bootstrap (keys - control, each at its OWN optimum) ===")
for s, v in sel.items():
    print(f"{s}: control argmin k={v['control_argmin_k']}, keys argmin k={v['keys_argmin_k']}, "
          f"point Delta={v['point_delta_at_own_optima']:+.4f}")
    f = v["fixed_k_bootstrap"]
    print(f"   fixed-k      : se={f['se']:.4f} CI95=[{f['ci95'][0]:+.4f},{f['ci95'][1]:+.4f}] "
          f"excl0={f['excludes_zero']}")
    a = v["selection_aware_bootstrap"]
    print(f"   selection-aware: se={a['se']:.4f} CI95=[{a['ci95'][0]:+.4f},{a['ci95'][1]:+.4f}] "
          f"excl0={a['excludes_zero']} bias={a['selection_bias_mean_minus_point']:+.4f}")
    print(f"   control argmin dist: {v['control_argmin_distribution']}")
    print(f"   keys    argmin dist: {v['keys_argmin_distribution']}")
print()
print("=== per-k keys - control ===")
for s in ["MT", "QA"]:
    print(s)
    for r in perk[s]:
        print(f"  k={r['k']:2d} d={r['delta']:+.4f} se={r['se']:.4f} "
              f"CI=[{r['ci95'][0]:+.4f},{r['ci95'][1]:+.4f}] measured_clear={r['clears_measured']} "
              f"pm015_clear={r['clears_pm015']}")
print()
print("=== eval-packing jitter ===")
print(json.dumps({k: v for k, v in out["eval_packing_jitter"].items() if k != "rows"}, indent=1))
for p in out["eval_packing_jitter"]["rows"]:
    print(f"  {p['doc']:7s} {p['quantity']:14s} subset_eval={p['subset_eval']:.6f} "
          f"full_eval_subset={p['subset_of_full_eval']:.6f} |d|={p['abs_diff']:.6f}")
print()
print("=== composite ===")
print(json.dumps(out["composite_resolution"], indent=1))
