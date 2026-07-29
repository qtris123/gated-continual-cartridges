"""DIAG-NOISE: bootstrap the sampling distribution of the reported metric and re-score
the board's load-bearing claims against the MEASURED resolution."""
import json
import numpy as np

P = json.load(open("/tmp/diag_noise/perexample_all.json"))
P3 = json.load(open("/tmp/diag_noise/perexample_step16.json"))
RES = dict(P["results"])
RES.update(P3["results"])
PAPERS = json.load(open("/tmp/diag_noise/paper_ids.json"))

rng = np.random.default_rng(20260729)
B = 200_000

CLUST = {s: np.array(PAPERS[s]) for s in ["MT", "QA"]}
UNIQ = {s: sorted(set(CLUST[s].tolist())) for s in CLUST}
CIDX = {s: np.array([UNIQ[s].index(p) for p in CLUST[s]]) for s in CLUST}


def arr(label, split):
    r = RES[label][split]
    return np.array(r["per_example_loss"], float), np.array(r["per_example_tokens"], float)


def dm_se(c, t):
    """delta-method SE of the ratio estimator R = sum(c)/sum(t) under iid unit sampling."""
    n = len(c)
    T = t.sum()
    R = c.sum() / T
    return float(np.sqrt(np.sum((c - R * t) ** 2) / (n - 1)) * np.sqrt(n) / T)


def collapse(c, t, cidx, G):
    C = np.bincount(cidx, weights=c, minlength=G)
    T = np.bincount(cidx, weights=t, minlength=G)
    return C, T


def boot_ratio(c, t, B, rng):
    n = len(c)
    idx = rng.integers(0, n, size=(B, n))
    return c[idx].sum(1) / t[idx].sum(1)


def summarize(vals, point):
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return {
        "point": float(point),
        "bootstrap_se": float(vals.std(ddof=1)),
        "ci95_lo": float(lo),
        "ci95_hi": float(hi),
        "ci95_halfwidth": float((hi - lo) / 2),
        "excludes_zero": bool((lo > 0) or (hi < 0)),
    }


# ---------------------------------------------------------------- single arms --
single = {}
for split in ["MT", "QA"]:
    n = len(CLUST[split])
    G = len(UNIQ[split])
    for label in RES:
        if split not in RES[label]:
            continue
        l, t = arr(label, split)
        c = l * t
        pt = c.sum() / t.sum()
        vals = boot_ratio(c, t, 50_000, rng)
        # cluster (by paper) bootstrap
        Cc, Tc = collapse(c, t, CIDX[split], G)
        valsc = boot_ratio(Cc, Tc, 50_000, rng)
        # unweighted macro mean over examples
        macro = float(l.mean())
        macro_se = float(l.std(ddof=1) / np.sqrt(n))
        single[f"{label}|{split}"] = {
            "n_examples": int(n),
            "n_scored_tokens": int(t.sum()),
            "token_weighted_loss": float(pt),
            "question_bootstrap": summarize(vals, pt),
            "paper_cluster_bootstrap": summarize(valsc, pt),
            "delta_method_se": dm_se(c, t),
            "delta_method_se_clustered": dm_se(Cc, Tc),
            "unweighted_macro_mean": macro,
            "unweighted_macro_se": macro_se,
            "weighting_shift_macro_minus_micro": macro - float(pt),
            "per_example_loss_sd": float(l.std(ddof=1)),
        }

# ------------------------------------------------------------------- claims ----
CLAIMS = [
    # (id, label_A, label_B, split, board_delta, board_verdict, description)
    ("MECH-KEYS_dMT_k16", "keys_k16", "ctrl_k16", "MT", -0.199, "clears (top edge)",
     "keys+reposition vs frozen-key control, both at k=16"),
    ("MECH-KEYS_dQA_k16", "keys_k16", "ctrl_k16", "QA", -0.125, "inside band",
     "keys+reposition vs frozen-key control, both at k=16"),
    ("MECH-KEYS_dMT_k12matched", "keys_k12", "ctrl_k12", "MT", -0.198, "clears",
     "matched-k comparison at k=12 (DIAG-KEYCURVE correction table)"),
    ("MECH-KEYS_dQA_k12matched", "keys_k12", "ctrl_k12", "QA", -0.118, "inside band",
     "matched-k comparison at k=12"),
    ("DIAG-CONTROLCURVE_dMT_optima", "keys_k12", "ctrl_k10", "MT", -0.1363, "INSIDE band -> REFUTED",
     "each arm at its own MT optimum (keys k=12 vs control k=10) -- the refutation"),
    ("DIAG-CONTROLCURVE_dQA_optima", "keys_k12", "ctrl_k10", "QA", -0.0115, "inside band",
     "QA at the two MT-optimal points"),
    ("DIAG-KEYCURVE_dMT_edge", "keys_k12", "ctrl_k8", "MT", -0.1816, "clears",
     "keys k=12 vs control pinned at the EDGE point k=8 (the superseded comparison)"),
    ("DIAG-KEYCURVE_dQA_edge", "keys_k12", "ctrl_k8", "QA", +0.0230, "inside band",
     "QA at those two points (control BETTER)"),
    ("keys_k12_vs_k16_MT", "keys_k12", "keys_k16", "MT", -0.0585, "inside band",
     "does the keys arm's own k=12 optimum beat its k=16 endpoint?"),
    ("keys_k12_vs_k16_QA", "keys_k12", "keys_k16", "QA", -0.0789, "inside band",
     "same, QA axis"),
    ("DIAG-SEQUENCE_tailrise_MT", "seq1e4_k16", "seq1e4_k12", "MT", +0.1172, "inside band",
     "value-only theta=1e4: MT gets WORSE from k=12 to k=16"),
    ("DIAG-SEQUENCE_tailrise_QA", "seq1e4_k16", "seq1e4_k12", "QA", +0.1350, "inside band",
     "same, QA axis (board quotes +0.153 from its k=13 min)"),
    ("MECH-BETA_dMT", "betaOn_k16", "betaRope_k16", "MT", +0.282, "far outside",
     "beta ON vs beta OFF at theta=5e6, k=16"),
    ("MECH-BETA_dQA", "betaOn_k16", "betaRope_k16", "QA", +0.581, "far outside",
     "same, QA axis"),
    ("DIAG-CONTENT_gap_k4", "contentB_k4", "seq1e4_k4", "MT", +0.232, "outside",
     "content-free arm B minus canonical, matched k=4"),
    ("DIAG-CONTENT_gap_k8", "contentB_k8", "seq1e4_k8", "MT", +0.288, "outside",
     "matched k=8"),
    ("DIAG-CONTENT_gap_k12", "contentB_k12", "seq1e4_k12", "MT", +0.608, "outside",
     "matched k=12"),
    ("DIAG-CONTENT_gap_k16", "contentB_k16", "seq1e4_k16", "MT", +0.882, "outside",
     "matched k=16"),
    ("DIAG-CONTENT_armC_sequence", "contentC_k16", "seq1e4_k16", "MT", +0.340, "outside",
     "reversed document order, content held constant"),
    ("MECH-SEQUENTIAL_k12_MT", "onpolkeys_k12", "keys_k12", "MT", +0.176, "the only delta escaping noise",
     "on-policy layer-sequential composed with keys, at k=12"),
    ("MECH-SEQUENTIAL_k12_QA", "onpolkeys_k12", "keys_k12", "QA", +0.188, "escapes noise",
     "same, QA axis"),
    ("DIAG-ROPE_dMT", "rope5e6_k16", "rope1e4_k16", "MT", -0.019, "inside band",
     "theta=5e6 vs theta=1e4 at k=16 (standalone evals of the byte-identical caches)"),
    ("DIAG-ROPE_dQA", "rope5e6_k16", "rope1e4_k16", "QA", -0.023, "inside band",
     "same, QA axis"),
    ("keys_k16_vs_ORACLEWRITE_proxy", "keys_k16", "ctrl_k10", "MT", None, "n/a",
     "keys endpoint vs control optimum (context row)"),
]

claims = {}
for cid, la, lb, split, board_delta, board_verdict, desc in CLAIMS:
    if la not in RES or lb not in RES:
        claims[cid] = {"error": f"missing {la} or {lb}"}
        continue
    la_l, t = arr(la, split)
    lb_l, t2 = arr(lb, split)
    assert np.array_equal(t, t2), "token counts must match across arms"
    n = len(t)
    G = len(UNIQ[split])
    d = la_l - lb_l
    delta = float((t * d).sum() / t.sum())

    idx = rng.integers(0, n, size=(B, n))
    tt = t[idx]
    paired_vals = (tt * d[idx]).sum(1) / tt.sum(1)

    # independent (unpaired) resampling of the two arms
    idxa = rng.integers(0, n, size=(50_000, n))
    idxb = rng.integers(0, n, size=(50_000, n))
    ca, cb = la_l * t, lb_l * t
    unp = ca[idxa].sum(1) / t[idxa].sum(1) - cb[idxb].sum(1) / t[idxb].sum(1)

    # cluster-by-paper paired bootstrap
    Cd, Tc = collapse(t * d, t, CIDX[split], G)
    idxg = rng.integers(0, G, size=(50_000, G))
    clust_vals = Cd[idxg].sum(1) / Tc[idxg].sum(1)

    # delta-method paired SE + double bootstrap on that SE
    se_dm = dm_se(t * d, t)
    outer = rng.integers(0, n, size=(2000, n))
    se_dist = np.array([dm_se((t * d)[o], t[o]) for o in outer])

    claims[cid] = {
        "description": desc,
        "arm_A": la, "arm_B": lb, "split": split,
        "board_delta": board_delta,
        "board_verdict_vs_pm0.15": board_verdict,
        "measured_delta": delta,
        "paired": summarize(paired_vals, delta),
        "paired_delta_method_se": se_dm,
        "paired_se_95ci": [float(np.percentile(se_dist, 2.5)), float(np.percentile(se_dist, 97.5))],
        "unpaired": summarize(unp, delta),
        "paper_cluster_paired": summarize(clust_vals, delta),
        "per_example_corr_between_arms": float(np.corrcoef(la_l, lb_l)[0, 1]),
        "n_examples_favouring_A": int((d < 0).sum()),
        "n_examples": int(n),
        "unweighted_macro_delta": float(d.mean()),
        "weighting_effect_on_delta": float(d.mean() - delta),
        "clears_measured_resolution_paired": bool(
            paired_vals.min() is not None and (np.percentile(paired_vals, 2.5) > 0
                                               or np.percentile(paired_vals, 97.5) < 0)),
        "clears_measured_resolution_paper_clustered": bool(
            np.percentile(clust_vals, 2.5) > 0 or np.percentile(clust_vals, 97.5) < 0),
        "clears_measured_resolution_unpaired": bool(
            np.percentile(unp, 2.5) > 0 or np.percentile(unp, 97.5) < 0),
        "clears_pm0.15_rule": (abs(delta) > 0.15),
    }

# ------------------------------------------------------- DIAG-PERDOC re-score --
DOCS = {"doc000": "1910.11471", "doc009": "2002.02427", "doc011": "1903.03467",
        "doc013": "1909.01013", "doc015": "1910.10408"}
perdoc = {}
lk16, tmt = arr("seq1e4_k16", "MT")
lk12, _ = arr("seq1e4_k12", "MT")
lph1, _ = arr("phase1_k0", "MT")
mtp = CLUST["MT"]
all_mask = np.zeros(len(mtp), bool)
pooled_num, pooled_den = 0.0, 0.0
for dkey, pid in DOCS.items():
    solo_l, _ = arr(f"solo_A_{dkey}", "MT")
    m = (mtp == pid)
    all_mask |= m
    tt = tmt[m]
    d = (solo_l - lk16)[m]
    delta = float((tt * d).sum() / tt.sum())
    nb = int(m.sum())
    idx = rng.integers(0, nb, size=(50_000, nb))
    vals = (tt[idx] * d[idx]).sum(1) / tt[idx].sum(1)
    perdoc[dkey] = {
        "paper_id": pid, "n_questions": nb, "n_tokens": int(tt.sum()),
        "solo_loss": float((solo_l[m] * tt).sum() / tt.sum()),
        "k16_loss": float((lk16[m] * tt).sum() / tt.sum()),
        "k12_loss": float((lk12[m] * tt).sum() / tt.sum()),
        "phase1_loss": float((lph1[m] * tt).sum() / tt.sum()),
        "paired_delta_solo_minus_k16": delta,
        "paired": summarize(vals, delta),
        "n_questions_favouring_solo": int((d < 0).sum()),
    }
    pooled_num += float((tt * d).sum())
    pooled_den += float(tt.sum())

tt = tmt[all_mask]
d = (np.concatenate([arr(f"solo_A_{k}", "MT")[0][mtp == p] for k, p in DOCS.items()])
     - np.concatenate([lk16[mtp == p] for p in DOCS.values()]))
tt_ord = np.concatenate([tmt[mtp == p] for p in DOCS.values()])
delta = float((tt_ord * d).sum() / tt_ord.sum())
nb = len(d)
idx = rng.integers(0, nb, size=(B, nb))
vals = (tt_ord[idx] * d[idx]).sum(1) / tt_ord[idx].sum(1)
# cluster by document (5 clusters)
dclust = np.concatenate([np.full((mtp == p).sum(), i) for i, p in enumerate(DOCS.values())])
Cd = np.bincount(dclust, weights=tt_ord * d, minlength=5)
Tc = np.bincount(dclust, weights=tt_ord, minlength=5)
idxg = rng.integers(0, 5, size=(50_000, 5))
vals_c = Cd[idxg].sum(1) / Tc[idxg].sum(1)
perdoc["_pooled_5docs_27questions"] = {
    "board_value": 1.101, "board_se": 0.083,
    "measured_paired_delta_solo_minus_k16": delta,
    "paired": summarize(vals, delta),
    "document_cluster_bootstrap": summarize(vals_c, delta),
    "n_questions": nb,
    "n_questions_favouring_solo": int((d < 0).sum()),
}
# doc-015 LOO cross-check against DIAG-PERDOC's recovered per-example losses
loo_ref = json.load(open("/localhome/local-triv/gated-continual-cartridges_explore/"
                        "research_loop/state/diagnostics/DIAG-PERDOC.json"))["leave_one_out_uncertainty"]
m15 = (mtp == DOCS["doc015"])
ours_solo = arr("solo_A_doc015", "MT")[0][m15]
ours_k16 = lk16[m15]
ref_solo = np.array(loo_ref["per_example_loss"]["A_doc015"])
ref_k16 = np.array(loo_ref["per_example_loss"]["K16A"])
perdoc["_loo_crosscheck_doc015"] = {
    "note": ("DIAG-PERDOC recovered these 7 per-example losses by leave-one-out over eval subsets; "
             "we recover them directly from the full-MT forward pass. Orders may differ, so both "
             "sorted and unsorted comparisons are given."),
    "ours_solo_sorted": sorted(ours_solo.tolist()),
    "diagperdoc_solo_sorted": sorted(ref_solo.tolist()),
    "max_abs_diff_solo_sorted": float(np.max(np.abs(np.sort(ours_solo) - np.sort(ref_solo)))),
    "ours_k16_sorted": sorted(ours_k16.tolist()),
    "diagperdoc_k16_sorted": sorted(ref_k16.tolist()),
    "max_abs_diff_k16_sorted": float(np.max(np.abs(np.sort(ours_k16) - np.sort(ref_k16)))),
}

# --------------------------------------------- non-sampling variance components --
nonsampling = {
    "in_run_vs_standalone_eval_of_a_BYTE_IDENTICAL_cartridge": {
        "theta1e4_k16": {"in_run_published": [2.1766157150268555, 2.5483615398406982],
                         "standalone_measured": [RES["seq1e4_k16"]["QA"]["mean_ce"],
                                                 RES["seq1e4_k16"]["MT"]["mean_ce"]],
                         "abs_diff_QA": abs(2.1766157150268555 - RES["seq1e4_k16"]["QA"]["mean_ce"]),
                         "abs_diff_MT": abs(2.5483615398406982 - RES["seq1e4_k16"]["MT"]["mean_ce"])},
        "theta5e6_k16": {"in_run_published": [2.15320, 2.52961],
                         "standalone_measured": [RES["ctrl_k16"]["QA"]["mean_ce"],
                                                 RES["ctrl_k16"]["MT"]["mean_ce"]],
                         "abs_diff_QA": abs(2.15320 - RES["ctrl_k16"]["QA"]["mean_ce"]),
                         "abs_diff_MT": abs(2.52961 - RES["ctrl_k16"]["MT"]["mean_ce"])},
    },
    "reference_draw_replicate_doc015_solo": {
        "full_MT_draw1": RES["solo_A_doc015"]["MT"]["mean_ce"],
        "full_MT_draw2": RES["solo_A_doc015shuf"]["MT"]["mean_ce"],
        "abs_diff_full_MT": abs(RES["solo_A_doc015"]["MT"]["mean_ce"]
                                - RES["solo_A_doc015shuf"]["MT"]["mean_ce"]),
        "full_QA_draw1": RES["solo_A_doc015"]["QA"]["mean_ce"],
        "full_QA_draw2": RES["solo_A_doc015shuf"]["QA"]["mean_ce"],
        "abs_diff_full_QA": abs(RES["solo_A_doc015"]["QA"]["mean_ce"]
                                - RES["solo_A_doc015shuf"]["QA"]["mean_ce"]),
    },
}

out = {"single_arms": single, "claims": claims, "perdoc": perdoc,
       "non_sampling_variance": nonsampling,
       "bootstrap_B": B, "rng_seed": 20260729}
json.dump(out, open("/tmp/diag_noise/analysis.json", "w"), indent=1)

# ------------------------------------------------------------------- printout --
print("=== SINGLE-ARM RESOLUTION (token-weighted mean CE) ===")
print(f"{'arm|split':28s} {'loss':>9s} {'se_q':>7s} {'95%hw':>7s} {'se_paper':>8s} {'95%hw_p':>8s} {'macro-micro':>11s}")
for k, v in single.items():
    print(f"{k:28s} {v['token_weighted_loss']:9.4f} {v['question_bootstrap']['bootstrap_se']:7.4f} "
          f"{v['question_bootstrap']['ci95_halfwidth']:7.4f} "
          f"{v['paper_cluster_bootstrap']['bootstrap_se']:8.4f} "
          f"{v['paper_cluster_bootstrap']['ci95_halfwidth']:8.4f} "
          f"{v['weighting_shift_macro_minus_micro']:+11.4f}")

print()
print("=== CLAIM RE-SCORE ===")
hdr = (f"{'claim':34s} {'board':>8s} {'meas':>8s} {'se_pair':>8s} {'95%CI paired':>22s} "
       f"{'se_unp':>7s} {'se_clust':>8s} {'r':>6s} {'pair?':>6s} {'clust?':>7s} {'unp?':>5s} {'>0.15?':>6s}")
print(hdr)
for cid, v in claims.items():
    if "error" in v:
        print(cid, v["error"]); continue
    p = v["paired"]
    print(f"{cid:34s} {(v['board_delta'] if v['board_delta'] is not None else float('nan')):8.4f} "
          f"{v['measured_delta']:8.4f} {p['bootstrap_se']:8.4f} "
          f"[{p['ci95_lo']:+8.4f},{p['ci95_hi']:+8.4f}] "
          f"{v['unpaired']['bootstrap_se']:7.4f} {v['paper_cluster_paired']['bootstrap_se']:8.4f} "
          f"{v['per_example_corr_between_arms']:6.3f} "
          f"{str(v['clears_measured_resolution_paired']):>6s} "
          f"{str(v['clears_measured_resolution_paper_clustered']):>7s} "
          f"{str(v['clears_measured_resolution_unpaired']):>5s} "
          f"{str(v['clears_pm0.15_rule']):>6s}")

print()
print("=== DIAG-PERDOC ===")
for k, v in perdoc.items():
    if k.startswith("_loo"):
        print(k, "max|diff| solo", round(v["max_abs_diff_solo_sorted"], 6),
              "k16", round(v["max_abs_diff_k16_sorted"], 6))
    elif k.startswith("_"):
        print(k, json.dumps({kk: vv for kk, vv in v.items() if kk != "paired"}, default=str)[:400])
        print("   paired:", v["paired"])
    else:
        print(f"{k} n={v['n_questions']} delta={v['paired_delta_solo_minus_k16']:+.4f} "
              f"se={v['paired']['bootstrap_se']:.4f} CI=[{v['paired']['ci95_lo']:+.4f},{v['paired']['ci95_hi']:+.4f}]")

print()
print("=== NON-SAMPLING ===")
print(json.dumps(nonsampling, indent=1))
