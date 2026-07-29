"""DIAG-NOISE: assemble the two deliverable bundles."""
import json
import subprocess
import datetime
import numpy as np

REPO = "/localhome/local-triv/gated-continual-cartridges_explore"
A1 = json.load(open("/tmp/diag_noise/analysis.json"))
A2 = json.load(open("/tmp/diag_noise/analysis2.json"))
PX = json.load(open("/tmp/diag_noise/perexample_all.json"))
PX3 = json.load(open("/tmp/diag_noise/perexample_step16.json"))
PXC = json.load(open("/tmp/diag_noise/perexample_curves.json"))

claims = A1["claims"]
single = A1["single_arms"]
comp = A2["composite_resolution"]
RES_MT_PAIRED_95 = 1.96 * comp["eval_sampling_paired_se_median"]["MT"]
RES_QA_PAIRED_95 = 1.96 * comp["eval_sampling_paired_se_median"]["QA"]
COMP_MT = comp["composite_95pct_resolution"]["MT"]
COMP_QA = comp["composite_95pct_resolution"]["QA"]


def band(split):
    return (RES_MT_PAIRED_95, COMP_MT) if split == "MT" else (RES_QA_PAIRED_95, COMP_QA)


table = []
for cid, v in claims.items():
    if "error" in v or cid.endswith("proxy"):
        continue
    s = v["split"]
    r_pair, r_comp = band(s)
    d = v["measured_delta"]
    table.append({
        "claim": cid,
        "description": v["description"],
        "split": s,
        "board_delta": v["board_delta"],
        "board_verdict_vs_pm0.15": v["board_verdict_vs_pm0.15"],
        "measured_delta": d,
        "paired_se": v["paired"]["bootstrap_se"],
        "paired_ci95": [v["paired"]["ci95_lo"], v["paired"]["ci95_hi"]],
        "paired_z": d / v["paired"]["bootstrap_se"],
        "paper_cluster_se": v["paper_cluster_paired"]["bootstrap_se"],
        "unpaired_se": v["unpaired"]["bootstrap_se"],
        "n_examples_favouring_A": v["n_examples_favouring_A"],
        "n_examples": v["n_examples"],
        "verdict_vs_pm0.15_rule": "clears" if abs(d) > 0.15 else "INSIDE (called noise)",
        "verdict_vs_measured_paired": ("RESOLVED" if v["clears_measured_resolution_paired"]
                                       else "not resolvable"),
        "verdict_vs_paper_clustered": ("RESOLVED" if v["clears_measured_resolution_paper_clustered"]
                                       else "not resolvable"),
        "verdict_vs_composite_incl_pipeline": ("RESOLVED" if abs(d) > r_comp else "not resolvable"),
        "adjudicated_on_wrong_side_by_pm0.15": bool((abs(d) > 0.15) != v["clears_measured_resolution_paired"]),
    })

n_wrong = sum(1 for t in table if t["adjudicated_on_wrong_side_by_pm0.15"])
n_inside_015 = sum(1 for t in table if abs(t["measured_delta"]) <= 0.15)
n_inside_meas = sum(1 for t in table if t["verdict_vs_measured_paired"] != "RESOLVED")
n_inside_comp = sum(1 for t in table if t["verdict_vs_composite_incl_pipeline"] != "RESOLVED")

perk = A2["per_k_keys_minus_control"]
perk_counts = {
    s: {"n_clearing_pm0.15": sum(1 for r in perk[s] if r["clears_pm015"]),
        "n_clearing_measured": sum(1 for r in perk[s] if r["clears_measured"]),
        "n_k": len(perk[s])}
    for s in perk
}

head = subprocess.run(["git", "-C", REPO, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
manifest = subprocess.run(
    "cd /tmp/amsnap_DIAG-NOISE && find cartridges examples -type f -name '*.py' | LC_ALL=C sort "
    "| xargs sha256sum | awk '{print $1\"  \"$2}' | LC_ALL=C sort | sha256sum",
    shell=True, capture_output=True, text=True).stdout.split()[0]
dirty = subprocess.run(["git", "-C", REPO, "status", "--porcelain"], capture_output=True,
                       text=True).stdout.strip()

WANDB = {
    "pass1_13ckpt": "https://wandb.ai/vqtri-purdue-university/SEACrowd/runs/i7ae5gnw",
    "pass2_28ckpt": "https://wandb.ai/vqtri-purdue-university/SEACrowd/runs/eu5uy2bq",
    "pass3_step16_files": "https://wandb.ai/vqtri-purdue-university/SEACrowd/runs/xqb0ptbq",
    "pass4_full_k_curves_32ckpt": "https://wandb.ai/vqtri-purdue-university/SEACrowd/runs/u54zfh9p",
}

diag = {
    "id": "DIAG-NOISE",
    "board_entry": "methodology (applies to every claim in the investigation)",
    "question": ("What is the eval's actual resolution? Estimate the sampling distribution of the "
                 "reported token-weighted mean CE from exactly-recovered per-example losses, and "
                 "re-score every load-bearing claim against the MEASURED resolution instead of the "
                 "inherited +-0.1-0.2 rule of thumb."),

    "headline": {
        "the_reported_metric_is": ("token-weighted (micro) mean cross-entropy: L = sum_e c_e / sum_e t_e, "
                                   "c_e = summed CE over example e's scored assistant tokens, t_e its token "
                                   "count. MT: 69 examples / 2562 scored tokens. QA: 78 / 2619."),
        "recovery_method": ("direct per-example decomposition of the harness's own arithmetic: one forward "
                            "pass per packed batch, ce_by_token bucketed by element id. NOT leave-one-out: "
                            "45 checkpoints x 2 splits reproduce the published standalone `Eval loss` to "
                            "<= 4.0e-07 (max over 90 evals), so the decomposition is exact, not approximate."),
        "resolution_MT_paired_95pct": RES_MT_PAIRED_95,
        "resolution_QA_paired_95pct": RES_QA_PAIRED_95,
        "resolution_MT_single_arm_95pct": float(np.median(
            [v["question_bootstrap"]["ci95_halfwidth"] for k, v in single.items() if k.endswith("|MT")])),
        "resolution_QA_single_arm_95pct": float(np.median(
            [v["question_bootstrap"]["ci95_halfwidth"] for k, v in single.items() if k.endswith("|QA")])),
        "resolution_MT_unpaired_diff_95pct": 1.96 * float(np.median(
            [v["unpaired"]["bootstrap_se"] for v in claims.values() if v.get("split") == "MT"])),
        "resolution_QA_unpaired_diff_95pct": 1.96 * float(np.median(
            [v["unpaired"]["bootstrap_se"] for v in claims.values() if v.get("split") == "QA"])),
        "composite_MT_95pct_incl_one_pipeline_replicate": COMP_MT,
        "composite_QA_95pct_incl_one_pipeline_replicate": COMP_QA,
        "one_line": (f"the +-0.15 band is ~{0.15/RES_MT_PAIRED_95:.1f}x TOO WIDE for the paired "
                     f"comparisons the board actually makes (measured MT resolution +-"
                     f"{RES_MT_PAIRED_95:.3f} at 95%) and ~"
                     f"{1.96*float(np.median([v['unpaired']['bootstrap_se'] for v in claims.values() if v.get('split')=='MT']))/0.15:.1f}x "
                     f"TOO NARROW for an independent one. Every board claim is paired."),
        "which_the_board_should_have_used": ("PAIRED. Both arms are scored on the SAME 69/78 examples with "
                                             "byte-identical token counts (verified: t_e arrays are equal "
                                             "across every arm), so the difference is a within-subject "
                                             "contrast. Per-example losses correlate r = 0.92-0.99 between "
                                             "arms, which is exactly why the paired SE is 4-12x smaller "
                                             "than the independent one."),
    },

    "resolution_table": {
        "MT_n69": {
            "single_arm_bootstrap_se": {"min": min(v["question_bootstrap"]["bootstrap_se"] for k, v in single.items() if k.endswith("|MT")),
                                        "median": float(np.median([v["question_bootstrap"]["bootstrap_se"] for k, v in single.items() if k.endswith("|MT")])),
                                        "max": max(v["question_bootstrap"]["bootstrap_se"] for k, v in single.items() if k.endswith("|MT"))},
            "single_arm_se_clustered_by_paper16": float(np.median([v["paper_cluster_bootstrap"]["bootstrap_se"] for k, v in single.items() if k.endswith("|MT")])),
            "paired_diff_se": {"min": min(v["paired"]["bootstrap_se"] for v in claims.values() if v.get("split") == "MT"),
                               "median": float(np.median([v["paired"]["bootstrap_se"] for v in claims.values() if v.get("split") == "MT"])),
                               "max": max(v["paired"]["bootstrap_se"] for v in claims.values() if v.get("split") == "MT")},
            "paired_diff_se_clustered_by_paper16": float(np.median([v["paper_cluster_paired"]["bootstrap_se"] for v in claims.values() if v.get("split") == "MT"])),
            "independent_diff_se": float(np.median([v["unpaired"]["bootstrap_se"] for v in claims.values() if v.get("split") == "MT"])),
        },
        "QA_n78": {
            "single_arm_bootstrap_se": {"min": min(v["question_bootstrap"]["bootstrap_se"] for k, v in single.items() if k.endswith("|QA")),
                                        "median": float(np.median([v["question_bootstrap"]["bootstrap_se"] for k, v in single.items() if k.endswith("|QA")])),
                                        "max": max(v["question_bootstrap"]["bootstrap_se"] for k, v in single.items() if k.endswith("|QA"))},
            "single_arm_se_clustered_by_paper16": float(np.median([v["paper_cluster_bootstrap"]["bootstrap_se"] for k, v in single.items() if k.endswith("|QA")])),
            "paired_diff_se": {"min": min(v["paired"]["bootstrap_se"] for v in claims.values() if v.get("split") == "QA"),
                               "median": float(np.median([v["paired"]["bootstrap_se"] for v in claims.values() if v.get("split") == "QA"])),
                               "max": max(v["paired"]["bootstrap_se"] for v in claims.values() if v.get("split") == "QA")},
            "paired_diff_se_clustered_by_paper16": float(np.median([v["paper_cluster_paired"]["bootstrap_se"] for v in claims.values() if v.get("split") == "QA"])),
            "independent_diff_se": float(np.median([v["unpaired"]["bootstrap_se"] for v in claims.values() if v.get("split") == "QA"])),
        },
        "uncertainty_of_the_resolution_itself": {
            "method": "double bootstrap: 2000 outer example-resamples, delta-method paired SE on each",
            "examples": {cid: {"se": claims[cid]["paired"]["bootstrap_se"],
                               "se_95ci": claims[cid]["paired_se_95ci"]}
                         for cid in ["DIAG-CONTROLCURVE_dMT_optima", "MECH-KEYS_dMT_k16",
                                     "keys_k12_vs_k16_MT", "DIAG-SEQUENCE_tailrise_MT"]},
            "reading": "the SE is itself uncertain by roughly +-25%; the resolution is +-0.04 to +-0.07 MT, not a point.",
        },
        "token_weighting": {
            "macro_minus_micro_single_arm_MT_median": float(np.median([v["weighting_shift_macro_minus_micro"] for k, v in single.items() if k.endswith("|MT")])),
            "macro_minus_micro_single_arm_QA_median": float(np.median([v["weighting_shift_macro_minus_micro"] for k, v in single.items() if k.endswith("|QA")])),
            "effect_on_a_DIFFERENCE_MT_median": float(np.median([abs(v["weighting_effect_on_delta"]) for v in claims.values() if v.get("split") == "MT"])),
            "effect_on_a_DIFFERENCE_MT_max": float(max(abs(v["weighting_effect_on_delta"]) for v in claims.values() if v.get("split") == "MT")),
            "reading": ("token weighting moves an ARM's reported loss by 0.03-0.18 relative to an "
                        "unweighted per-question mean (always upward: long answers are harder), i.e. by "
                        "as much as the whole old noise band. It moves a DIFFERENCE by a median 0.006 and "
                        "at most 0.058, so it is nearly irrelevant to every comparison the board makes."),
        },
    },

    "rescored_claims": table,
    "rescore_counts": {
        "n_claims_scored": len(table),
        "n_inside_the_old_pm0.15_rule": n_inside_015,
        "n_not_resolvable_at_the_MEASURED_paired_resolution": n_inside_meas,
        "n_not_resolvable_under_the_composite_resolution": n_inside_comp,
        "n_adjudicated_on_the_WRONG_side_by_pm0.15": n_wrong,
        "which_flipped": [t["claim"] for t in table if t["adjudicated_on_wrong_side_by_pm0.15"]],
    },

    "argmin_selection_test": A2["argmin_selection"],
    "per_k_keys_minus_control": perk,
    "per_k_counts": perk_counts,
    "diag_perdoc_rescore": A1["perdoc"],

    "non_sampling_variance_components": {
        "run_to_run_variance_of_the_SOLVE_is_ZERO": {
            "evidence": ("the theta=1e4 k=16 cartridge is BYTE-IDENTICAL across 3 independent runs "
                         "(sha256 581d8e9b... for cache-step16.pt of f705f46c and 4b0edcab; "
                         "b30ee18d... for cache-after-doc-015 of 4b0edcab and e6660cc4) and the "
                         "theta=5e6 k=16 cartridge across 3 more (8b793be8... for 5ff66d3f, "
                         "508b8b0f, 55af444a). Their standalone evals agree to 16 digits."),
        },
        "in_run_eval_vs_standalone_eval_of_a_byte_identical_cartridge": {
            **A1["non_sampling_variance"]["in_run_vs_standalone_eval_of_a_BYTE_IDENTICAL_cartridge"],
            "reading": ("results.csv MIXES two harness paths. DIAG-ROPE-A/B and MECH-BETA quote in-run "
                        "numbers (2.17662/2.54836 and 2.15320/2.52961); MECH-KEYS, DIAG-KEYCURVE, "
                        "DIAG-CONTROLCURVE, DIAG-SEQUENCE quote standalone eval_forgetting.py numbers "
                        "(2.17718/2.55242 and 2.15972/2.52962) for the SAME byte-identical caches. "
                        "The gap is 0.0006-0.0065 loss -- an order of magnitude below the measured "
                        "paired resolution, so it flips nothing, but it is a real cross-row inconsistency."),
        },
        "eval_packing_jitter": A2["eval_packing_jitter"],
        "reference_draw_sensitivity": A1["non_sampling_variance"]["reference_draw_replicate_doc015_solo"],
        "seed": "no seed variation is possible in this codebase (established by DIAG-KEYCURVE by direct probe).",
    },

    "what_this_bootstrap_does_NOT_cover": [
        "seed / stochastic training variance -- there is none: the AM solve is bitwise deterministic (proved above).",
        ("the arbitrary choice of WHICH 32 of a document's ~532 synthesis conversations become reference "
         "queries. One replicate pair exists (DIAG-PERDOC's doc-015 solo write): it moves full-MT by 0.0476 "
         "and full-QA by 0.0965 -- 1.9x and 3.5x the paired eval-sampling SE. This is the dominant "
         "un-quantified term and it rests on ONE pair, on a SOLO write."),
        ("the eval set is FIXED. The bootstrap answers 'would this delta survive a fresh draw of 69 "
         "comparable questions'. If one instead regards these 69 questions AS the population, the "
         "sampling SE is zero and every delta is exact. The former frame is the only one that supports "
         "any claim of generality; it is the one used here."),
        ("questions cluster in 16 papers (3-7 each). Clustering by paper is reported throughout; it "
         "changes the paired SE by at most ~35% and flips no verdict."),
        "model/decoding nondeterminism beyond bf16 packing order (not separately isolated).",
    ],

    "provenance": {
        "snapshot_path": "/tmp/amsnap_DIAG-NOISE",
        "snapshot_contents": f"git archive HEAD cartridges examples at HEAD={head}",
        "snapshot_manifest_sha256": manifest,
        "snapshot_manifest_recipe": ("cd $SNAP && find cartridges examples -type f -name '*.py' | "
                                     "LC_ALL=C sort | xargs sha256sum | awk '{print $1\"  \"$2}' | "
                                     "LC_ALL=C sort | sha256sum"),
        "verified_import_path": "/tmp/amsnap_DIAG-NOISE/cartridges (probe asserts this at start; run from /tmp, not the repo root)",
        "repo_HEAD": head,
        "repo_dirty_files_at_run": dirty,
        "concurrent_editor": ("MECH-SEED was editing cartridges/am/continual.py, cartridges/am/finetune.py "
                              "and examples/qasper2/train/continual_am_sparse.py in the working tree; this "
                              "job imported only from the frozen git-archive snapshot."),
        "no_source_file_edited": True,
        "gpu_claimed_via_flock": "/tmp/gpu_locks_$USER/gpu0.lock (CUDA_VISIBLE_DEVICES=0)",
        "nothing_retrained": ("all 45 checkpoints are pre-existing cached artefacts (cache-after-doc-*.pt / "
                              "cache-step16.pt / phase1 cache_last.pt). No solve, no training, no data edit."),
        "instrumentation": ("a read-only probe in /tmp replicating cartridges.train.evaluate_perplexity's "
                            "loop and bucketing ce_by_token by batch.element_ids; the repo's own "
                            "LossEvalDataset(seed=42, packed_seq_length=2048) supplies the batches."),
        "wandb_runs": WANDB,
        "exactness_gate": {
            "n_evals_with_a_published_standalone_reference": 90,
            "max_abs_err": 4.018e-07,
            "note": ("includes all 64 points of the DIAG-CONTROLCURVE control k-curve and the "
                     "DIAG-KEYCURVE keys k-curve, both splits, max err 4.0e-07."),
        },
        "curve_gate": A2["curve_gate"],
        "bootstrap": {"B_paired": A1["bootstrap_B"], "B_other": 50000, "rng_seed": A1["rng_seed"],
                      "kind": "nonparametric, over examples, percentile intervals; ratio estimator sum(c)/sum(t)"},
    },
}

json.dump(diag, open(f"{REPO}/research_loop/state/diagnostics/DIAG-NOISE.json", "w"), indent=1)

# raw per-example arrays (separate file; keeps the main json readable)
raw = {"note": ("per-example token-weighted CE decomposition. loss[e] = ce_sum[e]/tokens[e]; "
                "the harness's reported Eval loss = sum(ce_sum)/sum(tokens)."),
       "paper_ids": json.load(open("/tmp/diag_noise/paper_ids.json")),
       "batches": PX["batches"], "n_elements": PX["n_elements"],
       "checkpoints": {}}
for src in (PX, PX3, PXC):
    for label, e in src["results"].items():
        if "error" in e:
            continue
        raw["checkpoints"][label] = {
            "path": e["path"], "provenance": e["provenance"],
            **{s: {"per_example_loss": e[s]["per_example_loss"],
                   "per_example_tokens": e[s]["per_example_tokens"],
                   "mean_ce": e[s]["mean_ce"],
                   "published": e[s]["published"],
                   "abs_err_vs_published": e[s]["abs_err_vs_published"]}
               for s in ("QA", "MT") if s in e},
        }
json.dump(raw, open(f"{REPO}/research_loop/state/diagnostics/DIAG-NOISE_per_example.json", "w"))

print("n claims", len(table), "| inside +-0.15:", n_inside_015,
      "| not resolvable (measured paired):", n_inside_meas,
      "| not resolvable (composite):", n_inside_comp,
      "| WRONG SIDE:", n_wrong)
print("flipped:", [t["claim"] for t in table if t["adjudicated_on_wrong_side_by_pm0.15"]])
print("MT paired 95%:", RES_MT_PAIRED_95, " QA paired 95%:", RES_QA_PAIRED_95)
print("MT single 95%:", diag["headline"]["resolution_MT_single_arm_95pct"],
      " MT unpaired 95%:", diag["headline"]["resolution_MT_unpaired_diff_95pct"])
print("composite:", COMP_MT, COMP_QA)
print("perk counts:", perk_counts)
print("n checkpoints dumped:", len(raw["checkpoints"]))
