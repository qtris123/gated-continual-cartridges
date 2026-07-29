import json
import os

REPO = "/localhome/local-triv/gated-continual-cartridges_explore"
D = json.load(open(f"{REPO}/research_loop/state/diagnostics/DIAG-NOISE.json"))
h = D["headline"]
c = D["rescore_counts"]
os.makedirs(f"{REPO}/research_loop/results/DIAG-NOISE", exist_ok=True)

res = {
    "id": "DIAG-NOISE",
    "role": "verify",
    "board_entry": "methodology (applies to every claim in the investigation)",
    "question": D["question"],
    "variable_under_test": ("the eval's own resolution: the sampling distribution of the reported "
                            "token-weighted mean CE, single-arm vs PAIRED vs independent, estimated "
                            "from exactly-recovered per-example losses (no re-training, no re-solve)"),
    "baseline_ref": ("the inherited +-0.1-0.2 'noise band' rule of thumb (RUNBOOK §1/§6.9, "
                     "WORKERS.md); DIAG-CONTROLCURVE used +-0.15"),
    "status": "done",
    "numbers": {
        "qa_forgetting_loss": None,
        "mt_acquisition_loss": None,
        "solve_s": 0,
        "phase2_e2e_s": 0,
        "gradient_steps": 0,
        "measured_resolution_MT_paired_95pct": h["resolution_MT_paired_95pct"],
        "measured_resolution_QA_paired_95pct": h["resolution_QA_paired_95pct"],
        "measured_resolution_MT_single_arm_95pct": h["resolution_MT_single_arm_95pct"],
        "measured_resolution_QA_single_arm_95pct": h["resolution_QA_single_arm_95pct"],
        "measured_resolution_MT_independent_diff_95pct": h["resolution_MT_unpaired_diff_95pct"],
        "measured_resolution_QA_independent_diff_95pct": h["resolution_QA_unpaired_diff_95pct"],
        "composite_resolution_MT_95pct": h["composite_MT_95pct_incl_one_pipeline_replicate"],
        "composite_resolution_QA_95pct": h["composite_QA_95pct_incl_one_pipeline_replicate"],
        "n_claims_rescored": c["n_claims_scored"],
        "n_claims_the_pm015_rule_put_on_the_WRONG_side": c["n_adjudicated_on_the_WRONG_side_by_pm0.15"],
        "n_claims_not_resolvable_at_measured_paired_resolution": c["n_not_resolvable_at_the_MEASURED_paired_resolution"],
        "n_claims_not_resolvable_under_composite_resolution": c["n_not_resolvable_under_the_composite_resolution"],
        "exactness_gate_max_abs_err_vs_published": D["provenance"]["exactness_gate"]["max_abs_err"],
        "n_standalone_referenced_evals_reproduced": D["provenance"]["exactness_gate"]["standalone_referenced_evals"],
        "n_distinct_checkpoints_evaluated": D["provenance"]["n_distinct_checkpoint_files_evaluated"],
    },
    "diagnostics": {
        "metric_definition": h["the_reported_metric_is"],
        "MT_n_examples": 69, "MT_n_scored_tokens": 2562,
        "QA_n_examples": 78, "QA_n_scored_tokens": 2619,
        "paired_se_MT_median": D["resolution_table"]["MT_n69"]["paired_diff_se"]["median"],
        "paired_se_MT_range": [D["resolution_table"]["MT_n69"]["paired_diff_se"]["min"],
                                D["resolution_table"]["MT_n69"]["paired_diff_se"]["max"]],
        "paired_se_QA_median": D["resolution_table"]["QA_n78"]["paired_diff_se"]["median"],
        "paired_se_QA_range": [D["resolution_table"]["QA_n78"]["paired_diff_se"]["min"],
                                D["resolution_table"]["QA_n78"]["paired_diff_se"]["max"]],
        "single_arm_se_MT_median": D["resolution_table"]["MT_n69"]["single_arm_bootstrap_se"]["median"],
        "single_arm_se_QA_median": D["resolution_table"]["QA_n78"]["single_arm_bootstrap_se"]["median"],
        "independent_diff_se_MT_median": D["resolution_table"]["MT_n69"]["independent_diff_se"],
        "independent_diff_se_QA_median": D["resolution_table"]["QA_n78"]["independent_diff_se"],
        "paired_se_clustered_by_paper_MT_median": D["resolution_table"]["MT_n69"]["paired_diff_se_clustered_by_paper16"],
        "paired_se_clustered_by_paper_QA_median": D["resolution_table"]["QA_n78"]["paired_diff_se_clustered_by_paper16"],
        "per_example_loss_correlation_between_arms_range": [0.921, 0.992],
        "se_of_the_se_double_bootstrap": D["resolution_table"]["uncertainty_of_the_resolution_itself"]["examples"],
        "token_weighting_shift_on_an_ARM_MT_median": D["resolution_table"]["token_weighting"]["macro_minus_micro_single_arm_MT_median"],
        "token_weighting_shift_on_a_DIFFERENCE_MT_median": D["resolution_table"]["token_weighting"]["effect_on_a_DIFFERENCE_MT_median"],
        "argmin_selection_MT": {
            "point_delta_keys_k12_minus_control_k10": D["argmin_selection_test"]["MT"]["point_delta_at_own_optima"],
            "selection_aware_ci95": D["argmin_selection_test"]["MT"]["selection_aware_bootstrap"]["ci95"],
            "selection_bias": D["argmin_selection_test"]["MT"]["selection_aware_bootstrap"]["selection_bias_mean_minus_point"],
        },
        "per_k_keys_minus_control_counts": D["per_k_counts"],
        "diag_perdoc_pooled_solo_minus_k16": D["diag_perdoc_rescore"]["_pooled_5docs_27questions"]["measured_paired_delta_solo_minus_k16"],
        "diag_perdoc_pooled_paired_se": D["diag_perdoc_rescore"]["_pooled_5docs_27questions"]["paired"]["bootstrap_se"],
        "run_to_run_solve_variance": 0.0,
        "in_run_vs_standalone_eval_max_gap": 0.006524712464226035,
        "eval_packing_jitter_median": D["non_sampling_variance_components"]["eval_packing_jitter"]["median_abs_diff"],
        "eval_packing_jitter_max": D["non_sampling_variance_components"]["eval_packing_jitter"]["max_abs_diff"],
        "reference_draw_replicate_abs_diff_MT": 0.04757609440959776,
        "reference_draw_replicate_abs_diff_QA": 0.09651849566635295,
        "mass_on_S": ("not applicable — no AM run was performed; this task re-scored cached checkpoints "
                      "and re-used the routing figures already in the DIAG-KEYCURVE / DIAG-CONTROLCURVE bundles"),
    },
    "wandb_run_url": D["provenance"]["wandb_runs"]["pass4_full_k_curves_32ckpt"],
    "wandb_run_id": "u54zfh9p",
    "wandb_all_runs": D["provenance"]["wandb_runs"],
    "artifacts": [
        "research_loop/state/diagnostics/DIAG-NOISE.json",
        "research_loop/state/diagnostics/DIAG-NOISE_per_example.json",
        "research_loop/results/DIAG-NOISE/result.json",
        "research_loop/results/DIAG-NOISE/perex_probe.py",
        "research_loop/results/DIAG-NOISE/analyze.py",
        "research_loop/results/DIAG-NOISE/analyze2.py",
        "research_loop/results/DIAG-NOISE/probe_logs/",
    ],
    "command": (
        "frozen-snapshot pin: mkdir -p /tmp/amsnap_DIAG-NOISE && git archive HEAD cartridges examples | "
        "tar -x -C /tmp/amsnap_DIAG-NOISE; cd /tmp && PYTHONPATH=/tmp/amsnap_DIAG-NOISE python -c "
        "'import cartridges,os;print(os.path.dirname(cartridges.__file__))'  -> "
        "/tmp/amsnap_DIAG-NOISE/cartridges. Then 4 GPU passes under "
        "flock /tmp/gpu_locks_$USER/gpu0.lock, each running a read-only probe from /tmp that replicates "
        "cartridges.train.evaluate_perplexity and buckets ce_by_token by batch.element_ids: "
        "perex_probe.py (13 checkpoints), perex_probe2.py (28), perex_probe3.py (6 cache-step16 files), "
        "perex_probe4.py (the full 16-point control and keys k-curves, 32). "
        "Analysis on CPU: analyze.py (bootstraps + claim table), analyze2.py (argmin-selection bootstrap, "
        "packing jitter, composite resolution), finalize.py (bundles). NO SOURCE FILE EDITED."
    ),
    "observations": (
        "The reported metric is a token-weighted micro-average CE over 69 MT / 78 QA examples, and it "
        "decomposes exactly: bucketing the harness's own ce_by_token by element id reproduces the "
        f"published standalone `Eval loss` for {D['provenance']['exactness_gate']['standalone_referenced_evals']} "
        f"evals (60 distinct cached checkpoints, including all 64 points of the two k-curves) to "
        f"{D['provenance']['exactness_gate']['max_abs_err']:.1e} — so no leave-one-out approximation was needed and "
        "nothing was re-trained. "
        "MEASURED RESOLUTION (nonparametric bootstrap over examples, 95%): a SINGLE arm's loss is "
        f"+-{h['resolution_MT_single_arm_95pct']:.3f} MT / +-{h['resolution_QA_single_arm_95pct']:.3f} QA; "
        f"an INDEPENDENT difference is +-{h['resolution_MT_unpaired_diff_95pct']:.3f} MT / "
        f"+-{h['resolution_QA_unpaired_diff_95pct']:.3f} QA; a PAIRED difference is "
        f"+-{h['resolution_MT_paired_95pct']:.3f} MT / +-{h['resolution_QA_paired_95pct']:.3f} QA "
        "(median over the 23 comparisons; per-comparison range 0.021-0.090 MT). Every comparison the board "
        "makes is paired — the two arms are scored on the same examples with byte-identical token counts, "
        "and their per-example losses correlate r=0.92-0.99 — so the paired ruler is the correct one, and "
        "the inherited +-0.15 band is about 3x TOO WIDE for it (while being ~1.7x too narrow for an "
        "independent comparison, which is presumably where the rule of thumb came from). "
        "RE-SCORE: of 23 load-bearing deltas, 11 sat inside +-0.15 and were called noise; only 4 are "
        "actually unresolvable at the measured resolution (DIAG-CONTROLCURVE's QA-at-optima -0.0115, "
        "DIAG-KEYCURVE's QA-at-edge +0.0230, and both DIAG-ROPE deltas). Seven were adjudicated on the "
        "WRONG side of the true threshold, the most consequential being DIAG-CONTROLCURVE's own -0.1363: "
        "its 95% CI is [-0.181, -0.094], 6.1 SE from zero, and an argmin-selection-aware bootstrap that "
        "re-chooses each arm's best k on every resample leaves it at [-0.182, -0.094] with a selection bias "
        "of only -0.0005. The MT half of the key-side refutation therefore does not stand on noise grounds; "
        "the QA half does (-0.0115, and control-better +0.0430 at each arm's QA-best, both unresolvable). "
        "Per k, keys-minus-control clears +-0.15 at 7/16 k on MT and 0/16 on QA, but clears the MEASURED "
        "resolution at 16/16 MT (all negative) and 7/16 QA. "
        "ADVERSARIAL LIMITS, stated plainly: the bootstrap SE is itself uncertain by about +-25% (double "
        "bootstrap 95% CI on the DIAG-CONTROLCURVE SE = [0.017, 0.027]), so the MT paired resolution is a "
        "range +-0.04 to +-0.07, not a point; clustering the 69 questions in their 16 papers changes the "
        "paired SE by at most ~35% and flips no verdict; and the bootstrap covers eval-question sampling "
        "ONLY. Three non-sampling components were measured instead of assumed: run-to-run variance of the "
        "solve is exactly ZERO (six nominally identical runs produce byte-identical caches, two sha256 "
        "groups), in-run vs standalone eval of a byte-identical cartridge differs by up to 0.0065 (and "
        "results.csv mixes the two conventions across rows), and eval packing changes a per-subset loss by "
        "a median 0.0045 / max 0.0216. The one component that is NOT small is the arbitrary choice of which "
        "32 conversations become reference queries: DIAG-PERDOC's single replicate pair differs by 0.0476 "
        "on full MT and 0.0965 on full QA — 1.9x and 3.5x the paired sampling SE. Folding that one "
        f"observation in gives a conservative composite resolution of +-{h['composite_MT_95pct_incl_one_pipeline_replicate']:.3f} MT / "
        f"+-{h['composite_QA_95pct_incl_one_pipeline_replicate']:.3f} QA, under which 10 of the 23 deltas "
        "become unresolvable — including the keys arm's k=12-vs-k=16 gap (0.058/0.079), MECH-KEYS' QA "
        "-0.125, and DIAG-SEQUENCE's QA tail rise. DIAG-CONTROLCURVE's -0.1363 survives even that. "
        "DIAG-PERDOC's headline re-derives independently here as +1.1296 (paired se 0.082, CI "
        "[0.963, 1.283], 0/28 questions favouring solo; board quoted +1.101 se 0.083 over '27 questions' — "
        "the five subsets contain 28). Finally, token weighting shifts an ARM's reported loss by 0.03-0.18 "
        "versus an unweighted per-question mean (always upward), i.e. by as much as the whole old band, but "
        "shifts a DIFFERENCE by a median 0.006 and at most 0.058."
    ),
}
json.dump(res, open(f"{REPO}/research_loop/results/DIAG-NOISE/result.json", "w"), indent=1)
print("written")
