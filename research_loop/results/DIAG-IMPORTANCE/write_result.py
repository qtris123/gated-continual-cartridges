"""Assemble research_loop/results/DIAG-IMPORTANCE/result.json from the diagnostic dump.

Every number is read out of research_loop/state/diagnostics/DIAG-IMPORTANCE.json (and the
wandb URL out of the wrapper log) so nothing is transcribed by hand.
"""
import json
import os
import re
import sys

REPO = "/localhome/local-triv/gated-continual-cartridges_explore"
D = json.load(open(f"{REPO}/research_loop/state/diagnostics/DIAG-IMPORTANCE.json"))
LOG = open(f"{REPO}/research_loop/results/DIAG-IMPORTANCE/wrapper_full.log", errors="replace").read()
url = re.search(r"WANDB_RUN_URL=(\S+)", LOG)
rid = re.search(r"WANDB_RUN_ID=(\S+)", LOG)
if url is None:
    print("NO WANDB RUN URL IN LOG -- refusing to write a bundle", file=sys.stderr)
    sys.exit(2)

hl = D["headline"]
pl = D["spearman_per_layer_mean"]
sd = D["spearman_per_layer_sd"]
ov = D["top32_overlap_vs_incumbent_eval_ranker"]
ovm = D["top32_overlap_matrix_descending"]
sb = D["safe_acquisition_budget"]["tf_mass_mt_top32"]
sb2 = D["safe_acquisition_budget"]["ranker_tf_mt_top32"]
co = D["concentration"]
bo = D["bootstrap"]
rel = D["split_half_reliability_spearman"]
xv = D["cross_validation_vs_DIAG_ROUTING"]

def r(x, n=4):
    return round(float(x), n)

diagnostics = {
    "definitions_see": "research_loop/results/DIAG-IMPORTANCE/METRICS.md (exact formulas + code paths)",
    "headline_spearman_pooled": {k: r(v, 4) for k, v in hl.items() if k.endswith("_pooled")},
    "headline_spearman_per_layer_mean": {
        "tf_mass_qa__fisher": r(pl["tf_mass_qa"]["fisher"]),
        "tf_mass_qa__kl_loo": r(pl["tf_mass_qa"]["kl_loo"]),
        "tf_mass_qa__entropy": r(pl["tf_mass_qa"]["entropy"]),
        "tf_mass_qa__redundancy": r(pl["tf_mass_qa"]["redundancy"]),
        "tf_mass_qa__contrast": r(pl["tf_mass_qa"]["contrast"]),
        "tf_mass_qa__tf_mass_mt": r(pl["tf_mass_qa"]["tf_mass_mt"]),
        "fisher__kl_loo": r(pl["fisher"]["kl_loo"]),
        "fisher__redundancy": r(pl["fisher"]["redundancy"]),
        "fisher__entropy": r(pl["fisher"]["entropy"]),
        "fisher__contrast": r(pl["fisher"]["contrast"]),
        "_sd_over_36_layers": {
            "tf_mass_qa__fisher": r(sd["tf_mass_qa"]["fisher"]),
            "tf_mass_qa__kl_loo": r(sd["tf_mass_qa"]["kl_loo"]),
        },
    },
    "bootstrap_ci95_spearman_over_eval_examples": {
        k: [r(v["lo95"]), r(v["hi95"])] for k, v in bo["spearman"].items()
    },
    "split_half_reliability_spearman": {
        s: {k: r(v) for k, v in d.items()} for s, d in rel.items()
    },
    "top32_overlap_vs_incumbent_ranker_selection": {
        k: {"acquisition_direction": r(v["mean"]),
            "descending": r(v["mean_if_taken_descending"])}
        for k, v in ov.items()
    },
    "top32_overlap_vs_canonical_run_actual_selections": {
        k: (r(v["mean_over_16_docs"]) if isinstance(v, dict) else r(v))
        for k, v in D["top32_overlap_vs_actual_run_selections"].items()
    },
    "top32_overlap_qa_vs_mt_tf_mass_per_layer": r(ovm["tf_mass_qa"]["tf_mass_mt"]),
    "safe_acquisition_budget_mt_top32_by_tf_mass": {
        m: {k: r(v) for k, v in d.items()} for m, d in sb.items()
    },
    "safe_acquisition_budget_mt_top32_by_ranker_tf": {
        m: {k: r(v) for k, v in d.items()} for m, d in sb2.items()
    },
    "safe_budget_bootstrap_ci95": {
        m: {k: [r(v["lo95"]), r(v["hi95"])] for k, v in d.items()}
        for m, d in bo["safe_budget_tf_mass_mt_top32"].items()
    },
    "concentration_of_qa_importance": {
        m: {k: r(v) for k, v in d.items() if k.startswith("mean")} for m, d in co.items()
    },
    "cross_validation_vs_DIAG_ROUTING": {
        "measured": {k: r(v, 5) for k, v in xv.items() if isinstance(v, float)},
        "reference": xv["DIAG-ROUTING_reference"],
    },
    "redundancy_ridge_sensitivity": {
        k: r(v, 6) for k, v in D["redundancy_ridge_sensitivity"].items()
    },
    "fisher_cost": D["fisher_cost"],
    "attention_pass_cost_s": D["attention_pass_cost"],
    "per_example_micro_mean_ce_from_fisher_pass": {
        k: r(v["micro_mean_ce"], 6) for k, v in D["per_example_losses_from_fisher_pass"].items()
    },
    "n_eval_examples": D["provenance"]["n_eval_examples"],
    "n_bootstrap": bo["n_bootstrap"],
    "attenuation_corrected_spearman": D["attenuation_corrected_spearman"],
    "selector_tradeoff_table": D["selector_tradeoff_table"],
    "bandwidth_under_a_fisher_constraint": D["bandwidth_under_a_fisher_constraint"],
    "joint_availability_mean_slots_per_layer": D["joint_availability"],
}

out = {
    "id": "DIAG-IMPORTANCE",
    "role": "measure",
    "board_entry": "B-GATE",
    "question": (
        "Does the incumbent attention-mass (TF) ranker already capture slot IMPORTANCE, or is "
        "there headroom for an information-theoretic selector? Concretely: how do six per-slot "
        "scores (tf_mass, entropy, leave-one-out KL, diagonal Fisher of the QA loss, linear "
        "redundancy of the value vector, MT/QA contrast) rank-correlate on the Phase-1 "
        "cartridge, how differently would each select the top-32 slots, and how much larger is "
        "the 'safe acquisition budget' under an importance metric than under attention mass "
        "(~9% from DIAG-ROUTING's 0.914 top-32 overlap)?"
    ),
    "variable_under_test": (
        "none - forward passes plus diagnostic backward passes on the untouched Phase-1 "
        "cartridge; QA vs MT eval splits. No knob was changed and nothing was trained."
    ),
    "baseline_ref": "DIAG-ROUTING (same cartridge, same eval-time softmax convention) / "
                    "DIAG-OVERWRITE (the incumbent ranker's actual per-document selections)",
    "status": "done",
    "numbers": {
        "qa_forgetting_loss": None,
        "mt_acquisition_loss": None,
        "solve_s": 0,
        "phase2_e2e_s": 0,
        "gradient_steps": 0,
        "note": (
            "No training and no eval-harness run: no CE loss is reported as a result "
            "(WORKERS.md: never fabricate a number). The Fisher pass does recompute the "
            "harness's own per-token CE, and its token-weighted micro-average is reported "
            "under diagnostics.per_example_micro_mean_ce_from_fisher_pass as a plumbing check. "
            "mass_on_S is not applicable: no AM write was performed; the closest quantity, the "
            "per-slot eval-time attention mass, IS the primary measurement here."
        ),
        "wall_clock_total_s": round(float(D["wall_clock_total_s"]), 1),
        "fisher_backward_s_total": round(float(D["fisher_cost"]["total_s"]), 1),
        "n_backward_passes": D["fisher_cost"]["n_backward_passes"],
    },
    "diagnostics": diagnostics,
    "wandb_run_url": url.group(1),
    "wandb_run_id": rid.group(1) if rid else "",
    "artifacts": [
        "research_loop/state/diagnostics/DIAG-IMPORTANCE.json",
        "research_loop/state/diagnostics/DIAG-IMPORTANCE.npz",
        "research_loop/results/DIAG-IMPORTANCE/METRICS.md",
        "research_loop/results/DIAG-IMPORTANCE/measure_slot_importance.py",
        "research_loop/results/DIAG-IMPORTANCE/launch_diag_importance.sh",
        "research_loop/results/DIAG-IMPORTANCE/wrapper_full.log",
        "research_loop/results/DIAG-IMPORTANCE/wrapper_smoke.log",
    ],
    "command": (
        "TAG=full NBOOT=500 FP32_CHECK=1 bash "
        "research_loop/results/DIAG-IMPORTANCE/launch_diag_importance.sh  (backgrounded; claims "
        "one GPU by flock, exports PYTHONPATH=/tmp/amsnap_DIAG-IMPORTANCE from `git archive "
        "HEAD`, then runs research_loop/results/DIAG-IMPORTANCE/measure_slot_importance.py on "
        "outputs/phase1_selfdistill_qwen512/cache_last.pt over both eval parquets). NO file "
        "under cartridges/ or examples/ was edited; three new standalone files were created "
        "under research_loop/results/DIAG-IMPORTANCE/."
    ),
    "provenance": D["provenance"],
    "observations": "FILL",
}

json.dump(out, open(f"{REPO}/research_loop/results/DIAG-IMPORTANCE/result.json", "w"), indent=2)
print("wrote result.json")
print(json.dumps(diagnostics["headline_spearman_per_layer_mean"], indent=1))
