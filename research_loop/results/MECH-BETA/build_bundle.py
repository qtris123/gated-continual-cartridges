"""MECH-BETA: assemble result.json + the diagnostics bundle from run artifacts.

Reads wrapper.log (wandb URLs, run dirs, arm wall-clocks), the six eval logs,
`state/diagnostics/MECH-BETA.json` (per-doc/per-layer collector) and
`state/diagnostics/MECH-BETA_route_mass.json` (eval-time mass_on_S, beta-aware),
and writes `results/MECH-BETA/result.json`.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

REPO = Path("/localhome/local-triv/gated-continual-cartridges_explore")
RES = REPO / "research_loop/results/MECH-BETA"
DIAG = REPO / "research_loop/state/diagnostics"
ARMS = ["control", "rope", "beta"]
BASELINE = {"qa_forgetting": 2.1766157150268555, "mt_acquisition": 2.5483615398406982}
DIAG_ROPE_B = {"qa": 2.15320, "mt": 2.52961}

wrapper = (RES / "wrapper.log").read_text()


def wl(pat, default=None):
    m = re.search(pat, wrapper)
    return m.group(1) if m else default


def eval_loss(arm, split):
    p = RES / f"eval_{arm}_{split}.log"
    if not p.exists():
        return None, None
    txt = p.read_text()
    m = re.findall(r"Eval loss - ([0-9.]+)", txt)
    u = re.findall(r"https://wandb\.ai/\S+/runs/[a-z0-9]+", txt)
    return (float(m[-1]) if m else None), (u[-1] if u else None)


collector = json.loads((DIAG / "MECH-BETA.json").read_text())
route = json.loads((DIAG / "MECH-BETA_route_mass.json").read_text())
sanity = json.loads((RES / "sanity_cuda.json").read_text())

arms = {}
for a in ARMS:
    rundir = wl(rf"MECH_BETA_{a}_RUNDIR=(\S+)")
    t0 = wl(rf"MECH_BETA_{a}_START_EPOCH=(\d+)")
    t1 = wl(rf"MECH_BETA_{a}_END_EPOCH=(\d+)")
    c = collector["arms"].get(a, {})
    ps = c.get("phase2_summary", {}) or {}
    ev = ps.get("eval_metrics", {}) or {}
    qa, qa_url = eval_loss(a, "QA")
    mt, mt_url = eval_loss(a, "MT")
    rm = {}
    for split in ("QA", "MT"):
        r = route["runs"].get(f"{a}|{split}")
        if r:
            rm[split] = {
                "mean_mass_on_S": r["summary"]["mean_mass_on_S"],
                "mean_mass_on_S_scored": r["summary"]["mean_mass_on_S_scored"],
                "mean_mass_on_cart": r["summary"]["mean_mass_on_cart"],
                "max_mass_on_S": r["summary"]["max_mass_on_S"],
                "argmax_layer": r["summary"]["argmax_layer"],
            }
    arms[a] = {
        "run_dir": rundir,
        "wandb_run_url": wl(rf"MECH_BETA_{a}_WANDB=(\S+)"),
        "wall_clock_s": c.get("wall_clock_s"),
        "phase2_e2e_s": (int(t1) - int(t0)) if (t0 and t1) else None,
        "solve_s": c.get("doc_total_s_sum"),
        "prefill_s_total": c.get("prefill_s_total"),
        "gradient_steps": 0,
        "eval_in_run": {k: ev.get(k, {}).get("loss") for k in BASELINE},
        "eval_standalone": {"qa_forgetting": qa, "mt_acquisition": mt},
        "eval_wandb": {"qa": qa_url, "mt": mt_url},
        "am_mean_mse_over_docs": c.get("mean_mse_over_documents"),
        "value_global_max_abs": (ps.get("value_norms", {}) or {}).get("global_max_abs"),
        "ref_mass_on_S_mean_over_layers": c.get("mass_on_S_mean_over_layers"),
        "eval_mass_on_S": rm,
        "beta_summary": c.get("beta_summary"),
    }

ratios = route.get("ratios", {})
phase1 = {}
for split in ("QA", "MT"):
    r = route["runs"].get(f"phase1|{split}")
    if r:
        phase1[split] = {
            "mean_mass_on_S": r["summary"]["mean_mass_on_S"],
            "mean_mass_on_cart": r["summary"]["mean_mass_on_cart"],
        }

status = "done"
fail = None
ctrl = arms["control"]["eval_in_run"]
repro = all(ctrl.get(k) == BASELINE[k] for k in BASELINE)

out = {
    "id": "MECH-BETA",
    "role": "build+test",
    "board_entry": "B-SOLVE",
    "question": (
        "With the AM paper's own safeguards restored (box on beta, rank-revealing "
        "NNLS warm start) and the corrected rotary base, does per-key log-bias mass "
        "matching raise eval-time mass_on_S -- and does more bandwidth convert into "
        "MT acquisition?"
    ),
    "variable_under_test": (
        "ENABLE_BETA=1 with boxed NNLS (AM_BETA_BOX=3.0, AM_NNLS_ITERS=2, "
        "AM_NNLS_DRIVER=gelsd) at AM_ROPE_THETA=5e6, vs beta-off at 5e6 (rope-only) "
        "and beta-off at 1e4 (canonical control). All else canonical top32."
    ),
    "baseline_ref": "EXP-007-top32 / DIAG-ROPE arm A (QA 2.1766157150268555 / MT 2.5483615398406982)",
    "status": status,
    "numbers": {
        "qa_forgetting_loss": arms["beta"]["eval_standalone"]["qa_forgetting"],
        "mt_acquisition_loss": arms["beta"]["eval_standalone"]["mt_acquisition"],
        "solve_s": arms["beta"]["solve_s"],
        "phase2_e2e_s": arms["beta"]["phase2_e2e_s"],
        "gradient_steps": 0,
    },
    "arms": arms,
    "baseline_reproduced": collector.get("baseline_reproduced"),
    "diag_rope_arm_b_reference": DIAG_ROPE_B,
    "phase1_reference_mass": phase1,
    "mt_over_qa_mass_ratio": ratios,
    "part_a_sanity": {
        "device": sanity.get("device"),
        "nnls_default_bit_identical": sanity.get("nnls_default_bit_identical"),
        "refit_default_bit_identical": sanity.get("refit_default_bit_identical"),
        "rank_deficient_case": sanity.get("fixed_path_info"),
        "old_path_beta_range": [
            sanity.get("old_path_beta_min"), sanity.get("old_path_beta_max")
        ],
        "boxed_beta_range": [
            sanity.get("refit_boxed_beta", {}).get("min"),
            sanity.get("refit_boxed_beta", {}).get("max"),
        ],
        "end_to_end_mass_on_S_ratio": sanity.get("end_to_end", {}).get(
            "mass_on_S_ratio_on_over_off"
        ),
        "should_fit_beta": sanity.get("should_fit_beta"),
    },
    "wandb_run_url": arms["beta"]["wandb_run_url"],
    "wandb_run_id": (arms["beta"]["wandb_run_url"] or "").rsplit("/", 1)[-1] or None,
    "artifacts": [
        str(RES / "wrapper.log"),
        str(RES / "sanity_cuda.json"),
        str(DIAG / "MECH-BETA.json"),
        str(DIAG / "MECH-BETA_route_mass.json"),
    ] + [f"{arms[a]['run_dir']}/cache_last.pt" for a in ARMS if arms[a]["run_dir"]],
    "command": "bash research_loop/results/MECH-BETA/launch_mech_beta.sh",
    "observations": "TBD",
}
if fail:
    out["failure_cause"] = fail

# The named deliverable is `state/diagnostics/MECH-BETA.json`; fold the eval-time
# mass_on_S / MT-QA ratio / phase-1 reference into it so it is self-contained.
collector["eval_time_mass_on_S"] = {
    k: {
        "summary": v["summary"],
        "per_layer": v["per_layer"],
        "beta": v.get("beta"),
    }
    for k, v in route["runs"].items()
}
collector["mt_over_qa_mass_ratio"] = ratios
collector["phase1_reference_mass"] = phase1
collector["part_a_sanity"] = sanity
(DIAG / "MECH-BETA.json").write_text(json.dumps(collector, indent=1))

(RES / "result.json").write_text(json.dumps(out, indent=2))
print(json.dumps({k: out[k] for k in ("numbers", "baseline_reproduced")}, indent=2))
print("control repro exact:", repro)
for a in ARMS:
    x = arms[a]
    print(
        a,
        "in-run QA/MT", x["eval_in_run"],
        "| standalone", x["eval_standalone"],
        "| mse", x["am_mean_mse_over_docs"],
        "| |v|max", x["value_global_max_abs"],
        "| ref_mass", x["ref_mass_on_S_mean_over_layers"],
        "| eval_mass", {k: round(v["mean_mass_on_S"], 5) for k, v in x["eval_mass_on_S"].items()},
        "| beta", (x["beta_summary"] or {}).get("last_doc_beta"),
    )
print("ratios:", json.dumps(ratios, indent=1))
print("phase1:", json.dumps(phase1, indent=1))
print("WROTE", RES / "result.json")
