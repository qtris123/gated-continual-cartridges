"""MECH-KEYS: assemble result.json + the diagnostics bundle from run artifacts.

Reads wrapper.log (wandb URLs, run dirs, arm wall-clocks), the eval logs,
`state/diagnostics/MECH-KEYS.json` (per-doc/per-layer collector) and
`state/diagnostics/MECH-KEYS_route_mass.json` (eval-time mass_on_S + MT/QA ratio),
and writes `results/MECH-KEYS/result.json`.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

REPO = Path("/localhome/local-triv/gated-continual-cartridges_explore")
RES = REPO / "research_loop/results/MECH-KEYS"
DIAG = REPO / "research_loop/state/diagnostics"
ARMS = ["control", "keys_norepos", "keys_repos", "omp"]
# DIAG-ROPE arm B (theta = 5e6, KEY_MODE=freeze, top_t=32), in-run
BASELINE = {"qa_forgetting": 2.15320086479187, "mt_acquisition": 2.5296061038970947}

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


collector = json.loads((DIAG / "MECH-KEYS.json").read_text())
route = json.loads((DIAG / "MECH-KEYS_route_mass.json").read_text())
sanity_path = RES / "sanity_cuda.json"
if not sanity_path.exists() or not sanity_path.read_text().strip().startswith("{"):
    sanity_path = RES / "sanity_cpu.json"
sanity = json.loads(sanity_path.read_text())

arms = {}
for a in ARMS:
    rundir = wl(rf"MECH_KEYS_{a}_RUNDIR=(\S+)")
    t0 = wl(rf"MECH_KEYS_{a}_START_EPOCH=(\d+)")
    t1 = wl(rf"MECH_KEYS_{a}_END_EPOCH=(\d+)")
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
    kr = c.get("key_rewrite_summary")
    arms[a] = {
        "run_dir": rundir,
        "wandb_run_url": wl(rf"MECH_KEYS_{a}_WANDB=(\S+)"),
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
        "eval_mt_over_qa_mass_ratio": (
            rm["MT"]["mean_mass_on_S"] / rm["QA"]["mean_mass_on_S"]
            if rm.get("MT") and rm.get("QA") and rm["QA"]["mean_mass_on_S"]
            else None
        ),
        "keys_rewritten": kr,
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

ctrl = arms["control"]["eval_in_run"]
repro = all(ctrl.get(k) == BASELINE[k] for k in BASELINE)

part_a = {
    "device": sanity.get("device"),
    "rope_theta": sanity.get("rope_theta"),
    "headline_check": {
        "what": (
            "<q, k_installed> (student frame, raw query) vs <R_offset q, k_doc> "
            "(the logit the selector actually used). Equal <=> the installed key "
            "delivers the score it was chosen for."
        ),
        "fp32": {
            "max_abs_score_err_doc_rows_off": sanity["rewrite_fp32"]["off"][
                "doc_rows_max_abs_score_err"],
            "mean_abs_score_err_doc_rows_off": sanity["rewrite_fp32"]["off"][
                "doc_rows_mean_abs_score_err"],
            "max_abs_score_err_doc_rows_on": sanity["rewrite_fp32"]["on"][
                "doc_rows_max_abs_score_err"],
            "mean_abs_score_err_doc_rows_on": sanity["rewrite_fp32"]["on"][
                "doc_rows_mean_abs_score_err"],
            "cartridge_rows_err_off": sanity["rewrite_fp32"]["off"][
                "cartridge_rows_max_abs_score_err"],
            "cartridge_rows_err_on": sanity["rewrite_fp32"]["on"][
                "cartridge_rows_max_abs_score_err"],
            "score_scale_maxabs": sanity["rewrite_fp32"]["on"]["score_scale_maxabs"],
            "n_from_doc": sanity["rewrite_fp32"]["n_from_doc"],
            "n_from_cartridge": sanity["rewrite_fp32"]["n_from_cartridge"],
        },
        "bf16_cache_dtype": {
            "max_abs_score_err_doc_rows_off": sanity["rewrite_bf16"]["off"][
                "doc_rows_max_abs_score_err"],
            "max_abs_score_err_doc_rows_on": sanity["rewrite_bf16"]["on"][
                "doc_rows_max_abs_score_err"],
            "mean_abs_score_err_doc_rows_on": sanity["rewrite_bf16"]["on"][
                "doc_rows_mean_abs_score_err"],
        },
    },
    "primitive_by_offset": sanity.get("primitive"),
    "bit_identical_when_off": sanity.get("bit_identity"),
    "freeze_path_vs_HEAD_bit_identical": sanity.get("freeze_vs_HEAD", {}).get(
        "bit_identical"),
    "config": sanity.get("config"),
    "end_to_end_stub": {
        k: {kk: v[kk] for kk in ("mean_mse", "finite", "mass_on_S", "key_rewrite")
            if kk in v}
        for k, v in sanity.get("end_to_end", {}).items()
        if isinstance(v, dict) and "mean_mse" in v
    },
}

primary = "keys_repos"
out = {
    "id": "MECH-KEYS",
    "role": "build+test",
    "board_entry": "B-ROUTE",
    "question": (
        "The value side is closed (a perfect value write reaches only MT 2.381; 4.2x "
        "more bandwidth via beta made MT WORSE because beta is query-independent). "
        "Selectivity is the binding constraint. Can a KEY change supply it -- i.e. "
        "does KEY_MODE != freeze move the MT/QA eval-time mass_on_S ratio above ~1.05 "
        "and MT below 2.53 -- once the RoPE counter-rotation hazard H2 is fixed?"
    ),
    "variable_under_test": (
        "KEY_MODE in {freeze, highest_attention, omp} x AM_KEY_REPOSITION in {0,1}, "
        "all at AM_ROPE_THETA=5e6, ENABLE_BETA=0, canonical top32 "
        "(GRANULARITY=per_layer, SLOT_SELECTION=tfidf, USE_IDF=0, RIDGE_LAMBDA=1e-4 "
        "spectral, DELTA_WEIGHT=1e-2, MAX_QUERIES_PER_HEAD=64)."
    ),
    "baseline_ref": (
        "DIAG-ROPE arm B / MECH-BETA rope arm (KEY_MODE=freeze, theta=5e6): "
        "QA 2.15320086479187 / MT 2.5296061038970947 in-run"
    ),
    "status": "done",
    "numbers": {
        "qa_forgetting_loss": arms[primary]["eval_standalone"]["qa_forgetting"],
        "mt_acquisition_loss": arms[primary]["eval_standalone"]["mt_acquisition"],
        "solve_s": arms[primary]["solve_s"],
        "phase2_e2e_s": arms[primary]["phase2_e2e_s"],
        "gradient_steps": 0,
    },
    "arms": arms,
    "baseline_reproduced": collector.get("baseline_reproduced"),
    "phase1_reference_mass": phase1,
    "mt_over_qa_mass_ratio": ratios,
    "part_a_sanity": part_a,
    "wandb_run_url": arms[primary]["wandb_run_url"],
    "wandb_run_id": (arms[primary]["wandb_run_url"] or "").rsplit("/", 1)[-1] or None,
    "artifacts": [
        str(RES / "wrapper.log"),
        str(sanity_path),
        str(RES / "config_probe.json"),
        str(DIAG / "MECH-KEYS.json"),
        str(DIAG / "MECH-KEYS_route_mass.json"),
    ] + [f"{arms[a]['run_dir']}/cache_last.pt" for a in ARMS if arms[a]["run_dir"]],
    "command": "bash research_loop/results/MECH-KEYS/launch_mech_keys.sh",
    "observations": "TBD",
}

collector["eval_time_mass_on_S"] = {
    k: {"summary": v["summary"], "per_layer": v["per_layer"]}
    for k, v in route["runs"].items()
}
collector["mt_over_qa_mass_ratio"] = ratios
collector["phase1_reference_mass"] = phase1
collector["part_a_sanity"] = part_a
(DIAG / "MECH-KEYS.json").write_text(json.dumps(collector, indent=1))

(RES / "result.json").write_text(json.dumps(out, indent=2))
print("control repro exact:", repro)
print(json.dumps(out["baseline_reproduced"], indent=1))
for a in ARMS:
    x = arms[a]
    print(
        a,
        "| in-run", {k: (round(v, 5) if v else v) for k, v in x["eval_in_run"].items()},
        "| standalone", {k: (round(v, 5) if v else v)
                         for k, v in x["eval_standalone"].items()},
        "| mse", x["am_mean_mse_over_docs"],
        "| |v|max", x["value_global_max_abs"],
        "| eval_mass", {k: round(v["mean_mass_on_S"], 5)
                        for k, v in x["eval_mass_on_S"].items()},
        "| MT/QA", x["eval_mt_over_qa_mass_ratio"],
        "| keysrw", (x["keys_rewritten"] or {}).get("doc_fraction_of_support"),
    )
print("ratios:", json.dumps(ratios, indent=1))
print("phase1:", json.dumps(phase1, indent=1))
print("WROTE", RES / "result.json")
