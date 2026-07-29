"""MECH-SEQUENTIAL: assemble result.json + the diagnostics bundle from artifacts.

Reads wrapper.log (wandb URLs, run dirs, arm wall-clocks), the eval logs + curve.tsv,
`state/diagnostics/MECH-SEQUENTIAL.json` (per-doc/per-layer collector, incl. the
on-policy query-drift block) and `state/diagnostics/MECH-SEQUENTIAL_route_mass.json`
(eval-time mass_on_S, total cartridge mass, MT/QA ratio), and writes
`results/MECH-SEQUENTIAL/result.json`.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

REPO = Path("/localhome/local-triv/gated-continual-cartridges_explore")
RES = REPO / "research_loop/results/MECH-SEQUENTIAL"
DIAG = REPO / "research_loop/state/diagnostics"
ARMS = ["control", "onpolicy", "onpolicy_keys"]
# MECH-KEYS control == DIAG-ROPE arm B (theta=5e6, KEY_MODE=freeze, top_t=32)
BASELINE = {"qa_forgetting": 2.15320086479187, "mt_acquisition": 2.5296061038970947}
BASELINE_STANDALONE = {
    "qa_forgetting": 2.159724712371826,
    "mt_acquisition": 2.529625177383423,
}
# MECH-KEYS best gradient-free point (keys+reposition), standalone
MECH005 = {"qa_forgetting": 2.034916400909424, "mt_acquisition": 2.330503463745117}

wrapper = (RES / "wrapper.log").read_text()


def wl(pat, default=None):
    m = re.search(pat, wrapper)
    return m.group(1) if m else default


def eval_loss(arm, split, tag=""):
    p = RES / "evals" / f"eval_{arm}{tag}_{split}.log"
    if not p.exists():
        return None, None
    txt = p.read_text()
    m = re.findall(r"Eval loss - ([0-9.]+)", txt)
    u = re.findall(r"https://wandb\.ai/\S+/runs/[A-Za-z0-9]+", txt)
    return (float(m[-1]) if m else None), (u[-1] if u else None)


collector = json.loads((DIAG / "MECH-SEQUENTIAL.json").read_text())
route = json.loads((DIAG / "MECH-SEQUENTIAL_route_mass.json").read_text())
sanity_path = RES / "sanity_cuda.json"
if not sanity_path.exists() or not sanity_path.read_text().strip().startswith("{"):
    sanity_path = RES / "sanity_cpu.json"
sanity = json.loads(sanity_path.read_text())

arms = {}
for a in ARMS:
    rundir = wl(rf"MS_{a}_RUNDIR=(\S+)")
    t0 = wl(rf"MS_{a}_START_EPOCH=(\d+)")
    t1 = wl(rf"MS_{a}_END_EPOCH=(\d+)")
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
    op = c.get("onpolicy_summary")
    curve = {}
    for k in ("8", "12"):
        q, _ = eval_loss(a, "QA", f"_k{k}")
        m, _ = eval_loss(a, "MT", f"_k{k}")
        if q is not None or m is not None:
            curve[k] = {"qa_forgetting": q, "mt_acquisition": m}
    if qa is not None or mt is not None:
        curve["16"] = {"qa_forgetting": qa, "mt_acquisition": mt}
    arms[a] = {
        "run_dir": rundir,
        "wandb_run_url": wl(rf"MS_{a}_WANDB=(\S+)"),
        "wall_clock_s": c.get("wall_clock_s"),
        "phase2_e2e_s": (int(t1) - int(t0)) if (t0 and t1) else None,
        "solve_s": c.get("doc_total_s_sum"),
        "prefill_s_total": c.get("prefill_s_total"),
        "onpolicy_refresh_s_total": (op or {}).get("refresh_s_total_all_docs"),
        "gradient_steps": 0,
        "eval_in_run": {k: ev.get(k, {}).get("loss") for k in BASELINE},
        "eval_standalone": {"qa_forgetting": qa, "mt_acquisition": mt},
        "eval_wandb": {"qa": qa_url, "mt": mt_url},
        "k_curve_standalone": curve,
        "am_mean_mse_over_docs": c.get("mean_mse_over_documents"),
        "value_global_max_abs": (ps.get("value_norms", {}) or {}).get("global_max_abs"),
        "ref_mass_on_S_mean_over_layers": c.get("mass_on_S_mean_over_layers"),
        "eval_mass_on_S": rm,
        "eval_mt_over_qa_mass_ratio": (
            rm["MT"]["mean_mass_on_S"] / rm["QA"]["mean_mass_on_S"]
            if rm.get("MT") and rm.get("QA") and rm["QA"]["mean_mass_on_S"]
            else None
        ),
        "eval_total_cartridge_mass": {
            k: v["mean_mass_on_cart"] for k, v in rm.items()
        },
        "onpolicy": op,
        "keys_rewritten": c.get("key_rewrite_summary"),
    }

# cost multiples vs the control
ctrl_solve = arms["control"]["solve_s"] or 0.0
ctrl_e2e = arms["control"]["phase2_e2e_s"] or 0
for a in ARMS:
    arms[a]["solve_s_multiple_vs_control"] = (
        (arms[a]["solve_s"] / ctrl_solve) if (ctrl_solve and arms[a]["solve_s"]) else None
    )
    arms[a]["phase2_e2e_s_multiple_vs_control"] = (
        (arms[a]["phase2_e2e_s"] / ctrl_e2e)
        if (ctrl_e2e and arms[a]["phase2_e2e_s"]) else None
    )

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
repro_inrun = all(ctrl.get(k) == BASELINE[k] for k in BASELINE)
sa = arms["control"]["eval_standalone"]
repro_standalone = {
    k: (
        None if sa.get(k) is None
        else round(sa[k] - BASELINE_STANDALONE[k], 8)
    )
    for k in BASELINE_STANDALONE
}

part_a = {
    "device": sanity.get("device"),
    "config": sanity.get("config"),
    "checks": sanity.get("arms", {}).get("_checks"),
    "fail_loud": sanity.get("fail_loud"),
    "off_vs_HEAD_bit_identical": sanity.get("off_vs_HEAD", {}).get("bit_identical"),
    "stub_mean_mse": {
        k: v.get("mean_mse")
        for k, v in (sanity.get("arms") or {}).items()
        if isinstance(v, dict) and "mean_mse" in v
    },
}

primary = "onpolicy_keys"
out = {
    "id": "MECH-SEQUENTIAL",
    "role": "build+test",
    "board_entry": "B-CASCADE (query-distribution)",
    "question": (
        "Values, key selectivity and allocation are all closed. DIAG-KEYSPACE "
        "explicitly did NOT bound mechanisms that change the QUERY DISTRIBUTION. "
        "We collect reference queries in ONE forward pass over the pre-write "
        "cartridge and then solve all 36 layers against them, but writing layer l "
        "perturbs the residual stream, so the true queries at layers l+1..35 are "
        "not the ones we fitted (the AM paper does on-policy layer-sequential "
        "re-extraction; SCOUT-AM divergence #3 / LIT-006). Does re-extracting the "
        "reference queries from the UPDATED cache move MT materially below 2.33?"
    ),
    "variable_under_test": (
        "AM_ONPOLICY_LAYERS in {unset, 4} (group size 4 -> 8 re-extractions per "
        "document, before layers 4,8,...,32), crossed with the MECH-005 best point "
        "(KEY_MODE=highest_attention + AM_KEY_REPOSITION=1). All at "
        "AM_ROPE_THETA=5e6, ENABLE_BETA=0, canonical top32 (GRANULARITY=per_layer, "
        "SLOT_SELECTION=tfidf, USE_IDF=0, RIDGE_LAMBDA=1e-4 spectral, "
        "DELTA_WEIGHT=1e-2, MAX_QUERIES_PER_HEAD=64)."
    ),
    "baseline_ref": (
        "MECH-KEYS control (KEY_MODE=freeze, theta=5e6): in-run QA 2.15320086479187 "
        "/ MT 2.5296061038970947, standalone QA 2.159724712371826 / MT "
        "2.529625177383423; and MECH-005 best point standalone QA 2.0349 / MT 2.3305"
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
    "deltas_vs_reference": {
        "onpolicy_minus_control": {
            k: (
                None
                if (arms["onpolicy"]["eval_standalone"][k] is None
                    or arms["control"]["eval_standalone"][k] is None)
                else arms["onpolicy"]["eval_standalone"][k]
                - arms["control"]["eval_standalone"][k]
            )
            for k in BASELINE
        },
        "onpolicy_keys_minus_MECH005": {
            k: (
                None
                if arms["onpolicy_keys"]["eval_standalone"][k] is None
                else arms["onpolicy_keys"]["eval_standalone"][k] - MECH005[k]
            )
            for k in BASELINE
        },
        "noise_band_loss": [0.1, 0.2],
        "eval_n": {"MT": 69, "QA": 78},
    },
    "baseline_reproduced": {
        "in_run_bit_identical": repro_inrun,
        "in_run_measured": ctrl,
        "in_run_reference": BASELINE,
        "standalone_measured": sa,
        "standalone_reference": BASELINE_STANDALONE,
        "standalone_delta": repro_standalone,
    },
    "phase1_reference_mass": phase1,
    "mt_over_qa_mass_ratio": ratios,
    "part_a_sanity": part_a,
    "wandb_run_url": arms[primary]["wandb_run_url"],
    "wandb_run_id": (arms[primary]["wandb_run_url"] or "").rsplit("/", 1)[-1] or None,
    "artifacts": [
        str(RES / "wrapper.log"),
        str(sanity_path),
        str(RES / "sanity_cpu.json"),
        str(RES / "config_probe.json"),
        str(RES / "curve.tsv"),
        str(DIAG / "MECH-SEQUENTIAL.json"),
        str(DIAG / "MECH-SEQUENTIAL_route_mass.json"),
    ] + [f"{arms[a]['run_dir']}/cache_last.pt" for a in ARMS if arms[a]["run_dir"]],
    "command": "bash research_loop/results/MECH-SEQUENTIAL/launch_mech_sequential.sh",
    "observations": "TBD",
}

collector["eval_time_mass_on_S"] = {
    k: {"summary": v["summary"], "per_layer": v["per_layer"]}
    for k, v in route["runs"].items()
}
collector["mt_over_qa_mass_ratio"] = ratios
collector["phase1_reference_mass"] = phase1
collector["part_a_sanity"] = part_a
collector["cost_breakdown"] = {
    a: {
        "wall_clock_s": arms[a]["wall_clock_s"],
        "solve_s": arms[a]["solve_s"],
        "phase2_e2e_s": arms[a]["phase2_e2e_s"],
        "prefill_s_total": arms[a]["prefill_s_total"],
        "onpolicy_refresh_s_total": arms[a]["onpolicy_refresh_s_total"],
        "solve_s_multiple_vs_control": arms[a]["solve_s_multiple_vs_control"],
        "phase2_e2e_s_multiple_vs_control": arms[a]["phase2_e2e_s_multiple_vs_control"],
        "gradient_steps": 0,
    }
    for a in ARMS
}
collector["k_curve_standalone"] = {a: arms[a]["k_curve_standalone"] for a in ARMS}
(DIAG / "MECH-SEQUENTIAL.json").write_text(json.dumps(collector, indent=1))

(RES / "result.json").write_text(json.dumps(out, indent=2))
print("control repro in-run exact:", repro_inrun)
print("control standalone delta:", json.dumps(repro_standalone))
for a in ARMS:
    x = arms[a]
    print(
        a,
        "| in-run", {k: (round(v, 5) if v else v) for k, v in x["eval_in_run"].items()},
        "| standalone", {k: (round(v, 5) if v else v)
                         for k, v in x["eval_standalone"].items()},
        "| mse", x["am_mean_mse_over_docs"],
        "| |v|max", x["value_global_max_abs"],
        "| eval_mass_S", {k: round(v["mean_mass_on_S"], 5)
                          for k, v in x["eval_mass_on_S"].items()},
        "| cart_mass", {k: round(v, 5) for k, v in x["eval_total_cartridge_mass"].items()},
        "| MT/QA", x["eval_mt_over_qa_mass_ratio"],
        "| solve_s", x["solve_s"], "x", x["solve_s_multiple_vs_control"],
    )
print("onpolicy drift:", json.dumps(
    {a: (arms[a]["onpolicy"] or {}) for a in ARMS if arms[a]["onpolicy"]}, indent=1)[:2000])
print("k-curve:", json.dumps(collector["k_curve_standalone"], indent=1))
print("ratios:", json.dumps(ratios, indent=1))
print("phase1:", json.dumps(phase1, indent=1))
print("WROTE", RES / "result.json")
