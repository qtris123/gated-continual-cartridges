#!/usr/bin/env python
"""MECH-INFOGATE — assemble result.json + state/diagnostics/MECH-INFOGATE.json."""

from __future__ import annotations

import collections
import json
import os
import re

REPO = os.environ.get("CARTRIDGES_DIR", "/localhome/local-triv/gated-continual-cartridges_explore")
RES = os.path.join(REPO, "research_loop/results/MECH-INFOGATE")

# DIAG-NOISE measured paired resolution (NOT the old +-0.15 rule of thumb)
RES_MT, RES_QA = 0.049, 0.054
# the incumbent best point (MECH-005 / DIAG-KEYCURVE k=12) and the mission budget
INC = {"QA": 1.9560121297836304, "MT": 2.2720184326171875}
INC16 = {"QA": 2.034916400909424, "MT": 2.330503463745117}
BUDGET = {"QA": 2.52, "MT": 2.02}


def load_curve():
    rows = []
    for line in open(os.path.join(RES, "curve.tsv")).read().splitlines()[1:]:
        f = line.split("\t")
        rows.append({"arm": f[0], "selector": f[1], "top_t": int(f[2]), "k": int(f[3]),
                     "split": f[4], "loss": None if f[5] == "MISSING" else float(f[5]),
                     "ckpt": f[6], "wandb_run_url": f[7]})
    return rows


def main():
    rows = load_curve()
    arms = json.load(open(os.path.join(RES, "arms_analysis.json")))
    wrapper = open(os.path.join(RES, "wrapper.log")).read()

    # ---- per-arm k-curves ----------------------------------------------------
    curves: dict = collections.defaultdict(lambda: collections.defaultdict(dict))
    for r in rows:
        curves[r["arm"]][r["split"]][r["k"]] = r["loss"]

    per_arm = {}
    for arm, a in arms.items():
        c = {sp: dict(sorted(curves[arm][sp].items())) for sp in ("QA", "MT")}
        mtv = {k: v for k, v in c.get("MT", {}).items() if v is not None}
        best_k = min(mtv, key=mtv.get) if mtv else None
        per_arm[arm] = {
            "selector": a.get("selector"),
            "top_t": a.get("top_t"),
            "k_curve": c,
            "best_k_by_MT": best_k,
            "best_MT": mtv.get(best_k) if best_k else None,
            "QA_at_best_k": c.get("QA", {}).get(best_k) if best_k else None,
            "MT_at_k16": c.get("MT", {}).get(16),
            "QA_at_k16": c.get("QA", {}).get(16),
            "realised_frac_writable_MT_routing_mass": a.get("realised_frac_writable_MT_routing_mass"),
            "realised_frac_total_QA_Fisher_mass": a.get("realised_frac_total_QA_Fisher_mass"),
            "slots_selected_per_layer_per_doc": a.get("slots_selected_per_layer_per_doc"),
            "union_slots_per_layer": a.get("union_slots_per_layer"),
            "mass_on_S_union_MT_evalgeom": a.get("mass_on_S_union_MT_evalgeom"),
            "mass_on_S_union_QA_evalgeom": a.get("mass_on_S_union_QA_evalgeom"),
            "selectivity_MT_over_QA_evalgeom": a.get("selectivity_MT_over_QA_evalgeom"),
            "ref_mass_on_S_mean": a.get("ref_mass_on_S_mean"),
            "am_mean_mse": a.get("am_mean_mse"),
            "v_absmax_max": a.get("v_absmax_max"),
            "solve_s": a.get("solve_s"),
            "phase2_e2e_s": a.get("wall_s"),
            "gradient_steps": 0,
            "wandb_run_url": a.get("wandb_run_url"),
            "run_dir": a.get("run_dir"),
        }

    ctrl = per_arm.get("C0_control", {})
    ctrl_solve = ctrl.get("solve_s") or 1.0
    for arm, p in per_arm.items():
        p["solve_cost_multiple_vs_control"] = (
            round(p["solve_s"] / ctrl_solve, 3) if p.get("solve_s") else None
        )
        for sp, ref, res in (("MT", INC["MT"], RES_MT), ("QA", INC["QA"], RES_QA)):
            v = p["QA_at_best_k"] if sp == "QA" else p["best_MT"]
            if v is None:
                continue
            d = v - ref
            p[f"delta_{sp}_vs_incumbent_best"] = round(d, 5)
            p[f"delta_{sp}_clears_paired_resolution"] = bool(abs(d) > res)
            p[f"{sp}_meets_budget"] = bool(v <= BUDGET[sp])

    # ---- baseline reproduction block ----------------------------------------
    gate = re.search(r"CONTROL_GATE=(\S.*)", wrapper)
    repro = {
        "gate_line": gate.group(1).strip() if gate else None,
        "expected": {"k12": INC, "k16": INC16},
        "observed": {
            "k12": {"QA": curves["C0_control"]["QA"].get(12),
                    "MT": curves["C0_control"]["MT"].get(12)},
            "k16": {"QA": curves["C0_control"]["QA"].get(16),
                    "MT": curves["C0_control"]["MT"].get(16)},
        },
    }
    repro["bit_identical_when_default"] = bool(
        repro["gate_line"] and repro["gate_line"].startswith("GATE_PASS")
    )

    diagnostics = {
        "id": "MECH-INFOGATE",
        "mechanism": "MECH-008",
        "baseline_reproduced": repro,
        "paired_resolution_used": {"MT": RES_MT, "QA": RES_QA, "source": "DIAG-NOISE"},
        "incumbent_best_point": {"k": 12, **INC},
        "per_arm": per_arm,
        "raw_curve_rows": rows,
        "sanity": json.load(open(os.path.join(RES, "sanity.json"))),
        "arms_analysis": arms,
    }
    dpath = os.path.join(REPO, "research_loop/state/diagnostics/MECH-INFOGATE.json")
    with open(dpath, "w") as fh:
        json.dump(diagnostics, fh, indent=2)
    print(f"wrote {dpath}")

    # ---- console table -------------------------------------------------------
    print(f"\n{'arm':<14}{'sel':<19}{'t':>4}{'bestk':>6}{'MT':>9}{'QA':>9}"
          f"{'dMT':>9}{'dQA':>9}{'MTmass':>8}{'QAfish':>8}{'solve':>8}{'x':>6}")
    for arm, p in per_arm.items():
        print(f"{arm:<14}{str(p['selector']):<19}{p['top_t']:>4}{str(p['best_k_by_MT']):>6}"
              f"{(p['best_MT'] or 0):>9.4f}{(p['QA_at_best_k'] or 0):>9.4f}"
              f"{p.get('delta_MT_vs_incumbent_best', 0):>+9.4f}"
              f"{p.get('delta_QA_vs_incumbent_best', 0):>+9.4f}"
              f"{(p['realised_frac_writable_MT_routing_mass'] or 0):>8.3f}"
              f"{(p['realised_frac_total_QA_Fisher_mass'] or 0):>8.3f}"
              f"{(p['solve_s'] or 0):>8.0f}{(p['solve_cost_multiple_vs_control'] or 0):>6.2f}")
    return per_arm, repro


if __name__ == "__main__":
    main()
