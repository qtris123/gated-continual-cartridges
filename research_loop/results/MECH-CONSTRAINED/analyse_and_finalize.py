#!/usr/bin/env python
"""MECH-CONSTRAINED post-hoc: what did each arm actually select, and what did it cost?

Read-only, CPU-only. Two halves:

  A. `analyse_run_dir` -- for every arm's run directory, read the 16 saved
     `am_doc_*.pt` payloads (each holds the `GradientMask` the ranker produced
     for that document plus `am_stats.extra`) and report DIAG-IMPORTANCE's two
     Pareto axes applied to the ACTUAL per-document selections:
       * realised_frac_writable_MT_routing_mass  (write bandwidth)
       * realised_frac_total_QA_Fisher_mass      (retention exposure)
     plus eval-geometry `mass_on_S` on the union of written slots, the in-run
     `ref_mass_on_S`, `am/mean_mse`, `|v|max` and per-document wall clock.
     Definitions are byte-identical to MECH-INFOGATE's `analyse_arms.py`, so the
     numbers drop straight into its published table.

  B. assemble `state/diagnostics/MECH-CONSTRAINED.json` (k-curves, own-optimum
     comparison against the incumbent, the `baseline_reproduced` gate block, the
     dose-response test, provenance) and print the console table.
"""

from __future__ import annotations

import collections
import glob
import json
import os
import re
import sys

import numpy as np
import torch

REPO = os.environ.get(
    "CARTRIDGES_DIR", "/localhome/local-triv/gated-continual-cartridges_explore"
)
sys.path.insert(0, REPO)
RES = os.path.join(REPO, "research_loop/results/MECH-CONSTRAINED")
NPZ = os.path.join(REPO, "research_loop/state/diagnostics/DIAG-IMPORTANCE.npz")

# DIAG-NOISE measured PAIRED resolution (not the old +-0.15 rule of thumb)
RES_MT, RES_QA = 0.049, 0.054
# the incumbent best point (MECH-005 / DIAG-KEYCURVE) and the mission budget
INC = {"QA": 1.9560121297836304, "MT": 2.2720184326171875}
INC16 = {"QA": 2.034916400909424, "MT": 2.330503463745117}
BUDGET = {"QA": 2.52, "MT": 2.02}


# ============================================================ A. what was written
def analyse_run_dir(run_dir: str) -> dict:
    z = np.load(NPZ)
    mt_mass = z["score_tf_mass_mt"]      # (36, 511) routing mass, MT queries
    qa_fisher = z["score_fisher"]        # (36, 511) diagonal Fisher, QA loss
    w_mt = z["score_w_mass_mt"]          # (36, 511) eval-time mass incl. prefix
    w_qa = z["score_w_mass_qa"]
    red = z["score_redundancy"]
    n_layers, n_slots = mt_mass.shape

    files = sorted(glob.glob(os.path.join(run_dir, "am_doc_*.pt")))
    if not files:
        return {"error": f"no am_doc_*.pt in {run_dir}"}

    per_doc_mt, per_doc_fi, per_doc_sz, per_doc_red = [], [], [], []
    union = [set() for _ in range(n_layers)]
    ref_mass, mean_mse, vmax, doc_s = [], [], [], []

    for f in files:
        d = torch.load(f, map_location="cpu", weights_only=False)
        pos = d["ranking_info"].mask.positions_per_layer
        mt_l, fi_l, sz_l, rd_l = [], [], [], []
        for l in range(n_layers):
            p = pos.get(l)
            if p is None:
                continue
            idx = np.unique(np.asarray(torch.as_tensor(p).flatten().tolist(), dtype=int))
            union[l].update(idx.tolist())
            mt_l.append(mt_mass[l, idx].sum() / mt_mass[l].sum())
            fi_l.append(qa_fisher[l, idx].sum() / qa_fisher[l].sum())
            rd_l.append(red[l, idx].mean())
            sz_l.append(len(idx))
        per_doc_mt.append(float(np.mean(mt_l)))
        per_doc_fi.append(float(np.mean(fi_l)))
        per_doc_red.append(float(np.mean(rd_l)))
        per_doc_sz.append(float(np.mean(sz_l)))

        extra = d["am_stats"].extra or {}
        rm = extra.get("ref_mass_on_S_per_layer")
        if rm:
            ref_mass.append(float(np.mean(list(rm.values()))))
        vm = extra.get("v_selected_absmax_after_per_layer")
        if vm:
            vmax.append(float(max(vm.values())))
        mean_mse.append(float(d["am_stats"].mean_mse))
        doc_s.append(float(d["doc_record"]["timing_s"]["total_s"]))

    u_mt = float(np.mean([w_mt[l, sorted(union[l])].sum() for l in range(n_layers)]))
    u_qa = float(np.mean([w_qa[l, sorted(union[l])].sum() for l in range(n_layers)]))
    u_sz = float(np.mean([len(union[l]) for l in range(n_layers)]))

    return {
        "run_dir": run_dir,
        "n_documents": len(files),
        "slots_selected_per_layer_per_doc": float(np.mean(per_doc_sz)),
        "realised_frac_writable_MT_routing_mass": float(np.mean(per_doc_mt)),
        "realised_frac_total_QA_Fisher_mass": float(np.mean(per_doc_fi)),
        "mean_redundancy_of_selected": float(np.mean(per_doc_red)),
        "realised_MT_mass_per_doc": [round(x, 5) for x in per_doc_mt],
        "realised_QA_Fisher_per_doc": [round(x, 5) for x in per_doc_fi],
        "union_slots_per_layer": u_sz,
        "union_frac_writable_slots": u_sz / n_slots,
        "mass_on_S_union_MT_evalgeom": u_mt,
        "mass_on_S_union_QA_evalgeom": u_qa,
        "selectivity_MT_over_QA_evalgeom": (u_mt / u_qa) if u_qa else None,
        "ref_mass_on_S_mean": float(np.mean(ref_mass)) if ref_mass else None,
        "am_mean_mse": float(np.mean(mean_mse)),
        "am_mean_mse_per_doc": [round(x, 6) for x in mean_mse],
        "v_absmax_max": float(max(vmax)) if vmax else None,
        "doc_wall_s_total": float(np.sum(doc_s)),
        "doc_wall_s_mean": float(np.mean(doc_s)),
    }


def load_curve():
    rows = []
    for line in open(os.path.join(RES, "curve.tsv")).read().splitlines()[1:]:
        f = line.split("\t")
        rows.append({
            "arm": f[0], "selector": f[1], "safe_fraction": float(f[2]),
            "top_t": int(f[3]), "k": int(f[4]), "split": f[5],
            "loss": None if f[6] == "MISSING" else float(f[6]),
            "ckpt": f[7], "wandb_run_url": f[8],
        })
    return rows


def main() -> int:
    from scipy.stats import pearsonr, spearmanr

    arms = {}
    order = []
    for line in open(os.path.join(RES, "rundirs.tsv")).read().splitlines()[1:]:
        f = line.split("\t")
        arm, sel, q, topt, run_dir = f[0], f[1], float(f[2]), int(f[3]), f[4]
        order.append(arm)
        if not run_dir or not os.path.isdir(run_dir):
            arms[arm] = {"error": f"missing run_dir {run_dir!r}"}
            continue
        r = analyse_run_dir(run_dir)
        r.update({"selector": sel, "safe_fraction": q, "top_t": topt,
                  "n_candidate_slots_per_layer": (
                      511 if sel != "constrained_mass"
                      else max(topt, int(np.floor(q * 511)))),
                  "rc": int(f[5]), "wall_s": float(f[6]), "solve_s": float(f[7]),
                  "wandb_run_url": f[8]})
        arms[arm] = r
    with open(os.path.join(RES, "arms_analysis.json"), "w") as fh:
        json.dump(arms, fh, indent=2)

    rows = load_curve()
    wrapper = open(os.path.join(RES, "wrapper.log")).read()
    curves: dict = collections.defaultdict(lambda: collections.defaultdict(dict))
    for r in rows:
        curves[r["arm"]][r["split"]][r["k"]] = r["loss"]

    def best(arm, sp):
        d = {k: v for k, v in curves[arm][sp].items() if v is not None}
        if not d:
            return None, None
        k = min(d, key=d.get)
        return d[k], k

    per_arm = {}
    for arm in order:
        a = arms[arm]
        if "error" in a:
            per_arm[arm] = a
            continue
        c = {sp: dict(sorted(curves[arm][sp].items())) for sp in ("QA", "MT")}
        mt, mtk = best(arm, "MT")
        qa, qak = best(arm, "QA")
        per_arm[arm] = {
            "selector": a["selector"],
            "safe_fraction": a["safe_fraction"],
            "top_t": a["top_t"],
            "n_candidate_slots_per_layer": a["n_candidate_slots_per_layer"],
            "k_curve_MT": c["MT"], "k_curve_QA": c["QA"],
            "own_optimum": {"MT": mt, "MT_k": mtk, "QA": qa, "QA_k": qak},
            "realised_selection": {
                "frac_writable_MT_routing_mass":
                    a["realised_frac_writable_MT_routing_mass"],
                "frac_total_QA_Fisher_mass": a["realised_frac_total_QA_Fisher_mass"],
                "mean_redundancy_of_selected": a["mean_redundancy_of_selected"],
                "slots_per_layer_per_doc": a["slots_selected_per_layer_per_doc"],
                "union_slots_per_layer": a["union_slots_per_layer"],
                "union_frac_of_511_writable": a["union_frac_writable_slots"],
            },
            "mass_on_S": {
                "union_MT_phase1_evalgeom": a["mass_on_S_union_MT_evalgeom"],
                "union_QA_phase1_evalgeom": a["mass_on_S_union_QA_evalgeom"],
                "selectivity_MT_over_QA": a["selectivity_MT_over_QA_evalgeom"],
                "ref_mass_on_S_mean_in_run": a["ref_mass_on_S_mean"],
            },
            "am_mean_mse": a["am_mean_mse"],
            "v_absmax_max": a["v_absmax_max"],
            "solve_s": a["solve_s"],
            "phase2_e2e_s": a["wall_s"],
            "gradient_steps": 0,
            "wandb_run_url": a["wandb_run_url"],
            "run_dir": a["run_dir"],
        }

    ctrl = per_arm.get("C0_control", {})
    csolve = ctrl.get("solve_s") or 1.0
    cmt, cmtk = best("C0_control", "MT")
    cqa, cqak = best("C0_control", "QA")
    for arm, p in per_arm.items():
        if "error" in p:
            continue
        p["solve_cost_multiple_vs_control"] = round(p["solve_s"] / csolve, 3)
        mt, qa = p["own_optimum"]["MT"], p["own_optimum"]["QA"]
        dmt, dqa = (mt - cmt) if mt is not None else None, (qa - cqa) if qa is not None else None
        p["vs_control_at_own_optima"] = {
            "dMT": round(dmt, 5) if dmt is not None else None,
            "dMT_clears_0.049": bool(dmt is not None and abs(dmt) > RES_MT),
            "dQA": round(dqa, 5) if dqa is not None else None,
            "dQA_clears_0.054": bool(dqa is not None and abs(dqa) > RES_QA),
            "MT_meets_budget": bool(mt is not None and mt <= BUDGET["MT"]),
            "QA_meets_budget": bool(qa is not None and qa <= BUDGET["QA"]),
        }

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
        repro["gate_line"] and repro["gate_line"].startswith("GATE_PASS"))

    # ------------------------------- the dose-response test the brief asks for
    sweep_arms = [a for a in ("C0_control", "Q75", "Q50", "Q25")
                  if a in per_arm and "error" not in per_arm[a]]
    qs = [per_arm[a]["safe_fraction"] for a in sweep_arms]
    mts = [per_arm[a]["own_optimum"]["MT"] for a in sweep_arms]
    mass = [per_arm[a]["realised_selection"]["frac_writable_MT_routing_mass"]
            for a in sweep_arms]
    fish = [per_arm[a]["realised_selection"]["frac_total_QA_Fisher_mass"]
            for a in sweep_arms]
    qas = [per_arm[a]["own_optimum"]["QA"] for a in sweep_arms]
    dose = {
        "_definition": (
            "TOP_T=32 held fixed; only the constraint strength q moves "
            "(1.00 = the incumbent tfidf control, then 0.75 / 0.50 / 0.25). "
            "A monotone degradation as q tightens, tracking the realised MT "
            "routing mass, is the dose-response form of MECH-INFOGATE's "
            "log(mass) -> MT relationship (Pearson -0.877 over its six arms)."
        ),
        "arms": sweep_arms,
        "q": qs,
        "best_MT": mts,
        "best_QA": qas,
        "realised_MT_routing_mass": mass,
        "realised_QA_Fisher_exposure": fish,
        "MT_monotone_worsening_as_q_tightens": all(
            mts[i] <= mts[i + 1] + 1e-12 for i in range(len(mts) - 1)),
        "MT_mass_monotone_falling_as_q_tightens": all(
            mass[i] >= mass[i + 1] - 1e-12 for i in range(len(mass) - 1)),
        "QA_Fisher_monotone_falling_as_q_tightens": all(
            fish[i] >= fish[i + 1] - 1e-12 for i in range(len(fish) - 1)),
    }
    if len(sweep_arms) >= 3:
        dose["pearson_logMTmass_vs_bestMT_sweep_only"] = float(
            pearsonr(np.log(mass), mts)[0])
        dose["spearman_q_vs_bestMT"] = float(spearmanr(qs, mts).statistic)

    # ---- pooled with MECH-INFOGATE's six arms (11 points on one relationship)
    pooled = {}
    try:
        mi = json.load(open(os.path.join(
            REPO, "research_loop/results/MECH-INFOGATE/result.json")))
        pm, pmt, pf, pqa, labels = [], [], [], [], []
        for k, v in mi["arms"].items():
            pm.append(v["realised_selection"]["frac_writable_MT_routing_mass"])
            pf.append(v["realised_selection"]["frac_total_QA_Fisher_mass"])
            pmt.append(v["own_optimum"]["MT"])
            pqa.append(v["own_optimum"]["QA"])
            labels.append("MECH-INFOGATE:" + k)
        for a in per_arm:
            if "error" in per_arm[a] or a == "C0_control":
                continue
            pm.append(per_arm[a]["realised_selection"]["frac_writable_MT_routing_mass"])
            pf.append(per_arm[a]["realised_selection"]["frac_total_QA_Fisher_mass"])
            pmt.append(per_arm[a]["own_optimum"]["MT"])
            pqa.append(per_arm[a]["own_optimum"]["QA"])
            labels.append("MECH-CONSTRAINED:" + a)
        pooled = {
            "_definition": (
                "MECH-INFOGATE's six arms plus this bundle's four new selectors, "
                "on the identical axes and the identical control."),
            "n_points": len(pm),
            "labels": labels,
            "pearson_logMTmass_vs_bestMT": float(pearsonr(np.log(pm), pmt)[0]),
            "spearman_MTmass_vs_bestMT": float(spearmanr(pm, pmt).statistic),
            "pearson_QAfisher_vs_bestQA": float(pearsonr(pf, pqa)[0]),
            "spearman_QAfisher_vs_bestQA": float(spearmanr(pf, pqa).statistic),
            "MECH_INFOGATE_published_pearson_logMTmass_vs_bestMT": -0.8766490587087726,
        }
    except Exception as e:  # pragma: no cover
        pooled = {"error": repr(e)}

    diagnostics = {
        "id": "MECH-CONSTRAINED",
        "mechanism": "MECH-009",
        "selector": (
            "constrained_mass: per layer, restrict to the safest floor(q*511) "
            "slots by redundancy (descending), then take the top_t by attention "
            "mass within that candidate set."),
        "baseline_reproduced": repro,
        "paired_resolution_used": {"MT": RES_MT, "QA": RES_QA, "source": "DIAG-NOISE"},
        "incumbent_best_point": {"k": 12, **INC},
        "control_own_optimum": {"MT": cmt, "MT_k": cmtk, "QA": cqa, "QA_k": cqak},
        "budget": BUDGET,
        "per_arm": per_arm,
        "dose_response": dose,
        "pooled_with_MECH_INFOGATE": pooled,
        "raw_curve_rows": rows,
        "sanity": json.load(open(os.path.join(RES, "sanity.json"))),
        "arms_analysis": arms,
        "provenance": {
            "repo": REPO,
            "branch": "trivo-explore-research-work",
            "head_at_launch": (re.search(r"REPO_HEAD_AT_LAUNCH=(\S+)", wrapper) or [None, None])[1],
            "head_after": (re.search(r"REPO_HEAD_AFTER=(\S+)", wrapper) or [None, None])[1],
            "import_probe_from_tmp": (re.search(r"IMPORT_PROBE_FROM_TMP=(\S+)", wrapper) or [None, None])[1],
            "gpu": (re.search(r"MC_CLAIMED_GPU=(\d)", wrapper) or [None, None])[1],
            "phase1_cache": "outputs/phase1_selfdistill_qwen512/cache_last.pt",
            "synth_data": "data/qasper/train/qwen_qasper_MT_task_8192.parquet",
            "diag_importance_npz": NPZ,
            "worktree_diff": os.path.join(RES, "worktree_diff.patch"),
        },
    }
    dpath = os.path.join(REPO, "research_loop/state/diagnostics/MECH-CONSTRAINED.json")
    with open(dpath, "w") as fh:
        json.dump(diagnostics, fh, indent=2)
    print(f"wrote {dpath}")

    print(f"\n{'arm':<12}{'sel':<18}{'q':>6}{'t':>5}{'cand':>6}{'bestk':>6}"
          f"{'MT':>9}{'QA':>9}{'dMT':>9}{'dQA':>9}{'MTmass':>8}{'QAfish':>8}"
          f"{'mse':>9}{'|v|max':>8}{'solve':>8}")
    for arm in order:
        p = per_arm[arm]
        if "error" in p:
            print(f"{arm:<12} ERROR {p['error']}")
            continue
        v = p["vs_control_at_own_optima"]
        print(f"{arm:<12}{p['selector']:<18}{p['safe_fraction']:>6.2f}{p['top_t']:>5}"
              f"{p['n_candidate_slots_per_layer']:>6}{str(p['own_optimum']['MT_k']):>6}"
              f"{p['own_optimum']['MT']:>9.4f}{p['own_optimum']['QA']:>9.4f}"
              f"{v['dMT']:>+9.4f}{v['dQA']:>+9.4f}"
              f"{p['realised_selection']['frac_writable_MT_routing_mass']:>8.4f}"
              f"{p['realised_selection']['frac_total_QA_Fisher_mass']:>8.4f}"
              f"{p['am_mean_mse']:>9.5f}{(p['v_absmax_max'] or 0):>8.1f}"
              f"{p['solve_s']:>8.1f}")
    print(f"\nGATE: {repro['gate_line']}")
    print(f"DOSE-RESPONSE: MT monotone worsening as q tightens = "
          f"{dose['MT_monotone_worsening_as_q_tightens']}; "
          f"MT mass monotone falling = {dose['MT_mass_monotone_falling_as_q_tightens']}")
    print(f"  q            = {dose['q']}")
    print(f"  best MT      = {[round(x,4) for x in dose['best_MT']]}")
    print(f"  best QA      = {[round(x,4) for x in dose['best_QA']]}")
    print(f"  MT mass      = {[round(x,4) for x in dose['realised_MT_routing_mass']]}")
    print(f"  QA Fisher    = {[round(x,4) for x in dose['realised_QA_Fisher_exposure']]}")
    if "pearson_logMTmass_vs_bestMT_sweep_only" in dose:
        print(f"  pearson log(MTmass) vs best MT (sweep) = "
              f"{dose['pearson_logMTmass_vs_bestMT_sweep_only']:.4f}")
    if "pearson_logMTmass_vs_bestMT" in pooled:
        print(f"POOLED n={pooled['n_points']}: pearson log(MTmass) vs bestMT = "
              f"{pooled['pearson_logMTmass_vs_bestMT']:.4f} "
              f"(MECH-INFOGATE alone: -0.8766)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
