#!/usr/bin/env python
"""MECH-INFOGATE post-hoc: what did each arm actually select, and what did it cost?

Read-only, CPU-only. For every arm's run directory it reads the 16 saved
`am_doc_*.pt` payloads (each holds the `GradientMask` the ranker produced for
that document, plus `am_stats.extra`) and reports, per arm:

  * realised_frac_writable_MT_routing_mass -- DIAG-IMPORTANCE's
    `selector_tradeoff_table` definition applied to the ACTUAL per-document
    selections: per layer, the share of the 511 writable slots' mean MT routing
    mass (`score_tf_mass_mt`) captured by the selected set, averaged over layers
    and documents. This is the "write bandwidth" axis of the measured Pareto
    trade, so it can be checked directly against the projection.
  * realised_frac_total_QA_Fisher_mass -- the same for `score_fisher` (QA), the
    "retention exposure" axis.
  * mass_on_S (eval-time geometry) -- attention mass landing on the UNION of all
    written slots, computed with DIAG-IMPORTANCE's `w_mass_{qa,mt}` (the real
    eval-time softmax including the prefix), for MT and QA queries, plus the
    MT/QA selectivity ratio.
  * ref_mass_on_S -- the in-run quantity the solve itself reports
    (`extra.ref_mass_on_S_per_layer`, `value_solve.py:146`), i.e. the mass on S
    under the REFERENCE queries.
  * mean_mse, |v|max, per-document wall clock.
"""

from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
import torch

REPO = os.environ.get("CARTRIDGES_DIR", "/localhome/local-triv/gated-continual-cartridges_explore")
sys.path.insert(0, REPO)
NPZ = os.path.join(REPO, "research_loop/state/diagnostics/DIAG-IMPORTANCE.npz")


def analyse_run_dir(run_dir: str) -> dict:
    z = np.load(NPZ)
    mt_mass = z["score_tf_mass_mt"]      # (36, 511) routing mass, MT queries
    qa_fisher = z["score_fisher"]        # (36, 511) diagonal Fisher, QA loss
    w_mt = z["score_w_mass_mt"]          # (36, 511) eval-time mass incl. prefix
    w_qa = z["score_w_mass_qa"]
    n_layers, n_slots = mt_mass.shape

    files = sorted(glob.glob(os.path.join(run_dir, "am_doc_*.pt")))
    if not files:
        return {"error": f"no am_doc_*.pt in {run_dir}"}

    per_doc_mt, per_doc_fi, per_doc_sz = [], [], []
    union = [set() for _ in range(n_layers)]
    ref_mass, mean_mse, vmax, doc_s = [], [], [], []

    for f in files:
        d = torch.load(f, map_location="cpu", weights_only=False)
        mask = d["ranking_info"].mask
        pos = mask.positions_per_layer
        mt_l, fi_l, sz_l = [], [], []
        for l in range(n_layers):
            p = pos.get(l)
            if p is None:
                continue
            idx = np.asarray(torch.as_tensor(p).flatten().tolist(), dtype=int)
            idx = np.unique(idx)
            union[l].update(idx.tolist())
            mt_l.append(mt_mass[l, idx].sum() / mt_mass[l].sum())
            fi_l.append(qa_fisher[l, idx].sum() / qa_fisher[l].sum())
            sz_l.append(len(idx))
        per_doc_mt.append(float(np.mean(mt_l)))
        per_doc_fi.append(float(np.mean(fi_l)))
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


def main() -> int:
    rundirs = os.path.join(REPO, "research_loop/results/MECH-INFOGATE/rundirs.tsv")
    out = {}
    for line in open(rundirs).read().splitlines()[1:]:
        f = line.split("\t")
        arm, sel, topt, run_dir = f[0], f[1], f[2], f[3]
        if not run_dir or not os.path.isdir(run_dir):
            out[arm] = {"error": f"missing run_dir {run_dir!r}"}
            continue
        r = analyse_run_dir(run_dir)
        r.update({"selector": sel, "top_t": int(topt), "rc": int(f[4]),
                  "wall_s": float(f[5]), "solve_s": float(f[6]), "wandb_run_url": f[7]})
        out[arm] = r
        print(f"{arm:>12} sel={sel:>18} t={topt:>3}  "
              f"MTmass={r['realised_frac_writable_MT_routing_mass']:.4f}  "
              f"QAfisher={r['realised_frac_total_QA_Fisher_mass']:.4f}  "
              f"massS(MT/QA)={r['mass_on_S_union_MT_evalgeom']:.4f}/"
              f"{r['mass_on_S_union_QA_evalgeom']:.4f}="
              f"{r['selectivity_MT_over_QA_evalgeom']:.4f}  "
              f"mse={r['am_mean_mse']:.4f}  |v|max={r['v_absmax_max']}  "
              f"solve={r['solve_s']:.0f}s")
    dest = os.path.join(REPO, "research_loop/results/MECH-INFOGATE/arms_analysis.json")
    with open(dest, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
