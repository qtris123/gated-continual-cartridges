#!/usr/bin/env python
"""VERIFY-GATE post-hoc: what did each seed-offset arm actually select and write?

Read-only, CPU-only. Same definitions as MECH-INFOGATE/analyse_arms.py, so the
numbers are directly comparable to the published B-GATE table:

  * realised_frac_writable_MT_routing_mass -- per layer, the share of the 511
    writable slots' mean MT routing mass (`score_tf_mass_mt`) captured by the
    ACTUAL per-document selection, averaged over layers and documents. This is
    the quantity MECH-INFOGATE found predicts best-MT loss at Pearson -0.877.
  * realised_frac_total_QA_Fisher_mass -- the retention-exposure axis.
  * mass_on_S (eval-time geometry) for the UNION of written slots, MT and QA,
    plus the MT/QA selectivity ratio.
  * ref_mass_on_S -- the in-run quantity the solve reports (value_solve.py:146).
  * mean_mse, |v|max, wall clock.
"""

from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
import torch

REPO = "/localhome/local-triv/gated-continual-cartridges_explore"
sys.path.insert(0, os.environ.get("VG_SNAP", "/tmp/amsnap_verifygate"))
NPZ = os.path.join(REPO, "research_loop/state/diagnostics/DIAG-IMPORTANCE.npz")
RESDIR = os.path.join(REPO, "research_loop/results/VERIFY-GATE")


def analyse_run_dir(run_dir: str) -> dict:
    z = np.load(NPZ)
    mt_mass = z["score_tf_mass_mt"]
    qa_fisher = z["score_fisher"]
    w_mt = z["score_w_mass_mt"]
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
            idx = np.unique(np.asarray(torch.as_tensor(p).flatten().tolist(), dtype=int))
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
        "union_slots_per_layer": u_sz,
        "union_frac_writable_slots": u_sz / n_slots,
        "mass_on_S_union_MT_evalgeom": u_mt,
        "mass_on_S_union_QA_evalgeom": u_qa,
        "selectivity_MT_over_QA_evalgeom": (u_mt / u_qa) if u_qa else None,
        "ref_mass_on_S_mean": float(np.mean(ref_mass)) if ref_mass else None,
        "ref_mass_on_S_per_doc": [round(x, 5) for x in ref_mass],
        "am_mean_mse": float(np.mean(mean_mse)),
        "v_absmax_max": float(max(vmax)) if vmax else None,
        "doc_wall_s_total": float(np.sum(doc_s)),
    }


def main() -> int:
    rundirs = os.path.join(RESDIR, "rundirs.tsv")
    out = {}
    for line in open(rundirs).read().splitlines()[1:]:
        f = line.split("\t")
        arm, sel, off, run_dir = f[0], f[1], f[2], f[3]
        if not run_dir or not os.path.isdir(run_dir):
            out[arm] = {"error": f"missing run_dir {run_dir!r}"}
            continue
        r = analyse_run_dir(run_dir)
        r.update({"selector": sel, "seed_offset": int(off), "rc": int(f[4]),
                  "wall_s": float(f[5]), "solve_s": float(f[6]), "wandb_run_url": f[7]})
        out[arm] = r
        print(f"{arm:>12} sel={sel:>12} off={off:>5}  "
              f"MTmass={r['realised_frac_writable_MT_routing_mass']:.4f}  "
              f"QAfisher={r['realised_frac_total_QA_Fisher_mass']:.4f}  "
              f"massS(MT/QA)={r['mass_on_S_union_MT_evalgeom']:.4f}/"
              f"{r['mass_on_S_union_QA_evalgeom']:.4f}="
              f"{r['selectivity_MT_over_QA_evalgeom']:.4f}  "
              f"refmassS={r['ref_mass_on_S_mean']:.4f}  "
              f"mse={r['am_mean_mse']:.5f}  |v|max={r['v_absmax_max']}  "
              f"solve={r['solve_s']:.0f}s")

    # the MECH-INFOGATE offset-0 arms, recomputed here so the comparison is on
    # one code path (C0_control is the tfidf offset-0 arm this bundle reuses).
    for arm, rd in {
        "T_off0_MECH-INFOGATE_C0": "/localhome/local-triv/gated-continual-cartridges_explore/outputs/2026-07-29-07-17-36-continual_am_sparse/c28044a8-c9d7-452c-a18e-44117e5f6f7b",
        "R_off0_MECH-INFOGATE_R32": "/localhome/local-triv/gated-continual-cartridges_explore/outputs/2026-07-29-07-52-51-continual_am_sparse/ef58d93f-2b0f-47b8-98a9-2fcde6f59484",
    }.items():
        if os.path.isdir(rd):
            out[arm] = analyse_run_dir(rd)
            r = out[arm]
            print(f"{arm:>28}  MTmass={r['realised_frac_writable_MT_routing_mass']:.4f}  "
                  f"QAfisher={r['realised_frac_total_QA_Fisher_mass']:.4f}  "
                  f"refmassS={r['ref_mass_on_S_mean']:.4f}")

    dest = os.path.join(RESDIR, "arms_analysis.json")
    with open(dest, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
