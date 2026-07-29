#!/usr/bin/env python
"""DIAG-CONTROLCURVE finalizer.

Builds:
  research_loop/state/diagnostics/DIAG-CONTROLCURVE.json   (full curves + fair comparison + provenance)
  research_loop/results/DIAG-CONTROLCURVE/result.json      (WORKERS.md bundle)

Reads nothing from the GPU: pure post-processing of curve.tsv, the eval logs, the
route-mass json, and DIAG-KEYCURVE's bundle (for the keys arm, reused as-is).
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys

REPO = "/localhome/local-triv/gated-continual-cartridges_explore"
RESDIR = f"{REPO}/research_loop/results/DIAG-CONTROLCURVE"
DIAGDIR = f"{REPO}/research_loop/state/diagnostics"
BAND = 0.15  # the loop's stated +-0.15 band (MT n=69 / QA n=78)

KEYCURVE = json.load(open(f"{REPO}/research_loop/results/DIAG-KEYCURVE/result.json"))
KCD = KEYCURVE["diagnostics"]

# --------------------------------------------------------------------------- curves
rows = []
for line in open(f"{RESDIR}/curve.tsv").read().splitlines()[1:]:
    parts = line.split("\t")
    if len(parts) < 6:
        continue
    arm, k, split, loss, ckpt, url = parts[:6]
    rows.append({"arm": arm, "k": int(k), "split": split,
                 "loss": None if loss == "MISSING" else float(loss),
                 "loss_str": loss, "ckpt": ckpt,
                 "wandb_url": None if url == "MISSING" else url})

by = {}
for r in rows:
    by.setdefault((r["arm"], r["k"], r["split"]), r)

ctrl_mt = {r["k"]: r["loss"] for r in rows if r["arm"] == "control" and r["split"] == "MT"}
ctrl_qa = {r["k"]: r["loss"] for r in rows if r["arm"] == "control" and r["split"] == "QA"}
p1 = {r["split"]: r["loss"] for r in rows if r["arm"] == "phase1"}

# --------------------------------------------------------------------------- gates
GATE_EXPECT = {
    ("control", 16, "QA"): "2.159724712371826",
    ("control", 16, "MT"): "2.529625177383423",
    ("phase1", 0, "QA"): "2.23880672454834",
    ("phase1", 0, "MT"): "3.7825491428375244",
}
gates = {}
gate_pass = True
for (arm, k, sp), want in GATE_EXPECT.items():
    have = by.get((arm, k, sp), {}).get("loss_str", "MISSING")
    ok = have == want
    gate_pass &= ok
    gates[f"{arm}_k{k}_{sp}"] = {"expected": want, "observed": have, "exact_match": ok}

# ------------------------------------------------- reproduction of DIAG-KEYCURVE's 4 k
repro = {}
for k in (8, 12, 14, 16):
    for sp, src in (("MT", KCD["control_freeze_mt_by_k"]), ("QA", KCD["control_freeze_qa_by_k"])):
        prev = src[str(k)]
        now = (ctrl_mt if sp == "MT" else ctrl_qa).get(k)
        repro[f"k{k}_{sp}"] = {
            "DIAG_KEYCURVE": prev, "DIAG_CONTROLCURVE": now,
            "delta": None if now is None else now - prev,
            "exact_match": now is not None and repr(now) == repr(prev),
        }

# --------------------------------------------------------------------------- argmins
def argmin(d):
    ks = [k for k in d if d[k] is not None]
    if not ks:
        return None, None
    kb = min(ks, key=lambda k: d[k])
    return kb, d[kb]

c_mt_k, c_mt_v = argmin(ctrl_mt)
c_qa_k, c_qa_v = argmin(ctrl_qa)

keys_mt = {int(k): v for k, v in KCD["keys_repos_mt_by_k"].items()}
keys_qa = {int(k): v for k, v in KCD["keys_repos_qa_by_k"].items()}
k_mt_k, k_mt_v = argmin(keys_mt)
k_qa_k, k_qa_v = argmin(keys_qa)

# --------------------------------------------------------------- per-k keys vs control
per_k = []
for k in range(1, 17):
    if ctrl_mt.get(k) is None or ctrl_qa.get(k) is None:
        continue
    d_mt = keys_mt[k] - ctrl_mt[k]
    d_qa = keys_qa[k] - ctrl_qa[k]
    per_k.append({
        "k": k,
        "keys_qa": keys_qa[k], "control_qa": ctrl_qa[k], "d_qa": d_qa,
        "keys_mt": keys_mt[k], "control_mt": ctrl_mt[k], "d_mt": d_mt,
        "d_qa_clears_band": abs(d_qa) > BAND,
        "d_mt_clears_band": abs(d_mt) > BAND,
        "d_qa_favours_keys": d_qa < 0,
        "d_mt_favours_keys": d_mt < 0,
    })

# ---------------------------------------------------- fair each-arm-at-its-own-best
fair = {
    "keys_best_k": k_mt_k, "keys_best_mt": k_mt_v, "keys_qa_at_that_k": keys_qa[k_mt_k],
    "control_best_k": c_mt_k, "control_best_mt": c_mt_v, "control_qa_at_that_k": ctrl_qa[c_mt_k],
    "d_mt_at_own_bests": k_mt_v - c_mt_v,
    "d_qa_at_own_bests": keys_qa[k_mt_k] - ctrl_qa[c_mt_k],
    "control_k_now_fully_mapped": sorted(ctrl_mt),
    "prev_control_best_k_MEASURED_by_DIAG_KEYCURVE": 8,
    "prev_control_best_mt_MEASURED_by_DIAG_KEYCURVE": KCD["control_freeze_mt_by_k"]["8"],
    "prev_d_mt_at_own_bests": KCD["keys_vs_control_at_each_arms_own_best_k"]["d_mt_at_own_bests"],
    "prev_d_qa_at_own_bests": KCD["keys_vs_control_at_each_arms_own_best_k"]["d_qa_at_own_bests"],
}
fair["d_mt_clears_band"] = abs(fair["d_mt_at_own_bests"]) > BAND
fair["d_qa_clears_band"] = abs(fair["d_qa_at_own_bests"]) > BAND
# also: each arm at its own QA-best
fair["qa_axis"] = {
    "keys_best_qa_k": k_qa_k, "keys_best_qa": k_qa_v,
    "control_best_qa_k": c_qa_k, "control_best_qa": c_qa_v,
    "d_qa_at_own_qa_bests": k_qa_v - c_qa_v,
    "d_qa_at_own_qa_bests_clears_band": abs(k_qa_v - c_qa_v) > BAND,
}
# and the strictest honest framing: control's best MT vs keys' best MT, plus the
# control's best point on EITHER axis
fair["control_pareto_note"] = {
    "control_argmin_mt_k": c_mt_k, "control_argmin_mt": c_mt_v,
    "control_argmin_qa_k": c_qa_k, "control_argmin_qa": c_qa_v,
}

# --------------------------------------------------------------------------- route mass
RM_PATH = f"{DIAGDIR}/DIAG-CONTROLCURVE_route_mass.json"
mass = {}
if os.path.exists(RM_PATH):
    rm = json.load(open(RM_PATH))
    for label, r in rm["ratios"].items():
        mass[label] = r
ctrl_mass_mt = {}
ctrl_mass_qa = {}
ctrl_ratio = {}
ctrl_cart_mt = {}
ctrl_cart_qa = {}
for k in range(1, 17):
    lab = f"control_k{k}"
    if lab in mass:
        ctrl_mass_mt[k] = mass[lab]["mass_on_S_MT"]
        ctrl_mass_qa[k] = mass[lab]["mass_on_S_QA"]
        ctrl_ratio[k] = mass[lab]["mt_over_qa_mass_on_S"]
        ctrl_cart_mt[k] = mass[lab]["mass_on_cart_MT"]
        ctrl_cart_qa[k] = mass[lab]["mass_on_cart_QA"]

# written-support union per k for the control (from am_doc payloads)
union_by_k = {}
try:
    import torch
    from collections import defaultdict
    from glob import glob
    DIR_C = f"{REPO}/outputs/2026-07-28-23-19-56-continual_am_sparse/508b8b0f-075a-429b-b757-2b11bfb57124"
    files = sorted(glob(os.path.join(DIR_C, "am_doc_*.pt")))
    per_layer = defaultdict(set)
    for i, f in enumerate(files, start=1):
        payload = torch.load(f, map_location="cpu", weights_only=False)
        m = payload["am_stats"].mask
        for layer_idx, pos in m.positions_per_layer.items():
            per_layer[int(layer_idx)].update(pos.flatten().tolist())
        union_by_k[i] = sum(len(v) for v in per_layer.values()) / len(per_layer)
except Exception as e:  # pragma: no cover
    union_by_k = {"error": str(e)}

# --------------------------------------------------------------------------- provenance
def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout.strip()
    except Exception as e:
        return f"ERR {e}"

wl = open(f"{RESDIR}/wrapper.log").read()
def grab(pat, default=None):
    m = re.search(pat, wl)
    return m.group(1) if m else default

# confirm every eval loaded its intended checkpoint
ckpt_check = {"n_evals": 0, "n_confirmed": 0, "mismatches": []}
for r in rows:
    log = f"{RESDIR}/evals/eval_{r['arm']}_doc{r['k']}_{r['split']}.log"
    ckpt_check["n_evals"] += 1
    try:
        txt = open(log).read()
    except OSError:
        ckpt_check["mismatches"].append({"eval": os.path.basename(log), "reason": "log missing"})
        continue
    if r["ckpt"] in txt:
        ckpt_check["n_confirmed"] += 1
    else:
        ckpt_check["mismatches"].append({"eval": os.path.basename(log), "expected": r["ckpt"]})
ckpt_check["all_confirmed"] = not ckpt_check["mismatches"]

snap_sha = {}
for line in wl.splitlines():
    m = re.match(r"^([0-9a-f]{64})\s+(\S+cache-after-doc-\S+\.pt)$", line)
    if m:
        snap_sha[os.path.basename(m.group(2))] = m.group(1)

provenance = {
    "snapshot_path": grab(r"SNAPSHOT_PATH=(\S+)"),
    "snapshot_manifest_at_launch": grab(r"SNAPSHOT_MANIFEST_AT_LAUNCH=(\S+)"),
    "snapshot_manifest_after": grab(r"SNAPSHOT_MANIFEST_AFTER=(\S+)"),
    "snapshot_manifest_unchanged": grab(r"SNAPSHOT_MANIFEST_AT_LAUNCH=(\S+)") == grab(r"SNAPSHOT_MANIFEST_AFTER=(\S+)"),
    "snapshot_manifest_matches_DIAG_KEYCURVE": grab(r"SNAPSHOT_MANIFEST_AT_LAUNCH=(\S+)")
        == "9d13a49c76ebc1c289377d611eb472765c916ced68748d934b6a6f377960b7d1",
    "repo_head_at_launch": grab(r"REPO_HEAD_AT_LAUNCH=(\S+)"),
    "repo_head_after": grab(r"REPO_HEAD_AFTER=(\S+)"),
    "cartridges_import_path": grab(r"CARTRIDGES_IMPORT_PATH=(\S+)"),
    "gpu_claimed": grab(r"CC_CLAIMED_GPU=(\S+)"),
    "start_ts": grab(r"CC_START_TS=(\S+)"),
    "done_ts": grab(r"CC_DONE_TS=(\S+)"),
    "control_run_dir": grab(r"CC_DIR_control=(\S+)"),
    "phase1_cache": grab(r"CC_PHASE1=(\S+)"),
    "control_snapshot_sha256": snap_sha,
    "control_snapshots_distinct": len(set(snap_sha.values())) == len(snap_sha),
    "n_control_snapshots": len(snap_sha),
    "route_mass_script_sha256": grab(r"CC_ROUTEMASS_SCRIPT_SHA256=(\S+)"),
    "eval_checkpoint_confirmation": ckpt_check,
    "no_source_file_edited": sh(f"git -C {REPO} status --porcelain -- cartridges examples") or "(clean under cartridges/ and examples/ from this job's perspective; any diff listed here was made by MECH-SEQUENTIAL, not this job, and cannot reach the pinned snapshot)",
    "concurrent_editor": "MECH-SEQUENTIAL is modifying cartridges/am/finetune.py, cartridges/am/continual.py and examples/qasper2/train/continual_am_sparse.py in the working tree; this job imported only from the frozen git-archive snapshot.",
}

wandb_runs = {f"{r['arm']}_k{r['k']}_{r['split']}": r["wandb_url"] for r in rows}
json.dump(wandb_runs, open(f"{RESDIR}/wandb_runs.json", "w"), indent=1)

# --------------------------------------------------------------------------- outputs
diag = {
    "id": "DIAG-CONTROLCURVE",
    "what": "the frozen-key control's per-document k-curve on both splits, and an "
            "each-arm-at-its-own-best comparison against the keys+reposition arm now that "
            "BOTH curves are fully mapped.",
    "arms": {
        "control": "KEY_MODE=freeze, AM_ROPE_THETA=5e6, ENABLE_BETA=0, top32 per_layer tfidf "
                   "(MECH-KEYS `control`). Per-document cache-after-doc-*.pt snapshots, nothing re-trained.",
        "keys": "KEY_MODE=highest_attention + AM_KEY_REPOSITION=1, same everything else "
                "(MECH-KEYS `keys_repos`). Numbers REUSED verbatim from DIAG-KEYCURVE.",
    },
    "noise_band": BAND,
    "noise_note": "MT n=69 / QA n=78; |delta| <= 0.15 is inside the band. One seed throughout "
                  "(DIAG-KEYCURVE established by direct probe that no seed variation is possible "
                  "in this codebase).",
    "correctness_gates": gates,
    "correctness_gate_passed": gate_pass,
    "control_mt_by_k": ctrl_mt,
    "control_qa_by_k": ctrl_qa,
    "phase1_k0": p1,
    "control_argmin": {
        "mt_k": c_mt_k, "mt": c_mt_v, "qa_at_mt_argmin": ctrl_qa.get(c_mt_k),
        "qa_k": c_qa_k, "qa": c_qa_v, "mt_at_qa_argmin": ctrl_mt.get(c_qa_k),
    },
    "which_numbers_are_reused_vs_new": {
        "control_losses": "ALL 16 k were re-measured in THIS session. The 13 k DIAG-KEYCURVE "
                          "never measured (1..7, 9, 10, 11, 13, 15) are new; k=8/12/14/16 were "
                          "re-run as reproduction checks and are reported both ways "
                          "(see control_k_reproduction_vs_DIAG_KEYCURVE).",
        "keys_losses": "REUSED VERBATIM from DIAG-KEYCURVE's bundle for all 16 k -- not re-run here.",
        "keys_route_mass": "REUSED VERBATIM from DIAG-KEYCURVE.",
        "control_route_mass": "ALL 16 k measured here; DIAG-KEYCURVE had only k=8/12/14/16.",
    },
    "control_k_newly_measured_here": sorted(ctrl_mt),
    "control_k_reproduction_vs_DIAG_KEYCURVE": repro,
    "keys_mt_by_k": keys_mt,
    "keys_qa_by_k": keys_qa,
    "keys_argmin": {"mt_k": k_mt_k, "mt": k_mt_v, "qa_k": k_qa_k, "qa": k_qa_v},
    "keys_vs_control_by_k": per_k,
    "fair_comparison_each_arm_at_own_best": fair,
    "control_mass_on_S_MT_by_k": ctrl_mass_mt,
    "control_mass_on_S_QA_by_k": ctrl_mass_qa,
    "control_mt_over_qa_mass_ratio_by_k": ctrl_ratio,
    "control_total_cartridge_mass_MT_by_k": ctrl_cart_mt,
    "control_total_cartridge_mass_QA_by_k": ctrl_cart_qa,
    "control_written_support_union_slots_per_layer_by_k": union_by_k,
    "keys_mass_on_S_MT_by_k": {int(k): v for k, v in KCD["mass_on_S_MT_by_k"].items()},
    "keys_mt_over_qa_mass_ratio_by_k": {int(k): v for k, v in KCD["mt_over_qa_mass_ratio_by_k"].items()},
    "keys_total_cartridge_mass_MT_by_k": {int(k): v for k, v in KCD["total_cartridge_mass_MT_by_k"].items()},
    "phase1_route_mass": mass.get("phase1"),
    "wandb_runs": wandb_runs,
    "provenance": provenance,
}
os.makedirs(DIAGDIR, exist_ok=True)
json.dump(diag, open(f"{DIAGDIR}/DIAG-CONTROLCURVE.json", "w"), indent=2)
print(f"[wrote] {DIAGDIR}/DIAG-CONTROLCURVE.json")

# ------- human-readable table to stdout so the worker can read the shape at a glance
print("\n k |  control MT |  control QA |    keys MT |    keys QA |    dMT |    dQA")
for k in range(1, 17):
    if k not in ctrl_mt:
        continue
    print(f"{k:2d} | {ctrl_mt[k]:11.4f} | {ctrl_qa[k]:11.4f} | {keys_mt[k]:10.4f} | "
          f"{keys_qa[k]:10.4f} | {keys_mt[k]-ctrl_mt[k]:+6.3f} | {keys_qa[k]-ctrl_qa[k]:+6.3f}")
print(f"\ncontrol argmin MT: k={c_mt_k} -> {c_mt_v:.4f} (QA {ctrl_qa[c_mt_k]:.4f})")
print(f"keys    argmin MT: k={k_mt_k} -> {k_mt_v:.4f} (QA {keys_qa[k_mt_k]:.4f})")
print(f"FAIR dMT = {fair['d_mt_at_own_bests']:+.4f}  clears band: {fair['d_mt_clears_band']}")
print(f"FAIR dQA = {fair['d_qa_at_own_bests']:+.4f}  clears band: {fair['d_qa_clears_band']}")
print(f"\nmass_on_S(MT) control: " +
      ", ".join(f"k{k}={ctrl_mass_mt[k]:.4f}" for k in sorted(ctrl_mass_mt)))
print(f"GATE PASSED: {gate_pass}")
