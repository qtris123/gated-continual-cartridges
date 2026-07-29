"""DIAG-KEYCURVE finalize: build state/diagnostics/DIAG-KEYCURVE.json + results/.../result.json.

Pure post-processing of curve.tsv / wrapper.log / route-mass JSON. No GPU, no source edit.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from pathlib import Path

REPO = Path("/localhome/local-triv/gated-continual-cartridges_explore")
RESDIR = REPO / "research_loop/results/DIAG-KEYCURVE"
SNAP = Path("/tmp/amsnap_kcurve")
DIAGDIR = REPO / "research_loop/state/diagnostics"

NOISE = 0.15  # RUNBOOK 1 / board: MT n=69, QA n=78; deltas below ~0.15 are inside the band

# ---- the baselines this result is scored against ---------------------------------
CONTROL_K16 = {"qa": 2.15972, "mt": 2.52963}          # MECH-KEYS freeze control, standalone
MECHKEYS_K16 = {"qa": 2.034916400909424, "mt": 2.330503463745117}  # the correctness gate
TARGET = {"qa": 2.52, "mt": 2.02}                      # mission bar
ORACLE_PERFECT_VALUE_WRITE_MT = 2.3810                 # ORACLE-WRITE
# DIAG-SEQUENCE, value-only (KEY_MODE=freeze, theta=1e4) canonical curve
VALUE_ONLY_MT = {0: 3.7825491428375244, 1: 3.7445480823516846, 4: 3.0338, 8: 2.7099,
                 12: 2.4352, 13: 2.4490, 14: 2.4550, 15: 2.5000, 16: 2.5524}
VALUE_ONLY_QA = {0: 2.23880672454834, 1: 2.2298452854156494, 4: 2.113, 8: 2.047,
                 12: 2.042, 13: 2.024, 16: 2.1772}


def manifest(root: Path) -> str:
    cmd = ("find cartridges examples -type f -name '*.py' | LC_ALL=C sort | xargs sha256sum "
           "| awk '{print $1\"  \"$2}' | LC_ALL=C sort | sha256sum | cut -d' ' -f1")
    return subprocess.run(["bash", "-c", cmd], cwd=root, capture_output=True, text=True).stdout.strip()


def git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True).stdout.strip()


# ---------------------------------------------------------------- read the run
wrapper = (RESDIR / "wrapper.log").read_text(errors="ignore")


def wline(prefix: str):
    for ln in wrapper.splitlines():
        if ln.startswith(prefix):
            return ln.split("=", 1)[1].split(" ")[0]
    return None


rows = []
for line in (RESDIR / "curve.tsv").read_text().splitlines()[1:]:
    arm, k, split, loss, ckpt, url = line.split("\t")
    rows.append({"arm": arm, "k": int(k), "split": split,
                 "loss": None if loss == "MISSING" else float(loss),
                 "ckpt": ckpt, "wandb_url": None if url == "MISSING" else url})

by = {}
for r in rows:
    by.setdefault(r["arm"], {}).setdefault(r["k"], {})[r["split"]] = r

# ---------------------------------------------------------------- ckpt provenance
loaded_ok, ckpt_sha = {}, {}
for r in rows:
    log = RESDIR / "evals" / f"eval_{r['arm']}_doc{r['k']}_{r['split']}.log"
    txt = log.read_text(errors="ignore") if log.exists() else ""
    loaded_ok[f"{r['arm']}_k{r['k']}_{r['split']}"] = r["ckpt"] in txt
    if r["ckpt"] not in ckpt_sha and os.path.exists(r["ckpt"]):
        h = hashlib.sha256()
        with open(r["ckpt"], "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 22), b""):
                h.update(chunk)
        ckpt_sha[r["ckpt"]] = h.hexdigest()
n_ckpt, n_uniq_sha = len(ckpt_sha), len(set(ckpt_sha.values()))
_inv = {}
for _p, _s in ckpt_sha.items():
    _inv.setdefault(_s, []).append(_p)
identical_groups = [sorted(v) for v in _inv.values() if len(v) > 1]

# ---------------------------------------------------------------- route mass
rm_path = DIAGDIR / "DIAG-KEYCURVE_route_mass.json"
rm = json.load(open(rm_path)) if rm_path.exists() else {"runs": {}}


def mass_for(label: str):
    runs = rm.get("runs", {})
    run = runs.get(f"{label}|MT")
    if not run:
        return None
    qa = (runs.get(f"{label}|QA") or {}).get("summary")
    mt = run.get("summary")
    if not qa or not mt:
        return None
    out = {
        "mass_on_S_MT": mt["mean_mass_on_S"], "mass_on_S_QA": qa["mean_mass_on_S"],
        "mt_over_qa_mass_ratio": (mt["mean_mass_on_S"] / qa["mean_mass_on_S"]
                                  if qa["mean_mass_on_S"] else None),
        "mass_on_cart_MT": mt["mean_mass_on_cart"], "mass_on_cart_QA": qa["mean_mass_on_cart"],
        "mass_on_S_MT_scored": mt["mean_mass_on_S_scored"],
        "mass_on_S_QA_scored": qa["mean_mass_on_S_scored"],
    }
    nsl = {k2: v for k2, v in (run.get("n_slots_per_layer") or {}).items()}
    if nsl:
        v = list(nsl.values())
        out["slots_per_layer_mean"] = sum(v) / len(v)
        out["slots_per_layer_min"], out["slots_per_layer_max"] = min(v), max(v)
    return out


# ---------------------------------------------------------------- curves
def build_curve(arm: str, label_prefix: str):
    out = []
    for k in sorted(by.get(arm, {})):
        e = by[arm][k]
        rec = {"k": k,
               "qa": e.get("QA", {}).get("loss"), "mt": e.get("MT", {}).get("loss"),
               "ckpt": e.get("MT", e.get("QA"))["ckpt"],
               "wandb_qa": e.get("QA", {}).get("wandb_url"),
               "wandb_mt": e.get("MT", {}).get("wandb_url")}
        m = mass_for("phase1" if arm == "phase1" else f"{label_prefix}_k{k}")
        if m:
            rec["route_mass"] = m
        out.append(rec)
    return out


keys_curve = build_curve("keys_repos", "keys_repos")
ctrl_curve = build_curve("control", "control")
rerun_curve = build_curve("rerun", "rerun")
phase1_curve = build_curve("phase1", "phase1")

mt_pts = [(c["k"], c["mt"]) for c in keys_curve if c["mt"] is not None]
qa_pts = [(c["k"], c["qa"]) for c in keys_curve if c["qa"] is not None]
argmin_mt = min(mt_pts, key=lambda t: t[1]) if mt_pts else (None, None)
argmin_qa = min(qa_pts, key=lambda t: t[1]) if qa_pts else (None, None)
best = next((c for c in keys_curve if c["k"] == argmin_mt[0]), None)

k16 = next((c for c in keys_curve if c["k"] == 16), None)
gate = {
    "expected_qa_k16": MECHKEYS_K16["qa"], "observed_qa_k16": k16["qa"] if k16 else None,
    "expected_mt_k16": MECHKEYS_K16["mt"], "observed_mt_k16": k16["mt"] if k16 else None,
    "qa_exact_match": bool(k16 and k16["qa"] == MECHKEYS_K16["qa"]),
    "mt_exact_match": bool(k16 and k16["mt"] == MECHKEYS_K16["mt"]),
    "note": ("the k=16 snapshot is a DIFFERENT FILE from the run's cache_last.pt "
             "(cache-after-doc-015-*.pt vs cache-step16.pt) but must carry the same cartridge"),
}

# ---------------------------------------------------------------- verification arm
rerun_dir = (RESDIR / "rundir_rerun.txt").read_text().strip() if (RESDIR / "rundir_rerun.txt").exists() else None
rerun_summary = None
if rerun_dir and os.path.exists(os.path.join(rerun_dir, "phase2_summary.json")):
    rerun_summary = json.load(open(os.path.join(rerun_dir, "phase2_summary.json")))
cfg_diff = (RESDIR / "config_diff.txt").read_text() if (RESDIR / "config_diff.txt").exists() else ""
seed_probe = json.load(open(RESDIR / "seed_probe.json"))

rr16 = next((c for c in rerun_curve if c["k"] == 16), None)
rr_argmin = next((c for c in rerun_curve if c["k"] == argmin_mt[0]), None)


def delta(a, b):
    return None if (a is None or b is None) else a - b


verification = {
    "claim_under_test": "KEY_MODE=highest_attention + AM_KEY_REPOSITION=1 beats KEY_MODE=freeze on BOTH axes",
    "control_reference_MECH_KEYS_standalone": CONTROL_K16,
    "seed_variation": {
        "available": False,
        "evidence": seed_probe,
        "plain_statement": (
            "NO seed variation happened, and none is possible without a code change. "
            "`continual_am_sparse.py` calls `pydrantic.main` ONLY when "
            "AM_EXECUTION_MODE=train_loop; the canonical per_document mode calls "
            "run_per_document_phase2(config) directly, so the RUNBOOK 9c `seed=<n>` CLI "
            "override is silently ignored (probe: config.seed stayed 42). There is no SEED "
            "env knob. And even if config.seed could be moved, the AM write would not change: "
            "cartridges/am/continual.py:150 seeds the per-document reference draw with "
            "`seed=doc_idx`, a constant. The re-run below is therefore a SAME-SEED, "
            "fresh-process reproduction and confirms plumbing + run-to-run determinism only."
        ),
    },
    "fresh_process_rerun": {
        "run_dir": rerun_dir,
        "config_diff_vs_original_arm": cfg_diff.strip() or "(identical on every field except run_dir/name/wandb)",
        "config_diff_n_lines": len(cfg_diff.strip().splitlines()),
        "in_run_eval": (rerun_summary or {}).get("eval_metrics"),
        "standalone_k16": {"qa": rr16["qa"] if rr16 else None, "mt": rr16["mt"] if rr16 else None},
        "standalone_at_keys_argmin_k": (
            {"k": rr_argmin["k"], "qa": rr_argmin["qa"], "mt": rr_argmin["mt"]} if rr_argmin else None
        ),
        "delta_vs_original_k16": {
            "qa": delta(rr16["qa"] if rr16 else None, MECHKEYS_K16["qa"]),
            "mt": delta(rr16["mt"] if rr16 else None, MECHKEYS_K16["mt"]),
        },
    },
}

# ---------------------------------------------------------------- comparisons
ctrl_by_k = {c["k"]: c for c in ctrl_curve}
comparisons = []
for c in keys_curve:
    ck = ctrl_by_k.get(c["k"])
    if not ck:
        continue
    dq, dm = delta(c["qa"], ck["qa"]), delta(c["mt"], ck["mt"])
    comparisons.append({
        "k": c["k"],
        "keys_qa": c["qa"], "control_qa": ck["qa"], "d_qa": dq,
        "keys_mt": c["mt"], "control_mt": ck["mt"], "d_mt": dm,
        "d_qa_clears_noise_band": None if dq is None else abs(dq) > NOISE,
        "d_mt_clears_noise_band": None if dm is None else abs(dm) > NOISE,
    })

# ---- per-arm snapshot distinctness + full re-run bit-identity ---------------------
within_arm = {}
for arm_label, d in (("keys_repos", wline("KC_DIR_keys_repos=")), ("control", wline("KC_DIR_control="))):
    fs = sorted(Path(d).glob("cache-after-doc-*.pt"))
    shas = []
    for f in fs:
        h = hashlib.sha256()
        with open(f, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 22), b""):
                h.update(chunk)
        shas.append(h.hexdigest())
    within_arm[arm_label] = {"n_snapshots": len(fs), "n_distinct_sha256": len(set(shas)),
                             "all_distinct": len(fs) == len(set(shas))}

rerun_bit_identity = None
if rerun_dir:
    same, tot = 0, 0
    src = Path(wline("KC_DIR_keys_repos="))
    for f in sorted(Path(rerun_dir).glob("cache-after-doc-*.pt")):
        o = src / f.name
        if not o.exists():
            continue
        tot += 1
        ha, hb = hashlib.sha256(), hashlib.sha256()
        with open(f, "rb") as fh:
            for c in iter(lambda: fh.read(1 << 22), b""):
                ha.update(c)
        with open(o, "rb") as fh:
            for c in iter(lambda: fh.read(1 << 22), b""):
                hb.update(c)
        same += int(ha.hexdigest() == hb.hexdigest())
    rerun_bit_identity = {"snapshots_compared": tot, "byte_identical": same,
                          "all_byte_identical": tot > 0 and same == tot}
verification["fresh_process_rerun"]["per_document_snapshot_bit_identity_vs_original"] = rerun_bit_identity

# ---- shape of the keys curve ------------------------------------------------------
mt_by_k = {c["k"]: c["mt"] for c in keys_curve if c["mt"] is not None}
qa_by_k = {c["k"]: c["qa"] for c in keys_curve if c["qa"] is not None}
ks = sorted(mt_by_k)
marginal = [{"k": k, "d_mt_from_prev": mt_by_k[k] - mt_by_k[ks[i - 1]],
             "d_qa_from_prev": (qa_by_k[k] - qa_by_k[ks[i - 1]])
             if (k in qa_by_k and ks[i - 1] in qa_by_k) else None}
            for i, k in enumerate(ks) if i > 0]
ratios = [c["route_mass"]["mt_over_qa_mass_ratio"] for c in keys_curve if c.get("route_mass")]
masses = [c["route_mass"]["mass_on_S_MT"] for c in keys_curve if c.get("route_mass")]
ctrl_ratios = [c["route_mass"]["mt_over_qa_mass_ratio"] for c in ctrl_curve if c.get("route_mass")]
shape = {
    "mt_monotone_decreasing_to_argmin": all(
        mt_by_k[ks[i]] < mt_by_k[ks[i - 1]] for i in range(1, ks.index(argmin_mt[0]) + 1)),
    "mt_strictly_increasing_after_argmin": all(
        mt_by_k[ks[i]] > mt_by_k[ks[i - 1]] for i in range(ks.index(argmin_mt[0]) + 1, len(ks))),
    "mt_tail_rise_argmin_to_16": mt_by_k[16] - mt_by_k[argmin_mt[0]],
    "qa_tail_rise_argminqa_to_16": qa_by_k[16] - argmin_qa[1],
    "marginal_deltas": marginal,
    "mt_over_qa_mass_ratio_range_over_k": {"min": min(ratios), "max": max(ratios),
                                           "span": max(ratios) - min(ratios)} if ratios else None,
    "control_mt_over_qa_mass_ratio_range": {"min": min(ctrl_ratios), "max": max(ctrl_ratios)}
    if ctrl_ratios else None,
    "mass_on_S_MT_range_over_k": {"min": min(masses), "max": max(masses)} if masses else None,
    "phase1_mt_over_qa_mass_ratio": (phase1_curve[0].get("route_mass") or {}).get("mt_over_qa_mass_ratio")
    if phase1_curve else None,
}

# ---- each arm at its OWN best k (the harder comparison) ---------------------------
ctrl_mt = {c["k"]: c["mt"] for c in ctrl_curve if c["mt"] is not None}
ctrl_best_k = min(ctrl_mt, key=ctrl_mt.get) if ctrl_mt else None
ctrl_best = ctrl_by_k.get(ctrl_best_k)
own_best = {
    "keys_best": {"k": argmin_mt[0], "qa": best["qa"], "mt": best["mt"]} if best else None,
    "control_best_MEASURED": ({"k": ctrl_best_k, "qa": ctrl_best["qa"], "mt": ctrl_best["mt"]}
                              if ctrl_best else None),
    "control_k_measured": sorted(ctrl_mt),
    "control_minimum_is_NOT_located": (
        "only k in {8,12,14,16} were evaluated for the freeze control, and its lowest MT is at "
        "the EDGE of that set (k=8), so the control's true minimum may lie at k<8 and this "
        "comparison is an UPPER bound on the control, i.e. it FAVOURS the keys arm"),
    "d_mt_at_own_bests": delta(best["mt"] if best else None, ctrl_best["mt"] if ctrl_best else None),
    "d_qa_at_own_bests": delta(best["qa"] if best else None, ctrl_best["qa"] if ctrl_best else None),
}

headline = {
    "keys_repos_best_point": (
        {"k": best["k"], "qa": best["qa"], "mt": best["mt"]} if best else None
    ),
    "keys_repos_k16_endpoint": {"qa": k16["qa"] if k16 else None, "mt": k16["mt"] if k16 else None},
    "improvement_of_best_over_own_k16": {
        "d_mt": delta(best["mt"] if best else None, k16["mt"] if k16 else None),
        "d_qa": delta(best["qa"] if best else None, k16["qa"] if k16 else None),
    },
    "argmin_mt_k": argmin_mt[0], "argmin_mt_value": argmin_mt[1],
    "argmin_qa_k": argmin_qa[0], "argmin_qa_value": argmin_qa[1],
    "vs_MECH_KEYS_frozen_control_k16": {
        "d_qa": delta(best["qa"] if best else None, CONTROL_K16["qa"]),
        "d_mt": delta(best["mt"] if best else None, CONTROL_K16["mt"]),
    },
    "vs_mission_target": {
        "target_qa": TARGET["qa"], "target_mt": TARGET["mt"],
        "qa_slack_at_best": None if not best else TARGET["qa"] - best["qa"],
        "mt_short_by_at_best": None if not best else best["mt"] - TARGET["mt"],
    },
    "vs_oracle_perfect_value_write_mt": (
        None if not best else best["mt"] - ORACLE_PERFECT_VALUE_WRITE_MT
    ),
    "vs_value_only_k12_minimum_mt": None if not best else best["mt"] - VALUE_ONLY_MT[12],
    "vs_value_only_k16_endpoint_mt": None if not best else best["mt"] - VALUE_ONLY_MT[16],
    "vs_dense_4ep_bar_mt": None if not best else best["mt"] - 1.8725,
    "qa_vs_untouched_phase1_floor": None if not best else best["qa"] - 2.23880672454834,
    "vs_value_only_curve_mt": {
        str(c["k"]): {"keys": c["mt"], "value_only": VALUE_ONLY_MT.get(c["k"]),
                      "d": delta(c["mt"], VALUE_ONLY_MT.get(c["k"]))}
        for c in keys_curve if c["k"] in VALUE_ONLY_MT
    },
    "noise_band": NOISE,
}

out = {
    "id": "DIAG-KEYCURVE",
    "board_entry": "B-ROUTE (key side) / B-OVERWRITE",
    "question": (
        "Where does MECH-KEYS' best arm (KEY_MODE=highest_attention + AM_KEY_REPOSITION=1) "
        "actually bottom on the 16-document sequence, and does its headline k=16 point "
        "(QA 2.0349 / MT 2.3305) survive a fresh-process re-run and a same-session "
        "mechanism-off control?"
    ),
    "provenance": {
        "snapshot_path": str(SNAP),
        "snapshot_contents": f"git archive HEAD {{cartridges,examples}} taken at HEAD={wline('REPO_HEAD_AT_LAUNCH=')}",
        "snapshot_manifest_sha256_at_launch": wline("SNAPSHOT_MANIFEST_AT_LAUNCH="),
        "snapshot_manifest_sha256_after": wline("SNAPSHOT_MANIFEST_AFTER=") or manifest(SNAP),
        "snapshot_manifest_recipe": (
            "cd $SNAP && find cartridges examples -type f -name '*.py' | LC_ALL=C sort | "
            "xargs sha256sum | awk '{print $1\"  \"$2}' | LC_ALL=C sort | sha256sum"),
        "verified_import_path": wline("CARTRIDGES_IMPORT_PATH=") ,
        "CARTRIDGES_DIR": str(SNAP),
        "repo_HEAD_at_launch": wline("REPO_HEAD_AT_LAUNCH="),
        "repo_HEAD_after": git("rev-parse", "HEAD"),
        "repo_HEAD_moved_during_run": wline("REPO_HEAD_AT_LAUNCH=") != git("rev-parse", "HEAD"),
        "repo_HEAD_move_touched_code": bool(
            git("diff", "--name-only", wline("REPO_HEAD_AT_LAUNCH=") or "HEAD", "HEAD",
                "--", "cartridges", "examples")),
        "gpu_claimed_via_flock": wline("KC_CLAIMED_GPU="),
        "concurrent_worker": "D0-ICL (ICL prefills) held the other GPU for the whole session",
        "source_run_dir_keys_repos": wline("KC_DIR_keys_repos="),
        "source_run_dir_control": wline("KC_DIR_control="),
        "source_run_bundle": "research_loop/results/MECH-KEYS/result.json",
        "snapshots_are_stock_artefacts": (
            "cache-after-doc-*.pt saved by SAVE_AFTER_EACH_DOCUMENT=1 in MECH-KEYS' own runs; "
            "Part 1/2a re-use them unchanged - no re-run, no instrumentation, no source edit"),
        "instrumentation": "NONE",
        "eval_convention": (
            "standalone examples/qasper2/train/eval_forgetting.py (EVAL_MODE=cartridge), "
            "BATCH_SIZE=4 (default), MODEL_NAME=Qwen/Qwen3-4B-Instruct-2507, one fresh process "
            "per (arm,k,split); the 'Eval loss' mean-CE line is parsed from the log and the PID "
            "is then killed (RUNBOOK 1)"),
        "each_eval_loaded_intended_checkpoint": loaded_ok,
        "all_evals_loaded_intended_checkpoint": all(loaded_ok.values()),
        "n_checkpoint_paths_evaluated": n_ckpt,
        "n_distinct_checkpoint_sha256": n_uniq_sha,
        "byte_identical_checkpoint_groups": identical_groups,
        "byte_identical_checkpoint_note": (
            "the ONLY sha256 collision among the 23 evaluated checkpoint paths is the "
            "fresh re-run's cache-after-doc-011 against MECH-KEYS' own cache-after-doc-011 "
            "-- i.e. the re-run reproduced the k=12 cartridge BYTE FOR BYTE. Within each arm "
            "all 16 per-document snapshots are pairwise distinct (verified separately over "
            "the 16 keys_repos + 16 control files)."),
        "route_mass_script": "copy of research_loop/results/MECH-KEYS/measure_route_mass.py (unmodified)",
        "route_mass_script_sha256": wline("KC_ROUTEMASS_SCRIPT_SHA256="),
        "route_mass_per_k_slot_set": (
            "S(k) = union over the FIRST k documents' per-layer top-32 selections, built as a "
            "symlink directory holding only am_doc_doc-000..doc-(k-1) - no code change"),
    },
    "correctness_gate": gate,
    "keys_repos_curve": keys_curve,
    "control_freeze_curve": ctrl_curve,
    "rerun_curve": rerun_curve,
    "phase1_anchor": phase1_curve,
    "keys_vs_control_by_k": comparisons,
    "keys_vs_control_at_each_arms_own_best_k": own_best,
    "curve_shape": shape,
    "per_arm_snapshot_distinctness": within_arm,
    "headline": headline,
    "verification": verification,
    "value_only_reference_curve": {"mt": VALUE_ONLY_MT, "qa": VALUE_ONLY_QA,
                                   "source": "research_loop/state/diagnostics/DIAG-SEQUENCE.json "
                                             "(KEY_MODE=freeze, theta=1e4)"},
    "noise_discipline": {
        "eval_sizes": {"MT_examples": 69, "QA_examples": 78},
        "band": "deltas below ~0.15 loss are inside the band (RUNBOOK 1, 9d; eval size is fixed)",
        "seeds": 1,
    },
    "wandb": {f"{r['arm']}_k{r['k']}_{r['split']}": r["wandb_url"] for r in rows},
}
DIAGDIR.mkdir(parents=True, exist_ok=True)
(DIAGDIR / "DIAG-KEYCURVE.json").write_text(json.dumps(out, indent=1))
print("wrote", DIAGDIR / "DIAG-KEYCURVE.json")
print(json.dumps(headline, indent=1))
print(json.dumps(comparisons, indent=1))
