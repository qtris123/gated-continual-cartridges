#!/usr/bin/env python
"""Equivalence harness for the AM package restructure.

Three independent checks, none of which needs a GPU:

  A. **Leaf bodies are byte-identical.** Every math function that moved was
     supposed to move *verbatim*; only its call sites were allowed to change.
     This diffs each one's source against a pristine checkout.

  B. **The same environment produces the same values.** The old flat
     `AttentionMatchingFinetuningConfig` and the new nested `AMContinualConfig`
     are both built from the MECH-KEYS winning arm's environment, then compared
     field by field along the old -> new rename map.

  C. **The rewired write is numerically identical.** A and B together imply
     identical arguments reaching identical bodies, but not that the call sites
     were reconnected correctly. So one full per-document write runs on tiny
     synthetic tensors in each tree, across four arms that between them cover
     every branch of `ValueObjective.solve`, and the resulting cartridge
     tensors are compared elementwise.

Usage::

    cd $CARTRIDGES_DIR
    git worktree add -f --detach /tmp/am_baseline_tree <pre-refactor-ref>
    .venv/bin/python research_loop/verify_am_restructure.py /tmp/am_baseline_tree
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from pathlib import Path

# MECH-KEYS `keys_repos`, the winning arm. Canonical source:
# research_loop/results/MECH-KEYS/launch_mech_keys.sh lines 64-90.
MECH_KEYS_ENV = {
    "AM_ROPE_THETA": "5000000",
    "MAX_QUERIES_PER_HEAD": "64",
    "SLOT_SELECTION": "tfidf",
    "USE_IDF": "0",
    "GRANULARITY": "per_layer",
    "TOP_T": "32",
    "KEY_MODE": "highest_attention",
    "AM_KEY_REPOSITION": "1",
    "ENABLE_BETA": "0",
    "BETA_FIT_SCOPE": "selected",
    "RIDGE_LAMBDA": "1e-4",
    "RIDGE_SCALE": "spectral",
    "RIDGE_LAMBDA_MIN": "0.0",
    "DELTA_WEIGHT": "1e-2",
    "SAVE_AFTER_EACH_DOCUMENT": "1",
    # Inputs the driver requires but that no leaf argument depends on.
    "PHASE1_CACHE_PATH": "/tmp/phase1_cache.pt",
    "SYNTH_DATA_PATH": "/tmp/mt.parquet",
    "MODEL_NAME": "Qwen/Qwen3-4B-Instruct-2507",
    "EVAL_QA_PATH": "/tmp/qa_eval.parquet",
    "EVAL_MT_PATH": "/tmp/mt_eval.parquet",
    # Only read by the old driver, which still built a TrainConfig costume.
    "AM_EXECUTION_MODE": "per_document",
    "TARGET_MODE": "cartridge_plus_doc",
}

# Every function that was required to move verbatim: old module -> new module.
LEAF_MOVES = [
    ("cartridges/am/ranking.py", "cartridges/am/components/slots.py", [
        "_rank_attention_mass_per_layer", "_rank_residual_budget_per_layer",
        "compute_slot_redundancy", "load_slot_fisher_scores", "_unit_rank",
        "_mask_from_topk", "_safety_prior", "_rank_slot_prior_per_layer",
        "_check_prior_shape",
    ]),
    ("cartridges/am/key_select.py", "cartridges/am/components/keys.py", [
        "select_keys_highest_attention", "select_keys_omp",
    ]),
    ("cartridges/am/key_select.py", "cartridges/am/components/beta.py", [
        "nnls_projected_gradient", "refit_beta_nnls", "_record_beta_info",
    ]),
    ("cartridges/am/value_solve.py", "cartridges/am/components/objective.py", [
        "sparse_am_value_update", "guarded_sparse_am_value_update",
        "oracle_teacher_value_write", "refine_kv_head_values",
        "evaluate_attention_reconstruction",
    ]),
    ("cartridges/am/teacher.py", "cartridges/am/components/teacher.py", [
        "_tokenize_system_prompt", "prefill_document_kv_cache", "concat_teacher_kv",
        "compute_teacher_targets", "compute_teacher_log_mass", "compute_teacher_mass",
    ]),
    ("cartridges/am/query_accum.py", "cartridges/am/components/queries.py", [
        "AMQueryAccumulator", "AMTargetAccumulator",
        "install_teacher_attention_capture_hooks",
    ]),
    ("cartridges/am/reference_data.py", "cartridges/am/components/queries.py", [
        "load_conversations", "_extract_tag", "document_key",
        "group_conversations_by_document", "group_conversations_by_system_prompt",
        "limit_conversations", "load_old_reference_bank",
    ]),
    ("cartridges/am/compaction.py", "cartridges/am/initial/compaction.py", [
        "naive_compaction_c2_update", "compute_compaction_c2", "compact_kv_head",
    ]),
    ("cartridges/am/phase1.py", "cartridges/am/initial/compaction.py", [
        "_collect_compaction_queries",
    ]),
    ("cartridges/am/phase1.py", "cartridges/am/initial/refine.py", [
        "refine_cache_am_phase1",
    ]),
    ("cartridges/am/phase1.py", "cartridges/am/core.py", [
        "_rope_reposition",
    ]),
    ("cartridges/am/finetune.py", "cartridges/am/continual/write.py", [
        "_onpolicy_refresh",
    ]),
    ("cartridges/am_stability_probe.py", "cartridges/am/diagnostics.py", [
        "SlotMassProbe", "SolveProbe", "DocProbeSummary", "_safe_spearman",
        "probe_tfidf_vs_absolute_mass", "probe_sparse_solve",
        "summarize_slot_mass_probes", "summarize_solve_probes",
    ]),
]

# Leaves whose body legitimately changed, with the reason. Anything not listed
# here and not byte-identical is a bug.
EXPECTED_BODY_CHANGES = {
    "rewrite_keys_on_support": (
        "`_rope_reposition` moved to am.core, so the deferred local import that "
        "broke the phase1 <-> key_select cycle is now a module-level import."
    ),
    "rank_am_slots": (
        "reads config fields directly; the `getattr(config, ...)` defensive "
        "layer existed only to survive a sibling-checkout import (Section 0b)."
    ),
    "build_reference_dataloader": (
        "`os` / `tempfile` / `write_conversations` are imported at module scope "
        "instead of inside the function body."
    ),
    "cleanup_reference_parquet": ("`os` is imported at module scope."),
    "compact_cache_am_phase1": (
        "`dataclasses` and the reference-data helpers are imported at module "
        "scope; the docstring points at the new module path. The document prompt "
        "is also read from the QASPER paper (`full_paper_prompt`) instead of "
        "merged out of the synthesis rows' sampled sections -- byte-identical on "
        "the stock parquets, pinned by "
        "`test_document_prompts_match_golden_hashes`."
    ),
}

# old flat field -> new dotted path on AMContinualConfig.
FIELD_MAP = {
    # slots
    "top_t": "slots.top_t",
    "granularity": "slots.granularity",
    "slot_selection": "slots.slot_selection",
    "use_idf": "slots.use_idf",
    "idf_smoothing": "slots.idf_smoothing",
    "background_top_k_per_batch": "slots.background_top_k_per_batch",
    "background_indices_path": "slots.background_indices_path",
    "num_background_batches": "slots.num_background_batches",
    "redundancy_ridge_rel": "slots.redundancy_ridge_rel",
    "mass_redundancy_alpha": "slots.mass_redundancy_alpha",
    "slot_fisher_path": "slots.slot_fisher_path",
    "safe_fraction": "slots.safe_fraction",
    "safe_metric": "slots.safe_metric",
    "idf_prior_weight": "slots.idf_prior_weight",
    "min_top_t_per_layer": "slots.min_top_t_per_layer",
    # queries
    "max_ref_examples_per_doc": "queries.max_ref_examples_per_doc",
    "queries_per_batch": "queries.queries_per_batch",
    "max_queries_per_head": "queries.max_queries_per_head",
    "seed_offset": "queries.seed_offset",
    "onpolicy_layers": "queries.onpolicy_layers",
    "onpolicy_refresh_doc_kv": "queries.onpolicy_refresh_doc_kv",
    "decoupled_ref_batches": "queries.ref_batch_limit",
    # teacher (hoisted to the top level)
    "rope_theta": "rope_theta",
    # keys
    "key_mode": "keys.key_mode",
    "key_reposition": "keys.key_reposition",
    # beta
    "enable_beta": "beta.enabled",
    "beta_fit_scope": "beta.fit_scope",
    "beta_box": "beta.beta_box",
    "nnls_iters": "beta.nnls_iters",
    "nnls_driver": "beta.nnls_driver",
    "beta_target": "beta.target_mode",
    # objective
    "ridge_lambda": "objective.ridge_lambda",
    "ridge_scale": "objective.ridge_scale",
    "ridge_lambda_min": "objective.ridge_lambda_min",
    "delta_weight": "objective.delta_weight",
    "enable_old_reference_guard": "objective.enable_old_reference_guard",
    "old_ref_data_path": "objective.old_ref_data_path",
    "old_ref_max_examples": "objective.old_ref_max_examples",
    "old_reference_weight": "objective.old_reference_weight",
    "oracle_write": "objective.oracle_write",
    "oracle_write_assign": "objective.oracle_write_assign",
    # bookkeeping
    "save_after_each_document": "save_after_each_document",
    "compute_update_stats": "compute_update_stats",
}

# Deliberately removed by the restructure (plan Step 4).
DROPPED_FIELDS = {
    "enabled": "vestigial in a dedicated runner",
    "execution_mode": "a dispatch hack, not a hyperparameter",
    "target_mode": "inert on the per-document path",
    "freeze_keys": "superseded by key_mode",
    "update_interval": "train_loop only",
    "collect_background_stats": "the live flag is on SparseCacheFinetuningConfig",
    # Declared but read by nothing, at any commit; deleted before this refactor.
    "max_am_steps": "dead field, never read",
}

# Fields the beta stage reads. Inert unless `BetaFitter.should_fit` is True, so
# they are reported rather than asserted when beta is off.
BETA_ONLY_FIELDS = {"beta_box", "nnls_iters", "nnls_driver", "beta_target"}


# ======================================================================================
# A. leaf bodies
# ======================================================================================
def _defs(path: Path) -> dict[str, str]:
    """Every top-level def/class in `path`, as normalized source text."""
    src = path.read_text()
    tree = ast.parse(src)
    out = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out[node.name] = ast.get_source_segment(src, node)
    return out


def check_leaf_bodies(old_root: Path, new_root: Path) -> list[str]:
    problems = []
    checked = 0
    for old_rel, new_rel, names in LEAF_MOVES:
        old_defs = _defs(old_root / old_rel)
        new_defs = _defs(new_root / new_rel)
        for name in names:
            checked += 1
            if name not in old_defs:
                problems.append(f"{name}: missing from baseline {old_rel}")
                continue
            if name not in new_defs:
                problems.append(f"{name}: missing from {new_rel}")
                continue
            if old_defs[name] != new_defs[name]:
                problems.append(
                    f"{name}: body differs between {old_rel} and {new_rel} "
                    "(leaf math must move verbatim)"
                )
    print(f"[A] leaf bodies: {checked} checked, {len(problems)} unexpected diffs")
    for name, reason in EXPECTED_BODY_CHANGES.items():
        print(f"    known-changed  {name}: {reason}")
    return problems


# ======================================================================================
# B. config values
# ======================================================================================
_DUMP_OLD = r"""
import json, os, runpy, sys
mod = runpy.run_path(sys.argv[1], run_name="__equivalence_probe__")
cfg = mod["config"].attention_matching_finetuning
print("###JSON###" + json.dumps(cfg.model_dump(), default=str))
"""

_DUMP_NEW = r"""
import json, os, runpy, sys
mod = runpy.run_path(sys.argv[1], run_name="__equivalence_probe__")
cfg = mod["config"]
out = {"rope_theta": cfg.rope_theta,
       "save_after_each_document": cfg.save_after_each_document,
       "compute_update_stats": cfg.compute_update_stats}
for stage in ("slots", "queries", "teacher", "keys", "beta", "objective"):
    sub = getattr(cfg, stage).model_dump(exclude={"target", "kwargs"})
    for k, v in sub.items():
        out[f"{stage}.{k}"] = v
print("###JSON###" + json.dumps(out, default=str))
"""


def _dump(tree: Path, driver_rel: str, program: str, python: str) -> dict:
    env = {**os.environ, **MECH_KEYS_ENV}
    env["CARTRIDGES_DIR"] = str(tree)
    env["CARTRIDGES_OUTPUT_DIR"] = str(tree / "outputs")
    env["PYTHONPATH"] = str(tree)
    proc = subprocess.run(
        [python, "-c", program, str(tree / driver_rel)],
        cwd=tree, env=env, capture_output=True, text=True,
    )
    marker = "###JSON###"
    for line in proc.stdout.splitlines():
        if line.startswith(marker):
            return json.loads(line[len(marker):])
    raise RuntimeError(
        f"config dump failed in {tree}\n--- stdout ---\n{proc.stdout[-3000:]}"
        f"\n--- stderr ---\n{proc.stderr[-3000:]}"
    )


def check_config_values(old_root: Path, new_root: Path, python: str) -> list[str]:
    driver = "examples/qasper2/train/continual_am_sparse.py"
    old = _dump(old_root, driver, _DUMP_OLD, python)
    new = _dump(new_root, driver, _DUMP_NEW, python)

    problems, inert = [], []
    for old_key, new_key in FIELD_MAP.items():
        if old_key not in old:
            problems.append(f"{old_key}: absent from the baseline config")
            continue
        if new_key not in new:
            problems.append(f"{new_key}: absent from the new config")
            continue
        if old[old_key] != new[new_key]:
            line = f"{old_key} = {old[old_key]!r} -> {new_key} = {new[new_key]!r}"
            (inert if old_key in BETA_ONLY_FIELDS else problems).append(line)

    unmapped = set(old) - set(FIELD_MAP) - set(DROPPED_FIELDS)
    if unmapped:
        problems.append(f"baseline fields with no destination: {sorted(unmapped)}")
    resurrected = [f for f in DROPPED_FIELDS if f in new]
    if resurrected:
        problems.append(f"fields that should be gone are still present: {resurrected}")

    print(
        f"[B] config values: {len(FIELD_MAP)} mapped fields compared, "
        f"{len(problems)} mismatches"
    )
    print(f"    dropped as designed: {', '.join(sorted(DROPPED_FIELDS))}")
    if inert:
        # `ENABLE_BETA=0` on this arm, so `BetaFitter.should_fit` is False and
        # none of these reaches `refit_beta_nnls`.
        print("    inert differences (beta is OFF on this arm, so unread):")
        for line in inert:
            print(f"      {line}")
        assert old["enable_beta"] is False and new["beta.enabled"] is False, (
            "beta knobs differ AND beta is enabled -- these are NOT inert"
        )
    return problems


# ======================================================================================
# C. the rewired write
# ======================================================================================
WRITE_ARMS = ("keys_repos", "plain", "beta", "oracle")


def _write_dump(tree: Path, probe: Path, mode: str, arm: str, python: str) -> dict:
    env = {**os.environ}
    env["CARTRIDGES_DIR"] = str(tree)
    env["CARTRIDGES_OUTPUT_DIR"] = str(tree / "outputs")
    env["PYTHONPATH"] = str(tree)
    proc = subprocess.run(
        [python, str(probe), mode, arm], cwd=tree, env=env,
        capture_output=True, text=True,
    )
    marker = "###JSON###"
    for line in proc.stdout.splitlines():
        if line.startswith(marker):
            return json.loads(line[len(marker):])
    raise RuntimeError(
        f"write probe failed ({mode}/{arm}) in {tree}\n{proc.stderr[-3000:]}"
    )


def check_write_equivalence(old_root: Path, new_root: Path, python: str) -> list[str]:
    probe = Path(__file__).parent / "verify_am_write_probe.py"
    problems = []
    for arm in WRITE_ARMS:
        old = _write_dump(old_root, probe, "old", arm, python)
        new = _write_dump(new_root, probe, "new", arm, python)
        diffs = [k for k in old if old[k] != new.get(k)]
        n = sum(len(x) for x in old["keys"]) + sum(len(x) for x in old["values"])
        if old.get("beta"):
            n += sum(len(x) for x in old["beta"])
        if diffs:
            problems.append(f"write arm {arm!r} differs in: {diffs}")
        print(
            f"    {arm:11s} {'identical' if not diffs else 'DIFFERS'}  "
            f"({n} tensor entries, mean_mse={old['mean_mse']})"
        )
    print(f"[C] rewired write: {len(WRITE_ARMS)} arms, {len(problems)} mismatches")
    return problems


def main() -> int:
    old_root = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/am_baseline_tree")
    new_root = Path(os.environ.get("CARTRIDGES_DIR", ".")).resolve()
    python = sys.executable
    if not (old_root / "cartridges/am/finetune.py").exists():
        print(f"error: {old_root} is not a pre-refactor checkout", file=sys.stderr)
        return 2

    print(f"baseline: {old_root}\ncurrent:  {new_root}\n")
    problems = check_leaf_bodies(old_root, new_root)
    print()
    problems += check_config_values(old_root, new_root, python)
    print()
    problems += check_write_equivalence(old_root, new_root, python)

    print()
    if problems:
        print(f"FAILED ({len(problems)} problems)")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("PASSED: identical arguments reaching identical function bodies.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
