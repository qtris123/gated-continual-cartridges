"""D0-ICL: assemble result.json + state/diagnostics/D0-ICL.json from the per-cell JSONs."""

import glob
import json
import os

OUT = os.path.dirname(os.path.abspath(__file__))
REPO = "/localhome/local-triv/gated-continual-cartridges_explore"
DIAG = os.path.join(REPO, "research_loop/state/diagnostics/D0-ICL.json")

cells = {}
for p in sorted(glob.glob(os.path.join(OUT, "cell_*.json"))):
    rec = json.load(open(p))
    cells[rec["cell"]] = rec

PRIOR = {"icl_QA|QA": 1.9734, "icl_MT|MT": 1.8960}
PRIOR_OFFDIAG = {"icl_QA|MT": 2.7761, "icl_MT|QA": 2.9025}
ANCHORS = {
    "phase1_start": {"QA": 2.2388, "MT": 3.7825},
    "dense_4ep_bar": {"QA": 2.3721, "MT": 1.8725},
    "best_gradient_free_k12": {"QA": 2.0422, "MT": 2.4352},
    "sparse_gradient_ref": {"QA": 1.6169, "MT": 1.9664},
}

payload = {
    "id": "D0-ICL",
    "board_entry": "reference lines",
    "question": (
        "Does the full-context ICL 'ceiling' the board quotes (QA 1.9734 / MT 1.8960, measured with "
        "experiments/qasper_loss_benchmark.evaluate_loss_chunked) survive being re-measured on the "
        "loop's OWN harness (eval_forgetting.py EVAL_MODE=icl), the same ruler every cartridge/AM/dense "
        "number uses?"
    ),
    "harness": "examples/qasper2/train/eval_forgetting.py with EVAL_MODE=icl (-> examples/qasper2/train/eval_icl.py::run_icl_loss_eval)",
    "model": "Qwen/Qwen3-4B-Instruct-2507 (raw, NO cartridge)",
    "prefill_chunk_size": 2048,
    "max_context_tokens": None,
    "noise_band_loss": 0.15,
    "cells": cells,
    "diagonals": {
        k: cells[k]["eval_loss_mean_ce"] for k in ("icl_QA|QA", "icl_MT|MT") if k in cells
    },
    "off_diagonals": {
        k: cells[k]["eval_loss_mean_ce"] for k in ("icl_QA|MT", "icl_MT|QA") if k in cells
    },
    "prior_board_numbers": PRIOR,
    "prior_board_off_diagonals": PRIOR_OFFDIAG,
    "delta_vs_prior": {
        k: round(cells[k]["eval_loss_mean_ce"] - PRIOR[k], 6)
        for k in PRIOR
        if k in cells
    },
    "delta_vs_prior_off_diagonals": {
        k: round(cells[k]["eval_loss_mean_ce"] - PRIOR_OFFDIAG[k], 6)
        for k in PRIOR_OFFDIAG
        if k in cells
    },
    "anchors_eval_forgetting_mean_ce": ANCHORS,
    "ruler_check": {
        "claim": "eval_icl scores the same soft-CE-vs-teacher-top-k as cartridges/train.py::evaluate_perplexity",
        "scorer_formula_both": "-p_teacher(x).exp() * log q_model(x), summed over teacher top-k entries / n_entries",
        "scored_target_entries_cartridge_path": {"QA": 2619, "MT": 2562},
        "scored_target_entries_icl_path": {
            "QA": cells.get("icl_QA|QA", {}).get("num_target_tokens"),
            "MT": cells.get("icl_MT|MT", {}).get("num_target_tokens"),
        },
        "note": (
            "Denominators counted directly from LossEvalDataset(packed_seq_length=2048) on the same "
            "parquets (QA 6 batches/2619 entries, MT 5 batches/2562 entries). Identical to the ICL "
            "path's num_target_tokens => same scored token set, same formula, same files. The only "
            "difference is the forward pass (packed 2048-token cartridge forward vs per-conversation "
            "77-103k-token chunked prefill), which is intrinsic to ICL, not a harness discrepancy."
        ),
    },
}

payload["provenance"] = {
    "snapshot_path": "/tmp/amsnap_d0",
    "snapshot_commit": "a238cd304ce7d3cc90b4136666788e0d98091fdd",
    "snapshot_created_from": "git -C $CARTRIDGES_DIR archive HEAD cartridges examples | tar -x -C /tmp/amsnap_d0",
    "verified_import_path": "/tmp/amsnap_d0/cartridges (probe run from /tmp, printed into every cell log as 'RESOLVED cartridges -> ...')",
    "driver_executed": "/tmp/amsnap_d0/examples/qasper2/train/eval_forgetting.py (driver ALSO comes from the snapshot, not the worktree)",
    "snapshot_manifest_sha256_before_and_after": "c6e90e3e98d81b80a6faf340496ad5e6f98cb30b40c9d1f7b98e4b466103f693",
    "manifest_verify_method": "sha256 over sorted per-file sha256 of cartridges/ + examples/ (excluding __pycache__), compared against a fresh `git archive a238cd3` extraction after the runs; diff -rq also clean",
    "repo_head_at_start": "d1dbf7cd10c92b9d0734273937b8b7a672253da8",
    "repo_head_when_snapshot_taken": "a238cd304ce7d3cc90b4136666788e0d98091fdd",
    "repo_head_at_end": "383720786dfab33ba04012b9e6f9c747ed17cdd5",
    "head_moved_during_run": True,
    "concurrent_editor": "MECH-KEYS editing cartridges/am/key_select.py",
    "gpu_claim": "repo flock /tmp/gpu_locks_$USER/gpu{0,1}.lock, held for the life of each cell launcher; QA|QA + MT|MT on gpu0 (serialized), QA|MT + MT|QA on gpu1",
    "all_cells_exit_code": 0,
    "note_on_hang": "EVAL_MODE=icl does NOT hang (unlike the pydrantic cartridge path, RUNBOOK §1): all four processes returned rc=0 on their own, so no PID had to be killed.",
    "wandb_note": (
        "eval_forgetting.py's ICL branch (_run_icl) never initialises wandb — it only prints "
        "'Eval loss - <f>' and '[icl] {metrics}'. Under the no-source-edit rule the run record was "
        "created by research_loop/results/D0-ICL/log_cell_to_wandb.py, which parses those exact "
        "printed lines and logs them (project SEACrowd, entity vqtri-purdue-university, group REF-ICL, "
        "tag 'diagnostic'). The losses are the harness's own, not recomputed."
    ),
}
payload["interpretation"] = {
    "harness_change_moves_the_numbers": False,
    "max_abs_delta_vs_prior_board": 4.6e-05,
    "ceiling_correctly_placed": True,
    "icl_mt_vs_dense_bar": "ICL MT 1.89605 is ABOVE (worse than) the dense@4ep bar MT 1.87250 by 0.0235 loss — inside the ~0.15 noise band, i.e. a tie, but the point estimate has the 'ceiling' behind the bar.",
    "icl_qa_vs_sparse_gradient": "ICL QA 1.97341 is ABOVE (worse than) the sparse-gradient reference QA 1.61690 by 0.3565 loss — outside the noise band, i.e. ICL is NOT the QA ceiling.",
    "off_diagonal_topic_general_bound": (
        "Wrong-topic full context is far worse than right-topic: icl_QA|MT 2.77610 vs icl_MT|MT 1.89605 "
        "(+0.880), icl_MT|QA 2.90249 vs icl_QA|QA 1.97341 (+0.929). Relative to the Phase-1 start, "
        "wrong-topic QA papers still buy 3.78255 - 2.77610 = 1.00645 of the MT drop that right-topic "
        "papers buy (3.78255 - 1.89605 = 1.88650), i.e. 53.3% of ICL's own MT gain is available with "
        "content that is topic-mismatched by construction."
    ),
}

with open(DIAG, "w") as f:
    json.dump(payload, f, indent=2)
print("wrote", DIAG)
print(json.dumps(payload["diagonals"], indent=2))
print(json.dumps(payload["off_diagonals"], indent=2))
print(json.dumps(payload["delta_vs_prior"], indent=2))
