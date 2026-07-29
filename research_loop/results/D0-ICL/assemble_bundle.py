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

with open(DIAG, "w") as f:
    json.dump(payload, f, indent=2)
print("wrote", DIAG)
print(json.dumps(payload["diagonals"], indent=2))
print(json.dumps(payload["off_diagonals"], indent=2))
print(json.dumps(payload["delta_vs_prior"], indent=2))
