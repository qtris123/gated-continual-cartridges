#!/usr/bin/env python3
"""REF-ICL-v2: re-measure the two DIAGONAL full-context ICL ceilings on the SAME
eval parquets that eval_forgetting.py / the cartridge & AM numbers use, so ICL is
apples-to-apples with the rest of the 4B study.

Diagonals only (off-diagonals skipped to save time):
  * icl_QA_raw | QA  = QA papers full-context, evaluated on qasper_eval_QA.parquet
  * icl_MT_raw | MT  = MT papers full-context, evaluated on qasper_eval_MT.parquet

Reuses the EXACT functions from qasper_loss_benchmark.py so the mean-CE (assistant
tokens) metric matches eval_forgetting.py by design. Raw Qwen3-4B-Instruct-2507,
NO cartridge, all topic papers placed in the system prompt, chunked prefill 2048,
bf16, teacher-forced mean cross-entropy on assistant tokens.
"""
import json
import sys
from pathlib import Path

REPO = "/localhome/local-triv/gated-continual-cartridges_explore"
sys.path.insert(0, f"{REPO}/examples/qasper2/experiments")

import qasper_loss_benchmark as qlb  # noqa: E402

MODEL = "Qwen/Qwen3-4B-Instruct-2507"
DEVICE = "cuda"
QA_EVAL = Path(f"{REPO}/data/qasper/eval/qasper_eval_QA.parquet")
MT_EVAL = Path(f"{REPO}/data/qasper/eval/qasper_eval_MT.parquet")
OUT = Path(f"{REPO}/research_loop/results/REF-ICL-v2/raw_metrics.json")

print(f"[cfg] model={MODEL} qa_eval={QA_EVAL} mt_eval={MT_EVAL}", flush=True)

model, tokenizer = qlb.load_model_and_tokenizer(MODEL, DEVICE)
model_dev = qlb._model_device(model)
print(f"[loaded] model on {model_dev}", flush=True)

# Full-paper ICL system prompts (topic-matched; no truncation).
qa_sys = qlb.build_qasper_system_prompt("QA", tokenizer=tokenizer, max_context_tokens=None)
mt_sys = qlb.build_qasper_system_prompt("MT", tokenizer=tokenizer, max_context_tokens=None)
qa_ctx_tok = len(tokenizer.encode(qa_sys, add_special_tokens=False))
mt_ctx_tok = len(tokenizer.encode(mt_sys, add_special_tokens=False))
print(f"CONTEXT_TOKENS QA_sysprompt={qa_ctx_tok} MT_sysprompt={mt_ctx_tok}", flush=True)

# --- Diagonal 1: QA papers full-context on QA eval (the QA ceiling) ---
qa_res = qlb.evaluate_loss_chunked(
    model=model,
    cartridge=None,
    tokenizer=tokenizer,
    eval_path=QA_EVAL,
    system_prompt=qa_sys,
    device=str(model_dev),
    desc="icl_QA_raw|QA",
    prefill_chunk_size=2048,
)
print("RESULT_DICT_QA", qa_res, flush=True)
print(
    f"FINAL_QA_ICL loss={qa_res['loss']:.4f} ppl={qa_res['perplexity']:.4f} "
    f"n_tok={qa_res['num_target_tokens']} n_ex={qa_res['num_examples']} "
    f"sys_tok={qa_res['system_prompt_tokens']} max_seq={qa_res['max_sequence_tokens']}",
    flush=True,
)
# checkpoint after QA in case MT is killed
OUT.write_text(json.dumps({"qa": qa_res, "mt": None,
                           "qa_context_tokens": qa_ctx_tok,
                           "mt_context_tokens": mt_ctx_tok}, indent=2))

# --- Diagonal 2: MT papers full-context on MT eval (the MT ceiling) ---
mt_res = qlb.evaluate_loss_chunked(
    model=model,
    cartridge=None,
    tokenizer=tokenizer,
    eval_path=MT_EVAL,
    system_prompt=mt_sys,
    device=str(model_dev),
    desc="icl_MT_raw|MT",
    prefill_chunk_size=2048,
)
print("RESULT_DICT_MT", mt_res, flush=True)
print(
    f"FINAL_MT_ICL loss={mt_res['loss']:.4f} ppl={mt_res['perplexity']:.4f} "
    f"n_tok={mt_res['num_target_tokens']} n_ex={mt_res['num_examples']} "
    f"sys_tok={mt_res['system_prompt_tokens']} max_seq={mt_res['max_sequence_tokens']}",
    flush=True,
)

OUT.write_text(json.dumps({"qa": qa_res, "mt": mt_res,
                           "qa_context_tokens": qa_ctx_tok,
                           "mt_context_tokens": mt_ctx_tok}, indent=2))
print(f"[wrote] {OUT}", flush=True)
print("[done] rc=0", flush=True)
