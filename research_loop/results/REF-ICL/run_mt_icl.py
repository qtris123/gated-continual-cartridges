#!/usr/bin/env python3
"""REF-ICL: rerun ONLY the missing icl_MT_raw|MT eval (MT papers full-context, raw model,
no cartridge) as mean-CE loss on data/qasper/eval/qasper_eval_MT.parquet.

Reuses the exact functions from qasper_loss_benchmark.py so the metric matches the
QA-split number already captured (icl_QA_raw|QA loss=1.9734). Also prints the QA/MT
full-context system-prompt token counts for the observations.
"""
import sys
from pathlib import Path

REPO = "/localhome/local-triv/gated-continual-cartridges_explore"
sys.path.insert(0, f"{REPO}/examples/qasper2/experiments")

import qasper_loss_benchmark as qlb  # noqa: E402

MODEL = "Qwen/Qwen3-4B-Instruct-2507"
DEVICE = "cuda"

model, tokenizer = qlb.load_model_and_tokenizer(MODEL, DEVICE)
model_dev = qlb._model_device(model)

# Full-paper ICL system prompts (topic-matched).
qa_sys = qlb.build_qasper_system_prompt("QA", tokenizer=tokenizer, max_context_tokens=None)
mt_sys = qlb.build_qasper_system_prompt("MT", tokenizer=tokenizer, max_context_tokens=None)
qa_ctx_tok = len(tokenizer.encode(qa_sys, add_special_tokens=False))
mt_ctx_tok = len(tokenizer.encode(mt_sys, add_special_tokens=False))
print(f"CONTEXT_TOKENS QA_sysprompt={qa_ctx_tok} MT_sysprompt={mt_ctx_tok}", flush=True)

res = qlb.evaluate_loss_chunked(
    model=model,
    cartridge=None,
    tokenizer=tokenizer,
    eval_path=Path(f"{REPO}/data/qasper/eval/qasper_eval_MT.parquet"),
    system_prompt=mt_sys,
    device=str(model_dev),
    desc="icl_MT_raw|MT",
    prefill_chunk_size=2048,
)
print("RESULT_DICT", res, flush=True)
print(
    f"FINAL_MT_ICL loss={res['loss']:.4f} ppl={res['perplexity']:.4f} "
    f"n_tok={res['num_target_tokens']} n_ex={res['num_examples']} "
    f"sys_tok={res['system_prompt_tokens']} max_seq={res['max_sequence_tokens']}",
    flush=True,
)
print("[done] rc=0", flush=True)
