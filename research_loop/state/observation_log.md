# OBSERVATION LOG

Append-only. One dated entry per observation. Observation = what was measured, raw.
Keep interpretation in the hypothesis ledger, not here (research-companion discipline).

Format: `## <YYYY-MM-DD HH:MM> [EXP-ID] observation` then 1-4 lines of raw fact + artifact link.

---
## 2026-07-25 06:16 [EXP-000-verify] HF Phase-1 cache = QA cache (provenance)
HF `qtris123/qwen_qasper-QA-task_8192_512_no-cartridge_10-epochs` cache_last.pt eval'd on both splits
(Qwen3-4B, this harness): QA loss 2.239 (ppl ~9.4), MT loss 3.783 (ppl ~44). QA≪MT → QA Phase-1 cache;
repo name correct, bundled config.yaml (MT-labelled) is stale. Cache = 511 trainable + 1 frozen slot.
Artifact: research_loop/results/EXP-000-verify/result.json, /tmp/exp000_{qa,mt}.log.

## 2026-07-25 06:16 [infra] eval_forgetting.py hangs post-completion; nvidia-smi flaky
eval_forgetting.py printed `Eval loss` then never exited (held GPU 1 ~78 min until killed). nvidia-smi
repeatedly >20s/hangs on this box. Guardrails added to RUNBOOK §1/§6. Harness `Eval loss` = ln(ppl).

## 2026-07-25 18:02 [EXP-001] AM-sparse no-IDF Phase-2 = first efficiency+quality point
Closed-form AM-sparse (USE_IDF=0, tfidf ranker, per_layer, top_t=64, freeze keys, ridge spectral) from
outputs/phase1_selfdistill_qwen512/cache_last.pt over 16 MT docs: QA forgetting loss 2.2521 (+0.013 vs 2.239
floor), MT acquisition loss 2.5426 (−1.240 vs 3.783 floor; ppl 44→12.7). solve_s 181.4 / phase2_e2e_s 217 /
gradient_steps 0. value_global_max_abs 816 (deep layers) despite intact QA. Artifacts:
research_loop/results/EXP-001/result.json ; outputs/2026-07-25-18-02-29-continual_am_sparse/efa3bc76.../cache_last.pt.
