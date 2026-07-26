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

## 2026-07-25 18:50 [EXP-003] with-IDF canonical — IDF HURTS both axes (HYP-G1 supported)
AM-sparse USE_IDF=1 (bg_stats over Phase-1 QA corpus, tfidf, per_layer, top_t=64, freeze, ridge 1e-4 spectral),
single var vs EXP-001 (USE_IDF=0): QA-forgetting 2.6351 (ppl 13.9), MT-acquisition 3.0073 (ppl 20.2). vs EXP-001
(QA 2.2521 / MT 2.5426): QA +0.383, MT +0.465 — both >2× the tiny-eval noise band. value_global_max_abs 486
(vs 816) confirms IDF changed slot selection. solve 178.4s / e2e 201s / 0 grad. ⇒ canonical = no-IDF (EXP-001).
Artifacts: research_loop/results/EXP-003/result.json ; outputs/2026-07-25-18-47-23-continual_am_sparse/5989fef9.../.

## 2026-07-25 18:40 [SETUP-BG1] bg_stats collected over the Phase-1 self-distilled cache (infra)
Built examples/qasper2/train/collect_bg_stats.py; ran forward-only over the full Phase-1 QA corpus
(data/qasper/train/qwen_qasper_QA_task_8192.parquet, 1990 packed seqs) against
outputs/phase1_selfdistill_qwen512/cache_last.pt → outputs/phase1_selfdistill_qwen512/bg_stats.pt (292MB,
per_layer, access counts (1990,36,511) int64; derived IDF (36,511) finite). collect 265.3s. Verified loadable
by continual_am_sparse.py's bg_tracker.load()+CacheTFIDFRanker. Fix vs sketch: NUM_BG_BATCHES=full-corpus,
cache.to(local_rank) after from_pretrained. Artifact: research_loop/results/SETUP-BG1/result.json.

## 2026-07-25 18:20 [EXP-002] wandb cross-check + AM-paper facts (research)
Historical DENSE self-distillation Phase-2 (wandb SEACrowd scf175an; dense/no-sparse, adam lr2e-2, 10ep):
MT-acquisition loss 2.891 (ppl 18.01); QA-forgetting split NOT logged. Qwen Phase-1-start/no-P2 reference:
QA loss 2.051 (ppl 7.78) / MT loss 3.583 (ppl 36.0). Dense P2 train wall-clock 1805s (~30min); MT synth
10.3–11.7 ks. AM paper (2602.16284): L2 ridge on the VALUE-solve degrades ∀λ>0; β (NNLS mass-bias) is the
low-mass cure; per-head budget = top ablation. bg_stats background corpus = Phase-1 QA parquet, per_layer.
Artifact: research_loop/results/EXP-002/result.json.

## 2026-07-25 18:02 [EXP-001] AM-sparse no-IDF Phase-2 = first efficiency+quality point
Closed-form AM-sparse (USE_IDF=0, tfidf ranker, per_layer, top_t=64, freeze keys, ridge spectral) from
outputs/phase1_selfdistill_qwen512/cache_last.pt over 16 MT docs: QA forgetting loss 2.2521 (+0.013 vs 2.239
floor), MT acquisition loss 2.5426 (−1.240 vs 3.783 floor; ppl 44→12.7). solve_s 181.4 / phase2_e2e_s 217 /
gradient_steps 0. value_global_max_abs 816 (deep layers) despite intact QA. Artifacts:
research_loop/results/EXP-001/result.json ; outputs/2026-07-25-18-02-29-continual_am_sparse/efa3bc76.../cache_last.pt.
