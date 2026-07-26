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

## 2026-07-26 10:2x [EXP-006/β] β NNLS fit produces NaN — β deferred; import foot-gun found
EXP-006 (β + opt-in clamp[-3,3]) died `AssertionError: beta clamp failed: max|beta|=nan > 3.0` — the NNLS β-fit
itself yields NaN, so clamping the output can't help. β failed 3 ways (EXP-005 cholesky not-PD @66.6; EXP-005b
lstsq NaN; EXP-006 fit NaN). β DEFERRED (needs numerical guards in refit_beta_nnls). Separately: the β-clamp edit
made continual_am_sparse.py pass beta_clamp_abs UNCONDITIONALLY, which crashed ALL AM runs (incl EXP-007 top_t)
when `import cartridges` resolved to the SIBLING repo (no field). Edit REVERTED; foot-guns added RUNBOOK §6.10/6.11.

## 2026-07-26 09:2x [EXP-000] dense 10-epoch REF-CART bar = QA 2.6991 / MT 2.2137 (OVERFIT)
Final 10-epoch dense self-distillation checkpoint (cache-step624, ~101min train) eval'd both splits:
QA-forgetting 2.6991, MT-acquisition 2.2137. Both WORSE than dense@4ep (QA 2.3721 / MT 1.8725) → 10ep overfit
(QA-forgetting grows with epochs; MT regressed past ~4ep). Best dense point = @4ep. vs AM EXP-001 (QA 2.2521 /
MT 2.5426): AM retains QA better than both dense points; dense@4ep acquires MT better (gap 0.67). Artifact: results/EXP-000/result.json.

## 2026-07-26 08:51 [EXP-005b] β at RIDGE_LAMBDA=0 also failed (NaNs) — confirms unclamped β is root cause
ENABLE_BETA=1 with RIDGE_LAMBDA=0 (robust lstsq/gels path) crashed with "RuntimeError: NaNs in ridge lstsq
solution". So the failing solve branch is not the cause — the unclamped β log-weights themselves are. Both β
runs (λ=1e-4 cholesky, λ=0 lstsq) die. HYP-T2 needs the β-clamp (AM paper [-3,3]) → EXP-006 implements it.

## 2026-07-26 07:31 [EXP-005] ENABLE_BETA=1 crashed (β numerically unstable at λ=1e-4)
β engaged (max|β_logweight| 66.6, mean 4.23, ~19% nonzero, all 36 layers) but crashed on doc 3 with
linalg.cholesky not-positive-definite (am/core.py:225 _ridge_lstsq). Mechanism: unclamped β added to attention
logits pre-softmax → near-one-hot softmax → rank-deficient value-solve design matrix → XtX indefinite → cholesky
fails (primary + raised-floor fallback; λ=1e-4 too small). Docs 0-2 succeeded. No eval numbers (crash pre-eval).
AM paper clamps β∈[-3,3]; our impl doesn't. Retry EXP-005b (β at λ=0, robust lstsq). Artifact: results/EXP-005/result.json.

## 2026-07-26 01:18 [EXP-004] RIDGE_LAMBDA=0 vs 1e-4 on no-IDF canonical = WASH
Pure OLS value-solve (λ=0, ridge truly zeroed via core.py L144-145 gels branch), single var vs EXP-001:
QA-forgetting 2.2619 (ppl 9.6), MT-acquisition 2.5569 (ppl 12.9). vs EXP-001 (QA 2.2521 / MT 2.5426):
+0.010 / +0.014 — an order of magnitude under the ~0.1-0.2 noise band. solve 162.9s / e2e 186s / 0 grad.
value_global_max_abs 848. ⇒ λ=1e-4 ridge already negligible; keep canonical. Artifact: results/EXP-004/result.json.

## 2026-07-26 01:08 [infra][EXP-000] dense REF-CART run died mid-train + agent restarted from scratch
Attempt-1 dense baseline_continual (18:02 run dir) reached ~Epoch 5-6 (step ~329) then its process died; the
parked executor agent restarted training from Epoch 1 (new run dir 2026-07-26-01-07-07, gpu0) on a ~55-min
re-arm cycle. 10-epoch (~110min) dense may not finish in one window on this box; restarts lose progress.
Usable artifact: attempt-1 cache-step256.pt (~4 dense epochs). Dense MT bar already known (wandb scf175an
MT-acq 2.891). Plan: eval step256 for dense QA-forgetting rather than wait on a full run.

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
