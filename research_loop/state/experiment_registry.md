# EXPERIMENT REGISTRY

One `### EXP-XXX` block per experiment. Fill BEFORE dispatch (status `dispatched`), complete on ingest.
Format mirrors `.cursor/rules/research-companion.mdc`.

## Template (copy for each new experiment)
```
### EXP-XXX: <short-name>
- Date:
- Research question:
- Hypothesis tested (HYP-ID):
- Variable under test (the ONE thing changed):
- Baseline compared to (EXP-ID / config):
- Code branch+commit:
- Dataset / Model:  Qasper QA→MT / Qwen3-4B
- Config knobs (full):  slot_selection= use_idf= granularity= top_t= target_mode= key_mode=
                        ridge_lambda= ridge_scale= delta_weight= freeze_keys= ...
- Command:
- Metrics reported:  qa_forgetting_loss / mt_acquisition_loss (mean CE) / runtime_s
- Expected:
- Actual:
- Interpretation:
- Confounders considered:
- Status:  dispatched | done | failed
- Follow-up:
- Artifacts:  outputs/<run-dir>/ , logs/<log> , research_loop/results/EXP-XXX/result.json
```

---
<!-- experiments appended below by the orchestrator -->

### EXP-000: REF-CART — dense self-distillation Phase-2 (the quality BAR)
- Date: 2026-07-25 (cycle 0)
- Research question: What QA-forgetting / MT-acquisition eval-loss does the cartridge's own dense
  self-distillation Phase-2 reach from the verified Phase-1 cache — and at what training cost?
- Hypothesis tested (HYP-ID): n/a (reference-line anchor)
- Variable under test: none — this IS the baseline all AM-sparse configs must match.
- Baseline compared to: PHASE1 start point (QA loss 2.239 retention floor / MT loss 3.783 ceiling).
- Code branch+commit: trivo-explore-research-work @ 11d1217
- Dataset / Model: Qasper QA→MT / Qwen3-4B (Qwen/Qwen3-4B-Instruct-2507), 512 slots
- Config knobs (full): dense gradient self-distillation; LR 2e-2, EPOCHS 10, GLOBAL_BATCH_SIZE 32,
  DISTRIBUTED_BACKEND=gloo, NUM_TOKENS 512, PHASE1_CACHE=outputs/phase1_selfdistill_qwen512/cache_last.pt,
  SYNTH=qwen_qasper_MT_task_8192.parquet. (no AM/sparse knobs — this is baseline_continual.py)
- Command: RUNBOOK §3(a) validated pattern, real run (drop max_optimizer_steps, EPOCHS=10, eval BOTH splits)
- Metrics reported: qa_forgetting_loss / mt_acquisition_loss (mean CE) + train wall-clock/gpu_s (T-cart cost)
- Expected: MT loss ≪ 3.783 (acquisition), QA loss stays near/above 2.239 (some forgetting). This defines
  the target: AM-sparse must reach ≤ QA+0.15 AND ≤ MT+0.15 at materially lower train cost.
- Actual: _pending_
- Interpretation: _pending_
- Confounders considered: LAST≠BEST ckpt (§6.3) — eval the intended step; final-barrier hang (§3a) — ckpt
  saved before it, kill PID after eval; tiny eval set (n=5-6) — sub-0.2 deltas are noise.
- Status: dispatched
- Follow-up: cross-check vs wandb SEACrowd historical dense P2 (EXP-002).
- Artifacts: outputs/<run-dir>/ , logs/ , research_loop/results/EXP-000/result.json

### EXP-001: AM-sparse Phase-2 anchor (no-IDF) — first efficiency+quality point
- Date: 2026-07-25 (cycle 0)
- Research question: What QA/MT eval-loss + end-to-end cost does closed-form AM-sparse Phase-2 reach
  from the verified self-distilled Phase-1 cache, through the SAME eval harness as EXP-000?
- Hypothesis tested (HYP-ID): doubles as HYP-G1 "attention_mass / pure-TF (no IDF)" arm.
- Variable under test: method = AM-sparse closed-form (vs EXP-000 dense gradient) — the efficiency axis.
- Baseline compared to: EXP-000 (REF-CART) for quality; PHASE1 for retention floor.
- Code branch+commit: trivo-explore-research-work @ 11d1217
- Dataset / Model: Qasper QA→MT / Qwen3-4B, 512 slots
- Config knobs (full): slot_selection=tfidf, USE_IDF=0 (no bg_stats needed — pure TF ranking),
  granularity=per_layer, top_t=64, target_mode=cartridge_plus_doc, key_mode=freeze (freeze_keys),
  ridge_lambda=1e-4, ridge_scale=spectral, delta_weight=1e-2, AM_EXECUTION_MODE=per_document,
  PHASE1_CACHE=outputs/phase1_selfdistill_qwen512/cache_last.pt, SYNTH=qwen_qasper_MT_task_8192.parquet.
- Command: VAR=val ... bash examples/qasper2/scripts/train_continual_am_sparse.sh (RUNBOOK §3b), USE_IDF=0.
- Metrics reported: qa_forgetting_loss / mt_acquisition_loss (mean CE) + solve_s (T1) + phase2_e2e_s (T2) + gpu_s.
- Expected: closed-form, gradient_steps=0; cost ≪ EXP-000; quality somewhere between PHASE1 and REF-CART.
- Actual: QA forgetting 2.2521 (+0.013 vs 2.239 floor → no forgetting, within noise); MT acquisition
  2.5426 (−1.240 vs 3.783 floor; ppl 44→12.7). solve_s 181.4 / phase2_e2e_s 217 / gradient_steps 0 (16 docs).
- Interpretation: closed-form, backprop-free AM-sparse achieves large MT acquisition with intact QA
  retention (freeze-keys) in ~3.6 min — a strong first efficiency+quality point. Caveat: value_global_max_abs
  816 (deep layers) despite intact QA. Not yet comparable to REF-CART (EXP-000 pending) or with-IDF (EXP-003).
- Confounders considered: NOT the with-IDF canonical (that needs bg_stats over THIS cache — deferred EXP-003);
  historical "current best" 7.13/7.28 was top32 per_head — different config, not a strict repro target;
  tiny eval (QA n=6, MT n=5) so the QA +0.013 is noise-level.
- Status: done
- Follow-up: with-IDF canonical (EXP-003) next cycle once bg_stats-over-our-cache recipe is confirmed (EXP-002).
- Artifacts: outputs/<run-dir>/ , logs/ , research_loop/results/EXP-001/result.json

### EXP-002: RESEARCH — cited papers, cartridge cost figures, wandb cross-check, bg_stats recipe
- Date: 2026-07-25 (cycle 0)
- Research question: (a) concrete, knob-mapped gating/target ideas from AM.pdf + TF-IDF.pdf; (b) cartridge
  paper (2506.06266) synthesis + Phase-2 train-cost figures for the T3 story; (c) wandb SEACrowd historical
  dense Phase-2 numbers to cross-check EXP-000; (d) the EXACT recipe to collect bg_stats over OUR
  self-distilled Phase-1 cache (which corpus = background, how initial_am does it, is a standalone
  collector script needed) so the with-IDF canonical is unblocked next cycle.
- Hypothesis tested (HYP-ID): feeds HYP-G1..G4, GATE-N1..N3 (candidate hypotheses only — no decisions).
- Variable under test: n/a (research)
- Baseline compared to: n/a
- Command: read PDFs (repo root), WebSearch/WebFetch as needed, fetch_wandb_data.py, read
  cartridges/sparse_cache_finetuning.py collect_background_stats + examples/.../initial_am.py.
- Metrics reported: extracted numbers (cartridge synth/train cost, historical dense P2 loss) in numbers{}.
- Expected: a T3 cost table skeleton + a confirmed bg_stats collection recipe + 3-6 knob-mapped ideas.
- Actual: DELIVERED. (a) AM-faithful teacher = orig attention OUTPUT + MASS; value-solve = closed-form OLS —
  **paper: L2 ridge on the value-solve HURTS ∀λ>0** (distinct from Phase-1 recon λ=2.0); β mass-bias (NNLS)
  is AM's cure for low-mass (reweight retained high-mass slots), NOT gating into low-mass; per-head budget =
  AM's #1 ablation. TF-IDF: specificity>magnitude, IDF matters most at small TOP_T, background = PRESERVE set.
  (b) T3: dense P2 train ~1805s (wandb scf175an, ~30min) + MT synth ~10-12ks/task; AM EXP-001 e2e=217s.
  (c) dense P2 MT-acquisition loss 2.891 (ppl 18.0, scf175an); QA-forgetting NOT logged → EXP-000 authoritative.
  (d) bg_stats recipe CONFIRMED: background = Phase-1 QA parquet (qwen_qasper_QA_task_8192), per_layer,
  ~25-line standalone collector → outputs/phase1_selfdistill_qwen512/bg_stats.pt (sketch in bundle).
- Interpretation: EXP-001's MT 2.5426 already BEATS the historical dense P2 MT 2.891 backprop-free — pending
  EXP-000's authoritative local re-run. New single-var levers surfaced: RIDGE_LAMBDA=0, ENABLE_BETA=1, per-head budget.
- Status: done
- Follow-up: SETUP-BG1 (collect bg_stats now) → EXP-003 (with-IDF canonical). Add ridge/β/per-head to backlog.
- Artifacts: research_loop/results/EXP-002/result.json

### SETUP-BG1: collect bg_stats over the self-distilled Phase-1 cache (infra, unblocks USE_IDF=1)
- Date: 2026-07-25 (cycle 0, pipelined on gpu1 during EXP-000)
- Research question: n/a — infrastructure. Produce the IDF background statistics for OUR Phase-1 cache so
  any USE_IDF=1 AM-sparse run can load them.
- Variable under test: n/a
- Baseline compared to: n/a
- Dataset / Model: background = data/qasper/train/qwen_qasper_QA_task_8192.parquet (Phase-1 QA PRESERVE set),
  cache = outputs/phase1_selfdistill_qwen512/cache_last.pt, Qwen3-4B, granularity=per_layer.
- Command: implement ~25-line standalone collector (per EXP-002 sketch) → collect_background_stats(...,
  granularity='per_layer', save_path='outputs/phase1_selfdistill_qwen512/bg_stats.pt'); single-process, GPU forward-only.
- Metrics reported: bg_stats.pt exists + shape/layer/head sanity (36 layers, 8 KV heads, per_layer stats); collect wall-clock.
- Expected: a loadable bg_stats.pt reusable by all future USE_IDF=1 gating experiments.
- Actual: DONE. bg_stats.pt written (292MB), per_layer, access-counts shape (1990,36,511) int64 over the full
  QA corpus (1990 packed seqs); derived IDF shape (36,511) finite. Sanity-verified via the exact Phase-2 path
  (BackgroundAccessTracker.load + CacheTFIDFRanker). collect wall-clock 265.3s (cold flex_attention compile).
  Collector: examples/qasper2/train/collect_bg_stats.py. Two adaptations: NUM_BG_BATCHES=999999999 (full corpus,
  not 1000 which truncates coverage) + cache.to(local_rank) after from_pretrained (buffers left on CPU) — mirrors continual_am_sparse.py.
- Status: done
- Follow-up: EXP-003 = with-IDF canonical (tfidf, USE_IDF=1, per_layer, top_t=64) loading this bg_stats.pt.
- Artifacts: outputs/phase1_selfdistill_qwen512/bg_stats.pt , examples/qasper2/train/collect_bg_stats.py , research_loop/results/SETUP-BG1/result.json

### EXP-003: AM-sparse with-IDF canonical (the TRUE Lever-1 baseline + HYP-G1 with-IDF arm)
- Date: 2026-07-25 (cycle 0, pipelined on gpu1 during EXP-000)
- Research question: does IDF gating (tfidf, USE_IDF=1) beat pure-TF (EXP-001, USE_IDF=0) on MT-acquisition at
  equal-or-lower QA-forgetting? Establishes the canonical AM-sparse baseline all later gating experiments compare to.
- Hypothesis tested (HYP-ID): HYP-G1 (this is the with-IDF arm; EXP-001 is the no-IDF arm).
- Variable under test: USE_IDF (0→1, loading SETUP-BG1 bg_stats.pt). EVERYTHING else identical to EXP-001.
- Baseline compared to: EXP-001 (no-IDF, QA 2.2521 / MT 2.5426); also PHASE1 floor + REF-CART (EXP-000).
- Code branch+commit: trivo-explore-research-work @ 92fcb24
- Dataset / Model: Qasper QA→MT / Qwen3-4B, 512 slots
- Config knobs (full): slot_selection=tfidf, USE_IDF=1, BG_STATS_PATH=outputs/phase1_selfdistill_qwen512/bg_stats.pt,
  granularity=per_layer, top_t=64, target_mode=cartridge_plus_doc, key_mode=freeze, ridge_lambda=1e-4,
  ridge_scale=spectral, delta_weight=1e-2, AM_EXECUTION_MODE=per_document, PHASE1_CACHE=.../cache_last.pt, SYNTH=MT parquet.
- Command: VAR=val ... bash examples/qasper2/scripts/train_continual_am_sparse.sh (RUNBOOK §3b), USE_IDF=1 + BG_STATS_PATH set.
- Metrics reported: qa_forgetting_loss / mt_acquisition_loss (mean CE) + solve_s + phase2_e2e_s + gpu_s + gradient_steps:0.
- Expected: closed-form; either lower MT at ≤ QA vs EXP-001 (IDF helps) or a wash (IDF matters more at small top_t → HYP-a5).
- Actual: IDF HURTS BOTH. QA-forgetting 2.6351 (ppl 13.9) = +0.383 vs EXP-001; MT-acquisition 3.0073 (ppl 20.2)
  = +0.465 vs EXP-001. Both deltas >2× noise band. solve_s 178.4 / e2e 201 / 0 grad. value_max 486 (vs 816) ⇒ IDF
  really changed slot selection (valid single-var test).
- Interpretation: In the compressed-KV-cartridge setting the IDF/document-frequency term is the WRONG gate — it
  worsened acquisition AND (counterintuitively) retention. Pure attention-mass (TF) on the new doc selects slots that
  are both better for acquisition and less disruptive to Phase-1. Directly validates the project's core problem.
  ⇒ CANONICAL AM-sparse baseline going forward = the NO-IDF config (EXP-001), NOT this with-IDF one.
- Confounders considered: tiny eval (QA n=6, MT n=5) but deltas are large; single seed; only top_t=64 tested
  (HYP-a5 = IDF at top_t=32 is the natural follow-up since literature says IDF helps most at small support).
- Status: done
- Follow-up: HYP-G1 → SUPPORTED. Next: HYP-R0 (RIDGE_LAMBDA=0) + HYP-T2 (ENABLE_BETA=1) on the no-IDF canonical.
- Artifacts: outputs/<run-dir>/ , research_loop/results/EXP-003/result.json

### EXP-004: HYP-R0 — RIDGE_LAMBDA=0 vs 1e-4 on the no-IDF canonical
- Date: 2026-07-25 (cycle 0→1, pipelined on gpu1 during EXP-000)
- Research question: does removing the L2 ridge on the value-solve (λ=0, pure OLS) beat the canonical λ=1e-4?
  AM paper reports ridge on the (β,C_v) value-solve DEGRADES for ALL λ>0.
- Hypothesis tested (HYP-ID): HYP-R0.
- Variable under test: RIDGE_LAMBDA (1e-4 → 0). EVERYTHING else = the no-IDF canonical (EXP-001).
- Baseline compared to: EXP-001 (no-IDF canonical: QA 2.2521 / MT 2.5426).
- Code branch+commit: trivo-explore-research-work @ (post-EXP-003 ingest)
- Dataset / Model: Qasper QA→MT / Qwen3-4B, 512 slots
- Config knobs (full): slot_selection=tfidf, USE_IDF=0, granularity=per_layer, top_t=64, target_mode=cartridge_plus_doc,
  key_mode=freeze, RIDGE_LAMBDA=0, ridge_scale=spectral (moot at λ=0), ridge_lambda_min=0, delta_weight=1e-2,
  AM_EXECUTION_MODE=per_document, PHASE1_CACHE=.../cache_last.pt, SYNTH=MT parquet. NO BG_STATS_PATH (USE_IDF=0).
- Command: VAR=val ... bash examples/qasper2/scripts/train_continual_am_sparse.sh (RUNBOOK §3b), RIDGE_LAMBDA=0, USE_IDF=0.
- Metrics reported: qa_forgetting_loss / mt_acquisition_loss (mean CE) + solve_s + phase2_e2e_s + gradient_steps:0.
- Expected: per AM paper, λ=0 ≥ canonical on quality (esp. acquisition) at ~equal cost; or a wash if 1e-4 is already ~0.
- Actual: WASH. QA 2.2619 / MT 2.5569 vs EXP-001 (QA 2.2521 / MT 2.5426) → +0.010 / +0.014, ≪ noise band.
  solve_s 162.9 / e2e 186 / 0 grad. Executor verified ridge truly zeroed (effective λ short-circuits to 0 at
  core.py L144-145, `_ridge_lstsq` OLS gels branch); DELTA_WEIGHT trust-region held identical → isolates ridge only.
- Interpretation: at λ=1e-4 the L2 ridge on the value-solve is already negligible; removing it neither helps nor
  hurts. The AM-paper "ridge hurts ∀λ>0" claim doesn't bite at our tiny default. Keep canonical (λ indifferent);
  do not sweep ridge further on this axis.
- Confounders considered: tiny eval; single seed; λ=1e-4 is already tiny so effect may be small (confirmed: it is).
- Status: done
- Follow-up: HYP-R0 resolved (null/wash). Next lever: HYP-T2 (ENABLE_BETA=1, AM mass-bias).
- Artifacts: outputs/<run-dir>/ , research_loop/results/EXP-004/result.json

### EXP-005: HYP-T2 — ENABLE_BETA=1 (AM per-token mass-bias) on the no-IDF canonical
- Date: 2026-07-26 (cycle 1, gpu1 during dense full run)
- Research question: does the AM per-token β mass-bias (NNLS, w_j=exp(β_j)≥0 so retained slots carry the
  attention mass lost by subsetting) close the MT-ACQUISITION gap vs dense, without hurting QA retention?
- Hypothesis tested (HYP-ID): HYP-T2 (elevated by EXP-002 — β is AM's core low-mass cure).
- Variable under test: ENABLE_BETA (unset/0 → 1). EVERYTHING else = the no-IDF canonical (EXP-001).
- Baseline compared to: EXP-001 (no-IDF canonical: QA 2.2521 / MT 2.5426). Target = dense MT (REFCART-EVAL 1.8725 @4ep / full 10ep pending).
- Dataset / Model: Qasper QA→MT / Qwen3-4B, 512 slots
- Config knobs (full): slot_selection=tfidf, USE_IDF=0, granularity=per_layer, top_t=64, target_mode=cartridge_plus_doc,
  key_mode=freeze, ENABLE_BETA=1, BETA_FIT_SCOPE=selected, ridge_lambda=1e-4, ridge_scale=spectral, delta_weight=1e-2,
  AM_EXECUTION_MODE=per_document, PHASE1_CACHE=.../cache_last.pt, SYNTH=MT parquet. NO BG_STATS (USE_IDF=0).
- Command: VAR=val ... bash examples/qasper2/scripts/train_continual_am_sparse.sh, ENABLE_BETA=1, USE_IDF=0.
- Metrics reported: qa_forgetting_loss / mt_acquisition_loss (mean CE) + solve_s + phase2_e2e_s + gradient_steps:0.
- Expected: β improves MT-acquisition (lower MT loss) vs EXP-001; watch QA doesn't regress; still backprop-free.
- Actual: **FAILED (crash), HYP-T2 UNANSWERED.** β WAS engaged (max|β_logweight| 66.6, mean 4.23, ~19% nonzero,
  all 36 layers) but crashed on document 3 with linalg.cholesky not-positive-definite (am/core.py:225 _ridge_lstsq).
- Interpretation: our β path fits LARGE unclamped NNLS log-weights (~66.6); added to attention logits pre-softmax
  they collapse the softmax to near-one-hot → value-solve design matrix rank-deficient → XtX indefinite → cholesky
  fails (both primary + raised-floor fallback; λ=1e-4 too small to rescue). The AM paper CLAMPS β to [-3,3] for
  exactly this stability reason (EXP-002); our impl doesn't. Docs 0-2 succeeded before the crash.
- Confounders considered: single variable (EXP-001 same config w/ β off ran clean) → attributable to β.
- Status: failed
- Follow-up: (1) EXP-005b = β at RIDGE_LAMBDA=0 (robust lstsq/gels branch) — does λ=0 alone avoid the crash?
  (2) if λ=0 completes but is degenerate, or still crashes → EDIT: clamp β to [-3,3] in the NNLS fit path, then retest.
- Artifacts: outputs/<run-dir>/ , research_loop/results/EXP-005/result.json
