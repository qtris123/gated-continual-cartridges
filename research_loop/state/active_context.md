# ACTIVE CONTEXT — resume-from-this-file-alone

> Orchestrator: read this first every cycle; rewrite the top block every cycle.
> This is the single source of truth for "where are we and what's next."

## HEADLINE (update every cycle)
- **Status:** CYCLE 0 IN PROGRESS (anchoring) — dispatched 2026-07-25. **EXP-001 DONE + ingested** (first
  AM-sparse point). EXP-000 (REF-CART, train) + EXP-002 (research) still running. No new batch until both land.
- **Goal (TWO axes — see NORTH_STAR.md):** a fast, (near) training-free continual update for the
  KV-cartridge whose GATING gives cartridge-comparable forgetting/acquisition — winning on BOTH
  (1) training efficiency and (2) CL quality, vs cartridge (bar) and ICL (ceiling). Gating is the novelty.
- **Target (stop):** AM-sparse QA-loss ≤ REF-CART_QA + 0.15 AND MT-loss ≤ REF-CART_MT + 0.15,
  **at materially lower training cost than cartridge** (Pareto). Parity-but-expensive ≠ done.
- **Reference lines (to establish cycle 0):** REF-CART (dense self-distill P2) = _TBD_; REF-ICL
  (full-context ceiling) = _TBD_; cartridge P2 train-cost (efficiency baseline) = _TBD_.
- **CANONICAL AM-sparse = EXP-001 (no-IDF/pure-TF):** QA **2.2521** / MT **2.5426**, backprop-free, e2e 217s.
  HYP-G1 SUPPORTED — IDF gating (EXP-003) is WORSE on both axes (QA 2.6351 / MT 3.0073), so all further gating
  builds on the NO-IDF config. This IS the project's core thesis confirmed: IDF is the wrong gate in compressed-KV.
- **DENSE BAR ESTABLISHED (both dense points measured).** dense@10ep FINAL (REF-CART): QA 2.6991 / MT 2.2137 —
  but it OVERFIT (worse on BOTH than dense@4ep). **Best dense operating point = dense@4ep: QA 2.3721 / MT 1.8725.**
  AM canonical EXP-001: QA 2.2521 / MT 2.5426. ⇒ **Pareto: AM RETAINS QA better than even dense@4ep (2.25 < 2.37)
  and crushes dense@10ep on QA (2.25 vs 2.70); dense@4ep ACQUIRES MT better (1.87 vs 2.54, gap 0.67); AM ~8×
  cheaper, backprop-free.** THE CHALLENGE = close AM's MT-acquisition gap (target ~1.87) without losing QA/cost edge.
- **Gap to target:** MT-acquisition gap ≈ 0.67 (AM ~2.54 vs dense-best 1.87). **KEY (EXP-007): the acquisition
  gap is NOT support-limited** — MT is flat ~2.54 across top_t {32,64,128}; more support only adds forgetting.
  So gating/support help FORGETTING, not ACQUISITION; the acquisition bottleneck is the TARGET / closed-form
  value-solve. **Best AM point now = top32** (QA 2.1766 ~= floor / MT 2.5484, cheaper).
  **HYP-T1 NULL (EXP-008): target_mode {cartridge_plus_doc/self/teacher_attention} bit-identical → no-op.** So
  closed-form AM acquisition ~2.54 is a HARD CEILING (unmoved by gating, support, ridge, target). ⇒ Only remaining
  acquisition lever = **a few sparse GRADIENT steps (HYP-SG1, costed Pareto point)** — dispatched. β DEFERRED (broken).
- **REF-ICL ceiling (PARTIAL):** QA ICL = 1.9734 (ppl 7.2, < cartridge floor 2.239 ✓). MT ICL still computing (~20min/eval).

## BUDGET
- Soft budget: run unattended until target met or ~40 cycles / ~48 GPU-hours, whichever first.
  (Human may adjust in this file.)

## DONE PRE-LAUNCH (do not redo)
- ✅ Phase-1 cache downloaded + PROVENANCE VERIFIED (EXP-000-verify): it IS the QA Phase-1 self-distilled
  cartridge. QA loss **2.239** / MT loss **3.783** (this harness, loss=ln ppl). Staged at
  `outputs/phase1_selfdistill_qwen512/cache_last.pt`. Config.yaml is mislabelled — ignore it.
  → This is the Phase-2 START point (init) AND the retention floor (QA loss 2.24 to preserve).
- ✅ REF-CART command path SMOKE-TESTED (baseline_continual.py, Qwen, gloo, capped 2 steps): loads the
  Phase-1 cache, MT data, runs dense steps, evals, saves. Validated launch pattern in RUNBOOK §3(a).
  EXP-000 = just run it for real (EPOCHS=10, eval both splits, record loss + train cost). Note the
  "final barrier" hang — checkpoint saves before it.
- ✅ Cited-paper PDFs staged: `TF-IDF.pdf` (2510.15103v1), `AM.pdf` (2602.16284) at repo root (NORTH_STAR).

## IN-FLIGHT (experiments dispatched, awaiting result bundles)
- ~~EXP-000~~ **DONE + INGESTED** (dense REF-CART bar). dense@10ep QA 2.6991 / MT 2.2137 (OVERFIT); dense@4ep
  (best) QA 2.3721 / MT 1.8725. Bar established. results/EXP-000/ + REF-CART row in results.csv.
- ~~EXP-005 / EXP-005b~~ **BOTH FAILED** — β numerically pathological. EXP-005 (β@λ1e-4) cholesky not-PD;
  EXP-005b (β@λ0) → "NaNs in ridge lstsq solution". CONCLUSIVE: λ doesn't matter; the UNCLAMPED β log-weights
  (~66.6) are the problem. HYP-T2 requires the β-clamp fix. → **EXP-006** (implement opt-in β clamp [-3,3] +
  run β+clamp, RIDGE_LAMBDA=0, baseline EXP-004) dispatched.
- ~~EXP-006 / β~~ **FAILED → β DEFERRED.** β-clamp edit didn't help: EXP-006 died `AssertionError max|beta|=NaN` —
  the NNLS β-fit ITSELF produces NaN (not just huge values), so an output clamp can't fix it. β is a deep numerical
  rabbit hole (needs guards inside refit_beta_nnls) → DEFERRED (RUNBOOK §6.11, backlog). **β-clamp edit REVERTED.**
  Also exposed the dual-cartridges import foot-gun (RUNBOOK §6.10) — the unconditional beta_clamp_abs arg crashed
  ALL AM runs (incl. EXP-007's first attempt) when the sibling cartridges was imported.
- ~~EXP-007~~ **DONE + INGESTED** (2 executors, same numbers): top32 QA 2.1766/MT 2.5484, top128 QA 2.4837/MT 2.6860.
  HYP-S1 REJECTED for acquisition (MT flat; forgetting∝top_t). top32 = best AM point.
- **REF-ICL** (EVAL, GPU0) — CEILING. PARTIAL: QA ICL 1.9734 done; MT ICL still computing (long, ~20min/eval). Bundle: results/REF-ICL/.
- ~~EXP-008~~ **DONE + INGESTED** — HYP-T1 NULL (target_mode no-op; all 3 modes bit-identical QA 2.2521/MT 2.5426).
- **EXP-009 / HYP-SG1** (TRAIN, GPU1) — the sparse-GRADIENT costed Pareto point: a FEW sparse grad steps
  (continual_sparse.py, value-only, tfidf top-64) — does gradient close the acquisition gap AM's closed-form can't? Bundle: results/EXP-009/.
- NOTE ON CADENCE: /loop re-invocations arrive irregularly (observed a ~4h gap 02:5x→07:0x); ScheduleWakeup
  fallback isn't reliably firing. Make each invocation maximally productive; keep BOTH GPUs loaded to avoid long idle.
- ~~EXP-001~~ **DONE + INGESTED** (18:02) — AM-sparse no-IDF anchor: QA 2.2521 / MT 2.5426, e2e 217s / 0 grad.
- ~~EXP-002~~ **DONE + INGESTED** (18:20) — research. Key: (c) dense P2 MT=2.891 (ppl18, scf175an) — **EXP-001's
  2.5426 already beats it backprop-free**, pending EXP-000's QA-forgetting; (b) T3 = dense 1805s train + ~10-12ks
  synth vs AM 217s; (a) new levers RIDGE_LAMBDA=0 / ENABLE_BETA=1 / per-head budget (added to backlog); (d) bg_stats recipe.
- ~~SETUP-BG1~~ **DONE + INGESTED** (18:40) — bg_stats.pt (292MB, per_layer, IDF (36,511) finite) written +
  sanity-verified via the Phase-2 load path. Collector: examples/qasper2/train/collect_bg_stats.py. gpu1 freed.
- ~~EXP-003~~ **DONE + INGESTED** (18:50) — with-IDF: QA 2.6351 / MT 3.0073. **IDF HURTS both** (QA +0.383,
  MT +0.465 vs EXP-001, >2× noise). HYP-G1 SUPPORTED. ⇒ **canonical = no-IDF (EXP-001)**. (Bundle written by orchestrator.)
- ~~EXP-004~~ **DONE + INGESTED** (01:18) — HYP-R0 RIDGE_LAMBDA=0 = WASH (QA 2.2619 / MT 2.5569, ≈EXP-001).
  Keep canonical λ. gpu1 freed.
- ~~REFCART-EVAL~~ **DONE + INGESTED** (02:32) — dense@4ep: QA 2.3721 / MT 1.8725. Reframed the story (dense
  acquires MT better; AM cheaper + retains better). Full 10ep dense = authoritative bar (finishing now).
- **EXP-005** (TRAIN, GPU1) — HYP-T2: ENABLE_BETA=1 (AM per-token mass-bias) on the no-IDF canonical. Targets
  the MT-acquisition gap directly (β lets retained slots carry missing attention mass). Single var vs EXP-001. Bundle: results/EXP-005/.

## NEXT ACTIONS (what the next cycle should do)
0. **⚠️ SECURE THE DENSE BAR (EXP-000) — the full run is unreliable (restart-loop, see IN-FLIGHT).** Plan:
   (a) as soon as a GPU frees, dispatch a small EVAL executor to eval the attempt-1 dense checkpoint
   outputs/2026-07-25-18-02-35-baseline_continual/dfed5e8b-.../cache-step256.pt on BOTH splits (QA forgetting +
   MT acquisition; parse `Eval loss` mean-CE, kill the PID — eval hangs). Record as REF-CART@~4ep. That + the
   wandb 10-epoch dense MT=2.891 give a usable dense bar (dense forgetting only worsens with more epochs, so a
   4-epoch QA-forgetting is a conservative bar for "AM beats dense on QA"). (b) If the 01:07 run happens to
   complete a full 10 epochs uninterrupted, prefer that as the authoritative bar; if it restarts AGAIN, stop the
   EXP-000 agent (a4255d85247212da7) + reclaim gpu0, and use (a)+wandb as the dense bar. Headline comparison:
   EXP-001 (QA 2.2521 / MT 2.5426) vs dense (QA=step256 eval / MT 2.891).
1. **Ingest SETUP-BG1** when it lands → then TRAIN EXP-003 = with-IDF canonical (tfidf, USE_IDF=1, per_layer,
   top_t=64, BG_STATS_PATH=outputs/phase1_selfdistill_qwen512/bg_stats.pt) — the true Lever-1 baseline.
   Resolve HYP-G1: EXP-003 (with-IDF) vs EXP-001 (no-IDF), single var USE_IDF.
2. **Deferred anchor:** TRAIN EXP-000b (REF-ICL) full-context ICL eval on QA+MT → ceiling (measure ONCE), on
   whichever GPU frees first.
3. Then Lever-1 gating + the new EXP-002 levers (HYP-R0 RIDGE_LAMBDA=0, HYP-T2 ENABLE_BETA=1, HYP-PH1 per-head).
   Assemble the quality×cost Pareto plot (REF-CART, REF-ICL, PHASE1, AM points).
- NOTE: with-IDF needs bg_stats over OUR cache (SETUP-BG1); the July-10 bg_stats.pt are a DIFFERENT cache — never reuse.

## DECISIONS LOCKED (by the human, do not revisit)
- Signal = perplexity/eval-loss only (no inference-server task-accuracy).
- Model = Qwen3-4B; QA→MT 2-stage scope (SA/N-stage is future). Baseline = re-run locally + wandb cross-check.
- Phase-1 self-distilled cache = HF `qtris123/qwen_qasper-QA-task_8192_512_no-cartridge_10-epochs`
  (believed mislabelled QA-vs-MT; being re-verified by dual-split eval).
- **Two success axes:** training efficiency (3 tiers; synthesis relaxed — no re-synth) AND CL quality.
- **Fast is the requirement; gradient-free strongly preferred** — a few sparse gradient steps only as
  a distinct costed Pareto point.
- **Gating is the novelty — may design & implement NEW gaters autonomously on-branch** after
  existing-knob sweeps are informative.
- Measure **ICL ceiling once**. Full autonomy on branch `trivo-explore-research-work` only; may commit
  here; never touch other branches.
- Task domains: QA=Question Answering (P1), MT=Machine Translation (P2), SA=Sentiment Analysis (future P3).
