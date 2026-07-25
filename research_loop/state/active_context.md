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
- **Current best AM-sparse (MEASURED in-harness):** EXP-001 — QA **2.2521** / MT **2.5426** loss
  (Qwen, no-IDF/pure-TF, top64 per_layer, cartridge_plus_doc, freeze, ridge spectral), backprop-free,
  solve 181s / e2e 217s. Supersedes the old ambiguous-units 7.13/7.28 claim (top32 per_head, not re-measured).
- **Gap to target:** QA already at the 2.239 floor (no forgetting). MT 2.5426 vs REF-CART bar = _TBD (EXP-000
  pending)_ and vs REF-ICL ceiling = _TBD (EXP-000b pending)_. Cost side already strong (backprop-free, ~3.6 min).

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
- **EXP-000** (TRAIN, GPU0) — REF-CART dense self-distillation Phase-2 → quality BAR + train cost. STILL
  RUNNING: ~step 110/~620 (epoch ~2/10) at 18:20, ~11 min/epoch → ETA ~1.5–2h (this box slower than the
  wandb scf175an ~30min). The executor AGENT returned (parked) but the training process is alive (launcher
  3629349 holds gpu0 flock). ⚠️ MUST ensure its bundle gets written — see NEXT ACTIONS #0. Bundle: results/EXP-000/.
- ~~EXP-001~~ **DONE + INGESTED** (18:02) — AM-sparse no-IDF anchor: QA 2.2521 / MT 2.5426, e2e 217s / 0 grad.
- ~~EXP-002~~ **DONE + INGESTED** (18:20) — research. Key: (c) dense P2 MT=2.891 (ppl18, scf175an) — **EXP-001's
  2.5426 already beats it backprop-free**, pending EXP-000's QA-forgetting; (b) T3 = dense 1805s train + ~10-12ks
  synth vs AM 217s; (a) new levers RIDGE_LAMBDA=0 / ENABLE_BETA=1 / per-head budget (added to backlog); (d) bg_stats recipe.
- **SETUP-BG1** (TRAIN/EDIT, GPU1) — pipelined during EXP-000: build + run the standalone bg_stats collector over
  the Phase-1 QA parquet → outputs/phase1_selfdistill_qwen512/bg_stats.pt (per_layer). Unblocks EXP-003 (with-IDF). Bundle: results/SETUP-BG1/.

## NEXT ACTIONS (what the next cycle should do)
0. **⚠️ ENSURE EXP-000 COMPLETES + INGESTS.** Its executor agent is parked while training runs (~ETA 1.5–2h
   from 18:02). When the checkpoint saves (cache_last.pt / cache-step*.pt under
   outputs/2026-07-25-18-02-35-baseline_continual/…, saved BEFORE the final-barrier hang), if no
   results/EXP-000/result.json appears: either SendMessage the parked EXP-000 agent (id a4255d85247212da7) to
   eval BOTH splits + write the bundle, OR dispatch a small eval executor on the saved ckpt (eval_forgetting on
   QA+MT, parse `Eval loss`, kill PID). Then ingest → the quality BAR + T3 dense-train-cost anchor. Headline
   comparison: EXP-001 MT 2.5426 vs REF-CART MT (and vs the wandb dense 2.891).
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
