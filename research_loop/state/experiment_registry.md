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
- Actual: _pending_
- Status: dispatched
- Follow-up: bg_stats recipe → EXP-003 (with-IDF canonical) next cycle.
- Artifacts: research_loop/results/EXP-002/result.json
