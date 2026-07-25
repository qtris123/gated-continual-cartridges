# BACKLOG — prioritized, single-variable experiment queue

Orchestrator refines this each cycle. Each item isolates ONE variable vs a named baseline.
Priority order follows the mission: gating → teacher targets → support allocation → regularization → synthesis.
Do NOT re-test settled priors (RUNBOOK §7).

## Lever 0 — ANCHORING (do first)
- [ ] EXP-000 dense self-distillation Phase-2 baseline (local) — the number to match.
- [ ] EXP-001 reproduce current AM-sparse Phase-2 canonical config through the same eval harness.
- [ ] EXP-002 (research) wandb cross-check of dense Phase-2 + AM paper gating/target read.

## Lever 1 — GATING (the stated core problem: low-attention-mass slot selection)
Baseline for all: AM-sparse canonical (tfidf, use_idf=1, per_layer, top_t=64, freeze keys, ridge spectral).
- [ ] HYP-G1: `slot_selection=attention_mass` (pure TF, drop IDF) improves acquisition at fixed top_t
      by avoiding low-mass high-IDF slots. Compare vs tfidf. (one var: slot_selection)
- [ ] HYP-G2: soft IDF prior (`residual_budget` + `idf_prior_weight>0`) beats the hard multiplicative
      IDF gate. (one var: slot_selection=residual_budget vs tfidf, prior weight swept separately)
- [ ] HYP-G3: an **attention-mass floor** — never select a slot whose TF mass < τ — reduces wasted
      support. If no knob exists → EDIT executor adds an opt-in floor to the ranker, then TRAIN.
- [ ] DIAG-G0: instrument selected-slot attention-mass distribution (per layer) to quantify the
      "insufficient mass" problem directly. (log-only; cheap; informs G1–G3.)
- [ ] HYP-G4: granularity × top_t interaction under the best gater (small grid, per_layer vs global).

## Lever 2 — TEACHER TARGETS
- [ ] HYP-T1: `target_mode` cartridge_plus_doc vs self vs (others in enum) at fixed gating.
- [ ] HYP-T2: `enable_beta` (mass matching) on vs off with keys frozen.
- [ ] HYP-T3: key_mode freeze vs highest_attention vs omp — but note prior: touching keys hurts
      forgetting; frame as an acquisition-vs-forgetting trade curve, keep freeze as reference.

## Lever 3 — SUPPORT ALLOCATION
- [ ] HYP-S1: top_t ∈ {32,64,128} sweep under best gater — find the acquisition/forgetting knee.
- [ ] HYP-S2: `residual_budget` with `min_top_t_per_layer` floor vs uniform per-layer top_t.
- [ ] HYP-S3: `max_queries_per_head` / `max_ref_examples_per_doc` — does more reference support help?

## Lever 4 — REGULARIZATION
- [ ] HYP-R1: `ridge_lambda` × `ridge_scale` (test λ≈2.0 spectral — Phase-1 sweet spot — in Phase-2).
- [ ] HYP-R2: `delta_weight` trust region sweep (0, 1e-2, 1e-1) — trades acquisition vs stability.
- [ ] HYP-R3: `enable_old_reference_guard` on/off (explicit old-query preservation block).

## Lever 5 — SYNTHESIS
- [ ] Combine the winning setting of each lever; confirm on both eval splits with a clean re-run;
      run a small robustness check (2 seeds / a larger eval batch count if noise is a concern);
      write the final `notes/` entry and results table.
