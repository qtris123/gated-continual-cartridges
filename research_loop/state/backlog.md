# BACKLOG — prioritized, single-variable experiment queue

Orchestrator refines this each cycle. Each item isolates ONE variable vs a named baseline.
Priority order follows the mission: gating → teacher targets → support allocation → regularization → synthesis.
Do NOT re-test settled priors (RUNBOOK §7).

## Lever 0 — ANCHORING (do first) — establishes BOTH axes' reference lines
- [x] EXP-000-verify Phase-1 cache provenance — DONE: QA loss 2.239 / MT 3.783, it's the QA Phase-1
      cache; staged at outputs/phase1_selfdistill_qwen512/. (retention floor = QA loss 2.24)
- [ ] EXP-000 dense self-distillation Phase-2 baseline (REF-CART) from the staged Phase-1 cache — the
      quality BAR. Record its train cost (T2/T3) too, as the efficiency baseline to beat.
- [ ] EXP-000b ICL / full-context upper bound (REF-ICL) — measure ONCE on QA+MT (icl_eval.py /
      qasper_loss_benchmark fullctx). The ceiling.
- [ ] EXP-001 reproduce current AM-sparse Phase-2 canonical config through the same eval harness;
      record qa/mt loss AND solve_s/phase2_e2e_s (first efficiency data point).
- [ ] EXP-002 (research) pull the two cited papers (*Sparse Memory Finetuning* gating, *Fast KV
      Compaction via AM*) + cartridge paper synthesis/train-cost figures; wandb cross-check dense P2;
      mine any recorded synthesis time (do NOT re-synthesize).
- [ ] EFF-0 (research/log-mining) assemble the T3 cost comparison table (AM Phase-2 vs cartridge
      Phase-2 train time; synthesis cost estimated/qualitative per NORTH_STAR).

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

## Lever 2 — TEACHER TARGETS  ← NOW THE PRIORITY (acquisition is target-limited, per EXP-007)
- [~] HYP-T1 (EXP-008, RUNNING): `target_mode` cartridge_plus_doc vs self vs (others in enum) at fixed gating.
      This is THE key remaining closed-form acquisition lever — EXP-007 showed support doesn't move MT, so WHAT
      the value-solve matches (the target) is the bottleneck. Single var: TARGET_MODE.
- [ ] SPARSE-GRAD (costed Pareto point, NORTH_STAR-sanctioned): a FEW sparse gradient steps (SGD, value-only,
      top-t slots) on top of the AM init — does it close the MT-acquisition gap cheaply? Logged as a distinct,
      more-expensive Pareto point (not the default). Prefer SGD (EXP-002: SGD≫Adam for sparse). Try only if HYP-T1 stalls.
- [ ] HYP-T2: `enable_beta` (mass matching) on vs off with keys frozen.
- [ ] HYP-T3: key_mode freeze vs highest_attention vs omp — but note prior: touching keys hurts
      forgetting; frame as an acquisition-vs-forgetting trade curve, keep freeze as reference.

## Lever 3 — SUPPORT ALLOCATION
- [x] HYP-S1 (EXP-007): top_t {32,64,128} — REJECTED for acquisition. MT flat ~2.54 (worse at 128); QA-forgetting
      ∝ top_t. Acquisition is NOT support-limited → it's TARGET/solve-limited. top32 = marginally best point.
- [ ] HYP-S2: `residual_budget` with `min_top_t_per_layer` floor vs uniform per-layer top_t.
- [ ] HYP-S3: `max_queries_per_head` / `max_ref_examples_per_doc` — does more reference support help?

## Lever 4 — REGULARIZATION
- [x] **HYP-R0 (EXP-004): `RIDGE_LAMBDA=0` vs 1e-4 → WASH.** QA 2.2619 / MT 2.5569 vs EXP-001 2.2521/2.5426
      (+0.010/+0.014 ≪ noise). At λ=1e-4 the ridge is already negligible; removing it doesn't help. Keep canonical.
      (Distinct from HYP-R1's λ=2.0 Phase-1 RECON solve — not re-tested.)
- [ ] HYP-R1: `ridge_lambda` × `ridge_scale` (test λ≈2.0 spectral — Phase-1 sweet spot — in Phase-2).
- [ ] HYP-R2: `delta_weight` trust region sweep (0, 1e-2, 1e-1) — trades acquisition vs stability.
- [ ] HYP-R3: `enable_old_reference_guard` on/off (explicit old-query preservation block).

## Lever 2.5 — AM-FAITHFULNESS (from EXP-002 reading of AM.pdf; candidate single-var tests)
Baseline for all: AM-sparse canonical. Priority per AM paper's own ablations.
- [DEFERRED] HYP-T2: `ENABLE_BETA=1` (AM per-token β mass-bias, NNLS) — **β implementation is numerically broken.**
      Failed 3 ways: EXP-005 (λ=1e-4) cholesky not-PD (β~66.6); EXP-005b (λ=0) NaN in lstsq; EXP-006 (β-clamp added)
      `AssertionError max|beta|=NaN` — **the NNLS β-fit itself produces NaN**, so an output clamp can't help. A real
      fix needs numerical guards INSIDE refit_beta_nnls (input clamps / regularization / NaN guards) — a deep dive
      into someone else's AM numerics. DEFERRED as a rabbit hole; pursue tractable acquisition levers first (top_t,
      novel gaters). Revisit β only if support/gating don't close the MT gap. (β-clamp edit was reverted.)
- [ ] HYP-PH1: nonuniform PER-HEAD support budget (AM's #1 ablation: head sensitivity ~input-invariant →
      precomputed greedy budget). Map: GRANULARITY=per_head + a per-head TOP_T schedule (MIN_TOP_T_PER_LAYER /
      residual_budget). Likely needs an EDIT for a per-head budget schedule if no knob suffices.
- [ ] HYP-a5: at TOP_T=32, USE_IDF=1 vs 0 — AM/TF-IDF both say IDF matters MORE at small support. One var: USE_IDF at fixed small TOP_T.

## Lever 1.5 — NOVEL GATING DESIGN (the core novelty; open once existing-knob sweeps are informative)
Not just sweeping knobs — DESIGNING a gater specific to our setting (compressed embedding in attention).
Each is an EDIT executor (opt-in flag) + TRAIN executor to evaluate. Draw from the two cited papers
+ observed attention-mass/slot evidence. Keep gradient-free; keep it cheap (efficiency axis).
- [ ] GATE-N1: mass-aware gate — combine attention-mass (TF) with a *learned/derived* rarity signal
      that doesn't over-select near-zero-mass slots (fixes the core TF-IDF failure directly).
- [ ] GATE-N2: conflict-aware gate — select slots by old-vs-new attention conflict (protect
      load-bearing Phase-1 slots), inspired by the residual_budget idea but as a first-class gater.
- [ ] GATE-N3: coverage-budgeted gate — cap cumulative coverage (r≈0.91 with forgetting) explicitly,
      allocating support to maximize acquisition per unit coverage.
- [ ] (add more as the trajectory + literature suggest — this is where autonomy is expected)

## Lever 5 — SYNTHESIS (two-axis)
- [ ] Combine the winning gater + best target/support/reg; confirm on both eval splits with a clean
      re-run; place the final point on the **quality×cost Pareto plot** vs REF-CART and REF-ICL.
- [ ] Robustness: 2 seeds / larger eval-batch count if noise is a concern (evals are only a few batches).
- [ ] Write the final `notes/` entry + results table + Pareto figure.
