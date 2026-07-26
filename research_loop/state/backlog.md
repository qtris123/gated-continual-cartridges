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
- [x] **HYP-R0 (EXP-004): `RIDGE_LAMBDA=0` vs 1e-4 → WASH.** QA 2.2619 / MT 2.5569 vs EXP-001 2.2521/2.5426
      (+0.010/+0.014 ≪ noise). At λ=1e-4 the ridge is already negligible; removing it doesn't help. Keep canonical.
      (Distinct from HYP-R1's λ=2.0 Phase-1 RECON solve — not re-tested.)
- [ ] HYP-R1: `ridge_lambda` × `ridge_scale` (test λ≈2.0 spectral — Phase-1 sweet spot — in Phase-2).
- [ ] HYP-R2: `delta_weight` trust region sweep (0, 1e-2, 1e-1) — trades acquisition vs stability.
- [ ] HYP-R3: `enable_old_reference_guard` on/off (explicit old-query preservation block).

## Lever 2.5 — AM-FAITHFULNESS (from EXP-002 reading of AM.pdf; candidate single-var tests)
Baseline for all: AM-sparse canonical. Priority per AM paper's own ablations.
- [ ] HYP-T2 (ELEVATED by EXP-002): `ENABLE_BETA=1` vs unset — the per-token β mass-bias (NNLS) is AM's
      CORE mechanism to stop subsetting from underestimating future attention mass. AM claims dropping tokens
      w/o β systematically hurts. One var: ENABLE_BETA. (keys frozen; strong candidate right after gating.)
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
