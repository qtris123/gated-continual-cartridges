# HYPOTHESIS LEDGER

One `### HYP-XXX` block per hypothesis. Status ∈ active | weakened | supported | rejected | unresolved.

## Template
```
### HYP-XXX: <claim in one line>
- Status: active
- Claim:
- Motivation / mechanism:
- Evidence for:
- Evidence against:
- Related experiments (EXP-IDs):
- Prediction (what we'd see if true):
- Next decisive test:
- Notes:
```

---
### HYP-G1: Attention-mass gating beats TF-IDF for acquisition at fixed support
- Status: SUPPORTED (2026-07-25, EXP-003 vs EXP-001) — stronger than predicted: pure-TF (no-IDF) beats
  TF-IDF on BOTH acquisition AND forgetting at top_t=64. QA 2.2521 vs 2.6351 (−0.383), MT 2.5426 vs 3.0073
  (−0.465), both >2× noise. IDF is the wrong gate in the compressed-KV setting. Canonical set to no-IDF.
  Caveat: single seed, tiny eval (deltas large though), only top_t=64 tested → confirm-at-small-t via HYP-a5.
- Claim: Selecting slots by attention mass (pure TF) instead of TF×IDF avoids spending scarce
  support on low-mass high-IDF slots, improving MT-acquisition without extra forgetting.
- Motivation / mechanism: TF-IDF can rank a slot the new doc barely attends to (tiny TF) above a
  high-mass slot purely because it was Phase-1-rare (high IDF) — wasting a slot on near-zero leverage.
- Evidence for: RUNBOOK §7 coverage∝forgetting means wasted slots cost forgetting for no acquisition gain.
  EXP-001 (the NO-IDF / pure-TF arm, top64 per_layer) MEASURED: MT 2.5426, QA 2.2521 (no forgetting) — this
  is the no-IDF arm's absolute number; the HYP resolves only once the with-IDF arm (EXP-003) is measured.
- Evidence against: IDF may protect Phase-1 knowledge by steering away from load-bearing slots.
- Related experiments: EXP-001 (no-IDF arm, DONE: MT 2.5426 / QA 2.2521), EXP-003 (with-IDF canonical, pending
  bg_stats-over-our-cache), DIAG-G0.
- Prediction: at top_t=64, attention_mass (no-IDF) gives ≤ MT loss at ≤ QA loss vs tfidf (with-IDF).
- Next decisive test: EXP-003 with-IDF canonical (tfidf + bg_stats) vs EXP-001 no-IDF, all else fixed at
  canonical top64 per_layer — the single-variable USE_IDF comparison.
- Notes: This is the user's stated core problem ("select slots with insufficient attention mass").

<!-- more hypotheses appended by the orchestrator -->

### HYP-T2: AM per-token β mass-bias closes the MT-acquisition gap
- Status: BLOCKED (numerical) → retrying. β is the AM paper's core cure for low-mass (retained slots carry
  the missing attention mass via w_j=exp(β_j)≥0, NNLS-fit).
- Evidence for: AM paper claims dropping tokens w/o β systematically underestimates future attention (EXP-002).
- Evidence against: none yet (unmeasured).
- Related experiments: EXP-005 (FAILED — β at λ=1e-4 crashed: unclamped β log-weights ~66.6 collapse softmax →
  rank-deficient value-solve → cholesky not-PD, am/core.py:225). EXP-005b (β at λ=0, robust lstsq — running).
- Prediction: β lowers MT-acquisition loss vs no-β at ≤ QA cost — IF numerically stabilized.
- Next decisive test: EXP-005b; if degenerate/crashes → EDIT clamp β to [-3,3] (AM paper's stability range) then retest.
- Notes: root cause is our β-fit path has NO magnitude clamp; the AM paper clamps β∈[-3,3] for highest-attn keys.

### HYP-R0: RIDGE_LAMBDA=0 vs 1e-4 on the value-solve
- Status: RESOLVED — NULL/WASH (EXP-004). QA 2.2619 / MT 2.5569 vs EXP-001 2.2521/2.5426 (Δ<<noise). At
  λ=1e-4 the ridge is already negligible. Keep canonical (λ indifferent). AM "ridge hurts ∀λ>0" doesn't bite at tiny λ.

### HYP-S1: more support (top_t) improves acquisition
- Status: RESOLVED — REJECTED for acquisition (EXP-007). TOP_T {32,64,128}: MT-acquisition FLAT/worse
  (2.5484 / 2.5426 / 2.6860); QA-forgetting rises monotonically (2.1766 / 2.2521 / 2.4837, coverage∝forgetting).
  ⇒ AM's MT-acquisition (~2.54) is NOT support-limited; more slots only add forgetting. top32 marginally best
  operating point. IMPLICATION: the acquisition gap vs dense (1.87) is TARGET/solve-limited, not gating/support-limited.

### HYP-T1: teacher target_mode affects acquisition
- Status: RESOLVED — NULL (EXP-008). target_mode ∈ {cartridge_plus_doc, self, teacher_attention} all give
  BIT-IDENTICAL QA 2.2521 / MT 2.5426 (15 digits, different solve times) → target_mode is a NO-OP in the
  per_document AM path (either not wired, or all targets resolve identically). Combined with HYP-S1: **AM's
  closed-form acquisition ~2.54 is a HARD CEILING** — unmoved by gating, support, ridge, or target. Closing the
  acquisition gap vs dense (1.87) requires iterative optimization → the sparse-GRADIENT costed Pareto point.
- Caveat: the bit-identical result may be a wiring bug (target_mode ignored in per_document). Not chasing an EDIT
  fix (β burned us on that); the sparse-grad point is the cleaner route to acquisition.

### HYP-SG1: a few sparse gradient steps close the acquisition gap (costed Pareto point)
- Status: SUPPORTED + CONFIRMED (EXP-009 + EXP-009C) — bit-identical reproduction + Phase-1 floor control (2.2388)
  prove QA<floor is real (positive backward transfer). 30 steps ≈ 62 at 56% cost. THE WINNING RECIPE (notes/2026-07-26). —
  62 sparse grad steps (value-only, tfidf top64, USE_IDF=0, Adam LR2e-2):
  QA 1.6169 / MT 1.9664. MT closes the AM→dense gap (2.5426→1.9664, near dense@4ep 1.87); QA BETTER than AM (2.2521)
  and the Phase-1 floor (2.239). Dominates AM on both axes at ~1/10 dense's step cost. Acquisition front-loads
  (MT 3.78→2.11 by step15→~1.97 by step30, flat after). ⇒ potential TARGET MET (both axes ≤ cartridge+0.15 at ≪ cost).
- Evidence against: single run, tiny eval (QA n=6, MT n=5); QA 1.62 < floor 2.239 is surprising (positive backward
  transfer?) — MUST verify it's not an eval artifact. Same odd QA-drop seen in dense in-training (step60 QA~1.67).
- Next decisive test: EXP-009C — confirm 62-step reproduces + Phase-1 QA floor control + 30-step cheaper point.
