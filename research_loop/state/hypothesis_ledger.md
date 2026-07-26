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
