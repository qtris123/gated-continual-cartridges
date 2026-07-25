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
- Status: active
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
