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
- Evidence against: IDF may protect Phase-1 knowledge by steering away from load-bearing slots.
- Related experiments: EXP (Lever-1 HYP-G1), DIAG-G0.
- Prediction: at top_t=64, attention_mass gives lower MT loss at equal-or-lower QA loss vs tfidf.
- Next decisive test: slot_selection ∈ {tfidf, attention_mass}, all else fixed at canonical.
- Notes: This is the user's stated core problem ("select slots with insufficient attention mass").

<!-- more hypotheses appended by the orchestrator -->
