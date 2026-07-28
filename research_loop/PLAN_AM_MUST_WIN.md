# PLAN — make closed-form AM the winning mechanism (2026-07-28)

**Objective:** find an **AM / closed-form** config (backprop-free, `gradient_steps = 0`) whose continual-learning
quality **matches the self-distillation cartridge on BOTH axes** — i.e. it *is* the winning mechanism, not the
sparse-gradient method. A few gradient steps are now only a **reference point**, no longer an allowed winner.

> Supersedes the prior loop's stop (which accepted the sparse-gradient winner). Full prior write-up:
> `notes/2026-07-26-sparse-grad-win.md`. This plan targets the *value-solve itself*, since every knob is exhausted.

---
## Current standing (mean-CE loss; QA = forgetting, MT = acquisition; lower better)

| technique | QA | MT | grad steps | note |
|---|---|---|---|---|
| ICL full-context (ceiling?) | 1.973 | 1.896 | 0 | different context source; **NOT the ceiling on QA** |
| Cartridge dense @4ep (BAR, best) | 2.372 | **1.873** | 256 | the quality bar to match |
| Cartridge dense @10ep | 2.699 | 2.214 | 624 | overfit |
| **Sparse-gradient winner** | **1.617** | **1.966** | 62 | the reference to beat, gradient-free |
| **AM closed-form top32 (best)** | **2.177** | **2.548** | 0 | our starting point |
| AM closed-form top64 | 2.252 | 2.543 | 0 | canonical |
| Phase-1 start (floor) | 2.239 | 3.783 | 0 | retention floor |

**AM's bottleneck = ACQUISITION.** Retention is competitive (QA 2.18–2.25, even out-retains dense cartridge 2.37).
But **MT 2.54 is the worst of every real method** — ~0.65 behind the ~1.9 acquisition cluster (ICL/dense/sparse-grad).
The one-shot linear value-solve captures Phase-1 structure but can't *fit new MT docs* like iterative gradient can.
Confirmed **immovable** by gating (HYP-G1), support/top_t (HYP-S1), target_mode (HYP-T1 null), ridge (HYP-R0 null);
β (mass-matching) is numerically broken (NaN). ⇒ the fix must change the **solve / target / keys**, not the knobs.

## Win condition (STOP-success)
An AM config, **0 gradient steps**, reaches **MT ≤ cartridge_MT + 0.15 AND QA ≤ cartridge_QA + 0.15** on the
**unified harness**, confirmed by a clean re-run. Concretely: **MT 2.54 → ≤ ~2.0** while **QA ≤ ~2.4**.
Stretch: match/beat the sparse-grad winner (QA 1.62 / MT 1.97) gradient-free.
**Honesty gate:** if unreachable after β + keys + target + support, deliver a *diagnosed* "closed-form is
acquisition-limited, here's the mechanism" — a real negative result.

---
## Phases

**Phase 0 — Unified re-baseline (also answers "ICL ≈ cartridge?").**
- Re-measure **ICL on `data/qasper/eval/qasper_eval_{QA,MT}.parquet`** (the same 78/69 examples as everything else),
  report the diagonals `icl_QA_raw|QA` / `icl_MT_raw|MT`. (Metric already matches eval_forgetting by design.)
- Optional parity check: eval the Phase-1 cartridge QA through BOTH `eval_forgetting.py` and the benchmark's
  cartridge path → confirm they agree (validates ICL-vs-cartridge comparability).
- Output: one ruler for ICL / cartridge / AM / sparse-grad.

**Phase 1 — Diagnose WHY the value-solve caps at MT 2.54 (no blind sweeps).**
- Instrument, on the new MT queries: (a) the solve's **reconstruction error vs its target**; (b) **attention mass on
  the newly-written slots** (with frozen keys, are the new values even attended to?); (c) confirm whether
  `target_mode` genuinely changes the solved target or is a **wiring no-op** (bit-identical result was suspicious).
- Decides the lever: target-limited vs routing-limited (frozen keys) vs capacity-limited.

**Phase 2 — Fix β / mass-matching (highest-potential; currently NaN).**
- EDIT: numerical guards in `refit_beta_nnls` (input clamp / regularize / NaN-guard) so β runs finite. β is AM's
  *explicit acquisition mechanism* (retained slots carry the missing attention mass). Test β on/off, keys frozen.

**Phase 3 — Open the two structural levers Phase 1 points at.**
- **Keys:** `KEY_MODE ∈ {freeze, highest_attention, omp}` → acquisition/forgetting **trade curve** (moving keys
  routes attention to new content = acquisition, at a retention cost — settled prior: keys can collapse QA).
- **Target/reference:** if target-limited, build an **acquisition-weighted target** (up-weight new-doc queries / use
  the doc's own outputs); sweep `MAX_REF_EXAMPLES_PER_DOC`, per_document vs decoupled, on-policy re-extraction.

**Phase 4 — Synthesis + honest verdict.**
- Best gradient-free AM point on the quality×cost Pareto vs cartridge / unified-ICL / sparse-grad. Clears the win
  condition → confirm + STOP. Else → diagnosed-limit synthesis.

## Mechanics
Orchestrator + executors; ≤2 training GPUs; one variable per experiment; adversarial verify anything that flips
direction; state in `research_loop/state/*`. Reframe `NORTH_STAR.md` to "closed-form AM must be the winner" if we
wire this into the live loop.

## Risk (honest)
May not succeed. Keys-unfreezing trades away AM's retention strength (its current advantage); β may be insufficient
even once fixed; a one-shot linear projection is inherently weaker than iterative fitting for *learning new* content.
Most likely landing: "β + a smarter target closes ~half the acquisition gap gradient-free; the rest needs keys
(costing retention) or a few gradient steps," with a crisp diagnosis of the cap.
