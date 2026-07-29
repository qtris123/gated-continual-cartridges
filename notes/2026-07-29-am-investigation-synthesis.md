# Closed-form AM for continual cartridges — full investigation synthesis

**Date:** 2026-07-29 · **Status:** COMPLETE — every in-scope family closed and confirmed · **Branch:** `trivo-explore-research-work`
**Live state:** `research_loop/state/bottleneck_board.md` · **Terms & reproduction:** `research_loop/GLOSSARY.md`
**Supersedes:** `notes/2026-07-28-am-diagnosis-cycle1.md` (cycle 1 only) and `notes/2026-07-26-sparse-grad-win.md`

## The question and the answer

**Question:** can a **gradient-free** closed-form AM update match the dense self-distillation cartridge
on both axes — QA ≤ 2.52 **and** MT ≤ 2.02?

**Answer: no, not within this scope.** Best point **QA 1.9468 / MT 2.2681** (seed-mean).
Retention clears its budget by **0.573 at every seed**; acquisition falls **0.248 short at every seed**.

## What is confirmed

**One mechanism works and is confirmed to the mission's own standard** — fresh-process reproduction,
changed seed (×2), seed-varied mechanism-off ablation, and the Phase-1 floor control:

**Key installation with RoPE repositioning** (`KEY_MODE=highest_attention` + `AM_KEY_REPOSITION=1`,
MECH-005) buys **ΔMT −0.1545** (mean over seeds, each arm at its own per-seed argmin — the strictest
fair form), at **~1.9×** the control's solve cost. All three seeds clear the measured resolution by
2.8–3.6×. **The QA side is not confirmed** — consistently negative, consistently inside the band.

The "keys collapse QA" folklore is false here, and the RoPE counter-rotation is load-bearing: without it
the same mechanism reads 2.424 instead of 2.331 (unit check: repositioned logit error **4.8e-07** vs
**1.054** uncorrected; doing the rotation at the *wrong* θ gives 4.4–5.1, worse than not correcting).

## What is closed, and why

Five mechanism families, each with a measured cause rather than a failed sweep:

| family | verdict | the measurement that closed it |
|---|---|---|
| **values** | closed | a *perfect* write (teacher's own KV) reaches only MT 2.381; better fitting is **anti-correlated** with CE (18× lower MSE → +13 loss) |
| **bandwidth / β** | closed | β raised `mass_on_S` **4.23×** and made **both axes worse**; β is query-independent, so it moved selectivity only 1.043 → 1.051 |
| **keys / selectivity** | closed | ρ_key sits **at its held-out control floor** everywhere; QA/MT query separation is **2.5× smaller than sampling noise**, 0/288 heads otherwise; an *optimally placed* key buys 1.22 vs an incumbent 1.083 |
| **allocation / capacity** | closed | a solo write into **uncontested** slots is **+1.101 worse** on its own document than writing it among 16; 71–93% of a solo write's gain lands on *other* documents' questions |
| **query distribution** | closed | on-policy layer-sequential re-extraction (the paper's own procedure) moves ΔMT −0.021, and is *worse* composed; cross-layer drift is cos 0.9982 and largest at the **shallowest** layer |
| **gating / selection** | closed | see below |

### The gating result (the human's hypothesis, tested directly)
Attention mass is **not** importance — Spearman(`tf_mass`, `fisher`) = **0.579**, so the family was not
bounded a priori. But every information-theoretic selector **loses acquisition**, by 2.2–8.1× the
resolution, and the cause is close to a proof: across six arms **`log(MT routing mass)` predicts MT loss
at Pearson −0.877**. The incumbent ranker **maximises MT routing mass by construction**, so any
importance-aware modification captures less of it and necessarily loses. Redundancy-gating at **4× the
budget** still captures **less** MT mass than the incumbent at 1×. `fisher` gives the best retention ever
measured here (QA **1.8641**, 0.04% QA-Fisher exposure) and pays **+0.397 MT** — the anti-alignment
end to end. Protecting Phase-1 slots *works*; it simply buys the axis that was already 0.57 ahead.

## Why acquisition resists everything

The write is **not storing retrievable per-document content**. It produces a **cumulative, largely
document-agnostic adaptation** to the MT distribution. Evidence: documents with **zero MT content**
reproduce 28–73% of the MT gain (and full-context ICL shows **53.3%**, so this is a benchmark property,
not an AM defect); **writing a paper in makes the model worse at that paper**; the curve saturates at
k≈12 then degrades; sequence order alone is worth 0.340; 4.7% slot survival costs almost nothing.

## Methodological findings (these outlast the negative result)

1. 🔴 **A 500× rotary-base error** (`rope_theta` 10000 hard-coded vs the model's 5e6) sat in the
   teacher-target path of **every AM run this project ever did**. CE-neutral, but the target became
   **8.5× more fittable** once fixed — so every conclusion argued *from internals* had rested on a
   corrupted target.
2. 🔴 **`target_mode` never reached the write path.** EXP-008 was three replicas of EXP-001.
3. 🔴 **Seed variation was impossible** — `pydrantic.main` is never reached in `per_document` mode, so
   `seed=N` on argv was silently ignored. **Every number predating MECH-007 is one draw**, including the
   prior loop's "confirmed" win. Once possible, seed spread proved **small** (0.0094 MT at the operating
   point) — but that was luck, not diligence.
4. 🔴 **The ±0.15 noise band was ~3× too wide** for the *paired* comparisons everyone was making
   (measured: **±0.049 MT / ±0.054 QA**). Seven of 23 load-bearing deltas had been adjudicated on the
   wrong side, including one refutation that this reversed.
5. **`mass_on_S` was already computed in the code** and went unrecorded for nine experiments — the one
   number three literatures agree bounds any value-only write.
6. **Frozen-snapshot import pins are mandatory** when workers overlap: HEAD moved mid-run on five jobs,
   and unpinned imports on this box resolve to a *sibling* repo. The verification probe is a false
   negative from the repo root (cwd is `sys.path[0]`).

## Corrections made during the investigation

**Prior work:** HYP-T1 (wiring bug), HYP-S1 (confounded by a hard-coded query cap), "β is a numerical
rabbit hole" (a `gels`-driver silent NaN). HYP-R0 stands.
**My own:** B-CASCADE was the rope bug, not a mechanism; "16 documents are worth one" was refuted by the
loop's own curve; "keys beat frozen keys on both axes" was corrected twice; `ORACLE-WRITE-512` was
misdesigned (full support makes 16 documents mutually annihilate); and I pinned the query cap in
`MECH-BUDGET`, reproducing the exact underdetermination I had warned that worker about.

## Final confirmations (added 2026-07-29, after the synthesis was first written)

- **Support is genuinely not a lever, confound removed** (MECH-BUDGET-B). With determinacy proven per arm,
  the matched-n contrast t128 vs t64 gives ΔMT inside the resolution at all four k and **not even
  sign-consistent**. Three escapes closed: more queries, determined solve, and an n-scaled trust region.
- **The gating family closed by a controlled dose-response** (MECH-CONSTRAINED). Tightening the safety
  constraint monotonically cuts MT routing mass (0.2799 → 0.1661 → 0.1327 → 0.1000) and monotonically
  raises MT loss (2.2720 → 2.3288 → 2.3693 → 2.4388), Pearson **−0.9775**. **The curve's optimum is the
  incumbent**, so no untried setting remains.
- **The gating negative survives seed variation** (VERIFY-GATE). ΔMT +0.167 / +0.163 / +0.200 across
  offsets; **all 12 matched-k cells clear even the conservative composite**, and so does the adversarial
  pairing. Worth stating plainly: **this negative is better established than the project's own positive**,
  whose QA axis does not clear the composite.
- **The mechanism holds under three independent perturbations:** `log(MT routing mass) → MT loss` gives
  Pearson **−0.877** across *selectors*, **−0.9775** across *constraint strength*, **−0.986** across
  *seeds*. The incumbent maximises that quantity by construction, which is why nothing beats it.

## What remains

**In scope:** nothing untested. **Out of scope, and where a win most likely lives:** non-fixed-slot
formulations. `DIAG-KEYSPACE` explicitly did not bound those, and every in-scope route is now closed
with a measured cause.
