# Report — rebuilding the research loop from sweeping to designing

**Date:** 2026-07-29 · **Scope:** the *method*, not the science (science: `2026-07-29-am-investigation-synthesis.md`)
**Spec:** `research_loop/{PLAN_AM_MUST_WIN,DESIGN,ORCHESTRATOR,WORKERS,GLOSSARY}.md`

---
## 1. What was changed, and why

The prior loop searched the **configuration space**: pick knobs, sweep, take the best, stop when the
knobs run out. It terminated by declaring closed-form AM "acquisition-capped at MT ~2.54, immovable by
gating / support / target / ridge," and accepted a sparse-gradient winner instead.

That framing has a specific failure mode: **it cannot distinguish "this lever does nothing" from "this
lever is not connected."** A sweep over a disconnected knob returns a clean null, and a clean null reads
as knowledge. Three of the prior loop's four pillars turned out to be exactly that.

The rebuild replaced config search with **cause search**:

> **MEASURE the cause → READ how the field fixes that cause → BUILD it → TEST one variable →
> VERIFY adversarially → re-measure.**

Two gates enforce it. **No mechanism may be built for a cause with no measured number.** **No sweep
without a stated discriminand** — you must write down which hypothesis the sweep separates and what each
outcome implies, or it is drift. And the stop conditions were narrowed to two: a verified PASS, or a
human stop. "Out of knobs" triggers an escalation ladder instead of an ending.

## 2. Architecture

**One orchestrator, N workers, five roles.** The orchestrator plans and decides but never runs GPU work;
workers do one task, write a structured bundle, and stop — they never choose what comes next.

| role | GPU | purpose |
|---|---|---|
| **W1 MEASURE** | yes | instrument the model; **oracle ceilings** that bound whole families in one run |
| **W2 SCOUT** | no | import mechanisms from outside the repo; every entry must predict *our* signature |
| **W3 BUILD** | no | implement one mechanism as an opt-in flag, default-off, bit-identical when off |
| **W4 TEST** | yes | one variable, named baseline, both eval splits, wandb-logged |
| **W5 VERIFY** | yes | try to **kill** the result — repro, seed-vary, ablate, control |

**State lives on disk** because each cycle may start from a compacted context: a live `bottleneck_board`
(the causal picture), a `literature_ledger`, a `mechanism_registry`, per-experiment bundles, raw
diagnostics, and `results.csv`.

**Volume produced:** 50 result bundles, 64 recorded results, 51 raw diagnostic dumps, 28 literature
entries, 11 registered mechanisms, 39 commits, and a 1168-line causal board.

## 3. What the design produced that a sweep could not

**Six defects, none of which a parameter sweep can surface.**

| defect | consequence |
|---|---|
| `rope_theta` hard-coded **10000** vs the model's **5e6** | live in the teacher-target path of **every AM run this project ever did** |
| `target_mode` never reached the write path | EXP-008 was three replicas of EXP-001; a "null result" that was a wiring bug |
| `lstsq` default `gels` driver returns NaN **without raising** | three failed β experiments blamed on "numerics" |
| **seed variation was impossible** (`pydrantic.main` never reached in `per_document`) | every number predating MECH-007 was one draw, including the prior loop's "confirmed" win |
| the **±0.15 noise band was never measured** — it is ±0.049 paired | 7 of 23 load-bearing deltas adjudicated on the wrong side |
| `mass_on_S` computed in-code but **never recorded** | the one quantity that bounds any value-only write, unlogged for nine experiments |

**Oracles instead of sweeps.** Asking "what is the best this family could do?" bounds it in one run.
The write-ceiling oracle (write the *teacher's own* KV) capped value-only writing at MT 2.381 — a sweep
would have wandered the value-solve's knobs indefinitely without ever learning that.

**Dose-response instead of point estimates.** The gating hypothesis closed not with three arms that
happened to lose, but with a monotone curve: tightening the safety constraint monotonically reduced MT
routing mass (0.2799 → 0.1661 → 0.1327 → 0.1000) and monotonically raised MT loss (2.2720 → 2.3288 →
2.3693 → 2.4388), Pearson **−0.9775** within the sweep. **The curve's optimum is the incumbent** — so
there is no untried setting where it might have worked.

**Mechanisms, not verdicts.** Every closure carries a reason: β can't buy selectivity because it is
*query-independent*; importance-gating can't win because the incumbent **maximises MT routing mass by
construction** and MT loss tracks that quantity.

## 4. Evidence the adversarial discipline paid

Workers repeatedly argued against their own results — the behaviour the W5 role and the
"report the objection" instruction were designed to induce:

- `DIAG-KEYSPACE` free-form optimisation produced per-head selectivity ratios up to **1e26**; the worker
  added a held-out split, watched them collapse to 0.885–1.12, and **published the closed-form bound its
  own optimiser could not beat** instead.
- `DIAG-CONTROLCURVE` refuted a headline result — then flagged that its own comparison used a control
  whose optimum was unlocated. `DIAG-NOISE` later reversed the refutation on the MT axis.
- `MECH-BUDGET` reported its own confound: the query cap I had pinned made `t128` under-determined —
  the exact flaw the brief cited as invalidating prior work.
- `MECH-BUDGET-B` corrected my *mechanism*: the failure was not the solver's min-norm branch (never
  taken when `DELTA_WEIGHT > 0`) but a rank-64 data block leaving **half of every head's support written
  carrying no new information**.
- `MECH-SEQUENTIAL` verified bit-identity by running the knob **on** with a no-op refresh, isolating the
  restructuring from the mechanism.
- Independent cross-checks landed to **4–6 significant figures**, and one control reproduced another
  worker's run to **16 digits** in a different process.

## 5. Where it cost, and what I got wrong

**Orchestrator errors — three, all mine:**
1. `ORACLE-WRITE-512` was misdesigned: at full support all 16 documents overwrite the whole cartridge,
   so only the last survives. The run bounds "full support as implemented," not bandwidth.
2. `MECH-BUDGET` inherited a pinned query cap from my base config, reproducing the very
   underdetermination I had warned that worker about.
3. I over-declared the space closed and **left a GPU idle ~3 hours** while reporting "nothing
   dispatchable" — my own charter calls that an orchestrator bug.

**Worker-runtime failure mode:** two workers ended their turns expecting a wake-up that never comes
(ending a turn *terminates* the task). Written instructions did not prevent it; only a resume with the
execution model spelled out did. **Fix for next time: state it in the dispatch template, not the prose.**

**Overhead:** duplicate/stale monitor notifications produced several no-op cycles. The frozen-snapshot
import pin was needed on **five** jobs — HEAD moved mid-run repeatedly — and the verification probe is a
false negative from the repo root (`cwd` is `sys.path[0]`).

## 6. Did it work?

**On the target: no.** Best gradient-free point **QA 1.9468 / MT 2.2681** — retention clear by 0.573,
acquisition **0.248 short**, at every seed.

**On the method: yes, and the negative is worth more than the prior loop's positive.** The prior result
rested on four pillars; two were artefacts and one was confounded. What replaced it is six families
closed with measured causes, one mechanism confirmed across seeds, and an instrument whose resolution is
now known rather than assumed. That is a result someone can build on — and, unlike a swept null, one
that says *why*.

**The honest caveat:** the loop found six defects in its own substrate. A prudent reader should assume
there are more, and treat every number here as resting on an instrument that has been audited once.
