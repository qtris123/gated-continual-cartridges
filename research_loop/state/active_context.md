# ACTIVE CONTEXT — resume-from-this-file-alone

> Orchestrator: read this first every cycle; rewrite the top block every cycle.
> This is the single source of truth for "where are we and what's next."

## HEADLINE — updated 2026-07-28 late (cycle 1 complete, cycle 2 in flight)
- **The loop is a diagnosis engine** (`PLAN_AM_MUST_WIN.md`, `DESIGN.md`): measure the cause → import a
  fix from the literature → build → single-variable test → adversarially verify → re-measure.
  Stop conditions are only a **verified PASS** or a **human stop**.
- **WIN = `gradient_steps = 0` with QA ≤ 2.52 AND MT ≤ 2.02** (dense@4ep bar 2.3721 / 1.8725, +0.15).
- 🏁 **THE VALUE SIDE IS CLOSED.** Every value-side lever has been measured and none reaches MT ≤ 2.02:
  perfect teacher values **2.381**; 4.2× bandwidth via β **2.812 (worse)**; 256× reference queries
  **~2.55 (nothing)**; better fit **15.95 (catastrophic)**; correct rotary base **−0.019 (noise)**; more
  support **worse**; target mode **was a wiring no-op**.
  **Mechanistic reason (MECH-BETA):** β is *query-independent*, so 4.23× more mass on the written slots
  moved the MT/QA selectivity ratio only **1.043 → 1.051**. **Bandwidth is not the constraint;
  selectivity is, and no query-independent operator can supply it.**
- ⭐ **BEST GRADIENT-FREE POINT IS NOT THE ONE WE HAVE BEEN REPORTING.** The canonical run's own
  per-document curve bottoms at **k=12: QA 2.0422 / MT 2.4352**, then *degrades* to QA 2.1772 / MT 2.5524
  by k=16. **QA already clears its budget with 0.48 to spare; MT is 0.42 short.**
- **Four prior conclusions retracted or corrected this cycle:** HYP-T1 (`target_mode` no-op → **wiring
  bug**); HYP-S1 (support → **confounded** by a hard-coded 64-query cap); "β is a numerical rabbit hole"
  (→ a `gels`-driver silent NaN, now **running clean**); and the orchestrator's own "16 documents are
  worth one" (→ **refuted**, they accumulate; capacity saturates at ≈12).
- 🔴 **One confirmed bug, CE-neutral but internals-large:** `rope_theta` was hard-coded 10000 against the
  model's **5e6** in every AM entry point. ΔMT −0.019 (noise), but the target became **8.5× more
  fittable**, the write 5.5× gentler, and the routing collapse fully repaired. Every conclusion argued
  *from the internals* rested on a corrupted target (blast radius listed on the board).
- **THE ONLY AXIS LEFT IS KEYS**, and it has never been tested here (every row in `results.csv` is
  `KEY_MODE=freeze`; "keys collapse QA" is pre-loop Llama-era folklore). Two hazards that would have made
  the first key experiment fail for the wrong reason are now fixed (H4) or being fixed (H2, the missing
  RoPE counter-rotation on installed document keys).

## BUDGET
- **GPUs are free to use** (2× GH200, no GPU-hour cap). Keep both busy; an idle GPU with an open
  board entry is an orchestrator bug.
- Runs until a **verified PASS** or a **human stop**. Human: write `STOP` here to halt the loop.
- **wandb is mandatory** for every GPU run (RUNBOOK §0b). No run URL ⇒ invalid result, re-run it.

## STANDINGS (all on the `eval_forgetting.py` ruler; mean CE = ln ppl, lower better)
| point | QA | MT | grad steps | note |
|---|---|---|---|---|
| ICL full-context | 1.9734 | 1.8960 | 0 | ⚠️ different harness + context source → re-ruler (D0) |
| **dense @4ep — THE BAR** | **2.3721** | **1.8725** | 256 | best dense operating point |
| dense @10ep | 2.6991 | 2.2137 | 624 | overfit |
| sparse-gradient (62 steps) | 1.6169 | 1.9664 | 62 | reference point only; 30 steps ≈ same (1.5993 / 1.9840) |
| **AM top32 — best gradient-free** | **2.1766** | **2.5484** | 0 | the line to move |
| AM top64 (canonical) | 2.2521 | 2.5426 | 0 | e2e 217s |
| Phase-1 start | 2.2388 | 3.7826 | 0 | retention floor / untrained MT |

## IN-FLIGHT (cycle 2)
- **MECH-KEYS** (GPU, code-editing) → B-ROUTE key-side. **The first key experiment in this project's
  history.** Fixes hazard H2 (installed document keys get no RoPE counter-rotation; `phase1.py::
  _rope_reposition` already implements it) behind opt-in `AM_KEY_REPOSITION`, then runs 4 arms:
  freeze control (must reproduce QA 2.15320 / MT 2.52961), `highest_attention` without and with
  reposition, and `omp`. **Target metric is the MT/QA mass ratio, not loss** — β proved bandwidth alone
  is worthless, so only selectivity above ~1.05 counts. Bundle: `results/MECH-KEYS/`.
- **DIAG-CONTENT** (GPU, env-only, pinned) → B-OVERWRITE / B-ROUTE. **A control on whether the write
  stores content at all:** writes the *QA-topic* corpus and evaluates **MT**, where any gain is
  content-free by construction. Motivated by two anomalies — docs 1–3 hold 18.8% of the eval questions
  but deliver 56.7% of the MT gain (Spearman 0.51), and QA sits below its own floor at every k despite
  the QA eval papers being **entirely disjoint** from the Phase-2 documents. Bundle: `results/DIAG-CONTENT/`.

## QUEUED (next, in priority order) — updated 2026-07-29
0. **MECH-SEQUENTIAL** ← *the last mechanism inside scope.* The two escape hatches DIAG-KEYSPACE
   explicitly did **not** bound are "mechanisms that change the query distribution" and non-fixed-slot
   formulations (out of scope). The first has a concrete, gradient-free instance already identified:
   **SCOUT-AM's divergence #3** — the AM paper performs **on-policy, layer-sequential re-extraction** of
   reference queries, and we do not. We solve every layer against queries collected from **one** forward
   pass on the *pre-write* cartridge, so layers 1…35 are fitted against an activation distribution our
   own writes destroy (`continual.py:172-182`). Gradient descent gets this for free — it is exactly the
   cross-layer effect a per-layer closed-form solve misses, and it is the most plausible remaining
   explanation for the 62-step sparse-gradient reference reaching MT 1.966 where every closed-form
   variant stalls at 2.33–2.55.
   **Build:** after writing layer *l*, re-extract the reference queries for layers > *l* from the
   *updated* cache (opt-in flag, default off; bit-identical when off). Cost: one extra forward pass per
   layer-group, still `gradient_steps = 0`. **Compose with MECH-005** (`KEY_MODE=highest_attention` +
   `AM_KEY_REPOSITION=1`, θ=5e6), which is the current best point.
   **Prediction:** if the cascade is what closed-form is missing, MT should move materially below 2.33;
   if it does not, the gradient/closed-form gap is not cross-layer and the mission has a clean negative.
1. **VERIFY-BEST** — MECH-005's point (QA 2.0349 / MT 2.3305) is the mission's best and is **one seed**
   with ΔMT at the top edge of the noise band. Partly covered by DIAG-KEYCURVE's Part 2; if that returns
   without genuine seed variation (no `SEED` knob exists — see RUNBOOK §9c), a small BUILD adds one.
2. **COMPOSE-K** — the value-only curve bottoms at k=12 and degrades 0.117 by k=16. If DIAG-KEYCURVE
   shows the keys arm has the same shape, its own minimum is the number to quote, not its endpoint.
3. **MECH-METRIC** (LIT-009/010) — `C₀` from the old routing vectors' second moment instead of `w·I`.
   Now a **retention** mechanism only (DIAG-ROUTING's ρ→0 killed its acquisition story); its value is
   converting QA slack — currently **0.49** — into support.

## STANDINGS (all on the `eval_forgetting.py` ruler; mean CE = ln ppl, lower better)
| point | QA | MT | grad steps | note |
|---|---|---|---|---|
| ICL full-context | 1.9734 | 1.8960 | 0 | ⚠️ different harness + context source → re-ruler (D0) |
| **dense @4ep — THE BAR** | **2.3721** | **1.8725** | 256 | best dense operating point |
| dense @10ep | 2.6991 | 2.2137 | 624 | overfit |
| sparse-gradient (62 steps) | 1.6169 | 1.9664 | 62 | reference point only; 30 steps ≈ same (1.5993 / 1.9840) |
| **AM top32 — best gradient-free** | **2.1766** | **2.5484** | 0 | the line to move |
| AM top64 (canonical) | 2.2521 | 2.5426 | 0 | e2e 217s |
| Phase-1 start | 2.2388 | 3.7826 | 0 | retention floor / untrained MT |

## IN-FLIGHT (cycle 2)
- **MECH-KEYS** (GPU, code-editing) → B-ROUTE key-side. **The first key experiment in this project's
  history.** Fixes hazard H2 (installed document keys get no RoPE counter-rotation; `phase1.py::
  _rope_reposition` already implements it) behind opt-in `AM_KEY_REPOSITION`, then runs 4 arms:
  freeze control (must reproduce QA 2.15320 / MT 2.52961), `highest_attention` without and with
  reposition, and `omp`. **Target metric is the MT/QA mass ratio, not loss** — β proved bandwidth alone
  is worthless, so only selectivity above ~1.05 counts. Bundle: `results/MECH-KEYS/`.
- **DIAG-CONTENT** (GPU, env-only, pinned) → B-OVERWRITE / B-ROUTE. **A control on whether the write
  stores content at all:** writes the *QA-topic* corpus and evaluates **MT**, where any gain is
  content-free by construction. Motivated by two anomalies — docs 1–3 hold 18.8% of the eval questions
  but deliver 56.7% of the MT gain (Spearman 0.51), and QA sits below its own floor at every k despite
  the QA eval papers being **entirely disjoint** from the Phase-2 documents. Bundle: `results/DIAG-CONTENT/`.

## QUEUED (next, in priority order)
1. **MECH-DISJOINT** → B-OVERWRITE. Disjoint per-document allocation (16 docs × 32 slots = 512 = exactly
   the cartridge), with **explicit exclusion** rather than score-based separation, because the documents
   barely disagree (Spearman 0.958 between their slot scores; 87.4% overlap with a document-independent
   top-32). Predicted to push the k≈12 saturation point out and remove the k=12→16 tail regression.
2. **VERIFY-K12** → the mission's confirmation standard. `k=12` (QA 2.0422 / MT 2.4352) dominates the
   reported end state on both axes but is **one seed, and the tail regression is inside the noise band**.
   Needs a seed-varied re-run before it is quoted as the operating point.
3. **MECH-METRIC** (LIT-009/010) → replace `C₀ = w·I` with the old routing vectors' second moment from a
   single QA forward pass. Now a **retention** mechanism only (DIAG-ROUTING's ρ→0 killed its acquisition
   story), so its value is converting QA slack into support.
4. **DIAG-KEYSPACE** (SCOUT-KEYS' cheapest discriminator) → `ρ_key = tr(P⊥_r Q₀^MT)/tr(Q₀^MT)` from the
   128×128 per-head query second moments; ~86 s, no solve, no eval. Decides whether key-side selectivity
   is geometrically available before more is built on it.

## SETTLED FACTS (do not redo)
- ✅ Phase-1 cache provenance verified (EXP-000-verify) — it IS the QA Phase-1 self-distilled cartridge,
  staged at `outputs/phase1_selfdistill_qwen512/cache_last.pt`; its bundled `config.yaml` is mislabelled
  (ignore it). QA 2.2388 / MT 3.7826 on this harness.
- ✅ `bg_stats.pt` collected over OUR cache (SETUP-BG1, 292MB, per_layer, finite IDF). Never reuse the
  July-10 bg_stats — different cache.
- ✅ Dense bar established at both operating points (EXP-000 / REFCART-EVAL); @4ep is the bar.
- ✅ Knob-level explanations eliminated: gating (HYP-G1 — no-IDF wins both axes), support (HYP-S1 — MT
  flat in `top_t`), ridge (HYP-R0 — wash), `target_mode` (HYP-T1 — *bit-identical, see B-WIRE*).
- ❌ β / `ENABLE_BETA=1` is broken: NaN produced **inside** `refit_beta_nnls` (EXP-005/005b/006). The
  clamp edit was reverted. Needs guards inside the fit, not at its boundary.
- ⚠️ Untested, despite sounding settled: **every run so far used `KEY_MODE=freeze`.** "Keys collapse
  QA" is a pre-loop, Llama-era claim (RUNBOOK §7) — re-measure it, don't inherit it.
- ⚠️ Foot-gun that has bitten twice: the **dual-`cartridges` import** (RUNBOOK §6.10). Set
  `PYTHONPATH="$CARTRIDGES_DIR:$PYTHONPATH"`, verify `cartridges.__file__`, pass new kwargs conditionally.
- ⚠️ `eval_forgetting.py` prints `Eval loss` and then **hangs** — parse the line, kill the PID (§1).
- ⚠️ Eval splits are small and **fixed in size** (§9d): sub-0.1–0.2 deltas are noise, and no knob buys
  that down.

## DECISIONS LOCKED (by the human, do not revisit)
- Signal = perplexity / mean-CE eval loss only (no inference-server task accuracy).
- Model = Qwen3-4B; QA→MT 2-stage scope (SA / N-stage is future work).
- **Gradient-free is the requirement**, not a preference: a winner must have `gradient_steps = 0`.
- Two success axes: training efficiency (T1/T2/T3) AND continual-learning quality — judged as a Pareto trade.
- The loop **diagnoses and imports**; it does not tune. Sweeps are legal only as named discriminators.
- Gating remains the novelty budget, now including the *write rule itself* (target, keys, projection,
  number of closed-form rounds).
- GPUs free to use; every GPU run logged to wandb; investigation records are a deliverable.
- Full autonomy on branch `trivo-explore-research-work` only; may commit; **never** touch other branches.
- Task domains: QA = Question Answering (P1), MT = Machine Translation (P2), SA = Sentiment Analysis (future P3).

## PRIOR LOOP (superseded, kept for provenance)
The 2026-07-26 loop stopped after accepting a **sparse-gradient** winner (62 steps, QA 1.6169 /
MT 1.9664, reproduced bit-identically; Phase-1 floor control 2.2388 confirmed the sub-floor QA is a
real positive-backward-transfer effect, not an eval artifact). Full write-up:
`notes/2026-07-26-sparse-grad-win.md`. That result is now a **reference point**, not the goal — the
mission is to reach comparable quality with **zero** gradient steps.
