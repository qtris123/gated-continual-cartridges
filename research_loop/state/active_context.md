# ACTIVE CONTEXT — resume-from-this-file-alone

> Orchestrator: read this first every cycle; rewrite the top block every cycle.
> This is the single source of truth for "where are we and what's next."

## HEADLINE
- **Status (2026-07-28): LOOP RE-ARMED under a NEW MISSION and a NEW METHOD.** The human rejected the
  knob-tuning framing. The loop is now a **diagnosis engine**: measure the cause → read the literature
  for a fix → build it → test one variable → verify adversarially → re-measure.
  Read `research_loop/PLAN_AM_MUST_WIN.md` (mission) and `DESIGN.md` (architecture) before acting.
- **Objective:** a **gradient-free** (`gradient_steps = 0`) closed-form AM update that matches the
  dense cartridge on **both** axes. The sparse-gradient result is now only a reference point.
- **WIN = QA ≤ 2.52 AND MT ≤ 2.02** (dense@4ep bar 2.3721 / 1.8725, +0.15 each), **verified** by a W5
  worker: fresh process + changed seed + floor control + mechanism-off ablation.
- **The whole deficit is ACQUISITION:** best gradient-free point (AM top32) is QA 2.1766 (already
  better than the bar) / **MT 2.5484 (−0.53 to go)**.
- **Stop conditions are only two:** verified PASS, or a human stop. "Out of knobs" is not an ending —
  it triggers the escalation ladder (MISSION §6).
- **The board is the work:** `state/bottleneck_board.md` holds six candidate causes (B-WIRE, B-OBJ,
  B-ROUTE, B-TARGET, B-CAP, B-SOLVE), each with a signature to measure and an oracle that brackets it.
  **No mechanism may be built for a cause with no measured number.**

## NEXT ACTIONS (first cycle of the new loop)
The board is at its initial state — every entry is stage **A MEASURE**. Open with a batch that uses
both GPUs plus GPU-free workers:
1. **B-WIRE (no GPU, cheapest, do first):** read `cartridges/am/finetune.py` `target_mode` handling
   (~L108-190) and the `per_document` path. EXP-008 found all three modes **bit-identical to 15
   digits** — three different targets cannot give one solution. If it is a bug: fix, re-run EXP-008,
   and **retract HYP-T1's "NULL"** (several conclusions rest on it).
2. **B-ROUTE write-ceiling oracle (GPU-1) — the single most informative run in the mission:** write the
   *teacher's own* KV for the new doc into the selected slots, eval both splits. MT still ~2.5 ⇒ no
   value-only frozen-key write can ever win, and the answer must involve keys.
3. **B-OBJ (GPU-0):** read `am/mean_mse` against realized ΔCE; then the MSE→0 oracle (unbounded
   support, `RIDGE_LAMBDA=0`). MSE→0 with flat MT confirms objective mismatch.
4. **SCOUT (no GPU):** `AM.pdf` first — what does the paper's target actually consist of, does our
   `per_document` path implement it faithfully, and what exactly does β do? Then the B-ROUTE family
   (delta rule / fast weights) and the null-space editing family (AlphaEdit, MEMIT). Write LIT-XXX
   entries with **predictions about our signatures**.
5. **D0 unified ICL re-ruler (GPU, when free):** REF-ICL was measured on a *different* harness
   (`qasper_loss_benchmark.evaluate_loss_chunked`). Re-measure with `EVAL_MODE=icl` in
   `eval_forgetting.py` on `qasper_eval_{QA,MT}.parquet` so the ceiling sits on the same ruler.
   Informational — it gates nothing.

Also queued (do not start before the board says so): a `SEED` env knob for the confirmation standard
(RUNBOOK §9c); β guards *inside* `refit_beta_nnls` (only if the board says mass-matching matters).

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

## IN-FLIGHT
_(nothing dispatched — the loop was re-armed 2026-07-28 and has not run a cycle under the new spec)_

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
