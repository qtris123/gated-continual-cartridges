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

## CYCLE 1 RESULT SO FAR (2026-07-28)
- ✅ **B-WIRE CLOSED — confirmed bug (DIAG-WIRE).** `target_mode` never reaches the per-document write
  path; the per-doc write is hard-wired to `cartridge_plus_doc` and *structurally cannot* express
  `teacher_attention`. **EXP-008 is three replicas of EXP-001; HYP-T1's "NULL" is RETRACTED.**
  ⇒ **B-TARGET is untested, not dead** — it is promoted to the front of the queue.
  Fix sketched, **not applied**: blocked until ORACLE-WRITE stops editing `finetune.py`
  (two editors on one file is how MECH-000 lost a batch).
- Second-order finding: EXP-008 ran `WANDB_DISABLED=1`, so it has no wandb run and no result bundle —
  exactly the failure the §0b wandb mandate now blocks. Two other arms of the loop may share this gap.
- ✅ **SCOUT-AM: WE HAVE BEEN RUNNING AM'S WEAKEST ABLATION.** The paper fits attention **output** *and*
  attention **mass** (mass = the routing weight of the block against all future tokens, App. A.2).
  Our canonical config runs **neither**: (1) `KEY_MODE=freeze` **silently disables β**
  (`finetune.py:256-261` `_should_fit_beta`), so β was never off *by choice*; (2)
  `max_queries_per_head=64` is **hard-coded with no env knob** (`finetune.py:81`) vs the paper's
  **16k–50k queries per KV-head** — so `top_t=64` solves an **exactly-determined 64×64** system and
  `top_t=128` an **underdetermined** one (min-norm branch), which **explains EXP-007's top128 MT
  regression and confounds HYP-S1's refutation** (support and query count were never separated).
- ✅ **β's NaN root cause is named, not mysterious (LIT-002):** `key_select.py:35` uses
  `torch.linalg.lstsq` with the default **`gels` driver**, which on CUDA returns NaN for rank-deficient
  input **without raising** (so `except RuntimeError` at `:36` never fires — this is why EXP-006's
  output clamp failed); plus no upper bound (β floors at −27.6 vs the paper's box [−3,3], `iters=2`);
  plus a residual-target `clamp_min(1e-12)` at `:215` **absent from the paper** that zeroes the whole
  NNLS target whenever the untouched slots over-supply mass.
- ⚠️ **AM is not designed for our regime at all.** Its `Ck` is always a subset of the keys *of the block
  being compacted*; there is **no procedure in the paper** for writing new content into keys fitted to
  different content, and §6 names our setting as future work. Its only multi-shot mode re-compacts the
  whole cache **rather than freezing** prior portions. ⇒ the mechanism we need must be **imported**
  (SCOUT-EDIT dispatched: null-space editing, delta-rule writes, MEMIT-style multi-edit) or designed
  here. That is the novelty budget, now backed by a measurement rather than an aspiration.

- ✅ **SCOUT-EDIT (LIT-009…LIT-018): our write is already MEMIT with the wrong metric.**
  `continual_am_sparse.py:96` sets `DELTA_WEIGHT=1e-2`, overriding the `0.0` dataclass default
  (`finetune.py:87`), so every AM row went through `guarded_sparse_am_value_update` =
  **MEMIT's `Δ = R K₁ᵀ(C₀ + K₁K₁ᵀ)⁻¹` with `C₀ = w·I`**. The editing literature's one statement is that
  `C₀` must be the **second moment of the old keys**, not the identity — and our "key" is the simplex
  routing vector `a_S(q) = alpha[:,S]` (`value_solve.py:86`), so `C₀` is a `t×t` routing Gram from **one
  QA forward pass** with hooks we already have. `DELTA_WEIGHT` has **never been swept** (HYP-R0 refuted
  `RIDGE_LAMBDA`, a different knob), so DIAG-OBJ-c (`DELTA_WEIGHT=0`, in flight) doubles as this
  family's free negative control.
- 🔥 **`mass_on_S` is already computed and has never been recorded** (`value_solve.py:146`,
  `finetune.py:628`). It is the quantity three independent literatures say bounds any value-only write.
  Now a standing bundle requirement (WORKERS.md).
- **Null-space value editing is identically zero at `top_t=64`** (`C₀` full rank ⇒ `P = 0`); the
  retention-preserving projectors only exist at large support. `top_t` is now entangled with three
  mechanisms (query count, min-norm rank, null-space existence) and cannot be swept as a scalar.
- **Projection/preconditioning are retention mechanisms** — they cannot move MT alone. Their role is to
  convert our **0.34 of QA slack** (2.177 vs the 2.52 budget) into support without EXP-007's QA cost.

## QUEUED FOR THE NEXT CYCLE (all blocked on ORACLE-WRITE releasing `finetune.py`)
Serialize these — one code-editing worker at a time (MECH-000 lost a batch to two editors on one file).
**Priority 0 (no code edit, GPU-only — dispatch the moment a GPU frees):**
- **DIAG-ROUTING** — the cheapest discriminator SCOUT-EDIT named: one forward-pass-only run yielding
  four numbers — eigenspectrum/effective rank of the QA routing Gram at `top_t=512`; `ρ` = MT routing
  energy in its approximate null space; `mass_on_S` + routing entropies for MT vs QA; and the QA/MT
  mean-routing overlap. **Jointly confirms or refutes LIT-011/012/013 and bounds every value-only
  mechanism, with no training.**
**Then, serialized code edits:**
1. **MECH-QUERIES** (likely the biggest single lever): expose `max_queries_per_head` as an env knob and
   run `n ∈ {256, 1024, 4096}` at fixed `top_t=32`. An exactly-determined 64×64 solve has **zero**
   generalisation headroom; the paper uses ~250× more rows. Then re-test `top_t` jointly at `n ≫ t`.
2. **MECH-BETA**: apply LIT-002's three fixes (`driver='gelsd'`, box `w ∈ [e⁻³, e³]` with `iters=2`,
   drop the residual-target clamp in favour of the paper's full-mass fit) **and** decouple
   `_should_fit_beta` from `key_mode` so β can run with frozen keys. β has never actually executed.
3. **MECH-TARGET**: the DIAG-WIRE fix (opt-in target branch at `finetune.py:477`, `target_accumulator`
   plumbing for `teacher_attention`, fail-loud guard in `continual.py`), then re-run EXP-008 for real.
4. **MECH-METRIC** (LIT-009/010): replace `C0 = w*I` in `guarded_sparse_am_value_update` with the
   second moment of the OLD routing vectors, collected in one QA forward pass. Small change, large
   literature. LIT-010 predicts a **sign flip on the EXP-007 top128 anomaly**, which is exactly the
   identity-metric min-norm interpolant at `core.py:198-204` — a sharp, falsifiable prediction.
5. **MECH-NULLKEY** (LIT-017, the most promising and the most work): null-space **key** placement — the
   only imported operator that attacks routing and retention together and escapes the nonparametric
   bound. Gated on beta actually working (B-SOLVE's three fixes).

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

## IN-FLIGHT (cycle 1 under the diagnosis spec — dispatched 2026-07-28)
All four items are stage **A MEASURE** / **B SEARCH**; nothing is being tuned.
- **DIAG-WIRE** (no GPU) → **B-WIRE**. Is `target_mode` a wiring no-op? Code read + CPU unit-level
  assertion that the three modes build different targets. Reports a fix sketch; applies nothing.
  Bundle: `results/DIAG-WIRE/result.json`.
- **SCOUT-AM** (no GPU) → **B-WIRE/B-TARGET/B-SOLVE/B-ROUTE**. `AM.pdf` in depth: what the target
  really is, whether our `per_document` path is faithful, the exact β procedure and its safeguards,
  and **whether AM is designed to acquire new content at all** (vs compact existing content) with
  frozen keys. Writes LIT entries. Bundle: `results/SCOUT-AM/result.json`.
- **DIAG-OBJ** (GPU, env-only, no code edits) → **B-OBJ**. (a) reproduce canonical top32
  (expect QA 2.1766 / MT 2.5484); (b) MSE→0 oracle (`RIDGE_LAMBDA=0`, `TOP_T=512`). Reads
  `am/mean_mse` against realized ΔCE. Bundle: `results/DIAG-OBJ/result.json`.
- **ORACLE-WRITE** (GPU, the only code-editing worker this cycle) → **B-ROUTE** ⭐. Opt-in
  `AM_ORACLE_WRITE` flag writing the *teacher's own* KV for the new doc into the selected slots, plus
  eval-time attention-mass on those slots (MT vs QA queries). MT ≈ 2.5 ⇒ no value-only frozen-key
  write can win. Bundle: `results/ORACLE-WRITE/result.json`.

**Concurrency note:** ORACLE-WRITE is the only worker allowed to edit source this cycle (avoids the
dual-`cartridges` foot-gun colliding with DIAG-OBJ's runs). Both GPUs are claimed via flock.

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
