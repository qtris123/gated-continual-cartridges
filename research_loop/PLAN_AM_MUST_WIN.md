# MISSION — make closed-form AM the winning mechanism

**Rewritten 2026-07-28.** Supersedes both the prior loop's STOP (`notes/2026-07-26-sparse-grad-win.md`)
and the first draft of this plan, which was a knob-tuning phase list. This version is
**diagnosis-driven**: we do not search the config space, we search the **cause space**, and we import
mechanisms from outside this repo to attack the causes we measure.

---
## 1. Objective

Find an **AM / closed-form** continual update — **backprop-free, `gradient_steps = 0`** — whose
continual-learning quality matches the self-distillation cartridge on **both** axes, so that the
**closed-form solve itself is the winning mechanism**. A few sparse gradient steps remain a *costed
reference point on the Pareto plot*, never an acceptable answer.

## 2. Standing (mean-CE eval loss; QA = forgetting, MT = acquisition; lower is better)

| technique | QA | MT | grad steps | note |
|---|---|---|---|---|
| ICL full-context | 1.973 | 1.896 | 0 | different context source **and different eval harness** → ballpark; re-ruler in D0 |
| **Cartridge dense @4ep** | **2.372** | **1.873** | 256 | ← **THE BAR** (best dense operating point) |
| Cartridge dense @10ep | 2.699 | 2.214 | 624 | overfit; not the bar |
| Sparse-gradient (62 steps) | 1.617 | 1.966 | 62 | prior loop's winner; now only a reference point |
| **AM closed-form top32** | **2.177** | **2.548** | 0 | best gradient-free point = our starting line |
| AM closed-form top64 | 2.252 | 2.543 | 0 | canonical config |
| Phase-1 start | 2.239 | 3.783 | 0 | retention floor / untrained acquisition ceiling |

**The gap is entirely in ACQUISITION.** AM *out-retains* the dense cartridge (2.18 vs 2.37) but sits
~0.67 behind it on MT. Every knob-level explanation is eliminated: gating (HYP-G1), support/`top_t`
(HYP-S1), `target_mode` (HYP-T1 — bit-identical, therefore **suspicious**, see B-WIRE), ridge
(HYP-R0). β is numerically broken (NaN inside `refit_beta_nnls`).

⇒ **The remaining explanations are structural.** Structural causes are what this mission measures,
and structural fixes are what it imports.

## 3. Win condition

**PASS** = an AM config with **`gradient_steps = 0`** reaching, on the unified `eval_forgetting.py`
harness, against the **dense @4ep bar (QA 2.3721 / MT 1.8725)**:

> **QA ≤ 2.52  AND  MT ≤ 2.02**   (bar + 0.15 on each axis)

**Confirmation standard (required before any PASS is reported):** a W5 VERIFY worker reproduces it in
a fresh process **and** with a changed seed (RUNBOOK §9c — add a `SEED` knob if none exists), plus two
controls: the Phase-1 floor control (QA on the untouched Phase-1 cache) and the mechanism-off
ablation. The eval set is fixed and small (a handful of batches per split) and **cannot** be enlarged,
so margins under ~0.1–0.2 are labelled "within noise" regardless of how much we want the win.

**Stretch:** match or beat the sparse-gradient reference (QA 1.62 / MT 1.97) with zero gradients.

**Not a win:** parity bought with a giant reference bank, dozens of re-solves, or a runtime
approaching the dense cartridge's. Cost is an axis (NORTH_STAR) — report it on every result.

## 4. The bottleneck ladder — the cause space we are actually searching

Acquisition failure has these candidate causes. Each has a **signature** (what a diagnostic must
show) and an **oracle test** (an upper-bound run that brackets it). This ladder seeds
`state/bottleneck_board.md`; the orchestrator maintains it and adds causes as evidence demands.

| id | cause | signature to measure | oracle test (upper bound) |
|---|---|---|---|
| **B-OBJ** | **Objective mismatch** — the solve minimizes value/attention-space MSE; we are scored on token CE. A perfect solve can leave CE untouched. | achieved solve MSE vs realized ΔCE, per layer | drive MSE→~0 (unbounded support, λ=0) and read MT. Flat MT ⇒ confirmed |
| **B-ROUTE** | **Routing** — with **frozen keys**, eval-time MT queries may not attend to the rewritten slots at all; anything written is unreadable. | attention mass on rewritten slots at eval time, per layer, MT vs QA queries | **write-ceiling oracle**: replace the selected slots' values with the *teacher's own* KV for the new doc, then eval. If MT stays ~2.5, **no value-only frozen-key write can win** — the whole family is capped |
| **B-TARGET** | **Target/reference** — the reference queries the solve fits don't represent the eval distribution, so we fit the wrong thing well. | recon error on held-out MT queries vs on reference queries (generalization gap) | solve using the *eval* MT queries as the reference (deliberate cheat). Large MT drop ⇒ reference-limited |
| **B-CAP** | **Capacity** — the selected support cannot represent the new content. | residual energy / spectrum of the solve system after fitting; effective rank vs `top_t` | `top_t = 512` (all slots), ignore the QA cost, read MT alone |
| **B-SOLVE** | **Numerics** — ill-conditioning, ridge, or the NaN β path means we never reach our own optimum. | condition number / Gram spectrum of the query matrix; β finiteness | solve the *same* objective iteratively (LSQR/CG) and compare to the closed form |
| **B-WIRE** | **Wiring bug** — `target_mode` gave **bit-identical** results across all three modes. That is a no-op, not a null result. | read the path in `cartridges/am/finetune.py`; assert the solved target actually differs across modes | if it is a bug: fix it, re-run EXP-008, and **retract HYP-T1's "NULL"** in the ledger |

**Order of attack:** B-WIRE and B-OBJ first (cheap; either one can invalidate prior conclusions),
then B-ROUTE's write-ceiling oracle — **the single most informative run in this mission**, because it
bounds the entire method family in one shot — then B-TARGET / B-CAP / B-SOLVE as the oracles direct.

**Untested prior, flagged:** every experiment in `results.csv` ran `KEY_MODE=freeze`. The claim "keys
collapse QA" comes from a pre-loop session, so within this investigation the key-side levers
(`highest_attention`, `omp`, and imported key-write rules) are **unmeasured**, not settled.

## 5. The literature mandate (this is what makes it research, not tuning)

Once a cause is *measured*, the orchestrator dispatches **W2 SCOUT** workers to find how the field
already solves that specific failure; the fix is then imported, adapted, and cited. Local PDFs first
(`AM.pdf` = Fast KV Compaction via Attention Matching; `TF-IDF.pdf` = Continual Learning via Sparse
Memory Finetuning, both at repo root), then external search (WebSearch/WebFetch, arXiv, OSS code).

Seed directions per cause — SCOUT is expected to go **beyond** this list:

- **B-OBJ** → objectives that provably track end-loss: logit/KL-matched closed-form fits,
  Gauss-Newton / natural-gradient closed forms, layerwise least squares with output-space weighting.
- **B-ROUTE** → the **delta rule / fast-weight** literature (DeltaNet, delta-rule linear attention,
  Fast Weight Programmers): closed-form *associative writes* where the key is chosen so the write is
  retrievable; Hopfield/associative-memory capacity; closed-form key-side editing (our `KEY_MODE ∈
  {highest_attention, omp}` handle is untested).
- **B-TARGET** → on-policy / self-generated reference construction, test-time training,
  coverage-driven reference selection.
- **B-CAP / B-SOLVE** → closed-form model editing: **ROME / MEMIT** (covariance-preconditioned
  least-squares multi-edit), **AlphaEdit / null-space-projected editing**, Adam-NSCL-style null-space
  continual learning (write in the null space of old keys ⇒ acquisition *without* forgetting — exactly
  our two axes), recursive least squares / Kalman updates, OMP and matching pursuit.
- **KV-compression side** → H2O, SnapKV, PyramidKV, Scissorhands: what "which slot matters" means in a
  KV cache, and how they decide it.

Every source read becomes a `LIT-XXX` entry in `state/literature_ledger.md`: claim, mechanism in one
paragraph, **how it maps to our code** (file + function + proposed flag), and **what it predicts for
our measured signature**. A LIT entry with no prediction is not usable — send it back.

## 6. Escalation, not quitting

There is no "we ran out of knobs" ending. If every board entry is closed and the gap survives, the
loop **escalates** — in this order — and keeps running:

1. **Widen the cause space.** The board is wrong or incomplete: dispatch W1 MEASURE on the residual
   (what does a *winning* run do that ours doesn't? diff the sparse-gradient winner's internals
   against AM's — same slots, same eval, different update rule).
2. **Widen the mechanism space.** New family from the literature (§5), built and tested. Two
   independent families minimum per closed cause.
3. **Widen the definition of the write.** Keys, β, multiple alternating closed-form rounds
   (ALS/EM-style re-solves are still `gradient_steps = 0`), per-head allocation, null-space projection
   — anything that stays backprop-free is in scope.
4. **Report and continue.** Write the current mechanistic account into `active_context.md` +
   `notes/`, flag it for the human, and go back to step 1 with the new board.

The loop stops only on a **verified PASS** or a **human stop**. Honesty is a reporting requirement,
not an exit: never claim a win inside noise, never close a cause without a mechanism, always say what
is still unexplained.

## 7. Operating rules

- **GPUs are free to use.** Both GH200s should be busy whenever dispatchable GPU work exists; an idle
  GPU with an open board entry is an orchestrator bug.
- **Every GPU run is logged to wandb** per the naming/grouping contract in RUNBOOK §0b. No wandb URL
  in a bundle ⇒ the result is invalid and gets re-run.
- **One variable per experiment**, named baseline, both eval splits, T1/T2 timings recorded.
- **No mechanism is built before its target cause has a measured number** on the board.
- **No cause is closed without a mechanistic explanation** written into the board.
- Full autonomy on branch `trivo-explore-research-work` only; commit each coherent unit of work.
