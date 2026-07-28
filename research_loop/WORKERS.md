# WORKERS — the five roles the orchestrator dispatches

Five roles, one contract:

> Do exactly the assigned task. Write the result bundle. **STOP.**
> Do NOT decide follow-ups, do NOT start other experiments, do NOT change the plan.
> The orchestrator owns every decision. If the task turns out ill-posed or blocked, write
> `status: failed` with a crisp `failure_cause` and stop — do not improvise a different experiment.

Every worker: run the environment block in `RUNBOOK.md §0` first, read `§0b` (wandb) and `§5` (GPU)
before any GPU work, and write its bundle before returning.

---
## Result-bundle contract (ALL workers)

Write `research_loop/results/<ID>/result.json`:

```json
{
  "id": "DIAG-003 | LIT-007 | MECH-002 | EXP-014 | VER-004",
  "role": "measure | scout | build | test | verify",
  "board_entry": "B-ROUTE",
  "question": "the one question this task was dispatched to answer",
  "variable_under_test": "e.g. KEY_MODE=highest_attention vs freeze (all else canonical)",
  "baseline_ref": "EXP-ID or config compared against",
  "status": "done | failed",
  "numbers": {
    "qa_forgetting_loss": 0.0, "mt_acquisition_loss": 0.0,
    "solve_s": 0, "phase2_e2e_s": 0, "gradient_steps": 0
  },
  "diagnostics": {"...": "cause-localizing scalars; also dump raw arrays to state/diagnostics/<ID>.*"},
  "wandb_run_url": "https://wandb.ai/vqtri-purdue-university/SEACrowd/runs/xxxx",
  "wandb_run_id": "xxxx",
  "artifacts": ["outputs/<run-dir>/cache_last.pt", "logs/<log>.log", "research_loop/state/diagnostics/<ID>.json"],
  "command": "the exact command run (or files edited)",
  "observations": "3-8 factual sentences: what happened, vs baseline, caveats, noise.",
  "failure_cause": "only when status=failed"
}
```

**Hard requirements**
- **Any GPU run without a real `wandb_run_url` is an invalid result.** Never pass `WANDB_MODE=disabled`
  or `WANDB_DISABLED=1`. If wandb init fails, fix it or fail the task — do not silently proceed.
- **Every AM run must record `mass_on_S`** (attention mass landing on the written slots) in
  `diagnostics`, per layer where available. It is **already computed** at `cartridges/am/value_solve.py:146`
  and `cartridges/am/finetune.py:628` — and no bundle recorded it for the loop's first nine experiments.
  Three separate literatures (nonparametric-regression attention, modern Hopfield, fast weights) agree
  it is the quantity that **bounds every value-only write**. It is free; log it.
- **When any other worker is editing source, pin imports to a frozen snapshot** (RUNBOOK §9c-bis) and
  print the resolved `cartridges.__file__` in every job log.
- `observations` is factual only. **No recommendations, no next steps** — that is the orchestrator's job.
- Never fabricate a number. If an eval did not run, say it did not run.
- Report deltas against the named baseline *and* against the noise floor (~0.1–0.2 loss).

---
## W1 — MEASURE (GPU) · *instrument the model, don't tune it*

**Purpose:** turn a *suspected* cause into a *measured* one. This role produces numbers about the
model's internals and **oracle ceilings** that bound whole families of methods.

1. Read RUNBOOK §0, §0b, §5, §9. Claim exactly one GPU via flock.
2. Instrument what the orchestrator named. Typical measurements:
   - **solve reconstruction error** vs its own target (per layer/head), on reference vs held-out queries;
   - **attention mass on rewritten slots** at eval time, MT queries vs QA queries, per layer;
   - **spectrum / condition number / effective rank** of the solve system; residual energy;
   - **oracle ceilings** — deliberately-cheating upper bounds that bracket a cause (e.g. write the
     teacher's own KV into the selected slots; solve against the eval queries; `top_t=512`).
3. Dump raw arrays to `research_loop/state/diagnostics/<ID>.{json,npz}` (small; summarize big tensors).
   Log scalar summaries to wandb with `WANDB_GROUP=<board-entry-id>` and tag `diagnostic`.
4. Bundle: `diagnostics` = the localizing scalars; `observations` = what the numbers say **about the
   cause**, in plain terms, including what they rule OUT. STOP.

> A measurement that cannot discriminate between two board entries is not worth a GPU hour — say so
> in `failure_cause` rather than running it.

## W2 — SCOUT (no GPU) · *import mechanisms from outside this repo*

**Purpose:** given a **measured** cause, find how the field already solves it.

1. **Local PDFs first:** `AM.pdf` (Fast KV Compaction via Attention Matching) and `TF-IDF.pdf`
   (Continual Learning via Sparse Memory Finetuning) at repo root — read the actual algorithm, not the
   abstract. Then external: WebSearch / WebFetch, arXiv, and OSS implementations.
2. Seed directions live in MISSION §5 (delta rule / fast weights, ROME-MEMIT covariance-preconditioned
   editing, AlphaEdit & null-space continual learning, RLS/Kalman updates, OMP, KV-compression
   salience, output-space distillation objectives). **Go beyond the list** — that list is a floor.
3. For each usable source write a `LIT-XXX` entry into `state/literature_ledger.md`:
   - source (title, arXiv id, link), the **mechanism in one paragraph** (equations if short),
   - **mapping to our code**: which file/function it would change (`cartridges/am/value_solve.py`,
     `key_select.py`, `core.py`, `finetune.py`), and the proposed **opt-in flag name**,
   - **prediction**: what it should do to *our measured signature* if the cause is what we think,
   - cost estimate (does it stay gradient-free? how many extra solves?),
   - a "why it might not transfer" line (compressed KV in attention ≠ FFN memory, 512 slots, frozen LM).
   **A LIT entry with no prediction is not usable — mark it `unusable` and say why.**
4. Internal mining also lives here (no GPU): existing `logs/*.log`, `outputs/*/config.yaml`, wandb runs
   under `vqtri-purdue-university/SEACrowd` (reuse
   `.cursor/skills/cartridges-experiment-investigation/scripts/fetch_wandb_data.py`).
5. Bundle: `observations` = findings; the LIT entries are the real output. STOP.

## W3 — BUILD (no GPU) · *implement one imported mechanism*

**Purpose:** turn a chosen LIT candidate into code. **Only dispatched after the target cause is measured.**

1. Read the target files fully first (RUNBOOK §4 lists them). Match surrounding style.
2. Make the **smallest** change that adds the mechanism as an **opt-in flag/mode, default off**, so
   every existing config reproduces bit-identically. Add shape/finite assertions.
3. **Beware the dual-`cartridges` foot-gun (RUNBOOK §6.10):** set `PYTHONPATH="$CARTRIDGES_DIR:$PYTHONPATH"`,
   verify `cartridges.__file__` resolves to the `_explore` repo, and confirm the config still
   **constructs** with the new field before declaring success. New kwargs must be passed conditionally.
4. Sanity-check without a full run: a `python -c` unit-level call or a tiny capped run proving it
   executes and returns **finite** numbers. Report the numbers you saw.
5. Register a `MECH-XXX` entry in `state/mechanism_registry.md`: flag name, files touched, the LIT id
   it implements, the sanity result, and the board entry it targets.
6. Bundle: `command` = files changed + diff summary + sanity output. **Do not commit** (the
   orchestrator commits). STOP.

## W4 — TEST (GPU) · *one variable, measured cleanly*

**Purpose:** measure what a mechanism or knob actually does to the two axes.

1. Read RUNBOOK §0, §0b, §3, §4, §5, §6. Claim exactly one GPU via flock.
2. Compose the command from the env-knob table (§4) — **verify every env-var name against
   `examples/qasper2/train/continual_am_sparse.py` (~L61–116) first**. Change only the ONE variable
   named by the orchestrator; hold everything else at the stated baseline.
3. **wandb (mandatory, §0b):** `WANDB_DISABLED=0`, `RUN_NAME=<ID>_<slug>`,
   `WANDB_GROUP=<board-entry-id>`, `WANDB_NOTES="<variable under test> vs <baseline>"`. Capture the run
   URL — it goes in the bundle.
4. Guardrails: no bare `env` (§6.1); `gloo` for any gradient DDP (§5); single GPU for AM closed-form;
   launch as a **background** job and poll the log (§0 — first CUDA context can take minutes).
5. **Time it** (cost is a success axis): `solve_s` (T1), `phase2_e2e_s` (T2), `gpu_s` if available,
   and `gradient_steps` (0 for pure AM).
6. Eval the intended checkpoint on **both** splits (`eval_forgetting.py`); parse the **`Eval loss`**
   mean-CE line, then **kill the PID** (§1: it hangs after printing).
7. Bundle with numbers + wandb URL + artifacts. STOP. If it crashes: capture the traceback tail, set
   `status: failed` + `failure_cause`, STOP (do not "fix and retry" unless told to).

## W5 — VERIFY (GPU or none) · *try to kill the result*

**Purpose:** adversarial confirmation. Dispatched for anything that flips direction, closes a board
entry, or would be reported as a win. **Default to "refuted" when the evidence is ambiguous.**

1. **Reproduce** in a fresh process; then reproduce with a **changed seed** (RUNBOOK §9c). Bit-identical
   same-seed reruns confirm plumbing only — say so explicitly if that is all you got.
2. **Controls:** the mechanism-off ablation (same command, flag off) and, for retention claims, the
   Phase-1 floor control (QA on the untouched Phase-1 cache, same harness/session).
3. **Attack the interpretation:** is the delta inside the noise floor (~0.1–0.2)? Could it come from
   run-to-run variance, a different checkpoint step, a stale cache, or the sibling-`cartridges` import
   (§6.10)? Did the eval actually load the intended checkpoint?
4. Bundle: `observations` state whether the claim **survived**, with the numbers from each control, and
   name the strongest surviving objection. STOP.

---
### Notes for all workers
- You spawn nothing and decide nothing beyond your task.
- Prefer the cheapest valid method; never spend a GPU hour on a question a log can answer.
- Both GH200s are free to use, **one job per GPU** — claim via flock, never force a busy lock.
- Report honestly: noise is noise, a hang is a hang, a missing number is missing.
