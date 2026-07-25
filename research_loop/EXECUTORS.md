# EXECUTORS — role templates the orchestrator embeds when dispatching

Three archetypes. **Every** executor obeys the same contract:
> Do exactly the assigned task. Report results in the required bundle. **STOP.**
> Do NOT decide follow-ups, do NOT start other experiments, do NOT change the plan.
> The orchestrator owns all decisions.

Every executor first runs the environment block in `research_loop/RUNBOOK.md §0` and reads the
relevant RUNBOOK sections. Every executor writes its result bundle before returning.

## Result-bundle contract (ALL executors)
Write `research_loop/results/<EXP-ID>/result.json`:
```json
{
  "exp_id": "EXP-XXX",
  "type": "train | research | edit",
  "variable_under_test": "e.g. slot_selection=attention_mass vs tfidf",
  "baseline_ref": "EXP-ID or config this is compared to",
  "status": "done | failed",
  "numbers": {"qa_forgetting_loss": 0.0, "mt_acquisition_loss": 0.0, "runtime_s": 0, "...": 0},
  "artifacts": ["outputs/<run-dir>/cache_last.pt", "logs/<log>.log"],
  "command": "the exact command run (or files edited)",
  "observations": "3-8 sentences: what happened, vs baseline, caveats/noise.",
  "failure_cause": "only if status=failed"
}
```
Also append the raw run log path. Keep prose in `observations`; keep it factual, no recommendations.

---
## A) TRAINING executor (uses GPU)
Task: run ONE experiment = ONE training/eval config, report the two eval-loss numbers.
1. Read RUNBOOK §0,§3,§4,§5,§6. **Claim exactly one GPU** via the flock mechanism
   (`/tmp/gpu_locks_$USER/`); if none free, wait/retry, do not force.
2. Compose the command from the env-knob table (§4) — but **confirm every env-var name against
   `examples/qasper2/train/continual_am_sparse.py` ~L73-116 first**. Change only the ONE variable the
   orchestrator named; hold all else at the stated baseline.
3. Guardrails: no bare `env` (§6.1); gloo for any gradient DDP (§5); single GPU for AM closed-form.
4. **Time it (efficiency is a first-class success axis — NORTH_STAR).** Record:
   - `solve_s` (T1) = the closed-form value-solve wall-clock (from the phase2 summary / your own timer),
   - `phase2_e2e_s` (T2) = ref-query collection + bg_stats (if collected this run) + solve + eval,
   - `gpu_s` if available, and `gradient_steps` (0 for pure AM).
5. Run to completion. Then **eval** the intended checkpoint (`eval_forgetting`) on BOTH splits.
   Parse the **`Eval loss` mean-CE** for QA and MT — NOT the `perplexity` field (§1).
6. Write the result bundle: numbers = {qa_forgetting_loss, mt_acquisition_loss, solve_s,
   phase2_e2e_s, gpu_s, gradient_steps}, plus artifacts. Report a 2-line summary + bundle path. STOP.
If the run crashes: capture the traceback tail, set status=failed + failure_cause, STOP (do not "fix and retry" unless the orchestrator told you to).

## B) RESEARCH executor (no GPU — external / literature / log-mining)
Task: answer ONE research question the orchestrator posed. Two flavors:
- **External:** WebSearch/WebFetch the AM paper (arXiv 2506.06266 & the Attention-Matching algorithm),
  closed-form model/KV editing (ROME/MEMIT-style least-squares value updates), sparse-routing / salience
  alternatives to TF-IDF (BM25, attention-mass gating, OMP), continual-learning regularizers. Extract
  concrete, testable ideas mapped to our knobs (§4).
- **Internal (log-mining / wandb):** pull existing `logs/*.log`, `outputs/*/config.yaml`, or wandb runs
  (`vqtri-purdue-university/SEACrowd`) to extract numbers WITHOUT spending GPU. Reuse
  `.cursor/skills/cartridges-experiment-investigation/scripts/fetch_wandb_data.py`.
3. Write the bundle: `observations` = findings; `numbers` = any extracted metrics; propose (in
   observations only, as candidate hypotheses — NOT decisions) how each idea maps to a single-variable
   experiment. STOP.

## C) EDIT executor (no GPU — code change; only when no knob exists)
Task: implement ONE mechanism the orchestrator specified (e.g. a new `slot_selection` mode, an
attention-mass floor on gating, a new teacher target). Confined to branch `trivo-explore-research-work`.
1. Read the target file(s) fully first (RUNBOOK §4 lists them). Match surrounding style.
2. Make the **smallest** change that adds the mechanism as an OPT-IN flag/mode (default off, so
   existing configs are unchanged). Add an assertion / shape check.
3. Do a **50-step / tiny sanity run** (or a `python -c` import + unit-level call) to prove it executes
   and produces finite numbers — do NOT run a full sweep (that's a training executor's job next cycle).
4. Write the bundle: `command` = files changed + the diff summary + the sanity-check result. Do NOT
   commit (the orchestrator commits). STOP.

---
### Notes for all executors
- You may spawn NOTHING and decide NOTHING beyond your task. If you discover the task is ill-posed or
  blocked, set status=failed with a crisp `failure_cause` and STOP — the orchestrator will re-plan.
- Prefer the cheapest valid method. Never fabricate numbers; if an eval didn't run, say so.
