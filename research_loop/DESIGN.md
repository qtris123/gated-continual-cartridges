# DESIGN — Orchestrated AM-sparse research loop

## Problem
Make **AM / TF-IDF sparse Phase-2 finetuning match cartridge self-distillation** on Qasper
2-stage (Qwen3-4B), without gradients, catastrophic forgetting, or excessive runtime. Explore
four levers: **gating, teacher targets, support allocation, regularization.** Metric = eval loss
(mean CE) on QA-forgetting + MT-acquisition splits (perplexity-only; no inference server).

## Architecture: 1 orchestrator + N executor threads
```
                       ┌───────────────────────────────────────────┐
   /loop (tmux) ─────► │  ORCHESTRATOR  (plans, reasons, DECIDES)    │
                       │  - reads state files (its only memory)      │
                       │  - decides next batch (1 var / experiment)  │
                       │  - dispatches executors, integrates reports │
                       │  - NEVER trains/evals itself                │
                       └───────┬───────────────┬───────────────┬─────┘
                    dispatch   │               │               │  (concurrent, 1 message)
                       ┌───────▼──────┐ ┌───────▼──────┐ ┌──────▼───────┐
                       │ TRAIN exec   │ │ RESEARCH exec│ │  EDIT exec   │
                       │ (GPU, ≤2)    │ │ (web/wandb/  │ │ (new mechanism│
                       │ run+eval one │ │  log-mining) │ │  only if no   │
                       │ config       │ │              │ │  knob exists) │
                       └──────┬───────┘ └──────┬───────┘ └──────┬───────┘
                              └── result.json ──┴──── report ────┘
                                        (executors STOP after reporting;
                                         they never decide what's next)
```
- **Only the orchestrator decides.** Executors do one task, write a result bundle, stop.
- **State lives on disk** (`research_loop/state/`), because each `/loop` cycle may start with a
  fresh/compacted context. The orchestrator reconstructs everything from files each cycle.
- **GPU cap = 2** (2× GH200). ≤2 training executors concurrently; research/edit are GPU-free and
  run alongside. Training executors claim a GPU via the repo's flock lockfiles.

## The cycle (see ORCHESTRATOR.md for the exact steps)
load state → reconcile in-flight results → check stop conditions → decide next batch
(1 variable each) → dispatch executors concurrently → collect & integrate → persist state +
journal + commit → schedule next wakeup (or STOP when target met / budget spent).

## Files
```
research_loop/
├── DESIGN.md          ← this file (human overview)
├── START_HERE.md      ← how to launch the loop in tmux + materials checklist
├── ORCHESTRATOR.md    ← the prompt /loop runs each cycle (the core spec)
├── EXECUTORS.md       ← the 3 executor role templates + result-bundle contract
├── RUNBOOK.md         ← exact commands, knobs (4 levers), guardrails, settled priors, numbers
├── state/
│   ├── active_context.md      ← resume-from-this-alone headline + next actions (rewritten each cycle)
│   ├── backlog.md             ← prioritized single-variable experiment queue
│   ├── experiment_registry.md ← EXP-XXX ledger
│   ├── hypothesis_ledger.md   ← HYP-XXX ledger
│   ├── observation_log.md     ← raw observations (append-only)
│   └── results.csv            ← the numbers
└── results/<EXP-ID>/result.json  ← executor output bundles (created at runtime)
```
Reuses existing repo conventions: `notes/` + `JOURNAL.md` for durable narrative,
`.cursor/rules/research-companion.mdc` discipline (Observation→…→Decision), the
`cartridges-experiment-investigation` skill's wandb scripts, and the result-bundle idea from
`finance-qasper-investigation/`.

## Design choices (locked with the human)
- Signal: eval-loss only (fast, local, matches repo numbers). Task-accuracy deferred (needs server).
- Model: Qwen3-4B. Baseline: re-run dense self-distillation locally + cross-check wandb.
- Autonomy: full, confined to branch `trivo-explore-research-work`; may commit; never other branches.
- Most experiments are **config/env sweeps → no code edits**. Edit executors are the exception,
  used only when a hypothesis needs a mechanism no knob provides (e.g. an attention-mass floor).

## What "done" means (TWO axes — see NORTH_STAR.md)
An AM-sparse config that matches the self-distillation cartridge on quality (QA-loss ≤ REF-CART_QA
+ 0.15 AND MT-loss ≤ REF-CART_MT + 0.15) **at materially lower training cost** — i.e. it Pareto-wins
on the (training-cost × forgetting/acquisition) plane vs the cartridge, reaching toward the ICL
ceiling — driven by a good (possibly novel) GATING mechanism. Or an honest diminishing-returns
synthesis with the Pareto frontier plotted. The direction anchor is `NORTH_STAR.md`, re-read each cycle.
