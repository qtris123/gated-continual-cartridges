# DESIGN — the orchestrator × worker research engine

## What this loop is (and what it is NOT)

This is a **diagnosis engine**, not a sweeper.

> **The loop's job is to explain, then fix.** Every cycle must either (a) *measure* something that
> localizes the bottleneck, (b) *import* a mechanism from outside this repo that targets a measured
> bottleneck, (c) *build* that mechanism, or (d) *test/refute* one. A cycle that only re-tunes an
> existing knob is a wasted cycle.

**Banned by construction:**
- ❌ "Sweep the knobs, pick the best, declare success/failure." Knob sweeps are *diagnostics with a
  bad signal-to-cost ratio*; they are allowed only when the orchestrator names which bottleneck
  hypothesis the sweep discriminates.
- ❌ "All existing knobs are exhausted → stop." Knob exhaustion is the **beginning** of the real work
  (measure → read the literature → build a new mechanism), never a stop condition.
- ❌ Declaring a hypothesis dead without a *mechanistic* reason ("it didn't help" is not a reason;
  "the solve already reaches 1e-4 MSE, so the target — not the solve — is the binding constraint" is).

## The scientific cycle the engine implements

```
        ┌──────────────────────────────────────────────────────────────────────┐
        │                        BOTTLENECK BOARD                              │
        │  every candidate cause, its measured signature, its status           │
        │  state/bottleneck_board.md  ← the loop's central artifact            │
        └───────────▲──────────────────────────────────────────┬───────────────┘
                    │                                          │
      (E) VERIFY ───┘                                          ▼  (A) MEASURE
      refute / repro / ablate                       instrument the real model:
            ▲                                       recon error, attention mass,
            │                                       rank/energy, oracle ceilings
            │                                                  │
     (D) TEST ◄──────── (C) BUILD ◄──────── (B) SEARCH ◄────────┘
     single-variable    implement it as    read papers/code OUTSIDE this repo
     run, wandb-logged  an opt-in flag     for how the field solves THIS cause
```

A bottleneck moves through **A→B→C→D→E**. The loop always knows which stage each open bottleneck is
in. When (E) confirms a fix, the board is re-measured and the *next* binding constraint surfaces —
that is the loop's real output, whether or not the win condition falls.

## Architecture: 1 orchestrator + N workers

```
                        ┌───────────────────────────────────────────────┐
   /loop (tmux) ──────► │  ORCHESTRATOR — plans, reasons, DECIDES       │
                        │  • state files are its ONLY memory            │
                        │  • owns the bottleneck board                  │
                        │  • dispatches workers, integrates bundles     │
                        │  • NEVER runs GPU work itself                 │
                        └──┬────┬────┬────┬────┬────────────────────────┘
        dispatch (parallel, one message)                                
     ┌─────────┘    │    │    │    └──────────┐
     ▼              ▼    ▼    ▼               ▼
┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐
│ W1 MEASURE│ │ W2 SCOUT │ │ W3 BUILD │ │ W4 TEST  │ │ W5 VERIFY│
│ (GPU)     │ │ (no GPU) │ │ (no GPU) │ │ (GPU)    │ │ (GPU)    │
│ instrument│ │ external │ │ implement│ │ run one  │ │ refute / │
│ + oracle  │ │ papers,  │ │ mechanism│ │ variable,│ │ reproduce│
│ ceilings  │ │ code, OSS│ │ opt-in   │ │ → wandb  │ │ / ablate │
└─────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘
      └────────────┴───results/<ID>/result.json───┴───────┘
                   (workers report and STOP; they never decide what's next)
```

- **Only the orchestrator decides.** A worker does one task, writes one bundle, stops.
- **State lives on disk.** Each `/loop` cycle may start from a fresh/compacted context; the
  orchestrator rebuilds its entire world from `research_loop/state/`.
- **GPUs are free to use** (2× GH200 144GB). Both should be busy whenever there is dispatchable GPU
  work — an idle GPU with an open board entry is an orchestrator bug. One job per GPU (flock).
- **Every GPU run lands in wandb**, cleanly named and grouped (RUNBOOK §0b). A run with no wandb URL
  in its bundle is an invalid result.

## Record-keeping (the investigation is a deliverable)

The point is not just to win; it is to be able to *show the reasoning*. Files, all under
`research_loop/`:

```
research_loop/
├── DESIGN.md              ← this file: architecture + the cycle
├── NORTH_STAR.md          ← direction / anti-drift anchor (re-read every cycle)
├── PLAN_AM_MUST_WIN.md    ← the MISSION: objective, win condition, bottleneck ladder
├── ORCHESTRATOR.md        ← the exact per-cycle spec /loop runs
├── WORKERS.md             ← the 5 worker role templates + bundle contract
├── RUNBOOK.md             ← commands, env knobs, wandb mandate, GPU policy, foot-guns
├── PROTOCOL.md            ← the Qasper 2-stage task definition
├── START_HERE.md          ← how a human launches / steers / stops the loop
├── state/
│   ├── active_context.md      ← resume-from-this-alone headline + next actions
│   ├── bottleneck_board.md    ← ★ candidate causes, measured signatures, stage A–E, status
│   ├── literature_ledger.md   ← ★ LIT-XXX: source → claim → mapping to our code → status
│   ├── mechanism_registry.md  ← ★ MECH-XXX: what we built, flag, files, sanity, verdict + WHY
│   ├── experiment_registry.md ← EXP-XXX ledger (one variable each)
│   ├── hypothesis_ledger.md   ← HYP-XXX ledger
│   ├── observation_log.md     ← append-only raw observations
│   ├── diagnostics/           ← ★ raw diagnostic dumps (DIAG-XXX.json/npz) + index
│   └── results.csv            ← the numbers
└── results/<ID>/result.json   ← worker output bundles (runtime)
```
★ = added by this rewrite. Durable narrative still goes to `notes/<date>-<slug>.md` +
a one-line row prepended to `notes/JOURNAL.md`; commits land on `trivo-explore-research-work`.

## Why five worker roles

| role | GPU | exists because |
|---|---|---|
| **W1 MEASURE** | yes | You cannot fix what you have not localized. Produces numbers *about the model's internals*, not just eval loss — including **oracle ceilings** that bound whole families of methods. |
| **W2 SCOUT** | no | The fix for a measured bottleneck usually already exists in the literature (delta-rule updates, null-space edits, covariance-preconditioned closed-form editing…). Reading is far cheaper than sweeping. |
| **W3 BUILD** | no | Imported mechanisms need code. Always opt-in flags, default-off, sanity-checked. |
| **W4 TEST** | yes | One variable, named baseline, wandb-logged, both eval splits. |
| **W5 VERIFY** | yes | Anything that flips direction or would be reported as a win must survive an adversarial attempt to refute it (repro w/ different seed, ablation, control). |

## Stop policy — escalate, don't quit

The loop stops on **exactly two** conditions (ORCHESTRATOR.md STEP 3):
1. **Verified PASS** — the MISSION win condition, confirmed by a W5 VERIFY worker, or
2. **Human stop** — the human edits `active_context.md` BUDGET / kills the session.

"Out of knobs", "out of ideas", and "diminishing returns" are **not** stop conditions. When every
board entry is closed and the gap survives, the orchestrator runs the **escalation ladder**
(MISSION §6): widen the cause space → widen the mechanism space (a new literature family) → widen
the definition of the write (keys, β, alternating re-solves, null-space projection — all still
`gradient_steps = 0`) → report the current mechanistic account to the human and start the next
board. Honesty is a *reporting* obligation, not an exit.
