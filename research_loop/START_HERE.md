# START HERE — launching, watching, and steering the loop

**What this loop is:** a **diagnosis engine** for one question — *why does the gradient-free
closed-form AM update cap at MT ~2.54, and what mechanism from the literature fixes it?*
It measures causes, imports fixes, builds them, tests one variable at a time, and tries to refute its
own results. It does **not** sweep knobs, and it does **not** stop because it ran out of them.

Read in this order: `PLAN_AM_MUST_WIN.md` (mission) → `DESIGN.md` (architecture) →
`ORCHESTRATOR.md` (the cycle) → `WORKERS.md` (roles) → `RUNBOOK.md` (mechanics).

## 0. Preconditions (one-time)
- On branch **`trivo-explore-research-work`** — the loop commits here only:
  ```bash
  cd /localhome/local-triv/gated-continual-cartridges_explore && git branch --show-current
  ```
- Environment (RUNBOOK §0 re-exports these, but set them in your shell too):
  ```bash
  export CARTRIDGES_DIR=/localhome/local-triv/gated-continual-cartridges_explore
  export CARTRIDGES_OUTPUT_DIR=$CARTRIDGES_DIR/outputs
  export CARTRIDGES_WANDB_PROJECT=SEACrowd
  export CARTRIDGES_WANDB_ENTITY=vqtri-purdue-university
  ```
- `HF_TOKEN` and `WANDB_API_KEY` in env, `~/.netrc` logged in — **wandb is mandatory** (RUNBOOK §0b);
  a GPU run without a wandb URL is treated as an invalid result.
- Both GH200s free. Warm up CUDA once (first import ~1–2 min, then fast):
  ```bash
  $CARTRIDGES_DIR/.venv/bin/python -c "import os;os.environ.setdefault('CARTRIDGES_DIR','$CARTRIDGES_DIR');import cartridges,torch;print(torch.cuda.device_count())"
  ```

## 1. Launch in tmux
```bash
tmux new -s amloop
cd /localhome/local-triv/gated-continual-cartridges_explore
claude   # then, at the prompt:
```
```
/loop Run ONE cycle of the orchestrator defined in research_loop/ORCHESTRATOR.md. Follow that file's steps exactly. State lives in research_loop/state/ — treat those files as your only memory.
```
(Omit an interval so the loop self-paces via ScheduleWakeup, as ORCHESTRATOR.md STEP 7.6 directs.)
Detach with `Ctrl-b d`; re-attach with `tmux attach -t amloop`.

## 2. Watch progress
| where | what you'll see |
|---|---|
| `state/bottleneck_board.md` | ★ **the real story** — which cause is active, its stage, what has been measured, what is ruled out |
| `state/active_context.md` | headline standings + next actions (rewritten every cycle) |
| `state/literature_ledger.md` | what the loop has read and what it plans to import |
| `state/mechanism_registry.md` | what it has built, and the mechanistic verdict on each |
| `state/results.csv` | the numbers, one row per GPU run, each with its wandb URL |
| wandb `vqtri-purdue-university/SEACrowd` | training/solve curves, grouped by board entry (`B-ROUTE`, …) |
| `notes/JOURNAL.md` | one line per milestone, newest first |
| `git log --oneline` | committed units of work on the branch |

## 3. Steering and stopping
- **Steer:** edit `state/active_context.md` (NEXT ACTIONS / BUDGET) or `state/bottleneck_board.md`
  (add a cause, re-prioritize the active one). The orchestrator re-reads both every cycle.
- **Stop:** write `STOP` into the BUDGET block of `active_context.md`, or `Ctrl-c` the tmux pane.
  The loop stops on its own **only** for a verified PASS — "out of knobs" is not an ending
  (MISSION §6 escalation ladder).
- **The win it is chasing:** `gradient_steps = 0` with **QA ≤ 2.52 AND MT ≤ 2.02**, confirmed by a
  fresh-process + changed-seed re-run plus the floor control and mechanism-off ablation.

## 4. What the loop will do on its first cycle
Every board entry starts at stage **A MEASURE**, so cycle 1 is diagnosis, not training:
1. **B-WIRE** — is `target_mode` a wiring no-op? (EXP-008's three modes were bit-identical to 15
   digits; if that's a bug, HYP-T1's "NULL" gets retracted.) No GPU.
2. **B-ROUTE write-ceiling oracle** — write the *teacher's own* KV into the selected slots. If MT
   stays ~2.5, no value-only frozen-key write can win and the mission pivots to keys. GPU.
3. **B-OBJ** — is the solve already near-perfect in MSE while CE doesn't move? GPU.
4. **SCOUT** — `AM.pdf` in depth, then the delta-rule and null-space-editing families. No GPU.

## 5. Nothing is blocking
All materials are resolved: the Phase-1 cache is staged and provenance-verified
(`outputs/phase1_selfdistill_qwen512/cache_last.pt`), `bg_stats.pt` is collected over that cache, the
dense bar and the sparse-gradient reference are measured, and every lever knob is env-driven
(RUNBOOK §4). The one open infra item is an opt-in `SEED` knob needed for the confirmation standard
(RUNBOOK §9c) — the loop builds it when it needs it.
