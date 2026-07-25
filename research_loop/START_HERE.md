# START HERE — launching the unattended loop

## 0. Preconditions (one-time, verify)
- On branch **`trivo-explore-research-work`** (the loop commits here only):
  ```bash
  cd /localhome/local-triv/gated-continual-cartridges_explore && git branch --show-current
  ```
- Env vars exported (the loop's RUNBOOK §0 re-exports them, but set them in your shell too):
  ```bash
  export CARTRIDGES_DIR=/localhome/local-triv/gated-continual-cartridges_explore
  export CARTRIDGES_OUTPUT_DIR=$CARTRIDGES_DIR/outputs
  export CARTRIDGES_WANDB_PROJECT=SEACrowd
  export CARTRIDGES_WANDB_ENTITY=vqtri-purdue-university
  ```
- `HF_TOKEN`, `WANDB_API_KEY` already in env; `~/.netrc` logged in. 2× GH200 free.
- Warm up CUDA once (first import is slow ~1-2 min, then fast):
  ```bash
  $CARTRIDGES_DIR/.venv/bin/python -c "import os;os.environ.setdefault('CARTRIDGES_DIR','$CARTRIDGES_DIR');import cartridges,torch;print(torch.cuda.device_count())"
  ```

## 1. Launch in tmux
```bash
tmux new -s amloop
cd /localhome/local-triv/gated-continual-cartridges_explore
claude   # then, at the prompt:
```
In the Claude session, start the self-paced loop pointing at the orchestrator spec:
```
/loop Run ONE cycle of the orchestrator defined in research_loop/ORCHESTRATOR.md. Follow that file's steps exactly. State lives in research_loop/state/ — treat those files as your only memory.
```
(Omit an interval so the loop self-paces via ScheduleWakeup, as ORCHESTRATOR.md STEP 7 directs.)
Detach with `Ctrl-b d`; re-attach with `tmux attach -t amloop`.

## 2. Watch progress
- `research_loop/state/active_context.md` — headline standings + next actions (rewritten each cycle).
- `research_loop/state/results.csv` — the numbers.
- `notes/JOURNAL.md` — milestone narrative.
- `git log --oneline` on the branch — committed units of work.

## 3. Stopping / steering
- The loop self-stops when the target is met or the budget in `active_context.md` is exhausted.
- To steer mid-run: edit `research_loop/state/active_context.md` (NEXT ACTIONS / BUDGET) or
  `backlog.md` — the orchestrator re-reads them every cycle.
- To hard-stop: `Ctrl-c` in the tmux session (or kill the pane).

## 4. MATERIALS / THINGS I MAY NEED FROM YOU
Please confirm or provide:
1. **Dense self-distilled Phase-1 cache** (`cache_last.pt`) location — needed to init the
   `baseline_continual.py` self-distillation baseline (EXP-000). Local `outputs/*initial_am/` are
   AM-compaction caches, not dense self-distillation. Candidates to check: the sibling repo
   `/localhome/local-triv/gated-continual-cartridges/outputs/`, or a wandb artifact. If it doesn't
   exist anywhere, the loop will need to first re-run dense Phase-1 self-distillation (extra time) —
   tell me if you'd rather point me at an existing one.
2. **wandb run names** for the historical dense Phase-2 baseline, if you know them (pattern
   `qasper_baseline_phase2` / `qasper_phase2_*` under `vqtri-purdue-university/SEACrowd`). Speeds up EXP-002.
3. **Confirm the metric target** framing: "match self-distillation" = match its Phase-2 QA+MT eval
   loss within +0.15. If you have a specific target number in mind, put it in `active_context.md`.
4. **Budget** you're comfortable with (default written in `active_context.md`: ~40 cycles / ~48 GPU-h).
5. (Optional, only if you later want task-accuracy) a **tokasaurus/SGLang inference server** or Modal
   credentials — NOT needed for the perplexity-only plan we agreed on.

Nothing in this list blocks launching: the loop's cycle 0 will itself try to locate the Phase-1
cache and will escalate into `active_context.md` if it's truly missing.
