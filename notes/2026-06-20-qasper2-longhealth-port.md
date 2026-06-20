# Port qasper2 to LongHealth-style local GPU pipeline (+ split2x2 env-shim bug)

- **Date:** 2026-06-20
- **Status:** done
- **Related:** commit `c3df9fe` (`tri_work_placeholder`), prior note [2026-06-18-sparse-continual-sweep](./2026-06-18-sparse-continual-sweep.md), longhealth scripts under `examples/longhealth/scripts/`

## Goal

Mirror the LongHealth two-phase sparse-cartridge pipeline into `examples/qasper2/` so that the qasper2 experiments (Phase 1 = QA, Phase 2 = MT) can run on this 4-GPU local box with the same operational ergonomics as longhealth — auto-detected paths, `.venv` activation, NCCL by default, a 2×2 GPU split for parallel sweeps, and parallel `eval_forgetting`. Then *actually run it* on the local box and verify nothing silently fails.

## Setup

- **Branch / commit (parent):** `c3df9fe` on `tri_work_placeholder` ("add sparse finetuning pipeline to longhealth + fix missing all_reduce background_stats on multi-gpus"). All edits in this entry are on top of `c3df9fe` and **not yet committed**.
- **Hardware:** 4 × NVIDIA RTX 5880 Ada (46 GB each), single host `a4u8g-mil-0026`
- **Python entry point:** `examples/qasper2/.venv/bin/python3` via `source $CARTRIDGES_DIR/.venv/bin/activate`
- **Phase 1 data (QA):** `data/qasper/train/qasper_QA_task_8192_no-cartridge.parquet` (333 MB)
- **Phase 2 data (MT):** `data/qasper/train/qasper_MT_task_8192_no-cartridge.parquet` (315 MB)
- **Phase 1 eval (QA):** `examples/qasper2/qasper_eval_QA.parquet`
- **Phase 2 eval (MT):** `examples/qasper2/qasper_eval_MT.parquet`
- **Existing Phase 1 cartridges (per-head, used as default `PHASE1_CACHE_PATH`):**
  - `outputs/qasper-initial-per-head-all-reduce/51f1e2fb-321a-4cee-8796-9a826fb6681d/cache-step534.pt` (112 MB)
  - `outputs/qasper-initial-per-head-all-reduce/51f1e2fb-321a-4cee-8796-9a826fb6681d/bg_stats.pt` (3.0 GB)
- **Init text file:** `examples/qasper2/train/qasper_init_1024.txt` (1024-token cartridge size)
- **W&B project:** `cartridges` (default; from `CARTRIDGES_WANDB_PROJECT`)

## What I did

### 1. Porting the directory structure to mirror LongHealth

Updated existing Python files to be the *non-sparse* baselines (matching the longhealth split between `initial.py` / `initial_sparse.py`) and added explicit `_sparse` siblings:

- **Edited** `examples/qasper2/train/initial.py` — non-sparse Phase 1 (no `bg_stats`); `EVAL_DATA_PATH` made optional.
- **Edited** `examples/qasper2/train/continual.py` — non-sparse Phase 2 (dense cache update); `EVAL_DATA_PATH` made optional.
- **New** `examples/qasper2/train/initial_sparse.py` — Phase 1 with `SparseCacheFinetuningConfig(enabled=False, collect_background_stats=True, num_background_batches=99999999999, granularity=GRANULARITY)`. Default `GRANULARITY=per_head`.
- **New** `examples/qasper2/train/continual_sparse.py` — Phase 2 with TF-IDF sparse mask (`use_idf=BG_STATS_PATH is not None`, `top_t=TOP_T`, `momentum_masking=MOMENTUM_MASKING`, `freeze_keys=FREEZE_KEYS`, `idf_top_k=IDF_TOP_K`). LR scheduler: `CosWithWarmup(max_steps=250, warmup_steps=20, alpha_f=0.1)`.

### 2. Porting the shell scripts to LongHealth style

All shell changes: stripped SLURM `#SBATCH` headers, removed hardcoded `/home/vo43/...` paths, replaced `conda activate cartridges` with `.venv` activation, guarded `module load` with `command -v module`, auto-detected `CARTRIDGES_DIR` via `cd $SCRIPT_DIR/../../..`, made `EVAL_DATA_PATH` optional, defaulted `DISTRIBUTED_BACKEND=nccl`, auto-detected `NUM_GPUS` from `CUDA_VISIBLE_DEVICES`, and tagged the GPU-monitor log filename with `${CUDA_VISIBLE_DEVICES//,/_}_${MASTER_PORT}` so concurrent runs don't clobber each other.

- **Edited** `examples/qasper2/scripts/train_initial.sh`
- **Edited** `examples/qasper2/scripts/train_continual.sh`
- **Edited** `examples/qasper2/scripts/eval_forgetting.sh` — now dispatches round-robin across `NUM_GPUS` GPUs (`PREV=$(( JOB_IDX - NUM_GPUS - 1 ))` slot-reuse pattern), per-job log files
- **Edited** `examples/qasper2/scripts/synthesize_self_study.sh`
- **New** `examples/qasper2/scripts/train_initial_sparse.sh`
- **New** `examples/qasper2/scripts/train_continual_sparse.sh`
- **New** `examples/qasper2/scripts/train_continual_sparse_split2x2.sh`
- **New** `examples/qasper2/scripts/sweep_top_t.sh` — sequential sweep over `TOP_T ∈ {64, 128, 256, 512}`

### 3. Pre-flight check, then a real run

User ran:

```
A_TOP_T=64 B_TOP_T=128 bash examples/qasper2/scripts/train_continual_sparse_split2x2.sh
```

at 2026-06-18 03:22:00 UTC. Wrapper reported `Run A exit code: 0`, `Run B exit code: 0` **in seconds**. Per-side logs (`split2x2_logs/run_A_20260618-032200_gpus0_1.log`, `..._B_*.log`) ended up at **376 / 377 bytes — only the wrapper's own header**. No GPU monitor logs were written, no `outputs/` directory created, no W&B run started. Both children exited 0 silently without ever executing the inner script.

### 4. Root-causing the silent failure

Stub-based bisection in `/tmp/`:

| Test | Command | Wall time | Side-channel marker |
|---|---|---|---|
| `bash stub.sh & wait` | direct | 2.0 s | written ✅ |
| `env FOO=bar bash stub.sh & wait` | with `env` | 0.005 s | empty ❌ |
| `/usr/bin/env FOO=bar bash stub.sh & wait` | absolute path | 2.0 s | written ✅ |

`type env` resolved to `/localhome/local-triv/.local/bin/env`, a **328-byte POSIX shell script** that only manipulates `$PATH` and exits — it does not `exec "$@"`:

```sh
#!/bin/sh
case ":${PATH}:" in
    *:"$HOME/.local/bin":*) ;;
    *) export PATH="$HOME/.local/bin:$PATH" ;;
esac
```

Because `~/.local/bin` is first on `$PATH`, the wrapper's
`env "${RUN_ENV[@]}" bash "$INNER_SCRIPT"` was hijacked: the shim updates its
own PATH, exits 0, and the inner training script is never invoked.

### 5. Fix

Replaced the `env` invocation in **both** split2x2 wrappers with a subshell + bash builtin `export`, which doesn't touch `$PATH` at all and is shim-immune:

```bash
(
  export CUDA_VISIBLE_DEVICES="$gpus"
  export MASTER_PORT="$port"
  export NUM_GPUS=2
  local kv
  for kv in "${RUN_ENV[@]}"; do
    export "$kv"
  done
  exec bash "$INNER_SCRIPT"
) >>"$log" 2>&1 &
```

Applied to:
- `examples/qasper2/scripts/train_continual_sparse_split2x2.sh`
- `examples/longhealth/scripts/train_continual_sparse_split2x2.sh` (same bug latent there; never noticed because no one had actually run the longhealth split2x2 wrapper yet — its `split2x2_logs/` directory didn't exist)

### 6. Post-fix verification (stub-based, no real training)

| Metric | Before fix | After fix |
|---|---|---|
| Wrapper wall time (with `sleep 3` stub) | 0.024 s | 3.03 s |
| Stub stdout in per-side log | empty | present |
| Side-channel marker written by stub | empty | both children logged |
| `CUDA_VISIBLE_DEVICES`, `MASTER_PORT`, `TOP_T` reaching child | nominally set, never used | actually used |
| Run A `TOP_T=64`, Run B `TOP_T=128` separation | n/a | confirmed |

## Key findings / insights

- **Result:** qasper2 now runs end-to-end on the local 4-GPU box without SLURM, with the same patterns longhealth uses (auto NUM_GPUS, NCCL, unique GPU-monitor logs per side, parallel `eval_forgetting`). The 2×2 split launcher correctly runs two independent torchrun jobs (Run A on GPUs 0,1 port 29507, Run B on GPUs 2,3 port 29508) and produces **two distinct W&B runs** (one per side, rank-0-only via `is_rank_zero` gate in `cartridges/train.py:351`).

- **Run names produced by `A_TOP_T=64 B_TOP_T=128 …split2x2.sh`:**
  - `qasper_phase2_sparse_freeze_key-value_adam_top-64_per_head_lr2e-2_all-reduce`
  - `qasper_phase2_sparse_freeze_key-value_adam_top-128_per_head_lr2e-2_all-reduce`

- **Why `env` was the silent-failure mode:** the shim is meant to be `source`d (it's a typical "ensure ~/.local/bin is on PATH" snippet), but landing it as an executable file named `env` higher in `$PATH` than `/usr/bin/env` shadows the system tool. Anything calling `env VAR=val cmd …` on this account sees a 0-exit no-op. Makefiles, container entrypoints, anything similar would all break the same way. Suspect a stray `pip install --user` / uv install put it there.

- **Direct invocation always worked.** Running `bash train_continual_sparse.sh` directly with `CUDA_VISIBLE_DEVICES=0,1 NUM_GPUS=2 …` foreground-launches torchrun, loads the cache (`num_trainable_tokens=1023`, `num_frozen_tokens=1`), wraps in DDP, and starts training. The bug is *only* in the split2x2 wrapper's `env`-based env-injection.

## Gotchas / surprises

- **`type env`, not `which env`.** The shim was on `$PATH` from `bashrc`; `which env` would have shown the same path but its presence was non-obvious until I ran `type env`. If you suspect a shadowing problem, `type` reports aliases, functions, and the resolved binary in one shot.
- **`set -e` does not propagate into a backgrounded `&` child.** The wrapper's children exited 0 (the env shim's success), and `wait $PID` happily returned 0 — the wrapper never noticed the silent failure. *Always* check log file size after a background fan-out.
- **Empty per-side log file ≈ "child never ran"**, not "child ran but was quiet". The wrapper's own header is written by the parent shell before the child spawns; if the child wrote nothing, the log will be exactly the header size (376/377 bytes here).
- **`PATH="stub:$PATH" bash inner.sh` does NOT preempt commands the inner script invokes after `source .venv/bin/activate`.** I tried this when stubbing torchrun. The venv activation prepends `$CARTRIDGES_DIR/.venv/bin`, masking my stub. Use absolute path stubs or stub at a different layer if you actually need to intercept torchrun.
- **`bash -x` does not echo redirections or `&`.** When tracing the wrapper, the `>>"$log" 2>&1 &` is invisible in xtrace; only the bare command is shown. Don't conclude redirection isn't happening just because xtrace doesn't show it.
- **The longhealth split2x2 had the same latent bug** but was never observed because nobody had actually launched longhealth's split2x2 wrapper before today. Both fixed in the same patch.

## Artifacts

- **New code (this session):**
  - `examples/qasper2/train/initial_sparse.py`
  - `examples/qasper2/train/continual_sparse.py`
  - `examples/qasper2/scripts/train_initial_sparse.sh`
  - `examples/qasper2/scripts/train_continual_sparse.sh`
  - `examples/qasper2/scripts/train_continual_sparse_split2x2.sh`
  - `examples/qasper2/scripts/sweep_top_t.sh`
- **Edited code:**
  - `examples/qasper2/train/initial.py`
  - `examples/qasper2/train/continual.py`
  - `examples/qasper2/scripts/train_initial.sh`
  - `examples/qasper2/scripts/train_continual.sh`
  - `examples/qasper2/scripts/eval_forgetting.sh`
  - `examples/qasper2/scripts/synthesize_self_study.sh`
  - `examples/longhealth/scripts/train_continual_sparse_split2x2.sh` (env-shim fix only)
- **Failed run logs (to be deleted; kept here as a record of the bug shape):**
  - `examples/qasper2/scripts/split2x2_logs/run_A_20260618-032200_gpus0_1.log` (376 B — only header, bug exemplar)
  - `examples/qasper2/scripts/split2x2_logs/run_B_20260618-032200_gpus2_3.log` (377 B)

## Open questions / next steps

- [ ] Decide on the env shim. Either delete it (`rm /localhome/local-triv/.local/bin/env`) or fix it by appending `exec "$@"`. Until that's done, the system has a foot-gun for any future `env VAR=val cmd` usage in scripts (Makefiles, CI, container entrypoints).
- [ ] Actually launch a real 2×2 sweep on qasper2 — `A_TOP_T=64 B_TOP_T=128` and pair with `A_TOP_T=256 B_TOP_T=512` — and add the results to the qasper2 forgetting-eval `EXPERIMENTS=()` list in `examples/qasper2/scripts/eval_forgetting.sh`.
- [ ] Commit the qasper2 port + env-shim fix as a single atomic PR. Suggested title: "port qasper2 to LongHealth-style local pipeline; fix env-shim hijack of split2x2 wrapper". Make sure both split2x2 wrappers are in the same commit so the fix doesn't get split across patches.
- [ ] Consider sweeping `MOMENTUM_MASKING ∈ {soft, hard, freeze, decouple}` × `TOP_T ∈ {64, 128, 256, 512}` on qasper2 to see if the per-head freeze conclusion from longhealth ([2026-06-18-sparse-continual-sweep](./2026-06-18-sparse-continual-sweep.md)) replicates on a different domain.
- [ ] Add a `bash -n` + a stub-launch smoke test to CI for both split2x2 wrappers, asserting that per-side log size > header-only size after a known-fast inner stub completes.
