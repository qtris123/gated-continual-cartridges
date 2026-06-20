# Concurrent DDP on a 4-GPU box — gloo backend + flock GPU pool

- **Date:** 2026-06-20 (work done 2026-06-18)
- **Status:** done
- **Related:**
  [`2026-06-20-qasper2-longhealth-port.md`](./2026-06-20-qasper2-longhealth-port.md)
  (the *previous* bug on the same wrapper — an `env` PATH-shim hijack that
  caused the 03:22:00 UTC attempts to silently exit 0 with empty per-side
  logs; this note picks up at 03:49:34 UTC, after that fix made the inner
  script actually run and the NCCL hang therefore observable);
  [`2026-06-18-sparse-continual-sweep.md`](./2026-06-18-sparse-continual-sweep.md)
  (the longhealth-side sweep whose per-head/per-layer methodology this
  unblocked on qasper2); commit `c3df9fe` on `tri_work_placeholder`;
  `examples/qasper2/scripts/{train,pool}_continual_sparse.sh`.

## Goal

Make `train_continual_sparse.sh` self-arbitrate GPU and DDP rendezvous so a single
host (4 × RTX 5880 Ada, no NVLink) can run **two concurrent 2-GPU DDP training
jobs**, plus add an unattended queue script that fires the next config as soon
as the previous pair finishes. Concrete driver: clear the per-layer Qasper2
`top_t ∈ {64,128,256,512}` sweep without babysitting four sequential 70-minute
jobs.

## Setup

- **Host:** `a4u8g-mil-0026`, 4 × NVIDIA RTX 5880 Ada (46 GB), no NVLink, PCIe-only
- **Branch / commit:** `c3df9fe` on `tri_work_placeholder` (uncommitted: the new
  `pool_continual_sparse.sh`, edits to `train_continual_sparse.sh` in both
  `examples/qasper2/scripts/` and `examples/longhealth/scripts/`)
- **Code entry points:**
  - Wrapper: `examples/qasper2/scripts/train_continual_sparse.sh` (≈ identical
    copy in `examples/longhealth/scripts/train_continual_sparse.sh`)
  - Queue: `examples/qasper2/scripts/pool_continual_sparse.sh`
  - Python: `examples/qasper2/train/continual_sparse.py` →
    `cartridges/train.py` (DDP wrap at `wrapped_model = DDP(...)`, ~line 243)
- **Phase 1 cache used by the sweep this enabled:**
  - `outputs/qasper-initial-per-layer-all-reduce/d8103e75-4886-47a0-8af0-286ca4bec665/cache-step534.pt`
  - `bg_stats.pt` from same dir (the all-reduce-merged version from `c3df9fe`)
- **Hyperparameters (shared across all 8 enabled runs):**
  `global_batch_size=64`, `epochs=10`, `lr=2e-2`, `top_t ∈ {64,128,256,512}`,
  `momentum_masking=freeze`, `freeze_keys=1`, `granularity ∈ {per_layer, per_head}`,
  `idf_top_k=128`, `idf_smoothing=1.0`, `num_tokens=1024`, `max_steps=250`,
  `seed=42`, `distributed_backend=gloo`
- **Run IDs / W&B (project `SEACrowd`, entity `vqtri-purdue-university`):**
  | TOP_T | granularity | local UUID | wandb short |
  |---|---|---|---|
  | 64  | per_head  | `0a962537-…` | (manual launch — pre-pool) |
  | 128 | per_head  | `457e7585-…` | (manual launch — pre-pool) |
  | 256 | per_head  | `52148b46-…` | first pool, 06:11 → 07:24 UTC |
  | 512 | per_head  | `4a298966-…` | first pool, 06:11 → 07:24 UTC |
  | 64  | per_layer | `e7d58f58-…` | wandb `0mo75660` |
  | 128 | per_layer | `8395d484-…` | wandb `3ffh67tc` |
  | 256 | per_layer | `14ba2a6c-…` | second pool, 10:01 → 11:13 UTC |
  | 512 | per_layer | `15d1bc35-…` | second pool, 10:01 → 11:13 UTC |

## What I tried

- **Reproduce the now-visible split2x2 hang.** With the env-shim PATH hijack
  fixed (see related note `2026-06-20-qasper2-longhealth-port`),
  `train_continual_sparse_split2x2.sh` finally *did* invoke the inner script
  on both sides — and both runs hung in `DDP._sync_module_states` for the
  full 600 s NCCL watchdog window, then died with the misleading
  `RuntimeError: DDP expects same model across all ranks, but Rank X has Y params, while rank Z has inconsistent 0 params`.
  Logs: `examples/qasper2/scripts/split2x2_logs/run_{A,B}_20260618-034934_gpus*.log`
  (the *034934* timestamp; the earlier *032200* logs are the env-shim victims
  from the prior note and only contain the wrapper header).
- **First hypothesis: simultaneous launch / NCCL bootstrap collision.** Replaced
  the split2x2 wrapper with **per-script atomic GPU claim via flock**: each
  `train_continual_sparse.sh` invocation grabs `flock -n` on its picked GPUs'
  lockfiles under `/tmp/gpu_locks_${USER}/` and derives `MASTER_PORT=29500+lowest_gpu_idx`,
  so a second invocation in another terminal sees those GPUs as taken and
  picks the next pair on a different port. Held the FD for the lifetime of
  the bash process, so any exit (clean, Ctrl-C, OOM, kill -9) auto-releases.
- **Verified the simultaneity hypothesis was wrong.** Even with a 60-second
  manual stagger between launches, both jobs still hung at exactly the same
  DDP init step. Watchdog timeout was deterministic, not racy.
- **Compared `config.yaml`s of historically successful runs vs. failing ones**
  (rg `distributed_backend` across `outputs/`). Every successful Phase-1 /
  Phase-2 run on this host from 06-17 onward had `distributed_backend: gloo`;
  every failing run had `nccl`. The shell scripts had been switched to default
  `nccl` somewhere along the way, overriding the Python-side default of
  `gloo`.
- **Flipped the script default back to `gloo`.** Solo run completed DDP wrap
  in ~7 s. Two concurrent runs initialised, entered the training loop, and
  logged to W&B — no further changes.
- **Wrote `pool_continual_sparse.sh`** to handle the 4-config sweep
  unattended: takes a list of `TOP_T` values, polls `nvidia-smi + flock -n`
  to count claimable GPUs every 15 s, launches the next config as soon as
  ≥ `GPUS_PER_JOB` (default 2) come free, with a 20 s stagger between
  launches. Launched in `tmux new-session -d` so it survives the laptop
  closing.
- **End-to-end**: drove all 4 `top_t` per-layer runs from a single tmux
  session: `nohup ./pool_continual_sparse.sh 64 128 256 512`. 64+128 fired
  immediately, 256+512 fired automatically once the first pair released
  their flocks. Total wall clock ~2 h 25 min vs. ~5 h sequential.

## Key findings / insights

- **Root cause of the DDP hang was the backend, not concurrency.** NCCL
  deadlocks during `DDP._sync_module_states` on this host's PCIe-only
  topology (RTX 5880 Ada with no NVLink). Switching to gloo eliminates the
  hang — solo, concurrent, every variant. The "inconsistent 0 params" error
  is a downstream artifact of the watchdog timeout, **not** an actual
  parameter-list mismatch; ignore it as a diagnostic signal.
- **The flock-based GPU claim is independently useful** even with the
  backend fixed. It removes the need to remember which GPUs are free, picks
  a per-pair `MASTER_PORT`, and survives all exit paths via FD lifetime, so
  multiple terminals (or the pool + a manual run) can coexist without
  coordination.
- **The pool script never *holds* GPU locks** — it only opens-and-releases
  them via `flock -n … true` to *probe* which GPUs are claimable. The
  actual atomic claim happens inside the launched `train_continual_sparse.sh`.
  This means any number of pool processes (or pools + manual invocations)
  can coexist without race-claiming the same GPU pair.
- **Detached training jobs survive the launching tmux pane closing** via
  POSIX session-leader semantics. Once the controlling tty exits, the bash
  child gets reparented to PID 1 (init). All 8 runs in this thread had
  `PPID=1` for most of their lifetime — i.e. closing the laptop or killing
  Cursor's shell harness was strictly safe, no need for `nohup`.

## Gotchas / surprises

- **NCCL was the script's *active default*** — `train_continual_sparse.sh`
  was explicitly setting `DISTRIBUTED_BACKEND="${DISTRIBUTED_BACKEND:-nccl}"`,
  overriding `continual_sparse.py`'s gloo default. Always check both layers
  when a "default" looks wrong.
- **600 s NCCL watchdog == one entire wasted run.** This box's NCCL hang is
  silent: no error, no log entry, no GPU activity, just `running` state for
  10 minutes followed by a confusing param-mismatch traceback. If you ever
  see two ranks with `last_completed` counters that diverge by exactly 1
  collective op while sitting at high utilisation but zero progress, suspect
  the backend before the model.
- **`flock` is per-FD, not per-path.** The
  `exec {_fd}>"$_lockfile"; flock -n "$_fd"` idiom means the script must
  keep that FD open for the whole run; you can't `flock -n "$path" some_cmd`
  inside the same script and get the same effect, because the lock releases
  when the wrapper command exits.
- **Probing without race:** `flock -n "$lockfile" true 2>/dev/null` is
  safe to call repeatedly from a poller — it acquires and releases the
  lock atomically, never blocks, and never disturbs whoever currently holds
  it (or doesn't).
- **`MASTER_PORT=29500+lowest_gpu_idx` is fine for ≤ 4 GPUs but collides
  if someone hard-codes 29500/29502 elsewhere.** The Phase-1
  `train_initial_sparse.sh` historically used 29507; the previous static
  qasper2 default was also 29507. Pick a non-default port if you ever run
  Phase 1 + Phase 2 on the same host concurrently.
- **The split2x2 wrapper is superseded for this workflow.** Both
  `examples/qasper2/scripts/train_continual_sparse_split2x2.sh` and the
  longhealth copy still exist (and are now functionally correct after the
  env-shim fix in the related note), but for unattended sweeps the
  single-script flock model + `pool_continual_sparse.sh` are strictly
  better: GPU pairs aren't fixed at launch time, ports auto-derive, and the
  queue keeps going past 2 simultaneous jobs.
- **W&B run names from the pool default to `qasper_phase2_TOP_T_<N>_<TS>`**
  — prefix is configurable via `RUN_NAME_PREFIX`. The per-layer pool was
  launched with `RUN_NAME_PREFIX=qasper_phase2_perlayer_TOP_T` so the
  granularity is in the run name. Forgetting to set this on a future
  per-head pool would produce ambiguous wandb names.

## Artifacts

- **Code:**
  - `examples/qasper2/scripts/train_continual_sparse.sh` (modified — flock
    auto-claim, deterministic `MASTER_PORT`, gloo default)
  - `examples/longhealth/scripts/train_continual_sparse.sh` (same gloo
    default fix)
  - `examples/qasper2/scripts/pool_continual_sparse.sh` (new — unattended
    multi-config queue)
- **Sweep outputs (8 runs, all `exit=0`, `last_optimizer_step=250`):**
  - `outputs/2026-06-18-04-58-20-continual_sparse-qasper-per-head-top-64-…/`
  - `outputs/2026-06-18-04-59-17-continual_sparse-qasper-per-head-top-128-…/`
  - `outputs/2026-06-18-06-11-28-continual_sparse-qasper-per-head-top-256-…/`
  - `outputs/2026-06-18-06-11-48-continual_sparse-qasper-per-head-top-512-…/`
  - `outputs/2026-06-18-08-49-29-continual_sparse-qasper-per-layer-top-64-…/`
  - `outputs/2026-06-18-08-49-33-continual_sparse-qasper-per-layer-top-128-…/`
  - `outputs/2026-06-18-10-01-25-continual_sparse-qasper-per-layer-top-256-…/`
  - `outputs/2026-06-18-10-01-45-continual_sparse-qasper-per-layer-top-512-…/`
- **Pool / per-job logs:**
  `examples/qasper2/scripts/queue_logs/pool_master_2026061{8-060504,8-084902}.log`
  and `pool_*_top{64,128,256,512}.log` siblings.
- **Original failure logs (NCCL hang):**
  `examples/qasper2/scripts/split2x2_logs/run_{A,B}_20260618-034934_gpus*.log`

## Open questions / next steps

- [ ] Determine *why* NCCL hangs on this PCIe-only topology — try
  `DISTRIBUTED_BACKEND=nccl NCCL_P2P_DISABLE=1` on a smoke run to confirm the
  P2P path is the failing piece, vs. some `NCCL_DEBUG=INFO` quirk on this
  driver/CUDA/NCCL combo. If `NCCL_P2P_DISABLE=1` works, gloo is leaving
  measurable throughput on the table for any future bigger model.
- [ ] Generalise the pool over more than just `TOP_T`. Today it only sweeps
  one variable; making it accept `KEY=val,val,val` pairs (or a config matrix
  YAML) would let a single pool run a granularity × top_t product.
- [ ] Add a watchdog in the pool: if a child run hasn't written to its
  per-job log for >N seconds, print a warning and optionally kill it, so
  unattended overnight pools fail loud.
- [ ] Apply the same flock + gloo-default + pool pattern to the longhealth
  variants (`train_continual_sparse.sh` already gloo-fixed; the pool isn't
  ported there yet).
- [ ] Decide whether to commit the script changes onto `tri_work_placeholder`
  or split a small `infra/concurrent-ddp` branch — currently they live as
  uncommitted/untracked changes alongside a much larger `git status` diff.
