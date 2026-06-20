# Re-running the qasper2 value-only continual sweep at batch size 32

- **Date:** 2026-06-20 (sweep launched 2026-06-19 03:39 UTC, currently running)
- **Status:** in-progress (per_layer pool active; per_head pool queued sequentially behind it)
- **Related:**
  [`2026-06-20-ddp-gloo-flock-pool.md`](./2026-06-20-ddp-gloo-flock-pool.md)
  (the gloo + flock + `pool_continual_sparse.sh` infrastructure being reused
  unchanged here; original GBS=64 baseline whose numbers this sweep is meant
  to be compared against);
  [`2026-06-20-qasper2-longhealth-port.md`](./2026-06-20-qasper2-longhealth-port.md)
  (first occurrence of the `~/.local/bin/env` PATH-shim hijack in
  `train_continual_sparse_split2x2.sh`; this note records the second, mirror-image
  occurrence and the same fix);
  [`2026-06-18-sparse-continual-sweep.md`](./2026-06-18-sparse-continual-sweep.md)
  (the longhealth-side methodology this is repeating on qasper2);
  uncommitted on top of `c3df9fe` on `tri_work_placeholder`;
  `examples/qasper2/{train/continual_sparse.py, scripts/train_continual_sparse*.sh, scripts/launch_value-only_grid_b32.sh}`.

## Goal

Re-run the qasper2 Phase-2 value-only continual sweep at `GLOBAL_BATCH_SIZE=32`
(half of the existing GBS=64 baseline), holding everything else identical so
the dynamics analysis can compare cache-update behaviour at two batch scales.
Concrete grid: `granularity ∈ {per_layer, per_head}` × `TOP_T ∈ {64, 128, 256, 512}`
= 8 runs, value-only (`FREEZE_KEYS=1`), driven through the existing
`pool_continual_sparse.sh` queue at 2 concurrent 2-GPU jobs on the 4-GPU host.

## Setup

- **Host:** `a4u8g-mil-0026`, 4 × NVIDIA RTX 5880 Ada (46 GB), no NVLink, PCIe-only
  (same machine and same gloo-only constraint as 2026-06-18 sweep).
- **Branch / commit:** uncommitted changes on top of `c3df9fe` on
  `tri_work_placeholder`. Three plumbing edits + one new (broken) wrapper:
  - `examples/qasper2/train/continual_sparse.py` — added
    `MAX_STEPS = int(os.environ.get("MAX_STEPS", "250"))` near the other env
    reads and threaded it into `CosWithWarmup.Config(max_steps=MAX_STEPS, …)`.
    Replaced the hardcoded `max_steps=250  # for bsize 64; 500 for bsize 32`
    comment.
  - `examples/qasper2/scripts/train_continual_sparse.sh` — exposes
    `MAX_STEPS="${MAX_STEPS:-250}"`, echoes it in the startup banner, and
    forwards it in the `torchrun` env block.
  - `examples/qasper2/scripts/train_continual_sparse_split2x2.sh` — added
    `MAX_STEPS` to `PER_RUN_VARS` for parity (split2x2 is otherwise unused
    here; see "Gotchas").
  - `examples/qasper2/scripts/launch_value-only_grid_b32.sh` — **new but
    broken on this host** because of the env-shim bug (see "What I tried" /
    "Gotchas"); kept as a record of the intended layout, not as the runner.
- **Code entry points (unchanged from 2026-06-18 sweep):**
  - Wrapper: `examples/qasper2/scripts/train_continual_sparse.sh`
  - Pool driver: `examples/qasper2/scripts/pool_continual_sparse.sh`
  - Python: `examples/qasper2/train/continual_sparse.py`
- **Phase-1 caches (granularity must match):**
  - per_layer:
    `outputs/qasper-initial-per-layer-all-reduce/d8103e75-4886-47a0-8af0-286ca4bec665/{cache-step534.pt, bg_stats.pt}`
  - per_head:
    `outputs/qasper-initial-per-head-all-reduce/51f1e2fb-321a-4cee-8796-9a826fb6681d/{cache-step534.pt, bg_stats.pt}`
- **Hyperparameters (shared across all 8 runs):**
  `GLOBAL_BATCH_SIZE=32`, `MAX_STEPS=500`, `EVAL_EVERY_N_STEPS=16`,
  `EPOCHS=10`, `LR=2e-2`, `FREEZE_KEYS=1` (value-only),
  `MOMENTUM_MASKING=freeze`, `IDF_TOP_K=128`, `IDF_SMOOTHING=1.0`,
  `NUM_TOKENS=1024`, `distributed_backend=gloo`, `seed=42`. Schedule:
  `CosWithWarmup` with `warmup_steps=20`, `alpha_f=0.1`,
  `max_steps=MAX_STEPS=500`.
- **Active launch (Option A from the chat):** outer `nohup bash -c '…' > value-only_b32_grid.log 2>&1 & disown`
  with **two `/usr/bin/env "${COMMON_ENV[@]}" … bash pool_continual_sparse.sh 64 128 256 512`**
  invocations in sequence (per_layer first, then per_head). Outer PID was
  `843711`; pool driver PID was `843712`. Started 2026-06-19 03:39:20 UTC.
- **Output run dirs in progress at time of writing:**
  `outputs/2026-06-19-03-39-48-continual_sparse/` (TOP_T=64, per_layer)
  and `outputs/2026-06-19-03-39-52-continual_sparse/` (TOP_T=128, per_layer).
  Subsequent runs will land at adjacent timestamps as the pool flushes.

## What I tried

- **Plumbed `MAX_STEPS` through `continual_sparse.py` end-to-end.** The
  existing hardcoded `max_steps=250` was correct for GBS=64 (10 epochs ≈ 250
  optimizer steps) but at GBS=32 the per-epoch step count doubles, so the
  cosine schedule would terminate at step 250 of ~500 and pin LR at the
  `alpha_f=0.1·LR` floor for the second half — incomparable to the GBS=64
  baseline. The new env knob defaults to 250 (preserves all existing GBS=64
  scripts unchanged) and the GBS=32 launch sets it to 500 explicitly.
- **First launch attempt — `launch_value-only_grid_b32.sh`.** Wrote a
  wrapper that runs the two pool invocations sequentially with a per-pool
  log dir and a master log. Backgrounded with `nohup … & disown`, PID
  `841927`. **Exited rc=0 in <1 s with empty per-pool logs.** The
  `grid_master.log` truthfully reports "sub-grid per_layer DONE rc=0" within
  the same wall-clock second the pool was launched.
- **Second launch attempt — equivalent inline `nohup bash -c '…'`.** Same
  content as the script, dropped into a here-string. PID `842283` died
  instantly; `value-only_b32_grid.log` contained only `nohup: ignoring input`
  with nothing from the pool.
- **Diagnosed: `~/.local/bin/env` PATH-shim was hijacking the launches.**
  `which env` resolved to `/localhome/local-triv/.local/bin/env`, an 11-line
  `sh` script that ignores all arguments, prepends `~/.local/bin` to PATH if
  missing, and exits 0. So `env "${COMMON_ENV[@]}" GRANULARITY=… bash pool…`
  was eating the entire trailing command and returning success — `set -e`
  saw rc=0, the wrapper happily moved on. Same bug already documented in
  the in-file comment of `train_continual_sparse_split2x2.sh` and in
  `2026-06-20-qasper2-longhealth-port.md`; both new launch paths re-introduced
  it.
- **Third launch attempt — Option A: replace `env` with `/usr/bin/env`.**
  Bypasses the shim. PID `843711`, started 2026-06-19 03:39:20 UTC. The
  pool master logged `next: TOP_T=64` at 03:39:20, `launched TOP_T=64` at
  03:39:24, `launched TOP_T=128` at 03:39:44, then `waiting: 0/2 GPUs free`
  from 03:40:04 onward — exactly the desired schedule. All 4 GPUs claimed
  via flocks under `/tmp/gpu_locks_local-triv/gpu{0,1,2,3}.lock`; util
  reached 86–100% on all four after the IDF init phase.
- **Pedagogical detour: `kill -INT 843711` on the wrapper did nothing.**
  User typed it accidentally; the pipeline kept running. Verified by
  `pgrep -af` showing the entire DDP tree still alive plus pool logs still
  growing (147 KB and counting). Cause: backgrounded bash inherits
  `SIG_IGN` for SIGINT/SIGQUIT under POSIX job-control rules, so the kernel
  delivers the signal and bash discards it. No traps fire, no state
  changes, no cleanup queued. SIGTERM would have worked.

## Key findings / insights

- **`MAX_STEPS` must scale with `GLOBAL_BATCH_SIZE`** when `EPOCHS` and the
  dataset are fixed. Halving `GLOBAL_BATCH_SIZE` doubles the optimizer step
  count for the same data; if `max_steps` is hardcoded the cosine schedule
  no longer covers the run and the LR sits at the floor for the second half.
  Linear rule on this dataset at EPOCHS=10: `max_steps ≈ 16000 / GLOBAL_BATCH_SIZE`
  (250 ↔ 64, 500 ↔ 32). Everywhere else in this repo that hardcodes
  `max_steps` is now a latent bug under any future batch-size change.
- **`/usr/bin/env "${ARRAY[@]}" cmd` is the safe form on this account** as
  long as `~/.local/bin/env` exists. The bare `env "${ARRAY[@]}" cmd` form
  (the obvious form, the one almost every shell tutorial uses) is hijacked
  silently. Subshell + bash-builtin `export` (Option B from the chat) is
  also safe. The split2x2 wrapper already uses Option B; new scripts in
  this user's tree must pick one of those two and a comment explaining why.
- **`kill -INT <bg-bash-PID>` is silently a no-op.** The kernel delivers
  the signal, bash's SIG_IGN handler discards it, and nothing happens. The
  failure mode is "I asked it to stop and it didn't, must be a bug" — but
  there is no bug, just the wrong signal. Use `SIGTERM`, signal the
  process group (`kill -TERM -- -<PGID>`), or `pkill -f` the inner work.
- **Two pool processes on the same host are race-safe** (per the
  probe-vs-claim split from 2026-06-18) but I deliberately ran them
  sequentially via the wrapper anyway. Justification: with 4 GPUs and
  `GPUS_PER_JOB=2`, only 2 jobs can run concurrently regardless of how
  many pools schedule them, so two parallel pools provide zero throughput
  benefit while introducing real risk of interleaved logs and competing
  flock probes if anyone resizes `GPUS_PER_JOB` later.

## Gotchas / surprises

- **Same env-shim bug, second occurrence in two days.** The 2026-06-20
  `qasper2-longhealth-port` note documents it on `train_continual_sparse_split2x2.sh`;
  the new `launch_value-only_grid_b32.sh` re-introduced it because it was
  written without consulting the existing warning. Until the shim is
  removed (probably installed by `uv` — see "Open questions"), every
  script that says `env "${ARRAY[@]}" cmd` will silently no-op on this
  account.
- **Both broken launch attempts reported `rc=0` and "DONE".** The
  grid_master log says exactly what you'd expect a successful run to say,
  with timestamps within the same second. The only signal that something
  was wrong was the empty per-pool `pool_stdout.log` files. Future
  diagnostic heuristic: if `pool_stdout.log` is 0 bytes after a "DONE"
  line, suspect the shim, not the pool.
- **`launch_value-only_grid_b32.sh` is currently broken on this host** and
  should not be used. Either patch it to use `/usr/bin/env` or
  subshell+export, or delete it. The inline Option-A form is what's
  driving the live sweep.
- **`/usr/bin/env` inside `nohup bash -c '…'` is fine.** The shim only
  triggers on the bare `env ` token at the start of a command line, which
  resolves through PATH. Using the absolute path (or a builtin like
  `export`) sidesteps the resolution entirely.
- **Status checks must use `pgrep -af` or `ps`, not `jobs`.** `disown`
  removes the backgrounded job from bash's job table, so `jobs` won't
  show a `disown`'d nohup'd process even though it's still running. Used
  `pgrep -af 'pool_continual_sparse|train_continual_sparse|torchrun|continual_sparse.py'`
  throughout this thread.
- **Two `outputs/2026-06-19-03-39-{48,52}-continual_sparse/` dirs** got
  created during launch — one per concurrent rank-pair. The pool's
  TOP_T-based naming lives in W&B / RUN_NAME; on disk the run-id is just
  a UUID under a launch-id timestamped to the second. Don't try to map
  output-dirs → TOP_T by inspection alone; cross-reference with
  `pool_<TS>_top<N>.log`.

## Artifacts

- **Code (uncommitted):**
  - `examples/qasper2/train/continual_sparse.py`
  - `examples/qasper2/scripts/train_continual_sparse.sh`
  - `examples/qasper2/scripts/train_continual_sparse_split2x2.sh`
  - `examples/qasper2/scripts/launch_value-only_grid_b32.sh` (broken on
    this host — `env "${ARRAY[@]}"` form; do not invoke as-is)
  - longhealth tree changes from earlier in the same chat (NCCL default,
    `NUM_GPUS` auto-derive from `CUDA_VISIBLE_DEVICES`, GPU-monitor scoping
    via `nvidia-smi --id`, master-port discipline) on
    `examples/longhealth/scripts/train_continual_sparse{,_split2x2}.sh` —
    unrelated to the qasper2 sweep but recorded here because they were
    touched in the same session.
- **Live logs:**
  - Outer wrapper: `examples/qasper2/scripts/value-only_b32_grid.log`
  - Per-pool master:
    `examples/qasper2/scripts/queue_logs/value-only_per-{layer,head}_b32/pool_master_2026061{9,?}-*.log`
  - Per-job: `pool_<TS>_top{64,128,256,512}.log` siblings.
- **Output run dirs (in progress):**
  `outputs/2026-06-19-03-39-48-continual_sparse/` (TOP_T=64, per_layer)
  and `outputs/2026-06-19-03-39-52-continual_sparse/` (TOP_T=128, per_layer).
  Six more will follow as the pool drains.
- **Forensics for the failed launches:**
  `examples/qasper2/scripts/queue_logs/value-only_b32_grid_20260619-032617/`
  contains `grid_master.log` with the "DONE rc=0" line in the same second
  as launch, plus 0-byte `per_{layer,head}/pool_stdout.log` files. This
  is the canonical signature of an env-shim hijack on this account.

## Open questions / next steps

- [ ] When the sweep finishes (~03:39 + ~5 h ≈ 08:39 UTC on 2026-06-19,
  i.e. likely already done by the time this note is read), run
  `eval_forgetting.sh` against the 8 new checkpoints and append actual
  wall-clock + retention/acquisition numbers below this list, then flip
  status to `done`. Specific comparison: do the value-only retention
  numbers from the GBS=64 baseline (~5.5 at top-64, ~9.9 at top-512)
  shift meaningfully at GBS=32?
- [ ] Decide what to do about `launch_value-only_grid_b32.sh`: either fix
  it to use `/usr/bin/env` or subshell+export AND add a self-check at
  the top that aborts if `which env` doesn't resolve to `/usr/bin/env`,
  or delete it and standardise on the inline `nohup bash -c '…'` form.
- [ ] Globally banish `~/.local/bin/env`. It was almost certainly
  installed by `uv` or a similar `~/.local/bin`-managing tool; it adds
  zero functionality (the PATH it tries to inject is already on PATH on
  this host) and is responsible for two silent failures in this repo in
  the last 48 hours. Two options: remove it outright, or rename it to
  something other than `env`. Until then, treat every `env "${ARRAY[@]}"`
  in this repo as a latent bug.
- [ ] Generalise `pool_continual_sparse.sh` to accept `(GRANULARITY, TOP_T)`
  pairs (or a `KEY=val,val,val` matrix) so a single pool can drive a
  granularity × top_t product without an outer wrapper. This was item #2
  on `2026-06-20-ddp-gloo-flock-pool.md`'s open list and would have
  removed the entire `launch_value-only_grid_b32.sh` from existence.
- [ ] Add a watchdog to the pool: if a child run hasn't grown its log in
  >N seconds, warn (and optionally kill). Would have caught both
  shim-hijacked launches in <30 s instead of relying on a manual
  `pgrep` check.
- [ ] Commit the `MAX_STEPS` plumbing onto a real branch (it's currently
  uncommitted alongside a much larger `git status` diff). The python
  side is a 2-line behavioural change with a default that preserves
  every existing GBS=64 launch unchanged; should be safe to land.
