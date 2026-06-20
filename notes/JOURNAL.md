# Journal — index of `notes/`

Chronological index of investigations in this repo. **Most recent at the top.**

See [`README.md`](./README.md) for how to add an entry, and
[`TEMPLATE.md`](./TEMPLATE.md) for the per-entry structure.

| Date | Entry | Status | One-line summary |
|------|-------|--------|------------------|
| 2026-06-20 | [qasper-bsize32-grid-pool](./2026-06-20-qasper-bsize32-grid-pool.md) | in-progress | Re-launched the qasper2 value-only continual sweep at `GLOBAL_BATCH_SIZE=32` (8-run grid: granularity ∈ {per_layer, per_head} × top_t ∈ {64,128,256,512}); plumbed a new `MAX_STEPS` env knob through `continual_sparse.py` so the cosine schedule scales to the doubled step count, re-hit the same `~/.local/bin/env` PATH-shim hijack as `qasper2-longhealth-port` on the new wrapper and inline launches and fixed via `/usr/bin/env`, and verified that `kill -INT` on a `nohup`'d bash is a silent no-op (POSIX SIG_IGN inheritance) — sweep currently running. |
| 2026-06-20 | [qasper-forgetting-plots](./2026-06-20-qasper-forgetting-plots.md) | done | Plotted Phase 1 forgetting vs. Phase 2 learning perplexity as a function of active-slot budget (top-k ∈ {64,128,256,512}) for QASPER sparse continual cartridges across both granularities and both cache-update scopes — value-only shows a clean stability–plasticity trade-off (QA 6.6 → 13.5), key-value collapses Phase 1 to a flat ~16–19 ppl floor regardless of sparsity, indicating *touching keys* (not granularity) is the dominant driver of forgetting; investigation code lives outside the repo at `/localhome/local-triv/qasper-forgetting-investigation/`. |
| 2026-06-20 | [ddp-gloo-flock-pool](./2026-06-20-ddp-gloo-flock-pool.md) | done | After the env-shim fix exposed a deeper failure, root-caused the qasper2 split2x2 600 s NCCL watchdog hang to NCCL on PCIe-only RTX 5880 Ada (defaulted backend back to `gloo`), added flock-based GPU auto-claim + deterministic `MASTER_PORT` to `train_continual_sparse.sh`, and built `pool_continual_sparse.sh` for unattended multi-config sweeps — drove the 8-run per-layer/per-head TF-IDF sweep on qasper2 to completion (~2 h 25 min vs. ~5 h sequential). |
| 2026-06-20 | [dataset-train-eval-audit](./2026-06-20-dataset-train-eval-audit.md) | done | Audited QASPER vs LongHealth train↔eval distribution overlap; concluded LongHealth's stronger eval numbers are mostly identity-tuple retrieval + closed-vocab MCQ + fuzzy scoring (a dataset-design artifact), not a method-level effect — durable reference stored in `.cursor/skills/cartridges-experiment-investigation/DATASET_TRAIN_EVAL_AUDIT.md`. |
| 2026-06-20 | [qasper2-longhealth-port](./2026-06-20-qasper2-longhealth-port.md) | done | Ported `examples/qasper2/` to LongHealth's local 4-GPU pipeline (no SLURM, `.venv`, NCCL, parallel eval, 2×2 split launcher); root-caused a silent split2x2 failure to a stray `~/.local/bin/env` PATH-shim hijacking `env "${RUN_ENV[@]}" bash …` and replaced both qasper2 + longhealth split2x2 wrappers with a subshell + builtin `export`. |
| 2026-06-18 | [sparse-continual-sweep](./2026-06-18-sparse-continual-sweep.md) | done | Swept per-layer & per-head TF-IDF sparse continual cartridges on LongHealth; found and fixed critical bug where all DDP ranks independently saved biased `bg_stats.pt` instead of one all-reduced file. |
| _none yet — add entries above this row, newest first_ | | | |

<!--
Entry format (prepend, don't append):
| 2026-06-20 | [short-slug](./2026-06-20-short-slug.md) | done | One sentence — what was learned or decided. |
-->
