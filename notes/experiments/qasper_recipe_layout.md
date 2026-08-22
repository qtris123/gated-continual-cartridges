# qasper2 scripts

**Start here for Phase-2 AM:** [`AM_EXPERIMENTS.md`](AM_EXPERIMENTS.md) — the techniques implemented,
the command line for each experiment group, and what each one found.

Organized into four groups:

| Folder | Purpose |
|---|---|
| `core/` | Single-run launchers (Phase-1/2 train, eval, synth) |
| `sweeps/` | Ablation / sweep / GPU-pool orchestrators |
| `benchmarks/` | ICL + cartridge comparison tables |
| `infra/` | Shared helpers (e.g. bg_stats collection) |

Old flat paths (e.g. `examples/qasper/train_continual_am_sparse.sh`)
have been removed. Use the subfolder path instead, e.g.
`bash examples/qasper/pipelines/train_continual_am_sparse.sh`.
