# Data layout

- `data/<dataset>/` — raw dumps, eval splits, self-study synth runs, train parquets.
- `data/phases/` — ICL/export phase corpora (`phasek.txt`) and phase eval parquets. Keep this tree.
- `data/<dataset>/synth/<stage>/<technique>/` — Hydra self-study metadata (`config.yaml`, `source.json`).
- `data/<dataset>/train/` — AM train parquets (often a symlink to the synth artifact).
