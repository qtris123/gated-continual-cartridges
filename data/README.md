# Data layout

Everything for one stream lives under `data/<dataset>/`. There is no top-level
`phases/` tree; it was folded into each dataset.

- `data/<dataset>/phases/` — ICL/export corpus and eval sets for that stream:
  - `phase<k>.txt` — the phase resource's `to_string()`. Must stay byte-identical
    to the copy the eval sets were built against.
  - `phase<k>_eval.parquet` — that phase's questions.
  - `phase<k>_eval_anchored.parquet` — question stems rewritten to name their
    source document (FinQA and QuALITY only; see `cartridges/data/anchor_rewrite.py`).
  - `rewrite_raw/phase<k>_questions.parquet` — old/new question pairs from that rewrite.
  - `manifest.json` — per-phase token counts, question counts, cumulative tokens.
- `data/<dataset>/synth/p0<k>/<technique>/` — self-study runs: `config.yaml`,
  `source.json`, `repair.json`, `checkpoints/`, `artifact/dataset.parquet`.
- `data/<dataset>/train/` — AM train parquets (symlinks to the synth artifact).
- `data/qasper/eval/` — the QASPER phases are topic-named (`QA MT SA ASR KG` =
  phases 1–5), and the benchmark/sweep scripts address them that way, so
  `qasper_eval_<TOPIC>.parquet` are symlinks onto `../phases/phase<k>_eval.parquet`.
  Same 304 questions, one physical copy. `eval/reference/` is unrelated: the
  MCQ / yes-no CSVs used by the ICL baselines.
- `data/<dataset>/` also holds that stream's raw dumps (`raw/`, `init_text/`, `*.json`).

Regenerate a corpus or manifest with
`examples/maintenance/data/export_phase_corpora.py --dataset <name>`.
