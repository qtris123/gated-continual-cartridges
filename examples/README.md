# Workflow examples

`examples/` is organized by responsibility:

- `shared/` contains Python workflow engines grouped by capability.
- `qasper/`, `quality/`, `longhealth/`, `finqa/`, and `techqa/` contain
  dataset-owned Bash recipes only.
- `maintenance/` contains explicit one-off migration and data conversion tools.

Dataset recipes intentionally repeat environment variables, paths, and GPU
settings. They are executable run cards, not reusable libraries. Shared behavior
belongs in `examples/shared/` or, when it is part of the product API, in
`cartridges/`.

Datasets, prompt corpora, generated parquet files, and reference CSVs belong
under `data/<dataset>/`. Notebooks belong under `notebook/`; experiment reports
belong under `notes/`.

Run recipes from the repository root, for example:

```bash
bash examples/quality/pipelines/run_5phase_am.sh
bash examples/qasper/synthesize/self_study.sh
```
