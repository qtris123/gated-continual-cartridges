# Workflow examples

`examples/` is organized by responsibility:

- `shared/` contains Python workflow engines grouped by capability, including
  the continual-AM endpoints (`build_p01`, `run_chain`, `grad_step`) and the
  experiment manifests under `shared/am/manifests/`.
- `qasper/`, `quality/`, `longhealth/`, `finqa/`, and `techqa/` contain
  dataset-owned assets: synthesize recipes, benchmarks, and any dataset-specific
  helpers. Experiment workloads themselves live in manifests, not per-dataset
  wrapper scripts.
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
# Dataset synthesize recipe (Axis A):
bash examples/qasper/synthesize/self_study.sh

# Continual-AM experiment (Axis B), driven by a manifest:
python examples/shared/am/sweep.py check  --manifest examples/shared/am/manifests/soft_locality.yaml
python examples/shared/am/sweep.py launch --manifest examples/shared/am/manifests/soft_locality.yaml --stream <stream> --dataset <dataset>
```
