# Continual-AM experiment layout

**Start here for Phase-2 AM:** [`AM_EXPERIMENTS.md`](AM_EXPERIMENTS.md) — the techniques
implemented, the command line for each experiment group, and what each one found.

The old per-dataset wrapper scripts (`examples/qasper/pipelines/*.sh`,
`examples/qasper/sweeps/*.sh`, and the `run_*_5x5_continual.sh` wrappers) have been
removed. Experiments are now driven by two axes:

- **Axis A — dataset knowledge.** `examples/<dataset>/` owns only dataset-specific
  assets: `synthesize/` recipes and `benchmarks/`. What a dataset *is* (its five
  phases, synth paths, `p01_method`, and `scorer`) is declared once in
  `DatasetSpec` in [`examples/shared/am/continual_env.py`](../../examples/shared/am/continual_env.py).

- **Axis B — experiment workload.** Shared endpoints run any dataset:
  - [`build_p01.py`](../../examples/shared/am/build_p01.py) — stage-1 (p01) cartridge.
  - [`run_chain.py`](../../examples/shared/am/run_chain.py) — chain phases 2-5 and
    emit the 5x5 stage-by-eval matrix (`--method {am,gradient}`).
  - [`grad_step.py`](../../examples/shared/am/grad_step.py) — gradient/sparse step engine.

## Running an experiment

Experiments are manifests under
[`examples/shared/am/manifests/`](../../examples/shared/am/manifests/) (e.g.
`soft_locality.yaml`, `capacity_top_t.yaml`, `p01_rope.yaml`, `slots_per_doc.yaml`).
Each manifest names an `endpoint`, the `datasets:` to run, and the sweep `streams`.

```bash
python examples/shared/am/sweep.py check  --manifest examples/shared/am/manifests/soft_locality.yaml
python examples/shared/am/sweep.py launch --manifest examples/shared/am/manifests/soft_locality.yaml \
    --stream <stream> --dataset <dataset>
```

Configuration is recipes-as-config: a resolved `AMContinualConfig` YAML is written
to disk and passed to the endpoints via `RECIPE_CONFIG` (no per-knob env bridge).
To add a new selector, add its field to the relevant component `Config` and add a
stream to the manifest — any config field is a valid sweep axis.
