# Design: manifest-centric continual-AM pipeline (sweeps, folders, stage-1 config)

- **Date:** 2026-09-03
- **Status:** in-progress (design only — refactor to be done in a separate chat)
- **Related:** [p1-rope-continual](./2026-08-27-p1-rope-continual.md), [qasper_recipe_layout](./experiments/qasper_recipe_layout.md), `outputs/experiments/soft_locality/FINDINGS.md`
- **Scope note:** this doc is meant to be **merged with the other design doc**. It captures three threads: (A) the manifest/sweep architecture already prototyped for `soft_locality`, (B) the cross-benchmark folder + stage-1 config decision reached in the 2026-09-03 chat, and (C) the *unifying frame* worked out in the 2026-09-03 endpoint-consolidation chat — the two-axis matrix model, the corrected method/density taxonomy, "recipes-as-config kills the knob registry", and the coverage check. Thread (C) is the organizing frame; it refines §4 and the endpoint set below.

## Goal

Stop maintaining near-duplicate run/orchestration/analysis code per benchmark. Make the continual-AM sweep pipeline **manifest-driven**, so that a single YAML is the source of truth for what arms run, how they're launched, how they're evaluated, and how they're reported — and so that changing a selection function or adding a benchmark is a config edit, not a copy-paste of shell scripts.

**Guiding principle:** *push differences into DATA/CONFIG, keep CODE shared.* The question "do we still need per-benchmark folders?" resolves to: keep them only for what encodes genuine dataset knowledge (synthesis + benchmark eval); everything else (run/orchestration/analysis) collapses into `examples/shared/`.

## Motivation / what's wrong today

Evidence gathered by normalizing dataset names and diffing:

- `examples/*/pipelines/run_*_5x5_continual.sh` are **~95% identical** across finqa/quality/techqa/qasper. The only real deltas are the dataset name and `AM_PHASE` vs `AM_QUALITY_PHASE` — both **already** encoded in `continual_env.DatasetSpec`.
- The stage-1 compaction block inside those wrappers (`NUM_TOKENS=512, KEY_SELECT=highest_attention, ENABLE_BETA=0, REBAKE_KEY_POSITIONS=1, AM_ROPE_THETA=model, GLOBAL_TEACHER_POSITIONS=1, RIDGE_LAMBDA=1e-4, RIDGE_SCALE=spectral, …`) is a **shared constant** (arm-D recipe) — byte-identical across datasets.
- `qasper/run_qasper_5x5_continual.sh` (30 lines) is already the clean shape: no stage-1 build, just `--p01-cache … && exec continual_chain.py`.
- `examples/*/synthesize/self_study.sh` are **genuinely per-dataset** (~29/160 lines differ): data source, token limits, `docs_per_prompt`, quirks (FinQA NaN-scrubbing, TechQA prompt caps). These are irreducible.
- `qasper/pipelines/train_*.sh` (10 files) + `qasper/sweeps/*.sh` (9) + several `longhealth/pipelines/*` predate the shared `continual_chain.py` and are likely dead — need an audit.
- Slot-selection config was previously defined in ~4 places (recipe YAML, env mapping, `SlotSelector.Config`, analysis), inviting drift.

## Unifying frame: the two-axis matrix model

*(Added 2026-09-03, thread C. This is the organizing frame for everything below; it resolves the felt tension "do `<data>` folders and shared endpoints fight each other?" — they don't, they're perpendicular.)*

### Two orthogonal axes

The repo mixes two concerns that should be separated cleanly:

- **Axis A — dataset knowledge** ("what *is* this dataset"): how to synthesize its data, how to score its answers, its paths / phase-keys / p01 method. **Irreducibly per-dataset.** Lives in `examples/<data>/` (`synthesize/`, `benchmark/` scorer) + `DatasetSpec` (facts).
- **Axis B — experiment workload** ("what do we *do*"): build p01, run a chain, sweep an axis, report, benchmark. **Dataset-agnostic.** Lives in `examples/shared/` endpoints + `manifests/`.

They "fight" today **only** because `<data>/pipelines/` and `<data>/sweeps/` put Axis-B workloads inside Axis-A folders. That is the root cause of both symptoms: ~95% duplicated runners (maintenance) and lopsided coverage (qasper has 9 sweeps, finqa 0 → incompleteness). **The refactor deliberately changes the "separation by data": keep it for *knowledge*, remove it for *workloads*.**

### The matrix

| | build_p01 | run_chain | sweep | report | benchmark |
|---|---|---|---|---|---|
| qasper | · | · | · | · | · |
| quality | · | · | · | · | · |
| finqa | · | · | · | · | · |
| techqa | · | · | · | · | · |
| longhealth | · | · | · | · | · | *(5-phase, phase-keyed; synth train + p01 kvcache to be generated)* |

- **Rows** = dataset knowledge (`<data>/` + `DatasetSpec`).
- **Columns** = the shared endpoints (verbs), parameterized by `--dataset`.
- **A manifest** = a selection of cells for one experiment type (`datasets: [...]`).

Endpoint surface (the "few endpoints"): **`synthesize`, `build_p01`, `run_chain`, `sweep`, `report`, `benchmark`.** Everything a researcher runs is one of these + a manifest/recipe. `sweep` is not a peer of the others — it is the **outer loop** that composes them (`for each arm → (render recipe) → run_chain|build_p01 → eval`); a sweep of one arm with no axis *is* a single `run_chain`. It references the other endpoints by pointer (p01 cache), pipeline path, and eval wrapper — it embeds none of them.

### Completeness by construction

A dataset supports an experiment **iff** it has a `DatasetSpec` + synth data for the required phases + a scorer. This is checkable, not aspirational:

- `sweep check --manifest X` asserts every dataset in the manifest's `datasets:` has those three things → the matrix is provably dense.
- Adding a dataset to an experiment = add it to `datasets:` (given its `DatasetSpec`/synth/scorer exist). No per-dataset script to copy.

### Corrected method taxonomy: method × density is a 2×2

`method ∈ {AM, gradient}` is the **only** real dispatch axis; **density (dense/sparse) is a recipe knob**, not a third method. (Earlier framing wrongly treated "sparse" as a sibling of AM/gradient.)

| | dense | sparse |
|---|---|---|
| **AM** | `continual_write`, top_t = all (full-KV) | `continual_write` + selection |
| **gradient** | `baseline_continual` (all positions) | `continual_sparse` (masked) |

Consequence for the runner: **one `run_chain`** owns the shared parts (stage loop, per-stage 5×5 eval, `slots_written.pt` / matrix emission) and **dispatches only the per-stage *step*** to `am_write` or `grad_train`. Density/top_t is a config knob inside whichever engine. The genuine difference to absorb is *execution model* (AM = closed-form, single-process, per-doc loop; gradient = torchrun, optimizer, data-parallel), so the two step-engines each own their process model behind a common `--recipe-config`.

### Recipes-as-config supersedes the knob registry (revises §4)

`slot_knobs.py` exists to bridge `recipe → ENV → Config` because `sweep.py` shells out to a **bash pipeline** that passes config via **env vars**. The registry is a *symptom of the env indirection*, not a feature. Given the components are already objectified (`SlotSelector`, `KeyWriter`, `ValueObjective`, `BetaFitter`, `TeacherTarget`, `ReferenceQueries`):

- **Recipe = the serialized component-config tree** (YAML of `AMContinualConfig`).
- **Sweeping = dotted-path override** on that tree — which `sweep.py` already does (`_set_dotted(cfg, "value_objective.ridge_lambda", v)`).
- **Endpoints read a resolved recipe file** (`--recipe-config`, which AM already accepts; torchrun/gradient reads the same file) — **no env vars.**

Then **any field is a potential axis**, `is_axis` collapses to "field exists in the schema," and **`slot_knobs.py` + the env-plumbing in `continual_env.py` can be deleted** — instead of adding `p01_knobs`/`objective_knobs` sibling registries (the wrong direction: scaling up scaffolding). Ablations over ridge / rope / target_mode / num_docs then become *manifests*, with no new plumbing.

### Where this leaves `<data>/`

`<data>/` keeps **only** Axis-A knowledge:

```
examples/<data>/
  synthesize/self_study.sh    # data prep (irreducible)
  benchmark/scorer.py         # how to score THIS dataset's answers (irreducible)
```

`<data>/pipelines/` and `<data>/sweeps/` empty out entirely; genuine one-offs (e.g. `techqa/experiments/slots_per_doc`) are re-expressed as p01-axis manifests (launch=`build_p01`, axis=`p01.num_docs`) and only stay in `<data>/experiments/` if they encode logic no shared endpoint expresses.

### Endpoint surface → concrete modules (naming reconciliation)

The verbs in thread C are the stable names; here is what each maps to today vs. after. Use the verb in prose; the "target module" column is the file a fresh agent creates/keeps.

| verb | today (exists?) | target module | notes |
|---|---|---|---|
| `synthesize` | `examples/<data>/synthesize/self_study.sh` (exists, per-dataset) | unchanged (Axis-A) | stays per-dataset |
| `build_p01` | **does not exist**; p01 is built inline inside `run_*_5x5_continual.sh` for quality/finqa/techqa; qasper takes a pointer | **NEW** `examples/shared/am/build_p01.py` | Python, reads `--recipe-config`; dispatches on `DatasetSpec.p01_method` (compaction vs selfdistill); idempotent |
| `run_chain` | `examples/shared/am/continual_chain.py` (exists, **AM-only**) | evolve `continual_chain.py` → `run_chain` | add `method ∈ {am,gradient}` step dispatch; density is a recipe knob |
| `sweep` | `examples/shared/am/sweep.py` (exists) | keep; add `check` subcommand | drop the env-var pathway once endpoints read `--recipe-config` |
| `report` | `examples/shared/analysis/continual_sweep_report.py` + `slot_geometry.py` (exist) | keep as `report` | already manifest-driven |
| `benchmark` | `examples/shared/benchmark/*` (exist, **partly broken**) | consolidate + fix | dedup ICL/cartridge; fix `eval_icl` import + removed `cartridge_p1/p2` |

### Current state vs target (so a fresh agent doesn't rebuild what exists)

| component | status | action |
|---|---|---|
| `sweep.py` (arms/render/launch/eval over manifest) | **exists, works** | keep; add `check`; later drop env pathway |
| `soft_locality.yaml` manifest + `slot_knobs.py` registry | **exist, work** | keep during phase 1–2; **delete `slot_knobs.py`** in the env-bridge-removal phase |
| `cartridges/am/continual/run.py` emits `slots_written.pt` + `run_meta.json` | **exists (verified)** | keep; do not rebuild |
| `continual_sweep_report.py`, `slot_geometry.py` | **exist** | keep (this is `report`) |
| `continual_chain.py` (AM p01→p05, 5×5) | **exists, AM-only** | evolve into `run_chain` with method dispatch |
| `build_p01.py`, `run_5x5_continual.sh`/`run_chain` generic wrapper | **do not exist** | create (Python, `--recipe-config`) |
| `DatasetSpec` (`continual_env.py`) | **exists**: `name`, `task_names`, `synth_template`, `topics` for qasper/quality/finqa/techqa | **add** `p01_method` + `scorer`/eval hook; **register longhealth** (phase-keyed, `synth_template="data/longhealth/train/qwen_longhealth_p{key}_task_8192.parquet"`, `p01_method=compaction`, MCQ scorer) |
| gradient/sparse continual recipe schema | **does not exist** (only AM `AMContinualConfig`) | define a recipe schema for the gradient step |

**longhealth — IN, first-class 5-phase dataset (decided 2026-09-03).** The old `examples/longhealth/pipelines/*` (patients 1–10 / 11–20, gradient-only) are a *superseded* 2-phase framing and get deleted in Phase 4. The **current** infra already matches the other four datasets:
- `cartridges/data/longhealth/phases.py`: `NUM_PHASES=5`, `PATIENTS_PER_PHASE=4`, `PHASE_TO_PATIENT_IDS` (patients 01–20 in ID order, ~46–48k tokens/phase, **80 questions/phase** — uniquely equal across phases, the cleanest forgetting curves).
- `self_study_vllm.py` already supports `--dataset longhealth --phase 1..5` via `LongHealthResource.Config(phase=phase)`.
- On disk already: `data/longhealth/phases/phase{1..5}_eval.parquet` + `phase{1..5}.txt` init texts + `manifest.json` (matches `DatasetSpec.eval_path`'s `data/{name}/phases/phase{phase}_eval.parquet` convention exactly).

So longhealth runs through the **same** pipelines (`synthesize` → `build_p01` → `run_chain` → `report`) with no special-casing. **To-populate (data task, not code):** `data/longhealth/train/qwen_longhealth_p{phase}_task_8192.parquet` (generate via `synthesize`) and the p01 kvcache (via `build_p01`). **Code to add:** register it in `DATASETS`, and ensure `continual_write`/`continual_env` dataset routing accepts `longhealth` (currently absent). Its scorer is MCQ-generation (Axis-A `benchmark/scorer.py`), not F1/numeric.

## Core design

### 1. Folder split by *nature*, not by benchmark

| layer | contents | home |
|---|---|---|
| reusable **code** | pipeline (`continual_chain`→`run_chain`, `continual_write`, `continual_env`), orchestration (`sweep.py`), analysis (`continual_sweep_report`, `slot_geometry`), eval wrappers (~~slot knob registry~~ — deleted in Phase 3, see thread C) | `examples/shared/` + `cartridges/am/` |
| per-dataset **knowledge** | `synthesize/` (data prep), `benchmarks/` (dataset eval/scoring), genuine one-off experiments (e.g. `techqa/experiments/slots_per_doc_sweep.py`) | `examples/<data>/` |
| the **join** | per-dataset paths/phase-keying/p01 method (`DatasetSpec`); per-experiment arms + cache pointers (manifest) | `continual_env.DatasetSpec` + manifest |

Net: `examples/<data>/pipelines/` largely empties out; `synthesize/` + `benchmarks/` stay.

### 2. The manifest is the single source of truth for a sweep

Prototype: `examples/shared/am/manifests/soft_locality.yaml`. A manifest declares:

- `datasets:` which benchmarks this sweep applies to.
- `launch:` per-dataset pipeline entry point + p01 cache pointer.
- `streams:` each an **axis** (a dotted config path, e.g. `slots.top_t` or `slots.usage_penalty_lambda`), a `base_recipe`, optional `fixed:` knobs, a `label_fmt`, and a list of `arms` (value + optional `alias` to reuse an existing run dir).

Two streams captured today:
- `top_t` — naive selective-update capacity sweep (t = 32, 64, 128, 256, 512).
- `lambda` — usage-penalized "soft slot locality" sweep at fixed t=32 (λ = 0, 0.5, 1, 2, 4, 16).

### 3. `sweep.py` — one tool over the manifest

`examples/shared/am/sweep.py` resolves the manifest and does everything the old shell orchestrators did:

```
python -m examples.shared.am.sweep arms   --manifest M [--stream S]
python -m examples.shared.am.sweep render --manifest M [--stream S]   # write recipes for alias-less arms
python -m examples.shared.am.sweep launch --manifest M --stream S --dataset D --gpus 0,1,2,3 [--dry-run] [--force]
python -m examples.shared.am.sweep eval   --manifest M --stream S --dataset D --gpus 0,1,2,3 --kind accuracy|generations
```

- `derive_tag(axis, value, fixed)` → stable, human-readable run tags (no manual naming).
- alias → reuse an existing run dir; no alias → `render` writes a recipe from `base_recipe` + the axis value.

### 4. Slot-knob registry — single source of truth for selection config

> **Superseded by thread C ("Recipes-as-config supersedes the knob registry").** The registry below is the *current* prototype and correctly removes drift **while the env-var bridge exists**. The target design removes that bridge (recipe = serialized config tree, endpoints read `--recipe-config`, sweeping = dotted-path override), at which point this registry and the `continual_env` env-plumbing are deleted rather than extended. Do **not** add `p01_knobs`/`objective_knobs` sibling registries. Text kept for context.

`cartridges/am/components/slot_knobs.py` defines each slot hyperparameter **once** as a `SlotKnob(field, env, cast, default, axis)`:

- `slot_recipe_env(slots)` → env dict (used by `continual_env.recipe_env`).
- `slot_config_kwargs(environ)` → kwargs for `SlotSelector.Config` (used by `continual_write.py`).
- `is_axis(field)` → which knobs are legal sweep axes.

Adding/changing a selection knob is now a **one-line registry edit** that automatically flows recipe → env → Config. Bespoke fields (`use_idf`, `background_indices_path`, `num_background_batches`, `slot_fisher_path`) stay explicit.

### 5. Native slot-update emission + run identity

- `continual/run.py` always emits `slots_written.pt` per stage (`{stage, cumulative, n_layers, n_slots, usage_penalty_lambda, usage_decay}`) plus a legacy bare `slot_usage.pt`. This replaces fragile cache-diffing for overlap/coverage analysis.
- `_write_run_meta` emits `run_meta.json` (git SHA + resolved config) per stage for reproducibility.
- `slot_geometry.py` reads `slots_written.pt` natively (cache-diff fallback for legacy runs) and plots per-stage overlap/coverage heatmaps for any stream.

### 6. Analysis is manifest-driven too

`examples/shared/analysis/continual_sweep_report.py` (replaces the old `analyze.py` + `export_tables.py`): reads the manifest, produces loss/accuracy frontier tables and tidy CSVs across all arms — no hardcoded arm lists.

## Stage-1 (p01) config — the three-way split

The reason "DatasetSpec vs manifest" felt ambiguous: stage-1 config is **three things with three lifetimes**, each wanting a different home.

| config | example | lifetime | home |
|---|---|---|---|
| compaction **recipe** | arm-D constants (`KEY_SELECT`, `RIDGE`, `REBAKE`, rope=model, …) | shared constant | defaults in a shared `build_p01.sh` (not per-dataset) |
| per-dataset **facts** | data paths, phase-keying, p01 **method** (compaction vs quality self-distill) | per-dataset, stable across experiments | `continual_env.DatasetSpec` (typed) |
| p01 cache **pointer** | which `cache_last.pt` this run consumes | per-experiment (ablations may swap p01) | manifest `launch: p01_cache` |

**Decouple p01 construction from the sweep** (qasper-style). Two thin shared scripts replace the four per-dataset wrappers:

1. `examples/shared/am/build_p01.sh --dataset X` — arm-D compaction with constants as defaults, templated per dataset via `DatasetSpec`; idempotent (skip if cache exists). Handles quality's self-distill via a `DatasetSpec.p01_method` field.
2. `examples/shared/am/run_5x5_continual.sh --dataset X --p01-cache … --recipe …` — pure chain runner (what qasper already is).

The manifest `launch:` then targets `run_5x5_continual.sh` + the per-dataset `p01_cache`.

## Target layout (after refactor)

Reflects thread C: **Python endpoints reading `--recipe-config` (no env bridge)**; `slot_knobs.py` + `continual_env` env-plumbing are **removed**, not kept.

```
examples/
  shared/
    am/
      build_p01.py              # NEW: p01 endpoint (method=compaction|selfdistill via DatasetSpec), --recipe-config
      run_chain.py              # NEW/EVOLVED from continual_chain.py: generic p01->p05, method dispatch (am|gradient)
      continual_env.py          # DatasetSpec = per-dataset FACTS ONLY (+ p01_method, scorer); env-mapping helpers DELETED
      continual_write.py        # AM step engine; takes resolved config (no env parsing)
      grad_step.py              # gradient/sparse step engine (wraps existing train drivers), takes resolved config
      sweep.py                  # manifest arms/render/launch/eval + NEW `check`; no env-var pathway
      slot_geometry.py          # per-stage overlap/coverage (native slots_written.pt)
      manifests/
        soft_locality.yaml      # experiment: lambda + top_t
        capacity_top_t.yaml     # experiment
        p01_rope.yaml           # experiment (launch=build_p01, axis=p01.rope_theta)
        slots_per_doc.yaml      # experiment (launch=build_p01, axis=p01.num_docs)
    analysis/
      continual_sweep_report.py # = `report`: frontier tables + tidy CSVs, manifest-driven
    evaluate/                   # generic accuracy/generation wrappers (run_accuracy.sh, record_generations.sh)
    benchmark/                  # consolidated ICL-vs-cartridge answer benchmark (eval_icl + cartridge_p1/p2 fixed)
  <data>/                       # finqa, quality, techqa, qasper (+ longhealth: see caveat)
    synthesize/self_study.sh    # KEEP: Axis-A dataset knowledge
    benchmark/scorer.py         # KEEP: Axis-A dataset scoring
    # pipelines/ and sweeps/  -> DELETED (workloads move to manifests + shared endpoints)
cartridges/am/
  continual/run.py              # emits slots_written.pt + run_meta.json (exists)
  components/
    slots.py                    # SlotSelector (+ usage penalty)
    # slot_knobs.py -> DELETED once recipes-as-config lands (was the env-bridge symptom)
    # (objective/keys/values/teacher/reference component configs live here)
```

## Migration plan (phased — each phase independently landable)

Each phase has a **Definition of Done (DoD)** and an **equivalence gate** so a fresh agent can land it without regressions. **Golden reference for equivalence** (from `2026-08-27-p1-rope-continual.md`): the QuALITY 5×5 lineage `delta_ha_b0_idf0_ropefix_p01D`, whose matrix is at `outputs/evaluations/quality/delta_ha_b0_idf0_ropefix_p01D/teacher-forced-logppl-v1/matrix.json` (p05 row ≈ `1.660 / 2.224 / 1.848 / 1.617 / 2.344`, mean forgetting ≈ −0.336). Re-run must match within ~1e-2 nats (recompute under current code first — see Gotchas re: ~1% decode offset).

**Phase 0 — safe deletions & bug-fixes (no behavior change).** Land first; unblocks everything.
- Fix broken imports: `examples/shared/evaluate/forgetting.py:94` and `examples/shared/benchmark/qasper_loss_benchmark.py:56` import `eval_icl` which does not exist (it is `examples/shared/evaluate/icl.py`).
- Remove stale `cartridge_p1`/`cartridge_p2` methods no longer accepted by `qasper_loss_benchmark.py`: `dispatch_loss_benchmark.py:29-30`, `compare_benchmark_results.py:64-66`.
- Add missing `from pathlib import Path` to `examples/maintenance/cache/group_compacted_caches.py`.
- Delete confirmed dead stubs: `examples/shared/am/initial_fit.py` (SystemExit), `examples/qasper/pipelines/train_initial_am.sh` (exit 2).
- DoD: `python -c "import"`/`--help` smoke-passes on the touched modules; no import errors repo-wide (`rg "from eval_icl"` returns nothing).

**Phase 1 — endpoint skeletons (additive, nothing deleted).**
- Add `build_p01.py` (Python, `--recipe-config`, `--dataset`; dispatch on `DatasetSpec.p01_method`; idempotent skip-if-exists). Wrap existing `initial_compaction.py` / `train_initial_selfdistill.py`.
- Evolve `continual_chain.py` → `run_chain` with a `method` switch: `am` → `continual_write.py`; `gradient` → new `grad_step.py` (thin wrapper over `continual_perplexity.py`/`continual_sparse_perplexity.py`). Keep the shared stage loop + per-stage 5×5 eval + emission untouched.
- Extend `DatasetSpec`: add `p01_method` and a `scorer`/eval hook; **decide the longhealth caveat** (register with variable phase count, or mark out-of-scope).
- DoD: `build_p01 --dataset quality` reproduces the arm-D p01 cache; `run_chain --dataset quality --method am` reproduces the golden matrix within tol.

**Phase 2 — collapse the per-dataset wrappers into manifests.**
- Point every manifest `launch:` at the generic `run_chain` + per-dataset `p01_cache`; delete the four `run_{qasper,quality,finqa,techqa}_5x5_continual.sh`.
- Re-express ablations/one-offs as manifests: `p01_rope.yaml` (launch=`build_p01`, axis=`p01.rope_theta`), `slots_per_doc.yaml` (axis=`p01.num_docs`), `capacity_top_t.yaml`. Retire `sweep_initial_am_rope.sh`, `run_am_ablations*.sh`, `launch_am_*_e2e.py`, `techqa/experiments/slots_per_doc_sweep.py`.
- Add `sweep check --manifest X` (asserts `DatasetSpec` + synth data + scorer for each listed dataset).
- DoD: `sweep launch --manifest soft_locality --dataset quality` reproduces the existing lambda arms; `sweep check` passes for all four datasets.

**Phase 3 — remove the env bridge (recipes-as-config).** *Highest blast radius; do last, in one pass or behind a shim.*
- Make endpoints read a resolved recipe file directly into the component-config tree (`AMContinualConfig` + a gradient recipe schema); stop passing config via env vars.
- Delete `cartridges/am/components/slot_knobs.py` and the env-mapping helpers in `continual_env.py` (`slot_recipe_env`, `recipe_env`, `teacher_env`-as-env, etc.); `sweep.py` `_set_dotted` becomes the only override path; `is_axis` → "field exists in schema."
- Touches at once: `sweep.py` (launch), `continual_write.py`, `grad_step.py`, `continual_env.py`, all recipe YAMLs. Use a temporary shim if landing incrementally.
- DoD: golden QuALITY matrix still matches within tol with **zero** `os.environ` config reads in the AM path (`rg "os.environ" cartridges/am examples/shared/am` audited).

**Phase 4 — audit & remove remaining legacy; docs.**
- Audit-then-remove (grep for callers first): `qasper/pipelines/train_*.sh` (10), `qasper/sweeps/*.sh` (9), `qasper_five_phase.py`, `rebuild_qasper_inventory.py`, `collect_qasper_continual_matrix.py`, `sweeps/eval_p1_p2_sa_refs.py`, `longhealth/pipelines/init_kvcache.sh` (references a missing builder). Collapse duplicate trainers `baseline_initial.py`≈`initial_perplexity.py` and `baseline_continual.py`≈`continual_perplexity.py` into the single self-distill trainer (they are near-byte-identical; `baseline` becomes a tag/flag).
- Update `notes/experiments/qasper_recipe_layout.md` (still documents the old `core/sweeps/benchmarks/infra` layout).
- DoD: `rg` finds no references to any deleted file; `examples/<data>/pipelines` and `examples/<data>/sweeps` are empty/gone.

### Verified delete / fix inventory (as of this session)

All paths confirmed present. `[fix]` = edit in place, `[del]` = remove, `[merge]` = fold into a shared endpoint.

- `[fix]` `examples/shared/evaluate/forgetting.py` — `eval_icl` import (→ `evaluate/icl.py`).
- `[fix]` `examples/shared/benchmark/qasper_loss_benchmark.py` — `eval_icl` import.
- `[fix]` `examples/shared/benchmark/dispatch_loss_benchmark.py` — remove `cartridge_p1/p2` methods.
- `[fix]` `examples/shared/benchmark/compare_benchmark_results.py` — remove `cartridge_p1/p2` classification.
- `[fix]` `examples/maintenance/cache/group_compacted_caches.py` — missing `from pathlib import Path`.
- `[del]` `examples/shared/am/initial_fit.py` (deprecated SystemExit stub).
- `[del]` `examples/qasper/pipelines/train_initial_am.sh` (deprecated exit-2 shim).
- `[del]` `examples/shared/am/qasper_five_phase.py`, `rebuild_qasper_inventory.py`, `sweeps/eval_p1_p2_sa_refs.py` (hardcoded to the historical 24-lineage QASPER run; superseded by generic `run_chain`).
- `[del]` `examples/shared/evaluate/collect_qasper_continual_matrix.py` (hardcoded QASPER collector; superseded by `report`).
- `[del]` `examples/longhealth/pipelines/init_kvcache.sh` (references a missing builder).
- `[merge]` `examples/shared/train/baseline_initial.py` → the single self-distill p01 trainer.
- `[merge]` `examples/shared/train/baseline_continual.py` → the single self-distill continual trainer.
- `[audit+del]` `examples/qasper/pipelines/train_*.sh` (10), `examples/qasper/sweeps/*.sh` (9), stale `examples/longhealth/pipelines/*` — grep callers, then remove if pre-`continual_chain`.

## Gotchas / surprises

- **p01 provenance differs by dataset:** quality has a self-distill path (`train_initial_selfdistill.py`); others use RoPE compaction. That's a real per-dataset config → `DatasetSpec.p01_method`, not a hardcoded wrapper.
- Old-vs-new accuracy offset (~1%) on identical p01 bases came from an old decode/env difference; always recompute baselines under current code (`rm -rf` the eval dir) before comparing.
- Teacher-forced log-perplexity is misleading here — descending grids were mostly *under-coverage* (only ~11% of slots ever written), not acquisition. Report generation accuracy alongside logppl.
- `torch.load` needs `weights_only=False` on PyTorch 2.6 for these caches.

## Decisions resolved (2026-09-03, thread C)

- [x] **Organizing frame:** two-axis matrix model. `<data>/` = Axis-A knowledge (synth + scorer + `DatasetSpec` facts); shared endpoints = Axis-B workloads; manifests select cells. `<data>/pipelines/` and `<data>/sweeps/` empty out.
- [x] **Endpoint surface:** `synthesize`, `build_p01`, `run_chain`, `sweep`, `report`, `benchmark`. `sweep` composes the others (pointer/pipeline/wrapper), embeds none.
- [x] **Method taxonomy:** `method ∈ {AM, gradient}` is the only dispatch axis; dense/sparse (and `top_t`) is a recipe knob → 2×2, not three methods.
- [x] **run_chain:** one runner owns the shared stage loop + eval + emission; dispatches only the per-stage step to `am_write` / `grad_train` (each owning its own execution model behind `--recipe-config`).
- [x] **Knob registry:** do **not** add sibling registries. Move to recipes-as-config-tree + dotted-path override + `--recipe-config` files; then delete `slot_knobs.py` and the `continual_env` env plumbing.
- [x] **Completeness:** enforced via a `sweep check --manifest X` that asserts `DatasetSpec` + synth data + scorer for every listed dataset.
- [x] **longhealth in-scope:** first-class 5-phase phase-keyed dataset on the same pipelines (infra already exists in `cartridges/data/longhealth` + `data/longhealth/phases`); only synth train data + p01 kvcache remain to generate. Old 2-phase scripts superseded.

## Open questions (deferred)

- [x] Confirm the three-way stage-1 split (recipe defaults / `DatasetSpec` facts / manifest p01 pointer) — **resolved**: arm-D defaults live in `build_p01.py` *and* a checked-in base recipe (`examples/shared/am/manifests/p01_armD.yaml`); `DatasetSpec` holds facts (`p01_method`, `scorer`); manifests carry the p01 pointer/overrides.
- [x] Representation of p01 method: **resolved** — `DatasetSpec.p01_method` default (the recommended option), dispatched inside the single `build_p01` endpoint (no separate build scripts).
- [x] Migration ordering & the legacy-script audit/removal (qasper `train_*`/`sweeps/*`, longhealth) — **done** in Phase 4 (grep-then-delete; only live-dependency files kept: `quality/pipelines/train_initial_selfdistill.py` for `build_p01`, `longhealth/pipelines/init_kvcache.sh` for the kept benchmark).
- [ ] Whether synthesis (`self_study.sh`) can share a skeleton with per-dataset overrides, or stay fully separate (currently ~29/160 lines differ). (Stays Axis-A either way.)
- [x] Sequencing of the env-bridge removal: **resolved** — done in one pass (Phase 3) with recipes-as-config (`RECIPE_CONFIG` → resolved YAML), then the `continual_chain.py` shim removed in Phase 4.
- [x] **longhealth: IN** as a first-class 5-phase, phase-keyed dataset (same pipelines). Remaining is a *data* task, not a decision: generate `data/longhealth/train/qwen_longhealth_p{phase}_task_8192.parquet` (`synthesize`) + p01 kvcache (`build_p01`), register in `DATASETS`, add longhealth routing in `continual_write`/`continual_env`, and add its MCQ scorer. Old 2-phase `examples/longhealth/pipelines/*` are superseded (deleted in Phase 4).
- [ ] `benchmark` endpoint scope: how much of `examples/shared/benchmark/*` to consolidate vs leave (answer-accuracy ICL/cartridge is a distinct track from the teacher-forced 5×5 matrices).
