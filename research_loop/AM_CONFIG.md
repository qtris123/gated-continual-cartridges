# AM CONFIG REFERENCE — every field of `AMContinualConfig`

> **What this file is:** the field-by-field reference for
> `cartridges/am/continual/config.py::AMContinualConfig` and the six stage configs it nests. It
> holds the long-form notes that would otherwise bloat the class definitions.
>
> **Where the numbers are:** the rationale, sanity checks, ablation tables and verdict for every
> `MECH-00x` knob are in `research_loop/state/mechanism_registry.md`. Selector semantics and metric
> definitions are in `research_loop/GLOSSARY.md`. This file does not repeat them; it says what each
> field does, whether it is live on the default path, and where to read the evidence.

## How the config is organised

Phase 2 is a closed-form, backprop-free write, so it is its own `RunConfig` — not a `TrainConfig`
with the optimizer fields set to zero. Each stage of the write owns its knobs:

```
select slots → collect reference queries → build teacher target
             → (optionally) rewrite keys → (optionally) fit β → solve values
```

```python
class AMContinualConfig(RunConfig):
    model: ModelConfig
    kv_cache_initializer: KVCacheFactory.Config   # the Phase-1 cartridge
    document_data_path: str                       # the MT parquet

    rope_theta: float = 10000.0                   # shared by 3 stages, so top-level

    slots:     SlotSelector.Config
    queries:   ReferenceQueries.Config
    teacher:   TeacherTarget.Config
    keys:      KeyWriter.Config
    beta:      BetaFitter.Config
    objective: ValueObjective.Config
```

Each stage config lives next to the code it configures, and `config.build_stages()` instantiates all
six once per run. The objects are stateless — after `__init__` the only attribute is `self.config` —
so the document loop stays the only thing carrying state between documents.

**Every MECH knob defaults to the pre-loop behaviour**, so a config that sets none of them reproduces
the historical runs bit-identically.

One behaviour outside that guarantee: the teacher's document text is read from the QASPER paper
(`teacher.qasper_topic`) rather than merged out of the synthesis rows' sampled sections. It is
byte-identical on the stock parquets — pinned by golden hashes over all 16 documents in both the MT
and QA topics — so the historical numbers stand. It *does* diverge on a **subsampled** document slice
(a DIAG-PERDOC-style parquet with few rows per document), where the old merge saw only the sections
those rows happened to sample. That is the point: coverage no longer depends on the draw.

| Mechanism | Fields it added | Where they live now | Board entry |
|---|---|---|---|
| MECH-001 | `oracle_write`, `oracle_write_assign` | `objective` | B-ROUTE |
| MECH-002 | *(none — it only plumbed the existing `max_queries_per_head` to an env knob)* | `queries` | B-CASCADE |
| MECH-003 | `rope_theta` | **top level** | B-ROPE |
| MECH-004 | `beta_box`, `nnls_iters`, `nnls_driver`, `target_mode` | `beta` | B-SOLVE |
| MECH-005 | `key_reposition` | `keys` | B-ROUTE / B-ROPE |
| MECH-006 | `onpolicy_layers`, `onpolicy_refresh_doc_kv` | `queries` | B-TARGET |
| MECH-007 | `seed_offset` | `queries` | variance accounting |
| MECH-008 | `redundancy_ridge_rel`, `mass_redundancy_alpha`, `slot_fisher_path`, and the `redundancy` / `fisher` / `mass_x_redundancy` selectors | `slots` | B-GATE |
| MECH-009 | `safe_fraction`, `safe_metric`, and the `constrained_mass` selector | `slots` | B-GATE |

One constraint still shapes the layout: **`extra="forbid"`**. Moving a field onto a nested sub-object
is a breaking change for any caller that still passes it flat, which is why the restructure was done
as a clean break rather than incrementally. The frozen drivers under `research_loop/results/` are
deliberately *not* updated — they are historical records, reproducible at their original commits.

The old defensive discipline (`getattr(config, ..., default)` reads and `_*_kwargs()` conditional
helpers, guarding against `import cartridges` resolving to a sibling checkout — RUNBOOK §6.10 /
MECH-000) is **gone**. Pin the import with `PYTHONPATH=$CARTRIDGES_DIR` instead; the config now fails
loudly rather than silently running historical behaviour.

---

## Top level

| Field | Default | Notes |
|---|---|---|
| `model` | — | The frozen base model. |
| `kv_cache_initializer` | — | Produces the Phase-1 cartridge to write into. |
| `document_data_path` | — | The MT parquet, grouped into documents by paper title. A plain path, not a `DocumentSource` class: the loop calls `load_conversations` then `group_conversations_by_document`, and nothing else. |
| `loss_evals` | `[]` | Optional QA (forgetting) / MT (acquisition) perplexity evals, run once after the last document. |
| `save_after_each_document` | True | Checkpoint the cache after every document write. |
| `compute_update_stats` | True | Populate `AMUpdateStats` (per-layer MSE, value norms, routing mass). |
| `keep_last_n_saved` | 1 | Read by the shared `train.save_cache`. |
| `save_to_wandb` | False | Read by the shared `train.save_cache`. |
| `wandb` | `None` | Typed `None`: this run writes to disk only. |
| `device` | `cuda` | |
| `seed` | 42 | Seeds `seed_everything`. **Does not reach the per-document reference draw** — see `queries.seed_offset`. |

### MECH-003 — `rope_theta` (10000.0)

Rotary base used when the teacher path re-positions the reference queries by
`doc_rope_offset = T_doc` before scoring them against the document keys
(`core._apply_rope_offset_to_queries`). The AM package hard-coded 10000.0 everywhere, but the
model's own RoPE base comes from its HF config — `Qwen3-4B-Instruct-2507` uses **5e6**. Composing
two rotations only lands on a single absolute position when both use the same θ, so the hard-coded
value made the composite angle meaningless.

**It is top-level, not on `teacher`.** Three stages consume it — the teacher target, the key rewrite
(`keys.key_reposition`) and the oracle write (`objective.oracle_write`). Duplicating it across three
sub-configs would let them disagree, which is precisely the MECH-003/MECH-005 bug class.

**The default is the historical value, not the correct one.** Set it to the model's own base
(`AM_ROPE_THETA=model`) to opt in. MECH-003 measured the correction as an 8.5× more fittable target
and a 5.5× gentler write, for a CE move inside the noise floor.

---

## `slots` — `SlotSelector.Config`

Which cartridge slots the write is allowed to overwrite. Implemented in
`cartridges/am/components/slots.py`.

| Field | Default | Notes |
|---|---|---|
| `top_t` | 64 | Slots written per layer (at `granularity="per_layer"`). |
| `granularity` | `per_layer` | Budget scope: `global` / `per_layer` / `per_head`. Also fixes the shape of the access scores `queries` collects, which is why `ReferenceQueries.collect` takes it as an argument rather than duplicating it. |
| `slot_selection` | `tfidf` | Scorer. See below. |
| `use_idf` | True | IDF-weight the term frequencies. Requires `background_indices_path`. |
| `idf_smoothing` | 1.0 | Additive smoothing in the IDF term. |
| `background_top_k_per_batch` | 128 | Slots counted as "accessed" per background batch. |
| `background_indices_path` | None | Cached background access statistics. |
| `num_background_batches` | 1000 | Cap on background batches consumed. |

### `slot_selection` modes

Pre-loop: `tfidf` (default), `attention_mass`, `residual_budget`.
MECH-008: `redundancy`, `fisher`, `mass_x_redundancy`. MECH-009: `constrained_mass`.

**Why there is no `kl_loo` mode.** DIAG-IMPORTANCE showed that the exact leave-one-out KL of
dropping slot *j* is `-log(1 - w_j)`, a monotone function of the slot's own attention weight. It is
therefore *the same ranking* as `attention_mass` (measured ρ = 0.968), so it was deliberately never
added as a separate selector.

### MECH-008 knobs (inert unless `slot_selection` names a prior mode)

- `redundancy_ridge_rel` (1e-6) — relative ridge in the redundancy Gram solve. 1e-6 is
  DIAG-IMPORTANCE's own value.
- `mass_redundancy_alpha` (0.5) — blend for `mass_x_redundancy`: 0 = pure attention mass,
  1 = pure redundancy, 0.5 = geometric mean.
- `slot_fisher_path` (None) — a **cached** `(n_layers, n_slots)` diagonal-Fisher array. The scoring
  pass is a diagnostic backward over QA data; it constructs no optimizer and leaves `gradient_steps`
  at 0, but it costs real GPU time, so it is paid once offline rather than per run.

### MECH-009 knobs (inert unless `slot_selection="constrained_mass"`)

`constrained_mass` is mass ranking *within* a safety constraint.

- `safe_fraction` (1.0) — fraction of slots per layer that stay candidates, keeping the safest by
  `safe_metric`; in (0, 1]. **The default 1.0 excludes nothing, so `constrained_mass` at its default
  is identical to `attention_mass`** — nothing changes unless a fraction is set explicitly.
- `safe_metric` (`redundancy`) — which QA-importance scorer defines "safest": `redundancy`
  (gradient-free) or `fisher` (needs `slot_fisher_path`).

### `residual_budget` knobs

- `idf_prior_weight` (0.0) and `min_top_t_per_layer` (1) — read only when
  `slot_selection="residual_budget"`. They moved onto this config as options of that mode; they used
  to sit in the flat config's legacy group.

---

## `queries` — `ReferenceQueries.Config`

The `Q_ref` the write is fitted against, plus the reference-data plumbing that produces it.
Implemented in `cartridges/am/components/queries.py`.

| Field | Default | Notes |
|---|---|---|
| `max_ref_examples_per_doc` | 32 | Conversations drawn per document out of its ~500. |
| `queries_per_batch` | `all_tokens` | `all_tokens` uses every position's query; `last_token` keeps only the final one (the original AM-paper-style single-query variant). |
| `max_queries_per_head` | 64 | Reference queries per KV head fed to the solve. `n > top_t` makes the system over-determined. |
| `ref_batch_limit` | 5 | Batch cap for a collection pass when the caller gives no explicit one, **and** the cap on the old-reference bank. Renamed from `decoupled_ref_batches`, which was named for a dead execution mode; both use sites are live. |

### MECH-007 — `seed_offset` (0)

**The only stochastic choice left in the per-document write.** The document loop draws each
document's reference conversations with `seed=doc_idx` — a constant — and `config.seed` never
reaches that draw (DIAG-KEYCURVE/`seed_probe.json`). `seed_offset` shifts the draw to
`doc_idx + seed_offset`, which is what "changed seed" means for this method: a different subset of
each document's conversations. DIAG-PERDOC measured the resulting spread at **0.216 loss** on a
per-document subset, so it is a real source of variance. Must be `>= 0` — a negative offset would
alias document indices onto each other.

### MECH-006 — `onpolicy_layers` (0), `onpolicy_refresh_doc_kv` (False)

By default `Q_ref` is collected in ONE forward pass over the *pre-write* cartridge, and all
`n_layers` layers are then solved against it. But writing layer `l` perturbs the residual stream, so
the queries the model actually emits at layers `l+1 … n-1` are not the ones we fitted. The AM paper
compacts layers sequentially and re-extracts `Q_ref^l` with layers `< l` already written.

`onpolicy_layers = N > 0` splits the write into groups of N layers and re-extracts the reference
queries from the UPDATED cache before each group after the first (N = 1 → per layer, i.e. up to
`n_layers - 1` extra prefill passes per document; N = 4 → 8 passes for a 36-layer model). 0 = off,
a single pass.

`onpolicy_refresh_doc_kv` additionally re-prefills the DOCUMENT KV against the updated cartridge, so
the teacher `[cartridge ‖ doc]` is on-policy as well. The paper only re-extracts queries (its target
block is captured from the unmodified model), so this is a second, separately-testable axis and
stays off by default.

---

## `teacher` — `TeacherTarget.Config`

The stage's rotary knob, `rope_theta`, is hoisted to the top level (see above).

| Field | Default | Notes |
|---|---|---|
| `qasper_topic` | `MT` | Topic whose papers the document titles resolve against: `QA` / `MT` / `SA` / `all`. Must match `document_data_path`. |

Implemented in `cartridges/am/components/teacher.py`. The stage has four verbs: `document_prompt`
builds the document text, `prefill` captures its KV once per document, then `targets` and `log_mass`
read the `[cartridge ‖ doc]` teacher per (layer, head) — the latter feeds `BetaFitter.fit`.

### `qasper_topic` (`MT`)

The prefilled document is the **whole QASPER paper**, looked up by title
(`queries.full_paper_prompt`). It is deliberately *not* reconstructed from the synthesis rows: each
row was generated over a random subset of the paper's sections
(`qasper/resources.py::sample_prompt`), so merging them makes coverage a property of how many rows a
document happens to have. That merge is complete on the stock parquets (~500 rows per document) but
degrades on a subsampled slice, and a row with no `<title>` tag degenerates to a single subset.

**Sourcing the paper is numerically inert on the stock parquets.** The prompt joins sections with a
bare newline rather than `PAPER_TEMPLATE`'s `---Paper Title: …---` divider, which would add +3,642
tokens across the 16 MT documents (+5.0%; +65 to +500 per document). That is not cosmetic: `T_doc`
*is* `doc_rope_offset` (`continual/write.py`), so any token-count change re-rotates every document
key and moves every teacher target. The exact bytes are pinned by
`test_document_prompts_match_golden_hashes` over all 16 documents in both topics — a diff there means
the operating point moved, not that the test is stale.

A title missing from the configured topic raises rather than falling back; the usual cause is a topic
that does not match `document_data_path` (a QA parquet with `topic="MT"` resolves nothing). All
prompts are resolved up front in `run_documents`, so this fails before the first write lands. Titles
are unique across the three topics, so `all` cannot collide.

Phase 1 takes the same treatment: `compact_cache_am_phase1(qasper_topic="QA")`.

---

## `keys` — `KeyWriter.Config`

Implemented in `cartridges/am/components/keys.py`.

| Field | Default | Notes |
|---|---|---|
| `key_mode` | `freeze` | `freeze` leaves cartridge keys untouched (values-only write); `highest_attention` and `omp` select document keys to install into the selected slots. `KeyWriter.enabled` is exactly `key_mode != "freeze"`. |

### MECH-005 — `key_reposition` (False)

**B-ROPE hazard H2 / LIT-026.** When `key_mode != "freeze"`, a *document* key that wins selection is
installed into a cartridge slot with no counter-rotation. It was scored against a query rotated
forward by `doc_rope_offset = T_doc`, but the student and eval score it with the raw query, so the
installed key is off by a multi-thousand-position phase and the value solve is then fitted against
that corrupted routing. Because RoPE is orthogonal, `⟨R_Δ q, k⟩ = ⟨q, R_{−Δ} k⟩`, so installing
`R_{−Δ} k_doc` restores exactly the logit that selected the key.

`key_reposition=True` re-bases every doc-sourced row by `-doc_rope_offset` (`core._rope_reposition`,
the same operator `initial_am_compaction` already uses). Default False reproduces the historical
phase-corrupted behaviour. The driver **refuses** the flag with `key_mode="freeze"` (no key is ever
installed) and **refuses** it without an explicit `rope_theta` — the counter-rotation is only correct
in the model's own rotary frame, and at θ=1e4 it is worse than doing nothing.

---

## `beta` — `BetaFitter.Config`

Implemented in `cartridges/am/components/beta.py`, which also owns `nnls_projected_gradient` — the
shared projected-gradient NNLS solver that `keys` imports for its own selection fits.

| Field | Default | Notes |
|---|---|---|
| `enabled` | None | Tri-state; was `enable_beta`. **None = OFF.** See below. |
| `fit_scope` | `selected` | Fit β on the selected slots only, or on all of them. Was `beta_fit_scope`. |

### B-SOLVE — β is an independent axis

`BetaFitter.should_fit(key_mode=...)` returns True only when `enabled is True`. The historical
fallback for `enabled=None` was `key_mode != "freeze"`, which (a) silently switched the NNLS path ON
for every key experiment and (b) made "β with frozen keys" — the configuration B-ROUTE actually needs
— expressible only by setting the flag explicitly. Unset now means off, whatever the keys are doing.
Every run in `state/results.csv` used `key_mode="freeze"`, where both rules agree, so no historical
configuration changed behaviour. When `enabled` is unset *and* `key_mode != "freeze"`, the fitter
warns rather than guessing.

### MECH-004 knobs (AM paper App. C.2 "Stabilizing β", Algorithm 3)

All four default to the historical behaviour, so a run that does not set them is bit-identical.

- `beta_box` (None) — symmetric box on the fitted log-weights; the paper uses **3.0** for
  highest-attention keys. None = unbounded above and floored at `log(1e-12) = -27.6`.
- `nnls_iters` (200) — projected-gradient steps. The paper uses **2**; 200 is the historical value.
- `nnls_driver` (None) — lstsq driver for the NNLS warm start. None keeps torch's CUDA default
  `gels`, **which returns NaN for rank-deficient input without raising**; `gelsd` is rank-revealing.
  Note `gelsd` is CPU-only and raises on CUDA input, so the implementation retries the warm start on
  the CPU (`info["warm_start_on_cpu"]`); the matrices are small, so this costs ~1.3× `solve_s`.
- `target_mode` (`residual`) — `residual` subtracts the non-selected slots' mass from the teacher
  target (**this tree's own invention**); `full` fits the selected keys against the full teacher
  mass, as the paper does over all compacted keys. Named `target_mode` rather than `target` because
  `ObjectConfig` already owns a `target` field; it was `beta_target` on the flat config.

The driver forwards the paper's values (3.0 / 2 / `gelsd` / `residual`) unconditionally, so they are
recorded in every `config.yaml`. They are read only when `should_fit` is True, so a β-off run is
unaffected.

---

## `objective` — `ValueObjective.Config`

The closed-form value solve. Implemented in `cartridges/am/components/objective.py`.

| Field | Default | Notes |
|---|---|---|
| `ridge_lambda` | 1e-4 | Base ridge coefficient. |
| `ridge_scale` | `spectral` | How the base becomes the actual λ: `spectral` → `λ·‖X‖₂²` (scale-aware), `frobenius` → `λ·‖X‖_F²/k`, `fixed`/`absolute` → use `ridge_lambda` as-is. |
| `ridge_lambda_min` | 0.0 | Floor on the resulting λ. |
| `delta_weight` | 0.0 | Trust region penalising movement away from the current values. The canonical driver default is **1e-2**, not the dataclass 0.0. Note its pull is fixed at `top_t` rows while `XᵀX` grows with the query count, so its relative strength decays like `1/n` as `max_queries_per_head` rises (MECH-002). |

### `ValueObjective.solve` dispatch

One entry point, three write rules, in this precedence:

1. `oracle_write` → `oracle_teacher_value_write`
2. old-reference guard **or** `delta_weight > 0` → `guarded_sparse_am_value_update`
3. otherwise → `sparse_am_value_update`

Rule 2's `delta_weight > 0` clause matters: the canonical driver sets `DELTA_WEIGHT=1e-2`, so the
guarded rule is the live one even with the guard off. The rules differ in the `stats` dict they
return — rules 1 and 3 key the reconstruction error as `mse`, rule 2 as `mse_new` — which is why the
guard is keyed on whether the *run* supplied an old-reference bank rather than on whether this
particular (layer, head) had old queries.

### Old-reference guard

Stacks old-task (QA) queries into the solve so Phase-1 behaviour is preserved; the scale enters as
`√weight` on those rows. **Inactive in default runs** — all of it is inert unless
`enable_old_reference_guard` is set *and* `old_ref_data_path` is given, and the driver zeroes
`old_reference_weight` when the guard is off.

| Field | Default |
|---|---|
| `enable_old_reference_guard` | False |
| `old_ref_data_path` | None |
| `old_ref_max_examples` | 64 |
| `old_reference_weight` | 1.0 |

### Diagnostic oracle (MECH-001)

`oracle_write` (False) / `oracle_write_assign` (`mass_ranked`).

**Not a write rule — a measurement.** When on, the per-document write skips the closed-form solve
entirely and copies the teacher's own post-RoPE document value vectors into the selected slots. Keys
are untouched, so eval-time routing is exactly the routing the solve faces; this measures the
ceiling of "perfect content in the selected slots". `mass_ranked` pairs the highest-teacher-attention
document rows with the highest-student-attention slots; `sequential` is a positional copy.

---

## Renamed and deleted in the restructure

### Renamed

| Was (flat) | Is now |
|---|---|
| `decoupled_ref_batches` | `queries.ref_batch_limit` |
| `enable_beta` | `beta.enabled` |
| `beta_fit_scope` | `beta.fit_scope` |
| `beta_target` | `beta.target_mode` |

Everything else kept its name and moved onto the sub-config named for its stage; the full map is in
`research_loop/verify_am_restructure.py::FIELD_MAP`, which is what the equivalence harness asserts
against.

### Deleted

| Field | Was | Why it is gone |
|---|---|---|
| `enabled` | False | Vestigial in a dedicated runner: constructing an `AMContinualConfig` *is* enabling AM. |
| `execution_mode` | `per_document` | A dispatch hack, not a hyperparameter. `train_loop` was dead code (no live script used the AM branch in `train()`); `legacy_decoupled` was the only caller of `run_decoupled_tfidf_am_update`. Both paths and both functions are deleted. |
| `target_mode` | `cartridge_plus_doc` | Read only by the two dead execution modes. **Inert on the per-document path**, which always builds the `[cartridge ‖ doc]` teacher. |
| `freeze_keys` | True | Superseded by `keys.key_mode`; only the dead paths read it. |
| `update_interval` | 1 | Optimizer-step cadence — `train_loop` only. |
| `collect_background_stats` | False | Was dead on *this* config; the live flag of the same name is on `SparseCacheFinetuningConfig`. |
| `max_am_steps` | -1 | Defined but never read anywhere in the tree, at any commit. |

Two functions died with them: `apply_am_update_to_cache` (only the removed `train()` branch called
it) and `run_decoupled_tfidf_am_update` (only `execution_mode="legacy_decoupled"` called it).

### Env vars no longer read by the driver

`EPOCHS`, `GLOBAL_BATCH_SIZE`, `MAX_STEPS`, `EVAL_EVERY_N_STEPS`, `SAVE_EVERY_N_STEPS`,
`DISTRIBUTED_BACKEND` (all `TrainConfig` costume), plus `TARGET_MODE`, `AM_EXECUTION_MODE` and
`UPDATE_INTERVAL` (the deleted fields above). Every other variable the shell wrappers pass is
preserved. `AM_REF_BATCH_LIMIT` replaces `AM_DECOUPLED_REF_BATCHES`.

---

## Experiment logging

Weights & Biases was removed from the Phase-2 AM driver
(`examples/qasper2/train/continual_am_sparse.py`) and its wrapper script. Per-run results are still
written to disk: `phase2_summary.json`, `per_document_am_stats.pt`, and the per-document
`am_doc_*.pt` payloads under the run directory. `AMContinualConfig.wandb` is typed `None`, so it
cannot be switched on by accident; the shared `cartridges/train.py` still supports wandb for the
other examples.

---

## Verifying a change to this config

`research_loop/verify_am_restructure.py` compares the current tree against a pre-refactor checkout on
three axes: leaf math bodies are byte-identical, the same environment produces the same field values,
and one full per-document write is numerically identical across four arms (`keys_repos`, `plain`,
`beta`, `oracle`) that between them cover every branch of `ValueObjective.solve`. It needs no GPU.

```bash
git worktree add -f --detach /tmp/am_baseline_tree <pre-refactor-ref>
"$CARTRIDGES_DIR/.venv/bin/python" research_loop/verify_am_restructure.py /tmp/am_baseline_tree
```
