# GLOSSARY — terms, mechanisms and how to reproduce anything in this investigation

**Audience:** a human picking this up cold. Every term this loop invented is defined here, every code
mechanism it added is listed with its flag and default, and §4 shows how to reproduce any experiment.
When a definition here conflicts with the code, **the code wins** — fix this file.

---
## 1. Metrics and units

| term | meaning |
|---|---|
| **QA loss / "forgetting"** | mean cross-entropy on `data/qasper/eval/qasper_eval_QA.parquet` after Phase-2. Lower = less forgetting. **Units are ln(perplexity)**, not perplexity. |
| **MT loss / "acquisition"** | mean CE on `qasper_eval_MT.parquet`. Lower = more acquired. |
| **`Eval loss`** | the number to read from any eval log. **Never** use the `perplexity` field. |
| **the bar** | dense self-distillation cartridge @4ep: **QA 2.3721 / MT 1.8725**. |
| **the target / PASS** | `gradient_steps = 0` **and** QA ≤ 2.52 **and** MT ≤ 2.02 (bar + 0.15), verified. |
| **the floor** | untouched Phase-1 cartridge: **QA 2.2388 / MT 3.7825**. QA below this = positive backward transfer. |
| **`gradient_steps`** | optimizer steps taken **on the cartridge**. A *diagnostic* backward pass (e.g. Fisher scoring) is **not** an optimizer step and leaves this at 0 — but always report its cost. |

## 2. Terms this investigation introduced

| term | definition | where measured |
|---|---|---|
| **`mass_on_S`** | eval-time attention mass landing on the slots that were written (S = union of per-document selections). The quantity that bounds any value-only write. **Already computed at `cartridges/am/value_solve.py:146` and `finetune.py:628`** — it went unrecorded for the loop's first nine experiments. | DIAG-ROUTING, all later runs |
| **k-curve** | loss as a function of **k = number of documents written so far**, read off the `cache-after-doc-*.pt` snapshots the stock path already saves (`SAVE_AFTER_EACH_DOCUMENT=1`). Revealed that the reported k=16 endpoint is **not** the best point of a run. | DIAG-SEQUENCE, DIAG-KEYCURVE, DIAG-CONTROLCURVE |
| **write-ceiling oracle** | replace the solved values with the **teacher's own KV for the document** — the best any value-only write could do. Answers "is the solve the problem, or the write?" | ORACLE-WRITE (MECH-001) |
| **content-free control** | write documents from the *other* topic and evaluate MT. Any gain is content-free by construction. Measured 28.3–73.2% of the MT gain as content-free — **and full-context ICL shows 53.3%**, so it is a benchmark property, not an AM defect. | DIAG-CONTENT, D0-ICL |
| **paired resolution** | the 95% interval of a **paired** difference between two arms on the same examples: **±0.049 MT / ±0.054 QA**. The inherited "±0.15 noise band" was ~3× too wide for paired comparisons (and ~1.7× too *narrow* for independent ones). **Use paired.** | DIAG-NOISE |
| **ρ_key** | fraction of MT *query* energy outside the top-r eigenspace of the QA query second moment `Q₀` — i.e. is there a direction in 128-dim query space where MT lives and QA does not. Sat **at its held-out control floor** everywhere. | DIAG-KEYSPACE |
| **selectivity (MT/QA mass ratio)** | ratio of `mass_on_S` for MT queries vs QA queries. **1.04–1.05** for every mechanism tried, vs the untouched Phase-1 cartridge's own **1.0812**. β provably cannot move it (β is query-independent). | MECH-BETA, MECH-KEYS |
| **survival** | fraction of a document's written slots still holding its values after all 16 documents. **Mean 4.7%** — but this does *not* mean 4.7% information survival (DIAG-SEQUENCE refuted that). | DIAG-OVERWRITE |
| **frozen-snapshot pin** | `git archive HEAD cartridges examples` into `/tmp`, `PYTHONPATH`-pinned, probed **from `/tmp`**. Mandatory when any worker is editing source. Load-bearing four times this session. | RUNBOOK §9c-bis |

## 3. Code mechanisms added (all opt-in, default off, bit-identical when off)

| id | env flag | default | what it does | files |
|---|---|---|---|---|
| **MECH-001** | `AM_ORACLE_WRITE`, `AM_ORACLE_WRITE_ASSIGN` | off | writes the teacher's own document values into the selected slots instead of solving | `am/value_solve.py::oracle_teacher_value_write`, `am/finetune.py`, `am/continual.py` |
| **MECH-002** | `MAX_QUERIES_PER_HEAD` | **64** | exposes the reference-query cap that was hard-coded (paper uses 16k–50k). Also makes `mass_on_S` emit from the `DELTA_WEIGHT` branch | `am/finetune.py`, `am/value_solve.py` |
| **MECH-003** | `AM_ROPE_THETA` | **10000.0** | the rotary base used in the AM path. **The model's true value is 5000000** — the default is the historical (buggy) one, kept so old runs reproduce | `am/{core,teacher,key_select,finetune}.py` |
| **MECH-004** | `AM_BETA_BOX`, `AM_NNLS_ITERS`, `AM_NNLS_DRIVER`, `AM_BETA_TARGET` | 3.0 / 2 / `gelsd` / residual | boxed NNLS for β + CPU `gelsd` warm start, and decouples `_should_fit_beta` from `key_mode` | `am/key_select.py`, `am/finetune.py` |
| **MECH-005** | `AM_KEY_REPOSITION` | off | RoPE counter-rotation when a **document** key is installed into a cartridge slot. **Refuses to run without an explicit `AM_ROPE_THETA`** (rotating at 1e4 is worse than not correcting) | `am/key_select.py`, `am/finetune.py` |
| **MECH-006** | `AM_ONPOLICY_LAYERS`, `AM_ONPOLICY_DOCKV` | off | on-policy layer-sequential re-extraction of reference queries (group size 4) | `am/finetune.py`, `am/continual.py` |
| **MECH-007** | `AM_SEED_OFFSET` | 0 | offsets the per-document reference draw (`continual.py:150` seeds with the constant `doc_idx`). **Without this, seed variation is impossible** — `pydrantic.main` is never reached in `per_document` mode, so `seed=N` on argv is silently ignored | `am/continual.py` |

**Foot-gun that has bitten twice:** any new kwarg must be passed **conditionally** (`hasattr` guard). An
unconditional one crashed every AM run via the sibling-`cartridges` import path (RUNBOOK §6.10).

## 4. Reproducing an experiment

Every experiment has a bundle at `research_loop/results/<ID>/result.json` containing its exact
`command`, its wandb URL(s), and the checkpoint paths it evaluated. Raw per-layer arrays live in
`research_loop/state/diagnostics/<ID>.json`. `research_loop/state/results.csv` is the one-row-per-result
index (columns include `board_entry` and `wandb_run_url`).

```bash
# 0. environment (RUNBOOK §0)
cd /localhome/local-triv/gated-continual-cartridges_explore
export CARTRIDGES_DIR=$PWD CARTRIDGES_OUTPUT_DIR=$PWD/outputs
export CARTRIDGES_WANDB_PROJECT=SEACrowd CARTRIDGES_WANDB_ENTITY=vqtri-purdue-university

# 1. pin imports if anyone might be editing source (RUNBOOK §9c-bis)
mkdir -p /tmp/snap && git archive HEAD cartridges examples | tar -x -C /tmp/snap
export PYTHONPATH=/tmp/snap:$PYTHONPATH
cd /tmp && python -c "import cartridges,os;print(os.path.dirname(cartridges.__file__))"; cd -

# 2. the current best gradient-free point (MECH-005), wandb ON
CUDA_VISIBLE_DEVICES=0 \
PHASE1_CACHE_PATH=outputs/phase1_selfdistill_qwen512/cache_last.pt \
SYNTH_DATA_PATH=data/qasper/train/qwen_qasper_MT_task_8192.parquet \
TOP_T=32 GRANULARITY=per_layer SLOT_SELECTION=tfidf USE_IDF=0 \
KEY_MODE=highest_attention AM_KEY_REPOSITION=1 AM_ROPE_THETA=5000000 \
ENABLE_BETA=0 RIDGE_LAMBDA=1e-4 RIDGE_SCALE=spectral DELTA_WEIGHT=1e-2 \
MAX_QUERIES_PER_HEAD=64 \
WANDB_DISABLED=0 RUN_NAME=repro_best WANDB_GROUP=REPRO \
bash examples/qasper2/scripts/core/train_continual_am_sparse.sh

# 3. evaluate — BOTH splits, and note the k-curve: the k=16 endpoint is NOT the best point.
#    k=12 is the minimum; its snapshot is cache-after-doc-011-*.pt in the run dir.
CHECKPOINT_PATH=<run_dir>/cache-after-doc-011-*.pt \
EVAL_DATA_PATH=data/qasper/eval/qasper_eval_MT.parquet \
WANDB_DISABLED=0 RUN_NAME=repro_best_MT \
.venv/bin/python examples/qasper2/train/eval_forgetting.py
# parse the `Eval loss` line, then KILL THE PID — the script hangs after printing (RUNBOOK §1)
```

**Expected:** k=12 → QA 1.9560 / MT 2.2720; k=16 → QA 2.0349 / MT 2.3305. The solve is **deterministic**
— six nominally identical runs gave byte-identical caches — so a re-run that differs means the config
differs. To vary genuinely, set `AM_SEED_OFFSET` (MECH-007).

### wandb conventions
`WANDB_DISABLED=0` always. `RUN_NAME=<ID>_<slug>`, `WANDB_GROUP=<board-entry-id>` (e.g. `B-ROUTE`,
`B-GATE`), `WANDB_NOTES="<variable under test> vs <baseline>"`, tag `diagnostic` for non-training runs.
**A GPU run with no wandb URL is an invalid result and gets re-run** (RUNBOOK §0b).
⚠️ Known gap: `eval_forgetting.py`'s **ICL branch** (`_run_icl`) never initialises wandb — it only prints.

## 5. Where state lives
`state/bottleneck_board.md` (the live causal picture — read this first) · `state/results.csv` (numbers)
· `state/experiment_registry.md` (EXP/DIAG/MECH entries) · `state/literature_ledger.md` (LIT-001…027,
the imported mechanisms) · `state/mechanism_registry.md` (MECH-001…007, what was built and its verdict)
· `state/diagnostics/` (raw per-layer arrays) · `notes/` + `JOURNAL.md` (durable narrative).
