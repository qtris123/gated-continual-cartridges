# RUNBOOK — AM-sparse Qasper investigation

Ground truth for **how to run things**. Executors read this before composing any
command. Orchestrator reads this to decide what is cheap vs expensive.
When a fact here conflicts with the source, **the source wins** — fix this file.

Repo: `/localhome/local-triv/gated-continual-cartridges_explore` (branch `trivo-explore-research-work`)
Python: `.venv/bin/python` (torch 2.13.0+cu130, 2× GH200 144GB).

## 0. Environment (export at the top of EVERY shell)
```bash
cd /localhome/local-triv/gated-continual-cartridges_explore
export CARTRIDGES_DIR=$PWD
export CARTRIDGES_OUTPUT_DIR=$PWD/outputs
export CARTRIDGES_WANDB_PROJECT=SEACrowd
export CARTRIDGES_WANDB_ENTITY=vqtri-purdue-university
# HF_TOKEN and WANDB_API_KEY are already in the environment; ~/.netrc is logged in.
PY=$PWD/.venv/bin/python
```
Without `CARTRIDGES_DIR` set, `import cartridges` raises immediately.
First CUDA context on this box is **erratic — usually ~1–2 min but sometimes >2 min** (cold JIT);
warm runs are ~1s. **Executors: launch every train/eval as a BACKGROUND job and poll its log file;
never run GPU work synchronously under a short (<300s) timeout — you WILL get false timeouts.**
Do a throwaway warm-up import once per fresh machine before timing anything.

## 1. The task & the two axes ("task performance" here = perplexity)
Qasper 2-stage: **Phase 1 = QA task** builds the cartridge; **Phase 2 = MT task**
continually writes new docs into it. Two eval splits, both scored as **mean cross-entropy
("Eval loss"), lower is better**:
- `data/qasper/eval/qasper_eval_QA.parquet` → **forgetting** (retention of Phase-1 knowledge)
- `data/qasper/eval/qasper_eval_MT.parquet` → **acquisition** (new-doc learning)

✅ **FIXED (2026-07-28):** the wandb `num_elements`/`num_system_and_user_tokens`/
`num_assistant_tokens`/`macro_loss` fields used to stay at their dead-init value (0 / `nan` /
`None`) in `evaluate_perplexity` (`cartridges/train.py`), which made the accompanying
`perplexity` field look like garbage (e.g. `exp(11.29)=80019` for a genuinely-broken July AM
run, with `num_elements=0` as the tell). Root cause fixed: the counters are now actually
incremented. **`Eval loss` (mean-CE) was never affected** — it used the correctly-accumulated
`epoch_loss`/`epoch_denom` even before this fix — but it's still the primary number to read.
Wandb logging is also **back on by default** (`WANDB_DISABLED=0` in the AM wrapper scripts;
dense/sparse-grad scripts never disabled it — only individual one-off `launch_exp*.sh` files
did, via an explicit `WANDB_MODE=disabled` override that new launches should drop).

⚠️ **UNITS: the harness `Eval loss` is mean cross-entropy = ln(perplexity).** Many historical repo
figures (RUNBOOK §8, notes, CSVs) are quoted in **perplexity**, NOT loss. `ppl = exp(loss)`. **Only
ever compare LOSS-to-LOSS from the loop's own runs.** When citing a historical ppl target, convert it
(e.g. ppl 7.8 → loss 2.05; ppl 36 → loss 3.58) before comparing. The loop's optimization signal is LOSS.

⚠️ **`eval_forgetting.py` HANGS after printing `Eval loss`** (prints result, never exits — cost a prior
agent a 78-min stall waiting for exit). Executors: parse the `Eval loss` line from the log, then
**kill the PID**; never `wait` on exit. The eval itself is fast (~a few seconds over 5-6 batches).

⚠️ **`nvidia-smi` is chronically slow/flaky on this box** (often >20s, sometimes hangs). Prefer
`torch.cuda` or `/proc/driver/nvidia/gpus/` for GPU checks; if you must call nvidia-smi, wrap in
`timeout` and tolerate failure. Never let a GPU probe block the loop.

⚠️ **Eval set is TINY (QA n=6, MT n=5 batches).** Deltas < ~0.1-0.2 loss are within noise — don't
over-claim; confirm a "win" with a re-run or a larger eval before trusting it.

## 2. Model & data (fixed for this investigation)
- Model: **Qwen3-4B** (`FlexQwen3ForCausalLM`), `num_tokens=512`, `num_frozen_tokens=1`.
- Phase-1 (QA) synth: `data/qasper/train/qwen_qasper_QA_task_8192.parquet`
- Phase-2 (MT) synth: `data/qasper/train/qwen_qasper_MT_task_8192.parquet`
- Init text: `data/qasper/init_text/qasper_init_512.txt`

## 3. Commands (VERIFY env-var names against source before first use)
The **authoritative env→config mapping** is the parsing block in
`examples/qasper2/train/continual_am_sparse.py` (~lines 73–116). Read it before composing a
run; the table below is a summary, not a substitute.

### (a) Dense self-distillation baseline — Phase 2 (the number to MATCH) — ✅ VALIDATED
Driver: `examples/qasper2/train/baseline_continual.py` (dense gradient, no sparse/AM). Env vars:
`PHASE1_CACHE_PATH, SYNTH_DATA_PATH, EVAL_DATA_PATH, MODEL_NAME(default Llama→override Qwen),
NUM_TOKENS(512), LR(2e-2), EPOCHS(10), GLOBAL_BATCH_SIZE(32), EVAL_EVERY_N_STEPS(15),
DISTRIBUTED_BACKEND(gloo)`. Uses `pydrantic.main` → CLI overrides work (e.g. `max_optimizer_steps=N`).
**Validated launch pattern (smoke-tested; single GPU, gloo, capped):**
```bash
CUDA_VISIBLE_DEVICES=<gpu> PHASE1_CACHE_PATH=outputs/phase1_selfdistill_qwen512/cache_last.pt \
SYNTH_DATA_PATH=data/qasper/train/qwen_qasper_MT_task_8192.parquet \
EVAL_DATA_PATH=data/qasper/eval/qasper_eval_QA.parquet MODEL_NAME=Qwen/Qwen3-4B-Instruct-2507 \
NUM_TOKENS=512 DISTRIBUTED_BACKEND=gloo WANDB_MODE=disabled RUN_NAME=refcart_phase2 \
.venv/bin/torchrun --nproc_per_node=1 --master_port=2953X examples/qasper2/train/baseline_continual.py
```
For the REAL REF-CART run: drop `max_optimizer_steps`, keep EPOCHS=10 (or set MAX horizon), eval BOTH
splits (run once per EVAL_DATA_PATH, or use `eval_forgetting` on the saved ckpt). Grad-accum = ~32
micro-batches per optimizer step (dataloader len ~1999 at GBS 32). Wandb: set `WANDB_MODE=online` +
project/entity env if you want it logged (script attaches WandBConfig; `WANDB_MODE=disabled` for smoke).
> gradient DDP MUST use `DISTRIBUTED_BACKEND=gloo` (NCCL deadlocks on this PCIe box).
> ⚠️ Prints "Done training waiting for final barrier" and may HANG there — but **the checkpoint is
> saved BEFORE the barrier**, so parse the final eval + confirm the saved `cache-step*.pt`/`cache_last.pt`,
> then kill the PID. Don't wait-for-exit. (Same class as the eval hang.)

### (b) AM-sparse Phase 2 (the method we are improving) — single GPU, closed-form
```bash
PHASE1_CACHE_PATH=<phase1 cache_last.pt> \
BG_STATS_PATH=<bg_stats.pt> \
SYNTH_DATA_PATH=data/qasper/train/qwen_qasper_MT_task_8192.parquet \
TOP_T=64 GRANULARITY=per_layer \
bash examples/qasper2/scripts/core/train_continual_am_sparse.sh
```
### (c) Phase-1 AM cache + background stats (if a fresh Phase-1 is needed)
`examples/qasper2/scripts/core/train_initial_am.sh → train/initial_am.py`
(produces `cache_last.pt` + `bg_stats.pt`; `collect_background_stats`).

### (d) Eval (forgetting + acquisition) on any checkpoint
`examples/qasper2/scripts/core/eval_forgetting.sh` (round-robins QA+MT across GPUs) or
`train/eval_forgetting.py` with `CHECKPOINT_PATH=`, `EVAL_DATA_PATH=`. Metric key
`eval_qasper_perplexity/loss`. **Report the `loss` (mean CE), not `perplexity`.**

## 4. The knobs = the four levers (ALL env-driven → NO code edit for a sweep)
**Confirmed** env-var names from `examples/qasper2/train/continual_am_sparse.py` L61-112.
Set them as `VAR=val ... bash examples/qasper2/scripts/core/train_continual_am_sparse.sh` (never bare `env`).

| Lever | ENV VARS (default) |
|---|---|
| **Gating** | `SLOT_SELECTION`∈{tfidf,attention_mass,residual_budget} (tfidf), `USE_IDF`(1), `IDF_PRIOR_WEIGHT`(0.0), `GRANULARITY`∈{global,per_layer,per_head} (per_layer), `TOP_T`(64), `MIN_TOP_T_PER_LAYER`(1), `IDF_SMOOTHING`(1.0), `IDF_TOP_K`(128) |
| **Teacher targets** | `TARGET_MODE`∈{cartridge_plus_doc,self,…} (cartridge_plus_doc), `KEY_MODE`∈{freeze,highest_attention,omp} (freeze), `ENABLE_BETA`(unset), `BETA_FIT_SCOPE`(selected), `QUERIES_PER_BATCH`(all_tokens) |
| **Support allocation** | `TOP_T`, `MIN_TOP_T_PER_LAYER` (residual_budget floor), `MAX_REF_EXAMPLES_PER_DOC`(32) |
| **Regularization** | `RIDGE_LAMBDA`(1e-4), `RIDGE_SCALE`∈{spectral,frobenius,fixed} (spectral), `RIDGE_LAMBDA_MIN`(0.0), `DELTA_WEIGHT`(1e-2), `OLD_REFERENCE_WEIGHT`(1.0), `ENABLE_OLD_REFERENCE_GUARD`(0), `OLD_REF_DATA_PATH` |
| Run/exec | `AM_EXECUTION_MODE`(per_document), `EPOCHS`(10), `GLOBAL_BATCH_SIZE`(32), `MAX_STEPS`(550), `EVAL_EVERY_N_STEPS`(15), `NUM_TOKENS`, `MODEL_NAME`(Qwen/Qwen3-4B-Instruct-2507), `RUN_NAME`, `WANDB_GROUP` |
Config/code: `cartridges/am/finetune.py`, `cartridges/am/{value_solve,core,ranking,key_select}.py`,
`cartridges/sparse_cache_finetuning.py` (`CacheTFIDFRanker`). An EDIT executor is needed ONLY for a
mechanism outside this list (e.g. an attention-mass floor `τ` on the ranker — HYP-G3).

## 5. GPU discipline (2 GPUs — hard cap 2 concurrent training jobs)
- Claim a GPU with the repo's **flock** mechanism: lockfiles under `/tmp/gpu_locks_$USER/`,
  `MASTER_PORT=29500+gpu_idx`. Reuse the auto-claim in the train scripts / `pool_continual_sparse.sh`.
- Gradient DDP → `DISTRIBUTED_BACKEND=gloo` (NCCL deadlocks: PCIe-only, no NVLink).
  The July AM `config.yaml`s say `nccl` — **override to gloo for any multi-GPU gradient run**;
  AM closed-form runs are single-GPU so it's moot for them.

## 6. Guardrails / known foot-guns (from notes/ — do not rediscover)
1. **Never use bare `env VAR=val bash …`** — a `~/.local/bin/env` PATH-shim swallows it (rc=0, no-op).
   Use `VAR=val bash …` or a subshell with `export`.
2. **`kill -INT` on a backgrounded bash is a silent no-op** (SIG_IGN). Use SIGTERM / kill the PGID.
3. **Checkpoints saved are LAST, not BEST** — the cosine tail can regress; eval the intended step.
4. **`max_steps` in the cosine schedule is decoupled** from total steps → LR-floor tail if wrong
   (`MAX_STEPS ≈ 16000/GLOBAL_BATCH_SIZE`; 32→500, 64→250).
5. `FREEZE_KEYS=0` means keys **do** update. Keys drive forgetting → keep `freeze` unless testing keys.
6. pydrantic writes **one UUID run-dir per rank**; `cache_last.pt` may be a dangling symlink.
7. **`eval_forgetting.py` hangs after printing `Eval loss`** — parse the line, then kill the PID;
   never wait-for-exit (§1). 8. **`nvidia-smi` is flaky/slow** — use torch/`/proc` for GPU checks (§1).
9. Eval set is tiny (n=5-6) — treat sub-0.2-loss deltas as noise (§1).
10. **CODE EDITS may not take effect — dual `cartridges` package.** `import cartridges` can resolve to the
    SIBLING repo `/localhome/local-triv/gated-continual-cartridges/cartridges` (via the venv editable install),
    NOT this `_explore` repo, depending on which `.venv`/PYTHONPATH is active. Symptom: your edit to
    `_explore/cartridges/...` is invisible at runtime, OR a driver (explore) passes a new kwarg the imported
    (sibling) config rejects → `Extra inputs are not permitted` at pydantic construction, crashing ALL runs.
    Before running an EDIT-based experiment: (a) set `PYTHONPATH="$CARTRIDGES_DIR:$PYTHONPATH"`, (b) verify with
    `python -c "import cartridges,os; print(os.path.dirname(cartridges.__file__))"` (must be `_explore`), and
    (c) test the config CONSTRUCTS with the new field before a full run. Prefer making new kwargs OPT-IN and
    passed CONDITIONALLY (`hasattr`) so stock runs never break.
11. **β / ENABLE_BETA is numerically broken (deferred).** The NNLS β-fit produces NaN / huge (~66.6) log-weights
    → rank-deficient value-solve (cholesky not-PD or lstsq NaN). Do NOT enable β without a deep numerical fix
    inside `refit_beta_nnls`. (EXP-005/005b/006 all failed.)

## 7. Settled priors — DO NOT re-litigate (spend cycles elsewhere)
- Value-only (`freeze_keys`) preserves Phase-1; **key-value collapses QA to a ~16–19 floor**.
- **Cumulative slot coverage predicts forgetting, Pearson r≈0.91** — finer granularity & larger
  top_t both inflate the footprint. per-head vs per-layer retention is ~a wash.
- Phase-1 AM ridge sweet spot ≈ **λ=2.0 spectral** (recon eval 5.70); rebaking key positions is harmful.
- `freeze` momentum mode is Pareto-dominant among momentum modes.

## 8. Standing numbers (mind the UNITS — loss vs ppl — per §1)
- **VERIFIED Phase-1 self-distilled cartridge** (HF cache, EXP-000-verify, Qwen, this harness):
  **QA loss 2.239 (ppl ~9.4) / MT loss 3.783 (ppl ~44)**. QA≪MT confirms it's the QA Phase-1 cache;
  MT ~44 ppl = the untrained acquisition ceiling. This is the Phase-2 STARTING point + the retention floor.
- Historical (mostly **perplexity**, convert before comparing): Qwen Phase-1 QA ppl ~7.8; MT untrained
  ppl ~36–44. Best Llama TF-IDF sparse P2 (value-only top-64) QA 6.6 / MT 6.9 **ppl**. AM self-match P1 = 3.82.
- Current AM-sparse Phase-2 (Qwen, top32 per_head): QA 7.13 / MT 7.28 **loss** (from logs = loss units).
- **REF-CART (dense self-distillation Phase-2) = the quality bar** — establish locally from the verified
  Phase-1 cache (EXP-000), cross-check wandb `vqtri-purdue-university/SEACrowd` (`qasper_baseline_phase2`).
- **REF-ICL (full-context ceiling)** — measure once (EXP-000b).

## 8b. Open materials / unknowns to resolve early (flag to human if blocking)
- **DENSE self-distilled Phase-1 cache: RESOLVED (human-provided) — on HuggingFace:**
  `qtris123/qwen_qasper-QA-task_8192_512_no-cartridge_10-epochs` (files: `cache_last.pt`, `config.yaml`).
  Download once with `huggingface_hub.hf_hub_download(repo_id, "cache_last.pt")` (HF_TOKEN is set);
  stage it under `outputs/phase1_selfdistill_qwen512/cache_last.pt`. This is Qwen3-4B, 512 slots,
  dense adam, 10 epochs — the self-distilled cartridge to MATCH.
  ✅ **PROVENANCE VERIFIED (EXP-000-verify): this IS the QA Phase-1 cache.** Dual-split eval gave
  QA loss 2.239 (ppl ~9.4) ≪ MT loss 3.783 (ppl ~44) — the QA-specialist / MT-untrained signature.
  The repo NAME is correct; the bundled `config.yaml` (name=MT-task) is a stale mislabel — **ignore it.**
  Cache staged at `outputs/phase1_selfdistill_qwen512/cache_last.pt` (511 trainable + 1 frozen = 512 slots).
- Env-var names: RESOLVED — all lever knobs are env-driven (§4, confirmed from source).
