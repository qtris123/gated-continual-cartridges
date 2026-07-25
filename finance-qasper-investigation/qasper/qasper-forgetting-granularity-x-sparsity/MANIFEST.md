# qasper-forgetting-granularity-x-sparsity bundle

Distilled artifacts behind the Llama + Qwen forgetting/acquisition vs.
sparsity figures (ln(ppl) on the y-axis).

## Contents
- `plot_granularity_x_sparsity.py` — script that scrapes eval logs from the
  source `outputs/qasper_forgetting_eval_<model>_*` directories and writes the
  CSVs + PNGs in this bundle. Re-running it requires the raw eval folders
  (NOT included in this bundle); see the `MODELS` list at the top of the file
  for the expected paths.
- `llama_granularity_x_sparsity_summary.csv`,
  `qwen_granularity_x_sparsity_summary.csv` — per-granularity, per-top-k
  perplexity + ln(perplexity) values exactly as plotted, with per-granularity
  baselines and the shared averaged baseline (the dashed line in the figures).
  Schema: `model, granularity, top_k, forgetting_ppl, acquisition_ppl,
  forgetting_log_ppl, acquisition_log_ppl`.
- `llama_granularity_x_sparsity.png`, `qwen_granularity_x_sparsity.png` —
  the figures themselves.
- `README_investigation.md` — original README from
  `qasper-forgetting-investigation/`, describing the folder layout and how
  the plot script is wired up.

## Source eval folders (not in this bundle)
If you ever need to regenerate from scratch, the raw eval logs live in:
- `outputs/qasper_forgetting_eval_llama_granularity-x-sparsity_3273481/`
- `outputs/qasper_forgetting_eval_qwen_per-head-global_granularity-x-sparsity_3315625/`
- `outputs/qasper_forgetting_eval_qwen_per-layer_granularity-x-sparsity_3344506/`
The qwen plot script reads from a symlink-merged view at
`outputs/qasper_forgetting_eval_qwen_granularity-x-sparsity_3315625_3344506/`.
