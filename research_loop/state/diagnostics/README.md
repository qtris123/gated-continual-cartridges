# DIAGNOSTICS — raw dumps from W1 MEASURE workers

One file per diagnostic: `<ID>.json` (scalars, summaries) and optionally `<ID>.npz` (arrays).
Keep them small — summarize large tensors (per-layer means/quantiles), never dump full KV caches here.

Every file must carry, at minimum:
```json
{
  "id": "DIAG-003",
  "board_entry": "B-ROUTE",
  "question": "what this measurement was dispatched to answer",
  "config": {"...": "the exact env/config the measured run used"},
  "baseline_reproduced": {"qa": 0.0, "mt": 0.0, "matches_baseline": true},
  "measurements": {"...": "the localizing numbers"},
  "wandb_run_url": "..."
}
```
`baseline_reproduced` matters: instrumentation must not change numerical behaviour. If the
instrumented run does not reproduce its baseline losses, the diagnostic is invalid.

Index new dumps in `../bottleneck_board.md` under the board entry they serve.
