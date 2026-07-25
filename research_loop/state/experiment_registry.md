# EXPERIMENT REGISTRY

One `### EXP-XXX` block per experiment. Fill BEFORE dispatch (status `dispatched`), complete on ingest.
Format mirrors `.cursor/rules/research-companion.mdc`.

## Template (copy for each new experiment)
```
### EXP-XXX: <short-name>
- Date:
- Research question:
- Hypothesis tested (HYP-ID):
- Variable under test (the ONE thing changed):
- Baseline compared to (EXP-ID / config):
- Code branch+commit:
- Dataset / Model:  Qasper QA→MT / Qwen3-4B
- Config knobs (full):  slot_selection= use_idf= granularity= top_t= target_mode= key_mode=
                        ridge_lambda= ridge_scale= delta_weight= freeze_keys= ...
- Command:
- Metrics reported:  qa_forgetting_loss / mt_acquisition_loss (mean CE) / runtime_s
- Expected:
- Actual:
- Interpretation:
- Confounders considered:
- Status:  dispatched | done | failed
- Follow-up:
- Artifacts:  outputs/<run-dir>/ , logs/<log> , research_loop/results/EXP-XXX/result.json
```

---
<!-- experiments appended below by the orchestrator -->
