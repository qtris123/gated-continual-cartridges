# ACTIVE CONTEXT — resume-from-this-file-alone

> Orchestrator: read this first every cycle; rewrite the top block every cycle.
> This is the single source of truth for "where are we and what's next."

## HEADLINE (update every cycle)
- **Status:** NOT STARTED — cycle 0 not yet run.
- **Goal:** AM/TF-IDF sparse Phase-2 (Qwen3-4B) to MATCH self-distillation on Qasper, on both
  QA-forgetting loss and MT-acquisition loss.
- **Target (stop condition):** AM-sparse QA-loss ≤ baseline_QA + 0.15 AND MT-loss ≤ baseline_MT + 0.15.
- **Baseline (to be established, EXP-000):** dense self-distillation Phase-2 QA/MT loss = _TBD_.
- **Current best AM-sparse:** QA 7.13 / MT 7.28 loss (Qwen, top32 per_head, cartridge_plus_doc) —
  from `logs/phase2_from_compaction.log`, to be reproduced cleanly in EXP-001.
- **Gap to target:** _TBD after EXP-000/001._

## BUDGET
- Soft budget: run unattended until target met or ~40 cycles / ~48 GPU-hours, whichever first.
  (Human may adjust in this file.)

## IN-FLIGHT (experiments dispatched, awaiting result bundles)
- (none yet)

## NEXT ACTIONS (what the next cycle should do)
1. **Cycle 0 = anchoring.** Dispatch:
   - TRAIN EXP-000: (a) download the human-provided Phase-1 self-distilled cache from HF
     `qtris123/qwen_qasper-QA-task_8192_512_no-cartridge_10-epochs` → stage at
     `outputs/phase1_selfdistill_qwen512/cache_last.pt`; (b) **VERIFY provenance** — eval it on QA+MT
     splits; a real QA Phase-1 cache = low QA (~5-8) / high MT (~33). If MT is also low it's the wrong
     artifact → set EXP-000 blocked + escalate here (name/config mismatch, RUNBOOK §8b). (c) if
     verified, run `baseline_continual.py` Phase-2 (dense, gloo) from it → the self-distillation
     Phase-2 baseline QA/MT loss.
   - TRAIN EXP-001: reproduce the current AM-sparse Phase-2 canonical config end-to-end through the
     SAME eval harness, so EXP-000 vs EXP-001 are apples-to-apples.
   - RESEARCH EXP-002: pull the historical dense Phase-2 run from wandb (SEACrowd) to cross-check
     EXP-000; and read the AM paper's gating/target sections for lever ideas.
2. Once anchored, begin **Lever 1 (gating)** from `backlog.md` — the stated core problem
   ("TF-IDF selects slots with insufficient attention mass").

## DECISIONS LOCKED (by the human, do not revisit)
- Signal = perplexity/eval-loss only (no inference-server task-accuracy).
- Model = Qwen3-4B. Baseline = re-run locally + cross-check wandb.
- Full autonomy on branch `trivo-explore-research-work` only; may commit here; never touch other branches.
