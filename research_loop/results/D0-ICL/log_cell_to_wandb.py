"""D0-ICL: parse the ICL metrics line from a cell log and record it in wandb.

`eval_forgetting.py`'s ICL branch (`_run_icl`) does NOT initialise wandb itself — it
only prints `Eval loss - <float>` and `[icl] {metrics dict}`. The loop's wandb mandate
(RUNBOOK §0b) still applies, and this worker is under a no-source-edit rule, so the run
record is created here from the exact numbers the harness printed.

Env in: CELL, TOPIC, SPLIT, LOG, OUT, ELAPSED, GPU, SNAPHEAD.
"""

import ast
import json
import os
import re

CELL = os.environ["CELL"]
TOPIC = os.environ["TOPIC"]
SPLIT = os.environ["SPLIT"]
LOG = os.environ["LOG"]
OUT = os.environ["OUT"]
ELAPSED = int(os.environ.get("ELAPSED", "0"))
GPU = os.environ.get("GPU", "?")
SNAPHEAD = os.environ.get("SNAPHEAD", "?")

text = open(LOG, encoding="utf-8", errors="replace").read()

# take the LAST occurrence, so re-runs appending to the same log parse correctly
loss_matches = re.findall(r"^Eval loss - ([0-9.eE+-]+)\s*$", text, flags=re.M)
icl_matches = re.findall(r"^\[icl\] (\{.*\})\s*$", text, flags=re.M)
if not loss_matches or not icl_matches:
    raise SystemExit(f"D0-ICL: could not parse Eval loss / [icl] metrics from {LOG}")

loss = float(loss_matches[-1])
metrics = ast.literal_eval(icl_matches[-1])
assert abs(metrics["loss"] - loss) < 1e-9, (metrics["loss"], loss)

resolved = re.findall(r"^RESOLVED cartridges -> (.*)$", text, flags=re.M)
record = {
    "cell": CELL,
    "icl_topic": TOPIC,
    "eval_split": SPLIT,
    "eval_data_path": f"data/qasper/eval/qasper_eval_{SPLIT}.parquet",
    "harness": "examples/qasper2/train/eval_forgetting.py EVAL_MODE=icl (-> eval_icl.run_icl_loss_eval)",
    "eval_loss_mean_ce": loss,
    "elapsed_s": ELAPSED,
    "gpu": GPU,
    "snapshot_head": SNAPHEAD,
    "resolved_cartridges": resolved[-1] if resolved else None,
    **{k: v for k, v in metrics.items()},
}

import wandb

run = wandb.init(
    project=os.environ.get("CARTRIDGES_WANDB_PROJECT", "SEACrowd"),
    entity=os.environ.get("CARTRIDGES_WANDB_ENTITY"),
    name=f"D0-ICL_{TOPIC}on{SPLIT}",
    group="REF-ICL",
    tags=["diagnostic", "icl", "D0-ICL", f"cell:{CELL}"],
    notes=(
        f"D0-ICL re-ruler: full-context ICL ({TOPIC} papers) scored on "
        f"qasper_eval_{SPLIT}.parquet with the loop's own harness "
        f"(eval_forgetting.py EVAL_MODE=icl), vs the board's 1.9734/1.8960 "
        f"measured with experiments/qasper_loss_benchmark.evaluate_loss_chunked."
    ),
    config=record,
)
wandb.log(
    {
        "eval_qasper_perplexity/loss": loss,
        "eval_qasper_perplexity/perplexity": metrics["perplexity"],
        "diag/icl_loss": loss,
        "diag/icl_perplexity": metrics["perplexity"],
        "diag/num_target_tokens": metrics["num_target_tokens"],
        "diag/num_examples": metrics["num_examples"],
        "diag/system_prompt_tokens": metrics["system_prompt_tokens"],
        "diag/max_sequence_tokens": metrics["max_sequence_tokens"],
        "diag/elapsed_s": ELAPSED,
    }
)
record["wandb_run_url"] = run.url
record["wandb_run_id"] = run.id
run.finish()

path = os.path.join(OUT, f"cell_{TOPIC}on{SPLIT}.json")
with open(path, "w") as f:
    json.dump(record, f, indent=2)
print(f"[d0-icl] wrote {path}")
print(f"[d0-icl] WANDB_URL {run.url}")
print(f"[d0-icl] CELL {CELL} loss={loss}")
