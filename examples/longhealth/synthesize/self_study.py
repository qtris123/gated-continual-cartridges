"""
LongHealth self-study synthesis (Tokasaurus + LongHealthResource).

Same role as `experiments/synthesize/self_study.py`, but for synthetic clinical notes
instead of 10-K text. Run from repo root:

  python examples/longhealth/synthesize/self_study.py \\
    --model Qwen/Qwen3-4B \\
    --num-samples 8192 --batch-size 16 --max-num-batches 16

Environment:
  CARTRIDGES_TOKASAURUS_URL  Tokasaurus base URL (default http://localhost:8000)
  CARTRIDGES_OUTPUT_DIR      Output root for run directories
"""

from __future__ import annotations

import argparse
import os
import sys

import pydrantic
from pydrantic.variables import FormatStringVariable

from cartridges.clients.tokasaurus import TokasaurusClient
from cartridges.data.longhealth.resources import LongHealthResource
from cartridges.synthesize import SynthesizeConfig
from cartridges.synthesizers.self_study import SelfStudySynthesizer
from cartridges.utils.wandb import WandBConfig

_DEFAULT_PATIENTS = [
    "patient_01",
    "patient_02",
    "patient_03",
    "patient_04",
    "patient_05",
    "patient_06",
    "patient_07",
    "patient_08",
    "patient_09",
    "patient_10",
]
_DEFAULT_SEED_PROMPTS = [
    "structuring",
    "summarization",
    "question",
    "use_case",
    "creative",
]


def _parse_patient_ids(s: str) -> list[str]:
    return [p.strip() for p in s.split(",") if p.strip()]


parser = argparse.ArgumentParser(description="LongHealth self-study synthesis")
parser.add_argument(
    "--patient-ids",
    type=str,
    default=",".join(_DEFAULT_PATIENTS),
    help="Comma-separated LongHealth patient ids (default: patient_11..patient_20)",
)
parser.add_argument(
    "--max-notes-per-prompt",
    type=int,
    default=5,
    help="Max clinical notes sampled per prompt (LongHealthResource)",
)
parser.add_argument(
    "--min-notes-per-prompt",
    type=int,
    default=1,
    help="Min clinical notes sampled per prompt",
)
parser.add_argument(
    "--max-chars-per-note",
    type=int,
    default=None,
    help="Truncate each note to this many chars (omit for full notes)",
)
parser.add_argument(
    "--seed-prompts",
    nargs="+",
    default=_DEFAULT_SEED_PROMPTS,
    help="Seed prompt types passed to LongHealthResource",
)
parser.add_argument(
    "--model",
    type=str,
    default=os.environ.get("CARTRIDGES_SYNTH_MODEL", "Qwen/Qwen3-4B"),
    help="HF model id served by Tokasaurus",
)
parser.add_argument(
    "--tokasaurus-url",
    type=str,
    default=os.environ.get("CARTRIDGES_TOKASAURUS_URL", "http://localhost:8000"),
    help="Tokasaurus server URL",
)
parser.add_argument("--num-samples", type=int, required=True)
parser.add_argument("--batch-size", type=int, required=True)
parser.add_argument("--max-num-batches", type=int, required=True)
parser.add_argument(
    "--prob-thinking",
    type=float,
    default=0.2,
    help="Probability of chain-of-thought style generation",
)
parser.add_argument("--max-rounds", type=int, default=1)
parser.add_argument(
    "--upload-to-wandb",
    action="store_true",
    help="Upload dataset artifact to W&B",
)
parser.add_argument(
    "--run-name-prefix",
    type=str,
    default="longhealth_self_study",
    help="Prefix for run name / output folder pattern",
)

args, remaining = parser.parse_known_args()
sys.argv = [sys.argv[0]] + remaining

patient_ids = _parse_patient_ids(args.patient_ids)
patients_key = "-".join(p.replace("patient_", "p") for p in patient_ids[:3])
if len(patient_ids) > 3:
    patients_key += f"_plus{len(patient_ids) - 3}"

client = TokasaurusClient.Config(
    url=args.tokasaurus_url,
    model_name=args.model,
)

config = SynthesizeConfig(
    synthesizer=SelfStudySynthesizer.Config(
        client=client,
        max_rounds=args.max_rounds,
        prob_thinking=args.prob_thinking,
        tools=[],
        resources=[
            LongHealthResource.Config(
                patient_ids=patient_ids,
                max_notes_per_prompt=args.max_notes_per_prompt,
                min_notes_per_prompt=args.min_notes_per_prompt,
                max_chars_per_note=args.max_chars_per_note,
                seed_prompts=args.seed_prompts,
            )
        ],
    ),
    num_samples=args.num_samples,
    batch_size=args.batch_size,
    max_num_batches_in_parallel=args.max_num_batches,
    name=FormatStringVariable(
        f"{args.run_name_prefix}_{{synthesizer.client.model_name}}_{patients_key}_n{{num_samples}}"
    ),
    run_id=FormatStringVariable("{name}"),
    output_dir=os.environ.get("CARTRIDGES_OUTPUT_DIR", "."),
    wandb=WandBConfig(tags=["longhealth", "synthesis", "self_study"]),
    upload_to_wandb=args.upload_to_wandb,
    save_wandb_preview=False,
    upload_to_hf=False,
    hf_repo_id=None,
)


if __name__ == "__main__":
    pydrantic.main([config])
