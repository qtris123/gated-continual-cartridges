"""
QASPER self-study synthesis (Tokasaurus + QASPERResource).

Same role as `examples/qasper2/synthesize/self_study.py`, but configured for
QASPER topics (QA, MT, SA). Run from repo root:

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
from pathlib import Path
import sys

import pydrantic
from pydrantic.variables import FormatStringVariable

from cartridges.clients.tokasaurus import TokasaurusClient
from cartridges.synthesize import SynthesizeConfig
from cartridges.synthesizers.self_study import SelfStudySynthesizer
from cartridges.data.qasper.resources import QASPERResource
from cartridges.utils.wandb import WandBConfig


parser = argparse.ArgumentParser(description="Qasper self-study synthesis")

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
parser.add_argument("--topic", type=str, default="QA", help="Topic ID (QA, MT, SA)")
parser.add_argument(
    "--upload-to-wandb",
    action="store_true",
    help="Upload dataset artifact to W&B",
)
parser.add_argument(
    "--run-name-prefix",
    type=str,
    default="qasper_self_study",
    help="Prefix for run name / output folder pattern",
)

args, remaining = parser.parse_known_args()
sys.argv = [sys.argv[0]] + remaining

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
           QASPERResource.Config(
                topic=args.topic, # QA, MT, SA
                seed_prompts=["structuring", "summarization", "question", "use_case", "creative"],
            )
        ],
    ),
    num_samples=args.num_samples,
    batch_size=args.batch_size,
    max_num_batches_in_parallel=args.max_num_batches,
    name=FormatStringVariable(
        f"{args.run_name_prefix}_{{synthesizer.client.model_name}}_n{{num_samples}}"
    ),
    run_id=FormatStringVariable("{name}"),
    output_dir=os.environ.get("CARTRIDGES_OUTPUT_DIR", "."),
    wandb=WandBConfig(tags=["qasper", "synthesis", "self_study"]),
    upload_to_wandb=args.upload_to_wandb,
    save_wandb_preview=False,
    upload_to_hf=False,
    hf_repo_id=None,
)


if __name__ == "__main__":
    pydrantic.main([config])
