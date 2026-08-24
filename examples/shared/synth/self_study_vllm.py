"""Self-study synthesis against a vLLM OpenAI-compatible server.

One entry point for the five continual-learning streams (QASPER, LongHealth,
FinQA, QuALITY, TechQA). Each `--phase` maps onto that dataset's resource
(QASPER topics QA/MT/SA/ASR/KG; LongHealth four patients; FinQA/QuALITY/TechQA
the committed LPT splits).

OFF-POLICY: the document chunk is the system prompt from `SelfStudySynthesizer`;
the server hosts the plain base model. AM does not need stored logprobs.

Tokasaurus is unusable on this host (flashinfer vs CUDA 13.3).

  python examples/shared/synth/self_study_vllm.py \
    --dataset longhealth --phase 1 \
    --num-samples 8192 --batch-size 32 --max-num-batches 64

  python examples/shared/synth/self_study_vllm.py \
    --dataset qasper --phase 4 \
    --num-samples 8192 --batch-size 32 --max-num-batches 64
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional, Sequence

from examples.shared.paths import ROOT as _REPO_ROOT

import pydrantic
from pydrantic.variables import FormatStringVariable

from cartridges.clients.base import ClientConfig
from cartridges.clients.openai import OpenAIClient
from cartridges.data.finqa.resources import FinQAResource
from cartridges.data.longhealth.resources import LongHealthResource
from cartridges.data.qasper.resources import QASPERResource, TOPIC_TO_IDS
from cartridges.data.quality.resources import QuALITYResource
from cartridges.data.resources import Resource, SEED_TYPES
from cartridges.data.techqa.resources import TechQAResource
from cartridges.synthesize import SynthesizeConfig
from cartridges.synthesizers.self_study import SelfStudySynthesizer
from cartridges.utils.wandb import WandBConfig

DATASETS = ("qasper", "longhealth", "finqa", "quality", "techqa")
QASPER_PHASE_TO_TOPIC = {1: "QA", 2: "MT", 3: "SA", 4: "ASR", 5: "KG"}
DEFAULT_SEED_PROMPTS: list[SEED_TYPES] = [
    "structuring",
    "summarization",
    "question",
    "use_case",
    "creative",
]


def qasper_topic_for_phase(phase: int) -> str:
    if phase not in QASPER_PHASE_TO_TOPIC:
        raise ValueError(
            f"QASPER phase must be one of {sorted(QASPER_PHASE_TO_TOPIC)}, got {phase}"
        )
    topic = QASPER_PHASE_TO_TOPIC[phase]
    if topic not in TOPIC_TO_IDS:
        raise ValueError(f"QASPER topic {topic!r} is missing from TOPIC_TO_IDS")
    return topic


def build_resource_config(
    dataset: str,
    phase: int = 1,
    *,
    topic: Optional[str] = None,
    seed_prompts: Optional[Sequence[SEED_TYPES]] = None,
    docs_per_prompt: Optional[int] = None,
) -> Resource.Config:
    """Return the Resource.Config for one phase of one stream."""
    if dataset not in DATASETS:
        raise ValueError(f"dataset must be one of {DATASETS}, got {dataset!r}")
    if phase not in range(1, 6):
        raise ValueError(f"phase must be 1..5, got {phase}")
    seeds = list(seed_prompts) if seed_prompts is not None else list(DEFAULT_SEED_PROMPTS)

    if dataset == "qasper":
        resolved = topic or qasper_topic_for_phase(phase)
        return QASPERResource.Config(topic=resolved, seed_prompts=seeds)
    if dataset == "longhealth":
        return LongHealthResource.Config(phase=phase, seed_prompts=seeds)
    if dataset == "finqa":
        return FinQAResource.Config(phase=phase, seed_prompts=seeds)
    if dataset == "quality":
        return QuALITYResource.Config(phase=phase, seed_prompts=seeds)
    n_docs = 1 if docs_per_prompt is None else docs_per_prompt
    max_prompt = int(os.environ.get("TECHQA_MAX_PROMPT_TOKENS", "60000"))
    return TechQAResource.Config(
        phase=phase,
        seed_prompts=seeds,
        docs_per_prompt=n_docs,
        max_prompt_tokens=max_prompt,
    )


def build_synthesize_config(
    *,
    dataset: str,
    phase: int = 1,
    topic: Optional[str] = None,
    client: Optional[ClientConfig] = None,
    model: str = "Qwen/Qwen3-4B-Instruct-2507",
    base_url: str = "http://127.0.0.1:8000/v1",
    num_samples: int,
    batch_size: int,
    max_num_batches: int,
    prob_thinking: float = 0.2,
    max_rounds: int = 1,
    run_name: Optional[str] = None,
    upload_to_wandb: bool = False,
    seed_prompts: Optional[Sequence[SEED_TYPES]] = None,
    docs_per_prompt: Optional[int] = None,
    num_top_logprobs: Optional[int] = None,
    return_token_ids: bool = False,
    max_completion_tokens_b: int = 2048,
) -> SynthesizeConfig:
    if client is None:
        client = OpenAIClient.Config(
            model_name=model,
            base_url=base_url,
            api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"),
            return_tokens_as_token_ids=return_token_ids,
        )
    resource = build_resource_config(
        dataset,
        phase,
        topic=topic,
        seed_prompts=seed_prompts,
        docs_per_prompt=docs_per_prompt,
    )
    if dataset == "qasper":
        label = getattr(resource, "topic", topic or qasper_topic_for_phase(phase))
    else:
        label = f"p{phase}"
    name = run_name or f"{dataset}_{label}_self_study"
    return SynthesizeConfig(
        synthesizer=SelfStudySynthesizer.Config(
            client=client,
            max_rounds=max_rounds,
            prob_thinking=prob_thinking,
            tools=[],
            resources=[resource],
            num_top_logprobs=num_top_logprobs,
            max_completion_tokens_b=max_completion_tokens_b,
        ),
        num_samples=num_samples,
        batch_size=batch_size,
        max_num_batches_in_parallel=max_num_batches,
        worker_timeout=int(os.environ.get("SYNTH_WORKER_TIMEOUT", str(20 * 60))),
        batch_retries=int(os.environ.get("SYNTH_BATCH_RETRIES", "3")),
        name=FormatStringVariable(
            f"{name}_{{synthesizer.client.model_name}}_n{{num_samples}}"
        ),
        run_id=name,
        output_dir=os.environ.get("CARTRIDGES_OUTPUT_DIR", "."),
        wandb=WandBConfig(tags=[dataset, "synthesis", "self_study", label]),
        upload_to_wandb=upload_to_wandb,
        save_wandb_preview=False,
        upload_to_hf=False,
        hf_repo_id=None,
    )


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Self-study synthesis for the five phase streams (vLLM)"
    )
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument("--phase", type=int, default=1, help="Phase 1..5")
    parser.add_argument(
        "--topic",
        type=str,
        default=None,
        help="QASPER-only override (QA, MT, SA, ASR, KG). Default is the phase map.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=os.environ.get("CARTRIDGES_SYNTH_MODEL", "Qwen/Qwen3-4B-Instruct-2507"),
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=os.environ.get("CARTRIDGES_VLLM_URL", "http://127.0.0.1:8000/v1"),
    )
    parser.add_argument("--num-samples", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--max-num-batches", type=int, required=True)
    parser.add_argument("--prob-thinking", type=float, default=0.2)
    parser.add_argument("--max-rounds", type=int, default=1)
    parser.add_argument("--upload-to-wandb", action="store_true")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument(
        "--docs-per-prompt",
        type=int,
        default=None,
        help="TechQA only: max technotes per system prompt (default 1).",
    )
    parser.add_argument(
        "--resume-dir",
        type=str,
        default=os.environ.get("RESUME_DIR"),
        help="Reuse this run directory so nonempty checkpoints are kept",
    )
    parser.add_argument(
        "--num-top-logprobs",
        type=int,
        default=(
            int(os.environ["SYNTH_NUM_TOP_LOGPROBS"])
            if os.environ.get("SYNTH_NUM_TOP_LOGPROBS")
            else None
        ),
        help=(
            "Store top-k teacher logprobs per token. Omit for off-policy AM runs; "
            "set (e.g. 20) to match rows consumed by TrainDataset targets='logits'."
        ),
    )
    parser.add_argument(
        "--max-completion-tokens-b",
        type=int,
        default=int(os.environ.get("SYNTH_MAX_COMPLETION_TOKENS_B") or 2048),
        help=(
            "Cap on the assistant turn. At the old 1024, 47%% of QuALITY answers hit "
            "the cap and ended mid-sentence; 2048 leaves headroom for long-form streams."
        ),
    )
    parser.add_argument(
        "--return-token-ids",
        action="store_true",
        default=os.environ.get("SYNTH_RETURN_TOKEN_IDS", "0") == "1",
        help="Ask the server for 'token_id:N' tokens so top_logprobs carry true vocab ids.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _parse_args(argv)
    config = build_synthesize_config(
        dataset=args.dataset,
        phase=args.phase,
        topic=args.topic,
        model=args.model,
        base_url=args.base_url,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        max_num_batches=args.max_num_batches,
        prob_thinking=args.prob_thinking,
        max_rounds=args.max_rounds,
        run_name=args.run_name,
        upload_to_wandb=args.upload_to_wandb,
        docs_per_prompt=args.docs_per_prompt,
        num_top_logprobs=args.num_top_logprobs,
        return_token_ids=args.return_token_ids,
        max_completion_tokens_b=args.max_completion_tokens_b,
    )
    if args.resume_dir:
        from pathlib import Path

        config.run_dir = Path(args.resume_dir)
        config.run()
        return
    # pydrantic.main re-parses sys.argv; strip our flags so leftover keys
    # don't collide with SynthesizeConfig fields.
    sys.argv = [sys.argv[0]]
    pydrantic.main([config])


if __name__ == "__main__":
    main()
