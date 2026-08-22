"""Smoke tests for five-stream self-study resources and the vLLM synthesizer config.

Network: QASPER/QuALITY/TechQA hit Hugging Face on first load; FinQA downloads
GitHub JSON into $CARTRIDGES_DIR/data/finqa/; LongHealth reads the local
benchmark_v5.json cache.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cartridges.clients.base import Client, ClientConfig, ClientResponse, ClientSample
from cartridges.clients.usage import Usage
from cartridges.synthesizers.self_study import SelfStudySynthesizer
from examples.shared.synth.self_study_vllm import (
    DATASETS,
    QASPER_PHASE_TO_TOPIC,
    build_resource_config,
    build_synthesize_config,
    qasper_topic_for_phase,
)

@pytest.fixture(scope="session", autouse=True)
def _cartridges_dir():
    os.environ.setdefault("CARTRIDGES_DIR", str(REPO_ROOT))
    os.environ.setdefault("CARTRIDGES_OUTPUT_DIR", str(REPO_ROOT / "outputs"))


class FakeSynthClient(Client):
    class Config(ClientConfig):
        model_name: str = "fake-synth"
        response: str = "The answer is 42."

    async def chat(self, chats, **kwargs) -> ClientResponse:
        samples = [ClientSample(text=self.config.response) for _ in chats]
        return ClientResponse(
            samples=samples,
            usage=Usage(prompt_tokens=8, completion_tokens=4),
        )


def test_qasper_phase_map_matches_topics():
    assert [qasper_topic_for_phase(p) for p in range(1, 6)] == [
        "QA",
        "MT",
        "SA",
        "ASR",
        "KG",
    ]
    for topic in QASPER_PHASE_TO_TOPIC.values():
        cfg = build_resource_config("qasper", 1, topic=topic)
        assert cfg.topic == topic


def test_build_resource_rejects_bad_dataset_and_phase():
    with pytest.raises(ValueError, match="dataset"):
        build_resource_config("mmlu", 1)
    with pytest.raises(ValueError, match="phase"):
        build_resource_config("finqa", 0)
    with pytest.raises(ValueError, match="phase"):
        build_resource_config("finqa", 6)


@pytest.mark.parametrize("dataset", DATASETS)
def test_build_synthesize_config_wires_one_resource(dataset):
    cfg = build_synthesize_config(
        dataset=dataset,
        phase=1,
        num_samples=4,
        batch_size=2,
        max_num_batches=1,
        client=FakeSynthClient.Config(),
    )
    assert len(cfg.synthesizer.resources) == 1
    assert cfg.num_samples == 4
    assert cfg.batch_size == 2
    assert dataset in cfg.wandb.tags
    assert cfg.synthesizer.num_top_logprobs is None
    assert cfg.batch_retries >= 1
    assert cfg.worker_timeout >= 60


@pytest.mark.parametrize("dataset", DATASETS)
def test_phase1_resource_sample_prompt(dataset):
    resource = build_resource_config(dataset, phase=1).instantiate()
    ctx, seeds = asyncio.run(resource.sample_prompt(batch_size=2))
    assert isinstance(ctx, str) and len(ctx) > 50
    assert len(seeds) == 2
    assert all(isinstance(s, str) and s for s in seeds)

    if dataset == "qasper":
        assert "<paper>" in ctx.lower() or "paper" in ctx.lower()
    elif dataset == "longhealth":
        assert "patient" in ctx.lower() or "medical" in ctx.lower()
    elif dataset == "finqa":
        assert "filing" in ctx.lower() or "<page>" in ctx
    elif dataset == "quality":
        assert "story" in ctx.lower() or "<title>" in ctx
    elif dataset == "techqa":
        assert "document" in ctx.lower() or "<source>" in ctx


@pytest.mark.parametrize("dataset", DATASETS)
def test_phase1_self_study_one_convo_with_fake_client(dataset):
    synth_cfg = SelfStudySynthesizer.Config(
        client=FakeSynthClient.Config(),
        resources=[build_resource_config(dataset, phase=1)],
        tools=[],
        max_rounds=1,
        num_top_logprobs=None,
    )
    synth = synth_cfg.instantiate()

    async def _run():
        await synth.setup()
        try:
            return await synth.sample_convos(batch_idx=0, batch_size=1, total_batches=1)
        finally:
            await synth.cleanup()

    convos = asyncio.run(_run())
    assert len(convos) == 1
    convo = convos[0]
    assert convo.system_prompt
    assert any("42" in (m.content or "") for m in convo.messages)


def test_techqa_drops_notes_over_token_budget():
    from cartridges.data.techqa.resources import (
        Technote,
        approx_token_count,
        documents_within_token_budget,
    )

    short = Technote(filename="short.txt", text="ok " * 50)
    huge = Technote(filename="huge.txt", text="x" * 200_000)
    assert approx_token_count(huge.text) > 30_000
    kept = documents_within_token_budget([short, huge], max_tokens=30_000)
    assert kept == [short]
    assert documents_within_token_budget([huge], max_tokens=30_000) == []


def test_process_batch_retries_then_raises():
    from types import SimpleNamespace
    from cartridges.synthesize import _process_batch_async

    class Boom:
        n = 0

        async def sample_convos(self, *args, **kwargs):
            Boom.n += 1
            raise RuntimeError("Internal Server Error")

    async def _run():
        orig = asyncio.sleep
        asyncio.sleep = lambda *a, **k: orig(0)
        try:
            await _process_batch_async(
                batch_idx=0,
                total_batches=1,
                synthesizer=Boom(),
                config=SimpleNamespace(batch_size=4, num_samples=4, batch_retries=3),
            )
        finally:
            asyncio.sleep = orig

    Boom.n = 0
    with pytest.raises(RuntimeError, match="Internal Server Error"):
        asyncio.run(_run())
    assert Boom.n == 3


def test_process_batch_recovers_after_retry():
    from types import SimpleNamespace
    from cartridges.structs import Conversation
    from cartridges.synthesize import _process_batch_async

    ok = Conversation(messages=[], system_prompt="s", metadata={}, type="todo")

    class Flaky:
        n = 0

        async def sample_convos(self, *args, **kwargs):
            Flaky.n += 1
            if Flaky.n < 2:
                raise RuntimeError("500")
            return [ok]

    async def _run():
        orig = asyncio.sleep
        asyncio.sleep = lambda *a, **k: orig(0)
        try:
            return await _process_batch_async(
                batch_idx=0,
                total_batches=1,
                synthesizer=Flaky(),
                config=SimpleNamespace(batch_size=1, num_samples=1, batch_retries=3),
            )
        finally:
            asyncio.sleep = orig

    Flaky.n = 0
    rows = asyncio.run(_run())
    assert rows == [ok]
    assert Flaky.n == 2
