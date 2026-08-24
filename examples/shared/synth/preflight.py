"""Fail fast before a long self-study run instead of losing batches to it.

Phase-1 synthesis lost 3872 FinQA rows and 256 TechQA rows to two problems that
were only visible after the job had already exited 0:

  * a technote longer than the server's context window, which 400s every batch
    that samples it;
  * a server that answers without `token_id:{id}` tokens, so `top_logprobs` are
    silently dropped and the rows are unusable for `targets="logits"`.

Each check below corresponds to one of those failures. Exits non-zero so a
driver can gate the real run on it.

  python examples/shared/synth/preflight.py --dataset techqa --phase 2 \
      --num-samples 8192 --batch-size 32 --num-top-logprobs 20 \
      --resume-dir data/techqa/synth/p02/self_study-n8192
"""

from __future__ import annotations

import argparse
import asyncio
import glob
import math
import os
import random
import sys
from typing import Optional

import httpx

from cartridges.clients.openai import OpenAIClient

# Bot A asks for 512 completion tokens and Bot B for 1024; leave room for both
# plus the chat wrapper on top of the longest document.
COMPLETION_HEADROOM = 2048

_ok = True


def check(label: str, passed: bool, detail: str = "") -> None:
    global _ok
    mark = "PASS" if passed else "FAIL"
    if not passed:
        _ok = False
    print(f"[{mark}] {label}" + (f" — {detail}" if detail else ""))


def server_max_model_len(base_url: str, model: str) -> Optional[int]:
    root = base_url.rstrip("/")
    try:
        payload = httpx.get(f"{root}/models", timeout=30).json()
    except Exception as exc:
        check("server reachable", False, f"{type(exc).__name__}: {exc}")
        return None
    entries = {e["id"]: e for e in payload.get("data", [])}
    check("server reachable", True, f"{len(entries)} model(s) served")
    if model not in entries:
        check("model served", False, f"{model!r} not in {sorted(entries)}")
        return None
    max_len = entries[model].get("max_model_len")
    check("model served", True, f"{model} (max_model_len={max_len})")
    return max_len


async def probe_logprobs(base_url: str, model: str, k: Optional[int]) -> None:
    client = OpenAIClient.Config(
        model_name=model,
        base_url=base_url,
        api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"),
        return_tokens_as_token_ids=k is not None,
    ).instantiate()
    try:
        resp = await client.chat(
            [[{"role": "user", "content": "Reply with one short sentence."}]],
            temperature=0.0,
            top_logprobs=k,
            max_completion_tokens=16,
        )
    except Exception as exc:
        check("logprob probe", False, f"{type(exc).__name__}: {exc}")
        return
    sample = resp.samples[0]
    if k is None:
        check("generation works", bool(sample.text), repr(sample.text[:40]))
        return
    # Both of these go missing together when the server was launched without
    # --return-tokens-as-token-ids: the ids cannot be parsed, so the top-k block
    # is discarded and every row lands without a teacher distribution.
    check(
        "server returns true token ids (--return-tokens-as-token-ids)",
        sample.token_ids is not None,
        "" if sample.token_ids is not None else "relaunch the server with the flag",
    )
    got = None if sample.top_logprobs is None else sample.top_logprobs.logprobs.shape[1]
    check(f"top_logprobs width == {k}", got == k, f"got {got}")


async def probe_prompt_budget(
    dataset: str, phase: int, model: str, batches: int, max_model_len: Optional[int]
) -> None:
    from transformers import AutoTokenizer

    from examples.shared.synth.self_study_vllm import build_resource_config

    resource = build_resource_config(dataset, phase).instantiate()
    if hasattr(resource, "setup"):
        maybe = resource.setup()
        if asyncio.iscoroutine(maybe):
            await maybe

    tokenizer = AutoTokenizer.from_pretrained(model)
    # SelfStudySynthesizer seeds the global RNG with 82, so replaying it here
    # draws the same document sequence the real run will draw.
    random.seed(82)
    longest = 0
    for _ in range(batches):
        ctx, _seeds = await resource.sample_prompt(1)
        longest = max(longest, len(tokenizer.encode(ctx, add_special_tokens=False)))

    print(f"       longest sampled system prompt over {batches} draws: {longest:,} tokens")
    if max_model_len is None:
        check("prompt fits context window", False, "unknown max_model_len")
        return
    budget = max_model_len - COMPLETION_HEADROOM
    check(
        "prompt fits context window",
        longest <= budget,
        f"{longest:,} vs budget {budget:,} (max_model_len {max_model_len:,} - {COMPLETION_HEADROOM} headroom)",
    )


def report_resume(resume_dir: Optional[str], batches: int) -> None:
    if not resume_dir:
        print(f"       fresh run: {batches} batches to generate")
        return
    import pyarrow.parquet as pq

    files = sorted(glob.glob(os.path.join(resume_dir, "checkpoints", "*.parquet")))
    empty = [f for f in files if pq.ParquetFile(f).metadata.num_rows == 0]
    done = len(files) - len(empty)
    todo = batches - done
    print(
        f"       resume: {done} usable checkpoints, {len(empty)} empty "
        f"(regenerated automatically), {todo} batches to generate"
    )


async def main_async(args: argparse.Namespace) -> int:
    batches = math.ceil(args.num_samples / args.batch_size)
    max_model_len = server_max_model_len(args.base_url, args.model)
    await probe_logprobs(args.base_url, args.model, args.num_top_logprobs)
    await probe_prompt_budget(args.dataset, args.phase, args.model, batches, max_model_len)
    report_resume(args.resume_dir, batches)
    print("\nPREFLIGHT " + ("OK" if _ok else "FAILED"))
    return 0 if _ok else 1


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True)
    p.add_argument("--phase", type=int, required=True)
    p.add_argument("--model", default=os.environ.get("CARTRIDGES_SYNTH_MODEL", "Qwen/Qwen3-4B-Instruct-2507"))
    p.add_argument("--base-url", default=os.environ.get("CARTRIDGES_VLLM_URL", "http://127.0.0.1:8000/v1"))
    p.add_argument("--num-samples", type=int, default=8192)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-top-logprobs", type=int, default=None)
    p.add_argument("--resume-dir", default=None)
    sys.exit(asyncio.run(main_async(p.parse_args())))


if __name__ == "__main__":
    main()
