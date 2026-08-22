"""Shared ICL (full-context) loss-eval library.

Single home for the full-context / in-context-learning evaluation used by both
`eval_forgetting.py` (EVAL_MODE=icl) and
`examples/shared/benchmark/qasper_loss_benchmark.py`.

ICL differs from the cartridge eval only in the FORWARD PASS: the topic's papers
are placed in the prompt (77-103k tokens) and chunk-prefilled, because they will
not fit the single forward the canonical cartridge eval uses. The per-token
SCORER is the same soft cross-entropy against the teacher top-k
(`-p_teacher(x) * log q_model(x)`) as `cartridges/train.py::evaluate_perplexity`
and `eval_forgetting.py` — so ICL and cartridge numbers land on the same ruler.

This module lives under `examples/` and imports only from `cartridges/`; it does
NOT modify the core library.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from cartridges.cache import AttnConfig, TrainableCache
from cartridges.data.qasper.resources import QASPERResource
from cartridges.datasets import MODEL_TO_MESSAGE_CONVERTER
from cartridges.models import FlexLlamaForCausalLM, FlexQwen3ForCausalLM, HFModelConfig
from cartridges.structs import Conversation, read_conversations

QASPER_ICL_SYSTEM_TEMPLATE = """\
Please reference the scientific papers included below to answer questions about them.

{content}
"""


def build_qasper_system_prompt(
    topic: str,
    *,
    tokenizer,
    max_context_tokens: Optional[int] = None,
) -> str:
    """Build the full-paper ICL system prompt for a topic (e.g. 'QA', 'MT', 'QA+MT')."""
    topics = topic.split("+")
    panels = [
        QASPERResource(QASPERResource.Config(topic=panel_topic)).to_string()
        for panel_topic in topics
    ]
    if len(topics) == 1:
        ctx_text = panels[0]
    else:
        ctx_text = "\n\n".join(
            f'<topic-panel topic="{panel_topic}">\n{panel}\n</topic-panel>'
            for panel_topic, panel in zip(topics, panels)
        )
    if max_context_tokens is not None:
        ctx_text = tokenizer.decode(
            tokenizer.encode(ctx_text)[:max_context_tokens],
            add_special_tokens=False,
            max_length=999_999_999,
            truncation=True,
        )
    return QASPER_ICL_SYSTEM_TEMPLATE.format(content=ctx_text)


def _model_head_dim(model) -> int:
    if hasattr(model.config, "head_dim"):
        return int(model.config.head_dim)
    return model.config.hidden_size // model.config.num_attention_heads


def _model_device(model) -> torch.device:
    return next(model.parameters()).device


def _make_prompt_cache(
    model,
    cartridge: Optional[TrainableCache] = None,
    device: Optional[torch.device] = None,
) -> TrainableCache:
    if cartridge is not None:
        return cartridge
    dev = device or _model_device(model)
    return TrainableCache(
        config=AttnConfig(
            n_layers=model.config.num_hidden_layers,
            n_heads=model.config.num_key_value_heads,
            head_dim=_model_head_dim(model),
        ),
    ).to(dev)


def load_model_and_tokenizer(model_name: str, device: str, *, multi_gpu: bool = False):
    model_cls = FlexQwen3ForCausalLM if "qwen" in model_name.lower() else FlexLlamaForCausalLM
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    load_kwargs: dict[str, Any] = {}
    if multi_gpu:
        load_kwargs["device_map"] = "auto"
    model = HFModelConfig(
        pretrained_model_name_or_path=model_name,
        model_cls=model_cls,
        load_kwargs=load_kwargs,
    ).instantiate()
    if multi_gpu:
        model = model.to(torch.bfloat16)
    else:
        model = model.to(device).to(torch.bfloat16)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model, tokenizer


def evaluate_loss_chunked(
    *,
    model,
    cartridge: Optional[TrainableCache],
    tokenizer,
    eval_path: Path,
    system_prompt: str,
    device: str,
    desc: str,
    prefill_chunk_size: int = 2048,
) -> dict[str, Any]:
    """Score assistant tokens with full ICL context via chunked generate-mode prefill.

    Per-token loss is `-p_teacher(x) * log q_model(x)` over the teacher top-k —
    identical to the canonical `evaluate_perplexity` scorer.
    """
    conversations = read_conversations(str(eval_path))
    converter = MODEL_TO_MESSAGE_CONVERTER[tokenizer.name_or_path.lower()]
    model_dev = _model_device(model)

    total_loss = torch.tensor(0.0, device=model_dev)
    total_tokens = torch.tensor(0, device=model_dev)
    max_sequence_tokens = 0

    with torch.inference_mode():
        for convo in tqdm(conversations, total=len(conversations), desc=desc, leave=False):
            messages = [
                Conversation.Message(
                    role="system",
                    content=system_prompt,
                    token_ids=None,
                    top_logprobs=None,
                ),
                *convo.messages,
            ]
            element = converter(messages, retokenize=True, tokenizer=tokenizer)
            max_sequence_tokens = max(max_sequence_tokens, len(element.input_ids))

            kv_cache = _make_prompt_cache(model, cartridge, device=model_dev)
            if cartridge is not None:
                kv_cache.clear()

            target_idxs = element.topk_token_idxs
            target_ids = element.topk_token_ids
            target_logprobs = element.topk_logprobs
            assistant_start = int(target_idxs.min().item())

            last_logits: Optional[torch.Tensor] = None
            pos = 0
            while pos < assistant_start:
                end = min(pos + prefill_chunk_size, assistant_start)
                chunk_ids = element.input_ids[pos:end]
                chunk_len = len(chunk_ids)
                chunk_seq_ids = torch.zeros(chunk_len, dtype=torch.long)
                chunk_pos_ids = torch.arange(pos, pos + chunk_len, dtype=torch.long)

                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                    outputs = model(
                        input_ids=chunk_ids.to(model_dev),
                        seq_ids=chunk_seq_ids.to(model_dev),
                        position_ids=chunk_pos_ids.to(model_dev),
                        past_key_values=kv_cache,
                        use_cache=True,
                        mode="generate",
                    )
                last_logits = outputs.logits[0, -1, :].contiguous()
                pos = end

            for tidx, tid, hard_logprob in zip(target_idxs, target_ids, target_logprobs):
                tidx = int(tidx.item())
                tid = int(tid.item())

                if tidx == assistant_start:
                    logits = last_logits
                else:
                    prev_id = element.input_ids[tidx - 1 : tidx]
                    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                        outputs = model(
                            input_ids=prev_id.to(model_dev),
                            seq_ids=torch.zeros(1, dtype=torch.long, device=model_dev),
                            position_ids=torch.tensor([tidx - 1], dtype=torch.long, device=model_dev),
                            past_key_values=kv_cache,
                            use_cache=True,
                            mode="generate",
                        )
                    logits = outputs.logits[0, -1, :].contiguous()

                pred_logp = F.log_softmax(logits.float(), dim=-1)[tid]
                ce = -hard_logprob.to(model_dev).exp() * pred_logp
                total_loss += ce
                total_tokens += 1

            kv_cache.clear()

    loss = float((total_loss / total_tokens).item())
    return {
        "loss": loss,
        "perplexity": math.exp(loss),
        "num_target_tokens": int(total_tokens.item()),
        "num_examples": len(conversations),
        "max_sequence_tokens": max_sequence_tokens,
        "system_prompt_tokens": len(tokenizer.encode(system_prompt, add_special_tokens=False)),
        "context_mode": "full_topic_panel_chunked",
        "prefill_chunk_size": prefill_chunk_size,
    }


def run_icl_loss_eval(
    *,
    model_name: str,
    eval_path: str,
    topic: str,
    device: str = "cuda",
    prefill_chunk_size: int = 2048,
    max_context_tokens: Optional[int] = None,
    desc: Optional[str] = None,
) -> dict[str, Any]:
    """Full-context ICL loss eval on `eval_path` with the `topic` papers in context.

    No cartridge is used (raw model + full-paper prompt). Returns the metrics dict
    (mean-CE `loss`, `perplexity`, `num_examples`, context sizes, ...).
    """
    model, tokenizer = load_model_and_tokenizer(model_name, device)
    system_prompt = build_qasper_system_prompt(
        topic, tokenizer=tokenizer, max_context_tokens=max_context_tokens
    )
    return evaluate_loss_chunked(
        model=model,
        cartridge=None,
        tokenizer=tokenizer,
        eval_path=Path(eval_path),
        system_prompt=system_prompt,
        device=device,
        desc=desc or f"icl|{topic}",
        prefill_chunk_size=prefill_chunk_size,
    )
