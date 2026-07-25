"""Reference data helpers for AM Phase 2 per-document writes."""

from __future__ import annotations

import random
import re
from typing import Optional

from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerFast

from cartridges.datasets import DataSource, TrainDataset
from cartridges.structs import Conversation, read_conversations


def load_conversations(path: str) -> list[Conversation]:
    """Load conversations from a local parquet/pkl/csv path."""
    return read_conversations(path)


def _extract_tag(text: str, tag: str) -> Optional[str]:
    match = re.search(
        rf"<{tag}>(.*?)</{tag}>",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    return match.group(1).strip() if match else None


def document_key(conversation: Conversation) -> str:
    """Return a stable paper key despite randomized section subsets."""
    prompt = conversation.system_prompt or ""
    title = _extract_tag(prompt, "title")
    return title or prompt


def group_conversations_by_document(
    conversations: list[Conversation],
) -> dict[str, list[Conversation]]:
    """Group synthesis rows by paper title, not exact sampled section context."""
    groups: dict[str, list[Conversation]] = {}
    for convo in conversations:
        key = document_key(convo)
        groups.setdefault(key, []).append(convo)
    return groups


def group_conversations_by_system_prompt(
    conversations: list[Conversation],
) -> dict[str, list[Conversation]]:
    """Backward-compatible alias for document-level grouping."""
    return group_conversations_by_document(conversations)


def canonical_document_prompt(conversations: list[Conversation]) -> str:
    """Merge sampled QASPER section contexts into one canonical paper prompt."""
    if not conversations:
        raise ValueError("Cannot build a document prompt from no conversations")

    prompts = [convo.system_prompt or "" for convo in conversations]
    title = next((_extract_tag(prompt, "title") for prompt in prompts if _extract_tag(prompt, "title")), None)
    abstract = next(
        (_extract_tag(prompt, "abstract") for prompt in prompts if _extract_tag(prompt, "abstract")),
        "",
    )
    if not title:
        return prompts[0]

    sections: dict[str, str] = {}
    for prompt in prompts:
        for section in re.findall(
            r"<section>.*?</section>",
            prompt,
            flags=re.DOTALL | re.IGNORECASE,
        ):
            section_number = _extract_tag(section, "section-number")
            key = section_number or section
            sections.setdefault(key, section.strip())

    def section_sort_key(item: tuple[str, str]):
        key = item[0]
        try:
            return (0, int(key))
        except ValueError:
            return (1, key)

    merged_sections = "\n".join(
        section for _, section in sorted(sections.items(), key=section_sort_key)
    )
    return (
        "Below is (part of) a scientific paper. Please read it and be prepared "
        "to answer questions. \n"
        "<paper>\n"
        f"<title>{title}</title>\n"
        f"<abstract>{abstract}</abstract>\n"
        "<sections>\n"
        f"{merged_sections}\n"
        "</sections>\n"
        "</paper>\n"
    )


def limit_conversations(
    conversations: list[Conversation],
    max_examples: Optional[int],
    seed: int = 0,
) -> list[Conversation]:
    if max_examples is None or max_examples <= 0 or len(conversations) <= max_examples:
        return conversations
    rng = random.Random(seed)
    indices = rng.sample(range(len(conversations)), max_examples)
    return [conversations[i] for i in sorted(indices)]


def build_reference_dataloader(
    conversations: list[Conversation],
    tokenizer: PreTrainedTokenizerFast,
    *,
    packed_seq_length: int = 2048,
    packing_mode: str = "truncate",
    seed: int = 0,
    top_k_logits: int = 20,
    tmp_path: Optional[str] = None,
) -> tuple[DataLoader, str]:
    """Build a batch_size=1 dataloader over the given conversation subset.

    Returns (dataloader, parquet_path). Caller should delete parquet_path when done.
    """
    import os
    import tempfile

    from cartridges.structs import write_conversations

    if tmp_path is None:
        tmp = tempfile.NamedTemporaryFile(suffix=".parquet", delete=False)
        tmp_path = tmp.name
        tmp.close()

    write_conversations(conversations, tmp_path)
    dataset = TrainDataset.Config(
        data_sources=[DataSource(path=tmp_path, type="local")],
        top_k_logits=top_k_logits,
        packed_seq_length=packed_seq_length,
        packing_mode=packing_mode,
    ).instantiate(tokenizer=tokenizer, seed=seed)

    loader = DataLoader(
        dataset, batch_size=1, collate_fn=lambda batch: batch[0], num_workers=0,
    )
    return loader, tmp_path


def cleanup_reference_parquet(path: str) -> None:
    import os

    if path and os.path.exists(path):
        os.unlink(path)


def load_old_reference_bank(
    qa_parquet_path: str,
    max_examples: int = 64,
    seed: int = 0,
) -> list[Conversation]:
    """Load QA parquet rows for optional old-reference guard constraints."""
    conversations = load_conversations(qa_parquet_path)
    return limit_conversations(conversations, max_examples, seed=seed)
