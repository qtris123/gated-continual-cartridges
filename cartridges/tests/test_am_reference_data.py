"""Tests for AM reference data grouping and loaders."""

from __future__ import annotations

from cartridges.am_reference_data import (
    canonical_document_prompt,
    group_conversations_by_system_prompt,
    limit_conversations,
)
from cartridges.structs import Conversation


def _convo(system_prompt: str, user: str = "q") -> Conversation:
    return Conversation(
        messages=[
            Conversation.Message(content=user, role="user", token_ids=[1, 2]),
            Conversation.Message(content="a", role="assistant", token_ids=[3, 4]),
        ],
        system_prompt=system_prompt,
        metadata={},
        type="todo",
    )


def test_group_conversations_by_system_prompt():
    a = _convo("paper-a")
    b = _convo("paper-b")
    c = _convo("paper-a", user="q2")
    groups = group_conversations_by_system_prompt([a, b, c])
    assert len(groups) == 2
    assert len(groups["paper-a"]) == 2
    assert len(groups["paper-b"]) == 1


def test_limit_conversations():
    convos = [_convo(f"p{i}") for i in range(10)]
    limited = limit_conversations(convos, max_examples=3, seed=0)
    assert len(limited) == 3


def test_groups_section_subsets_by_paper_title():
    prompt_a = (
        "<paper><title>Same paper</title><abstract>A</abstract><sections>"
        "<section><section-number>0</section-number><paragraphs>x</paragraphs></section>"
        "</sections></paper>"
    )
    prompt_b = (
        "<paper><title>Same paper</title><abstract>A</abstract><sections>"
        "<section><section-number>1</section-number><paragraphs>y</paragraphs></section>"
        "</sections></paper>"
    )
    groups = group_conversations_by_system_prompt([_convo(prompt_a), _convo(prompt_b)])
    assert list(groups) == ["Same paper"]
    canonical = canonical_document_prompt(groups["Same paper"])
    assert "<section-number>0</section-number>" in canonical
    assert "<section-number>1</section-number>" in canonical
