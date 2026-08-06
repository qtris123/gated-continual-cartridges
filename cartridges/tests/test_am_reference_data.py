"""Tests for AM reference data grouping and loaders."""

from __future__ import annotations

import hashlib

import pytest

from cartridges.am.components.queries import (
    full_paper_prompt,
    group_conversations_by_system_prompt,
    limit_conversations,
)
from cartridges.structs import Conversation

# sha256 of every document prompt the teacher prefills, per topic. `T_doc` IS the
# `doc_rope_offset` rotation (`continual/write.py`), so a prompt that changes by
# even one token re-rotates every document key and moves every teacher target --
# which would silently invalidate every recorded loss. These hashes are the
# operating point the published numbers were measured at; a diff here means the
# prompt changed, not that the test is stale.
GOLDEN_PROMPT_HASHES = {
    "MT": {
        "28b764bdd64fac478aa0b8b48057f2c3c6075c7da08204cdb86b1739bbe608c7",
        "7373d761d4061d9b1c7988ebbfc755154b10bf4ef080d8df89c588ac62f1f6c4",
        "3f32e591b34e9e505cc20f6087d235f2df9eab97fefbd7c01aa89e6a94845008",
        "a457a5ef325b24a1f3b3215417a2ad397d515a1c3fc8ed19bcb81266b368254b",
        "bb8fb9dcdf900e48bbf0585d16ef7b80460ec6637eb6dbf6ad9074ee1b0211b2",
        "78151435aa83c0a73f7632613a73ef5bdd6324345ca84c7f0b9a3dd5753c297a",
        "d70b343149586537c4e4ab5d4c1d584daffde925195cbac0104d8e2316047b0d",
        "a0231be42946bb6127baaed469951489576ca4e87e853a4566ddb9c57f83e595",
        "d660b0cd9a81cf56b74f3b6e00244beb665fd976a6a77f3c83d38e3c65dbe8d0",
        "4e1da3684228bce75d6fc2dc7607f187d032e53646f853199f9de2cc3b71ff57",
        "56715d3665a0527e3311047eab31b08401bb0ba0707fe1f1ca3e444c7654b381",
        "5202df00928d1807c58b33d644f3ede9ca3ebee27e9eff85897d5ea9a59035f8",
        "833deebf22a89ee9c7e1c5c1f9f025ec9955ec07b2bb131a8209772faf0dd34f",
        "4d7823c4c5c92f201a3d85ecb1a376800492aeecd95111f553229f1879b6dec8",
        "83276392be2fa07fc01eb54794530b74eab81752449fb6bded2c53f33110d7de",
        "9910865f23828b47792e9a8bf7d84baa627c1dcc525f714cb808b9f25dce6e5d",
    },
    "QA": {
        "248405a00627e8cc0f3d9ee803f02208a6818b36baf93c198f6a2e402938f58b",
        "4d876b2100c648cd9d0366b9f76a53b69594ba2c1b7b1869f3fb397a0888cd67",
        "177fd934f6d2342422eae7407177327c329970cb4478d4fc772af632dd33bbb0",
        "8f162425bbd0e852c4f059ef443837cc45504e1b90e27c8fa44129400ce45082",
        "b39959717d833b2214c93a964f09eec80bb07469274c635111606ff12051a380",
        "e0f11070f203e30b8e1e30afcf2f8563d353fc8677ba66de68f2e93bd932280f",
        "c69ac38551edc97105fd8ed2a3a90720a9d4f548cc5fd39754b3456f872b0058",
        "0e8bc53dcf921d8a4f6f289e4ef184415802ed00ca179f26e2cd396a334922e0",
        "22b5b5a3de43a01a2cefcfb38378797c0ad924a5325576da8a8dd31a2cdfe338",
        "6f3e90c7dd14daa0de0845d0c3b47f7e366748d77ef581522c938a3f0f62338c",
        "143dac35ac3b738f368f6a8c713113eba8fe82a3e66f510417ac0319ce4e76a6",
        "b9ef323583a285ad1f91071ec295bbdc87db664771a30b9f944f2e09e1c89dab",
        "c700112421c63e8393d0e5bbe842380651a0dd6d951157ea928f5b8932a0f551",
        "feef3386c46dd60461f1332b4bbae1804e3a5b3ba16d9d3391bd0c53e9394963",
        "a50b0b3d6126364149e2085864ec7f60e4aa9ed31082fab96f387f68ed723f37",
        "3f3e5f2f82a3ac904175416e03192ed44a910c41161d7b1bf859f0a40b243ad0",
    },
}


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
    """Rows of one paper group together even though each samples other sections."""
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
    assert len(groups["Same paper"]) == 2


@pytest.mark.parametrize("topic", sorted(GOLDEN_PROMPT_HASHES))
def test_document_prompts_match_golden_hashes(topic):
    from cartridges.am.components.queries import _qasper_papers_by_title

    titles = _qasper_papers_by_title(topic)
    assert len(titles) == 16
    actual = {
        hashlib.sha256(full_paper_prompt(title, topic=topic).encode()).hexdigest()
        for title in titles
    }
    assert actual == GOLDEN_PROMPT_HASHES[topic]


def test_full_paper_prompt_covers_every_section():
    from cartridges.am.components.queries import _qasper_papers_by_title

    title, paper = next(iter(_qasper_papers_by_title("MT").items()))
    prompt = full_paper_prompt(title, topic="MT")
    assert len(paper.sections) > 1
    for section in paper.sections:
        assert f"<section-number>{section.section_number}</section-number>" in prompt


def test_full_paper_prompt_rejects_unknown_title():
    with pytest.raises(KeyError, match="not in QASPER topic"):
        full_paper_prompt("A Paper That Does Not Exist", topic="MT")


def test_full_paper_prompt_rejects_unknown_topic():
    with pytest.raises(ValueError, match="Unknown QASPER topic"):
        full_paper_prompt("anything", topic="NOPE")
