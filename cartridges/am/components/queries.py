"""Query stage: the reference queries Q that a write is fitted against.

Three things live here because they are one concern -- turning a corpus of
conversations into the query matrix each (layer, head) solve sees:

- the reference-data helpers that group conversations into documents and build
  a batch_size=1 loader over a subset,
- the accumulators that collect post-RoPE queries (and optional teacher
  attention outputs) off capture hooks,
- ``ReferenceQueries``, the stage object that binds the two.
"""

from __future__ import annotations

import os
import random
import re
import tempfile
from functools import lru_cache
from typing import Literal, Optional

import torch
import torch.nn as nn
from pydrantic import ObjectConfig
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerFast

from cartridges.datasets import DataSource, TrainDataset
from cartridges.models.attention import flex_attention_forward
from cartridges.sparse_cache_finetuning import _apply_rotary_pos_emb
from cartridges.structs import Conversation, read_conversations, write_conversations


# ======================================================================================
# Reference data: conversations -> documents -> a loader over one document's subset
# ======================================================================================
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


def document_key(conversation: Conversation, dataset: str = "qasper") -> str:
    """Return a stable document key despite randomized context subsets.

    QASPER/QuALITY synthesis rows carry a structural ``<title>`` and group by it.
    FinQA/TechQA rows have no structural title and group documents with
    ``<source>`` tags instead; critically, their document bodies (IBM technotes,
    SEC filings) can themselves contain ``<title>``/HTML, so those datasets must
    key on the first ``<source>`` rather than any ``<title>``. FinQA sources are
    ``TICKER/YEAR/page.pdf`` (one company per row, so the ticker is the
    document); TechQA sources are a single technote filename.
    """
    prompt = conversation.system_prompt or ""
    if dataset in ("finqa", "techqa"):
        source = _extract_tag(prompt, "source")
        if source:
            source = source.strip()
            return source.split("/")[0] if "/" in source else source
        return prompt
    title = _extract_tag(prompt, "title")
    return title or prompt


def group_conversations_by_document(
    conversations: list[Conversation],
    dataset: str = "qasper",
) -> dict[str, list[Conversation]]:
    """Group synthesis rows by document, not exact sampled context subset."""
    groups: dict[str, list[Conversation]] = {}
    for convo in conversations:
        key = document_key(convo, dataset)
        groups.setdefault(key, []).append(convo)
    return groups


def group_conversations_by_system_prompt(
    conversations: list[Conversation],
    dataset: str = "qasper",
) -> dict[str, list[Conversation]]:
    """Backward-compatible alias for document-level grouping."""
    return group_conversations_by_document(conversations, dataset)


@lru_cache(maxsize=None)
def _qasper_papers_by_title(topic: str) -> dict[str, "object"]:
    """Title -> QASPER ``Paper``, for one topic or ``"all"``.

    Cached because the loop asks per document and the underlying
    ``load_dataset`` is the expensive part. Titles are unique across the three
    topics, so the ``"all"`` union cannot collide.
    """
    # Local import: pulls in `datasets` and hits the HF cache.
    from cartridges.data.qasper.resources import TOPIC_TO_IDS, QASPERResource

    topics = list(TOPIC_TO_IDS) if topic == "all" else [topic]
    unknown = [t for t in topics if t not in TOPIC_TO_IDS]
    if unknown:
        raise ValueError(
            f"Unknown QASPER topic(s) {unknown}; expected one of "
            f"{sorted(TOPIC_TO_IDS)} or 'all'."
        )

    papers: dict[str, object] = {}
    for name in topics:
        resource = QASPERResource(QASPERResource.Config(topic=name))
        for paper in resource.papers:
            papers[paper.title] = paper
    return papers


def full_paper_prompt(title: str, *, topic: str = "MT") -> str:
    """The document text prefilled as the teacher, for one paper.

    Sourced from the QASPER dataset rather than from the synthesis rows, whose
    system prompts each carry only a RANDOM subset of the paper's sections
    (``qasper/resources.py::sample_prompt``), so coverage cannot depend on how
    many rows a document happens to have.

    The wording mirrors ``qasper.resources.SYSTEM_PROMPT_TEMPLATE`` -- what the
    rows were generated under -- but sections are joined with a bare newline,
    WITHOUT ``PAPER_TEMPLATE``'s ``---Paper Title: ...---`` divider. That divider
    is worth ~5% more tokens, and ``T_doc`` is not cosmetic here: it *is* the
    ``doc_rope_offset`` rotation (``continual/write.py``), so any token-count
    change re-rotates every document key and moves every teacher target.
    ``test_document_prompts_match_golden_hashes`` pins the exact bytes.
    """
    papers = _qasper_papers_by_title(topic)
    paper = papers.get(title)
    if paper is None:
        raise KeyError(
            f"Document title {title!r} is not in QASPER topic {topic!r} "
            f"({len(papers)} papers). The prompt is looked up by the <title> tag "
            "of the synthesis rows, so this usually means the topic does not "
            "match the data path (e.g. a QA parquet with topic='MT'). Set the "
            "topic to match, or use 'all'."
        )
    merged_sections = "\n".join(
        section.text.strip()
        for section in sorted(paper.sections, key=lambda s: s.section_number)
    )
    return (
        "Below is (part of) a scientific paper. Please read it and be prepared "
        "to answer questions. \n"
        "<paper>\n"
        f"<title>{paper.title}</title>\n"
        f"<abstract>{paper.abstract}</abstract>\n"
        "<sections>\n"
        f"{merged_sections}\n"
        "</sections>\n"
        "</paper>\n"
    )


@lru_cache(maxsize=None)
def _quality_articles_by_title(phase: int) -> dict[str, "object"]:
    """Title -> complete QuALITY article for one declared phase."""
    from cartridges.data.quality.resources import (
        PHASE_TO_ARTICLE_IDS,
        load_articles,
    )

    if phase not in PHASE_TO_ARTICLE_IDS:
        raise ValueError(
            f"Unknown QuALITY phase {phase}; expected one of "
            f"{sorted(PHASE_TO_ARTICLE_IDS)}."
        )
    articles = load_articles(PHASE_TO_ARTICLE_IDS[phase])
    by_title: dict[str, object] = {}
    for article in articles.values():
        if article.title in by_title:
            raise ValueError(
                f"Duplicate QuALITY title {article.title!r} in phase {phase}; "
                "title-based synthesis rows would be ambiguous."
            )
        by_title[article.title] = article
    return by_title


def full_quality_prompt(title: str, *, phase: int) -> str:
    """Complete QuALITY story prompt matching the resource used for synthesis."""
    from cartridges.data.quality.resources import SYSTEM_PROMPT_TEMPLATE

    articles = _quality_articles_by_title(phase)
    article = articles.get(title)
    if article is None:
        raise KeyError(
            f"Document title {title!r} is not in QuALITY phase {phase} "
            f"({len(articles)} articles). The configured phase must match the "
            "synthesis parquet."
        )
    return SYSTEM_PROMPT_TEMPLATE.format(story=article.to_string)


@lru_cache(maxsize=None)
def _finqa_pages_by_company(phase: int) -> dict[str, list]:
    """Company ticker -> its FinQA pages for one phase, filename-sorted.

    Cached like the QASPER loader because the compaction loop asks per document
    and ``load_phase`` re-reads the cached JSON splits each call.
    """
    from cartridges.data.finqa.resources import load_phase

    pages, _ = load_phase(phase)
    by_company: dict[str, list] = {}
    for page in pages:
        by_company.setdefault(page.company, []).append(page)
    return {
        company: sorted(company_pages, key=lambda p: p.filename)
        for company, company_pages in by_company.items()
    }


def full_finqa_prompt(company: str, *, phase: int) -> str:
    """The complete FinQA filing prompt for one company in ``phase``.

    Sourced from the FinQA resource, not the synthesis rows, whose system
    prompts each carry only a RANDOM subset of the company's pages
    (``finqa/resources.py::sample_prompt``). Mirrors ``SYSTEM_PROMPT_TEMPLATE``.
    """
    from cartridges.data.finqa.resources import FISCAL_YEAR, SYSTEM_PROMPT_TEMPLATE

    by_company = _finqa_pages_by_company(phase)
    pages = by_company.get(company)
    if pages is None:
        raise KeyError(
            f"Company {company!r} is not in FinQA phase {phase} "
            f"({len(by_company)} companies). The prompt is looked up by the "
            "<source> ticker of the synthesis rows, so the phase must match the "
            "synthesis parquet."
        )
    return SYSTEM_PROMPT_TEMPLATE.format(
        year=FISCAL_YEAR, filings="\n".join(page.text for page in pages)
    )


@lru_cache(maxsize=None)
def _techqa_notes_by_filename(phase: int) -> dict[str, object]:
    """Technote filename -> ``Technote`` for one phase."""
    from cartridges.data.techqa.resources import PHASE_TO_FILENAMES, load_corpus

    if phase not in PHASE_TO_FILENAMES:
        raise ValueError(
            f"Unknown TechQA phase {phase}; expected one of "
            f"{sorted(PHASE_TO_FILENAMES)}."
        )
    documents, _ = load_corpus()
    notes: dict[str, object] = {}
    for name in PHASE_TO_FILENAMES[phase]:
        if name not in documents:
            raise KeyError(f"TechQA technote {name!r} missing from corpus.")
        notes[name] = documents[name]
    return notes


def full_techqa_prompt(filename: str, *, phase: int) -> str:
    """The complete TechQA prompt for one technote in ``phase``."""
    from cartridges.data.techqa.resources import SYSTEM_PROMPT_TEMPLATE

    notes = _techqa_notes_by_filename(phase)
    note = notes.get(filename)
    if note is None:
        raise KeyError(
            f"Technote {filename!r} is not in TechQA phase {phase} "
            f"({len(notes)} technotes). The phase must match the synthesis parquet."
        )
    return SYSTEM_PROMPT_TEMPLATE.format(documents=note.to_string)


def full_document_prompt(
    title: str,
    *,
    dataset: str = "qasper",
    qasper_topic: str = "MT",
    quality_phase: Optional[int] = None,
    phase: Optional[int] = None,
) -> str:
    """Resolve a synthesis document key to the complete teacher prompt.

    ``phase`` carries the five-phase index for the phase-keyed datasets (finqa,
    techqa); ``quality_phase`` is the QuALITY-specific alias kept for
    backward-compatible configs.
    """
    if dataset == "qasper":
        return full_paper_prompt(title, topic=qasper_topic)
    if dataset == "quality":
        if quality_phase is None:
            raise ValueError("quality_phase is required when dataset='quality'")
        return full_quality_prompt(title, phase=quality_phase)
    if dataset == "finqa":
        if phase is None:
            raise ValueError("phase is required when dataset='finqa'")
        return full_finqa_prompt(title, phase=phase)
    if dataset == "techqa":
        if phase is None:
            raise ValueError("phase is required when dataset='techqa'")
        return full_techqa_prompt(title, phase=phase)
    raise ValueError(
        f"Unsupported AM teacher dataset {dataset!r}; expected 'qasper', "
        "'quality', 'finqa', or 'techqa'."
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


# ======================================================================================
# Accumulators: capture hooks -> per-(layer, head) query matrices
# ======================================================================================
class AMQueryAccumulator:
    """Accumulates reference queries from captured hooks across microbatches."""

    def __init__(
        self,
        granularity: Literal["global", "per_layer", "per_head"] = "per_layer",
        queries_per_batch: Literal["last_token", "all_tokens"] = "all_tokens",
        n_layers: int = 1,
        n_kv_heads: int = 1,
        device: torch.device = torch.device("cpu"),
    ):
        self.granularity = granularity
        self.queries_per_batch = queries_per_batch
        self.n_layers = n_layers
        self.n_kv_heads = n_kv_heads
        self.device = device
        # {layer_idx: list of (n_q_heads, n_tokens, head_dim) tensors}
        self._queries: dict[int, list[torch.Tensor]] = {l: [] for l in range(n_layers)}
        self._access_scores_layer: Optional[torch.Tensor] = None

    def reset(self):
        self._queries = {l: [] for l in range(self.n_layers)}
        self._access_scores_layer = None

    def accumulate_from_hooks(
        self,
        captured_queries: dict[int, torch.Tensor],
        cache: nn.Module,
        scaling: float,
        seq_ids: torch.Tensor,
        valid_len: Optional[int] = None,
    ):
        """Extract queries and TF scores from captured post-RoPE queries."""
        with torch.no_grad():
            if valid_len is not None:
                valid_len = min(int(valid_len), seq_ids.shape[0])
            else:
                valid_len = seq_ids.shape[0]

            if self.queries_per_batch == "last_token":
                valid_seq_ids = seq_ids[:valid_len]
                unique_ids = valid_seq_ids.unique()
                token_indices = []
                for uid in unique_ids:
                    positions = (valid_seq_ids == uid).nonzero(as_tuple=True)[0]
                    token_indices.append(positions[-1].item())
                token_indices_t = torch.tensor(token_indices, device=seq_ids.device)
            else:
                token_indices_t = torch.arange(valid_len, device=seq_ids.device)

            for layer_idx, q in captured_queries.items():
                if layer_idx >= self.n_layers:
                    continue
                # q: (1, n_q_heads, seq_len, head_dim)
                q_sel = q[:, :, token_indices_t, :]

                self._queries[layer_idx].append(q_sel.detach().cpu())

                # TF scores for TF-IDF ranking (same as CacheAccessTracker per_layer)
                cache_k = cache.trainable_keys[layer_idx]
                n_kv_heads = cache_k.shape[1]
                n_q_heads = q.shape[1]
                groups = n_q_heads // n_kv_heads

                q_for_score = q_sel

                q_grouped = q_for_score.view(1, n_kv_heads, groups, -1, q_for_score.shape[-1])
                q_mean = q_grouped.mean(dim=2)  # (1, n_kv_heads, n_tokens, d)
                raw_scores = torch.matmul(q_mean, cache_k.transpose(-1, -2)) * scaling
                attn_weights = raw_scores.softmax(dim=-1)
                if self.granularity == "per_head":
                    # Sum batch and query tokens, preserving each KV head.
                    # Shape: (n_kv_heads, n_cache_tokens).
                    layer_score = attn_weights.sum(dim=(0, 2))
                else:
                    layer_score = attn_weights.sum(dim=(0, 1, 2))

                if self._access_scores_layer is None:
                    if self.granularity == "per_head":
                        self._access_scores_layer = torch.zeros(
                            self.n_layers,
                            self.n_kv_heads,
                            layer_score.shape[-1],
                            device=self.device,
                        )
                    else:
                        self._access_scores_layer = torch.zeros(
                            self.n_layers, layer_score.shape[-1], device=self.device
                        )
                self._access_scores_layer[layer_idx] += layer_score.to(self.device)

    def get_access_scores(self) -> torch.Tensor:
        """TF scores shaped for TF-IDF ranker."""
        if self.granularity == "global":
            return self._access_scores_layer.sum(dim=0)
        elif self.granularity == "per_layer":
            return self._access_scores_layer
        else:
            return self._access_scores_layer

    def get_layer_head_queries(
        self,
        layer_idx: int,
        head_idx: int,
        n_q_heads: int,
        n_kv_heads: int,
    ) -> torch.Tensor:
        """Return (n, head_dim) queries for a specific KV head."""
        if not self._queries[layer_idx]:
            return torch.zeros(0, 0)

        groups = n_q_heads // n_kv_heads
        head_qs = []
        for q_batch in self._queries[layer_idx]:
            # q_batch: (1, n_q_heads, n_tokens, head_dim)
            q_h = q_batch[0, head_idx * groups : (head_idx + 1) * groups]
            q_h = q_h.reshape(-1, q_h.shape[-1])  # (n_tokens * groups, d)
            head_qs.append(q_h)

        if not head_qs:
            return torch.zeros(0, 0)
        return torch.cat(head_qs, dim=0)


class AMTargetAccumulator:
    """Accumulates per-query, per-head AM targets in the same order as queries."""

    def __init__(
        self,
        n_layers: int = 1,
        n_kv_heads: int = 1,
    ):
        self.n_layers = n_layers
        self.n_kv_heads = n_kv_heads
        self._targets: dict[int, list[torch.Tensor]] = {l: [] for l in range(n_layers)}

    def reset(self):
        self._targets = {l: [] for l in range(self.n_layers)}

    def accumulate_from_hooks(
        self,
        captured_targets: dict[int, torch.Tensor],
        valid_len: Optional[int] = None,
    ):
        with torch.no_grad():
            for layer_idx, target in captured_targets.items():
                if layer_idx >= self.n_layers:
                    continue
                # target: (1, n_q_heads, n_tokens, head_dim)
                if valid_len is not None:
                    t_sel = target[:, :, :valid_len, :]
                else:
                    t_sel = target
                self._targets[layer_idx].append(t_sel.detach().cpu())

    def get_layer_head_targets(
        self,
        layer_idx: int,
        head_idx: int,
        n_q_heads: int,
        n_kv_heads: int,
    ) -> torch.Tensor:
        """Return (n, head_dim) targets aligned to AMQueryAccumulator order."""
        if not self._targets[layer_idx]:
            return torch.zeros(0, 0)

        groups = n_q_heads // n_kv_heads
        head_targets = []
        for target_batch in self._targets[layer_idx]:
            target_h = target_batch[0, head_idx * groups : (head_idx + 1) * groups]
            target_h = target_h.reshape(-1, target_h.shape[-1])
            head_targets.append(target_h)

        if not head_targets:
            return torch.zeros(0, 0)
        return torch.cat(head_targets, dim=0)


def install_teacher_attention_capture_hooks(model: nn.Module) -> tuple[dict, list]:
    """Capture no-cache teacher attention outputs before the output projection.

    The sparse AM value solve operates per KV head in head-dim space. Capturing
    pre-``o_proj`` outputs keeps the target dimension compatible with the solver.
    """
    captured: dict[int, torch.Tensor] = {}
    handles = []

    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
    elif hasattr(model, "layers"):
        layers = model.layers
    else:
        raise ValueError(f"Cannot find decoder layers in model of type {type(model)}")

    for layer_idx, decoder_layer in enumerate(layers):
        attn_module = decoder_layer.self_attn

        def make_hook(idx, attn_mod):
            def hook_fn(module, args, output):
                batch = args[0] if args else None
                if batch is None:
                    return

                hidden_states = batch.hidden_states
                input_shape = hidden_states.shape[:-1]
                hidden_shape = (*input_shape, -1, attn_mod.head_dim)

                with torch.no_grad():
                    if hasattr(attn_mod, "q_norm"):
                        query_states = attn_mod.q_norm(
                            attn_mod.q_proj(hidden_states).view(hidden_shape)
                        ).transpose(1, 2)
                        key_states = attn_mod.k_norm(
                            attn_mod.k_proj(hidden_states).view(hidden_shape)
                        ).transpose(1, 2)
                    else:
                        query_states = attn_mod.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
                        key_states = attn_mod.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
                    value_states = attn_mod.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

                    cos, sin = batch.position_embeddings
                    query_states = _apply_rotary_pos_emb(query_states, cos, sin)
                    key_states = _apply_rotary_pos_emb(key_states, cos, sin)

                    attn_output = flex_attention_forward(
                        attn_mod,
                        query_states,
                        key_states,
                        value_states,
                        attention_mask=batch.attention_mask,
                        scaling=attn_mod.scaling,
                        mode=batch.mode,
                    )
                    # flex_attention_forward returns (batch, tokens, q_heads, head_dim).
                    captured[idx] = attn_output.transpose(1, 2).detach()

            return hook_fn

        h = attn_module.register_forward_hook(make_hook(layer_idx, attn_module))
        handles.append(h)

    return captured, handles


def collect_reference_queries(
    wrapped_model: nn.Module,
    cache: nn.Module,
    source_dataloader,
    *,
    granularity: str,
    queries_per_batch: str,
    n_layers: int,
    n_kv_heads: int,
    head_dim: int,
    device: torch.device,
    batch_limit: int,
    collect_teacher_targets: bool = False,
) -> tuple[AMQueryAccumulator, Optional[AMTargetAccumulator], int]:
    """Collect reference queries (and optional legacy teacher targets) from a dataloader."""
    query_acc = AMQueryAccumulator(
        granularity=granularity,
        queries_per_batch=queries_per_batch,
        n_layers=n_layers,
        n_kv_heads=n_kv_heads,
        device=device,
    )
    target_acc = (
        AMTargetAccumulator(n_layers=n_layers, n_kv_heads=n_kv_heads)
        if collect_teacher_targets
        else None
    )
    batch_count = 0
    with torch.no_grad():
        for batch in source_dataloader:
            if batch_count >= batch_limit:
                break

            input_ids = batch.input_ids.to(device)
            seq_ids = batch.element_ids.to(device)
            position_ids = batch.position_ids.to(device)
            valid_len = getattr(batch, "valid_len", None)
            if valid_len is None:
                valid_len = input_ids.shape[0]

            wrapped_model(
                input_ids=input_ids,
                seq_ids=seq_ids,
                position_ids=position_ids,
            )
            captured_q = wrapped_model.get_captured_queries()
            if captured_q:
                query_acc.accumulate_from_hooks(
                    captured_q,
                    cache,
                    scaling=head_dim ** -0.5,
                    seq_ids=seq_ids,
                    valid_len=valid_len,
                )

            if target_acc is not None:
                teacher_captured, teacher_handles = install_teacher_attention_capture_hooks(
                    wrapped_model.model
                )
                try:
                    wrapped_model.model(
                        input_ids=input_ids,
                        seq_ids=seq_ids,
                        position_ids=position_ids,
                        use_cache=False,
                        past_key_values=None,
                    )
                    if teacher_captured:
                        target_acc.accumulate_from_hooks(teacher_captured, valid_len=valid_len)
                finally:
                    for handle in teacher_handles:
                        handle.remove()

            batch_count += 1

    return query_acc, target_acc, batch_count


class ReferenceQueries:
    """Draws a document's reference conversations and extracts their queries."""

    class Config(ObjectConfig):
        _pass_as_config = True

        max_ref_examples_per_doc: int = 32
        queries_per_batch: Literal["last_token", "all_tokens"] = "all_tokens"
        max_queries_per_head: int = 64

        # MECH-007: the per-document draw is seeded `doc_idx + seed_offset`;
        # `config.seed` never reaches it, so this is the only stochastic choice
        # left in the write. Must be >= 0 or documents alias onto each other.
        seed_offset: int = 0

        # MECH-006: re-extract the reference queries from the UPDATED cartridge
        # every N layers, since writing layer `l` perturbs the queries at
        # `l+1...`. 0 = one pass for all layers. `..._refresh_doc_kv` also
        # re-prefills the document KV, so the teacher is on-policy too (a
        # separate axis; the paper only re-extracts queries).
        onpolicy_layers: int = 0
        onpolicy_refresh_doc_kv: bool = False

        # Cap on how many batches a collection pass consumes when the caller does
        # not give an explicit one. Was `decoupled_ref_batches`, named for a dead
        # execution mode; it is a general reference-collection cap and also
        # bounds the old-reference bank.
        ref_batch_limit: int = 5

    def __init__(self, config: Config):
        self.config = config

    def build_loader(
        self,
        conversations: list[Conversation],
        tokenizer: PreTrainedTokenizerFast,
        *,
        seed: int,
    ) -> tuple[DataLoader, str]:
        """Draw this document's reference subset and pack it into a loader.

        BOTH consumers take the same seed: `limit_conversations` chooses WHICH
        conversations are drawn and `build_reference_dataloader` seeds the
        packing/ordering of the drawn subset.
        """
        limited = limit_conversations(
            conversations,
            self.config.max_ref_examples_per_doc,
            seed=seed,
        )
        return build_reference_dataloader(limited, tokenizer, seed=seed)

    def draw_seed(self, doc_idx: int) -> int:
        """The sampling seed for document `doc_idx` (MECH-007)."""
        seed = doc_idx + self.config.seed_offset
        assert isinstance(seed, int) and seed >= 0, (
            f"per-document draw seed must be a non-negative int, got {seed!r}"
        )
        return seed

    def collect(
        self,
        wrapped_model: nn.Module,
        cache: nn.Module,
        loader,
        *,
        granularity: str,
        n_layers: int,
        n_kv_heads: int,
        head_dim: int,
        device: torch.device,
        max_batches: Optional[int] = None,
        collect_teacher_targets: bool = False,
    ) -> tuple[AMQueryAccumulator, Optional[AMTargetAccumulator], int]:
        """Run the loader through the model and accumulate its post-RoPE queries.

        ``granularity`` is passed in rather than configured here: the access
        scores this produces are consumed by ``SlotSelector``, and the two must
        agree on the shape. Duplicating the field would let them disagree.
        """
        return collect_reference_queries(
            wrapped_model,
            cache,
            loader,
            granularity=granularity,
            queries_per_batch=self.config.queries_per_batch,
            n_layers=n_layers,
            n_kv_heads=n_kv_heads,
            head_dim=head_dim,
            device=device,
            batch_limit=(
                max_batches if max_batches is not None else self.config.ref_batch_limit
            ),
            collect_teacher_targets=collect_teacher_targets,
        )
