"""Eval datasets and mode configuration for intervention experiments.

Two dataset classes:
  OpenEndedEvalDataset — *_original.parquet; short factual answers. Emits both
    the question-only input (for generation) and the full question+answer token
    sequence (for teacher-forced log-perplexity).
  ChoiceGenerateDataset — yes/no or MCQ parquets; strips the baked reasoning
    instruction and appends a terse answer suffix. The model's step-0 logits
    over the label tokens give the score.
"""
from __future__ import annotations

from cartridges.datasets import (
    DataSource,
    GenerateEvalDataset,
    GenerateEvalDatasetElement,
    MODEL_TO_CHAT_TEMPLATE,
    MODELS_WITH_THINKING,
)

MODE_CONFIG = {
    "yesno": {
        "labels": ["Yes", "No"],
        "suffix": "Answer with Yes or No only, no other text.",
        "baked_instruction": (
            "Explain your reasoning, then end your response with 'Answer: Yes' or 'Answer: No'."
        ),
    },
    "mcq": {
        "labels": ["A", "B", "C", "D"],
        "suffix": "Answer with A, B, C, or D only, no other text.",
        "baked_instruction": (
            "Explain your reasoning, then end your response with 'Answer: X' where X "
            "is the letter (A, B, C, or D)."
        ),
    },
    "openended": {
        "labels": None,
        "suffix": None,
        "baked_instruction": None,
    },
}


class OpenEndedEvalDataset(GenerateEvalDataset):
    """For *_original.parquet: user asks a question, assistant answer is a short
    string like '$103.4 billion'. Emits both the question-only input (for
    generation) and the full question+answer token sequence (for teacher-forced
    log-perplexity)."""

    def __getitem__(self, index: int) -> GenerateEvalDatasetElement:
        convo = self.data[index]
        assert len(convo.messages) >= 2
        assert convo.messages[-1].role == "assistant"

        user_msgs = [
            {"role": m.role, "content": m.content}
            for m in convo.messages[:-1]
        ]
        answer = convo.messages[-1].content
        full_msgs = user_msgs + [{"role": "assistant", "content": answer}]

        kwargs = {}
        if self.tokenizer.name_or_path in MODELS_WITH_THINKING:
            kwargs["enable_thinking"] = self.config.cot

        gen_input_ids = self.tokenizer.apply_chat_template(
            user_msgs,
            add_generation_prompt=True,
            return_tensors="pt",
            chat_template=MODEL_TO_CHAT_TEMPLATE.get(self.tokenizer.name_or_path, None),
            **kwargs,
        )
        full_input_ids = self.tokenizer.apply_chat_template(
            full_msgs,
            add_generation_prompt=False,
            return_tensors="pt",
            chat_template=MODEL_TO_CHAT_TEMPLATE.get(self.tokenizer.name_or_path, None),
            **kwargs,
        )
        answer_token_ids = self.tokenizer.encode(answer, add_special_tokens=False)
        answer_start = gen_input_ids.shape[-1]
        answer_end = answer_start + len(answer_token_ids)

        return GenerateEvalDatasetElement(
            input_ids=gen_input_ids,
            prompt=user_msgs,
            answer=answer,
            convo_id=convo.metadata.get("question_id", str(index)),
            metadata={
                **convo.metadata,
                "idx": index,
                "full_input_ids": full_input_ids.flatten().tolist(),
                "answer_start": int(answer_start),
                "answer_end": int(answer_end),
                "answer_token_ids": list(answer_token_ids),
            },
        )


class ChoiceGenerateDataset(GenerateEvalDataset):
    """GenerateEvalDataset that strips the baked reasoning instruction and
    appends a mode-specific terse answer suffix to the last user message."""

    def __init__(self, *args, baked_instruction: str, suffix: str, **kwargs):
        super().__init__(*args, **kwargs)
        self._baked_instruction = baked_instruction
        self._suffix = suffix

    def __getitem__(self, index: int) -> GenerateEvalDatasetElement:
        convo = self.data[index]
        assert len(convo.messages) > 1
        assert convo.messages[-1].role == "assistant"

        msgs = [{"role": m.role, "content": m.content} for m in convo.messages[:-1]]
        for m in reversed(msgs):
            if m["role"] == "user":
                stripped = m["content"].replace(self._baked_instruction, "").rstrip()
                m["content"] = f"{stripped}\n\n{self._suffix}"
                break

        kwargs = {}
        if self.tokenizer.name_or_path in MODELS_WITH_THINKING:
            kwargs["enable_thinking"] = self.config.cot

        input_ids = self.tokenizer.apply_chat_template(
            msgs,
            add_generation_prompt=True,
            return_tensors="pt",
            chat_template=MODEL_TO_CHAT_TEMPLATE.get(self.tokenizer.name_or_path, None),
            **kwargs,
        )

        return GenerateEvalDatasetElement(
            input_ids=input_ids,
            prompt=msgs,
            answer=convo.messages[-1].content,
            convo_id=convo.metadata.get("question_id", str(index)),
            metadata={**convo.metadata, "idx": index},
        )


def parse_eval_args(entries):
    """Parse a list of `name=/path/to.parquet` strings into a dict."""
    out = {}
    for entry in entries or []:
        name, _, path = entry.partition("=")
        if not path:
            raise ValueError(f"--eval must be name=/path, got {entry!r}")
        out[name] = path
    return out


def build_datasets(eval_files: dict, tokenizer, mode: str) -> dict:
    """Build the right dataset class per eval file based on mode."""
    mode_cfg = MODE_CONFIG[mode]
    datasets = {}
    for name, path in eval_files.items():
        cfg = GenerateEvalDataset.Config(data_source=DataSource(path=path, type="local"))
        if mode == "openended":
            datasets[name] = OpenEndedEvalDataset(cfg, tokenizer=tokenizer, seed=42)
        else:
            datasets[name] = ChoiceGenerateDataset(
                cfg,
                tokenizer=tokenizer,
                seed=42,
                baked_instruction=mode_cfg["baked_instruction"],
                suffix=mode_cfg["suffix"],
            )
    return datasets
