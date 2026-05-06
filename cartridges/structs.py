from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal, Optional, TypedDict

from cartridges.clients.base import FlatTopLogprobs

# Appended to MCQ / yes-no eval CSV user prompts; ChoiceGenerateDataset strips these verbatim.
CHOICE_MCQ_BAKED = (
    "Explain your reasoning, then end your response with 'Answer: X' where X "
    "is the letter (A, B, C, or D)."
)
CHOICE_YESNO_BAKED = (
    "Explain your reasoning, then end your response with 'Answer: Yes' or 'Answer: No'."
)
MCQ_SUFFIX = "\n\n" + CHOICE_MCQ_BAKED
YESNO_SUFFIX = "\n\n" + CHOICE_YESNO_BAKED


class MessageDict(TypedDict):
    """This is simply a convenience type for typehints for a message dictionary
    compatible with OpenAI-apis and tokenizer.apply_chat_template.

    It differs from Message, which is a dataclass that also has fields for token_ids and
    top_logprobs.
    """

    role: Literal["user", "assistant", "system"]
    content: str


@dataclass
class Conversation:
    messages: list[Conversation.Message]
    system_prompt: str
    metadata: dict
    type: Optional[str] = None

    @dataclass
    class Message:
        content: str
        role: Literal["user", "assistant", "system"]
        token_ids: Optional[List[int]]

        # Sparse dictionary of top logprobs for each token
        top_logprobs: Optional[FlatTopLogprobs] = None

        def to_message_dict(self) -> MessageDict:
            return {"content": self.content, "role": self.role}

    def _repr_html_(self) -> str:
        import markdown

        html = """
        <link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
        <div class='context-convo p-4'>
        """
        for message in self.messages:
            if message.role == "user":
                role_class = "bg-blue-100 text-blue-800"
            else:
                role_class = "bg-green-100 text-green-800"
            role_display = f"<strong style='font-size: 1.5em;'>{message.role.capitalize()}</strong>"
            content_html = markdown.markdown(message.content)
            html += f"""
            <div class='p-2 my-2 rounded {role_class}'>
                {role_display} {content_html}
            </div>
            """
        html += "</div>"
        return html

    def to_html(self) -> str:
        return self._repr_html_()

    @staticmethod
    def from_dict(row: dict) -> Conversation:
        return Conversation(
            messages=[
                Conversation.Message(
                    content=message["content"],
                    role=message["role"],
                    token_ids=message["token_ids"],
                    top_logprobs=(
                        FlatTopLogprobs(**message["top_logprobs"])
                        if message["top_logprobs"] is not None
                        else None
                    ),
                )
                for message in row["messages"]
            ],
            system_prompt=row["system_prompt"],
            metadata=row["metadata"],
            type=row["type"],
        )


def write_conversations(conversations: list[Conversation], path: str):
    path_str = str(path)
    if path_str.endswith(".parquet"):
        _conversations_to_parquet(conversations, path)
    elif path_str.endswith(".pkl"):
        _conversations_to_pkl(conversations, path)
    else:
        raise ValueError(f"Unsupported file extension: {path_str}")


def read_conversations(path: str) -> list[Conversation]:
    path_str = str(path)
    if path_str.endswith(".parquet"):
        return _conversations_from_parquet(path)
    elif path_str.endswith(".pkl"):
        return _conversations_from_pkl(path)
    elif path_str.lower().endswith(".csv"):
        return _conversations_from_csv(path_str)
    else:
        raise ValueError(f"Unsupported file extension: {path_str}")


def _conversations_to_parquet(conversations: list[Conversation], path: str):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from dataclasses import asdict

    rows = (asdict(row) for row in conversations)
    table = pa.Table.from_pylist(list(rows))
    pq.write_table(table, path, compression="snappy")


def _conversations_from_parquet(path: str) -> list[Conversation]:
    import pandas as pd

    rows = pd.read_parquet(path).to_dict(orient="records")
    return [Conversation.from_dict(row) for row in rows]


def _conversations_to_pkl(conversations: list[Conversation], path: str):
    """For backwards compatibility, we will eventually only support parquet as it is
    roughly half the size of pkl."""
    import pickle

    with open(path, "wb") as f:
        pickle.dump(conversations, f)


def _conversations_from_pkl(path: str) -> list[Conversation]:
    """For backwards compatibility, we will eventually only support parquet as it is
    roughly half the size of pkl."""
    import pickle

    with open(path, "rb") as f:
        data = pickle.load(f)

    if isinstance(data, dict) and "rows" in data:
        # backwards compatibility
        return data["rows"]
    else:
        return data


def _csv_eval_two_turn(*, user_text: str, answer: str, metadata: dict) -> Conversation:
    return Conversation(
        system_prompt="",
        messages=[
            Conversation.Message(
                content=user_text,
                role="user",
                token_ids=None,
                top_logprobs=None,
            ),
            Conversation.Message(
                content=str(answer),
                role="assistant",
                token_ids=None,
                top_logprobs=None,
            ),
        ],
        metadata=metadata,
        type="continual_eval",
    )


def _detect_csv_eval_kind(columns: list[str]) -> str:
    cols = set(columns)
    if "mcq_question" in cols and "option_a" in cols:
        return "mcq"
    if "yes_no_question" in cols:
        return "yes_no"
    if "original_question" in cols and "original_answer" in cols:
        return "original"
    raise ValueError(
        "Unrecognized eval CSV schema. Expected MCQ columns "
        "(mcq_question, option_a..d, answer, ...), yes/no (yes_no_question, answer, ...), "
        "or open-ended (original_question, original_answer, ...). "
        f"Got columns: {columns}"
    )


def _format_mcq_csv_row(row, *, doc_source: str) -> Conversation:
    q = row["mcq_question"]
    options = (
        f"\nA) {row['option_a']}\nB) {row['option_b']}"
        f"\nC) {row['option_c']}\nD) {row['option_d']}"
    )
    user_text = q + options + MCQ_SUFFIX
    metadata = {
        "category": row["category"],
        "question_id": row["id"],
        "question_type": "mcq",
        "doc_source": doc_source,
        "original_question": row["original_question"],
        "original_answer": row["original_answer"],
        "options": {
            "A": row["option_a"],
            "B": row["option_b"],
            "C": row["option_c"],
            "D": row["option_d"],
        },
    }
    return _csv_eval_two_turn(user_text=user_text, answer=row["answer"], metadata=metadata)


def _format_yesno_csv_row(row, *, doc_source: str) -> Conversation:
    user_text = row["yes_no_question"] + YESNO_SUFFIX
    metadata = {
        "category": row["category"],
        "question_id": row["id"],
        "question_type": "yes_no",
        "doc_source": doc_source,
        "original_question": row["original_question"],
        "original_answer": row["original_answer"],
    }
    return _csv_eval_two_turn(user_text=user_text, answer=row["answer"], metadata=metadata)


def _format_original_csv_row(row, *, doc_source: str) -> Conversation:
    metadata = {
        "category": row["category"],
        "question_id": row["id"],
        "question_type": "original",
        "doc_source": doc_source,
    }
    return _csv_eval_two_turn(
        user_text=str(row["original_question"]),
        answer=str(row["original_answer"]),
        metadata=metadata,
    )


def _conversations_from_csv(path: str) -> list[Conversation]:
    """Qasper-style eval CSV (MCQ, yes/no, or deduped original columns)."""
    import pandas as pd

    df = pd.read_csv(path)
    kind = _detect_csv_eval_kind(list(df.columns))
    doc_source = Path(path).stem
    out: list[Conversation] = []

    if kind == "mcq":
        for _, row in df.iterrows():
            out.append(_format_mcq_csv_row(row, doc_source=doc_source))
    elif kind == "yes_no":
        for _, row in df.iterrows():
            out.append(_format_yesno_csv_row(row, doc_source=doc_source))
    else:
        seen: set = set()
        for _, row in df.iterrows():
            qid = row["id"]
            if qid in seen:
                continue
            seen.add(qid)
            out.append(_format_original_csv_row(row, doc_source=doc_source))

    return out


class TrainingExample(Conversation):
    # backwards compatibility
    pass