from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass
import random

from pydrantic import ObjectConfig
from transformers import PreTrainedTokenizerFast

from cartridges.datasets import GenerateEvalDataset, GenerateEvalDatasetElement
from cartridges.data.quality.resources import (
    PHASE_TO_ARTICLE_IDS,
    QUALITY_DATASET,
    QUALITY_SPLITS,
    load_articles,
    Article,
)
from cartridges.initialization.tokenization_utils import MODEL_TO_CHAT_TEMPLATE, MODELS_WITH_THINKING
from cartridges.benchmark.scorers import resolve_mc_option


@dataclass
class QualityQuestion:
    question_id: str
    question: str
    correct: str
    options: list[str]
    article_id: str
    difficult: int = 0
    question_type: str = "original"


class QualityMultipleChoiceGenerateDataset(GenerateEvalDataset):
    class Config(GenerateEvalDataset.Config):
        _pass_as_config = True
        article_ids: Optional[List[str]] = None
        phase: Optional[int] = None
        max_questions: Optional[int] = None
        cot: bool = False

    def __init__(self, config: Config, tokenizer: PreTrainedTokenizerFast, seed: int):
        from datasets import load_dataset

        self.config = config
        self.tokenizer = tokenizer

        # Determine target article IDs
        target_ids: Optional[set[str]] = None
        if config.article_ids is not None:
            target_ids = set(config.article_ids)
        elif config.phase is not None:
            if config.phase not in PHASE_TO_ARTICLE_IDS:
                raise ValueError(
                    f"phase must be one of {sorted(PHASE_TO_ARTICLE_IDS)}, got {config.phase}"
                )
            target_ids = set(PHASE_TO_ARTICLE_IDS[config.phase])

        articles = load_articles(list(target_ids) if target_ids else None)

        OPTION_LETTERS = ["a", "b", "c", "d"]

        def wrap_question(question_text: str, options: list[str], story_info: str) -> str:
            options_str = "\n".join(
                f"({OPTION_LETTERS[i]}) {opt}" for i, opt in enumerate(options)
            )
            return (
                "Please answer the question below about the following story: "
                f"{story_info}"
                f"\n\n<question>\n{question_text}\n</question>"
                f"\n\n<options>\n{options_str}\n</options>"
                f"\nOutput only the letter of the correct option (e.g. (a), (b), (c), or (d)) along with the content of the option."
                f"\n\nAnswer:"
            )

        questions: list[QualityQuestion] = []
        for split in QUALITY_SPLITS:
            for row in load_dataset(QUALITY_DATASET, split=split):
                aid = str(row["article_id"])
                if target_ids is not None and aid not in target_ids:
                    continue
                if aid not in articles:
                    continue

                article = articles[aid]
                story_info = f"Title: {article.title}, Author: {article.author}"

                raw_options = list(row["options"])
                raw_label = row.get("gold_label")
                if raw_label is None or raw_label == "":
                    raw_label = row.get("writer_label", 1)
                # QuALITY gold_label is 1-indexed (1 to 4)
                correct_idx: int = int(raw_label) - 1
                if not (0 <= correct_idx < len(raw_options)):
                    correct_idx = 0
                correct_text = raw_options[correct_idx]
                q_text = row["question"]
                qid = str(row.get("question_unique_id") or row.get("question_id"))

                wrapped = wrap_question(q_text, raw_options, story_info)
                questions.append(
                    QualityQuestion(
                        question_id=qid,
                        question=wrapped,
                        correct=correct_text,
                        options=raw_options,
                        article_id=aid,
                        difficult=int(row.get("difficult", 0)),
                        question_type=str(row.get("question_type", "original")),
                    )
                )

        random.Random(seed).shuffle(questions)

        if self.config.max_questions is not None:
            questions = questions[:self.config.max_questions]

        self.questions = questions
        self.question_id_to_idx = {
            q.question_id: idx for idx, q in enumerate(self.questions)
        }

    def __getitem__(self, index: int) -> GenerateEvalDatasetElement:
        question: QualityQuestion = self.questions[index]

        kwargs: dict[str, Any] = {}
        if self.tokenizer.name_or_path in MODELS_WITH_THINKING:
            # Turn off reasoning/thinking for accuracy and generation evals
            kwargs["enable_thinking"] = False

        input_ids = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": question.question}],
            add_generation_prompt=True,
            return_tensors="pt",
            chat_template=MODEL_TO_CHAT_TEMPLATE.get(self.tokenizer.name_or_path, None),
            **kwargs,
        )

        return GenerateEvalDatasetElement(
            input_ids=input_ids,
            prompt=question.question,
            answer=question.correct,
            convo_id=question.question_id,
            metadata={
                "idx": index,
                "options": question.options,
                "article_id": question.article_id,
            },
        )

    def __len__(self) -> int:
        return len(self.questions)

    def score(
        self,
        pred: str,
        answer: str,
        convo_id: str,
    ) -> Tuple[bool, Dict[str, Optional[str]]]:
        """Score MCQ generations: first check option letter/number, then option content."""
        import re
        from difflib import SequenceMatcher

        question: QualityQuestion = self.questions[self.question_id_to_idx[convo_id]]
        OPTION_LETTERS = ["a", "b", "c", "d"]
        raw_options = question.options
        options_lower = [o.strip().lower() for o in raw_options]
        answer_lower = answer.strip().lower()

        pred_text = pred.strip()

        # 1. Try to extract letter (a)-(d) or number (1)-(4) — prefer last occurrence after "Answer:"
        letters_found = [
            (m.start(), m.group(1).lower())
            for m in re.finditer(r"\(([a-d1-4])\)", pred_text.lower())
        ]
        # Match leading letter/number like "a)", "a.", "1)", "1.", "Answer: a"
        leading = re.search(r"^(?:answer:\s*)?\(?([a-d1-4])(?:\.|\)|\:|\s)", pred_text.lower())
        if leading:
            letters_found.insert(0, (leading.start(1), leading.group(1).lower()))
        if not letters_found:
            exact = re.search(r"^(?:answer:\s*)?\(?([a-d1-4])\)?$", pred_text.lower())
            if exact:
                letters_found.append((exact.start(1), exact.group(1).lower()))

        answer_anchor_pos = pred_text.lower().rfind("answer")
        after_anchor = [
            letter
            for pos, letter in letters_found
            if answer_anchor_pos != -1 and pos > answer_anchor_pos
        ]
        chosen_indicator = (after_anchor or [ltr for _, ltr in letters_found] or [None])[-1] if (after_anchor or letters_found) else None

        if chosen_indicator is not None:
            if chosen_indicator in "1234":
                idx = int(chosen_indicator) - 1
            elif chosen_indicator in OPTION_LETTERS:
                idx = OPTION_LETTERS.index(chosen_indicator)
            else:
                idx = -1
            if 0 <= idx < len(raw_options):
                resolved = options_lower[idx]
                return resolved == answer_lower, {
                    "extracted_pred": raw_options[idx],
                    "method": "option_indicator",
                }

        # 2. Fallback: check option content
        pred_lower = pred_text.lower()
        # Exact substring match
        mentions = [
            (pred_lower.rfind(opt.replace(" ", "")), opt, i)
            for i, opt in enumerate(options_lower)
            if opt.replace(" ", "") in pred_lower.replace(" ", "")
        ]
        if mentions:
            _, best_opt, best_idx = max(mentions, key=lambda x: x[0])
            return best_opt == answer_lower, {
                "extracted_pred": raw_options[best_idx],
                "method": "substring",
            }

        # Fuzzy similarity fallback
        def find_best_match(reference: str, candidates: list[str]) -> str:
            return max(candidates, key=lambda x: SequenceMatcher(None, reference, x).ratio())

        best_match = find_best_match(pred_lower, options_lower)
        similarity = SequenceMatcher(None, pred_lower, best_match).ratio()
        if similarity >= 0.5:
            best_idx = options_lower.index(best_match)
            return best_match == answer_lower, {
                "extracted_pred": raw_options[best_idx],
                "method": "fuzzy",
            }

        return False, {"extracted_pred": None, "method": "unresolved"}
