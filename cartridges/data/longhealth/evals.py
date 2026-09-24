from typing import List, Optional, Tuple, Dict
import random

from pydrantic import ObjectConfig
from transformers import PreTrainedTokenizerFast

from cartridges.datasets import GenerateEvalDataset, GenerateEvalDatasetElement
from cartridges.data.longhealth.utils import LongHealthQuestion, LongHealthPatient, load_longhealth_dataset
from cartridges.initialization.tokenization_utils import MODEL_TO_CHAT_TEMPLATE, MODELS_WITH_THINKING



class LongHealthMultipleChoiceGenerateDataset(GenerateEvalDataset):
    class Config(GenerateEvalDataset.Config):
        _pass_as_config = True
        patient_ids: Optional[List[str]] = None
        max_questions: Optional[int] = None
        include_diagnosis: bool = True
        cot: bool = True


    def __init__(self, config: Config, tokenizer: PreTrainedTokenizerFast, seed: int):
        self.config = config
        self.tokenizer = tokenizer
        
        self.patients = load_longhealth_dataset(config.patient_ids)
        
        OPTION_LETTERS = ["a", "b", "c", "d", "e"]

        def wrap_question(question: LongHealthQuestion, patient: LongHealthPatient):
            raw_options = [
                question.answer_a,
                question.answer_b,
                question.answer_c,
                question.answer_d,
                question.answer_e,
            ]
            options = "\n".join(
                f"({OPTION_LETTERS[i]}) {opt}" for i, opt in enumerate(raw_options)
            )

            if self.config.include_diagnosis:
                patient_info = f"ID {patient.patient_id}, Name: {patient.name}, Birthday: {patient.birthday}, Diagnosis: {patient.diagnosis}"
            else:
                patient_info = f"ID {patient.patient_id}, Name: {patient.name}, Birthday: {patient.birthday}"

            return (
                "Please answer the question below about the following patient: "
                f"{patient_info}"
                f"\n\n<question>\n{question.question}\n</question>"
                f"\n\n<options>\n{options}\n</options>"
                f"\nOutput only the letter of the correct option (e.g. (a), (b), (c), (d), or (e)) along with the content of the option."
                f"\n\nAnswer:"
            )
         
        self.questions = [
            LongHealthQuestion(
                question_id=question.question_id,
                question=wrap_question(question, patient),
                correct=question.correct,
                answer_a=question.answer_a,
                answer_b=question.answer_b,
                answer_c=question.answer_c,
                answer_d=question.answer_d,
                answer_e=question.answer_e,
                answer_location=question.answer_location,
            )
            for patient in self.patients
            for question in patient.questions
        ]
        random.Random(seed).shuffle(self.questions)


        if self.config.max_questions is not None:
            self.questions = self.questions[:self.config.max_questions]
        self.question_id_to_idx = {
            question.question_id: idx for idx, question in enumerate(self.questions)
        }


        self.tokenizer = tokenizer


    def __getitem__(
        self, index: int
    ) -> GenerateEvalDatasetElement:
        # convo: ContextConvo = ContextConvo.model_validate(self.data[index])
        question: LongHealthQuestion = self.questions[index]

        kwargs = {}
        if self.tokenizer.name_or_path in MODELS_WITH_THINKING:
            # Always disable thinking for accuracy/generation evals
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
            metadata={"idx": index}
        )

    def __len__(self):
        return len(self.questions)

    def score(
        self,
        pred: str,
        answer: str,
        convo_id: str
    ) -> Tuple[bool, Dict[str, Optional[str]]]:
        import re
        from difflib import SequenceMatcher

        def find_best_match(reference, candidates):
            return max(candidates, key=lambda x: SequenceMatcher(None, reference, x).ratio())

        question: LongHealthQuestion = self.questions[self.question_id_to_idx[convo_id]]
        OPTION_LETTERS = ["a", "b", "c", "d", "e"]
        raw_options = [
            question.answer_a,
            question.answer_b,
            question.answer_c,
            question.answer_d,
            question.answer_e,
        ]
        options_lower = [o.strip().lower() for o in raw_options]
        answer_lower = answer.strip().lower()

        pred_text = pred.strip()

        # 1. Try to extract letter (a)/(b)/(c)/(d)/(e) — prefer last occurrence after "Answer:"
        letters_found = [
            (m.start(), m.group(1))
            for m in re.finditer(r"\(([a-e])\)", pred_text.lower())
        ]
        # Also match bare leading letter like "a)" or "a." or "Answer: a"
        leading = re.match(r"^\(?([a-e])[\)\.\:]\s", pred_text.lower())
        if leading:
            letters_found.insert(0, (0, leading.group(1)))
        # Prefer any letter that appears after "Answer:"
        answer_anchor_pos = pred_text.lower().rfind("answer")
        after_anchor = [
            letter
            for pos, letter in letters_found
            if answer_anchor_pos != -1 and pos > answer_anchor_pos
        ]
        chosen_letter = (after_anchor or [ltr for _, ltr in letters_found] or [None])[-1] if (after_anchor or letters_found) else None

        if chosen_letter is not None:
            idx = OPTION_LETTERS.index(chosen_letter)
            if idx < len(raw_options):
                resolved = options_lower[idx]
                return resolved == answer_lower, {"extracted_pred": raw_options[idx], "method": "letter"}

        # 2. Fallback: fuzzy content match against option texts
        pred_lower = pred_text.lower()
        # Exact substring match
        mentions = [
            (pred_lower.rfind(opt.replace(" ", "")), opt, i)
            for i, opt in enumerate(options_lower)
            if opt.replace(" ", "") in pred_lower.replace(" ", "")
        ]
        if mentions:
            _, best_opt, best_idx = max(mentions, key=lambda x: x[0])
            return best_opt == answer_lower, {"extracted_pred": raw_options[best_idx], "method": "substring"}

        # Fuzzy similarity fallback
        best_match = find_best_match(pred_lower, options_lower)
        similarity = SequenceMatcher(None, pred_lower, best_match).ratio()
        if similarity >= 0.5:
            best_idx = options_lower.index(best_match)
            return best_match == answer_lower, {"extracted_pred": raw_options[best_idx], "method": "fuzzy"}

        # Could not resolve
        return False, {"extracted_pred": None, "method": "unresolved"}
        