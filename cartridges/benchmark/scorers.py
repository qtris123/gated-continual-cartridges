from __future__ import annotations

import re
import string
from collections import Counter
from difflib import SequenceMatcher
from typing import Any, Callable


def exact_match(prediction: str, ground_truth: str, **kwargs) -> float:
    return float(prediction.strip().lower() == ground_truth.strip().lower())


def contains_match(prediction: str, ground_truth: str, **kwargs) -> float:
    return float(ground_truth.strip().lower() in prediction.strip().lower())


def f1_score(prediction: str, ground_truth: str, **kwargs) -> float:
    pred_tokens = _normalize_and_tokenize(prediction)
    gt_tokens = _normalize_and_tokenize(ground_truth)
    if not gt_tokens:
        return float(not pred_tokens)
    if not pred_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_common = sum(common.values())
    if num_common == 0:
        return 0.0
    precision = num_common / len(pred_tokens)
    recall = num_common / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)


def multiple_choice(prediction: str, ground_truth: str, **kwargs) -> float:
    """Score multiple-choice answers by extracting the letter (A/B/C/D/...)."""
    pred_letter = _extract_mc_letter(prediction)
    gt_letter = _extract_mc_letter(ground_truth)
    if pred_letter is None or gt_letter is None:
        return exact_match(prediction, ground_truth)
    return float(pred_letter == gt_letter)


def longhealth_mc(prediction: str, ground_truth: str, **kwargs) -> float:
    """Scoring logic from cartridges.data.longhealth.evals — extract from <answer>
    tags, fuzzy-match against the option list, compare to ground truth."""
    metadata: dict[str, Any] = kwargs.get("metadata", {})
    options: list[str] = metadata.get("options", [])

    extracted = _extract_answer_tag(prediction)

    if extracted is not None and options:
        closest = _find_best_match(extracted.strip().lower(),
                                   [o.strip().lower() for o in options])
        return float(closest == ground_truth.strip().lower())

    if extracted is not None:
        return float(extracted.strip().lower() == ground_truth.strip().lower())

    return 0.0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_answer_tag(text: str) -> str | None:
    m = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
    if m:
        return m.group(1)
    m = re.search(r'\{answer\}\s*\n\s*([^\n]+)', text, re.IGNORECASE)
    if m:
        return m.group(1)
    return None


def _find_best_match(reference: str, candidates: list[str]) -> str:
    return max(candidates, key=lambda x: SequenceMatcher(None, reference, x).ratio())


def _extract_mc_letter(text: str) -> str | None:
    text = text.strip()
    match = re.search(r'\b([A-E])\b', text.upper())
    if match:
        return match.group(1)
    if len(text) == 1 and text.upper() in "ABCDE":
        return text.upper()
    return None


def _normalize_and_tokenize(text: str) -> list[str]:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text.split()


ScorerFn = Callable[..., float]

SCORER_REGISTRY: dict[str, ScorerFn] = {
    "exact_match": exact_match,
    "contains": contains_match,
    "f1": f1_score,
    "multiple_choice": multiple_choice,
    "longhealth_mc": longhealth_mc,
}
