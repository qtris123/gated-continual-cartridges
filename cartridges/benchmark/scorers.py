from __future__ import annotations

import math
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


def yes_no_match(prediction: str, ground_truth: str, **kwargs) -> float:
    """Score yes/no answers by extracting yes or no from the prediction."""
    pred = _extract_yes_no(prediction)
    gt = ground_truth.strip().lower()
    if gt not in ("yes", "no"):
        gt = _extract_yes_no(ground_truth)
    if pred is None or gt is None:
        return exact_match(prediction, ground_truth)
    return float(pred == gt)


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


def resolve_mc_option(
    prediction: str,
    options: list[str],
) -> str | None:
    """Resolve a free-form generation to one of the supplied option texts.

    This is the phase-stream protocol from
    ``faridlazuarda/gated-continual-cartridges@7c11bf1``.  Keeping the resolver
    separate from the score makes malformed-answer rates auditable without
    changing the baseline-compatible accuracy definition.
    """
    if not options:
        return None
    normalized_options = [option.strip().lower() for option in options]
    text = _extract_answer_tag(prediction) or prediction
    normalized_text = " ".join(text.lower().split())

    def by_letter(letter: str) -> str | None:
        index = ord(letter) - ord("a")
        return normalized_options[index] if index < len(normalized_options) else None

    chosen = None
    letters = [
        (match.start(), match.group(1))
        for match in re.finditer(r"\(([a-e])\)", normalized_text)
    ]
    leading = re.match(r"^\(?([a-e])[\)\.\:]\s", normalized_text)
    if leading:
        letters.insert(0, (0, leading.group(1)))
    if letters:
        answer_at = normalized_text.rfind("answer")
        after = [
            letter
            for position, letter in letters
            if answer_at != -1 and position > answer_at
        ]
        chosen = by_letter(after[0] if after else letters[-1][1])

    if chosen is None:
        squashed = normalized_text.replace(" ", "")
        mentions = [
            (squashed.rfind(option.replace(" ", "")), option)
            for option in normalized_options
            if option.replace(" ", "") in squashed
        ]
        if mentions:
            chosen = max(mentions)[1]

    if chosen is None:
        best = _find_best_match(normalized_text, normalized_options)
        if SequenceMatcher(None, normalized_text, best).ratio() >= 0.5:
            chosen = best

    return options[normalized_options.index(chosen)] if chosen is not None else None


def mc_options(prediction: str, ground_truth: str, **kwargs) -> float:
    """Score MCQ generations against option text, not a fixed letter protocol."""
    metadata: dict[str, Any] = kwargs.get("metadata", {})
    raw_options = metadata.get("options")
    options = [str(option) for option in raw_options] if raw_options is not None else []
    chosen = resolve_mc_option(prediction, options)
    if chosen is None:
        return 0.0
    gt = _extract_answer_tag(ground_truth) or ground_truth
    return float(chosen.strip().lower() == gt.strip().lower())


def numeric_match(prediction: str, ground_truth: str, **kwargs) -> float:
    """Numeric answers (FinQA): compare the model's *concluding* number to the
    gold value -- never intermediate quantities.

    FinQA generations show their work, so scanning every number for a match
    credits answers whose headline is wrong but whose scratch work happens to
    contain the gold. Instead a single answer candidate is extracted, in
    order of preference:

      1. the first number after the final "answer" mention ("The answer is
         47 (see page 42)" must resolve to 47, not the citation)
      2. else the last number in the last number-bearing line
      3. else the last number in the text

    Normalisation: $ and thousands separators stripped; a trailing % divides
    by 100; the candidate is accepted in either the percent or the fraction
    convention (14.1 vs 0.141) since FinQA gold answers mix the two.
    Tolerance is loose-relative (0.5%) to absorb the rounding FinQA gold
    answers carry.
    """
    gt_vals = _extract_numbers(ground_truth)
    if not gt_vals:
        return contains_match(prediction, ground_truth)

    pred = None
    m = list(re.finditer(r'[Aa]nswer', prediction))
    if m:
        tail_vals = _extract_numbers(prediction[m[-1].end():])
        if tail_vals:
            pred = tail_vals[0]
    if pred is None:
        for line in reversed(prediction.splitlines()):
            line_vals = _extract_numbers(line)
            if line_vals:
                pred = line_vals[-1]
                break
    if pred is None:
        return 0.0

    gt = gt_vals[-1]
    for candidate in (pred, pred / 100.0, pred * 100.0):
        if math.isclose(candidate, gt, rel_tol=5e-3, abs_tol=5e-4):
            return 1.0
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
    # Try "Answer: X" format first (most reliable when model follows instructions)
    match = re.search(r'[Aa]nswer:\s*([A-E])\b', text)
    if match:
        return match.group(1).upper()
    # Fallback: any standalone letter A-E
    match = re.search(r'\b([A-E])\b', text.upper())
    if match:
        return match.group(1)
    if len(text) == 1 and text.upper() in "ABCDE":
        return text.upper()
    return None


def _extract_yes_no(text: str) -> str | None:
    text = text.strip()
    # Try "Answer: Yes/No" format first
    match = re.search(r'[Aa]nswer:\s*(yes|no)\b', text, re.IGNORECASE)
    if match:
        return match.group(1).lower()
    # Fallback: starts with yes/no
    lower = text.lower()
    if lower.startswith("yes"):
        return "yes"
    if lower.startswith("no"):
        return "no"
    # Fallback: any yes/no in text
    match = re.search(r'\b(yes|no)\b', text, re.IGNORECASE)
    return match.group(1).lower() if match else None


def _extract_numbers(text: str) -> list[float]:
    """All numbers in the text, $-signs and thousands separators stripped,
    trailing % applied as /100."""
    cleaned = text.replace("$", "").replace(",", "")
    out = []
    for m in re.finditer(r'(-?\d+(?:\.\d+)?)(\s*%)?', cleaned):
        val = float(m.group(1))
        if m.group(2):
            val /= 100.0
        out.append(val)
    return out


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
    "yes_no": yes_no_match,
    "longhealth_mc": longhealth_mc,
    "mc_options": mc_options,
    "numeric_match": numeric_match,
}

CATEGORY_SCORERS: dict[str, ScorerFn] = {
    "longhealth_mcq": mc_options,
    "quality_mcq": mc_options,
    "finqa_numeric": numeric_match,
}
