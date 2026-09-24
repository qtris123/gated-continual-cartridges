from cartridges.benchmark.scorers import (
    CATEGORY_SCORERS,
    mc_options,
    numeric_match,
    resolve_mc_option,
)


OPTIONS = [
    "Alpha is the correct explanation.",
    "Beta is the correct explanation.",
    "Gamma is the correct explanation.",
    "Delta is the correct explanation.",
]


def test_mc_options_accepts_option_text():
    prediction = "The answer is Gamma is the correct explanation."
    assert mc_options(
        prediction,
        OPTIONS[2],
        metadata={"options": OPTIONS},
    ) == 1.0


def test_mc_options_accepts_parenthesized_letter():
    assert resolve_mc_option("After considering them, the answer is (c).", OPTIONS) == (
        OPTIONS[2]
    )


def test_mc_options_uses_final_parenthesized_choice():
    prediction = "I compared (a) and (b). My final answer is (d)."
    assert resolve_mc_option(prediction, OPTIONS) == OPTIONS[3]


def test_mc_options_accepts_numeric_option():
    assert resolve_mc_option("(3) Gamma is the correct explanation.", OPTIONS) == OPTIONS[2]
    assert resolve_mc_option("3. Gamma is the correct explanation.", OPTIONS) == OPTIONS[2]
    assert resolve_mc_option("Answer: (2)", OPTIONS) == OPTIONS[1]
    assert resolve_mc_option("Answer: 1", OPTIONS) == OPTIONS[0]


def test_mc_options_option_first_over_conflicting_content():
    # Model specifies option (b), even though text mentions Alpha (option a)
    prediction = "Answer: (b) Alpha is mentioned here."
    assert resolve_mc_option(prediction, OPTIONS) == OPTIONS[1]


def test_mc_options_rejects_unparseable_output():
    prediction = "I cannot determine the answer from the supplied information."
    assert resolve_mc_option(prediction, OPTIONS) is None
    assert mc_options(
        prediction,
        OPTIONS[0],
        metadata={"options": OPTIONS},
    ) == 0.0


def test_mc_options_requires_options_metadata():
    assert mc_options("(a)", OPTIONS[0], metadata={}) == 0.0


def test_finqa_numeric_registered():
    # FinQA phase evals tag every row `finqa_numeric`; the accuracy route
    # dispatches on this, so a missing registration silently breaks scoring.
    assert CATEGORY_SCORERS["finqa_numeric"] is numeric_match


def test_numeric_match_prefers_number_after_final_answer():
    prediction = "First 3, then 12. The answer is 47 (see page 42)."
    assert numeric_match(prediction, "47") == 1.0


def test_numeric_match_ignores_scratch_work():
    # The gold is the concluding number, not an intermediate quantity.
    prediction = "Revenue was 1200 and cost 800.\nFinal answer: 400"
    assert numeric_match(prediction, "400") == 1.0
    assert numeric_match(prediction, "1200") == 0.0


def test_numeric_match_percent_and_fraction_conventions():
    # FinQA gold mixes 14.1 (%) and 0.141 (fraction); either should match.
    assert numeric_match("The answer is 14.1%", "0.141") == 1.0
    assert numeric_match("Answer: 0.141", "14.1") == 1.0


def test_numeric_match_strips_currency_and_separators():
    assert numeric_match("The answer is $1,234.5", "1234.5") == 1.0


def test_numeric_match_wrong_number_scores_zero():
    assert numeric_match("The answer is 50", "42") == 0.0


def test_numeric_match_no_number_scores_zero():
    assert numeric_match("I cannot compute this.", "42") == 0.0
