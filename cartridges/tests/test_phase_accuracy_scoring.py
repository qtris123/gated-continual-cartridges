from cartridges.benchmark.scorers import mc_options, resolve_mc_option


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
