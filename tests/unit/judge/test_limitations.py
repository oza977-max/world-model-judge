"""Tests for wmj.judge.limitations — the judge's own admissions, word for word (JU-10, ADR-J7).

In plain words: every verdict must say, in fixed wording that nobody can quietly soften, what
this judge cannot do. These tests pin the seven sentences exactly as the spec wrote them, in
order, and the four things the verdict lists as never tested.
"""

from __future__ import annotations

from wmj.judge.limitations import JU10_DISCLOSURES, NOT_TESTED

EXPECTED = (
    "This method validates the middle of the distribution, not the extremes — it says nothing about how the model behaves on rare, disaster-scale events.",
    "This toy world validates the judging harness, not any real-world model — a clean result here is not evidence that a real world model is trustworthy.",
    "The judge, the models, and the thresholds share a single author. The blinding in JU-1 means the judge cannot favour a model for its identity — it does not mean this evaluation has banking's organisational independence: a separate team, reporting separately, empowered to challenge the model builder.",
    "This project borrows banking's backtesting and ex-ante-threshold practices, not its ongoing monitoring or its power to force a stop. Deciding who has the authority to act on a bad verdict is exactly the gap this project's source essay names — it is not solved here.",
    "The judge's own thresholds and metric choices are modelling decisions. They were fixed in advance and are not arbitrary, but they are not beyond challenge either.",
    "The toy worlds contain no genuine off-model surprises by construction (no randomness, no hidden state, no high-dimensional input) — this judge's behaviour when a model meets a real surprise is untested.",
    "'Control' and 'planning' name the tolerance regime used to grade a passive prediction, not a closed-loop decision. This judge does not test whether a model can choose its own actions to reach a goal — the actions it is graded against were chosen by someone else.",
)


def test_tc_ju10_01_the_seven_disclosures_are_present_verbatim_and_in_order():
    assert JU10_DISCLOSURES == EXPECTED
    assert isinstance(JU10_DISCLOSURES, tuple) and len(JU10_DISCLOSURES) == 7


def test_the_disclosures_are_plain_strings_with_no_placeholders_or_formatting_slots():
    for text in JU10_DISCLOSURES:
        assert type(text) is str and text.strip() == text and text.endswith(".")
        assert "{" not in text and "}" not in text and "%s" not in text


def test_the_not_tested_list_is_the_four_entries_the_spec_names():
    assert NOT_TESTED == (
        "genuine off-model surprises (WD-8)",
        "closed-loop action selection",
        "ongoing monitoring",
        "the seven JU-10 disclosures cover the rest of this list",
    )


def test_the_disclosures_state_the_single_author_blind_not_independent_point_in_the_text_itself():
    third = JU10_DISCLOSURES[2].lower()
    assert "single author" in third and "does not mean" in third and "independence" in third
