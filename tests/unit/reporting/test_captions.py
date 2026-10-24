"""Tests for wmj.reporting.captions — fixed templates, never improvised.

Reporting ADR-R3: captions are authored English templates reviewed as
part of the spec. P2-C05's scoped Chart-2 caption is the template with
its switch-step sentence dropped — derived from the full template, not
retyped, so the two can never drift apart.
"""

from __future__ import annotations

from pathlib import Path

from wmj.reporting.captions import (
    CHART2_FULL,
    CHART2_SCOPED,
    sentence_count,
    write_caption,
)


def test_chart2_full_template_is_adr_r3_verbatim():
    assert CHART2_FULL == (
        "How wrong the model gets as it predicts further ahead. The dashed "
        "black line is how fast the world drifts away from itself — only the "
        "gap above that line is the model's fault. Past each task's marked "
        "step, exact paths stop being gradable and the judge switches to "
        "checking the overall pattern."
    )


def test_chart2_scoped_is_the_full_template_minus_its_last_sentence():
    assert CHART2_FULL.startswith(CHART2_SCOPED)
    assert "switch" not in CHART2_SCOPED
    assert "only the gap above that line is the model's fault" in CHART2_SCOPED


def test_both_templates_respect_rp5_sentence_limit():
    assert sentence_count(CHART2_SCOPED) == 2
    assert sentence_count(CHART2_FULL) == 3


def test_write_caption_creates_parents_and_writes_utf8(tmp_path):
    path = tmp_path / "out" / "captions" / "x.txt"
    written = write_caption(path, CHART2_SCOPED)
    assert written == path
    assert path.read_text(encoding="utf-8") == CHART2_SCOPED


def test_rp5_write_caption_refuses_a_fourth_sentence():
    """code-review-001 I5: RP-5's three-sentence limit had no test proving
    the `CaptionLengthError` guard fires — it was only ever exercised with
    the two-sentence scoped caption."""
    import pytest

    from wmj.reporting.captions import CaptionLengthError

    with pytest.raises(CaptionLengthError):
        write_caption(Path("unused.txt"), "One. Two. Three. Four.")


# --- P5-C01: the other three charts' captions (ADR-R3), filled by checked functions --------------------
# In plain words: each caption is fixed text with a few numbers dropped in. These tests pin the text,
# check every filled caption stays within three sentences, and check that a wrong number (a count
# above the number of runs, an unknown band, a confidence level the judge never uses) is refused
# instead of being printed under a chart.

import pytest

from wmj.reporting import captions as c

CHART1 = (
    "Each dot is one test run: how far the outcome landed from the model's guess, measured in the "
    "model's own stated error sizes and combined over everything it predicts. The line marks the edge "
    "of the range the model called 90% likely; dots above it are the misses, circled in red. A model this "
    "confident should miss about {expected} of {n} times — it missed {observed}, which lands it in the "
    "{band} band drawn on the right."
)
CHART3 = (
    "When the model said it was {level}% sure, was it right that often? Of every 100 ranges it drew at "
    "90% confidence, about 90 should contain the true outcome. The solid line is familiar territory; the "
    "dashed line is territory it never saw in training — the honest model's lines both hug the diagonal."
)


def test_the_templates_are_the_pinned_text():
    assert c.CHART1_TEMPLATE == CHART1  # amended from ADR-R3 for the A39 item 9 chart (backlog)
    assert c.CHART3_TEMPLATE == CHART3  # ADR-R3 verbatim
    assert c.CHART4_HEAD == "Ordinary accuracy and the judge's trust horizon, side by side."
    assert c.CHART4_DISAGREE == "Ordinary error says {A} and {B} are equals; the judge does not."
    assert c.CHART4_AGREE == (
        "On this run the two rankings agree — the interesting case did not occur, and saying so plainly "
        "beats manufacturing one."
    )


def test_chart1_fills_its_numbers_and_rounds_the_expected_count_for_display():
    text = c.chart1_caption(expected=20.0, n=200, observed=27, band="green")
    assert "should miss about 20 of 200 times — it missed 27, which lands it in the green band" in text
    assert c.sentence_count(text) == 3
    # the verdict's expected count is a float n x p; a value a hair off a whole number still reads "about 20"
    assert "about 20 of 200" in c.chart1_caption(expected=19.999999999999996, n=200, observed=0, band="red")
    assert "it missed 200," in c.chart1_caption(expected=20.0, n=200, observed=200, band="red")


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"expected": 20.0, "n": 200, "observed": 201, "band": "red"}, "observed"),
        ({"expected": 20.0, "n": 200, "observed": -1, "band": "red"}, "observed"),
        ({"expected": 20.0, "n": 200, "observed": True, "band": "red"}, "observed"),
        ({"expected": 20.0, "n": 0, "observed": 0, "band": "red"}, "n"),
        ({"expected": 20.0, "n": 200.0, "observed": 3, "band": "red"}, "n"),
        ({"expected": float("nan"), "n": 200, "observed": 3, "band": "red"}, "expected"),
        ({"expected": -1.0, "n": 200, "observed": 3, "band": "red"}, "expected"),
        ({"expected": 201.0, "n": 200, "observed": 3, "band": "red"}, "expected"),
        ({"expected": "20", "n": 200, "observed": 3, "band": "red"}, "expected"),
        ({"expected": 20.0, "n": 200, "observed": 3, "band": "GREEN"}, "band"),
        ({"expected": 20.0, "n": 200, "observed": 3, "band": "blue"}, "band"),
    ],
)
def test_chart1_refuses_a_number_that_cannot_be_true(kwargs, match):
    with pytest.raises(c.CaptionSlotError, match=match):
        c.chart1_caption(**kwargs)


def test_chart3_takes_only_the_four_levels_the_judge_reports():
    assert c.chart3_caption(level=90).startswith("When the model said it was 90% sure, was it right that often?")
    assert c.sentence_count(c.chart3_caption(level=95)) == 3
    for bad in (85, 0.9, 90.0, True, "90", 100):
        with pytest.raises(c.CaptionSlotError, match="level"):
            c.chart3_caption(level=bad)


def test_chart4_has_its_two_fixed_forms_and_names_the_two_models_it_compares():
    agree = c.chart4_caption(None)
    assert agree == c.CHART4_HEAD + " " + c.CHART4_AGREE and c.sentence_count(agree) == 2
    disagree = c.chart4_caption(("direct", "ensemble"))
    assert disagree == c.CHART4_HEAD + " Ordinary error says direct and ensemble are equals; the judge does not."
    assert c.sentence_count(disagree) == 2


def test_chart4_takes_display_names_as_given_including_fixture_prefixes_and_hyphens():
    text = c.chart4_caption(("FIXTURE: fx-honest-rough", "direct"))
    assert "Ordinary error says FIXTURE: fx-honest-rough and direct are equals" in text
    assert c.sentence_count(text) == 2


@pytest.mark.parametrize(
    "pair",
    [("direct",), ("direct", "direct"), ("", "ensemble"), ("direct", None), "de", ("a. B", "c"), ("a", "b", "c"),
     ("A.", "b"), ("a!", "b"), (" a", "b"), ("a", "b\n"), ("a", "b ")],
)
def test_chart4_refuses_a_pair_that_is_not_two_different_plain_names(pair):
    with pytest.raises(c.CaptionSlotError, match="pair|name"):
        c.chart4_caption(pair)


def test_every_filled_caption_is_within_three_sentences_and_has_no_unfilled_slot():
    filled = [
        c.chart1_caption(expected=20.0, n=200, observed=k, band=b)
        for k in (0, 7, 200) for b in ("green", "amber", "red")
    ] + [c.chart3_caption(level=v) for v in (50, 80, 90, 95)] + [c.chart4_caption(None), c.chart4_caption(("a", "b"))]
    for text in filled:
        assert c.sentence_count(text) <= c.MAX_SENTENCES
        assert "{" not in text and "}" not in text


def test_an_expected_count_that_is_not_whole_is_shown_to_one_decimal_never_rounded_away():
    assert "about 0.4 of 20 times — it missed 7" in c.chart1_caption(expected=0.4, n=20, observed=7, band="red")
    assert "about 2.5 of 25" in c.chart1_caption(expected=2.5, n=25, observed=2, band="green")  # not a half-to-even 2


def test_the_sentence_limit_guard_refuses_an_over_long_filled_caption(monkeypatch):
    # every validated slot keeps charts 1, 3 and 4 within three sentences; the guard is for a template edit
    monkeypatch.setattr(c, "CHART3_TEMPLATE", c.CHART3_TEMPLATE + " A fourth sentence.")
    with pytest.raises(c.CaptionLengthError):
        c.chart3_caption(level=90)
