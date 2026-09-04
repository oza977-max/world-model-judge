"""Tests for wmj.reporting.captions — fixed templates, never improvised.

Reporting ADR-R3: captions are authored English templates reviewed as
part of the spec. P2-C05's scoped Chart-2 caption is the template with
its switch-step sentence dropped — derived from the full template, not
retyped, so the two can never drift apart.
"""

from __future__ import annotations

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
