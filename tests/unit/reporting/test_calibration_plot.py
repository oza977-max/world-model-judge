"""Tests for wmj.reporting.calibration_plot — Chart 3, "was 90% sure right 90% of the time?" (ADR-R2, RP-3).

In plain words: for each task, the chart plots how often the truth actually landed inside the
model's ranges at the four stated confidence levels, one line per region, against a diagonal
labelled "perfectly honest", with how wide the 90% ranges were printed above it (sharpness, JU-5),
so honesty bought by vagueness is visible. The caption speaks in natural frequencies ("about 90 of
every 100", TC-RP3-01). These tests feed it blocks made by the real judge and read the saved SVG back.
"""

from __future__ import annotations

import copy
import math

import numpy as np
import pytest

from tests.unit.reporting._verdict_blocks import envelope
from wmj.reporting import calibration_plot as cp
from wmj.reporting import style


def test_tc_rp3_01_the_saved_chart_draws_and_labels_the_diagonal_and_speaks_in_natural_frequencies(tmp_path):
    env = envelope()
    paths = cp.render_calibration_chart(env, tmp_path / "charts", tmp_path / "captions")
    assert (paths.png.name, paths.svg.name, paths.caption.name) == ("lv-direct-calibration.png", "lv-direct-calibration.svg", "lv-direct-calibration.txt")
    svg = paths.svg.read_text(encoding="utf-8")
    assert "perfectly honest" in svg
    for entry in env["verdict"]["sharpness"]["per_task"]:  # JU-5: sharpness printed beside the coverage it qualifies
        assert "sharpness (mean 90% width)" in svg and entry["region"] in svg
    caption = paths.caption.read_text(encoding="utf-8")
    assert "about 90 should contain the true outcome" in caption and "Of every 100" in caption


def test_tc_rp8_01_a_fixtures_calibration_chart_carries_the_fixture_sentence(tmp_path):
    paths = cp.render_calibration_chart(envelope(name="fx-overconfident", is_fixture=True), tmp_path / "c", tmp_path / "t")
    svg = paths.svg.read_text(encoding="utf-8")
    assert style.FIXTURE_LABEL in svg and "FIXTURE: fx-overconfident" in svg
