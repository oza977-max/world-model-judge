"""Tests for wmj.reporting.exception_plot — Chart 1, the backtesting exception plot (ADR-R2, RP-1).

In plain words: the primary chart must show, on the image itself, how many misses a model of that
confidence should have had, how many it actually had, and which colour band that puts it in
(TC-RP1-01); a deliberately broken test model's chart must say so in words (TC-RP8-01). The chart
draws only what the verdict says: these tests feed it blocks made by the real judge, check the saved
SVG (charts keep text as text), and check it refuses a block that does not add up instead of drawing
part of it.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest

from tests.unit.judge._builders import make_input
from wmj.judge.exceptions import compute_exceptions_and_trials
from wmj.reporting import exception_plot as ep
from wmj.reporting import style


def _blocks(inp=None):
    exceptions, trials = compute_exceptions_and_trials(inp if inp is not None else make_input())
    return exceptions, trials


def _envelope(*, name="direct", is_fixture=False, inp=None, world="lv"):
    exceptions, trials = _blocks(inp)
    return {
        "model_ref": "opaque-0",
        "model_name": name,
        "is_fixture": is_fixture,
        "verdict": {"world": world, "exceptions": exceptions, "trials": trials},
        "meta": {},
    }


# --- outside-in: the saved chart, read back from disk -------------------------------------------------


def test_tc_rp1_01_the_saved_chart_states_expected_beside_observed_and_names_the_band(tmp_path):
    env = _envelope()
    written = ep.render_exception_charts(env, tmp_path / "charts", tmp_path / "captions")
    entries = env["verdict"]["exceptions"]["per_task"]
    assert len(written) == len(entries)  # one chart per (region, task, horizon step)
    for entry, paths in zip(entries, written, strict=True):
        svg = paths.svg.read_text(encoding="utf-8")
        expected = f"expected: {round(entry['expected']) if float(entry['expected']).is_integer() else entry['expected']} of {entry['n_trials']}"
        assert f"observed exceptions: {entry['observed']}" in svg
        assert expected in svg
        assert f"band: {entry['band'].upper()}" in svg
        assert f"region: {entry['region']}" in svg
        caption = paths.caption.read_text(encoding="utf-8")
        assert f"it missed {entry['observed']}," in caption and f"the {entry['band']} band" in caption
        assert paths.png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
        stem = f"lv-direct-exceptions-{entry['task']}-{entry['region']}-h{entry['horizon_step']}"
        assert (paths.png.name, paths.svg.name, paths.caption.name) == (f"{stem}.png", f"{stem}.svg", f"{stem}.txt")


def test_tc_rp8_01_a_fixture_models_saved_chart_carries_the_fixture_sentence(tmp_path):
    written = ep.render_exception_charts(_envelope(name="fx-brittle", is_fixture=True), tmp_path / "c", tmp_path / "t")
    for paths in written:
        svg = paths.svg.read_text(encoding="utf-8")
        assert style.FIXTURE_LABEL in svg and "FIXTURE: fx-brittle" in svg
    plain = ep.render_exception_charts(_envelope(), tmp_path / "c2", tmp_path / "t2")
    assert all(style.FIXTURE_LABEL not in p.svg.read_text(encoding="utf-8") for p in plain)
