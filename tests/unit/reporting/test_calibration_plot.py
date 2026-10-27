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

from tests.unit.reporting._svg import says
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
    assert says(svg, style.FIXTURE_LABEL) and "FIXTURE: fx-overconfident" in svg


def _figure(env=None):
    env = env or envelope()
    panels = cp.calibration_panels(env["verdict"]["calibration"], env["verdict"]["sharpness"])
    return env, cp.build_calibration_figure(panels, env["model_name"], env["is_fixture"])


def test_one_panel_per_task_and_one_line_per_region_with_the_verdicts_coverage():
    env, chart = _figure()
    tasks = list(dict.fromkeys(e["task"] for e in env["verdict"]["calibration"]["per_task"]))
    assert list(chart.tasks) == tasks and len(chart.panels) == len(tasks)
    for index, region, line in chart.region_lines:
        entry = next(e for e in env["verdict"]["calibration"]["per_task"] if e["task"] == tasks[index] and e["region"] == region)
        assert list(line.get_xdata()) == cp.LEVELS and list(line.get_ydata()) == entry["coverage"]
        assert line.get_linestyle() == ("-" if region == "training" else "--")


def test_the_diagonal_is_drawn_and_labelled_on_every_panel():
    _, chart = _figure()
    for ax, diagonal in zip(chart.panels, chart.diagonals, strict=True):
        xs, ys = diagonal.get_xdata(), diagonal.get_ydata()
        assert list(xs) == list(ys) and diagonal.axes is ax
        assert any(t.get_text() == "perfectly honest" for t in ax.texts)


def test_the_error_bars_are_one_standard_error_of_honest_luck_at_the_stated_level():
    env, chart = _figure()
    for index, region, bars in chart.error_bars:
        n = next(e["n_trials"] for e in env["verdict"]["calibration"]["per_task"] if e["task"] == chart.tasks[index] and e["region"] == region)
        assert bars == [pytest.approx(math.sqrt(level * (1 - level) / n)) for level in cp.LEVELS]
    assert cp.honest_luck(0.9, 200) == pytest.approx(0.0212, abs=1e-4)  # 90% at 200 runs: about 2 points either way


def test_the_header_strip_states_each_regions_sharpness_beside_its_task():
    env, chart = _figure()
    for header, task in zip(chart.headers, chart.tasks, strict=True):
        text = header.get_text()
        assert text.startswith(f"task {task}") and "sharpness (mean 90% width):" in text
        for entry in env["verdict"]["sharpness"]["per_task"]:
            if entry["task"] == task:
                assert f"{entry['region']} {entry['mean_width_90']:.3g}" in text


def _spoiled(change):
    env = envelope()
    calibration, sharpness = copy.deepcopy(env["verdict"]["calibration"]), copy.deepcopy(env["verdict"]["sharpness"])
    change(calibration["per_task"][0], sharpness["per_task"][0], calibration, sharpness)
    return calibration, sharpness


@pytest.mark.parametrize(
    "change, match",
    [
        (lambda c, s, C, S: c.update(levels=[0.5, 0.8, 0.9]), "levels"),
        (lambda c, s, C, S: c.update(levels=[0.5, 0.8, 0.95, 0.9]), "levels"),
        (lambda c, s, C, S: c.update(coverage=[0.5, 0.8, 0.9]), "coverage"),
        (lambda c, s, C, S: c["coverage"].__setitem__(0, 1.2), "coverage"),
        (lambda c, s, C, S: c["coverage"].__setitem__(0, float("nan")), "coverage"),
        (lambda c, s, C, S: c["coverage"].__setitem__(0, True), "coverage"),
        (lambda c, s, C, S: c.update(n_trials=0), "n_trials"),
        (lambda c, s, C, S: c.update(n_trials=200.0), "n_trials"),
        (lambda c, s, C, S: s.update(mean_width_90=-1.0), "mean_width_90"),
        (lambda c, s, C, S: s.update(mean_width_90=float("inf")), "mean_width_90"),
        (lambda c, s, C, S: S["per_task"].pop(0), "no sharpness"),
        (lambda c, s, C, S: C["per_task"].pop(0), "no calibration"),
        (lambda c, s, C, S: C["per_task"].append(copy.deepcopy(c)), "two calibration"),
        (lambda c, s, C, S: S["per_task"].append(copy.deepcopy(s)), "two sharpness"),
        (lambda c, s, C, S: c.pop("per_dimension"), "calibration entry"),
        (lambda c, s, C, S: c.update(task=""), "non-empty"),
        (lambda c, s, C, S: C.update(per_task=[]) or S.update(per_task=[]), "no calibration entries"),
    ],
)
def test_a_block_that_does_not_add_up_is_refused(change, match):
    calibration, sharpness = _spoiled(change)
    with pytest.raises(cp.CalibrationBlockError, match=match):
        cp.calibration_panels(calibration, sharpness)


def test_render_refuses_before_writing_anything(tmp_path):
    env = envelope()
    env["verdict"]["calibration"]["per_task"][-1]["coverage"][0] = -0.1
    with pytest.raises(cp.CalibrationBlockError):
        cp.render_calibration_chart(env, tmp_path / "charts", tmp_path / "captions")
    assert not (tmp_path / "charts").exists()
    with pytest.raises(cp.CalibrationBlockError):
        cp.render_calibration_chart({"model_name": "x"}, tmp_path / "c", tmp_path / "t")


def test_realistic_extremes_render_coverage_of_none_and_all_three_regions_and_a_fixture(tmp_path):
    regions = ("training", "out-near-inverted", "out-large-action")
    calibration = {"per_task": [
        {"task": t, "region": r, "levels": list(cp.LEVELS), "coverage": cov, "n_trials": 200, "per_dimension": []}
        for t in ("dp-control", "dp-planning")
        for r, cov in zip(regions, ([0.5, 0.8, 0.9, 0.95], [0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]), strict=True)
    ]}
    sharpness = {"per_task": [{"task": e["task"], "region": e["region"], "mean_width_90": 1e-6 if e["coverage"][0] == 0 else 12.5}
                              for e in calibration["per_task"]]}
    env = {"model_name": "fx-overconfident", "is_fixture": True,
           "verdict": {"world": "pendulum", "calibration": calibration, "sharpness": sharpness}}
    paths = cp.render_calibration_chart(env, tmp_path / "c", tmp_path / "t")
    svg = paths.svg.read_text(encoding="utf-8")
    assert all(r in svg for r in regions) and "1e-06" in svg
    _, chart = _figure(env)
    for ax, (index, region, line) in ((chart.panels[i], (i, r, ln)) for i, r, ln in chart.region_lines):
        top = ax.transAxes.inverted().transform(ax.transData.transform((0.95, max(line.get_ydata()))))[1]
        assert top <= 0.8  # 100% coverage sits below the fixture label
    assert np.all(np.isfinite([b for _, _, bars in chart.error_bars for b in bars]))


def test_a_name_too_long_for_the_chart_is_refused_before_anything_is_written(tmp_path):
    env = envelope()
    for block in ("calibration", "sharpness"):
        for k, entry in enumerate(env["verdict"][block]["per_task"]):
            entry["region"] = f"{k}" + "r" * 40
    with pytest.raises(cp.CalibrationBlockError, match="longer than 40"):
        cp.render_calibration_chart(env, tmp_path / "c", tmp_path / "t")
    assert not (tmp_path / "c").exists()


@pytest.mark.parametrize("n_trials", [1, 10, 200])
def test_every_point_and_its_whole_error_bar_lie_inside_the_panel(n_trials):
    # review of P5-C03: a fixed 0..1.3 range clipped a zero-coverage point and the lower half of its bar
    env = envelope()
    for entry in env["verdict"]["calibration"]["per_task"]:
        entry.update(coverage=[1.0, 0.0, 0.0, 1.0], n_trials=n_trials)
    _, chart = _figure(env)
    for (index, _region, bars), (_i, _r, line) in zip(chart.error_bars, chart.region_lines, strict=True):
        low, high = chart.panels[index].get_ylim()
        assert all(low < c - b and c + b < high for c, b in zip(line.get_ydata(), bars, strict=True))


@pytest.mark.parametrize("n_trials", [1, 10, 200])
def test_the_fixture_sentence_hides_no_point_or_bar_however_few_the_runs(n_trials):
    env = envelope(name="fx-x", is_fixture=True)
    for entry in env["verdict"]["calibration"]["per_task"]:
        entry.update(coverage=[1.0, 1.0, 1.0, 1.0], n_trials=n_trials)
    _, chart = _figure(env)
    chart.figure.draw_without_rendering()
    for (index, _region, bars), (_i, _r, line) in zip(chart.error_bars, chart.region_lines, strict=True):
        ax = chart.panels[index]
        sentence = next(t for t in ax.texts if t.get_bbox_patch() is not None).get_bbox_patch().get_window_extent()
        tallest = max(c + b for c, b in zip(line.get_ydata(), bars, strict=True))
        assert ax.transData.transform((0.5, tallest))[1] < sentence.y0


def test_the_diagonals_label_sits_outside_the_plotted_area_clear_of_the_region_labels():
    # review of P5-C03, pass 3: inside the panel, an overconfident region's 50% point and bar ran through it
    env = envelope()
    for entry in env["verdict"]["calibration"]["per_task"]:
        entry["coverage"] = [0.45, 0.62, 0.74, 1.0]
    _, chart = _figure(env)
    chart.figure.draw_without_rendering()
    for ax in chart.panels:
        label = next(t for t in ax.texts if t.get_text() == cp.DIAGONAL_LABEL).get_window_extent()
        assert label.x0 >= ax.get_window_extent().x1  # right of the plotted area: no point, bar or line reaches it
        others = [t.get_window_extent() for t in ax.texts if t.get_text() != cp.DIAGONAL_LABEL]
        assert others and not any(label.overlaps(o) for o in others)


def test_at_full_coverage_the_diagonals_label_keeps_the_top_slot_at_its_line_end():
    # review of P5-C03, pass 4: regions at 100% tied with the diagonal's end and pushed its label down
    env = envelope()
    for entry in env["verdict"]["calibration"]["per_task"]:
        entry["coverage"] = [1.0, 1.0, 1.0, 1.0]
    _, chart = _figure(env)
    chart.figure.draw_without_rendering()
    for ax in chart.panels:
        labels = sorted((t for t in ax.texts if t.get_text() and t.get_bbox_patch() is None), key=lambda t: -t.get_window_extent().y1)
        assert labels[0].get_text() == cp.DIAGONAL_LABEL



def test_the_caption_speaks_of_the_90_percent_level():
    from wmj.reporting.captions import chart3_caption

    _, chart = _figure()
    assert chart.caption == chart3_caption(level=90) and "said it was 90% sure" in chart.caption
