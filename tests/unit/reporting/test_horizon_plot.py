"""Tests for wmj.reporting.horizon_plot — Chart 2 at P2-C05 scope.

Reporting ADR-R2 Chart 2: one panel per region from
`error_vs_horizon.per_region`; X starts at step 1 (index 0 is zero by
construction and log(0) is undefined); world-time secondary axis =
step x `dt`; Y log scale (TC-RP2-01's legibility clause); model curve
solid dark blue, world divergence reference black dashed, direct
labels. §4: reporting draws what it is handed and computes nothing.
"""

from __future__ import annotations

import numpy as np
import pytest

from wmj.reporting import style
from wmj.reporting.horizon_plot import (
    HorizonBlockError,
    build_horizon_figure,
    render_horizon_chart,
)


def _block(n_regions: int = 2, horizon: int = 20) -> dict:
    steps = list(range(horizon + 1))
    per_region = []
    for i in range(n_regions):
        growth = np.linspace(0.0, 1.0, horizon + 1) * (i + 1)
        per_region.append(
            {
                "region": ["training", "out-high-amplitude"][i],
                "steps": steps,
                "median_error": (growth * 0.1).tolist(),
                "divergence_reference": (growth * 0.01).tolist(),
            }
        )
    return {"dt": 0.02, "per_region": per_region}


def test_one_panel_per_region_in_declared_order():
    chart = build_horizon_figure(_block(2), model_label="persistence")
    assert len(chart.panels) == 2
    assert [ax.get_title() for ax in chart.panels] == ["training", "out-high-amplitude"]


def test_y_axis_is_log_scale_for_early_gap_legibility():
    chart = build_horizon_figure(_block(1), model_label="persistence")
    assert chart.panels[0].get_yscale() == "log"


def test_x_starts_at_step_one_not_zero_and_arrays_are_untouched():
    block = _block(1)
    before = list(block["per_region"][0]["median_error"])
    chart = build_horizon_figure(block, model_label="persistence")
    for line in chart.panels[0].get_lines():
        assert line.get_xdata()[0] == 1
    assert block["per_region"][0]["median_error"] == before


def test_colour_semantics_model_blue_solid_reference_black_dashed():
    chart = build_horizon_figure(_block(1), model_label="persistence")
    lines = chart.panels[0].get_lines()
    styles = {(line.get_color(), line.get_linestyle()) for line in lines}
    assert (style.MODEL_COLOUR, "-") in styles
    assert (style.DIVERGENCE_COLOUR, "--") in styles


def test_direct_labels_name_the_model_and_the_reference():
    chart = build_horizon_figure(_block(1), model_label="persistence")
    texts = {t.get_text() for t in chart.panels[0].texts}
    assert "persistence" in texts
    assert any("divergence" in t for t in texts)


def test_world_time_secondary_axis_is_step_times_dt():
    chart = build_horizon_figure(_block(1, horizon=50), model_label="persistence")
    secondary = chart.secondary_axes[0]
    assert "world time" in secondary.get_xlabel()
    forward, _ = chart.world_time_functions
    assert forward(50) == pytest.approx(1.0)  # 50 steps x 0.02


def test_fixture_flag_calls_mark_fixture_once_per_panel():
    chart = build_horizon_figure(_block(2), model_label="fx-broken", is_fixture=True)
    for ax in chart.panels:
        assert sum(t.get_text().split() == style.FIXTURE_LABEL.split() for t in ax.texts) == 1


def test_non_fixture_carries_no_fixture_label():
    chart = build_horizon_figure(_block(1), model_label="persistence")
    assert all(t.get_text() != style.FIXTURE_LABEL for t in chart.panels[0].texts)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda b: b.pop("dt"),
        lambda b: b["per_region"][0].pop("divergence_reference"),
        lambda b: b["per_region"][0]["steps"].__setitem__(0, 1),
        lambda b: b["per_region"][0]["median_error"].pop(),
        lambda b: b["per_region"][0]["median_error"].__setitem__(5, 0.0),
        lambda b: b.__setitem__("per_region", []),
        # code-review-001 I5: every disjunct of every guard has a case
        # that reaches it, so no branch of the fail-loud path is a phantom.
        lambda b: b.__setitem__("dt", -1.0),
        lambda b: b.__setitem__("dt", float("nan")),
        lambda b: b.__setitem__("dt", "0.02"),
        lambda b: b.__setitem__("per_region", {"training": {}}),
        lambda b: b["per_region"].__setitem__(0, ["not", "a", "dict"]),
        lambda b: b["per_region"].__setitem__(
            0,
            {**b["per_region"][0], "steps": [0], "median_error": [0.0], "divergence_reference": [0.0]},
        ),
    ],
    ids=[
        "no-dt", "no-reference", "steps-not-from-zero", "length-mismatch", "zero-on-log", "no-regions",
        "dt-negative", "dt-nan", "dt-wrong-type", "per-region-not-a-list", "entry-not-a-dict",
        "single-step",
    ],
)
def test_malformed_block_fails_loudly(mutate):
    block = _block(1)
    mutate(block)
    with pytest.raises(HorizonBlockError):
        build_horizon_figure(block, model_label="persistence")


def test_block_that_is_not_a_dict_fails_loudly():
    """code-review-001 I5: the `not isinstance(block, dict)` disjunct."""
    with pytest.raises(HorizonBlockError):
        build_horizon_figure(["not", "a", "block"], model_label="persistence")


def test_render_horizon_chart_writes_both_files(tmp_path):
    png, svg = tmp_path / "c.png", tmp_path / "c.svg"
    render_horizon_chart(_block(2), "persistence", png, svg)
    assert png.stat().st_size > 0 and svg.stat().st_size > 0
    body = svg.read_bytes()
    assert b"dc:date" not in body
    assert b"training" in body  # svg.fonttype none keeps labels as text


# --- P5-C03: Chart 2 at full scope — every model, baselines in grey, switch lines, full caption ----------


def test_tc_rp2_01_full_the_saved_chart_marks_each_switch_and_the_caption_states_the_reading_rule(tmp_path):
    from tests.unit.reporting._verdict_blocks import baseline_envelopes, envelope
    from wmj.reporting.captions import CHART2_FULL
    from wmj.reporting.horizon_plot import render_horizon_charts

    env = envelope()
    paths = render_horizon_charts(env, baseline_envelopes(), tmp_path / "charts", tmp_path / "captions")
    assert (paths.png.name, paths.svg.name, paths.caption.name) == ("lv-direct-horizon.png", "lv-direct-horizon.svg", "lv-direct-horizon.txt")
    svg = paths.svg.read_text(encoding="utf-8")
    switching = [e for e in env["verdict"]["climatology"]["per_task"] if e["switch_step"] is not None]
    assert switching  # the judge builder's tasks do switch
    for entry in switching:
        assert f"{entry['task']} switches to climate here (step {entry['switch_step']})" in svg
    assert "persistence" in svg and "linear" in svg and "world divergence" in svg and "(reference)" in svg
    caption = paths.caption.read_text(encoding="utf-8")
    assert caption == CHART2_FULL and "only the gap above that line is the model's fault" in caption


def test_tc_rp8_01_a_fixtures_horizon_chart_carries_the_fixture_sentence(tmp_path):
    from tests.unit.reporting._verdict_blocks import baseline_envelopes, envelope
    from wmj.reporting.horizon_plot import render_horizon_charts

    paths = render_horizon_charts(envelope(name="fx-brittle", is_fixture=True), baseline_envelopes(), tmp_path / "c", tmp_path / "t")
    svg = paths.svg.read_text(encoding="utf-8")
    from tests.unit.reporting._svg import says

    assert says(svg, style.FIXTURE_LABEL) and "FIXTURE: fx-brittle" in svg


def _full(env=None, baselines=None):
    from tests.unit.reporting._verdict_blocks import baseline_envelopes, envelope
    from wmj.reporting.horizon_plot import horizon_switches

    env = env or envelope()
    baselines = baseline_envelopes() if baselines is None else baselines
    chart = build_horizon_figure(
        env["verdict"]["error_vs_horizon"], env["model_name"], env["is_fixture"],
        baselines=tuple((b["model_name"], b["verdict"]["error_vs_horizon"]) for b in baselines),
        switches=horizon_switches(env["verdict"]["climatology"]),
    )
    return env, chart


def test_tc_ju3_01_the_model_and_the_world_reference_share_one_axes_and_one_unit():
    env, chart = _full()
    for index, ax in enumerate(chart.panels):
        assert chart.model_lines[index].axes is ax and chart.reference_lines[index].axes is ax
        entry = env["verdict"]["error_vs_horizon"]["per_region"][index]
        assert list(chart.model_lines[index].get_ydata()) == entry["median_error"][1:]
        assert list(chart.reference_lines[index].get_ydata()) == entry["divergence_reference"][1:]
    assert "normalised" in chart.panels[0].get_ylabel()  # one label, one unit, for both curves


def test_baselines_are_grey_lines_drawn_from_their_own_blocks_on_every_panel():
    from tests.unit.reporting._verdict_blocks import baseline_envelopes

    baselines = baseline_envelopes()
    _, chart = _full(baselines=baselines)
    assert len(chart.baseline_lines) == len(baselines) * len(chart.panels)
    for index, name, line in chart.baseline_lines:
        block = next(b for b in baselines if b["model_name"] == name)["verdict"]["error_vs_horizon"]
        assert line.get_color() == style.BASELINE_COLOUR and line.axes is chart.panels[index]
        assert list(line.get_ydata()) == block["per_region"][index]["median_error"][1:]


def test_each_switch_step_is_a_vertical_line_at_its_step_on_its_own_regions_panel():
    env, chart = _full()
    expected = {(e["region"], e["task"], e["switch_step"]) for e in env["verdict"]["climatology"]["per_task"] if e["switch_step"] is not None}
    drawn = {(chart.panels[i].get_title(), task, line.get_xdata()[0]) for i, task, line in chart.switch_lines}
    assert drawn == expected


def test_the_caption_points_at_switch_marks_only_when_there_are_some(tmp_path):
    from tests.unit.reporting._verdict_blocks import envelope
    from wmj.reporting.captions import CHART2_SCOPED
    from wmj.reporting.horizon_plot import render_horizon_charts

    env = envelope()
    for entry in env["verdict"]["climatology"]["per_task"]:  # a world with no switch step anywhere (predator-prey)
        entry.update(switch_step=None, agreement_mean_abs_z=None, agrees=None)
    paths = render_horizon_charts(env, [], tmp_path / "c", tmp_path / "t")
    assert paths.caption.read_text(encoding="utf-8") == CHART2_SCOPED
    assert "switches to climate here" not in paths.svg.read_text(encoding="utf-8")


def test_a_pendulum_like_curve_over_ten_decades_keeps_the_early_gap_legible():
    # TC-RP2-01's legibility clause: on a log axis the first 100 steps are not flattened by the late blow-up
    steps = list(range(5001))
    s = np.arange(5001)
    reference = np.r_[0.0, 1e-6 * np.exp(s[1:] / 300)]
    error = np.r_[0.0, 1e-4 * np.exp(s[1:] / 400)]
    block = {"dt": 0.01, "per_region": [{"region": "training", "steps": steps, "median_error": error.tolist(),
                                         "divergence_reference": reference.tolist()}]}
    chart = build_horizon_figure(block, "ensemble", switches={"training": [("dp-control", 4750)]})
    ax = chart.panels[0]
    to_axes = ax.transAxes.inverted().transform
    early_gap = abs(to_axes(ax.transData.transform((100, error[100])))[1] - to_axes(ax.transData.transform((100, reference[100])))[1])
    assert ax.get_yscale() == "log" and early_gap > 0.1  # at least a tenth of the panel between the curves at step 100


@pytest.mark.parametrize(
    "spoil, match",
    [
        (lambda b: b["verdict"]["error_vs_horizon"].update(dt=0.5), "same world"),
        (lambda b: b["verdict"]["error_vs_horizon"]["per_region"].pop(), "same world"),
        (lambda b: b["verdict"]["error_vs_horizon"]["per_region"][0]["steps"].append(99), "steps"),
        (lambda b: b["verdict"].update(world="pendulum"), "world"),
        (lambda b: b.update(is_fixture=True), "fixture"),
    ],
)
def test_a_baseline_from_another_world_or_run_is_refused(tmp_path, spoil, match):
    from tests.unit.reporting._verdict_blocks import baseline_envelopes, envelope
    from wmj.reporting.horizon_plot import render_horizon_charts

    baselines = baseline_envelopes()
    spoil(baselines[0])
    with pytest.raises(HorizonBlockError, match=match):
        render_horizon_charts(envelope(), baselines, tmp_path / "c", tmp_path / "t")
    assert not (tmp_path / "c").exists()


@pytest.mark.parametrize(
    "spoil, match",
    [
        (lambda c: c["per_task"][0].update(switch_step=0), "switch_step"),
        (lambda c: c["per_task"][0].update(switch_step=999), "outside"),
        (lambda c: c["per_task"][0].update(switch_step=True), "switch_step"),
        (lambda c: c["per_task"][0].update(region="nowhere", switch_step=1), "no panel"),
        (lambda c: c["per_task"][0].pop("agrees"), "climatology entry"),
        (lambda c: c.update(extra=1), "per_task"),
    ],
)
def test_a_switch_step_that_cannot_be_drawn_is_refused(tmp_path, spoil, match):
    from tests.unit.reporting._verdict_blocks import envelope
    from wmj.reporting.horizon_plot import render_horizon_charts

    env = envelope()
    spoil(env["verdict"]["climatology"])
    with pytest.raises(HorizonBlockError, match=match):
        render_horizon_charts(env, [], tmp_path / "c", tmp_path / "t")


def test_a_fixture_horizon_chart_leaves_the_top_of_each_panel_clear_for_its_label():
    from tests.unit.reporting._verdict_blocks import envelope

    _, chart = _full(envelope(name="fx-brittle", is_fixture=True))
    for index, ax in enumerate(chart.panels):
        top = max(max(chart.model_lines[index].get_ydata()), max(chart.reference_lines[index].get_ydata()),
                  *(max(line.get_ydata()) for i, _, line in chart.baseline_lines if i == index))
        assert ax.transAxes.inverted().transform(ax.transData.transform((1, top)))[1] <= 0.8


def test_a_baseline_run_over_a_shorter_horizon_is_refused_not_drawn_on_the_wrong_steps(tmp_path):
    from tests.unit.reporting._verdict_blocks import baseline_envelopes, envelope
    from wmj.reporting.horizon_plot import render_horizon_charts

    baselines = baseline_envelopes()
    for entry in baselines[0]["verdict"]["error_vs_horizon"]["per_region"]:  # a valid block, one step shorter
        for key in ("steps", "median_error", "divergence_reference"):
            entry[key] = entry[key][:-1]
    with pytest.raises(HorizonBlockError, match="same world, regions and steps"):
        render_horizon_charts(envelope(), baselines, tmp_path / "c", tmp_path / "t")


def test_line_labels_that_would_overprint_are_moved_apart():
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    fig = style.new_figure()
    ax = fig.subplots()
    ax.plot([0, 1], [1.0, 2.0])
    texts = style.direct_labels(ax, [(1, 2.0, "persistence", "grey"), (1, 2.0, "linear", "grey"), (1, 1.999, "direct", "blue")])
    renderer = FigureCanvasAgg(fig).get_renderer()
    boxes = [t.get_window_extent(renderer) for t in texts]
    assert not any(a.overlaps(b) for i, a in enumerate(boxes) for b in boxes[i + 1:])


# --- review of P5-C03, pass 1 ------------------------------------------------------------------------------


def _switch_texts(ax):
    return [t for t in ax.texts if "switches to climate here" in t.get_text()]


def _pendulum_block(regions=("training", "out-near-inverted", "out-large-action"), horizon=5000):
    s = np.arange(horizon + 1)
    return {"dt": 0.01, "per_region": [
        {"region": r, "steps": list(range(horizon + 1)),
         "median_error": [0.0, *(1e-4 * np.exp(s[1:] / (400 + 50 * i))).tolist()],
         "divergence_reference": [0.0, *(1e-6 * np.exp(s[1:] / (300 + 50 * i))).tolist()]}
        for i, r in enumerate(regions)]}


@pytest.mark.parametrize("is_fixture", [False, True])
@pytest.mark.parametrize("steps", [(1, 1), (1, 2), (4750, 4900), (4999, 5000), (5000, 5000), (2500, 2501)])
def test_switch_labels_never_overprint_each_other_the_fixture_sentence_or_the_axis(steps, is_fixture):
    # review of P5-C03, I1: two tasks switching at one step, or a step apart, printed one label on top of the other
    block = _pendulum_block()
    switches = {r["region"]: [("dp-control", steps[0]), ("dp-planning", steps[1])] for r in block["per_region"]}
    chart = build_horizon_figure(block, "fx-x" if is_fixture else "ensemble", is_fixture, switches=switches)
    for ax in chart.panels:
        labels = _switch_texts(ax)
        assert sorted(t.get_text().split()[0] for t in labels) == ["dp-control", "dp-planning"]
        boxes = [t.get_window_extent() for t in labels]
        inside = ax.get_window_extent()
        assert all(inside.x0 <= b.x0 and b.x1 <= inside.x1 for b in boxes)  # never over the tick labels or the next panel
        assert not boxes[0].overlaps(boxes[1])
        others = [t.get_bbox_patch().get_window_extent() for t in ax.texts if t.get_bbox_patch() is not None]
        assert len(others) == is_fixture and not any(b.overlaps(o) for b in boxes for o in others)


def test_switch_labels_that_cannot_all_be_placed_readably_are_refused():
    block = _pendulum_block(regions=("training",), horizon=20)
    crowd = [(f"task-{k}", 1) for k in range(8)]  # far more labels at one step than the panel has room for
    with pytest.raises(HorizonBlockError, match="cannot be placed"):
        build_horizon_figure(block, "ensemble", switches={"training": crowd})


def test_a_two_line_label_and_its_neighbour_never_overprint_and_none_falls_below_the_panel():
    # review of P5-C03, I2: the gap was one line's height, so "(reference)" sat on the model's name
    fig = style.new_figure()
    ax = fig.subplots()
    ax.plot([0, 1], [0.0, 1.0])
    ax.set_ylim(0.0, 1.0)
    items = [(1, 0.01, "world divergence\n(reference)", "black"), (1, 0.0, "direct", "blue"), (1, 0.0, "persistence", "grey"),
             (1, 0.005, "two\nlines", "grey")]
    texts = style.direct_labels(ax, items, fontsize=7.5)
    boxes = [t.get_window_extent() for t in texts]
    assert not any(a.overlaps(b) for i, a in enumerate(boxes) for b in boxes[i + 1:])
    assert min(b.y0 for b in boxes) >= ax.get_window_extent().y0 - 1.0


def test_climatology_and_baseline_names_that_are_not_text_are_refused():
    from tests.unit.reporting._verdict_blocks import envelope
    from wmj.reporting.horizon_plot import horizon_switches

    climatology = envelope()["verdict"]["climatology"]
    climatology["per_task"][0]["task"] = 5
    with pytest.raises(HorizonBlockError, match="must be text"):
        horizon_switches(climatology)
    for name in (None, "", "  "):
        with pytest.raises(HorizonBlockError, match="needs a name"):
            build_horizon_figure(_block(), "ensemble", baselines=((name, _block()),))


def test_a_baseline_models_own_chart_draws_it_once_not_again_in_grey(tmp_path):
    from tests.unit.reporting._verdict_blocks import baseline_envelopes
    from wmj.reporting.horizon_plot import render_horizon_charts

    baselines = baseline_envelopes()
    with_self = render_horizon_charts(baselines[0], baselines, tmp_path / "a", tmp_path / "at")
    without = render_horizon_charts(baselines[0], baselines[1:], tmp_path / "b", tmp_path / "bt")
    assert with_self.svg.read_bytes() == without.svg.read_bytes()


@pytest.mark.parametrize("length", [12, 30, 40])
def test_a_long_task_name_on_a_fixtures_chart_is_placed_clear_of_the_fixture_sentence(length):
    block = _pendulum_block()
    tasks = ("c" * length, "p" * length)
    switches = {r["region"]: [(tasks[0], 4750), (tasks[1], 4900)] for r in block["per_region"]}
    chart = build_horizon_figure(block, "fx-x", True, switches=switches)
    for ax in chart.panels:
        labels = _switch_texts(ax)
        assert sorted(t.get_text().split()[0] for t in labels) == sorted(tasks)
        assert sorted(t.get_text().split()[-1] for t in labels) == ["4750)", "4900)"]  # each label names its own step
        fixture = next(t for t in ax.texts if t.get_bbox_patch() is not None).get_bbox_patch().get_window_extent()
        boxes = [t.get_window_extent() for t in labels]
        assert not boxes[0].overlaps(boxes[1]) and not any(b.overlaps(fixture) for b in boxes)
