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
from tests.unit.reporting._svg import says
from wmj.judge.exceptions import compute_exceptions_and_trials
from wmj.reporting import exception_plot as ep
from wmj.reporting import style
from wmj.reporting.captions import format_expected


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
        expected = f"expected: {format_expected(entry['expected'])} of {entry['n_trials']}"
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
        assert says(svg, style.FIXTURE_LABEL) and "FIXTURE: fx-brittle" in svg
    plain = ep.render_exception_charts(_envelope(), tmp_path / "c2", tmp_path / "t2")
    assert not any(says(p.svg.read_text(encoding="utf-8"), style.FIXTURE_LABEL) for p in plain)


# --- what the figure draws (before saving) -------------------------------------------------------------


def _one(env=None, index=0):
    env = env or _envelope()
    exceptions, trials = env["verdict"]["exceptions"], env["verdict"]["trials"]
    pairs = ep.chart1_panels(exceptions, trials)
    entry, trial = pairs[index]
    return entry, trial, ep.build_exception_figure(entry, trial, env["model_name"], env["is_fixture"])


def test_the_dots_are_the_verdicts_distances_in_trial_order_and_the_line_is_band_hi():
    entry, trial, chart = _one()
    x, y = chart.dots.get_offsets().T
    assert list(x) == list(range(entry["n_trials"])) and list(y) == trial["outcome_distance"]
    assert list(chart.line.get_ydata()) == trial["band_hi"]


def test_misses_are_circled_exactly_where_the_verdict_flags_them_never_re_derived():
    env = _envelope()
    trial = env["verdict"]["trials"]["per_task"][0]
    entry = env["verdict"]["exceptions"]["per_task"][0]
    # a flag that disagrees with "dot above the line" is still drawn as the verdict says (judge §5 warning)
    trial["is_exception"] = [not f for f in trial["is_exception"]]
    entry["observed"] = sum(trial["is_exception"])
    entry["band"] = ep._band_of(entry["observed"], entry["bands"])
    chart = ep.build_exception_figure(entry, trial, "direct", False)
    circled = [round(v) for v in chart.misses.get_offsets()[:, 0]]
    assert circled == [i for i, f in enumerate(trial["is_exception"]) if f]
    assert chart.misses.get_edgecolor()[0][:3] == pytest.approx(tuple(int(style.EXCEPTION_COLOUR[i:i + 2], 16) / 255 for i in (1, 3, 5)))


def test_the_header_strip_reads_like_the_spec_and_adds_the_hedging_note_when_flagged():
    entry, trial, chart = _one()
    assert chart.header.get_text() == (
        f"region: {entry['region']} · observed exceptions: {entry['observed']} · "
        f"expected: {format_expected(entry['expected'])} of {entry['n_trials']} · band: {entry['band'].upper()}"
    )
    entry = dict(entry, low_side_sharpness_flag=True)
    chart = ep.build_exception_figure(entry, trial, "direct", False)
    assert chart.header.get_text().endswith(f"\n{entry['observed']} misses — but see sharpness: possible hedging")


def test_the_inset_scale_marks_the_observed_count_inside_its_band():
    entry, _, chart = _one()
    assert chart.marker.get_ydata()[0] == entry["observed"]
    lo, hi = chart.inset.get_ylim()
    assert lo <= entry["observed"] <= hi and lo <= 0
    assert chart.inset.get_position().x0 > chart.panel.get_position().x1  # on the right, as the caption says


def test_a_fixture_figure_is_marked_and_named_and_an_ordinary_one_is_not():
    *_, chart = _one(_envelope(name="fx-brittle", is_fixture=True))
    assert style.has_fixture_mark(chart.figure)
    assert "FIXTURE: fx-brittle" in chart.figure.get_suptitle()
    *_, plain = _one()
    assert not style.has_fixture_mark(plain.figure) and style.fixture_traces(plain.figure) == []


# --- refusals: draw nothing rather than part of it (reporting §7) --------------------------------------


def _mutated(change):
    env = _envelope()
    exceptions, trials = copy.deepcopy(env["verdict"]["exceptions"]), copy.deepcopy(env["verdict"]["trials"])
    change(exceptions["per_task"][0], trials["per_task"][0], exceptions, trials)
    return exceptions, trials


@pytest.mark.parametrize(
    "change, match",
    [
        (lambda e, t, E, T: e.pop("band"), "keys"),
        (lambda e, t, E, T: t.update(extra=1), "keys"),
        (lambda e, t, E, T: t["outcome_distance"].pop(), "length"),
        (lambda e, t, E, T: e.update(n_trials=e["n_trials"] + 1), "length"),
        (lambda e, t, E, T: e.update(observed=e["observed"] + 1), "observed"),
        (lambda e, t, E, T: e.update(band="blue"), "band"),
        (lambda e, t, E, T: e.update(low_side_sharpness_flag=1), "flag"),
        (lambda e, t, E, T: t["is_exception"].__setitem__(0, 1), "is_exception"),
        (lambda e, t, E, T: t["outcome_distance"].__setitem__(0, float("nan")), "distance"),
        (lambda e, t, E, T: t["outcome_distance"].__setitem__(0, -0.5), "distance"),
        (lambda e, t, E, T: t["band_hi"].__setitem__(0, float("inf")), "distance"),
        (lambda e, t, E, T: t["band_lo"].__setitem__(0, 0.1), "band_lo"),
        (lambda e, t, E, T: T["per_task"].pop(0), "no trials entry"),
        (lambda e, t, E, T: T["per_task"].append(copy.deepcopy(t)), "trials entries"),
        (lambda e, t, E, T: E.update(extra=1), "per_task"),
        (lambda e, t, E, T: E["per_task"].pop(0), "no exceptions entry"),
        (lambda e, t, E, T: e.update(band={"green": "amber", "amber": "red", "red": "green"}[e["band"]]), "disagrees"),
        (lambda e, t, E, T: e["bands"].update(amber=[[0, 0], [5, 9]]), "touching"),
        (lambda e, t, E, T: (e.update(task=["x"]), t.update(task=["x"])), "text"),
        (lambda e, t, E, T: e.update(expected=e["n_trials"] + 1.0), "expected"),
        (lambda e, t, E, T: e.update(bands={"green": [1, 1]}), "bands"),
        (lambda e, t, E, T: (e.update(horizon_step=0), t.update(horizon_step=0)), "horizon_step"),
        (lambda e, t, E, T: (e.update(horizon_step=True), t.update(horizon_step=True)), "horizon_step"),
    ],
)
def test_a_block_that_does_not_add_up_is_refused(change, match):
    exceptions, trials = _mutated(change)
    with pytest.raises(ep.ExceptionBlockError, match=match):
        ep.chart1_panels(exceptions, trials)


def test_render_refuses_before_writing_anything(tmp_path):
    env = _envelope()
    env["verdict"]["trials"]["per_task"][-1]["outcome_distance"][0] = -1.0  # the last panel is bad
    with pytest.raises(ep.ExceptionBlockError):
        ep.render_exception_charts(env, tmp_path / "charts", tmp_path / "captions")
    assert not (tmp_path / "charts").exists() and not (tmp_path / "captions").exists()


@pytest.mark.parametrize("bad", [{"model_name": "x"}, None, {"model_name": "", "is_fixture": False, "verdict": {}},
                                 {"model_name": "direct", "is_fixture": 1, "verdict": {}}])
def test_render_refuses_an_envelope_without_the_fields_it_draws(tmp_path, bad):
    with pytest.raises(ep.ExceptionBlockError):
        ep.render_exception_charts(bad, tmp_path / "c", tmp_path / "t")


# --- realistic-fixture variants (TDD-3): real trial counts and awkward data ----------------------------


def _n200_blocks(distances, line=1.0):
    n = len(distances)
    flags = [bool(d > line) for d in distances]
    observed = sum(flags)
    band = "green" if 12 <= observed <= 29 else "amber" if 8 <= observed <= 35 else "red"
    key = {"task": "dp-control", "region": "out-near-inverted", "horizon_step": 4750}
    exceptions = {"per_task": [{**key, "n_trials": n, "expected": 20.0, "observed": observed, "band": band,
                                "low_side_sharpness_flag": False,
                                "bands": {"green": [12, 29], "amber": [[8, 11], [30, 35]], "red": "outside"}}]}
    trials = {"per_task": [{**key, "distance_unit": "stated-size", "outcome_distance": [float(d) for d in distances],
                            "band_lo": [0.0] * n, "band_hi": [line] * n, "is_exception": flags}]}
    return exceptions, trials


@pytest.mark.parametrize(
    "distances",
    [
        np.r_[np.full(190, 0.3), np.full(10, 1e6)],  # huge outliers: misses a million stated sizes away
        np.zeros(200),  # every outcome exactly on the guess: zero misses
        np.full(200, 5.0),  # every run a miss
        np.r_[np.full(180, 0.5), np.full(20, 2.0)],  # an honest-looking 20 of 200
    ],
)
def test_two_hundred_trial_blocks_with_awkward_data_render_and_the_numbers_survive(tmp_path, distances):
    exceptions, trials = _n200_blocks(distances)
    env = {"model_ref": "o", "model_name": "ensemble", "is_fixture": False,
           "verdict": {"world": "pendulum", "exceptions": exceptions, "trials": trials}, "meta": {}}
    (paths,) = ep.render_exception_charts(env, tmp_path / "c", tmp_path / "t")
    entry = exceptions["per_task"][0]
    svg = paths.svg.read_text(encoding="utf-8")
    assert f"observed exceptions: {entry['observed']}" in svg and "expected: 20 of 200" in svg
    assert paths.svg.name == "pendulum-ensemble-exceptions-dp-control-out-near-inverted-h4750.svg"
    chart = ep.build_exception_figure(entry, trials["per_task"][0], "ensemble", False)
    lo, hi = chart.inset.get_ylim()
    assert lo <= entry["observed"] <= hi


def test_a_three_region_verdict_gives_one_chart_per_region_task_and_step(tmp_path):
    from tests.unit.judge._builders import judge_input_kwargs
    from wmj.judge.types import (
        Bands,
        JudgeInput,
        RegionClimatology,
        RegionCurve,
        RegionLabel,
        Thresholds,
    )

    kw = judge_input_kwargs()
    names = ("training", "out-a", "out-b")
    kw["region_labels"] = tuple(RegionLabel(names[i % 3], None if i % 3 == 0 else "state") for i in range(len(kw["region_labels"])))
    curve = kw["divergence_curves"][0].curve
    kw["divergence_curves"] = tuple(RegionCurve(n, curve) for n in names)
    kw["climatology"] = tuple(RegionClimatology(n, kw["climatology"][0].bins) for n in names)
    old = kw["thresholds"]  # 6 trials over 3 regions: bands for n = 2 per region
    kw["thresholds"] = Thresholds(Bands(n=2, p=0.1, green=(1, 1), amber_outer=(0, 2)), old.sharpness_hedge_threshold, old.agreement_threshold)
    env = _envelope(inp=JudgeInput(**kw))
    written = ep.render_exception_charts(env, tmp_path / "c", tmp_path / "t")
    assert len(written) == len(env["verdict"]["exceptions"]["per_task"]) and len(written) >= 3 * 2
    assert any("out-a" in p.svg.name for p in written) and any("out-b" in p.svg.name for p in written)


def test_the_top_of_the_panel_is_left_empty_so_the_fixture_label_never_hides_a_dot():
    # seen on the first render: the label box covered a miss 10,000 stated sizes out
    for distances in (np.r_[np.full(199, 0.5), [1e4]], np.r_[np.full(199, 0.5), [3.0]]):
        exceptions, trials = _n200_blocks(distances)
        chart = ep.build_exception_figure(exceptions["per_task"][0], trials["per_task"][0], "fx-x", True)
        ax = chart.panel
        to_axes = ax.transAxes.inverted().transform
        top_dot = to_axes(ax.transData.transform((0, max(distances))))[1]
        assert top_dot <= 0.8, top_dot  # the label sits in the top ~12% of the panel


def test_the_inset_also_marks_the_expected_count():
    entry, _, chart = _one()
    lines = [ln for ln in chart.inset.get_lines() if ln is not chart.marker]
    assert any(list(ln.get_ydata()) == [entry["expected"], entry["expected"]] for ln in lines)


def test_the_axis_turns_logarithmic_above_the_line_only_when_a_miss_is_far_out():
    far, near = _n200_blocks(np.r_[np.full(199, 0.5), [1e6]]), _n200_blocks(np.r_[np.full(199, 0.5), [3.0]])
    chart = ep.build_exception_figure(far[0]["per_task"][0], far[1]["per_task"][0], "direct", False)
    assert chart.panel.get_yscale() == "symlog" and "log scale above the line" in chart.panel.get_ylabel()
    chart = ep.build_exception_figure(near[0]["per_task"][0], near[1]["per_task"][0], "direct", False)
    assert chart.panel.get_yscale() == "linear" and "log" not in chart.panel.get_ylabel()


@pytest.mark.parametrize("region", ["training", "out-near-inverted", "out-high-amplitude"])
def test_the_header_with_the_hedging_note_fits_inside_the_panel_and_clear_of_the_scale(region):
    # review pass 1: on one line the §7 note ran off the figure and into the scale's title
    exceptions, trials = _n200_blocks(np.zeros(200))
    entry = dict(exceptions["per_task"][0], region=region, low_side_sharpness_flag=True, band="red", observed=0)
    trial = dict(trials["per_task"][0], region=region)
    chart = ep.build_exception_figure(entry, trial, "direct", False)
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    renderer = FigureCanvasAgg(chart.figure).get_renderer()
    header = chart.header.get_window_extent(renderer)
    panel = chart.panel.get_window_extent(renderer)
    scale_title = chart.inset.title.get_window_extent(renderer)
    assert header.x0 >= panel.x0 - 1 and header.x1 <= panel.x1 and header.x1 < scale_title.x0
    assert "possible hedging" in chart.header.get_text().splitlines()[-1]


@pytest.mark.parametrize("far", [1e215, 1.7e308])
def test_a_miss_near_the_largest_float_is_drawn_not_crashed_on(tmp_path, far):
    # review pass 1: the log-axis headroom overflowed for misses beyond about 1e212 stated sizes
    exceptions, trials = _n200_blocks(np.r_[np.full(199, 0.5), [far]])
    env = {"model_ref": "o", "model_name": "direct", "is_fixture": False,
           "verdict": {"world": "lv", "exceptions": exceptions, "trials": trials}, "meta": {}}
    (paths,) = ep.render_exception_charts(env, tmp_path / "c", tmp_path / "t")
    chart = ep.build_exception_figure(exceptions["per_task"][0], trials["per_task"][0], "direct", False)
    assert chart.panel.get_ylim()[1] >= far and paths.svg.exists()


def test_misses_and_the_scale_marker_are_drawn_whole_at_the_edges():
    exceptions, trials = _n200_blocks(np.r_[[5.0], np.full(198, 0.5), [5.0]])
    chart = ep.build_exception_figure(exceptions["per_task"][0], trials["per_task"][0], "direct", False)
    assert chart.misses.get_clip_on() is False and chart.marker.get_clip_on() is False


def test_build_refuses_a_malformed_pair_with_its_own_error_not_a_key_error():
    with pytest.raises(ep.ExceptionBlockError, match="keys"):
        ep.build_exception_figure({}, {}, "direct", False)


def test_with_no_misses_no_red_is_drawn_anywhere_on_the_image(tmp_path):
    # seen on a render: an empty, unclipped circle layer stamped a red half-circle in the figure's corner
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    exceptions, trials = _n200_blocks(np.zeros(200))
    chart = ep.build_exception_figure(exceptions["per_task"][0], trials["per_task"][0], "direct", False)
    canvas = FigureCanvasAgg(chart.figure)
    canvas.draw()
    pixels = np.asarray(canvas.buffer_rgba())[..., :3].astype(int)
    red = np.array([int(style.EXCEPTION_COLOUR[i:i + 2], 16) for i in (1, 3, 5)])
    assert not np.any(np.all(np.abs(pixels - red) <= 40, axis=-1))


def test_on_a_log_axis_the_zero_and_the_line_never_overprint():
    # review pass 2: with misses 1e150 stated sizes out, "0" and "10^0" overprinted into a false "100"
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    exceptions, trials = _n200_blocks(np.r_[np.full(175, 0.5), np.full(25, 1e150)])
    chart = ep.build_exception_figure(exceptions["per_task"][0], trials["per_task"][0], "direct", False)
    renderer = FigureCanvasAgg(chart.figure).get_renderer()
    labels = [t for t in chart.panel.get_yticklabels() if t.get_text()]
    boxes = [t.get_window_extent(renderer) for t in labels]
    for i, a in enumerate(boxes):
        for b in boxes[i + 1:]:
            assert not a.overlaps(b), [t.get_text() for t in labels]


def test_a_region_name_too_long_for_the_header_is_refused(tmp_path):
    env = _envelope()
    for block in ("exceptions", "trials"):
        for entry in env["verdict"][block]["per_task"]:
            entry["region"] = "r" * 41
    with pytest.raises(ep.ExceptionBlockError, match="40"):
        ep.render_exception_charts(env, tmp_path / "c", tmp_path / "t")
