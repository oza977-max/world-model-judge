"""Chart 4, the comparison table: which rows are marked, what the caption says, and what is refused.

In plain words: the table is the project's headline. These tests pin down:
- which task and region the ordinary-error rank comes from;
- that a tie is never called a disagreement;
- that a fixture is never marked or named;
- which pair the caption names;
- that malformed input is refused, never drawn.
"""

from __future__ import annotations

import copy

import pytest

from tests.unit.reporting._verdict_blocks import (
    full_envelope,
    run_envelopes,
    set_crps,
    set_horizon,
)
from wmj.reporting import comparison_table as ct
from wmj.reporting import style
from wmj.reporting.captions import chart4_caption


def _flat(envelopes):
    for env in envelopes:
        for entry in env["verdict"]["trust_horizons"]["per_task"]:
            set_horizon(env, 0, entry["task"], entry["region"])
    return envelopes


def _named(envelopes):
    return {e["model_name"]: e for e in envelopes}


def _rows():
    envelopes = _flat(run_envelopes())
    for rank, env in enumerate(envelopes):
        set_crps(env, 0.1 * (rank + 1))  # persistence, linear, direct, ensemble, fx-brittle
    return envelopes


def test_the_rank_comes_from_the_control_task_in_the_training_region_and_says_so():
    envelopes = _rows()
    set_crps(_named(envelopes)["linear"], 9.0, "lv-planning", "training")  # another task's error is ignored
    comparison = ct.comparison_rows(envelopes)
    assert comparison.control_task == "lv-control"  # tolerance 0.1 against lv-planning's 0.3
    assert [r.name for r in comparison.rows] == ["persistence", "linear", "direct", "ensemble", "fx-brittle"]
    assert comparison.ranks == (1, 2, 3, 4, 5)
    figure = ct.build_comparison_figure(comparison)
    header = figure.table[0, 1].get_text().get_text()
    assert "lv-control" in header and "training" in header and "one step" in header


def test_equal_errors_share_a_rank():
    envelopes = _rows()
    set_crps(_named(envelopes)["direct"], 0.2)  # level with linear
    comparison = ct.comparison_rows(envelopes)
    assert dict(zip([r.name for r in comparison.rows], comparison.ranks, strict=True)) == {
        "persistence": 1, "direct": 2, "linear": 2, "ensemble": 4, "fx-brittle": 5}


def test_columns_put_training_first_in_each_task_then_the_other_regions():
    comparison = ct.comparison_rows(_rows())
    assert comparison.columns == (("lv-control", "training"), ("lv-control", "out-of-range"),
                                  ("lv-planning", "training"), ("lv-planning", "out-of-range"))


@pytest.mark.parametrize(
    "steps, crps, marked",
    [
        ((1, 2), (0.30, 0.40), {"direct", "ensemble"}),  # error says direct, the judge says ensemble
        ((2, 1), (0.30, 0.40), set()),  # both say direct
        ((1, 1), (0.30, 0.40), set()),  # a horizon tie is not a disagreement
        ((1, 2), (0.30, 0.30), set()),  # an error tie is not a disagreement
        ((2, 1), (0.30, 0.30), set()),  # whichever way round the horizons differ
    ],
)
def test_a_row_is_marked_only_for_a_strict_disagreement(steps, crps, marked):
    envelopes = _rows()
    by = _named(envelopes)
    for name in ("persistence", "linear"):  # ahead on error, and at least level on every horizon: never discordant
        set_horizon(by[name], 2, "lv-planning")
    for name, s, c in zip(("direct", "ensemble"), steps, crps, strict=True):
        set_crps(by[name], c)
        set_horizon(by[name], s, "lv-planning")
    comparison = ct.comparison_rows(envelopes)
    assert {r.name for r, m in zip(comparison.rows, comparison.marked, strict=True) if m} == marked
    assert comparison.caption_pair == (("direct", "ensemble") if marked else None)


def test_a_fixture_is_never_marked_nor_named_however_it_ranks():
    envelopes = _rows()
    fixture = _named(envelopes)["fx-brittle"]
    set_crps(fixture, 0.01)  # best on ordinary error
    set_horizon(fixture, 0, "lv-planning")
    for name, steps in (("persistence", 2), ("linear", 2), ("direct", 2), ("ensemble", 1)):  # real models concordant
        set_horizon(_named(envelopes)[name], steps, "lv-planning")
    comparison = ct.comparison_rows(envelopes)
    assert not any(comparison.marked) and comparison.caption_pair is None
    chart = ct.build_comparison_figure(comparison)
    assert chart.caption == chart4_caption(None)


def test_the_caption_names_the_disagreeing_pair_closest_on_ordinary_error_better_first():
    envelopes = _rows()
    by = _named(envelopes)
    set_crps(by["persistence"], 0.50)
    set_crps(by["linear"], 1.00)
    set_crps(by["direct"], 0.41)
    set_crps(by["ensemble"], 0.40)
    for name, steps in (("persistence", 2), ("linear", 2), ("direct", 2), ("ensemble", 1)):
        set_horizon(by[name], steps, "lv-planning")
    comparison = ct.comparison_rows(envelopes)
    # ensemble disagrees with all three; it is closest to direct (0.41 / 0.40), and has the lower error
    assert comparison.caption_pair == ("ensemble", "direct")
    marked = {r.name for r, m in zip(comparison.rows, comparison.marked, strict=True) if m}
    assert marked == {"ensemble", "direct", "persistence", "linear"}


def test_a_world_with_no_natural_cycle_shows_a_dash_not_a_zero():
    envelopes = _rows()
    for env in envelopes:
        for entry in env["verdict"]["trust_horizons"]["per_task"]:
            entry["natural_units"] = None
    chart = ct.build_comparison_figure(ct.comparison_rows(envelopes))
    assert chart.table[1, 2].get_text().get_text().endswith("—")


def test_fixture_rows_are_labelled_tinted_and_the_figure_carries_the_fixture_sentence():
    chart = ct.build_comparison_figure(ct.comparison_rows(_rows()))
    names = [chart.table[r, 0].get_text().get_text() for r in range(1, 6)]
    fixture_row = names.index("FIXTURE: fx-brittle") + 1
    assert chart.table[fixture_row, 0].get_text().get_color() == style.FIXTURE_COLOUR
    assert chart.table[fixture_row, 3].get_facecolor()[:3] != chart.table[1, 3].get_facecolor()[:3]
    assert style.has_fixture_mark(chart.figure)


def test_a_run_without_fixtures_carries_no_fixture_sentence(tmp_path):
    envelopes = [e for e in _rows() if not e["is_fixture"]]
    paths = ct.render_comparison_chart(envelopes, tmp_path / "c", tmp_path / "t")
    assert style.FIXTURE_LABEL not in paths.svg.read_text(encoding="utf-8")


def _spoil(change):
    envelopes = _rows()
    change(envelopes)
    return envelopes


@pytest.mark.parametrize(
    "change, match",
    [
        (lambda es: es[1]["verdict"].update(world="pendulum"), "one table per world"),
        (lambda es: es.append(copy.deepcopy(es[0])), "appears twice"),
        (lambda es: es[2]["verdict"]["trust_horizons"]["per_task"].pop(0), "different tasks"),
        (lambda es: es[2]["verdict"]["skill"]["per_task_region"].__setitem__(
            1, {**es[2]["verdict"]["skill"]["per_task_region"][1], "region": "elsewhere"}), "no skill entry"),
        (lambda es: [e["verdict"]["trust_horizons"]["per_task"][i].update(tolerance=0.1)
                     for e in es for i in range(4)], "ambiguous"),
        (lambda es: es[0]["verdict"]["trust_horizons"]["per_task"][0].update(tolerance=0.2), "two tolerances"),
        (lambda es: es[0]["verdict"]["trust_horizons"]["per_task"][0].update(steps=-1), "valid length"),
        (lambda es: es[0]["verdict"]["trust_horizons"]["per_task"][0].update(steps=1.0), "valid length"),
        (lambda es: es[0]["verdict"]["trust_horizons"]["per_task"][0].update(natural_units=3), "natural_units"),
        (lambda es: es[0]["verdict"]["skill"]["per_task_region"][1].update(crps=float("nan")), "finite"),
        (lambda es: es[0]["verdict"]["skill"]["per_task_region"][1].update(vs_linear=None), "finite"),
        (lambda es: es[0]["verdict"]["skill"].update(extra=[]), "skill"),
        (lambda es: es[0]["verdict"]["trust_horizons"]["per_task"][0].pop("tolerance"), "trust_horizons entry"),
        (lambda es: [e["verdict"]["trust_horizons"]["per_task"][i].update(region="r2")
                     for e in es for i in (1, 3)], "no 'training'"),
        (lambda es: es.__setitem__(slice(None), [e for e in es if e["is_fixture"]]), "every model is a fixture"),
        (lambda es: es[0].update(model_name="bad name"), "file name"),
    ],
)
def test_input_that_does_not_add_up_is_refused(change, match):
    with pytest.raises(ct.ComparisonBlockError, match=match):
        ct.comparison_rows(_spoil(change))


def test_render_refuses_before_writing_anything(tmp_path):
    with pytest.raises(ct.ComparisonBlockError):
        ct.render_comparison_chart([], tmp_path / "c", tmp_path / "t")
    assert not (tmp_path / "c").exists()


def test_a_realistic_world_three_regions_six_models_two_fixtures_and_long_names(tmp_path):
    regions = ("training", "out-near-inverted", "out-large-action")
    names = [("persistence", False), ("linear", False), ("direct-predicted-error", False),
             ("self-predicted-ens", False), ("fx-overconfident", True), ("f" * 40, True)]
    envelopes = []
    for i, (name, fixture) in enumerate(names):
        env = full_envelope(name=name, is_fixture=fixture, seed=10 + i, world="pendulum")
        skill = env["verdict"]["skill"]["per_task_region"]
        horizons = env["verdict"]["trust_horizons"]["per_task"]
        template_s, template_h = skill[1], horizons[1]  # (control, training) entries
        env["verdict"]["skill"]["per_task_region"] = [
            {**template_s, "task": t, "region": r} for t in ("dp-control", "dp-planning") for r in regions]
        env["verdict"]["trust_horizons"]["per_task"] = [
            {**template_h, "task": t, "region": r, "tolerance": 0.1 if t == "dp-control" else 0.3, "natural_units": None,
             "steps": i % 3, "world_time": (i % 3) * 0.02} for t in ("dp-control", "dp-planning") for r in regions]
        envelopes.append(env)
    paths = ct.render_comparison_chart(envelopes, tmp_path / "c", tmp_path / "t")
    svg = paths.svg.read_text(encoding="utf-8")
    assert "out-large-action" in svg and "FIXTURE: " + "f" * 40 in svg and "—" in svg
    chart = ct.build_comparison_figure(ct.comparison_rows(envelopes))
    chart.figure.draw_without_rendering()
    for (r, c), cell in chart.table.get_celld().items():  # every cell's text fits inside its cell
        text, box = cell.get_text().get_window_extent(), cell.get_window_extent()
        assert box.x0 - 1 <= text.x0 and text.x1 <= box.x1 + 1 and box.y0 - 1 <= text.y0 and text.y1 <= box.y1 + 1, (r, c)


def test_a_skill_a_hair_below_zero_reads_as_zero_not_minus_zero():
    envelopes = _rows()
    for entry in _named(envelopes)["direct"]["verdict"]["skill"]["per_task_region"]:
        entry["vs_persistence"] = -0.001
    chart = ct.build_comparison_figure(ct.comparison_rows(envelopes))
    texts = [cell.get_text().get_text() for cell in chart.table.get_celld().values()]
    assert not any("-0.00" in t for t in texts) and any("skill vs persistence 0.00" in t for t in texts)
