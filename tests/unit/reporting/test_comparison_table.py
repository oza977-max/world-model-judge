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

MARGIN = 0.05  # prereg/recipe.md's matching_margin


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
    comparison = ct.comparison_rows(envelopes, MARGIN)
    assert comparison.control_task == "lv-control"  # tolerance 0.1 against lv-planning's 0.3
    assert [r.name for r in comparison.rows] == ["persistence", "linear", "direct", "ensemble", "fx-brittle"]
    assert comparison.ranks == (1, 2, 3, 4, 5)
    figure = ct.build_comparison_figure(comparison)
    header = figure.table[0, 1].get_text().get_text()
    assert "lv-control" in header and "training" in header and "one step" in header


def test_equal_errors_share_a_rank():
    envelopes = _rows()
    set_crps(_named(envelopes)["direct"], 0.2)  # level with linear
    comparison = ct.comparison_rows(envelopes, MARGIN)
    assert dict(zip([r.name for r in comparison.rows], comparison.ranks, strict=True)) == {
        "persistence": 1, "direct": 2, "linear": 2, "ensemble": 4, "fx-brittle": 5}


def test_columns_put_training_first_in_each_task_then_the_other_regions():
    comparison = ct.comparison_rows(_rows(), MARGIN)
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
    comparison = ct.comparison_rows(envelopes, MARGIN)
    assert {r.name for r, m in zip(comparison.rows, comparison.marked, strict=True) if m} == marked
    assert comparison.caption_pair == (("direct", "ensemble") if marked else None)


def test_a_fixture_is_never_marked_nor_named_however_it_ranks():
    envelopes = _rows()
    fixture = _named(envelopes)["fx-brittle"]
    set_crps(fixture, 0.01)  # best on ordinary error
    set_horizon(fixture, 0, "lv-planning")
    for name, steps in (("persistence", 2), ("linear", 2), ("direct", 2), ("ensemble", 1)):  # real models concordant
        set_horizon(_named(envelopes)[name], steps, "lv-planning")
    comparison = ct.comparison_rows(envelopes, MARGIN)
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
    comparison = ct.comparison_rows(envelopes, MARGIN)
    # ensemble disagrees with all three; it is closest to direct (0.41 / 0.40), and has the lower error
    assert comparison.caption_pair == ("ensemble", "direct")
    marked = {r.name for r, m in zip(comparison.rows, comparison.marked, strict=True) if m}
    assert marked == {"ensemble", "direct", "persistence", "linear"}


def test_a_world_with_no_natural_cycle_shows_a_dash_not_a_zero():
    envelopes = _rows()
    for env in envelopes:
        for entry in env["verdict"]["trust_horizons"]["per_task"]:
            entry["natural_units"] = None
    chart = ct.build_comparison_figure(ct.comparison_rows(envelopes, MARGIN))
    assert chart.table[1, 2].get_text().get_text().endswith("—")


def test_fixture_rows_are_labelled_tinted_and_the_figure_carries_the_fixture_sentence():
    chart = ct.build_comparison_figure(ct.comparison_rows(_rows(), MARGIN))
    names = [chart.table[r, 0].get_text().get_text() for r in range(1, 6)]
    fixture_row = names.index("FIXTURE: fx-brittle") + 1
    assert chart.table[fixture_row, 0].get_text().get_color() == style.FIXTURE_TEXT_COLOUR  # readable on the tint
    assert chart.table[fixture_row, 3].get_facecolor()[:3] != chart.table[1, 3].get_facecolor()[:3]
    assert style.has_fixture_mark(chart.figure)


def test_a_run_without_fixtures_carries_no_fixture_sentence(tmp_path):
    envelopes = [e for e in _rows() if not e["is_fixture"]]
    paths = ct.render_comparison_chart(envelopes, tmp_path / "c", tmp_path / "t", MARGIN)
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
        ct.comparison_rows(_spoil(change), MARGIN)


def test_render_refuses_before_writing_anything(tmp_path):
    with pytest.raises(ct.ComparisonBlockError):
        ct.render_comparison_chart([], tmp_path / "c", tmp_path / "t", MARGIN)
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
    paths = ct.render_comparison_chart(envelopes, tmp_path / "c", tmp_path / "t", MARGIN)
    svg = paths.svg.read_text(encoding="utf-8")
    assert "out-large-action" in svg and "FIXTURE: " + "f" * 40 in svg and "—" in svg
    chart = ct.build_comparison_figure(ct.comparison_rows(envelopes, MARGIN))
    chart.figure.draw_without_rendering()
    for (r, c), cell in chart.table.get_celld().items():  # every cell's text fits inside its cell
        text, box = cell.get_text().get_window_extent(), cell.get_window_extent()
        assert box.x0 - 1 <= text.x0 and text.x1 <= box.x1 + 1 and box.y0 - 1 <= text.y0 and text.y1 <= box.y1 + 1, (r, c)


def test_a_skill_a_hair_below_zero_reads_as_zero_not_minus_zero():
    envelopes = _rows()
    for entry in _named(envelopes)["direct"]["verdict"]["skill"]["per_task_region"]:
        entry["vs_persistence"] = -0.001
    chart = ct.build_comparison_figure(ct.comparison_rows(envelopes, MARGIN))
    texts = [cell.get_text().get_text() for cell in chart.table.get_celld().values()]
    assert not any("-0.00" in t for t in texts) and any("skill vs persistence 0.00" in t for t in texts)


# --- review of P5-C04, pass 1 ------------------------------------------------------------------------------


def _discordant_pair(envelopes, a, b, crps, skill=None):
    """Make models a and b disagree (a ahead on error, b on the judge's horizon); optionally set a's skills."""
    by = _named(envelopes)
    for name in by:
        set_horizon(by[name], 2 if name in ("persistence", "linear") else 0, "lv-planning")
    set_crps(by[a], crps[0])
    set_crps(by[b], crps[1])
    set_horizon(by[b], 1, "lv-planning")
    if skill is not None:
        for entry in by[a]["verdict"]["skill"]["per_task_region"]:
            entry.update(vs_persistence=skill[0], vs_linear=skill[1])
    return envelopes


def test_equals_is_said_only_of_a_pair_inside_the_pre_registered_margin():
    # review of P5-C04, I1: a pair 4.5 times apart on error was called "equals"
    envelopes = _discordant_pair(_rows(), "direct", "ensemble", (0.2, 0.9), skill=(0.9, 0.9))
    chart = ct.build_comparison_figure(ct.comparison_rows(envelopes, MARGIN))
    assert chart.comparison.caption_pair == ("direct", "ensemble") and not chart.comparison.pair_matched
    assert "Ordinary error ranks direct ahead of ensemble; the judge trusts ensemble further ahead on lv-planning, training." in chart.caption
    assert "equals" not in chart.caption


def test_a_pair_equal_by_the_margin_says_equals():
    envelopes = _discordant_pair(_rows(), "direct", "ensemble", (0.40, 0.41))
    by = _named(envelopes)
    skills = next(e for e in by["ensemble"]["verdict"]["skill"]["per_task_region"] if (e["task"], e["region"]) == ("lv-control", "training"))
    for entry in by["direct"]["verdict"]["skill"]["per_task_region"]:
        entry.update(vs_persistence=skills["vs_persistence"] + 0.049, vs_linear=skills["vs_linear"] - 0.049)  # just inside
    comparison = ct.comparison_rows(envelopes, MARGIN)
    assert comparison.pair_matched and "are equals; the judge does not" in ct.build_comparison_figure(comparison).caption


def test_a_pair_equal_by_the_margin_is_named_before_a_closer_unequal_one():
    envelopes = _discordant_pair(_rows(), "direct", "ensemble", (0.40, 0.48))  # equal by skill, 1.2x apart on CRPS
    by = _named(envelopes)
    set_crps(by["linear"], 0.401)  # direct vs linear: 1.0025x apart, opposite orders (linear's horizon is 2), unequal skill
    for entry in by["linear"]["verdict"]["skill"]["per_task_region"]:
        entry.update(vs_persistence=-5.0)
    for entry in by["ensemble"]["verdict"]["skill"]["per_task_region"]:
        entry.update(vs_persistence=_skill(by["direct"])[0], vs_linear=_skill(by["direct"])[1])
    comparison = ct.comparison_rows(envelopes, MARGIN)
    assert comparison.caption_pair == ("direct", "ensemble") and comparison.pair_matched


def _skill(env):
    entry = next(e for e in env["verdict"]["skill"]["per_task_region"] if (e["task"], e["region"]) == ("lv-control", "training"))
    return entry["vs_persistence"], entry["vs_linear"]


@pytest.mark.parametrize("margin", [0.0, 1.0, -0.1, 0, "0.05", float("nan"), None])
def test_the_matching_margin_must_be_a_number_between_0_and_1(margin):
    with pytest.raises(ct.ComparisonBlockError, match="matching margin"):
        ct.comparison_rows(_rows(), margin)


def test_a_model_judged_with_another_tolerance_is_refused():
    envelopes = _rows()
    for entry in envelopes[2]["verdict"]["trust_horizons"]["per_task"]:
        if entry["task"] == "lv-planning":
            entry["tolerance"] = 0.5
    with pytest.raises(ct.ComparisonBlockError, match="tolerances"):
        ct.comparison_rows(envelopes, MARGIN)


def test_ties_on_error_are_listed_by_name_and_the_caption_tie_breaks_by_name():
    envelopes = _rows()
    by = _named(envelopes)
    set_crps(by["fx-brittle"], 0.9)
    for name in ("persistence", "linear", "direct", "ensemble"):
        set_crps(by[name], 0.5)
    comparison = ct.comparison_rows(envelopes, MARGIN)
    assert [r.name for r in comparison.rows][:4] == ["direct", "ensemble", "linear", "persistence"]
    set_crps(by["direct"], 0.4)
    set_crps(by["ensemble"], 0.4)
    set_crps(by["linear"], 0.6)
    set_crps(by["persistence"], 0.6)  # direct/linear and ensemble/persistence: the same 1.5x apart
    for name, steps in (("direct", 0), ("ensemble", 0), ("linear", 1), ("persistence", 1)):
        set_horizon(by[name], steps, "lv-planning")
    for env in (by["linear"], by["persistence"]):
        for entry in env["verdict"]["skill"]["per_task_region"]:
            entry.update(vs_persistence=9.0)  # unequal by skill: the caption ranks
    comparison = ct.comparison_rows(envelopes, MARGIN)
    assert comparison.caption_pair == ("direct", "linear")


@pytest.mark.parametrize(
    "b_skills, equal",
    [((0.33, 0.62), True), ((0.40, 0.62), False), ((0.33, 0.70), False), ((0.40, 0.70), False)],
)
def test_equal_needs_both_skill_scores_inside_the_margin(b_skills, equal):
    a = ct.Row("a", False, 0.4, 0.30, 0.60, ())
    b = ct.Row("b", False, 0.5, *b_skills, ())
    assert ct.matched(a, b, MARGIN) is equal



# --- review of P5-C04, pass 2 ------------------------------------------------------------------------------


def test_the_apart_caption_names_the_column_where_the_second_model_leads_even_when_it_leads_in_one_only():
    envelopes = _rows()
    by = _named(envelopes)
    for name in by:
        for task, region in (("lv-control", "training"), ("lv-control", "out-of-range"), ("lv-planning", "out-of-range")):
            set_horizon(by[name], 1 if name in ("direct", "persistence", "linear") else 0, task, region)
        set_horizon(by[name], 2 if name in ("persistence", "linear") else 0, "lv-planning", "training")
    set_crps(by["direct"], 0.3)
    set_crps(by["ensemble"], 0.6)
    set_horizon(by["ensemble"], 1, "lv-planning", "training")  # direct leads in three columns, ensemble in one
    for entry in by["ensemble"]["verdict"]["skill"]["per_task_region"]:
        entry.update(vs_persistence=9.0)  # not equal by the margin
    comparison = ct.comparison_rows(envelopes, MARGIN)
    assert comparison.caption_pair == ("direct", "ensemble") and comparison.lead_column == ("lv-planning", "training")
    assert ct.build_comparison_figure(comparison).caption.endswith(
        "the judge trusts ensemble further ahead on lv-planning, training.")


def test_the_lead_column_is_the_largest_lead_first_in_column_order_on_a_tie():
    envelopes = _rows()
    by = _named(envelopes)
    for name in by:
        for entry in by[name]["verdict"]["trust_horizons"]["per_task"]:
            set_horizon(by[name], 0, entry["task"], entry["region"])
    for name in ("persistence", "linear"):
        set_horizon(by[name], 2, "lv-planning")
        set_horizon(by[name], 1, "lv-control")
    set_horizon(by["ensemble"], 1, "lv-control")
    set_horizon(by["ensemble"], 1, "lv-planning")  # two columns with the same one-step lead
    for entry in by["ensemble"]["verdict"]["skill"]["per_task_region"]:
        entry.update(vs_persistence=9.0)
    assert ct.comparison_rows(envelopes, MARGIN).lead_column == ("lv-control", "training")
    set_horizon(by["ensemble"], 2, "lv-planning")  # now the planning lead is larger
    assert ct.comparison_rows(envelopes, MARGIN).lead_column == ("lv-planning", "training")


def test_a_pair_exactly_at_the_margin_is_equal_as_mu5_counts_it():
    a = ct.Row("a", False, 0.4, 0.25, 0.5, ())
    b = ct.Row("b", False, 0.5, 0.3125, 0.5625, ())  # differences of exactly 0.0625, a binary-exact margin
    assert ct.matched(a, b, 0.0625)


def test_closeness_is_the_ratio_of_errors_not_their_difference():
    rows = [ct.Row(n, False, c, 9.0 * i, 0.0, ()) for i, (n, c) in enumerate((("a", 0.1), ("b", 0.2), ("c", 1.0), ("d", 1.5)))]
    pairs = [(rows[0], rows[1]), (rows[2], rows[3])]  # 2x apart (0.1 difference) and 1.5x apart (0.5 difference)
    assert ct._closest(pairs, MARGIN) == (("c", "d"), False)


def test_a_world_time_of_exactly_one_is_singular():
    assert "1 time unit\n" in ct._horizon_text(ct.Horizon(50, 1.0, None))
    assert "2 time units\n" in ct._horizon_text(ct.Horizon(100, 2.0, None))
