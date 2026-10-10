"""Tests for wmj.judge.calibration — coverage at four levels (ADR-J2, TC-JU4-01).

In plain words: build forecasts whose true coverage is known exactly (outcomes placed at chosen
distances from the stated mean, in units of the stated spread — including exactly on each
confidence multiplier), and check the module counts them right: joint and per quantity, only at
the task's own judging step, only within the region, closed at the edge.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.unit.judge._builders import D, H, judge_input_kwargs, make_input
from wmj.judge._normal import Z_50, Z_80, Z_90, Z_95
from wmj.judge.calibration import (
    LEVELS,
    compute_calibration,
    region_rows,
    task_region_step,
)
from wmj.judge.errors import JudgeInputError
from wmj.judge.types import Forecasts, JudgeInput, RegionCurve, TaskSpec
from wmj.judge.verdict import _check_block, _check_calibration


def test_the_four_levels_and_their_fixed_multipliers_are_the_specs():
    assert LEVELS == (0.5, 0.8, 0.9, 0.95)
    assert (Z_50, Z_80, Z_90, Z_95) == (0.6745, 1.2816, 1.6449, 1.9600)


def _input_with_offsets(offsets_by_region, *, tasks=None, curves=None, spread=1.0):
    """Outcomes at the given offsets (in spreads) from a zero mean, for every step; offsets are [trial][dim]."""
    kwargs = judge_input_kwargs()
    n = len(kwargs["region_labels"])
    offsets = np.zeros((n, D))
    for region, rows in offsets_by_region.items():
        for row, off in zip(REGION_ROWS[region], rows, strict=True):
            offsets[row] = off
    outcomes = np.repeat(offsets[:, None, :], H, axis=1) * spread
    kwargs["outcomes"] = outcomes
    kwargs["predictions"] = Forecasts(np.zeros((n, H, D)), np.full((n, H, D), spread))
    if tasks is not None:
        kwargs["tasks"] = tasks
    if curves is not None:
        kwargs["divergence_curves"] = curves
    return JudgeInput(**kwargs)


REGION_ROWS = {"training": [0, 1, 2], "out-of-range": [3, 4, 5]}


def _entry(block, task, region):
    return next(e for e in block["per_task"] if e["task"] == task and e["region"] == region)


def test_coverage_is_counted_exactly_at_each_level_with_the_edge_inside():
    # one quantity is 0 (always inside); the other sits at 0, then 0.7 sigma, then exactly at 1.6449 sigma
    inp = _input_with_offsets({"training": [(0.0, 0.0), (0.0, 0.7), (0.0, Z_90)], "out-of-range": [(0.0, 0.0)] * 3})
    e = _entry(compute_calibration(inp), "lv-control", "training")
    # levels 50/80/90/95: trial0 always in; trial1 (0.7) in at 80+ but not 50 (0.6745); trial2 (1.6449) in at 90 (closed edge) and 95
    assert e["coverage"] == [pytest.approx(1 / 3), pytest.approx(2 / 3), pytest.approx(1.0), pytest.approx(1.0)]
    assert e["levels"] == [0.5, 0.8, 0.9, 0.95] and e["n_trials"] == 3


def test_a_point_just_outside_the_edge_is_not_covered():
    just_out = np.nextafter(Z_90, 10.0)
    inp = _input_with_offsets({"training": [(0.0, 0.0), (0.0, 0.0), (0.0, just_out)], "out-of-range": [(0.0, 0.0)] * 3})
    e = _entry(compute_calibration(inp), "lv-control", "training")
    assert e["coverage"][2] == pytest.approx(2 / 3)  # at 90% the just-outside point is out
    assert e["coverage"][3] == pytest.approx(1.0)  # at 95% (1.96) it is in


def test_joint_coverage_needs_every_quantity_inside_and_per_dimension_is_kept():
    inp = _input_with_offsets({"training": [(0.0, 0.0), (3.0, 0.0), (0.0, 3.0)], "out-of-range": [(0.0, 0.0)] * 3})
    e = _entry(compute_calibration(inp), "lv-control", "training")
    assert e["coverage"] == [pytest.approx(1 / 3)] * 4  # only trial 0 has both inside
    assert e["per_dimension"] == [[pytest.approx(2 / 3), pytest.approx(2 / 3)]] * 4  # each quantity alone: 2 of 3


def test_per_dimension_rows_follow_the_levels_in_order():
    inp = _input_with_offsets({"training": [(0.0, 0.0), (0.0, 0.7), (0.0, 1.5)], "out-of-range": [(0.0, 0.0)] * 3})
    e = _entry(compute_calibration(inp), "lv-control", "training")
    assert [row[0] for row in e["per_dimension"]] == [1.0, 1.0, 1.0, 1.0]
    assert [round(row[1], 6) for row in e["per_dimension"]] == [round(1 / 3, 6), round(2 / 3, 6), 1.0, 1.0]


def test_the_stated_spread_scales_the_interval():
    wide = _input_with_offsets({"training": [(0.0, 3.0)] * 3, "out-of-range": [(0.0, 0.0)] * 3}, spread=2.0)
    # offsets are in spreads, so the outcome is 3 spreads away whatever the spread: never covered
    assert _entry(compute_calibration(wide), "lv-control", "training")["coverage"] == [0.0] * 4
    kwargs = judge_input_kwargs()
    n = len(kwargs["region_labels"])
    kwargs["outcomes"] = np.full((n, H, D), 1.0)  # 1.0 from a zero mean
    kwargs["predictions"] = Forecasts(np.zeros((n, H, D)), np.full((n, H, D), 0.5))  # 2 sigma away -> outside all levels
    assert _entry(compute_calibration(JudgeInput(**kwargs)), "lv-control", "training")["coverage"] == [0.0] * 4
    kwargs["predictions"] = Forecasts(np.zeros((n, H, D)), np.full((n, H, D), 0.9))  # 1.11 sigma: in at 80, 90, 95 but not 50
    assert _entry(compute_calibration(JudgeInput(**kwargs)), "lv-control", "training")["coverage"] == [0.0, 1.0, 1.0, 1.0]


def test_each_task_is_judged_at_its_own_switch_step_and_no_other_step_matters():
    kwargs = judge_input_kwargs()
    n = len(kwargs["region_labels"])
    # training curve [0, .2, .4, .6, .8, 1.0]: control (tau .1) switches at step 1, planning (tau .3) at step 2
    outcomes = np.full((n, H, D), 99.0)  # far outside everywhere ...
    outcomes[:, 0, :] = 0.0  # ... except step 1 (index 0), where the control task is judged: all covered
    outcomes[:, 1, :] = 1.0  # step 2 (index 1): 1.0 away, covered only at spread >= 1.0/z
    kwargs["outcomes"] = outcomes
    kwargs["predictions"] = Forecasts(np.zeros((n, H, D)), np.full((n, H, D), 1.0))
    block = compute_calibration(JudgeInput(**kwargs))
    assert _entry(block, "lv-control", "training")["coverage"] == [1.0, 1.0, 1.0, 1.0]
    # planning at step 2: |1.0| <= z*1.0 is false at 50% (0.6745) and true from 80% up
    assert _entry(block, "lv-planning", "training")["coverage"] == [0.0, 1.0, 1.0, 1.0]
    assert task_region_step(JudgeInput(**kwargs), JudgeInput(**kwargs).tasks[0], "training") == 1
    assert task_region_step(JudgeInput(**kwargs), JudgeInput(**kwargs).tasks[1], "training") == 2
    assert task_region_step(JudgeInput(**kwargs), JudgeInput(**kwargs).tasks[1], "out-of-range") == 1  # a faster-drifting region


def test_a_task_that_never_switches_is_judged_at_the_end_of_its_horizon():
    kwargs = judge_input_kwargs()
    n = len(kwargs["region_labels"])
    flat = np.full(H + 1, 0.001)  # the world never drifts past any tolerance
    kwargs["divergence_curves"] = tuple(type(c)(c.region_name, flat) for c in kwargs["divergence_curves"])
    outcomes = np.full((n, H, D), 99.0)
    outcomes[:, 2, :] = 0.0  # step 3: the control task's horizon is 3
    outcomes[:, 4, :] = 0.0  # step 5: the planning task's horizon is 5
    kwargs["outcomes"] = outcomes
    kwargs["predictions"] = Forecasts(np.zeros((n, H, D)), np.ones((n, H, D)))
    block = compute_calibration(JudgeInput(**kwargs))
    assert all(e["coverage"] == [1.0] * 4 for e in block["per_task"])


def test_regions_are_scored_only_on_their_own_trials():
    base = _input_with_offsets({"training": [(0.0, 0.0)] * 3, "out-of-range": [(0.0, 0.0)] * 3})
    worse = _input_with_offsets({"training": [(0.0, 0.0)] * 3, "out-of-range": [(9.0, 9.0)] * 3})
    a, b = compute_calibration(base), compute_calibration(worse)
    assert _entry(a, "lv-control", "training") == _entry(b, "lv-control", "training")
    assert _entry(b, "lv-control", "out-of-range")["coverage"] == [0.0] * 4
    assert _entry(a, "lv-control", "out-of-range")["coverage"] == [1.0] * 4


def test_entries_are_one_per_task_and_region_tasks_in_order_regions_sorted():
    block = compute_calibration(make_input())
    assert [(e["task"], e["region"]) for e in block["per_task"]] == [
        ("lv-control", "out-of-range"), ("lv-control", "training"),
        ("lv-planning", "out-of-range"), ("lv-planning", "training"),
    ]
    assert list(region_rows(make_input())) == ["out-of-range", "training"]
    assert [int(r.size) for r in region_rows(make_input()).values()] == [3, 3]


def test_the_block_passes_the_verdict_doors_own_calibration_checks():
    block = compute_calibration(make_input())
    checked = _check_block("calibration", block)
    _check_calibration(checked)
    for e in block["per_task"]:
        assert all(type(v) is float for v in e["coverage"]) and type(e["n_trials"]) is int


def test_coverage_never_falls_as_the_interval_widens_for_random_inputs():
    rng = np.random.default_rng(11)
    for _ in range(25):
        kwargs = judge_input_kwargs()
        n = len(kwargs["region_labels"])
        kwargs["outcomes"] = rng.normal(size=(n, H, D)) * rng.uniform(0.1, 3.0)
        kwargs["predictions"] = Forecasts(rng.normal(size=(n, H, D)), rng.uniform(0.05, 2.0, size=(n, H, D)))
        for e in compute_calibration(JudgeInput(**kwargs))["per_task"]:
            _check_calibration({"per_task": [e]})


def test_tc_ju4_01_calibration_is_its_own_field_never_mixed_into_the_skill_score():
    from wmj.judge.skill import compute_skill

    inp = make_input()
    cal, skill = compute_calibration(inp), compute_skill(inp)
    assert set(cal["per_task"][0]) == {"task", "region", "levels", "coverage", "n_trials", "per_dimension"}
    assert not (set(cal["per_task"][0]) & {"vs_persistence", "vs_linear", "crps"})
    assert not (set(skill["per_task_region"][0]) & {"coverage", "levels", "per_dimension"})


def test_a_true_model_is_calibrated_at_every_level_on_a_large_sample():
    rng = np.random.default_rng(2026)
    n = 20_000
    sigma = rng.uniform(0.3, 2.0, size=(n, 1, 1))
    mean = rng.normal(size=(n, 1, 1))
    outcome = mean + sigma * rng.normal(size=(n, 1, 1))
    from wmj.judge.types import (
        Bands,
        ClimatologyBin,
        RegionClimatology,
        RegionCurve,
        RegionLabel,
        Thresholds,
    )

    inp = JudgeInput(
        world="w", dt=0.1, natural_cycle_length=None,
        predictions=Forecasts(mean, sigma), outcomes=outcome,
        persistence=Forecasts(mean * 0, np.ones_like(sigma)), linear=Forecasts(mean * 0, np.ones_like(sigma)),
        region_labels=tuple(RegionLabel("r", None) for _ in range(n)),
        divergence_curves=(RegionCurve("r", np.array([0.0, 1.0])),),
        climatology=(RegionClimatology("r", (ClimatologyBin(0.0, 1.0, np.zeros(1), np.ones(1), 5),)),),
        invariant_bins=np.zeros((n, 1), dtype=int), tasks=(TaskSpec("t", "control", 0.1, 1),),
        thresholds=Thresholds(Bands(n=n, p=0.1, green=(0, 1), amber_outer=(0, 2)), np.ones(1), 1.0),
    )
    coverage = compute_calibration(inp)["per_task"][0]["coverage"]
    assert coverage == [pytest.approx(p, abs=0.012) for p in (0.5, 0.8, 0.9, 0.95)]


def test_compute_calibration_takes_only_a_real_judge_input():
    for bad in (None, {}, "x"):
        with pytest.raises(JudgeInputError, match="JudgeInput"):
            compute_calibration(bad)


def _many_regions(names, per_region):
    from wmj.judge.types import (
        Bands,
        ClimatologyBin,
        RegionClimatology,
        RegionCurve,
        RegionLabel,
        Thresholds,
    )

    n = len(names) * per_region
    labels = tuple(RegionLabel(name, None) for name in names for _ in range(per_region))
    return JudgeInput(
        world="w", dt=0.1, natural_cycle_length=None,
        predictions=Forecasts(np.zeros((n, 2, 1)), np.ones((n, 2, 1))), outcomes=np.zeros((n, 2, 1)),
        persistence=Forecasts(np.zeros((n, 2, 1)), np.ones((n, 2, 1))), linear=Forecasts(np.zeros((n, 2, 1)), np.ones((n, 2, 1))),
        region_labels=labels,
        divergence_curves=tuple(RegionCurve(name, np.array([0.0, 0.5, 1.0])) for name in names),
        climatology=tuple(RegionClimatology(name, (ClimatologyBin(0.0, 1.0, np.zeros(1), np.ones(1), 5),)) for name in names),
        invariant_bins=np.zeros((n, 2), dtype=int), tasks=(TaskSpec("t", "control", 0.1, 2),),
        thresholds=Thresholds(Bands(n=per_region, p=0.1, green=(0, 1), amber_outer=(0, per_region)), np.ones(1), 1.0),
    )


def test_regions_come_out_in_sorted_name_order_whatever_the_order_they_arrive_in_and_trial_counts_are_the_regions():
    names = ["zeta", "alpha", "mu", "beta", "omega", "gamma", "kappa"]
    block = compute_calibration(_many_regions(names, 2))
    assert [e["region"] for e in block["per_task"]] == sorted(names)
    assert {e["n_trials"] for e in block["per_task"]} == {2}
    assert [e["region"] for e in compute_calibration(_many_regions(names[::-1], 4))["per_task"]] == sorted(names)
    assert {e["n_trials"] for e in compute_calibration(_many_regions(names, 4))["per_task"]} == {4}


# --- review pass 2: the step actually read, the task's own horizon, the tolerance edge ------------------


def _stepwise_input(curves, tasks, *, miss_step, miss_in_spreads):
    """Mean = s, spread = s at step s; the outcome sits `miss_in_spreads` spreads from the mean at `miss_step`, on the mean elsewhere."""
    kwargs = judge_input_kwargs()
    n = len(kwargs["region_labels"])
    steps = np.arange(1, H + 1, dtype=float)
    mean = np.broadcast_to(steps[None, :, None], (n, H, D)).copy()
    spread = mean.copy()
    outcome = mean.copy()
    outcome[:, miss_step - 1, :] += miss_in_spreads * spread[:, miss_step - 1, :]
    kwargs.update(
        predictions=Forecasts(mean, spread), outcomes=outcome, divergence_curves=curves, tasks=tasks,
    )
    return JudgeInput(**kwargs)


CURVES_SWITCH_AT_3_AND_4 = (
    RegionCurve("training", np.array([0.0, 0.01, 0.02, 0.2, 0.5, 0.6])),
    RegionCurve("out-of-range", np.array([0.0, 0.01, 0.02, 0.2, 0.5, 0.6])),
)


def test_the_mean_and_the_spread_are_read_at_the_judging_step_not_at_step_one_or_the_last():
    # control (tolerance 0.1) is judged at step 3, planning (0.3) at step 4; only that step's outcome is off, by 1.5 spreads
    for miss_step, task, covered in ((3, "lv-control", [0.0, 0.0, 1.0, 1.0]), (4, "lv-planning", [0.0, 0.0, 1.0, 1.0])):
        inp = _stepwise_input(CURVES_SWITCH_AT_3_AND_4, judge_input_kwargs()["tasks"], miss_step=miss_step, miss_in_spreads=1.5)
        assert _entry(compute_calibration(inp), task, "training")["coverage"] == covered
    # an outcome off at a step that is not the judging step is invisible
    inp = _stepwise_input(CURVES_SWITCH_AT_3_AND_4, judge_input_kwargs()["tasks"], miss_step=5, miss_in_spreads=9.0)
    assert _entry(compute_calibration(inp), "lv-control", "training")["coverage"] == [1.0, 1.0, 1.0, 1.0]


def test_a_task_is_searched_and_judged_inside_its_own_horizon_not_the_worlds():
    # the curve crosses at step 3, after this task's horizon of 2: no switch step, so it is judged at its last step, 2
    tasks = (TaskSpec("short", "control", 0.1, 2),)
    inp = _stepwise_input(CURVES_SWITCH_AT_3_AND_4, tasks, miss_step=3, miss_in_spreads=9.0)
    assert task_region_step(inp, tasks[0], "training") == 2
    assert _entry(compute_calibration(inp), "short", "training")["coverage"] == [1.0, 1.0, 1.0, 1.0]


def test_the_switch_step_uses_the_exact_tolerance_so_a_hair_over_is_a_switch_and_equal_is_not():
    tolerance = 0.1
    task = TaskSpec("t", "control", tolerance, 5)
    over = float(np.nextafter(tolerance, 1.0))
    for step_value, expected_step in ((over, 1), (tolerance, 5)):
        curves = (RegionCurve("training", np.array([0.0, step_value, tolerance, tolerance, tolerance, tolerance])),
                  RegionCurve("out-of-range", np.zeros(H + 1)))
        inp = _stepwise_input(curves, (task,), miss_step=1, miss_in_spreads=0.0)
        assert task_region_step(inp, task, "training") == expected_step


def test_a_gap_too_large_to_represent_is_refused_not_counted_as_covered():
    kwargs = judge_input_kwargs()
    n = len(kwargs["region_labels"])
    kwargs["predictions"] = Forecasts(np.full((n, H, D), -1.5e308), np.full((n, H, D), 1e308))
    kwargs["outcomes"] = np.full((n, H, D), 1.5e308)
    with pytest.raises(JudgeInputError, match="overflow"):
        compute_calibration(JudgeInput(**kwargs))


def test_an_overflowing_gap_is_refused_as_a_judge_error_even_when_warnings_are_errors():
    import warnings

    kwargs = judge_input_kwargs()
    n = len(kwargs["region_labels"])
    kwargs["predictions"] = Forecasts(np.full((n, H, D), -1.5e308), np.full((n, H, D), 1e308))
    kwargs["outcomes"] = np.full((n, H, D), 1.5e308)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(JudgeInputError, match="overflow"):
            compute_calibration(JudgeInput(**kwargs))


def test_a_single_overflowing_half_width_among_finite_ones_is_still_refused():
    kwargs = judge_input_kwargs()
    n = len(kwargs["region_labels"])
    spread = np.ones((n, H, D))
    spread[0, :, 0] = 1.7e308  # one trial, one quantity: its half-width is infinite and must not read as 'covered'
    kwargs.update(predictions=Forecasts(np.zeros((n, H, D)), spread), outcomes=np.full((n, H, D), 1e300))
    with pytest.raises(JudgeInputError, match="overflow"):
        compute_calibration(JudgeInput(**kwargs))
