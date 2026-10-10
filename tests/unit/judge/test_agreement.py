"""Tests for wmj.judge.climatology.compute_climatology — does the model still look like the world? (ADR-J5, JU-6)

In plain words: past the switch step nobody can fairly be asked to get the exact path, so the judge
asks whether the model's predicted state looks like the world's long-run behaviour *for the same
amount of the conserved quantity* (energy). These tests build predictions at exactly known numbers
of standard deviations from a chosen bin's mean, and check: nothing is reported when there is no
switch step; only steps from the switch to the task's horizon count; the bin is the one measured
at that trial and step (not frozen), from the trial's own region's table; the score is the
average absolute standardised gap; and the pass mark is closed at the threshold.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.unit.judge._builders import D, H, judge_input_kwargs, make_input
from wmj.judge.climatology import compute_climatology
from wmj.judge.errors import JudgeInputError
from wmj.judge.types import (
    ClimatologyBin,
    Forecasts,
    JudgeInput,
    RegionClimatology,
    RegionCurve,
    Thresholds,
)
from wmj.judge.verdict import _check_block, _check_climatology

ROWS = {"training": [0, 1, 2], "out-of-range": [3, 4, 5]}
# control (tolerance 0.1, horizon 3) switches at step 3; planning (tolerance 0.3, horizon 5) at step 4
CURVES = (
    RegionCurve("training", np.array([0.0, 0.01, 0.02, 0.2, 0.5, 0.6])),
    RegionCurve("out-of-range", np.array([0.0, 0.01, 0.02, 0.2, 0.5, 0.6])),
)


def _tables():
    """Two different tables (one per region): bin i has mean (1, 2) * (i + 1) * offset, sd (0.1, 0.2) * (i + 1) * scale."""
    def table(offset, scale):
        return tuple(
            ClimatologyBin(
                -np.inf if i == 0 else float(i), np.inf if i == 3 else float(i + 1),
                np.array([1.0, 2.0]) * (i + 1) * offset, np.array([0.1, 0.2]) * scale * (i + 1), 100)
            for i in range(4))
    return (RegionClimatology("training", table(1.0, 1.0)), RegionClimatology("out-of-range", table(10.0, 5.0)))


def _input(z_by_trial_step, bins_by_trial_step, *, curves=CURVES, tasks=None, threshold=1.0):
    """Predicted mean = the chosen bin's mean + z * sd (same z in both quantities) at every [trial, step]."""
    kwargs = judge_input_kwargs()
    n = len(kwargs["region_labels"])
    tables = _tables()
    by_region = {t.region_name: t for t in tables}
    regions = [lab.region_name for lab in kwargs["region_labels"]]
    mean = np.zeros((n, H, D))
    for i in range(n):
        for s in range(H):
            b = by_region[regions[i]].bins[bins_by_trial_step[i][s]]
            mean[i, s, :] = b.mean + np.asarray(z_by_trial_step[i][s], dtype=float) * b.sd
    kwargs.update(
        predictions=Forecasts(mean, np.ones((n, H, D))), climatology=tables, divergence_curves=curves,
        invariant_bins=np.array(bins_by_trial_step, dtype=np.int64),
    )
    if tasks is not None:
        kwargs["tasks"] = tasks
    kwargs["thresholds"] = Thresholds(kwargs["thresholds"].bands, kwargs["thresholds"].sharpness_hedge_threshold, threshold)
    return JudgeInput(**kwargs)


def _entry(block, task, region):
    return next(e for e in block["per_task"] if (e["task"], e["region"]) == (task, region))


def _flat(value, bins=0):
    return [[value] * H for _ in range(6)], [[bins] * H for _ in range(6)]


def test_a_task_that_never_leaves_trajectory_grading_reports_nothing_all_three_fields_null():
    quiet = (RegionCurve("training", np.zeros(H + 1)), RegionCurve("out-of-range", np.zeros(H + 1)))
    z, b = _flat(0.5)
    e = _entry(compute_climatology(_input(z, b, curves=quiet)), "lv-control", "training")
    assert (e["switch_step"], e["agreement_mean_abs_z"], e["agrees"]) == (None, None, None)


def test_the_switch_step_is_recorded_per_task_and_region():
    z, b = _flat(0.5)
    block = compute_climatology(_input(z, b))
    assert _entry(block, "lv-control", "training")["switch_step"] == 3
    assert _entry(block, "lv-planning", "out-of-range")["switch_step"] == 4
    shifted = (RegionCurve("training", np.array([0.0, 0.5, 0.5, 0.5, 0.5, 0.5])), CURVES[1])
    block = compute_climatology(_input(z, b, curves=shifted))
    assert _entry(block, "lv-control", "training")["switch_step"] == 1
    assert _entry(block, "lv-control", "out-of-range")["switch_step"] == 3


def test_only_steps_from_the_switch_to_the_tasks_horizon_are_averaged():
    # control: switch 3, horizon 3 -> only step 3 counts. Every other step is wildly off (z = 50) and must be ignored.
    z = [[50.0, 50.0, 0.25, 50.0, 50.0] for _ in range(6)]
    _, b = _flat(0, 2)
    e = _entry(compute_climatology(_input(z, b)), "lv-control", "training")
    assert e["agreement_mean_abs_z"] == pytest.approx(0.25, rel=1e-9)
    # planning: switch 4, horizon 5 -> steps 4 and 5: (0.5 + 1.5) / 2 = 1.0
    z = [[50.0, 50.0, 50.0, 0.5, 1.5] for _ in range(6)]
    e = _entry(compute_climatology(_input(z, b)), "lv-planning", "training")
    assert e["agreement_mean_abs_z"] == pytest.approx(1.0, rel=1e-9)


def test_the_score_is_the_mean_of_the_absolute_standardised_gaps_over_trials_steps_and_quantities():
    # planning, training trials 0-2, steps 4 and 5: z values chosen so mean(|z|) = 2 (median 1.5) while mean(z) is not
    z = [[0.0, 0.0, 0.0, -1.0, 2.0], [0.0, 0.0, 0.0, 1.0, -2.0], [0.0, 0.0, 0.0, 6.0, 0.0],
         [0.0] * 5, [0.0] * 5, [0.0] * 5]
    _, b = _flat(0, 0)
    e = _entry(compute_climatology(_input(z, b)), "lv-planning", "training")
    assert e["agreement_mean_abs_z"] == pytest.approx((1 + 2 + 1 + 2 + 6 + 0) / 6, rel=1e-9)


def test_the_bin_is_the_one_measured_at_that_trial_and_step_not_frozen_and_not_another_regions():
    # predictions sit exactly on bin 2's mean at step 3; if the judge used the bin recorded for another step
    # or another trial, the standardised gap would be large
    z, _ = _flat(0.0)
    bins = [[0, 1, 2, 3, 1] for _ in range(6)]
    bins[1][2] = 2
    inp = _input(z, bins)
    assert _entry(compute_climatology(inp), "lv-control", "training")["agreement_mean_abs_z"] == pytest.approx(0.0, abs=1e-9)
    # now ask the judge to read a different bin than the one the predictions were built on: gap is large
    wrong = _input(z, bins)
    kwargs = {f: getattr(wrong, f) for f in ("world", "dt", "natural_cycle_length", "predictions", "outcomes", "persistence",
                                              "linear", "region_labels", "divergence_curves", "climatology", "tasks", "thresholds")}
    other = np.array(bins, dtype=np.int64)
    other[:, 2] = 3
    e = _entry(compute_climatology(JudgeInput(invariant_bins=other, **kwargs)), "lv-control", "training")
    assert e["agreement_mean_abs_z"] > 2.0  # bin 3 is a different state, 2.5 spreads away


def test_each_quantity_is_standardised_by_its_own_spread():
    z = [[(0.0, 2.0)] * H for _ in range(6)]  # quantity 0 is on the mean, quantity 1 is two of its own spreads away
    _, b = _flat(0, 1)
    e = _entry(compute_climatology(_input(z, b)), "lv-control", "training")
    assert e["agreement_mean_abs_z"] == pytest.approx(1.0, rel=1e-9)


def test_each_regions_own_climatology_table_is_used():
    # out-of-range's table has 10x the means and 5x the sds; predictions built on it sit at z = 1
    z, b = _flat(1.0, bins=1)
    block = compute_climatology(_input(z, b))
    assert _entry(block, "lv-control", "out-of-range")["agreement_mean_abs_z"] == pytest.approx(1.0, rel=1e-9)
    assert _entry(block, "lv-control", "training")["agreement_mean_abs_z"] == pytest.approx(1.0, rel=1e-9)


def test_agreement_holds_exactly_at_the_threshold_and_fails_just_above_it():
    z, b = _flat(0.7)
    score = _entry(compute_climatology(_input(z, b)), "lv-control", "training")["agreement_mean_abs_z"]
    for threshold, agrees in ((score, True), (float(np.nextafter(score, 0.0)), False), (score * 2, True)):
        e = _entry(compute_climatology(_input(z, b, threshold=threshold)), "lv-control", "training")
        assert e["agrees"] is agrees


def test_the_block_passes_the_verdict_doors_own_checks_and_is_plain_data():
    block = compute_climatology(make_input())
    _check_climatology(_check_block("climatology", block))
    assert [(e["task"], e["region"]) for e in block["per_task"]] == [
        ("lv-control", "out-of-range"), ("lv-control", "training"), ("lv-planning", "out-of-range"), ("lv-planning", "training")]
    for e in block["per_task"]:
        assert e["switch_step"] is None or type(e["switch_step"]) is int
        assert e["agrees"] is None or type(e["agrees"]) is bool


def test_a_gap_too_large_to_represent_is_refused_not_scored():
    kwargs = judge_input_kwargs()
    n = len(kwargs["region_labels"])
    kwargs.update(predictions=Forecasts(np.full((n, H, D), 1.7e308), np.ones((n, H, D))))
    with pytest.raises(JudgeInputError, match="overflow"):
        compute_climatology(JudgeInput(**kwargs))


def test_it_takes_only_a_real_judge_input():
    for bad in (None, {}, "x"):
        with pytest.raises(JudgeInputError, match="JudgeInput"):
            compute_climatology(bad)


# --- review pass 1 --------------------------------------------------------------------------------------

BINS_PER_TRIAL = [[0, 1, 2, 3, 1], [1, 2, 3, 0, 2], [2, 3, 0, 1, 3], [3, 0, 1, 2, 0], [0, 3, 2, 1, 2], [1, 1, 3, 2, 3]]


def test_every_trial_and_step_is_compared_with_its_own_measured_bin_for_both_tasks():
    zero = [[0.0] * H for _ in range(6)]
    inp = _input(zero, BINS_PER_TRIAL)
    block = compute_climatology(inp)
    for task, region in (("lv-control", "training"), ("lv-planning", "training"), ("lv-planning", "out-of-range")):
        assert _entry(block, task, region)["agreement_mean_abs_z"] == pytest.approx(0.0, abs=1e-9), (task, region)
    kwargs = {f: getattr(inp, f) for f in ("world", "dt", "natural_cycle_length", "predictions", "outcomes", "persistence",
                                           "linear", "region_labels", "divergence_curves", "climatology", "tasks", "thresholds")}

    def score(bins):
        e = _entry(compute_climatology(JudgeInput(invariant_bins=np.array(bins, dtype=np.int64), **kwargs)), "lv-planning", "training")
        return e["agreement_mean_abs_z"]

    frozen_at_the_switch_step = [[row[3]] * H for row in BINS_PER_TRIAL]  # planning switches at step 4 (index 3)
    assert score(frozen_at_the_switch_step) > 1.0
    assert score(BINS_PER_TRIAL[::-1]) > 1.0  # trials swapped end for end


def test_a_switch_step_beyond_the_tasks_horizon_is_no_switch_step_for_that_task():
    late = (RegionCurve("training", np.array([0.0, 0.0, 0.0, 0.0, 5.0, 5.0])), RegionCurve("out-of-range", np.zeros(H + 1)))
    from wmj.judge.types import TaskSpec

    z, b = _flat(0.0)
    e = _entry(compute_climatology(_input(z, b, curves=late, tasks=(TaskSpec("short", "control", 1.0, 3),))), "short", "training")
    assert (e["switch_step"], e["agreement_mean_abs_z"], e["agrees"]) == (None, None, None)


def test_a_sum_of_huge_standardised_gaps_is_averaged_without_overflow():
    z = [[1e307] * H for _ in range(6)]
    _, b = _flat(0, 0)
    e = _entry(compute_climatology(_input(z, b)), "lv-control", "training")
    assert e["agreement_mean_abs_z"] == pytest.approx(1e307, rel=1e-9)
