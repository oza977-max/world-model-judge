"""Tests for wmj.judge.sharpness — how narrow the stated ranges are (ADR-J3, TC-JU5-01/02).

In plain words: sharpness is the average width of the model's 90% ranges at the task's judging
step: exactly 2 × 1.6449 × the stated spread. These tests check that formula by hand, that it
scales one-for-one with the spread, that a model with tight honest ranges is sharper than one
that hedges with wide ones at the same calibration, and that only the right step and region count.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.unit.judge._builders import D, H, judge_input_kwargs, make_input
from wmj.judge._normal import Z_90
from wmj.judge.calibration import compute_calibration
from wmj.judge.errors import JudgeInputError
from wmj.judge.sharpness import compute_sharpness
from wmj.judge.types import Forecasts, JudgeInput
from wmj.judge.verdict import _check_block


def _with_spread(spread):
    kwargs = judge_input_kwargs()
    n = len(kwargs["region_labels"])
    kwargs["predictions"] = Forecasts(np.zeros((n, H, D)), np.asarray(spread, dtype=float).reshape(n, H, D) if np.size(spread) == n * H * D else np.full((n, H, D), float(spread)))
    kwargs["outcomes"] = np.zeros((n, H, D))
    return JudgeInput(**kwargs)


def _entry(block, task, region):
    return next(e for e in block["per_task"] if e["task"] == task and e["region"] == region)


def test_the_width_is_two_times_the_ninety_percent_multiplier_times_the_spread():
    for spread in (0.1, 1.0, 3.7):
        e = _entry(compute_sharpness(_with_spread(spread)), "lv-control", "training")
        assert e["mean_width_90"] == pytest.approx(2 * 1.6449 * spread, rel=1e-12)
    assert Z_90 == 1.6449


def test_the_mean_is_over_trials_and_quantities_at_the_judging_step_only():
    kwargs = judge_input_kwargs()
    n = len(kwargs["region_labels"])
    spread = np.full((n, H, D), 100.0)  # huge everywhere ...
    spread[:3, 0, 0], spread[:3, 0, 1] = [1.0, 2.0, 3.0], [4.0, 5.0, 6.0]  # ... except step 1 of the three training trials
    kwargs["predictions"] = Forecasts(np.zeros((n, H, D)), spread)
    kwargs["outcomes"] = np.zeros((n, H, D))
    e = _entry(compute_sharpness(JudgeInput(**kwargs)), "lv-control", "training")  # judged at step 1
    assert e["mean_width_90"] == pytest.approx(2 * 1.6449 * np.mean([1, 2, 3, 4, 5, 6]))


def test_tc_ju5_02_width_is_monotone_in_the_stated_spread():
    previous = 0.0
    for spread in (0.01, 0.1, 0.5, 1.0, 2.0, 10.0):
        width = _entry(compute_sharpness(_with_spread(spread)), "lv-planning", "out-of-range")["mean_width_90"]
        assert width > previous
        previous = width
    rng = np.random.default_rng(4)
    for _ in range(20):
        base = rng.uniform(0.1, 2.0, size=(6, H, D))
        bigger = base * rng.uniform(1.01, 3.0)
        a = compute_sharpness(_with_spread(base))["per_task"]
        b = compute_sharpness(_with_spread(bigger))["per_task"]
        assert all(y["mean_width_90"] > x["mean_width_90"] for x, y in zip(a, b, strict=True))


def test_tc_ju5_01_a_tight_honest_model_is_sharper_than_a_hedging_one_at_equal_calibration():
    rng = np.random.default_rng(7)
    n = 6
    z = rng.normal(size=(n, H, D))
    tight = judge_input_kwargs()
    tight["predictions"] = Forecasts(np.zeros((n, H, D)), np.full((n, H, D), 1.0))
    tight["outcomes"] = z * 1.0
    wide = judge_input_kwargs()
    wide["predictions"] = Forecasts(np.zeros((n, H, D)), np.full((n, H, D), 50.0))
    wide["outcomes"] = z * 1.0
    tight_input, wide_input = JudgeInput(**tight), JudgeInput(**wide)
    # the always-safe model covers everything (calibration looks perfect or better)...
    assert all(e["coverage"] == [1.0] * 4 for e in compute_calibration(wide_input)["per_task"])
    # ...but its ranges are fifty times wider, which is what the verdict must show beside it
    for a, b in zip(compute_sharpness(tight_input)["per_task"], compute_sharpness(wide_input)["per_task"], strict=True):
        assert b["mean_width_90"] == pytest.approx(50 * a["mean_width_90"])


def test_each_task_and_region_is_judged_at_its_own_step():
    kwargs = judge_input_kwargs()
    n = len(kwargs["region_labels"])
    spread = np.ones((n, H, D))
    spread[:, 0, :] = 1.0  # step 1 (control in both regions; planning in out-of-range)
    spread[:, 1, :] = 2.0  # step 2 (planning in training)
    kwargs["predictions"] = Forecasts(np.zeros((n, H, D)), spread)
    kwargs["outcomes"] = np.zeros((n, H, D))
    block = compute_sharpness(JudgeInput(**kwargs))
    assert _entry(block, "lv-control", "training")["mean_width_90"] == pytest.approx(2 * 1.6449 * 1.0)
    assert _entry(block, "lv-planning", "training")["mean_width_90"] == pytest.approx(2 * 1.6449 * 2.0)
    assert _entry(block, "lv-planning", "out-of-range")["mean_width_90"] == pytest.approx(2 * 1.6449 * 1.0)


def test_entries_order_types_and_the_verdict_door():
    block = compute_sharpness(make_input())
    assert [(e["task"], e["region"]) for e in block["per_task"]] == [
        ("lv-control", "out-of-range"), ("lv-control", "training"),
        ("lv-planning", "out-of-range"), ("lv-planning", "training"),
    ]
    assert all(set(e) == {"task", "region", "mean_width_90"} and type(e["mean_width_90"]) is float for e in block["per_task"])
    assert _check_block("sharpness", block) == block


def test_compute_sharpness_takes_only_a_real_judge_input():
    for bad in (None, {}, "x"):
        with pytest.raises(JudgeInputError, match="JudgeInput"):
            compute_sharpness(bad)


def test_the_width_is_the_mean_not_the_median_over_trials():
    spread = np.ones((6, H, D))
    spread[2] = 10.0  # training trials: 1, 1, 10 -> mean 4, median 1
    spread[5] = 10.0
    e = _entry(compute_sharpness(_with_spread(spread)), "lv-control", "training")
    assert e["mean_width_90"] == pytest.approx(2 * Z_90 * 4.0, rel=1e-12)


def test_an_overflowing_width_is_refused_not_reported_as_infinity():
    for spread in (1e308, 1e307):
        with pytest.raises(JudgeInputError, match="overflow"):
            compute_sharpness(_with_spread(spread))
