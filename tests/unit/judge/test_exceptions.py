"""Tests for wmj.judge.exceptions — the backtest: count the misses, band them (ADR-J4, JU-8).

In plain words: each independent test start is a bet that the truth lands inside the model's
stated 90% range. These tests build outcomes at exactly known distances from the stated mean
(including exactly on the edge), and check the misses are counted right, that the count falls in
the right colour band at every band edge, that each task is checked at step 1 and at its own
switch step, that the per-trial list the chart draws and the count are one fact, that a padded
(too-wide) model is flagged instead of rewarded, and that a model that really is 90% honest lands
green about as often as the arithmetic says it should.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tests.unit.judge._builders import D, H, judge_input_kwargs, make_input
from wmj.judge._normal import Z_90
from wmj.judge.calibration import compute_calibration
from wmj.judge.errors import JudgeInputError
from wmj.judge.exceptions import compute_exceptions_and_trials
from wmj.judge.sharpness import compute_sharpness
from wmj.judge.skill import compute_skill
from wmj.judge.types import (
    Bands,
    ClimatologyBin,
    Forecasts,
    JudgeInput,
    RegionClimatology,
    RegionCurve,
    RegionLabel,
    TaskSpec,
    Thresholds,
)
from wmj.judge.verdict import (
    _check_block,
    _check_trials_and_exceptions,
    assemble_verdict,
)

# the builder's input: 3 + 3 trials, bands for n = 3: green [1, 1], amber-low [0, 0], amber-high [2, 2], red = 3
REGION_ROWS = {"training": [0, 1, 2], "out-of-range": [3, 4, 5]}


def _input(offsets_by_region, *, spread=1.0, step_offsets=None):
    """Outcomes `offsets` (in spreads, per quantity) from a zero mean at every step; `step_offsets` overrides per step."""
    kwargs = judge_input_kwargs()
    n = len(kwargs["region_labels"])
    offsets = np.zeros((n, H, D))
    for region, per_trial in offsets_by_region.items():
        for row, off in zip(REGION_ROWS[region], per_trial, strict=True):
            offsets[row, :, :] = off
    for (row, step), off in (step_offsets or {}).items():
        offsets[row, step - 1, :] = off
    kwargs["outcomes"] = offsets * spread
    kwargs["predictions"] = Forecasts(np.zeros((n, H, D)), np.full((n, H, D), spread))
    return JudgeInput(**kwargs)


def _get(block, task, region, step):
    return next(e for e in block["per_task"] if (e["task"], e["region"], e["horizon_step"]) == (task, region, step))


def test_misses_are_counted_per_trial_with_the_edge_inside_and_just_past_it_outside():
    just_out = float(np.nextafter(Z_90, 10.0))
    inp = _input({"training": [(0.0, 0.0), (0.0, Z_90), (0.0, just_out)], "out-of-range": [(0.0, 0.0)] * 3})
    exceptions, trials = compute_exceptions_and_trials(inp)
    e = _get(exceptions, "lv-control", "training", 1)
    t = _get(trials, "lv-control", "training", 1)
    assert t["is_exception"] == [False, False, True]  # exactly 1.6449 sigma is a hit; one ulp further is a miss
    assert e["observed"] == 1 and e["n_trials"] == 3


def test_any_one_quantity_outside_makes_the_trial_a_miss_and_it_is_one_miss_not_two():
    inp = _input({"training": [(2.0, 2.0), (2.0, 0.0), (0.0, 0.0)], "out-of-range": [(0.0, 0.0)] * 3})
    exceptions, trials = compute_exceptions_and_trials(inp)
    assert _get(trials, "lv-control", "training", 1)["is_exception"] == [True, True, False]
    assert _get(exceptions, "lv-control", "training", 1)["observed"] == 2


def test_the_band_follows_the_count_at_every_edge_of_the_pre_registered_bands():
    # bands for n = 3: 0 -> amber-low, 1 -> green, 2 -> amber-high, 3 -> red
    expected = {0: "amber", 1: "green", 2: "amber", 3: "red"}
    for misses, band in expected.items():
        offsets = [(9.0, 9.0)] * misses + [(0.0, 0.0)] * (3 - misses)
        inp = _input({"training": offsets, "out-of-range": [(0.0, 0.0)] * 3})
        e = _get(compute_exceptions_and_trials(inp)[0], "lv-control", "training", 1)
        assert e["observed"] == misses and e["band"] == band, misses


def test_the_bands_are_copied_from_the_thresholds_as_contiguous_ranges_with_the_expected_count():
    e = _get(compute_exceptions_and_trials(make_input())[0], "lv-control", "training", 1)
    assert e["bands"] == {"green": [1, 1], "amber": [[0, 0], [2, 2]], "red": "outside"}
    assert e["expected"] == pytest.approx(0.3) and type(e["expected"]) is float


def test_each_task_is_checked_at_step_one_and_at_its_own_switch_step_one_entry_when_they_coincide():
    inp = make_input()
    exceptions, trials = compute_exceptions_and_trials(inp)
    keys = [(e["task"], e["region"], e["horizon_step"]) for e in exceptions["per_task"]]
    assert keys == [
        ("lv-control", "out-of-range", 1), ("lv-control", "training", 1),  # control switches at step 1 in both regions
        ("lv-planning", "out-of-range", 1),  # planning switches at step 1 in the fast-drifting region: one entry
        ("lv-planning", "training", 1), ("lv-planning", "training", 2),  # ... and at step 2 in training: two entries
    ]
    assert [(t["task"], t["region"], t["horizon_step"]) for t in trials["per_task"]] == keys


def test_the_two_steps_of_a_task_are_scored_on_their_own_step():
    # trial 0 misses at step 2 only, trial 1 misses at step 1 only
    inp = _input({"training": [(0.0, 0.0)] * 3, "out-of-range": [(0.0, 0.0)] * 3}, step_offsets={(0, 2): (9.0, 0.0), (1, 1): (9.0, 0.0)})
    exceptions, trials = compute_exceptions_and_trials(inp)
    assert _get(trials, "lv-planning", "training", 1)["is_exception"] == [False, True, False]
    assert _get(trials, "lv-planning", "training", 2)["is_exception"] == [True, False, False]
    assert _get(exceptions, "lv-planning", "training", 1)["observed"] == 1
    assert _get(exceptions, "lv-planning", "training", 2)["observed"] == 1


def test_regions_are_counted_on_their_own_trials_only():
    inp = _input({"training": [(0.0, 0.0)] * 3, "out-of-range": [(9.0, 9.0)] * 3})
    exceptions, _ = compute_exceptions_and_trials(inp)
    assert _get(exceptions, "lv-control", "training", 1)["observed"] == 0
    assert _get(exceptions, "lv-control", "out-of-range", 1)["observed"] == 3


def test_the_chart_arrays_are_distances_from_the_mean_and_the_intervals_radius_in_the_same_unit():
    inp = _input({"training": [(3.0, 4.0), (0.0, 0.0), (6.0, 8.0)], "out-of-range": [(0.0, 0.0)] * 3}, spread=2.0)
    t = _get(compute_exceptions_and_trials(inp)[1], "lv-control", "training", 1)
    assert t["outcome_distance"] == [pytest.approx(2.0 * math.sqrt((9 + 16) / 2)), 0.0, pytest.approx(2.0 * math.sqrt((36 + 64) / 2))]
    assert t["band_hi"] == [pytest.approx(Z_90 * 2.0)] * 3  # RMS over quantities of z90 * sigma
    assert t["band_lo"] == [0.0] * 3
    assert t["distance_unit"].startswith("rms-normalised")
    assert all(type(x) is float for x in t["outcome_distance"] + t["band_hi"]) and all(type(f) is bool for f in t["is_exception"])


def test_the_interval_radius_is_the_rms_over_quantities_of_each_trials_own_spread():
    kwargs = judge_input_kwargs()
    n = len(kwargs["region_labels"])
    spread = np.ones((n, H, D))
    spread[0, 0, :] = [1.0, 7.0]
    kwargs["predictions"] = Forecasts(np.zeros((n, H, D)), spread)
    kwargs["outcomes"] = np.zeros((n, H, D))
    t = _get(compute_exceptions_and_trials(JudgeInput(**kwargs))[1], "lv-control", "training", 1)
    assert t["band_hi"][0] == pytest.approx(Z_90 * math.sqrt((1 + 49) / 2))
    assert t["band_hi"][1] == pytest.approx(Z_90)


def test_tc_ju9_03_observed_is_the_sum_of_the_flags_for_every_entry_over_many_random_inputs():
    rng = np.random.default_rng(5)
    for _ in range(40):
        kwargs = judge_input_kwargs()
        n = len(kwargs["region_labels"])
        kwargs["predictions"] = Forecasts(rng.normal(size=(n, H, D)), rng.uniform(0.2, 2.0, size=(n, H, D)))
        kwargs["outcomes"] = rng.normal(size=(n, H, D)) * rng.uniform(0.2, 3.0)
        exceptions, trials = compute_exceptions_and_trials(JudgeInput(**kwargs))
        _check_trials_and_exceptions(_check_block("trials", trials), _check_block("exceptions", exceptions))
        for e in exceptions["per_task"]:
            t = _get(trials, e["task"], e["region"], e["horizon_step"])
            assert e["observed"] == sum(t["is_exception"]) and e["n_trials"] == len(t["is_exception"])


def test_a_padded_model_with_a_low_count_is_flagged_not_rewarded_tc_ju8_03():
    wide = _input({"training": [(0.0, 0.0)] * 3, "out-of-range": [(0.0, 0.0)] * 3}, spread=10.0)  # width 33 >> 3.25
    tight = _input({"training": [(0.0, 0.0)] * 3, "out-of-range": [(0.0, 0.0)] * 3}, spread=0.5)  # width 1.6 < 3.25
    flags = lambda inp: [e["low_side_sharpness_flag"] for e in compute_exceptions_and_trials(inp)[0]["per_task"]]
    assert all(flags(wide))
    assert not any(flags(tight))


def test_the_hedging_flag_needs_a_green_or_better_count_and_a_width_over_the_threshold():
    # wide spread but the trials still miss a lot -> the count is above green, so not the 'padded' pattern
    kwargs = judge_input_kwargs()
    n = len(kwargs["region_labels"])
    kwargs["predictions"] = Forecasts(np.zeros((n, H, D)), np.full((n, H, D), 10.0))
    kwargs["outcomes"] = np.full((n, H, D), 500.0)
    e = _get(compute_exceptions_and_trials(JudgeInput(**kwargs))[0], "lv-control", "training", 1)
    assert e["observed"] == 3 and e["low_side_sharpness_flag"] is False
    # exactly one miss (green) with a wide spread: flagged
    inp = _input({"training": [(0.0, 0.0), (0.0, 0.0), (0.0, 0.0)], "out-of-range": [(0.0, 0.0)] * 3}, spread=10.0,
                 step_offsets={(0, 1): (9.0, 9.0)})
    e = _get(compute_exceptions_and_trials(inp)[0], "lv-control", "training", 1)
    assert e["observed"] == 1 and e["band"] == "green" and e["low_side_sharpness_flag"] is True


def test_the_hedge_threshold_is_compared_with_the_mean_width_from_the_sharpness_block():
    inp = make_input()
    widths = {(e["task"], e["region"]): e["mean_width_90"] for e in compute_sharpness(inp)["per_task"]}
    threshold = float(np.mean(inp.thresholds.sharpness_hedge_threshold))
    for e in compute_exceptions_and_trials(inp)[0]["per_task"]:
        assert e["low_side_sharpness_flag"] == (e["observed"] <= 1 and widths[(e["task"], e["region"])] > threshold)


def test_tc_ju8_01_a_single_long_rollout_is_refused_not_counted_as_independent_trials():
    kwargs = judge_input_kwargs()
    long_steps = 400
    kwargs.update(
        predictions=Forecasts(np.zeros((1, long_steps, D)), np.ones((1, long_steps, D))),
        outcomes=np.zeros((1, long_steps, D)),
        persistence=Forecasts(np.zeros((1, long_steps, D)), np.ones((1, long_steps, D))),
        linear=Forecasts(np.zeros((1, long_steps, D)), np.ones((1, long_steps, D))),
        region_labels=(RegionLabel("training", None),),
        divergence_curves=(RegionCurve("training", np.linspace(0, 1, long_steps + 1)),),
        climatology=(RegionClimatology("training", (ClimatologyBin(0.0, 1.0, np.zeros(D), np.ones(D), 5),)),),
        invariant_bins=np.zeros((1, long_steps), dtype=int),
        thresholds=Thresholds(Bands(n=200, p=0.1, green=(12, 29), amber_outer=(8, 35)), np.ones(D), 1.0),
        tasks=(TaskSpec("t", "control", 0.1, 5),),
    )
    with pytest.raises(JudgeInputError, match="pre-registered bands"):
        JudgeInput(**kwargs)


def _two_hundred(stated_sigma, true_sigma, rng):
    n = 200
    mean = np.zeros((n, 1, 1))
    outcome = rng.normal(size=(n, 1, 1)) * true_sigma
    spread = np.full((n, 1, 1), stated_sigma)
    return JudgeInput(
        world="w", dt=0.1, natural_cycle_length=None,
        predictions=Forecasts(mean, spread), outcomes=outcome,
        persistence=Forecasts(mean, np.ones((n, 1, 1))), linear=Forecasts(mean, np.ones((n, 1, 1))),
        region_labels=tuple(RegionLabel("r", None) for _ in range(n)),
        divergence_curves=(RegionCurve("r", np.array([0.0, 1.0])),),
        climatology=(RegionClimatology("r", (ClimatologyBin(0.0, 1.0, np.zeros(1), np.ones(1), 5),)),),
        invariant_bins=np.zeros((n, 1), dtype=int), tasks=(TaskSpec("t", "control", 0.1, 1),),
        thresholds=Thresholds(Bands(n=200, p=0.1, green=(12, 29), amber_outer=(8, 35)), np.array([1.0]), 1.0),
    )


def test_tc_ju8_02_an_honest_ninety_percent_model_lands_green_at_its_false_alarm_rate():
    rng = np.random.default_rng(20260825)
    bands = [compute_exceptions_and_trials(_two_hundred(1.0, 1.0, rng))[0]["per_task"][0]["band"] for _ in range(600)]
    green_rate = bands.count("green") / len(bands)
    assert 0.945 <= green_rate <= 0.985  # the exact binomial says 1 - 0.0331 = 0.9669
    assert "red" not in bands or bands.count("red") / len(bands) < 0.01


def test_a_seventy_percent_model_is_deep_red_and_a_padded_one_is_red_on_the_low_side_and_flagged():
    rng = np.random.default_rng(1)
    over = compute_exceptions_and_trials(_two_hundred(0.3, 1.0, rng))[0]["per_task"][0]
    assert over["band"] == "red" and over["observed"] > 100 and over["low_side_sharpness_flag"] is False
    padded = compute_exceptions_and_trials(_two_hundred(3.0, 1.0, rng))[0]["per_task"][0]
    assert padded["band"] == "red" and padded["observed"] < 8 and padded["low_side_sharpness_flag"] is True


def test_a_whole_verdict_can_be_assembled_from_the_computed_blocks_for_the_same_task_region_pairs():
    inp = make_input()
    skill, calibration, sharpness = compute_skill(inp), compute_calibration(inp), compute_sharpness(inp)
    exceptions, trials = compute_exceptions_and_trials(inp)
    pairs = [(e["task"], e["region"]) for e in skill["per_task_region"]]
    verdict = assemble_verdict(
        world=inp.world, skill=skill, calibration=calibration, sharpness=sharpness, exceptions=exceptions, trials=trials,
        error_vs_horizon={"dt": inp.dt, "per_region": [
            {"region": c.region_name, "steps": list(range(len(c.curve))), "median_error": [0.0] * len(c.curve),
             "divergence_reference": [float(v) for v in c.curve]} for c in inp.divergence_curves]},
        climatology={"per_task": [{"task": t, "region": r, "switch_step": None, "agreement_mean_abs_z": None, "agrees": None}
                                  for t, r in pairs]},
        trust_horizons={"per_task": [{"task": t, "region": r, "tolerance": 0.1, "steps": 0, "world_time": 0.0,
                                      "natural_units": None} for t, r in pairs]},
    )
    record = verdict.to_dict()
    assert len(record["exceptions"]["per_task"]) == 5 and len(record["skill"]["per_task_region"]) == 4


def test_compute_exceptions_and_trials_takes_only_a_real_judge_input():
    for bad in (None, {}, "x"):
        with pytest.raises(JudgeInputError, match="JudgeInput"):
            compute_exceptions_and_trials(bad)


def test_bands_with_no_amber_range_on_a_side_are_refused_rather_than_invented():
    kwargs = judge_input_kwargs()
    kwargs["thresholds"] = Thresholds(Bands(n=3, p=0.1, green=(0, 1), amber_outer=(0, 2)), np.array([4.0, 2.5]), 1.0)
    with pytest.raises(JudgeInputError, match="amber"):
        compute_exceptions_and_trials(JudgeInput(**kwargs))


def test_the_hedging_flag_is_off_for_an_amber_high_count_even_with_wide_ranges():
    # two misses = the amber-high band for n = 3: too many misses is over-confidence, never 'padding'
    inp = _input({"training": [(0.0, 0.0)] * 3, "out-of-range": [(0.0, 0.0)] * 3}, spread=10.0,
                 step_offsets={(0, 1): (9.0, 9.0), (1, 1): (9.0, 9.0)})
    e = _get(compute_exceptions_and_trials(inp)[0], "lv-control", "training", 1)
    assert e["observed"] == 2 and e["band"] == "amber" and e["low_side_sharpness_flag"] is False


def test_a_width_exactly_at_the_hedge_threshold_is_not_flagged():
    exact = float(2.0 * Z_90)  # the mean 90% width of unit spreads, bit for bit
    for threshold, flagged in ((exact, False), (float(np.nextafter(exact, 0.0)), True)):
        inp = _input({"training": [(0.0, 0.0)] * 3, "out-of-range": [(0.0, 0.0)] * 3})
        old = inp.thresholds
        inp = JudgeInput(**{**judge_input_kwargs(), "outcomes": inp.outcomes, "predictions": inp.predictions,
                            "thresholds": Thresholds(old.bands, np.array([threshold, threshold]), old.agreement_threshold)})
        e = _get(compute_exceptions_and_trials(inp)[0], "lv-control", "training", 1)
        assert e["low_side_sharpness_flag"] is flagged
