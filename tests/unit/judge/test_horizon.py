"""Tests for wmj.judge.horizon — the error-versus-horizon curve and the trust horizons (ADR-J5, JU-3, JU-7).

In plain words: build forecasts whose distance from the truth is known exactly at every step, and
check (1) the curve of typical (median) error per step, beside the world's own drift curve, and
(2) the trust horizon — the last step up to which the typical error stayed within the task's
tolerance at every step — at every edge: failing at step 1, failing part-way, landing exactly on
the tolerance, being capped at the switch step, and having no switch step at all; plus the units
(steps, world time, fraction of the natural cycle) and that the verdict door accepts the result.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tests.unit.judge._builders import D, H, judge_input_kwargs, make_input
from wmj.judge.calibration import compute_calibration
from wmj.judge.climatology import compute_climatology
from wmj.judge.errors import JudgeInputError
from wmj.judge.exceptions import compute_exceptions_and_trials
from wmj.judge.horizon import compute_error_vs_horizon, compute_trust_horizons
from wmj.judge.sharpness import compute_sharpness
from wmj.judge.skill import compute_skill
from wmj.judge.types import Forecasts, JudgeInput, RegionCurve, TaskSpec
from wmj.judge.verdict import (
    _check_block,
    _check_error_vs_horizon,
    _check_trust_horizons,
    assemble_verdict,
)

ROWS = {"training": [0, 1, 2], "out-of-range": [3, 4, 5]}
# the world's drift: control (tolerance 0.1) switches at step 3, planning (0.3) at step 4, in both regions
CURVES = (
    RegionCurve("training", np.array([0.0, 0.01, 0.02, 0.2, 0.5, 0.6])),
    RegionCurve("out-of-range", np.array([0.0, 0.01, 0.02, 0.2, 0.5, 0.6])),
)


def _with_distances(per_region, *, curves=CURVES, tasks=None, natural=3.5, dt=0.02):
    """Predictions all zero; the outcome of every trial in a region sits `d[step]` away in every quantity.

    With the same gap in both quantities the root-mean-square distance is exactly that gap."""
    kwargs = judge_input_kwargs()
    n = len(kwargs["region_labels"])
    outcomes = np.zeros((n, H, D))
    for region, gaps in per_region.items():
        for row in ROWS[region]:
            outcomes[row, :, :] = np.array(gaps, dtype=float)[:, None]
    kwargs.update(
        predictions=Forecasts(np.zeros((n, H, D)), np.ones((n, H, D))), outcomes=outcomes,
        divergence_curves=curves, natural_cycle_length=natural, dt=dt,
    )
    if tasks is not None:
        kwargs["tasks"] = tasks
    return JudgeInput(**kwargs)


def _trust(block, task, region):
    return next(e for e in block["per_task"] if (e["task"], e["region"]) == (task, region))


def _region(block, region):
    return next(e for e in block["per_region"] if e["region"] == region)


# ---- error_vs_horizon ----------------------------------------------------------------------------------


def test_the_curve_has_one_point_per_step_from_zero_with_a_zero_origin_and_the_worlds_drift_beside_it():
    inp = _with_distances({"training": [0.05, 0.08, 0.09, 0.2, 0.3], "out-of-range": [0.5] * 5})
    block = compute_error_vs_horizon(inp)
    assert block["dt"] == 0.02
    assert [e["region"] for e in block["per_region"]] == ["out-of-range", "training"]
    e = _region(block, "training")
    assert e["steps"] == [0, 1, 2, 3, 4, 5]
    assert e["median_error"] == [0.0] + [pytest.approx(x, rel=1e-12) for x in (0.05, 0.08, 0.09, 0.2, 0.3)]
    assert e["divergence_reference"] == [0.0, 0.01, 0.02, 0.2, 0.5, 0.6]
    assert all(type(x) is float for x in e["median_error"] + e["divergence_reference"])
    _check_error_vs_horizon(block)


def test_the_error_at_a_step_is_the_median_over_the_regions_trials_not_the_mean_and_only_its_own():
    kwargs = judge_input_kwargs()
    n = len(kwargs["region_labels"])
    outcomes = np.zeros((n, H, D))
    for row, gap in zip((0, 1, 2), (1.0, 1.0, 10.0), strict=True):  # training trials: median 1, mean 4
        outcomes[row] = gap
    outcomes[3:] = 100.0  # the other region must not leak in
    kwargs.update(predictions=Forecasts(np.zeros((n, H, D)), np.ones((n, H, D))), outcomes=outcomes)
    e = _region(compute_error_vs_horizon(JudgeInput(**kwargs)), "training")
    assert e["median_error"][1:] == [pytest.approx(1.0)] * H


def test_the_distance_is_the_root_mean_square_over_quantities_and_uses_the_step_it_names():
    kwargs = judge_input_kwargs()
    n = len(kwargs["region_labels"])
    outcomes = np.zeros((n, H, D))
    outcomes[:, 1, :] = [3.0, 4.0]  # step 2: rms of (3, 4) = sqrt(12.5), not 5 (norm) and not 3.5 (mean)
    predictions = np.zeros((n, H, D))
    predictions[:, 3, :] = [1.0, 1.0]  # step 4: the predicted mean matters, not only the outcome
    outcomes[:, 3, :] = [1.0, 1.0]  # ... so this step has distance 0
    kwargs.update(predictions=Forecasts(predictions, np.ones((n, H, D))), outcomes=outcomes)
    e = _region(compute_error_vs_horizon(JudgeInput(**kwargs)), "training")
    assert e["median_error"] == [0.0, 0.0, pytest.approx(math.sqrt(12.5)), 0.0, 0.0, 0.0]


def test_the_error_block_passes_the_verdict_doors_own_check_and_is_plain_data():
    block = compute_error_vs_horizon(make_input())
    _check_error_vs_horizon(block)
    assert all(type(e["steps"][0]) is int for e in block["per_region"])


def test_an_error_too_large_to_represent_is_refused_not_reported():
    kwargs = judge_input_kwargs()
    n = len(kwargs["region_labels"])
    kwargs.update(predictions=Forecasts(np.full((n, H, D), -1.5e308), np.ones((n, H, D))), outcomes=np.full((n, H, D), 1.5e308))
    with pytest.raises(JudgeInputError, match="overflow"):
        compute_error_vs_horizon(JudgeInput(**kwargs))


def test_the_functions_take_only_a_real_judge_input():
    for fn in (compute_error_vs_horizon, compute_trust_horizons):
        for bad in (None, {}, "x"):
            with pytest.raises(JudgeInputError, match="JudgeInput"):
                fn(bad)


# ---- trust horizons ------------------------------------------------------------------------------------


def test_trust_is_the_last_step_up_to_which_every_step_is_within_tolerance_capped_at_the_switch_step():
    # control: tolerance 0.1, switch 3. errors 0.05, 0.08, 0.09 are all within -> 3 (the cap), though step 4 is also fine
    inp = _with_distances({"training": [0.05, 0.08, 0.09, 0.05, 0.05], "out-of-range": [0.05] * 5})
    t = _trust(compute_trust_horizons(inp), "lv-control", "training")
    assert (t["steps"], t["tolerance"]) == (3, 0.1)
    # planning: tolerance 0.3, switch 4. error 0.31 at step 4 -> 3; it would be 4 without it
    inp = _with_distances({"training": [0.05, 0.08, 0.09, 0.31, 0.05], "out-of-range": [0.05] * 5})
    assert _trust(compute_trust_horizons(inp), "lv-planning", "training")["steps"] == 3
    inp = _with_distances({"training": [0.05, 0.08, 0.09, 0.29, 0.05], "out-of-range": [0.05] * 5})
    assert _trust(compute_trust_horizons(inp), "lv-planning", "training")["steps"] == 4  # capped at its switch step 4, not 5


def test_a_failure_after_the_switch_step_but_inside_the_horizon_cannot_extend_or_distort_the_trust():
    early = (RegionCurve("training", np.array([0.0, 0.01, 0.5, 0.6, 0.7, 0.8])), CURVES[1])  # planning (0.3) switches at step 2
    inp = _with_distances({"training": [0.05, 0.05, 0.05, 0.9, 0.9], "out-of-range": [0.05] * 5}, curves=early)
    assert _trust(compute_trust_horizons(inp), "lv-planning", "training")["steps"] == 2  # capped at the switch step, not 3 or 5


def test_one_bad_step_ends_the_trust_even_if_later_steps_look_fine():
    inp = _with_distances({"training": [0.05, 0.2, 0.05, 0.05, 0.05], "out-of-range": [0.05] * 5})
    assert _trust(compute_trust_horizons(inp), "lv-control", "training")["steps"] == 1


def test_failing_tolerance_at_step_one_is_an_explicit_zero():
    inp = _with_distances({"training": [0.5, 0.01, 0.01, 0.01, 0.01], "out-of-range": [0.05] * 5})
    t = _trust(compute_trust_horizons(inp), "lv-control", "training")
    assert t["steps"] == 0 and t["world_time"] == 0.0 and t["natural_units"] == "0.00 cycles"


def test_an_error_exactly_equal_to_the_tolerance_is_still_within_it_and_one_ulp_over_is_not():
    quiet = (RegionCurve("training", np.zeros(H + 1)), RegionCurve("out-of-range", np.zeros(H + 1)))
    gaps = {"training": [0.01, 0.01, 0.05, 0.9, 0.9], "out-of-range": [0.05] * 5}
    measured = _region(compute_error_vs_horizon(_with_distances(gaps, curves=quiet)), "training")["median_error"][3]
    for tolerance, expected in ((measured, 3), (float(np.nextafter(measured, 0.0)), 2)):
        inp = _with_distances(gaps, curves=quiet, tasks=(TaskSpec("t", "control", tolerance, 5),))
        assert _trust(compute_trust_horizons(inp), "t", "training")["steps"] == expected


def test_with_no_switch_step_the_trust_horizon_is_capped_at_the_tasks_own_horizon_not_the_worlds():
    quiet = (RegionCurve("training", np.zeros(H + 1)), RegionCurve("out-of-range", np.zeros(H + 1)))
    tasks = (TaskSpec("short", "control", 0.1, 3),)
    inp = _with_distances({"training": [0.01] * 5, "out-of-range": [0.01] * 5}, curves=quiet, tasks=tasks)
    assert _trust(compute_trust_horizons(inp), "short", "training")["steps"] == 3  # the world runs 5 steps; the task asks for 3


def test_each_region_gets_its_own_trust_horizon():
    inp = _with_distances({"training": [0.05, 0.08, 0.09, 0.05, 0.05], "out-of-range": [0.5, 0.01, 0.01, 0.01, 0.01]})
    block = compute_trust_horizons(inp)
    assert _trust(block, "lv-control", "training")["steps"] == 3
    assert _trust(block, "lv-control", "out-of-range")["steps"] == 0


def test_trust_is_reported_in_steps_world_time_and_fractions_of_the_natural_cycle():
    inp = _with_distances({"training": [0.05, 0.08, 0.09, 0.05, 0.05], "out-of-range": [0.05] * 5}, dt=0.5, natural=3.5)
    t = _trust(compute_trust_horizons(inp), "lv-control", "training")
    assert t["steps"] == 3 and t["world_time"] == 1.5 and t["natural_units"] == f"{1.5 / 3.5:.2f} cycles" == "0.43 cycles"
    assert type(t["world_time"]) is float and type(t["steps"]) is int
    pendulum = _with_distances({"training": [0.05, 0.08, 0.09, 0.05, 0.05], "out-of-range": [0.05] * 5}, natural=None)
    assert _trust(compute_trust_horizons(pendulum), "lv-control", "training")["natural_units"] is None


def test_the_entries_are_one_per_task_and_region_in_the_fixed_order_and_pass_the_doors_own_checks():
    inp = make_input()
    block = compute_trust_horizons(inp)
    assert [(e["task"], e["region"]) for e in block["per_task"]] == [
        ("lv-control", "out-of-range"), ("lv-control", "training"), ("lv-planning", "out-of-range"), ("lv-planning", "training")]
    _check_trust_horizons(_check_block("trust_horizons", block), inp.dt)


def test_a_whole_nine_group_verdict_assembles_from_the_computed_blocks_and_trust_never_passes_the_switch():
    inp = _with_distances({"training": [0.05] * 5, "out-of-range": [0.05] * 5})
    exceptions, trials = compute_exceptions_and_trials(inp)
    verdict = assemble_verdict(
        world=inp.world, skill=compute_skill(inp), error_vs_horizon=compute_error_vs_horizon(inp),
        calibration=compute_calibration(inp), sharpness=compute_sharpness(inp), exceptions=exceptions, trials=trials,
        climatology=compute_climatology(inp), trust_horizons=compute_trust_horizons(inp),
    )
    record = verdict.to_dict()
    for entry in record["trust_horizons"]["per_task"]:
        switch = next(c["switch_step"] for c in record["climatology"]["per_task"]
                      if (c["task"], c["region"]) == (entry["task"], entry["region"]))
        assert switch is None or entry["steps"] <= switch


# --- review pass 1 --------------------------------------------------------------------------------------


def _one_region(gaps_per_trial, *, bands, task=None, curve=None, steps=2):
    """One region, `len(gaps_per_trial)` trials, one quantity; trial i is `gaps_per_trial[i]` away at every step."""
    from wmj.judge.types import (
        Bands,
        ClimatologyBin,
        RegionClimatology,
        RegionLabel,
        Thresholds,
    )

    n = len(gaps_per_trial)
    outcomes = np.repeat(np.array(gaps_per_trial, dtype=float)[:, None, None], steps, axis=1)
    bins = (ClimatologyBin(-np.inf, np.inf, np.zeros(1), np.ones(1), 5),)
    return JudgeInput(
        world="w", dt=0.1, natural_cycle_length=None,
        predictions=Forecasts(np.zeros((n, steps, 1)), np.ones((n, steps, 1))), outcomes=outcomes,
        persistence=Forecasts(np.zeros((n, steps, 1)), np.ones((n, steps, 1))),
        linear=Forecasts(np.zeros((n, steps, 1)), np.ones((n, steps, 1))),
        region_labels=tuple(RegionLabel("r", None) for _ in range(n)),
        divergence_curves=(RegionCurve("r", np.zeros(steps + 1) if curve is None else curve),),
        climatology=(RegionClimatology("r", bins),), invariant_bins=np.zeros((n, steps), dtype=np.int64),
        tasks=(task or TaskSpec("t", "control", 1.0, steps),),
        thresholds=Thresholds(Bands(**bands), np.array([1.0]), 1.0),
    )


THREE = {"n": 3, "p": 0.1, "green": (1, 1), "amber_outer": (0, 2)}
FOUR = {"n": 4, "p": 0.1, "green": (1, 3), "amber_outer": (0, 4)}


def test_the_typical_error_is_the_true_median_for_odd_and_even_trial_counts_whatever_the_order():
    for gaps, expected in (
        ([1.0, 2.0, 4.0, 9.0], 3.0),  # even: the average of the two middle values (2 and 4), not either of them
        ([9.0, 4.0, 2.0, 1.0], 3.0),
        ([9.0, 1.0, 4.0], 4.0),  # odd, and trial 0 is not the middle
        ([4.0, 9.0, 1.0], 4.0),
        ([1.0, 9.0, 4.0], 4.0),
    ):
        inp = _one_region(gaps, bands=FOUR if len(gaps) == 4 else THREE)
        assert compute_error_vs_horizon(inp)["per_region"][0]["median_error"] == [0.0, pytest.approx(expected), pytest.approx(expected)]


def test_a_huge_but_representable_median_is_returned_for_even_counts_just_as_for_odd_ones():
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # the average of the two middle values overflows if added first
        even = compute_error_vs_horizon(_one_region([1.0e308, 1.2e308, 1.6e308, 1.7e308], bands=FOUR))
        odd = compute_error_vs_horizon(_one_region([1.7e308, 1.7e308, 1.7e308], bands=THREE))
    assert even["per_region"][0]["median_error"] == [0.0, pytest.approx(1.4e308), pytest.approx(1.4e308)]  # (1.2e308 + 1.6e308) / 2
    assert odd["per_region"][0]["median_error"] == [0.0, 1.7e308, 1.7e308]


def test_a_world_time_too_large_to_represent_is_refused_not_passed_on_as_infinity():
    kwargs = judge_input_kwargs()
    kwargs["dt"] = 1e308
    n = len(kwargs["region_labels"])
    kwargs.update(natural_cycle_length=None, predictions=Forecasts(np.zeros((n, H, D)), np.ones((n, H, D))), outcomes=np.zeros((n, H, D)))
    with pytest.raises(JudgeInputError, match="overflow"):
        compute_trust_horizons(JudgeInput(**kwargs))


def test_a_tie_at_the_tolerance_is_exact_even_when_the_gaps_are_unequal():
    # gaps (1, 1) and (5): the root-mean-square of (1, 1, 5) is exactly 3, so tolerance 3.0 is still within it
    inp = _one_region_multi_quantity([1.0, 1.0, 5.0], tolerance=3.0)
    assert compute_trust_horizons(inp)["per_task"][0]["steps"] == 2
    inp = _one_region_multi_quantity([1.0, 1.0, 5.0], tolerance=float(np.nextafter(3.0, 0.0)))
    assert compute_trust_horizons(inp)["per_task"][0]["steps"] == 0


def _one_region_multi_quantity(gaps, *, tolerance):
    """Three trials of one region and ONE step pair; the three quantities are `gaps` away in every trial."""
    from wmj.judge.types import (
        Bands,
        ClimatologyBin,
        RegionClimatology,
        RegionLabel,
        Thresholds,
    )

    n, d, steps = 3, len(gaps), 2
    outcomes = np.tile(np.array(gaps, dtype=float), (n, steps, 1))
    bins = (ClimatologyBin(-np.inf, np.inf, np.zeros(d), np.ones(d), 5),)
    return JudgeInput(
        world="w", dt=0.1, natural_cycle_length=None,
        predictions=Forecasts(np.zeros((n, steps, d)), np.ones((n, steps, d))), outcomes=outcomes,
        persistence=Forecasts(np.zeros((n, steps, d)), np.ones((n, steps, d))),
        linear=Forecasts(np.zeros((n, steps, d)), np.ones((n, steps, d))),
        region_labels=tuple(RegionLabel("r", None) for _ in range(n)),
        divergence_curves=(RegionCurve("r", np.zeros(steps + 1)),),
        climatology=(RegionClimatology("r", bins),), invariant_bins=np.zeros((n, steps), dtype=np.int64),
        tasks=(TaskSpec("t", "control", tolerance, steps),),
        thresholds=Thresholds(Bands(**THREE), np.ones(d), 1.0),
    )


def test_each_region_gets_its_own_divergence_reference_and_trials_are_paired_with_their_own_outcomes():
    different = (RegionCurve("training", np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])),
                 RegionCurve("out-of-range", np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])))
    kwargs = judge_input_kwargs()
    n = len(kwargs["region_labels"])
    predictions = np.zeros((n, H, D))
    predictions[:, :, :] = np.arange(n, dtype=float)[:, None, None]  # trial i predicts i in every quantity
    outcomes = predictions + np.array([1.0, 3.0, 2.0, 10.0, 30.0, 20.0])[:, None, None]  # trial i is off by its own gap
    kwargs.update(predictions=Forecasts(predictions, np.ones((n, H, D))), outcomes=outcomes, divergence_curves=different)
    block = compute_error_vs_horizon(JudgeInput(**kwargs))
    assert _region(block, "training")["divergence_reference"] == pytest.approx([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    assert _region(block, "out-of-range")["divergence_reference"] == pytest.approx([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    # a trial paired with another trial's outcome would give a different set of gaps and so a different median
    assert _region(block, "training")["median_error"][1:] == [pytest.approx(2.0)] * H
    assert _region(block, "out-of-range")["median_error"][1:] == [pytest.approx(20.0)] * H
def test_a_switch_step_beyond_the_tasks_horizon_does_not_exist_for_that_task_and_trust_stops_at_the_horizon():
    # the world first drifts past the tolerance at step 4, but this task only goes to step 3
    late = np.array([0.0, 0.0, 0.0, 0.0, 5.0, 5.0])
    task = TaskSpec("short", "control", 1.0, 3)
    inp = _one_region([0.0, 0.0, 0.0], bands=THREE, task=task, curve=late, steps=5)
    t = compute_trust_horizons(inp)["per_task"][0]
    assert t["steps"] == 3  # the task's horizon; not 5, and not "capped at the switch step 4"
    assert compute_climatology(inp)["per_task"][0]["switch_step"] is None


def test_a_natural_cycle_fraction_too_large_to_represent_is_refused():
    kwargs = judge_input_kwargs()
    kwargs["natural_cycle_length"] = 1e-300
    kwargs["dt"] = 1e300
    n = len(kwargs["region_labels"])
    kwargs.update(predictions=Forecasts(np.zeros((n, H, D)), np.ones((n, H, D))), outcomes=np.zeros((n, H, D)))
    with pytest.raises(JudgeInputError, match="overflow"):
        compute_trust_horizons(JudgeInput(**kwargs))
