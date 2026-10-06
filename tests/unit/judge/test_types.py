"""Tests for wmj.judge.types — the judge's input door (JU-1, MU-2, judge spec §4/§7).

In plain words: the judge may only be given predictions, outcomes, the two baselines, region
labels, drift curves, the climatology table, tasks and thresholds. These tests check that no
field could carry a model's identity (and none can be added), that the judge cannot be built
without both baselines, that arrays are copied and locked, and that every inconsistent input
is refused with a clear reason rather than passed on to the arithmetic.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from tests.unit.judge._builders import (
    D,
    H,
    N,
    forecasts,
    judge_input_kwargs,
    make_input,
)
from wmj.judge.errors import JudgeInputError, MissingBaselineError
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

ALL_TYPES = (Bands, ClimatologyBin, Forecasts, JudgeInput, RegionClimatology, RegionCurve, RegionLabel, TaskSpec, Thresholds)

# --- TC-JU1-01: blindness by type -------------------------------------------------------


def test_tc_ju1_01_the_input_has_exactly_the_documented_fields_and_none_can_carry_identity():
    assert [f.name for f in dataclasses.fields(JudgeInput)] == [
        "world", "dt", "natural_cycle_length", "predictions", "outcomes", "persistence", "linear",
        "region_labels", "divergence_curves", "climatology", "invariant_bins", "tasks", "thresholds",
    ]
    forbidden = ("model", "fixture", "arch", "train", "label_", "ref", "seed", "weights", "author", "meta", "id")
    for cls in ALL_TYPES:
        for f in dataclasses.fields(cls):
            assert not any(f.name == w or f.name.startswith(w + "_") for w in forbidden), (cls.__name__, f.name)


def test_tc_ju1_01_every_judge_type_is_frozen_and_slotted_so_no_field_can_be_attached_later():
    obj = make_input()
    for attempt in ("model_name", "is_fixture", "model_ref", "architecture", "training_history"):
        with pytest.raises((AttributeError, TypeError)):
            setattr(obj, attempt, "x")
        with pytest.raises(TypeError):
            JudgeInput(**{**judge_input_kwargs(), attempt: "x"})
    with pytest.raises(dataclasses.FrozenInstanceError):
        obj.world = "other"
    for cls in ALL_TYPES:
        assert hasattr(cls, "__slots__"), cls.__name__
        assert cls.__dataclass_params__.frozen is True, cls.__name__
    assert not hasattr(obj, "__dict__")


# --- TC-MU2-01: no verdict without both baselines ------------------------------------------


@pytest.mark.parametrize("missing", ["persistence", "linear"])
def test_tc_mu2_01_the_judge_refuses_an_input_without_either_baseline(missing):
    with pytest.raises(MissingBaselineError, match=missing):
        make_input(**{missing: None})


def test_tc_mu2_01_both_missing_is_also_refused_and_a_full_input_is_accepted():
    with pytest.raises(MissingBaselineError):
        make_input(persistence=None, linear=None)
    assert make_input().world == "lv"


def test_a_missing_baseline_error_is_a_judge_error_not_a_generic_value_error():
    from wmj.judge.errors import JudgeError

    assert issubclass(MissingBaselineError, JudgeError) and not issubclass(MissingBaselineError, ValueError)
    assert issubclass(JudgeInputError, JudgeError) and issubclass(JudgeInputError, ValueError)


# --- arrays are copied and locked ------------------------------------------------------------


def test_arrays_are_copied_and_read_only_so_the_callers_edits_cannot_reach_the_judge():
    kwargs = judge_input_kwargs()
    outcomes = np.array(kwargs["outcomes"])
    kwargs["outcomes"] = outcomes
    inp = JudgeInput(**kwargs)
    outcomes[:] = 999.0
    assert not np.any(inp.outcomes == 999.0)
    for arr in (inp.outcomes, inp.predictions.mean, inp.predictions.spread, inp.persistence.mean,
                inp.linear.spread, inp.invariant_bins, inp.divergence_curves[0].curve,
                inp.thresholds.sharpness_hedge_threshold, inp.climatology[0].bins[0].mean):
        assert arr.flags.writeable is False
        with pytest.raises(ValueError):
            arr[...] = 0


# --- shape and consistency refusals ------------------------------------------------------------


def test_the_baselines_and_predictions_must_match_the_outcomes_shape():
    for field in ("predictions", "persistence", "linear"):
        with pytest.raises(JudgeInputError, match=field):
            make_input(**{field: forecasts(0, n=N, h=H + 1)})
        with pytest.raises(JudgeInputError, match=field):
            make_input(**{field: forecasts(0, n=N + 1)})
        with pytest.raises(JudgeInputError, match=field):
            make_input(**{field: forecasts(0, d=D + 1)})


def test_outcomes_must_be_three_dimensional_finite_and_non_empty():
    with pytest.raises(JudgeInputError, match="dimension"):
        make_input(outcomes=np.zeros((N, H)))
    bad = np.zeros((N, H, D))
    bad[0, 0, 0] = np.nan
    with pytest.raises(JudgeInputError, match="NaN or infinity"):
        make_input(outcomes=bad)
    bad[0, 0, 0] = np.inf
    with pytest.raises(JudgeInputError, match="NaN or infinity"):
        make_input(outcomes=bad)


@pytest.mark.parametrize("bad", [0.0, -0.1, np.nan, np.inf])
def test_a_spread_with_no_width_or_that_is_not_finite_is_refused(bad):
    spread = np.full((N, H, D), 0.5)
    spread[2, 1, 0] = bad
    with pytest.raises(JudgeInputError):
        Forecasts(mean=np.zeros((N, H, D)), spread=spread)


def test_forecast_mean_and_spread_must_agree_in_shape():
    with pytest.raises(JudgeInputError, match="same shape"):
        Forecasts(mean=np.zeros((N, H, D)), spread=np.ones((N, H, D + 1)))


def test_region_labels_need_one_per_trial_and_valid_axes():
    labels = judge_input_kwargs()["region_labels"]
    with pytest.raises(JudgeInputError, match="one RegionLabel per trial"):
        make_input(region_labels=labels[:-1])
    for axis in ("state", "action", "both", None):
        RegionLabel("r", axis)
    with pytest.raises(JudgeInputError, match="axis"):
        RegionLabel("r", "diagonal")
    with pytest.raises(JudgeInputError, match="region_name"):
        RegionLabel("", None)


def test_divergence_curves_must_cover_exactly_the_regions_present_with_h_plus_one_points():
    curves = judge_input_kwargs()["divergence_curves"]
    with pytest.raises(JudgeInputError, match="exactly the regions"):
        make_input(divergence_curves=curves[:1])
    with pytest.raises(JudgeInputError, match="exactly the regions"):
        make_input(divergence_curves=curves + (RegionCurve("extra", np.zeros(H + 1)),))
    with pytest.raises(JudgeInputError, match="H\\+1"):
        make_input(divergence_curves=(RegionCurve("training", np.zeros(H)), curves[1]))
    with pytest.raises(JudgeInputError, match="none negative"):
        RegionCurve("training", np.array([0.0, -1.0, 1.0]))


def test_climatology_must_cover_the_regions_and_the_quantities_and_invariant_bins_must_index_it():
    tables = judge_input_kwargs()["climatology"]
    with pytest.raises(JudgeInputError, match="exactly the regions"):
        make_input(climatology=tables[:1])
    bins = np.zeros((N, H), dtype=int)
    bins[0, 0] = 4  # the table has 4 bins: index 4 is out of range
    with pytest.raises(JudgeInputError, match="outside its region"):
        make_input(invariant_bins=bins)
    bins[0, 0] = -1
    with pytest.raises(JudgeInputError, match="outside its region"):
        make_input(invariant_bins=bins)
    with pytest.raises(JudgeInputError, match=r"\[n_trials, H\]"):
        make_input(invariant_bins=np.zeros((N, H + 1), dtype=int))
    wide = ClimatologyBin(0.0, 1.0, np.zeros(D + 1), np.ones(D + 1), 10)
    with pytest.raises(JudgeInputError, match="quantities"):
        make_input(climatology=(RegionClimatology("training", (wide,)), tables[1]))


def test_climatology_bin_rules():
    ClimatologyBin(-np.inf, 1.0, np.zeros(2), np.ones(2), 5)
    ClimatologyBin(0.0, np.inf, np.zeros(2), np.ones(2), 5)
    for args in ((1.0, 1.0), (2.0, 1.0), (np.nan, 1.0)):
        with pytest.raises(JudgeInputError, match="invariant_lo < invariant_hi"):
            ClimatologyBin(*args, np.zeros(2), np.ones(2), 5)
    with pytest.raises(JudgeInputError, match="sd must be > 0"):
        ClimatologyBin(0.0, 1.0, np.zeros(2), np.array([1.0, 0.0]), 5)
    with pytest.raises(JudgeInputError, match="n_samples"):
        ClimatologyBin(0.0, 1.0, np.zeros(2), np.ones(2), 0)
    with pytest.raises(JudgeInputError, match="same length"):
        ClimatologyBin(0.0, 1.0, np.zeros(2), np.ones(3), 5)
    with pytest.raises(JudgeInputError, match="at least one"):
        RegionClimatology("r", ())


def test_tasks_must_be_valid_unique_and_fit_inside_the_rollouts():
    with pytest.raises(JudgeInputError, match="non-empty"):
        make_input(tasks=())
    t = TaskSpec("a", "control", 0.1, 3)
    with pytest.raises(JudgeInputError, match="unique"):
        make_input(tasks=(t, t))
    with pytest.raises(JudgeInputError, match="exceeds"):
        make_input(tasks=(TaskSpec("a", "control", 0.1, H + 1),))
    make_input(tasks=(TaskSpec("a", "control", 0.1, H),))  # exactly H is allowed (closed)
    with pytest.raises(JudgeInputError, match="kind"):
        TaskSpec("a", "navigation", 0.1, 3)
    for tol in (0.0, -1.0, np.nan, np.inf, True):
        with pytest.raises(JudgeInputError, match="tolerance"):
            TaskSpec("a", "control", tol, 3)
    for horizon in (0, -1, 2.5, True):
        with pytest.raises(JudgeInputError, match="horizon"):
            TaskSpec("a", "control", 0.1, horizon)
    with pytest.raises(JudgeInputError, match="name"):
        TaskSpec("", "control", 0.1, 3)


def test_thresholds_bands_must_nest_and_hedge_vector_must_match_the_quantities():
    Bands(n=200, p=0.1, green=(12, 29), amber_outer=(8, 35))
    for kwargs in (
        {"green": (12, 29), "amber_outer": (13, 35)},  # amber starts inside green
        {"green": (12, 29), "amber_outer": (8, 28)},  # amber ends inside green
        {"green": (30, 29), "amber_outer": (8, 35)},  # green inverted
        {"green": (12, 29), "amber_outer": (8, 201)},  # beyond n
        {"green": (12, 29), "amber_outer": (-1, 35)},
    ):
        with pytest.raises(JudgeInputError, match="nest"):
            Bands(n=200, p=0.1, **kwargs)
    for p in (0.0, 1.0, -0.1, True):
        with pytest.raises(JudgeInputError, match="p must"):
            Bands(n=200, p=p, green=(12, 29), amber_outer=(8, 35))
    with pytest.raises(JudgeInputError, match="n must"):
        Bands(n=0, p=0.1, green=(0, 0), amber_outer=(0, 0))
    bands = Bands(n=200, p=0.1, green=(12, 29), amber_outer=(8, 35))
    with pytest.raises(JudgeInputError, match="positive"):
        Thresholds(bands, np.array([1.0, 0.0]), 1.0)
    with pytest.raises(JudgeInputError, match="agreement_threshold"):
        Thresholds(bands, np.array([1.0, 1.0]), 0.0)
    with pytest.raises(JudgeInputError, match="one entry per quantity"):
        make_input(thresholds=Thresholds(bands, np.ones(D + 1), 1.0))


def test_world_dt_and_cycle_length_rules():
    with pytest.raises(JudgeInputError, match="world"):
        make_input(world="")
    for dt in (0.0, -0.02, np.nan, np.inf, True, "0.02"):
        with pytest.raises(JudgeInputError, match="dt"):
            make_input(dt=dt)
    assert make_input(natural_cycle_length=None).natural_cycle_length is None  # the pendulum has none
    for bad in (0.0, -1.0, np.nan):
        with pytest.raises(JudgeInputError, match="natural_cycle_length"):
            make_input(natural_cycle_length=bad)


def test_wrong_object_types_are_refused_not_crashed_on():
    with pytest.raises(JudgeInputError, match="predictions must be a Forecasts"):
        make_input(predictions="not forecasts")
    with pytest.raises(JudgeInputError, match="thresholds must be a Thresholds"):
        make_input(thresholds={"bands": 1})
    with pytest.raises(JudgeInputError, match="RegionLabel"):
        make_input(region_labels=("training",) * N)
    with pytest.raises(JudgeInputError, match="RegionCurve"):
        make_input(divergence_curves=("training", "out-of-range"))
    with pytest.raises(JudgeInputError, match="RegionClimatology"):
        make_input(climatology=("training", "out-of-range"))
    with pytest.raises(JudgeInputError, match="TaskSpec"):
        make_input(tasks=("lv-control",))


def test_numpy_integer_and_float_inputs_are_accepted_and_normalised_to_python_numbers():
    task = TaskSpec("a", "control", np.float64(0.1), np.int64(3))
    assert type(task.tolerance) is float and type(task.horizon) is int
    bands = Bands(n=np.int64(200), p=0.1, green=(np.int64(12), 29), amber_outer=(8, 35))
    assert type(bands.n) is int and bands.green == (12, 29)
