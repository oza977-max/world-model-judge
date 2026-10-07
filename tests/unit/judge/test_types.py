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


# --- review pass 1 ----------------------------------------------------------------------------------

EXACT_FIELDS = {
    RegionLabel: ["region_name", "axis"],
    TaskSpec: ["name", "kind", "tolerance", "horizon"],
    Forecasts: ["mean", "spread"],
    Bands: ["n", "p", "green", "amber_outer"],
    Thresholds: ["bands", "sharpness_hedge_threshold", "agreement_threshold"],
    ClimatologyBin: ["invariant_lo", "invariant_hi", "mean", "sd", "n_samples"],
    RegionClimatology: ["region_name", "bins"],
    RegionCurve: ["region_name", "curve"],
}


@pytest.mark.parametrize("cls", list(EXACT_FIELDS))
def test_tc_ju1_01_every_judge_input_type_has_exactly_its_documented_fields(cls):
    assert [f.name for f in dataclasses.fields(cls)] == EXACT_FIELDS[cls]


def test_the_only_text_fields_anywhere_in_the_input_are_the_world_region_task_names_kind_and_axis():
    text_fields = set()
    for cls in (JudgeInput, *EXACT_FIELDS):
        for f in dataclasses.fields(cls):
            if "str" in str(f.type):
                text_fields.add((cls.__name__, f.name))
    assert text_fields == {
        ("JudgeInput", "world"), ("RegionLabel", "region_name"), ("RegionLabel", "axis"), ("TaskSpec", "name"),
        ("TaskSpec", "kind"), ("RegionClimatology", "region_name"), ("RegionCurve", "region_name"),
    }


def test_a_subclass_or_a_str_subclass_cannot_carry_extra_state_into_the_judge():
    class S(str):
        pass

    for kwargs in ({"world": S("lv")},):
        with pytest.raises(JudgeInputError, match="plain non-blank string"):
            make_input(**kwargs)
    with pytest.raises(JudgeInputError, match="plain non-blank string"):
        TaskSpec(S("a"), "control", 0.1, 3)
    with pytest.raises(JudgeInputError, match="plain non-blank string"):
        RegionLabel(S("r"), None)
    with pytest.raises(JudgeInputError, match="plain non-blank string"):
        RegionCurve(S("r"), np.zeros(3))
    with pytest.raises(JudgeInputError, match="plain non-blank string"):
        RegionClimatology(S("r"), (ClimatologyBin(0.0, 1.0, np.zeros(2), np.ones(2), 3),))
    with pytest.raises(JudgeInputError, match="axis"):
        RegionLabel("r", S("state"))
    # subclasses cannot even be defined, so the exact-type checks are a second line of defence
    for cls in ALL_TYPES:
        with pytest.raises(TypeError, match="cannot be subclassed"):
            type("Sub", (cls,), {})


@pytest.mark.parametrize("name", ["", "  ", "\t"])
def test_blank_names_are_refused_everywhere(name):
    with pytest.raises(JudgeInputError):
        make_input(world=name)
    with pytest.raises(JudgeInputError):
        TaskSpec(name, "control", 0.1, 3)
    with pytest.raises(JudgeInputError):
        RegionLabel(name, None)
    with pytest.raises(JudgeInputError):
        RegionCurve(name, np.zeros(3))
    with pytest.raises(JudgeInputError):
        RegionClimatology(name, (ClimatologyBin(0.0, 1.0, np.zeros(2), np.ones(2), 3),))


def test_every_guard_inside_the_small_types_can_fail():
    with pytest.raises(JudgeInputError, match="non-empty in every dimension"):
        make_input(outcomes=np.zeros((0, H, D)))
    with pytest.raises(JudgeInputError, match="same number of quantities"):
        RegionClimatology("r", (
            ClimatologyBin(0.0, 1.0, np.zeros(2), np.ones(2), 3),
            ClimatologyBin(1.0, 2.0, np.zeros(3), np.ones(3), 3),
        ))
    for pair in ((12,), (12, 29, 30), "ab", None, 5):
        with pytest.raises(JudgeInputError, match="pair"):
            Bands(n=200, p=0.1, green=pair, amber_outer=(8, 35))
        with pytest.raises(JudgeInputError, match="pair"):
            Bands(n=200, p=0.1, green=(12, 29), amber_outer=pair)
    with pytest.raises(JudgeInputError, match="thresholds.bands must be a Bands"):
        Thresholds("bands", np.ones(2), 1.0)
    with pytest.raises(JudgeInputError, match="non-empty vector"):
        Thresholds(Bands(n=200, p=0.1, green=(12, 29), amber_outer=(8, 35)), np.array([]), 1.0)
    with pytest.raises(JudgeInputError, match="at least one ClimatologyBin"):
        RegionClimatology("r", ())
    with pytest.raises(JudgeInputError, match="at least one ClimatologyBin"):
        RegionClimatology("r", ("bin",))


def test_boundary_values_are_accepted_exactly_at_the_edge():
    TaskSpec("a", "planning", 1e-12, 1)
    Bands(n=1, p=0.5, green=(0, 1), amber_outer=(0, 1))
    Bands(n=200, p=0.1, green=(12, 12), amber_outer=(12, 12))  # every edge may coincide
    Bands(n=200, p=0.1, green=(12, 29), amber_outer=(12, 29))
    Bands(n=200, p=0.1, green=(0, 29), amber_outer=(0, 200))
    ClimatologyBin(0.0, 1e-9, np.zeros(1), np.ones(1), 1)
    RegionCurve("r", np.array([0.0, 0.0]))  # two points, zero allowed (a distance is never negative)
    Thresholds(Bands(n=200, p=0.1, green=(12, 29), amber_outer=(8, 35)), np.array([1e-9]), 1e-9)


def test_values_are_normalised_to_plain_tuples_and_numbers_and_locked():
    inp = make_input(
        region_labels=list(judge_input_kwargs()["region_labels"]),
        divergence_curves=list(judge_input_kwargs()["divergence_curves"]),
        climatology=list(judge_input_kwargs()["climatology"]),
        tasks=list(judge_input_kwargs()["tasks"]),
    )
    for field in (inp.region_labels, inp.divergence_curves, inp.climatology, inp.tasks):
        assert type(field) is tuple
    assert type(inp.climatology[0].bins) is tuple
    bands = Bands(n=200, p=0.1, green=[12, 29], amber_outer=[8, 35])
    assert bands.green == (12, 29) and type(bands.green) is tuple and type(bands.amber_outer) is tuple
    cb = inp.climatology[0].bins[0]
    assert cb.sd.flags.writeable is False and type(cb.n_samples) is int
    reg = RegionClimatology("r", [ClimatologyBin(0.0, 1.0, np.zeros(2), np.ones(2), np.int64(7))])
    assert type(reg.bins) is tuple and type(reg.bins[0].n_samples) is int


def test_no_silent_coercion_of_the_wrong_kind_of_number():
    for green in ((12.9, 29), ("12", "29"), (True, 29), (None, 29), (float("nan"), 29), (float("inf"), 29)):
        with pytest.raises(JudgeInputError):
            Bands(n=200, p=0.1, green=green, amber_outer=(8, 35))
    for bad in (np.full((N, H), 1.9), np.full((N, H), "1"), np.full((N, H), True), np.full((N, H), np.inf)):
        with pytest.raises(JudgeInputError):
            make_input(invariant_bins=bad)
    with pytest.raises(JudgeInputError, match="too large"):
        make_input(invariant_bins=np.full((N, H), 2**63, dtype=np.uint64))
    for outcomes in (np.full((N, H, D), "1.0"), np.full((N, H, D), True), np.full((N, H, D), 1j)):
        with pytest.raises(JudgeInputError, match="numbers"):
            make_input(outcomes=outcomes)
    for lo, hi in (("0", "1"), (None, 1.0), (True, 2.0), (0.0, "1")):
        with pytest.raises(JudgeInputError, match="must be a number"):
            ClimatologyBin(lo, hi, np.zeros(2), np.ones(2), 3)
    for tol in (10**400, 10**308 * 10):
        with pytest.raises(JudgeInputError, match="too large"):
            TaskSpec("a", "control", tol, 3)
    for n in (2.5, "3", None, True):
        with pytest.raises(JudgeInputError, match="whole number"):
            ClimatologyBin(0.0, 1.0, np.zeros(2), np.ones(2), n)


def test_copying_and_pickling_go_back_through_the_checks_and_keep_the_arrays_read_only():
    import copy
    import pickle

    inp = make_input()
    for clone in (copy.deepcopy(inp), pickle.loads(pickle.dumps(inp)), copy.copy(inp)):
        assert clone is not inp
        assert clone.world == inp.world and np.array_equal(clone.outcomes, inp.outcomes)
        for arr in (clone.outcomes, clone.predictions.mean, clone.persistence.spread, clone.linear.mean,
                    clone.invariant_bins, clone.divergence_curves[0].curve, clone.thresholds.sharpness_hedge_threshold,
                    clone.climatology[0].bins[0].mean, clone.climatology[0].bins[0].sd):
            assert arr.flags.writeable is False
    for part in (inp.predictions, inp.thresholds, inp.climatology[0].bins[0], inp.divergence_curves[0]):
        clone = copy.deepcopy(part)
        assert type(clone) is type(part)


def test_equality_and_hashing_are_by_identity_so_comparing_inputs_never_crashes():
    a, b = make_input(), make_input()
    assert a != b and (a == b) is False  # equality is identity: comparing two inputs never raises
    assert len({a, b}) == 2 and len({a.predictions, b.predictions}) == 2
    assert forecasts(1) != forecasts(1)  # identical content, still not "equal": compare the arrays explicitly


def test_the_smallest_possible_input_one_trial_one_step_one_quantity_is_accepted():
    bin_ = ClimatologyBin(0.0, 1.0, np.zeros(1), np.ones(1), 1)
    inp = JudgeInput(
        world="w", dt=0.1, natural_cycle_length=None,
        predictions=forecasts(0, n=1, h=1, d=1), outcomes=np.zeros((1, 1, 1)),
        persistence=forecasts(1, n=1, h=1, d=1), linear=forecasts(2, n=1, h=1, d=1),
        region_labels=(RegionLabel("r", None),),
        divergence_curves=(RegionCurve("r", np.array([0.0, 1.0])),),
        climatology=(RegionClimatology("r", (bin_,)),),
        invariant_bins=np.zeros((1, 1), dtype=int),
        tasks=(TaskSpec("t", "control", 0.1, 1),),
        thresholds=Thresholds(Bands(n=200, p=0.1, green=(12, 29), amber_outer=(8, 35)), np.ones(1), 1.0),
    )
    assert inp.outcomes.shape == (1, 1, 1)


# --- review pass 2 ------------------------------------------------------------------------------------


def test_copies_carry_the_same_values_not_just_the_same_lock():
    import copy
    import pickle

    inp = make_input()
    for clone in (copy.deepcopy(inp), pickle.loads(pickle.dumps(inp))):
        assert np.array_equal(clone.persistence.mean, inp.persistence.mean)
        assert np.array_equal(clone.linear.mean, inp.linear.mean)
        assert not np.array_equal(clone.persistence.mean, clone.linear.mean)
        assert np.array_equal(clone.predictions.spread, inp.predictions.spread)
        assert np.array_equal(clone.predictions.mean, inp.predictions.mean)
        assert not np.array_equal(clone.predictions.mean, clone.predictions.spread)
        assert np.array_equal(clone.thresholds.sharpness_hedge_threshold, inp.thresholds.sharpness_hedge_threshold)
        assert clone.thresholds.agreement_threshold == inp.thresholds.agreement_threshold == 1.0
        assert clone.thresholds.bands == inp.thresholds.bands
        assert clone.dt == inp.dt and clone.natural_cycle_length == inp.natural_cycle_length
        assert clone.world == inp.world and clone.tasks == inp.tasks and clone.region_labels == inp.region_labels
        assert np.array_equal(clone.invariant_bins, inp.invariant_bins)
        for a, b in zip(clone.climatology[0].bins, inp.climatology[0].bins, strict=True):
            assert (a.invariant_lo, a.invariant_hi, a.n_samples) == (b.invariant_lo, b.invariant_hi, b.n_samples)
            assert np.array_equal(a.mean, b.mean) and np.array_equal(a.sd, b.sd)
        for a, b in zip(clone.divergence_curves, inp.divergence_curves, strict=True):
            assert a.region_name == b.region_name and np.array_equal(a.curve, b.curve)


@pytest.mark.parametrize("make", [
    lambda: make_input().thresholds,
    lambda: make_input().climatology[0].bins[0],
    lambda: make_input().climatology[0],
    lambda: make_input().divergence_curves[0],
])
def test_array_bearing_types_compare_by_identity_so_comparison_never_raises(make):
    a, b = make(), make()
    assert (a == b) is False and a != b


def test_value_types_without_arrays_compare_by_value():
    assert RegionLabel("r", None) == RegionLabel("r", None)
    assert TaskSpec("a", "control", 0.1, 3) == TaskSpec("a", "control", 0.1, 3)
    assert Bands(200, 0.1, (12, 29), (8, 35)) == Bands(200, 0.1, (12, 29), (8, 35))


def test_bands_p_must_be_a_plain_number_and_the_message_says_which_problem():
    for p in ("0.1", None, np.float64(0.1), [0.1]):
        with pytest.raises(JudgeInputError, match="plain number"):
            Bands(n=200, p=p, green=(12, 29), amber_outer=(8, 35))
    with pytest.raises(JudgeInputError, match="strictly between"):
        Bands(n=200, p=1.5, green=(12, 29), amber_outer=(8, 35))


def test_bad_containers_and_huge_numbers_raise_the_judges_own_error():
    for field in ("region_labels", "divergence_curves", "climatology", "tasks"):
        for bad in (None, 5):
            with pytest.raises(JudgeInputError, match="sequence"):
                make_input(**{field: bad})
    with pytest.raises(JudgeInputError, match="sequence"):
        RegionClimatology("r", None)
    with pytest.raises(JudgeInputError, match="too large"):
        ClimatologyBin(10**400, 10**400 + 1, np.zeros(2), np.ones(2), 3)


def test_climatology_bin_ends_are_normalised_to_floats():
    cb = ClimatologyBin(0, 2, np.zeros(2), np.ones(2), 3)
    assert type(cb.invariant_lo) is float and type(cb.invariant_hi) is float


def test_a_judge_input_cannot_be_subclassed_to_carry_identity():
    with pytest.raises(TypeError, match="cannot be subclassed"):
        class Tagged(JudgeInput):
            model_name: str = "x"
