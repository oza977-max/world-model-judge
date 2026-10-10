"""Tests for wmj.judge.skill.compute_skill — the verdict's `skill` block (ADR-J1, TC-JU2-01, TC-JU4-02, TC-MU6-05(b)).

In plain words: skill says how much better a model's whole forecast — its guess and its stated
uncertainty together — is than two simple reference forecasts ("nothing changes" and "straight
line"). These tests compute the expected numbers by hand from the closed-form score, check that
only the first step ahead and only the right region's trials are used, that the block has exactly
the fields the spec names, and — the property the whole judge leans on — that stating a false
confidence can never beat stating the true one.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tests.unit.judge._builders import (
    D,
    H,
    forecasts,
    judge_input_kwargs,
    make_input,
)
from wmj.judge.errors import JudgeInputError
from wmj.judge.skill import compute_skill, crps_gaussian, skill_score
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
from wmj.judge.verdict import _check_block

SQRT_PI = math.sqrt(math.pi)


def _phi(z):
    return math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)


def _Phi(z):
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _crps(mu, sigma, y):
    """The closed form written out independently of the module under test (scalar maths only)."""
    z = (y - mu) / sigma
    return sigma * (z * (2.0 * _Phi(z) - 1.0) + 2.0 * _phi(z) - 1.0 / SQRT_PI)


def _expected_region_crps(fc: Forecasts, outcomes, rows):
    values = []
    for i in rows:
        per_dim = [_crps(fc.mean[i, 0, j], fc.spread[i, 0, j], outcomes[i, 0, j]) for j in range(outcomes.shape[2])]
        values.append(sum(per_dim) / len(per_dim))
    return sum(values) / len(values)


def test_the_block_matches_a_hand_computed_closed_form_for_every_region():
    inp = make_input()
    block = compute_skill(inp)
    regions = {"training": [0, 1, 2], "out-of-range": [3, 4, 5]}
    for entry in block["per_task_region"]:
        rows = regions[entry["region"]]
        model = _expected_region_crps(inp.predictions, inp.outcomes, rows)
        pers = _expected_region_crps(inp.persistence, inp.outcomes, rows)
        lin = _expected_region_crps(inp.linear, inp.outcomes, rows)
        assert entry["crps"] == pytest.approx(model, rel=1e-12)
        assert entry["vs_persistence"] == pytest.approx(1.0 - model / pers, rel=1e-12)
        assert entry["vs_linear"] == pytest.approx(1.0 - model / lin, rel=1e-12)


def test_the_block_has_exactly_the_spec_fields_one_entry_per_task_and_region_in_a_fixed_order():
    block = compute_skill(make_input())
    assert list(block) == ["per_task_region"]
    entries = block["per_task_region"]
    assert [(e["task"], e["region"]) for e in entries] == [
        ("lv-control", "out-of-range"), ("lv-control", "training"),
        ("lv-planning", "out-of-range"), ("lv-planning", "training"),
    ]
    for e in entries:
        assert set(e) == {"task", "region", "vs_persistence", "vs_linear", "crps"}  # TC-MU6-05(b): two comparators, no more
        assert all(type(e[k]) is float for k in ("vs_persistence", "vs_linear", "crps"))


def test_skill_is_one_step_so_every_task_of_a_region_gets_the_same_numbers():
    entries = compute_skill(make_input())["per_task_region"]
    for region in ("training", "out-of-range"):
        same = [e for e in entries if e["region"] == region]
        assert len(same) == 2
        assert {(e["vs_persistence"], e["vs_linear"], e["crps"]) for e in same} == {
            (same[0]["vs_persistence"], same[0]["vs_linear"], same[0]["crps"])}


def test_only_the_first_step_ahead_is_used_later_steps_cannot_move_the_score():
    kwargs = judge_input_kwargs()
    before = compute_skill(JudgeInput(**kwargs))
    rng = np.random.default_rng(5)
    changed = dict(kwargs)
    changed["outcomes"] = np.array(kwargs["outcomes"])
    changed["outcomes"][:, 1:, :] = rng.normal(size=changed["outcomes"][:, 1:, :].shape) * 50
    for key in ("predictions", "persistence", "linear"):
        fc = kwargs[key]
        mean, spread = np.array(fc.mean), np.array(fc.spread)
        mean[:, 1:, :] += 100.0
        spread[:, 1:, :] *= 7.0
        changed[key] = Forecasts(mean, spread)
    assert compute_skill(JudgeInput(**changed)) == before
    # ...and the first step does move it
    moved = dict(kwargs)
    moved["outcomes"] = np.array(kwargs["outcomes"])
    moved["outcomes"][:, 0, :] += 3.0
    assert compute_skill(JudgeInput(**moved)) != before


def test_each_region_is_scored_only_on_its_own_trials():
    kwargs = judge_input_kwargs()
    base = {(e["task"], e["region"]): e for e in compute_skill(JudgeInput(**kwargs))["per_task_region"]}
    changed = dict(kwargs)
    changed["outcomes"] = np.array(kwargs["outcomes"])
    changed["outcomes"][3:, 0, :] += 5.0  # only the out-of-range trials
    after = {(e["task"], e["region"]): e for e in compute_skill(JudgeInput(**changed))["per_task_region"]}
    assert after[("lv-control", "training")] == base[("lv-control", "training")]
    assert after[("lv-control", "out-of-range")] != base[("lv-control", "out-of-range")]


def test_a_model_that_is_the_persistence_baseline_has_exactly_zero_skill_against_it():
    kwargs = judge_input_kwargs()
    kwargs["predictions"] = kwargs["persistence"]
    for e in compute_skill(JudgeInput(**kwargs))["per_task_region"]:
        assert e["vs_persistence"] == 0.0 and e["vs_persistence"] != e["vs_linear"]


def test_a_near_perfect_confident_forecast_scores_near_one_and_a_terrible_one_scores_below_zero():
    kwargs = judge_input_kwargs()
    outcomes = np.array(kwargs["outcomes"])
    perfect = Forecasts(outcomes.copy(), np.full_like(outcomes, 1e-4))
    kwargs["predictions"] = perfect
    for e in compute_skill(JudgeInput(**kwargs))["per_task_region"]:
        assert e["vs_persistence"] > 0.999 and e["vs_linear"] > 0.999
    kwargs["predictions"] = Forecasts(outcomes + 50.0, np.full_like(outcomes, 0.1))
    for e in compute_skill(JudgeInput(**kwargs))["per_task_region"]:
        assert e["vs_persistence"] < 0.0 and e["vs_linear"] < 0.0


def test_the_output_passes_the_verdict_doors_own_skill_check():
    block = compute_skill(make_input())
    checked = _check_block("skill", block)
    assert checked == block


def test_tc_mu6_05_b_the_judge_cannot_be_given_or_name_a_third_comparator():
    with pytest.raises(TypeError):
        JudgeInput(**{**judge_input_kwargs(), "vs_newbaseline": forecasts(9)})
    names = {f for e in compute_skill(make_input())["per_task_region"] for f in e}
    assert not any(n.startswith("vs_") and n not in ("vs_persistence", "vs_linear") for n in names)


def test_tc_ju2_01_no_raw_absolute_error_comes_without_both_skills():
    for e in compute_skill(make_input())["per_task_region"]:
        assert "vs_persistence" in e and "vs_linear" in e
        assert not any("error" in k or "rmse" in k or "mae" in k for k in e)


def test_compute_skill_takes_only_a_real_judge_input():
    class Lookalike:
        pass

    for bad in (None, {}, Lookalike(), "input"):
        with pytest.raises(JudgeInputError, match="JudgeInput"):
            compute_skill(bad)


# --- TC-JU4-02: the score cannot be gamed (strictly proper), as a property over many distributions -----


@pytest.mark.parametrize("factor", [0.25, 0.5, 0.8, 1.25, 2.0, 4.0])
def test_tc_ju4_02_stating_the_true_spread_never_scores_worse_than_a_misstated_one(factor):
    rng = np.random.default_rng(20260825)
    for _ in range(40):
        mu = rng.normal(scale=3.0)
        sigma = float(np.exp(rng.normal(scale=1.0)))
        y = rng.normal(mu, sigma, size=40_000)
        true_score = float(np.mean(crps_gaussian(np.full_like(y, mu), np.full_like(y, sigma), y)))
        false_score = float(np.mean(crps_gaussian(np.full_like(y, mu), np.full_like(y, sigma * factor), y)))
        assert true_score < false_score


def test_tc_ju4_02_a_wrong_centre_cannot_be_rescued_by_hedging_and_overconfidence_is_punished_in_skill():
    rng = np.random.default_rng(3)
    n, steps, d = 200, 2, 2
    truth = rng.normal(size=(n, steps, d))
    honest = Forecasts(np.zeros((n, steps, d)), np.ones((n, steps, d)))  # N(0, 1) is exactly the truth's law
    cocky = Forecasts(np.zeros((n, steps, d)), np.full((n, steps, d), 0.25))  # same centre, a quarter the width
    base = Forecasts(np.zeros((n, steps, d)) + 0.5, np.ones((n, steps, d)))

    def skill_of(model):
        labels = tuple(RegionLabel("r", None) for _ in range(n))
        from wmj.judge.types import (
            ClimatologyBin,
            RegionClimatology,
            RegionCurve,
            TaskSpec,
        )

        inp = JudgeInput(
            world="w", dt=0.1, natural_cycle_length=None, predictions=model, outcomes=truth,
            persistence=base, linear=base, region_labels=labels,
            divergence_curves=(RegionCurve("r", np.linspace(0, 1, steps + 1)),),
            climatology=(RegionClimatology("r", (ClimatologyBin(0.0, 1.0, np.zeros(d), np.ones(d), 5),)),),
            invariant_bins=np.zeros((n, steps), dtype=int),
            tasks=(TaskSpec("t", "control", 0.1, steps),),
            thresholds=Thresholds(Bands(n=n, p=0.1, green=(0, 40), amber_outer=(0, 60)), np.ones(d), 1.0),
        )
        return compute_skill(inp)["per_task_region"][0]["vs_persistence"]

    assert skill_of(honest) > skill_of(cocky)
    assert skill_of(honest) > 0.0


def test_the_scalar_helpers_still_agree_with_the_independent_closed_form():
    for mu, sigma, y in ((0.0, 1.0, 0.3), (2.0, 0.1, 2.5), (-1.0, 5.0, 4.0)):
        assert float(crps_gaussian(np.array([mu]), np.array([sigma]), np.array([y]))[0]) == pytest.approx(_crps(mu, sigma, y), rel=1e-12)
    assert skill_score(0.5, 1.0) == 0.5


# --- review pass 1 --------------------------------------------------------------------------------------


def test_a_crps_that_overflows_is_a_loud_judge_error_never_a_skill_of_one():
    from wmj.judge.errors import JudgeError
    from wmj.judge.skill import NonFiniteScoreError

    kwargs = judge_input_kwargs()
    n = len(kwargs["region_labels"])
    tiny = np.full((n, H, D), 1e-320)  # a legal spread (> 0) so small that (outcome - mean) / spread overflows
    for field in ("persistence", "linear"):
        broken = dict(kwargs)
        broken[field] = Forecasts(np.array(kwargs[field].mean), tiny)
        with pytest.raises(NonFiniteScoreError) as err:
            compute_skill(JudgeInput(**broken))
        assert isinstance(err.value, JudgeError)
    broken = dict(kwargs)
    broken["predictions"] = Forecasts(np.array(kwargs["predictions"].mean), tiny)
    with pytest.raises(NonFiniteScoreError):
        compute_skill(JudgeInput(**broken))


def test_the_scalar_functions_refuse_non_finite_scores_and_their_errors_are_judge_errors():
    from wmj.judge.errors import JudgeError
    from wmj.judge.skill import (
        NonFiniteScoreError,
        NonPositiveBaselineError,
        NonPositiveSpreadError,
    )

    for bad in (float("inf"), float("nan")):
        with pytest.raises(NonFiniteScoreError):
            skill_score(bad, 1.0)
        with pytest.raises(NonFiniteScoreError):
            skill_score(1.0, bad)
    with pytest.raises(NonPositiveBaselineError):
        skill_score(0.5, 0.0)
    with pytest.raises(NonPositiveSpreadError):
        crps_gaussian(np.array([0.0]), np.array([0.0]), np.array([1.0]))
    for cls in (NonFiniteScoreError, NonPositiveBaselineError, NonPositiveSpreadError):
        assert issubclass(cls, JudgeError) and issubclass(cls, ValueError)


def test_crps_accepts_scalar_and_zero_dimensional_input():
    assert float(crps_gaussian(np.array(0.0), np.array(1.0), np.array(0.3))) == pytest.approx(_crps(0.0, 1.0, 0.3), rel=1e-12)
    assert float(crps_gaussian(0.0, 1.0, 0.3)) == pytest.approx(_crps(0.0, 1.0, 0.3), rel=1e-12)


def test_extreme_but_representable_cases_stay_finite_and_positive():
    for z in (-40.0, -8.0, 0.0, 1e-3, 8.0, 40.0):
        for sigma in (1e-12, 1e-3, 1.0, 1e3, 1e12):
            value = float(crps_gaussian(np.array([0.0]), np.array([sigma]), np.array([z * sigma]))[0])
            assert math.isfinite(value) and value > 0.0
            assert value == pytest.approx(_crps(0.0, sigma, z * sigma), rel=1e-9)


def test_region_names_that_are_prefixes_of_each_other_stay_separate():
    from wmj.judge.types import (
        Bands,
        ClimatologyBin,
        RegionClimatology,
        RegionCurve,
        Thresholds,
    )

    n_each, names = 2, ["a", "ab", "abc"]
    n = n_each * len(names)
    rng = np.random.default_rng(1)
    labels = tuple(RegionLabel(name, None) for name in names for _ in range(n_each))
    mean, spread = rng.normal(size=(n, 1, 1)), rng.uniform(0.5, 1.5, size=(n, 1, 1))
    inp = JudgeInput(
        world="w", dt=0.1, natural_cycle_length=None,
        predictions=Forecasts(mean, spread), outcomes=rng.normal(size=(n, 1, 1)),
        persistence=Forecasts(mean * 0, np.ones_like(spread)), linear=Forecasts(mean * 0 + 0.3, np.ones_like(spread)),
        region_labels=labels,
        divergence_curves=tuple(RegionCurve(name, np.array([0.0, 1.0])) for name in names),
        climatology=tuple(RegionClimatology(name, (ClimatologyBin(0.0, 1.0, np.zeros(1), np.ones(1), 5),)) for name in names),
        invariant_bins=np.zeros((n, 1), dtype=int), tasks=(TaskSpec("t", "control", 0.1, 1),),
        thresholds=Thresholds(Bands(n=n_each, p=0.1, green=(0, 1), amber_outer=(0, n_each)), np.ones(1), 1.0),
    )
    block = compute_skill(inp)["per_task_region"]
    assert [e["region"] for e in block] == names
    for i, name in enumerate(names):
        rows = [2 * i, 2 * i + 1]
        expected = np.mean([_crps(mean[r, 0, 0], spread[r, 0, 0], inp.outcomes[r, 0, 0]) for r in rows])
        assert block[i]["crps"] == pytest.approx(expected, rel=1e-12)


def test_a_duck_typed_lookalike_with_the_right_attributes_is_still_refused():
    import types as pytypes

    real = make_input()
    lookalike = pytypes.SimpleNamespace(**{f: getattr(real, f) for f in
        ("world", "predictions", "outcomes", "persistence", "linear", "region_labels", "tasks", "divergence_curves")})
    with pytest.raises(JudgeInputError, match="JudgeInput"):
        compute_skill(lookalike)


@pytest.mark.parametrize("factor", [0.9, 0.95, 1.05, 1.1])
def test_tc_ju4_02_even_small_misstatements_of_the_spread_score_worse(factor):
    rng = np.random.default_rng(99)
    worse = 0
    for _ in range(25):
        mu, sigma = rng.normal(scale=2.0), float(np.exp(rng.normal(scale=0.7)))
        y = rng.normal(mu, sigma, size=400_000)
        true_score = float(np.mean(crps_gaussian(np.full_like(y, mu), np.full_like(y, sigma), y)))
        false_score = float(np.mean(crps_gaussian(np.full_like(y, mu), np.full_like(y, sigma * factor), y)))
        worse += true_score < false_score
    assert worse == 25


# --- review pass 2 --------------------------------------------------------------------------------------


def test_crps_gaussian_itself_refuses_an_overflow_and_leaks_no_warning():
    import warnings

    from wmj.judge.skill import NonFiniteScoreError

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # a leaked RuntimeWarning would surface as a raw error, not the judge's refusal
        for mean, spread, outcome in ((0.0, 1e-320, 1.0), (-1.5e308, 1.0, 1.5e308)):
            with pytest.raises(NonFiniteScoreError):
                crps_gaussian(np.array([mean]), np.array([spread]), np.array([outcome]))


def test_skill_score_refuses_a_result_that_is_not_a_finite_number():
    from wmj.judge.skill import NonFiniteScoreError

    with pytest.raises(NonFiniteScoreError):
        skill_score(1e100, 1e-300)  # both inputs finite, the ratio is not
    assert skill_score(1e-300, 1e100) == 1.0  # a vanishing ratio is a plain, finite skill of 1


def test_quantities_are_averaged_by_the_mean_not_the_median_checked_with_three_quantities():
    n, d = 3, 3
    outcome = np.zeros((n, 1, d))
    outcome[:, 0, :] = [0.0, 1.0, 5.0]  # the three quantities score very differently: mean != median
    flat = lambda spread: Forecasts(np.zeros((n, 1, d)), np.full((n, 1, d), spread))
    bins = (ClimatologyBin(-np.inf, np.inf, np.zeros(d), np.ones(d), 5),)
    inp = JudgeInput(
        world="w", dt=0.1, natural_cycle_length=None,
        predictions=flat(1.0), outcomes=outcome, persistence=flat(2.0), linear=flat(3.0),
        region_labels=tuple(RegionLabel("r", None) for _ in range(n)),
        divergence_curves=(RegionCurve("r", np.array([0.0, 1.0])),),
        climatology=(RegionClimatology("r", bins),),
        invariant_bins=np.zeros((n, 1), dtype=int), tasks=(TaskSpec("t", "control", 0.1, 1),),
        thresholds=Thresholds(Bands(n=3, p=0.1, green=(1, 1), amber_outer=(0, 2)), np.ones(d), 1.0),
    )
    entry = compute_skill(inp)["per_task_region"][0]
    per_quantity = lambda sigma: [_crps(0.0, sigma, y) for y in (0.0, 1.0, 5.0)]
    mean = lambda values: sum(values) / len(values)
    assert entry["crps"] == pytest.approx(mean(per_quantity(1.0)), rel=1e-12)
    assert entry["vs_persistence"] == pytest.approx(1.0 - mean(per_quantity(1.0)) / mean(per_quantity(2.0)), rel=1e-12)
    assert entry["vs_linear"] == pytest.approx(1.0 - mean(per_quantity(1.0)) / mean(per_quantity(3.0)), rel=1e-12)
