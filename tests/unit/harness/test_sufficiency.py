"""Tests for wmj.harness.sufficiency — the one-time, blind "is M enough?" check (ADR-M3, TC-MU5-05).

In plain words: training each practice model on 50,000 and on 100,000
examples and comparing the error on examples neither saw tells us whether
50,000 is enough. These tests pin the formula, the decision rule, that the
check only ever looks at held-out examples (never the matching margin,
evaluation trials or skill scores), and that it scores both versions on
examples neither trained on.
"""

from __future__ import annotations

import ast
import inspect

import numpy as np
import pytest

from wmj.harness import sufficiency
from wmj.harness.sufficiency import (
    SufficiencyError,
    check_world,
    decide_subsample_pairs,
    held_out_error,
    kick_split_error,
    m_is_sufficient,
)
from wmj.harness.training import TrainingRecipe, build_training_data, make_world_context
from wmj.models.base import Pairs, SeedSource
from wmj.models.direct import DirectModel, train_direct
from wmj.worlds import lv

SEED = 20260825
SCALE = np.array([2.0, 4.0])


def _pairs(next_state, kicks=None):
    n = len(next_state)
    kicks = np.zeros(n, dtype=bool) if kicks is None else np.asarray(kicks, dtype=bool)
    action = np.where(kicks[:, None], 0.05, 0.0)
    return Pairs(
        state=np.zeros((n, 2)), action=action, next_state=np.asarray(next_state, float),
        is_kick=kicks,
    )


# --- the error formula ---


def test_held_out_error_is_the_mean_over_examples_of_the_mean_over_dimensions_of_scaled_squares():
    pairs = _pairs([[2.0, 0.0], [0.0, 4.0]])
    means = np.zeros((2, 2))
    # example 0: ((2/2)^2 + 0)/2 = 0.5 ; example 1: (0 + (4/4)^2)/2 = 0.5 -> 0.5
    assert held_out_error(means, pairs, SCALE) == pytest.approx(0.5)
    means = np.array([[1.0, 0.0], [0.0, 0.0]])
    # example 0: ((1/2)^2... (2-1)/2=0.5 -> 0.25 + 0 -> 0.125 ; example 1 -> 0.5 ; mean 0.3125
    assert held_out_error(means, pairs, SCALE) == pytest.approx(0.3125)


def test_held_out_error_refuses_bad_shapes_empty_sets_and_non_finite_values():
    pairs = _pairs([[1.0, 1.0]])
    with pytest.raises(SufficiencyError, match="shape"):
        held_out_error(np.zeros((2, 2)), pairs, SCALE)
    with pytest.raises(SufficiencyError, match="finite"):
        held_out_error(np.array([[np.nan, 0.0]]), pairs, SCALE)
    empty = Pairs(np.zeros((0, 2)), np.zeros((0, 1)), np.zeros((0, 2)), np.zeros(0, dtype=bool))
    with pytest.raises(SufficiencyError, match="empty"):
        held_out_error(np.zeros((0, 2)), empty, SCALE)


# --- the kick split ---


def test_the_kick_split_reports_each_group_and_its_size():
    pairs = _pairs([[2.0, 0.0], [0.0, 4.0], [2.0, 4.0]], kicks=[True, False, False])
    split = kick_split_error(np.zeros((3, 2)), pairs, SCALE)
    assert (split.n_kick, split.n_plain) == (1, 2)
    assert split.error_kick == pytest.approx(0.5)  # (1 + 0)/2
    assert split.error_plain == pytest.approx((0.5 + 1.0) / 2)  # ((0+1)/2 + (1+1)/2)/2


def test_a_split_with_no_kicked_examples_says_so_rather_than_inventing_a_number():
    split = kick_split_error(np.zeros((2, 2)), _pairs([[1.0, 1.0], [2.0, 2.0]]), SCALE)
    assert split.error_kick is None and split.n_kick == 0 and split.error_plain is not None
    all_kick = kick_split_error(np.zeros((1, 2)), _pairs([[1.0, 1.0]], kicks=[True]), SCALE)
    assert all_kick.error_plain is None and all_kick.n_plain == 0


# --- the decision rule ---


def test_m_is_sufficient_at_exactly_the_tolerance_and_not_beyond():
    assert m_is_sufficient(1.5, 1.0, 0.5) is True  # exactly (1 + 0.5) x
    assert m_is_sufficient(np.nextafter(1.5, 2.0), 1.0, 0.5) is False
    assert m_is_sufficient(1.0, 1.0) is True and m_is_sufficient(1.11, 1.0) is False


def test_the_default_tolerance_is_the_recipes_ten_percent():
    assert sufficiency.SUFFICIENCY_TOLERANCE == 0.10
    assert m_is_sufficient(1.1, 1.0) is True  # 1.10 x within float
    assert m_is_sufficient(1.2, 1.0) is False


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1.0, True, None, "1"])
def test_m_is_sufficient_refuses_unusable_numbers(bad):
    for args in ((bad, 1.0, 0.1), (1.0, bad, 0.1), (1.0, 1.0, bad)):
        with pytest.raises(SufficiencyError):
            m_is_sufficient(*args)


def test_the_only_two_outcomes_are_m_or_the_one_fallback():
    for flags in ([True], [True, True, True], [False], [True, False], [False, False]):
        assert decide_subsample_pairs(50000, 100000, flags) == (50000 if all(flags) else 100000)
    with pytest.raises(SufficiencyError, match="every unrigged model"):
        decide_subsample_pairs(50000, 100000, [])


# --- check_world: small real runs ---


RECIPE = TrainingRecipe(
    training_trajectories=100, subsample_pairs=1000, kick_pairs=40, heldout_pairs=300,
    gradcheck_pairs=16,
)


def _quick_direct(ctx, seeds, data):
    return DirectModel(ctx, train_direct(ctx, seeds, data, epochs=2))


@pytest.fixture(scope="module")
def result():
    return check_world("lv", lv.WORLD, _quick_direct, RECIPE, SEED, "direct", horizon=100)


def test_both_versions_are_scored_on_the_larger_builds_held_out_set(result):
    from dataclasses import replace

    big = replace(RECIPE, subsample_pairs=2000)
    data_big = build_training_data("lv", lv.WORLD, SeedSource(SEED, None), big, horizon=100)
    data_small = build_training_data("lv", lv.WORLD, SeedSource(SEED, None), RECIPE, horizon=100)
    ctx = make_world_context("lv", lv.WORLD)
    held = data_big.heldout_pairs
    scale = np.asarray(lv.WORLD.scale)
    for data, expected in ((data_small, result.err_m), (data_big, result.err_2m)):
        model = _quick_direct(ctx, SeedSource(SEED, "direct"), data)
        means, _ = model.predict_batch(held.state, held.action)
        assert held_out_error(means, held, scale) == expected
    # and that held-out set shares no example with either training set
    train_big = {r.tobytes() + a.tobytes() for r, a in zip(data_big.train_pairs.state, data_big.train_pairs.action)}
    held_rows = {r.tobytes() + a.tobytes() for r, a in zip(held.state, held.action)}
    assert train_big.isdisjoint(held_rows)
    assert data_small.train_pairs.state.tobytes() == data_big.train_pairs.state[:1000].tobytes()


def test_the_result_is_consistent_with_the_rule_and_reports_the_kick_split(result):
    assert result.world == "lv"
    assert result.sufficient is m_is_sufficient(result.err_m, result.err_2m)
    assert result.split_m.n_kick + result.split_m.n_plain == RECIPE.heldout_pairs
    assert result.split_m.n_kick == result.split_2m.n_kick  # same held-out set


def test_check_world_is_deterministic():
    again = check_world("lv", lv.WORLD, _quick_direct, RECIPE, SEED, "direct", horizon=100)
    base = check_world("lv", lv.WORLD, _quick_direct, RECIPE, SEED, "direct", horizon=100)
    assert again == base


class _BiasedWorldModel:
    """Predicts the true next state plus a bias that shrinks with the training size."""

    def __init__(self, bias):
        self.bias = bias

    def predict_batch(self, states, actions):
        return lv.transition_batch(states, actions) + self.bias, np.ones_like(states)


def _biased_factory(shrinks_with_data):
    def factory(ctx, seeds, data):
        n = data.train_pairs.state.shape[0]
        return _BiasedWorldModel(1.0 / np.sqrt(n) if shrinks_with_data else 0.01)

    return factory


def test_a_model_that_keeps_improving_with_more_data_fails_the_check():
    r = check_world("lv", lv.WORLD, _biased_factory(True), RECIPE, SEED, "x", horizon=100)
    assert r.err_m / r.err_2m == pytest.approx(2.0) and r.sufficient is False


def test_a_model_that_does_not_improve_with_more_data_passes_the_check():
    r = check_world("lv", lv.WORLD, _biased_factory(False), RECIPE, SEED, "x", horizon=100)
    assert r.err_m == r.err_2m and r.sufficient is True


def test_the_model_is_given_its_own_named_seed_stream():
    seen = []

    def factory(ctx, seeds, data):
        seen.append(seeds.my_name)
        return _BiasedWorldModel(0.01)

    check_world("lv", lv.WORLD, factory, RECIPE, SEED, "direct", horizon=100)
    assert seen == ["direct", "direct"]


# --- TC-MU5-05: it never looks at the matching margin, evaluation trials or skill scores ---


def test_tc_mu5_05_no_parameter_or_identifier_can_carry_the_margin_or_evaluation_results():
    params = set(inspect.signature(check_world).parameters)
    assert not any(any(w in p for w in ("margin", "skill", "trial", "eval", "verdict")) for p in params)
    tree = ast.parse(inspect.getsource(sufficiency))
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} | {
        n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)
    }
    imported = {a.name for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom)) for a in n.names}
    modules = {getattr(n, "module", None) or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    banned = ("margin", "skill", "prereg", "verdict", "judge")
    assert not [x for x in names | imported | modules if any(b in x.lower() for b in banned)]


def test_tc_mu5_05_the_check_runs_with_the_margin_machinery_unavailable(monkeypatch):
    from wmj.harness import prereg

    def unavailable(*_a, **_k):
        raise AssertionError("the sufficiency check reached for the matching margin")

    monkeypatch.setattr(prereg, "read_matching_margin", unavailable)
    monkeypatch.setattr(prereg, "within_matching_margin", unavailable)
    monkeypatch.setattr(prereg, "check_prereg", unavailable)
    r = check_world("lv", lv.WORLD, _biased_factory(False), RECIPE, SEED, "x", horizon=100)
    assert r.sufficient is True


# --- independent review, P3-C03 pass 1 ---


def _ratio_factory(bias_small, bias_big):
    """A model whose bias depends on the training size, so err(M)/err(2M) = (bias_small/bias_big)^2."""

    def factory(ctx, seeds, data):
        small = data.train_pairs.state.shape[0] == RECIPE.subsample_pairs
        return _BiasedWorldModel(bias_small if small else bias_big)

    return factory


def test_the_tolerance_argument_decides_the_flag():
    # ratio of errors = (1.1)^2 = 1.21
    factory = _ratio_factory(0.011, 0.010)
    assert check_world("lv", lv.WORLD, factory, RECIPE, SEED, "x", tolerance=0.25, horizon=100).sufficient is True
    assert check_world("lv", lv.WORLD, factory, RECIPE, SEED, "x", tolerance=0.15, horizon=100).sufficient is False
    # and the default is the recipe's 10%: 1.21 > 1.10
    assert check_world("lv", lv.WORLD, factory, RECIPE, SEED, "x", horizon=100).sufficient is False


def test_each_kick_split_comes_from_its_own_models_predictions():
    factory = _ratio_factory(0.02, 0.01)
    r = check_world("lv", lv.WORLD, factory, RECIPE, SEED, "x", horizon=100)
    assert r.split_m.error_plain == pytest.approx(4 * r.split_2m.error_plain, rel=1e-6)
    assert r.err_m == pytest.approx(4 * r.err_2m, rel=1e-6)


def test_the_kick_split_refuses_a_shape_mismatch():
    with pytest.raises(SufficiencyError, match="shape"):
        kick_split_error(np.zeros((2, 2)), _pairs([[1.0, 1.0]]), SCALE)
