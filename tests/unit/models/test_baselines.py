"""Tests for wmj.models.baselines — persistence and linear (models ADR-M2).

Persistence: mean = current state; spread = std of one-step training
changes. Linear: mean = current + (current - previous), falling back
to persistence on the first predict() of a rollout (no previous yet);
spread = std of that rule's own training residuals.
"""

from __future__ import annotations

import numpy as np
import pytest

from wmj.models.baselines import linear_factory, persistence_factory
from wmj.models.base import SeedSource, TrainingData, WorldContext


def _training_data() -> TrainingData:
    rng = np.random.default_rng(20260825)
    n, h, d, a = 4, 20, 2, 1
    states = np.zeros((n, h + 1, d))
    states[:, 0, :] = rng.uniform(3.0, 5.0, size=(n, d))
    for t in range(h):
        states[:, t + 1, :] = states[:, t, :] + rng.normal(0.0, 0.05, size=(n, d))
    actions = rng.uniform(-0.2, 0.2, size=(n, h, a))
    return TrainingData(states=states, actions=actions)


def _ctx() -> WorldContext:
    return WorldContext(
        world_name="lv",
        state_dim=2,
        action_dim=1,
        training_state_box=np.array([[2.0, 6.0], [1.0, 4.0]]),
        training_action_interval=np.array([[-0.5, 0.5]]),
        scale=np.array([4.0, 2.5]),
    )


def _seeds() -> SeedSource:
    return SeedSource(run_seed=20260825, my_name=None)


def test_persistence_predicts_current_state_unchanged():
    model = persistence_factory(_ctx(), _seeds(), _training_data())
    model.reset()

    state = np.array([4.2, 2.1])
    prediction = model.predict(state, np.array([0.0]))

    assert np.array_equal(prediction.mean, state)
    assert np.all(prediction.spread > 0.0)


def test_persistence_spread_is_a_fitted_constant_across_calls():
    model = persistence_factory(_ctx(), _seeds(), _training_data())
    model.reset()
    first = model.predict(np.array([4.0, 2.0]), np.array([0.0]))
    second = model.predict(np.array([5.0, 3.0]), np.array([0.0]))
    assert np.array_equal(first.spread, second.spread)


def test_linear_falls_back_to_persistence_on_first_call_after_reset():
    model = linear_factory(_ctx(), _seeds(), _training_data())
    model.reset()

    state = np.array([4.2, 2.1])
    prediction = model.predict(state, np.array([0.0]))

    assert np.array_equal(prediction.mean, state)


def test_linear_extrapolates_on_second_call():
    model = linear_factory(_ctx(), _seeds(), _training_data())
    model.reset()

    model.predict(np.array([4.0, 2.0]), np.array([0.0]))
    second = model.predict(np.array([4.2, 2.1]), np.array([0.0]))

    expected_mean = np.array([4.2, 2.1]) + (
        np.array([4.2, 2.1]) - np.array([4.0, 2.0])
    )
    assert second.mean == pytest.approx(expected_mean)


def test_linear_reset_clears_rollout_local_memory():
    """TC-MU1-03: reset() actually clears rollout-local state."""
    model = linear_factory(_ctx(), _seeds(), _training_data())
    model.reset()
    model.predict(np.array([4.0, 2.0]), np.array([0.0]))

    model.reset()
    state = np.array([9.0, 9.0])
    prediction = model.predict(state, np.array([0.0]))

    # if reset() had not cleared the previous state, this would
    # extrapolate from the stale (4.0, 2.0) instead of falling back
    assert np.array_equal(prediction.mean, state)


def test_both_baselines_are_flagged_as_baselines_not_fixtures():
    persistence = persistence_factory(_ctx(), _seeds(), _training_data())
    linear = linear_factory(_ctx(), _seeds(), _training_data())

    for model in (persistence, linear):
        assert model.is_baseline is True
        assert model.is_fixture is False


def test_tc_mu2_02_baselines_produce_different_predictions_from_each_other():
    """Supporting check for TC-MU2-02: the two baselines are genuinely
    different rules, not accidental duplicates of one another."""
    persistence = persistence_factory(_ctx(), _seeds(), _training_data())
    linear = linear_factory(_ctx(), _seeds(), _training_data())
    persistence.reset()
    linear.reset()

    persistence.predict(np.array([4.0, 2.0]), np.array([0.0]))
    linear.predict(np.array([4.0, 2.0]), np.array([0.0]))

    persistence_pred = persistence.predict(np.array([4.3, 2.2]), np.array([0.0]))
    linear_pred = linear.predict(np.array([4.3, 2.2]), np.array([0.0]))

    assert not np.array_equal(persistence_pred.mean, linear_pred.mean)


# --- P2-C05: persistence spread fit made explicit (ADR-M2, backlog A8) ---


def test_fit_persistence_spread_is_sample_std_of_one_step_changes():
    from wmj.models.baselines import fit_persistence_spread

    training = _training_data()
    changes = training.states[:, 1:, :] - training.states[:, :-1, :]
    expected = np.std(changes.reshape(-1, 2), axis=0, ddof=1)
    assert np.array_equal(fit_persistence_spread(training), expected)


def test_persistence_factory_uses_the_public_fit():
    from wmj.models.baselines import fit_persistence_spread

    model = persistence_factory(_ctx(), _seeds(), _training_data())
    prediction = model.predict(np.array([4.0, 2.0]), np.array([0.0]))
    assert np.array_equal(prediction.spread, fit_persistence_spread(_training_data()))


def test_tc_mu2_03_fit_persistence_spread_refuses_a_zero_variance_dimension():
    from wmj.models.baselines import DegenerateSpreadError, fit_persistence_spread

    states = np.zeros((3, 6, 2))
    states[:, :, 0] = np.arange(6) * 0.25  # exactly constant change -> std exactly 0
    states[:, :, 1] = np.random.default_rng(1).normal(size=(3, 6))
    with pytest.raises(DegenerateSpreadError):
        fit_persistence_spread(TrainingData(states=states, actions=np.zeros((3, 5, 1))))


def test_tc_mu2_03_fit_persistence_spread_refuses_too_few_changes_for_a_sample_std():
    from wmj.models.baselines import DegenerateSpreadError, fit_persistence_spread

    states = np.random.default_rng(2).normal(size=(1, 2, 2))  # one change only
    with pytest.raises(DegenerateSpreadError):
        fit_persistence_spread(TrainingData(states=states, actions=np.zeros((1, 1, 1))))


# --- design-review-009 C1: linear's spread fit now matches persistence's ---


def test_fit_linear_spread_is_sample_std_of_its_own_residuals():
    from wmj.models.baselines import fit_linear_spread

    training = _training_data()
    previous = training.states[:, :-2, :]
    current = training.states[:, 1:-1, :]
    actual_next = training.states[:, 2:, :]
    residual = actual_next - (current + (current - previous))
    expected = np.std(residual.reshape(-1, 2), axis=0, ddof=1)
    assert np.array_equal(fit_linear_spread(training), expected)


def test_linear_factory_uses_the_public_fit():
    from wmj.models.baselines import fit_linear_spread

    model = linear_factory(_ctx(), _seeds(), _training_data())
    model.reset()
    model.predict(np.array([4.0, 2.0]), np.array([0.0]))  # first call: persistence fallback
    prediction = model.predict(np.array([4.2, 2.1]), np.array([0.0]))
    assert np.array_equal(prediction.spread, fit_linear_spread(_training_data()))


def test_tc_mu2_03_fit_linear_spread_refuses_a_zero_variance_dimension():
    from wmj.models.baselines import DegenerateSpreadError, fit_linear_spread

    states = np.zeros((3, 7, 2))
    states[:, :, 0] = np.arange(7) * 0.25  # perfectly linear -> zero residual std
    states[:, :, 1] = np.random.default_rng(1).normal(size=(3, 7))
    with pytest.raises(DegenerateSpreadError):
        fit_linear_spread(TrainingData(states=states, actions=np.zeros((3, 6, 1))))


def test_tc_mu2_03_fit_linear_spread_refuses_too_few_residuals_for_a_sample_std():
    from wmj.models.baselines import DegenerateSpreadError, fit_linear_spread

    states = np.random.default_rng(2).normal(size=(1, 3, 2))  # one residual only
    with pytest.raises(DegenerateSpreadError):
        fit_linear_spread(TrainingData(states=states, actions=np.zeros((1, 2, 1))))


def test_tc_mu2_03_persistence_and_linear_spread_fits_both_refuse_the_same_degenerate_input():
    """TC-MU2-03 (negative/phantom-gate): the two baselines must not
    silently diverge on the fail-loud guard (design-review-009 C1 — the
    exact defect this test locks down: persistence was guarded, linear
    was not)."""
    from wmj.models.baselines import DegenerateSpreadError, fit_linear_spread, fit_persistence_spread

    degenerate = TrainingData(
        states=np.zeros((3, 6, 2)) + np.arange(6).reshape(1, 6, 1) * np.array([0.25, 0.0]),
        actions=np.zeros((3, 5, 1)),
    )
    with pytest.raises(DegenerateSpreadError):
        fit_persistence_spread(degenerate)
    with pytest.raises(DegenerateSpreadError):
        fit_linear_spread(degenerate)
