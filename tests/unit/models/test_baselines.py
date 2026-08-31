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
