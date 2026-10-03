"""Tests for `Pairs` and the extended `TrainingData` (models ADR-M1, design-review-010).

In plain words: the homework handed to every model is a bundle of states and
pushes plus a pre-picked training set, a held-back set and a few examples for
a sanity check. These tests make sure the bundle refuses to exist in a
broken shape, and that nobody can change it after it is handed out.
"""

from __future__ import annotations

import numpy as np
import pytest

from wmj.models.base import Pairs, TrainingData, TrainingDataShapeError


def _pairs(m: int = 5, d: int = 2, a: int = 1) -> Pairs:
    return Pairs(
        state=np.zeros((m, d)),
        action=np.zeros((m, a)),
        next_state=np.ones((m, d)),
        is_kick=np.zeros(m, dtype=bool),
    )


def _states_actions(n=2, h=3, d=2, a=1):
    return np.zeros((n, h + 1, d)), np.zeros((n, h, a))


def test_training_data_without_pairs_still_builds_for_the_older_callers():
    states, actions = _states_actions()
    data = TrainingData(states=states, actions=actions)
    assert data.train_pairs is None and data.heldout_pairs is None and data.gradcheck_index is None


def test_pairs_are_read_only():
    pairs = _pairs()
    for name in ("state", "action", "next_state", "is_kick"):
        with pytest.raises(ValueError):
            getattr(pairs, name)[0] = 1


def test_training_data_arrays_are_read_only():
    states, actions = _states_actions()
    data = TrainingData(
        states=states, actions=actions, train_pairs=_pairs(), heldout_pairs=_pairs(),
        gradcheck_index=np.array([0, 1], dtype=np.int64),
    )
    for array in (data.states, data.actions, data.gradcheck_index):
        with pytest.raises(ValueError):
            array[0] = 1


@pytest.mark.parametrize(
    "kwargs",
    [
        {"action": np.zeros((4, 1))},  # row count differs
        {"next_state": np.ones((5, 3))},  # state width differs
        {"is_kick": np.zeros(4, dtype=bool)},
        {"is_kick": np.zeros(5)},  # not boolean
        {"state": np.zeros(5)},  # not 2-D
    ],
)
def test_pairs_with_inconsistent_shapes_are_refused(kwargs):
    base = dict(
        state=np.zeros((5, 2)), action=np.zeros((5, 1)), next_state=np.ones((5, 2)),
        is_kick=np.zeros(5, dtype=bool),
    )
    base.update(kwargs)
    with pytest.raises(TrainingDataShapeError):
        Pairs(**base)


def test_non_finite_pair_values_are_refused():
    with pytest.raises(TrainingDataShapeError, match="finite"):
        Pairs(
            state=np.array([[np.nan, 0.0]]), action=np.zeros((1, 1)),
            next_state=np.zeros((1, 2)), is_kick=np.zeros(1, dtype=bool),
        )


@pytest.mark.parametrize(
    "index",
    [
        np.array([0, 5], dtype=np.int64),  # out of range (5 rows)
        np.array([-1, 0], dtype=np.int64),
        np.array([0.0, 1.0]),  # not an integer array
        np.array([[0, 1]], dtype=np.int64),  # not 1-D
    ],
)
def test_a_bad_gradcheck_index_is_refused(index):
    states, actions = _states_actions()
    with pytest.raises(TrainingDataShapeError, match="gradcheck"):
        TrainingData(
            states=states, actions=actions, train_pairs=_pairs(5), heldout_pairs=_pairs(2),
            gradcheck_index=index,
        )


def test_a_gradcheck_index_without_train_pairs_is_refused():
    states, actions = _states_actions()
    with pytest.raises(TrainingDataShapeError, match="gradcheck"):
        TrainingData(states=states, actions=actions, gradcheck_index=np.array([0], dtype=np.int64))


def test_trajectory_arrays_with_mismatched_lengths_are_refused():
    with pytest.raises(TrainingDataShapeError, match="actions"):
        TrainingData(states=np.zeros((2, 4, 2)), actions=np.zeros((2, 4, 1)))  # needs H=3
    with pytest.raises(TrainingDataShapeError, match="trajector"):
        TrainingData(states=np.zeros((2, 4, 2)), actions=np.zeros((3, 3, 1)))


def test_pairs_and_trajectories_must_agree_on_state_and_action_width():
    states, actions = _states_actions(d=2, a=1)
    with pytest.raises(TrainingDataShapeError, match="width"):
        TrainingData(states=states, actions=actions, train_pairs=_pairs(5, d=3), heldout_pairs=_pairs(2))
