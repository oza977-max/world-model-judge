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
    base = {
        "state": np.zeros((5, 2)), "action": np.zeros((5, 1)), "next_state": np.ones((5, 2)),
        "is_kick": np.zeros(5, dtype=bool),
    }
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


# --- independent review, P3-C06 pass 1 (I-4): every validation rule pinned ---


def _kw(**over):
    base = {
        "state": np.zeros((5, 2)), "action": np.zeros((5, 1)), "next_state": np.ones((5, 2)),
        "is_kick": np.zeros(5, dtype=bool),
    }
    base.update(over)
    return base


@pytest.mark.parametrize("field", ["state", "action", "next_state"])
@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_non_finite_values_are_refused_in_every_pairs_array(field, bad):
    array = _kw()[field].copy()
    array[0, 0] = bad
    with pytest.raises(TrainingDataShapeError, match="finite"):
        Pairs(**_kw(**{field: array}))


@pytest.mark.parametrize("field", ["state", "action", "next_state"])
@pytest.mark.parametrize("dtype", [np.float32, np.int64])
def test_pairs_arrays_must_be_float64(field, dtype):
    with pytest.raises(TrainingDataShapeError, match="float64"):
        Pairs(**_kw(**{field: _kw()[field].astype(dtype)}))


def test_is_kick_must_agree_with_the_actions():
    with pytest.raises(TrainingDataShapeError, match="is_kick"):
        Pairs(**_kw(is_kick=np.ones(5, dtype=bool)))  # actions are all zero
    action = np.zeros((5, 1))
    action[2, 0] = 0.05
    with pytest.raises(TrainingDataShapeError, match="is_kick"):
        Pairs(**_kw(action=action))  # a kick not flagged
    flags = np.zeros(5, dtype=bool)
    flags[2] = True
    Pairs(**_kw(action=action, is_kick=flags))  # consistent: accepted


def test_pairs_row_counts_and_widths_are_each_checked():
    with pytest.raises(TrainingDataShapeError):
        Pairs(**_kw(next_state=np.ones((4, 2))))  # next_state rows
    with pytest.raises(TrainingDataShapeError):
        Pairs(**_kw(next_state=np.ones((5, 1))))  # next_state narrower
    with pytest.raises(TrainingDataShapeError):
        Pairs(**_kw(is_kick=np.zeros((5, 1), dtype=bool)))  # 2-D flags


@pytest.mark.parametrize("name", ["states", "actions"])
@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_non_finite_trajectories_are_refused(name, bad):
    states, actions = _states_actions()
    arrays = {"states": states, "actions": actions}
    arrays[name][0, 0, 0] = bad
    with pytest.raises(TrainingDataShapeError, match="finite"):
        TrainingData(**arrays)


@pytest.mark.parametrize("name", ["states", "actions"])
@pytest.mark.parametrize("dtype", [np.float32, np.int64, bool, object])
def test_trajectory_arrays_must_be_float64(name, dtype):
    states, actions = _states_actions()
    arrays = {"states": states, "actions": actions}
    arrays[name] = arrays[name].astype(dtype)
    with pytest.raises(TrainingDataShapeError, match="float64"):
        TrainingData(**arrays)


@pytest.mark.parametrize("shape", [(2, 4), (2, 3, 1, 1)])
def test_trajectory_arrays_must_be_three_dimensional(shape):
    states, _ = _states_actions()
    with pytest.raises(TrainingDataShapeError, match="3-D"):
        TrainingData(states=states, actions=np.zeros(shape))


def test_heldout_width_and_type_are_checked_too():
    states, actions = _states_actions()
    with pytest.raises(TrainingDataShapeError, match="heldout_pairs"):
        TrainingData(states=states, actions=actions, train_pairs=_pairs(5), heldout_pairs=_pairs(2, d=3))
    with pytest.raises(TrainingDataShapeError, match="Pairs"):
        TrainingData(states=states, actions=actions, train_pairs=object())


@pytest.mark.parametrize(
    "index",
    [
        np.array([0, 0], dtype=np.int64),  # duplicates
        np.array([], dtype=np.int64),  # empty
        np.array([0, 1], dtype=np.int32),  # wrong width
        np.array([0, 1], dtype=np.uint64),
    ],
)
def test_gradcheck_index_must_be_distinct_nonempty_int64(index):
    states, actions = _states_actions()
    with pytest.raises(TrainingDataShapeError, match="gradcheck"):
        TrainingData(states=states, actions=actions, train_pairs=_pairs(5), gradcheck_index=index)


# --- independent review, P3-C06 pass 2 ---


def test_pairs_action_width_must_match_the_trajectories_action_width():
    states, actions = _states_actions(d=2, a=1)
    wide = Pairs(
        state=np.zeros((3, 2)), action=np.zeros((3, 2)), next_state=np.zeros((3, 2)),
        is_kick=np.zeros(3, dtype=bool),
    )
    with pytest.raises(TrainingDataShapeError, match="action"):
        TrainingData(states=states, actions=actions, train_pairs=wide, heldout_pairs=_pairs(2))


@pytest.mark.parametrize("states_steps", [3, 5, 6])  # H is 3: states must have exactly H+1 = 4
def test_states_must_have_exactly_one_more_step_than_actions(states_steps):
    with pytest.raises(TrainingDataShapeError, match="one more step"):
        TrainingData(states=np.zeros((2, states_steps, 2)), actions=np.zeros((2, 3, 1)))


def test_a_gradcheck_batch_as_large_as_the_whole_training_set_is_legal():
    states, actions = _states_actions()
    TrainingData(
        states=states, actions=actions, train_pairs=_pairs(4),
        gradcheck_index=np.array([3, 1, 0, 2], dtype=np.int64),
    )
