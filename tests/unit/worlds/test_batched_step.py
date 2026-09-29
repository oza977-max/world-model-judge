"""TC-WD3-04 — the batched world step is bit-identical to the single step.

In plain words: to make 2,000 training runs and 200 test runs per region
in seconds, the worlds can now advance many states at once
(`transition_batch`). That is only acceptable if it gives *exactly* the
same numbers as advancing each state on its own — not "close", the same
bits — because otherwise the training data and the truth the models are
graded against would quietly come from two slightly different physics
engines, the very trap WD-3 exists to close. Equality here is
`np.array_equal`, never a tolerance (worlds ADR-W1, design-review-010).
"""

from __future__ import annotations

import numpy as np
import pytest

from wmj.worlds import lv, pendulum
from wmj.worlds.actions import kick_sequence
from wmj.worlds.errors import (
    ActionRangeError,
    StateFloorClampError,
    WorldInputShapeError,
)
from wmj.worlds.integrator import rk4_step

WORLDS = [("lv", lv), ("pendulum", pendulum)]
BATCH_SIZES = [1, 2, 7, 64, 200]


def _starts_and_actions(module, n: int, band: str, seed: int):
    rng = np.random.default_rng(seed)
    box = module.regions().training_state_box
    states = rng.uniform(box[:, 0], box[:, 1], size=(n, box.shape[0]))
    umax = float(module.regions().training_action_interval[0, 1])
    # Kick every step so every row carries a non-null action.
    actions = kick_sequence(rng, horizon=n, p_step=1.0, band=band, umax=umax)
    return states, actions


@pytest.mark.parametrize(("name", "module"), WORLDS)
@pytest.mark.parametrize("n", BATCH_SIZES)
@pytest.mark.parametrize("band", ["in", "out"])
def test_tc_wd3_04_every_batched_row_equals_the_single_step_exactly(name, module, n, band):
    states, actions = _starts_and_actions(module, n, band, seed=1000 + n)
    batched = module.transition_batch(states, actions)
    assert batched.shape == states.shape
    for i in range(n):
        single = module.transition(states[i], actions[i])
        assert np.array_equal(batched[i], single), f"{name} row {i} of {n} ({band}) differs"


@pytest.mark.parametrize(("name", "module"), WORLDS)
def test_tc_wd3_04_multi_step_batched_rollout_equals_single_rollouts(name, module):
    # Bit-identity must survive compounding: 50 steps, 7 rows, kicked.
    rng = np.random.default_rng(77)
    box = module.regions().training_state_box
    states = rng.uniform(box[:, 0], box[:, 1], size=(7, box.shape[0]))
    umax = float(module.regions().training_action_interval[0, 1])
    kicks = np.stack(
        [kick_sequence(rng, horizon=50, p_step=0.3, band="in", umax=umax) for _ in range(7)]
    )  # [7, 50, 1]
    batch = states.copy()
    singles = [s.copy() for s in states]
    for t in range(50):
        batch = module.transition_batch(batch, kicks[:, t, :])
        singles = [module.transition(singles[i], kicks[i, t, :]) for i in range(7)]
    assert np.array_equal(batch, np.stack(singles))


@pytest.mark.parametrize(("name", "module"), WORLDS)
def test_both_paths_call_the_one_shared_integrator_with_the_world_dt(name, module, monkeypatch):
    # WD-3 / TC-WD3-01's identity half, checked by behaviour: replace the
    # module's rk4_step with a recording wrapper around the real one, then
    # step once each way. If either path stopped calling the shared
    # integrator (say, a second vectorised copy inside transition_batch),
    # its call would be missing here (independent review, P3-C09 pass 1).
    assert module.rk4_step is rk4_step
    calls = []

    def recording_rk4(deriv, state, dt):
        calls.append((deriv, np.shape(state), dt))
        return rk4_step(deriv, state, dt)

    monkeypatch.setattr(module, "rk4_step", recording_rk4)
    box = module.regions().training_state_box
    state = box.mean(axis=1)
    module.transition(state, np.array([0.0]))
    module.transition_batch(np.tile(state, (3, 1)), np.zeros((3, 1)))
    assert calls == [
        (module._deriv, state.shape, module.DT),
        (module._deriv, (3, state.shape[0]), module.DT),
    ]


@pytest.mark.parametrize(("name", "module"), WORLDS)
def test_transition_refuses_a_batched_state_instead_of_returning_a_wrong_shape(name, module):
    # design-review-010: lv.transition given a batch used to return a
    # wrong-shaped array silently.
    box = module.regions().training_state_box
    states = np.tile(box.mean(axis=1), (3, 1))
    with pytest.raises(WorldInputShapeError, match="WD-2"):
        module.transition(states, np.array([0.0]))


@pytest.mark.parametrize(("name", "module"), WORLDS)
def test_transition_refuses_a_wrong_length_state(name, module):
    with pytest.raises(WorldInputShapeError, match="WD-2"):
        module.transition(np.zeros(module.WORLD.d + 1), np.array([0.0]))


@pytest.mark.parametrize(("name", "module"), WORLDS)
def test_transition_batch_refuses_mismatched_shapes(name, module):
    box = module.regions().training_state_box
    states = np.tile(box.mean(axis=1), (4, 1))
    with pytest.raises(WorldInputShapeError, match="WD-2"):
        module.transition_batch(states, np.zeros((3, 1)))  # row count mismatch
    with pytest.raises(WorldInputShapeError, match="WD-2"):
        module.transition_batch(states[0], np.zeros((1, 1)))  # 1-D states
    with pytest.raises(WorldInputShapeError, match="WD-2"):
        module.transition_batch(states, np.zeros((4, 2)))  # wrong action width


@pytest.mark.parametrize(("name", "module"), WORLDS)
def test_transition_batch_refuses_any_row_outside_the_declared_action_range(name, module):
    box = module.regions().training_state_box
    states = np.tile(box.mean(axis=1), (5, 1))
    actions = np.zeros((5, 1))
    actions[3, 0] = module.ACTION_RANGE[1] + 0.5
    with pytest.raises(ActionRangeError, match="§7"):
        module.transition_batch(states, actions)


def test_lv_batch_with_one_row_breaching_the_floor_aborts_the_whole_call():
    # TDD-3 realistic fixture: one bad row among good ones. Worlds §7's
    # one rule — the whole call aborts, never a silently dropped row.
    states = np.array([[4.0, 2.0], [0.12, 2.0], [5.0, 3.0]])
    actions = np.array([[0.05], [-0.1], [0.0]])  # row 1: 0.12 - 0.1 = 0.02 < floor
    with pytest.raises(StateFloorClampError, match="§7"):
        lv.transition_batch(states, actions)


def test_transition_batch_does_not_mutate_its_inputs():
    states = np.array([[4.0, 2.0], [3.0, 1.5]])
    actions = np.array([[0.05], [-0.05]])
    s0, a0 = states.copy(), actions.copy()
    lv.transition_batch(states, actions)
    pendulum.transition_batch(np.zeros((2, 4)), np.array([[0.5], [-0.5]]))
    assert np.array_equal(states, s0) and np.array_equal(actions, a0)


@pytest.mark.parametrize(("name", "module"), WORLDS)
def test_world_object_exposes_transition_batch(name, module):
    states = np.tile(module.regions().training_state_box.mean(axis=1), (2, 1))
    actions = np.zeros((2, 1))
    assert np.array_equal(
        module.WORLD.transition_batch(states, actions), module.transition_batch(states, actions)
    )


@pytest.mark.parametrize(("name", "module"), WORLDS)
@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_non_finite_states_or_actions_are_refused_not_propagated(name, module, bad):
    # A NaN compares False against every bound, so without this check it
    # would pass the range and floor guards and spread silently.
    state = module.regions().training_state_box.mean(axis=1)
    with pytest.raises(WorldInputShapeError, match="finite"):
        module.transition(state, np.array([bad]))
    states = np.tile(state, (4, 1))
    states[2, 0] = bad
    with pytest.raises(WorldInputShapeError, match="row 2"):
        module.transition_batch(states, np.zeros((4, 1)))


def test_abort_messages_name_the_failing_row():
    states = np.array([[4.0, 2.0], [4.0, 2.0], [0.12, 2.0]])
    actions = np.array([[0.0], [0.0], [-0.1]])
    with pytest.raises(StateFloorClampError, match="row 2"):
        lv.transition_batch(states, actions)
    actions = np.array([[0.0], [5.0], [0.0]])
    with pytest.raises(ActionRangeError, match="row 1"):
        lv.transition_batch(np.tile([4.0, 2.0], (3, 1)), actions)


# --- independent review, P3-C09 pass 3: properties no earlier test pinned ---


def test_lv_floor_is_enforced_after_the_rk4_step_not_only_after_the_kick():
    # A null action (no impulse) from a state whose predator decays below
    # the floor during the step itself: §7's one rule must still abort.
    state = np.array([1.0, 0.0501])
    with pytest.raises(StateFloorClampError, match="RK4 step"):
        lv.transition(state, np.array([0.0]))
    batch = np.array([[4.0, 2.0], [1.0, 0.0501], [5.0, 3.0]])
    with pytest.raises(StateFloorClampError, match="RK4 step"):
        lv.transition_batch(batch, np.zeros((3, 1)))


@pytest.mark.parametrize("u", [0.07, -0.04])
def test_lv_kick_adds_u_to_prey_only(u):
    # worlds §4.1: x <- x + u, then one RK4 step of the bare equations.
    state = np.array([4.0, 2.0])
    expected = rk4_step(lv._deriv, state + np.array([u, 0.0]), lv.DT)
    assert np.array_equal(lv.transition(state, np.array([u])), expected)
    assert np.array_equal(lv.transition_batch(state[None, :], np.array([[u]]))[0], expected)


@pytest.mark.parametrize("u", [0.8, -1.5])
def test_pendulum_kick_adds_u_to_the_first_joints_angular_velocity_only(u):
    # worlds §4.2: omega1 <- omega1 + u, then one RK4 step.
    state = np.array([0.1, -0.2, 0.3, -0.4])
    expected = rk4_step(pendulum._deriv, state + np.array([0.0, 0.0, u, 0.0]), pendulum.DT)
    assert np.array_equal(pendulum.transition(state, np.array([u])), expected)
    assert np.array_equal(
        pendulum.transition_batch(state[None, :], np.array([[u]]))[0], expected
    )


@pytest.mark.parametrize(("name", "module"), WORLDS)
@pytest.mark.parametrize("action", [np.zeros(2), np.zeros((1, 1)), np.float64(0.0)])
def test_transition_refuses_a_wrongly_shaped_action(name, module, action):
    state = module.regions().training_state_box.mean(axis=1)
    with pytest.raises(WorldInputShapeError, match="WD-2"):
        module.transition(state, action)
