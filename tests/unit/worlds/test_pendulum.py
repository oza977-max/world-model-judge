"""Tests for wmj.worlds.pendulum — the double pendulum world (worlds §4.2).

TC-WD1-01: the no-action transition matches a hand-verified reference
value, computed by an independent plain-Python (no NumPy) RK4
implementation of the design-review-002-corrected equations of motion
— worlds.md's own §8 pinned value is stale (computed from the
pre-correction EOM) and is not used; this chunk computes a fresh one,
same discipline as P1-C02 did for LV.

TC-WD2-01: a non-null action changes the outcome.
"""

from __future__ import annotations

import numpy as np
import pytest

from wmj.worlds import pendulum


def test_tc_wd1_01_transition_matches_hand_verified_reference_no_action():
    # Independently computed via a plain-Python (no NumPy) RK4 of the
    # design-review-002-corrected EOM from (0.1, 0.1, 0.0, 0.0),
    # dt=0.002 -- see build/prompts/P2-C02.md for the derivation notes
    # and why (0.1, 0.1, 0, 0) is a genuinely non-degenerate point
    # (theta1_ddot != 0 there, even though delta=theta1-theta2=0).
    state = np.array([0.1, 0.1, 0.0, 0.0])
    action = np.array([0.0])

    next_state = pendulum.transition(state, action)

    expected = np.array(
        [0.0999980412811115, 0.0999999999872539, -0.0019587061423893717, -2.5492225466702998e-08]
    )
    assert next_state == pytest.approx(expected, rel=1e-9, abs=1e-12)


def test_tc_wd2_01_nonnull_action_changes_the_outcome():
    state = np.array([0.1, 0.1, 0.0, 0.0])

    no_action = pendulum.transition(state, np.array([0.0]))
    with_action = pendulum.transition(state, np.array([0.5]))

    assert not np.allclose(no_action, with_action)


def test_action_outside_declared_range_raises_action_range_error():
    state = np.array([0.1, 0.1, 0.0, 0.0])
    with pytest.raises(pendulum.ActionRangeError):
        pendulum.transition(state, np.array([3.0]))


def test_conserved_quantity_is_a_scalar():
    state = np.array([0.1, 0.1, 0.0, 0.0])
    value = pendulum.conserved(state)
    assert isinstance(value, float)


def test_energy_is_conserved_to_rk4_local_error_over_one_step():
    state = np.array([0.1, 0.1, 0.0, 0.0])
    next_state = pendulum.transition(state, np.array([0.0]))
    e0 = pendulum.conserved(state)
    e1 = pendulum.conserved(next_state)
    assert e1 == pytest.approx(e0, rel=1e-9)


def test_world_declares_dimensionality_and_scale():
    assert pendulum.WORLD.d == 4
    assert pendulum.WORLD.a == 1
    assert pendulum.WORLD.dt == pytest.approx(0.002)
    assert pendulum.WORLD.scale.shape == (4,)
    assert pendulum.WORLD.scale == pytest.approx(
        [np.pi, np.pi, 2 * np.pi, 2 * np.pi]
    )


def test_regions_declares_training_and_out_region():
    region_spec = pendulum.regions()
    assert region_spec.training_state_box.shape == (4, 2)
    assert region_spec.training_action_interval.shape == (1, 2)
    assert len(region_spec.out_regions) == 1
    assert region_spec.out_regions[0].region_name == "out-near-inverted"


def test_tasks_declares_control_and_planning_with_different_tolerances():
    tasks = {task.kind: task for task in pendulum.tasks()}
    assert set(tasks) == {"control", "planning"}
    assert tasks["control"].tolerance == pytest.approx(0.05)
    assert tasks["planning"].tolerance == pytest.approx(0.30)


def test_angles_are_stored_unwrapped_not_mod_2pi():
    """worlds §4.2: angles are stored unwrapped (not mod 2*pi) -- a
    rollout that winds theta1 past pi must keep the raw, unwrapped
    value, not fold it back into [-pi, pi]."""
    state = np.array([3.2, 0.0, 0.0, 0.0])  # theta1 already past pi
    next_state = pendulum.transition(state, np.array([0.0]))
    # if the world wrapped angles, this would come back near -3.08
    # (3.2 - 2*pi); unwrapped, one 0.002s step barely moves it
    assert next_state[0] > np.pi

    import ast
    import inspect

    tree = ast.parse(inspect.getsource(pendulum.transition))
    for node in ast.walk(tree):
        assert not isinstance(node, ast.Mod), "transition() must never wrap angles"
        if isinstance(node, ast.Attribute):
            assert node.attr not in ("mod", "fmod"), "transition() must never wrap angles"
