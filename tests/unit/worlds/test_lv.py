"""Tests for wmj.worlds.lv — the Lotka-Volterra world (worlds spec §4.1).

TC-WD1-01: the no-action transition matches a hand-verified reference
value, computed by an independent plain-Python (no NumPy) RK4
implementation of the exact same equations — a second, independent
reading of the pinned formulas, not a second call into the same code.

TC-WD2-01: a non-null action changes the outcome — proving the
interface is genuinely (state, action) -> next_state.
"""

from __future__ import annotations

import numpy as np
import pytest

from wmj.worlds import lv


def test_tc_wd1_01_transition_matches_hand_verified_reference_no_action():
    # Independently computed via a plain-Python (no NumPy) RK4 of
    # dx/dt = alpha*x - beta*x*y, dy/dt = delta*x*y - gamma*y from
    # (4.5, 2.0), dt=0.02 -- see build/handovers/P1-C02.md for the
    # standalone derivation script.
    state = np.array([4.5, 2.0])
    action = np.array([0.0])

    next_state = lv.transition(state, action)

    expected = np.array([4.517962843950525, 2.0040760496001093])
    assert next_state == pytest.approx(expected, rel=1e-9)


def test_tc_wd2_01_nonnull_action_changes_the_outcome():
    state = np.array([4.5, 2.0])

    no_action = lv.transition(state, np.array([0.0]))
    with_action = lv.transition(state, np.array([0.3]))

    assert not np.allclose(no_action, with_action)


def test_transition_state_floor_clamps_at_0_05():
    # a large negative impulse would otherwise drive prey below zero
    state = np.array([0.1, 2.0])
    next_state = lv.transition(state, np.array([-1.0]))
    assert next_state[0] >= 0.05


def test_conserved_quantity_is_a_scalar():
    state = np.array([4.5, 2.0])
    value = lv.conserved(state)
    assert isinstance(value, float)


def test_world_declares_dimensionality_and_scale():
    assert lv.WORLD.d == 2
    assert lv.WORLD.a == 1
    assert lv.WORLD.dt == pytest.approx(0.02)
    assert lv.WORLD.scale.shape == (2,)


def test_regions_declares_training_and_out_region():
    region_spec = lv.regions()
    assert region_spec.training_state_box.shape == (2, 2)
    assert region_spec.training_action_interval.shape == (1, 2)
    assert len(region_spec.out_regions) == 1
    assert region_spec.out_regions[0].region_name == "out-high-amplitude"


def test_tasks_declares_control_and_planning():
    task_names = {task.name for task in lv.tasks()}
    assert task_names == {"lv-control", "lv-planning"}
