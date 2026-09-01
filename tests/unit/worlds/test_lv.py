"""Tests for wmj.worlds.lv — the Lotka-Volterra world (worlds spec §4.1).

TC-WD1-01: the no-action transition matches a hand-verified reference
value, computed by an independent plain-Python (no NumPy) RK4
implementation of the exact same equations — a second, independent
reading of the pinned formulas, not a second call into the same code.

**Reference-state deviation, documented (independent-review finding,
P1-C02 pass 1):** worlds.md v1.4 §8 pins the LV reference state as
(4.0, 2.5) with u=0. That point is this world's own equilibrium
(dx/dt = dy/dt = 0 there exactly, verified independently) — RK4 of an
exactly-zero derivative returns the input state unchanged regardless
of whether the integrator's arithmetic is implemented correctly, so a
test built on it cannot catch an integrator bug (only a wrong-constant
bug, since a wrong alpha/beta/gamma/delta would no longer zero out at
that exact point). This test instead hand-verifies at (4.5, 2.0), a
genuinely non-degenerate point, against an independent plain-Python
RK4 implementation of the identical equations (build/prompts/P1-C02.md
records this scope decision; worlds.md's own reference-state choice is
a candidate for a future design-review correction, not amended here).

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
    # (4.5, 2.0), dt=0.02 -- see this file's module docstring above
    # for why (4.0, 2.5), worlds.md's own pinned state, is unsuitable.
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


def test_action_that_would_drive_state_below_floor_aborts_loudly():
    """Worlds §7: a state-floor excursion aborts the run, never a
    silent clamp — a large negative impulse would otherwise drive prey
    below the floor."""
    state = np.array([0.1, 2.0])
    with pytest.raises(lv.StateFloorClampError):
        lv.transition(state, np.array([-1.0]))


def test_action_outside_declared_range_raises_action_range_error():
    """Worlds §7: an action outside the world's declared range is a
    caller bug, not a legitimate out-of-trained-range trial."""
    state = np.array([4.5, 2.0])
    with pytest.raises(lv.ActionRangeError):
        lv.transition(state, np.array([1.5]))


def test_regions_satisfy_worlds_section_7_invariants():
    """Worlds §7: training box strictly above the state floor; every
    out-region disjoint from the training box on at least one axis."""
    region_spec = lv.regions()
    assert np.all(region_spec.training_state_box[:, 0] > lv.STATE_FLOOR)

    out_region = region_spec.out_regions[0]
    disjoint = np.any(
        (out_region.state_box[:, 0] > region_spec.training_state_box[:, 1])
        | (out_region.state_box[:, 1] < region_spec.training_state_box[:, 0])
    )
    assert disjoint


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
