"""Tests for wmj.judge.climatology.switch_step / evaluation_step (ADR-J5).

In plain words: the switch step is the first step where the world's own drift exceeds a task's
tolerance. These tests pin exactly which step that is, including the edge cases (equal to the
tolerance is still fine, step 0 never counts, no crossing means no switch step).
"""

from __future__ import annotations

import numpy as np
import pytest

from wmj.judge.climatology import evaluation_step, switch_step

CURVE = np.array([0.0, 0.05, 0.10, 0.20, 0.40, 0.80])


def test_the_first_step_strictly_above_the_tolerance_is_the_switch_step():
    assert switch_step(CURVE, 0.15, 5) == 3
    assert switch_step(CURVE, 0.01, 5) == 1
    assert switch_step(CURVE, 0.30, 5) == 4
    assert switch_step(CURVE, 0.60, 5) == 5


def test_a_distance_exactly_equal_to_the_tolerance_does_not_exceed_it():
    assert switch_step(CURVE, 0.10, 5) == 3  # curve[2] == 0.10 passes; curve[3] = 0.20 is the first above
    assert switch_step(CURVE, 0.80, 5) is None  # curve[5] == 0.80 passes


def test_step_zero_is_never_a_switch_step_even_if_the_curve_starts_above_the_tolerance():
    assert switch_step(np.array([5.0, 6.0, 7.0]), 1.0, 2) == 1


def test_no_crossing_within_the_horizon_gives_no_switch_step_even_if_it_crosses_later():
    assert switch_step(CURVE, 0.30, 3) is None  # it crosses at step 4, beyond a horizon of 3
    assert switch_step(CURVE, 0.30, 4) == 4  # the horizon is inclusive


def test_a_curve_that_does_not_reach_the_horizon_is_refused():
    with pytest.raises(ValueError, match="does not reach"):
        switch_step(CURVE, 0.1, 6)
    with pytest.raises(ValueError, match="does not reach"):
        switch_step(CURVE, 0.1, 0)


def test_the_evaluation_step_is_the_switch_step_or_the_horizon_when_there_is_none():
    assert evaluation_step(3, 5) == 3
    assert evaluation_step(None, 5) == 5
    assert evaluation_step(1, 1) == 1
