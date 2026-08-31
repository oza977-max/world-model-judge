"""Tests for wmj.worlds.integrator.rk4_step (worlds spec ADR-W1).

One shared, pure, fixed-step RK4 — the module every ground-truth
generator in the project calls, so its own correctness is checked here
against a problem with a known closed-form solution, independent of
any world-specific equations.
"""

from __future__ import annotations

import numpy as np
import pytest

from wmj.worlds.integrator import rk4_step


def test_rk4_step_matches_closed_form_exponential_decay():
    """dy/dt = -y has the closed form y(t) = y0 * exp(-t); RK4's local
    error is O(h^5), so one small step should match to several digits."""

    def deriv(state: np.ndarray) -> np.ndarray:
        return -state

    y0 = np.array([1.0])
    dt = 0.01
    next_state = rk4_step(deriv, y0, dt)

    expected = y0 * np.exp(-dt)
    assert next_state == pytest.approx(expected, rel=1e-8)


def test_rk4_step_is_pure_and_deterministic():
    def deriv(state: np.ndarray) -> np.ndarray:
        return np.array([state[1], -state[0]])

    state = np.array([1.0, 0.0])
    dt = 0.05

    first = rk4_step(deriv, state, dt)
    second = rk4_step(deriv, state, dt)

    assert np.array_equal(first, second)
    # the input array itself must not be mutated
    assert np.array_equal(state, np.array([1.0, 0.0]))


def test_rk4_step_returns_same_shape_as_input():
    def deriv(state: np.ndarray) -> np.ndarray:
        return np.zeros_like(state)

    state = np.array([1.0, 2.0, 3.0, 4.0])
    next_state = rk4_step(deriv, state, 0.1)
    assert next_state.shape == state.shape
