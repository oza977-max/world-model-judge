"""wmj.worlds.integrator — the one shared fixed-step RK4.

In plain words: every place in this project that needs to advance a
world forward in time — the ground truth, the training-data generator,
the benchmark generator — calls this exact function with the exact
same step size, so a chart never quietly mixes two different notions
of "one step forward" (worlds spec ADR-W1). It's a pure function: same
inputs always give the same outputs, and it never touches anything
outside its own arguments.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

DerivFn = Callable[[np.ndarray], np.ndarray]


def rk4_step(deriv_fn: DerivFn, state: np.ndarray, dt: float) -> np.ndarray:
    """Advance state by one classical fourth-order Runge-Kutta step.

    deriv_fn(state) -> derivative, autonomous (no explicit time
    dependence) — worlds ADR-W2 keeps the action out of the ODE itself
    by applying it as an impulse before this function is ever called.
    """
    k1 = deriv_fn(state)
    k2 = deriv_fn(state + (dt / 2.0) * k1)
    k3 = deriv_fn(state + (dt / 2.0) * k2)
    k4 = deriv_fn(state + dt * k3)
    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
