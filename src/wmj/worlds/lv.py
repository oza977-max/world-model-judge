"""wmj.worlds.lv — the Lotka-Volterra (foxes and rabbits) world.

In plain words: two populations, prey and predator, that rise and
fall in cycles. The "lever" (the action) lets you add or remove prey
at a step boundary; everything else is the classical predator-prey
equations, advanced by the one shared RK4 integrator every world in
this project uses (worlds spec ADR-W1, §4.1).
"""

from __future__ import annotations

import math

import numpy as np

from wmj.worlds.base import OutRegion, RegionSpec, Task
from wmj.worlds.integrator import rk4_step

ALPHA = 1.0
BETA = 0.4
GAMMA = 0.8
DELTA = 0.2

DT = 0.02
HORIZON = 700
SCALE = np.array([4.0, 2.5])

STATE_FLOOR = 0.05


def _deriv(state: np.ndarray) -> np.ndarray:
    x, y = state
    dx = ALPHA * x - BETA * x * y
    dy = DELTA * x * y - GAMMA * y
    return np.array([dx, dy])


def _apply_action(state: np.ndarray, action: np.ndarray) -> np.ndarray:
    """The lever: an instantaneous prey impulse (worlds ADR-W2)."""
    u = action[0]
    x = max(state[0] + u, STATE_FLOOR)
    return np.array([x, state[1]])


def transition(state: np.ndarray, action: np.ndarray) -> np.ndarray:
    """(state, action) -> next_state: impulse, then one RK4 step.

    A null action (0.0) makes the impulse the identity, so the
    no-action case is exactly one RK4 step of the bare equations
    (TC-WD1-01's reference value is computed against exactly this).
    """
    perturbed = _apply_action(state, action)
    next_state = rk4_step(_deriv, perturbed, DT)
    return np.maximum(next_state, STATE_FLOOR)


def conserved(state: np.ndarray) -> float:
    """V(x,y) = delta*x - gamma*ln(x) + beta*y - alpha*ln(y)."""
    x, y = state
    return float(DELTA * x - GAMMA * math.log(x) + BETA * y - ALPHA * math.log(y))


def regions() -> RegionSpec:
    return RegionSpec(
        training_state_box=np.array([[2.0, 6.0], [1.0, 4.0]]),
        training_action_interval=np.array([[-0.5, 0.5]]),
        out_regions=(
            OutRegion(
                region_name="out-high-amplitude",
                axis="state",
                state_box=np.array([[8.0, 12.0], [4.0, 6.0]]),
                action_box=np.array([[-0.5, 0.5]]),
            ),
        ),
    )


def tasks() -> tuple[Task, ...]:
    return (
        Task(name="lv-control", kind="control", tolerance=0.10, horizon=HORIZON),
        Task(name="lv-planning", kind="planning", tolerance=0.40, horizon=HORIZON),
    )


class LVWorld:
    """Satisfies the World protocol (worlds spec §4.3) as one object."""

    d = 2
    a = 1
    dt = DT
    scale = SCALE

    def transition(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        return transition(state, action)

    def conserved(self, state: np.ndarray) -> float:
        return conserved(state)

    def regions(self) -> RegionSpec:
        return regions()

    def tasks(self) -> tuple[Task, ...]:
        return tasks()


WORLD = LVWorld()
