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
from wmj.worlds.errors import ActionRangeError, RegionSpecError, StateFloorClampError
from wmj.worlds.integrator import rk4_step

ALPHA = 1.0
BETA = 0.4
GAMMA = 0.8
DELTA = 0.2

DT = 0.02
HORIZON = 700
SCALE = np.array([4.0, 2.5])

STATE_FLOOR = 0.05
ACTION_RANGE = (-1.0, 1.0)  # the world's full declared action range, worlds §4.1


def _deriv(state: np.ndarray) -> np.ndarray:
    x, y = state
    dx = ALPHA * x - BETA * x * y
    dy = DELTA * x * y - GAMMA * y
    return np.array([dx, dy])


def _apply_action(state: np.ndarray, action: np.ndarray) -> np.ndarray:
    """The lever: an instantaneous prey impulse (worlds ADR-W2).

    Does not clamp — a floor violation here is caught by transition()'s
    own check, which aborts loudly per worlds §7 rather than silently
    keeping an unphysical excursion.
    """
    u = action[0]
    return np.array([state[0] + u, state[1]])


def transition(state: np.ndarray, action: np.ndarray) -> np.ndarray:
    """(state, action) -> next_state: impulse, then one RK4 step.

    A null action (0.0) makes the impulse the identity, so the
    no-action case is exactly one RK4 step of the bare equations
    (TC-WD1-01's reference value is computed against exactly this).

    Worlds spec §7, "one rule, no scope exceptions": an action outside
    the world's declared range is a caller bug (ActionRangeError); a
    state-floor excursion means the region/action declarations allowed
    an unphysical excursion, which is a spec bug to fix, never data to
    train on or grade against (StateFloorClampError) — both abort the
    run loudly rather than being silently clamped.
    """
    u = float(action[0])
    if not (ACTION_RANGE[0] <= u <= ACTION_RANGE[1]):
        raise ActionRangeError(
            f"lv action {u!r} outside the world's declared range "
            f"{ACTION_RANGE} (worlds spec §7)"
        )
    perturbed = _apply_action(state, action)
    if np.any(perturbed < STATE_FLOOR):
        raise StateFloorClampError(
            f"lv prey impulse drove state to {perturbed!r}, below the "
            f"floor {STATE_FLOOR} (worlds spec §7)"
        )
    next_state = rk4_step(_deriv, perturbed, DT)
    if np.any(next_state < STATE_FLOOR):
        raise StateFloorClampError(
            f"lv RK4 step drove state to {next_state!r}, below the "
            f"floor {STATE_FLOOR} (worlds spec §7)"
        )
    return next_state


def conserved(state: np.ndarray) -> float:
    """V(x,y) = delta*x - gamma*ln(x) + beta*y - alpha*ln(y)."""
    x, y = state
    return float(DELTA * x - GAMMA * math.log(x) + BETA * y - ALPHA * math.log(y))


def _build_region_spec() -> RegionSpec:
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


def _validate_region_spec(region_spec: RegionSpec) -> None:
    """Worlds spec §7: regions are validated at construction.

    Training box strictly inside the state-floor-safe domain;
    out-region disjoint from training box on at least one axis.
    """
    if np.any(region_spec.training_state_box[:, 0] <= STATE_FLOOR):
        raise RegionSpecError(
            f"lv training_state_box {region_spec.training_state_box!r} is not "
            f"strictly above the state floor {STATE_FLOOR} (worlds spec §7)"
        )
    for out_region in region_spec.out_regions:
        disjoint = np.any(
            (out_region.state_box[:, 0] > region_spec.training_state_box[:, 1])
            | (out_region.state_box[:, 1] < region_spec.training_state_box[:, 0])
        )
        if not disjoint:
            raise RegionSpecError(
                f"lv out-region {out_region.region_name!r} is not disjoint "
                f"from the training box on any axis (worlds spec §7)"
            )


_REGION_SPEC = _build_region_spec()
_validate_region_spec(_REGION_SPEC)


def regions() -> RegionSpec:
    return _REGION_SPEC


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
