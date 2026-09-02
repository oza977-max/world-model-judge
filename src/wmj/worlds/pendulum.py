"""wmj.worlds.pendulum — the double pendulum world.

In plain words: two rigid links, hinged, swinging under gravity. Small
starting angles are gentle and predictable; nudged far enough, the
motion turns chaotic. The action is a "kick" to the first joint's
angular velocity. State is (theta1, theta2, omega1, omega2), advanced
by the one shared RK4 integrator every world uses (worlds spec ADR-W1,
§4.2) — the equations below are the design-review-002-corrected form
(the v1.1 draft's `2*m` coefficients were wrong; these use `3*m`,
re-derived from the Euler-Lagrange equations for equal masses/lengths).
"""

from __future__ import annotations

import math

import numpy as np

from wmj.worlds.base import OutRegion, RegionSpec, Task
from wmj.worlds.errors import ActionRangeError, RegionSpecError
from wmj.worlds.integrator import rk4_step

M = 1.0
L = 1.0
G = 9.81

DT = 0.002
HORIZON = 5000
SCALE = np.array([math.pi, math.pi, 2 * math.pi, 2 * math.pi])

ACTION_RANGE = (-2.0, 2.0)  # the world's full declared action range, worlds §4.2


def _deriv(state: np.ndarray) -> np.ndarray:
    theta1, theta2, omega1, omega2 = state
    delta = theta1 - theta2
    denom = L * (3.0 * M - M * math.cos(2.0 * delta))

    theta1_ddot = (
        -G * (3.0 * M) * math.sin(theta1)
        - M * G * math.sin(theta1 - 2.0 * theta2)
        - 2.0
        * math.sin(delta)
        * M
        * (omega2**2 * L + omega1**2 * L * math.cos(delta))
    ) / denom
    theta2_ddot = (
        2.0
        * math.sin(delta)
        * (
            omega1**2 * L * (2.0 * M)
            + G * (2.0 * M) * math.cos(theta1)
            + omega2**2 * L * M * math.cos(delta)
        )
    ) / denom

    return np.array([omega1, omega2, theta1_ddot, theta2_ddot])


def _apply_action(state: np.ndarray, action: np.ndarray) -> np.ndarray:
    """The lever: an instantaneous kick to the first joint's angular
    velocity (worlds ADR-W2)."""
    u = action[0]
    theta1, theta2, omega1, omega2 = state
    return np.array([theta1, theta2, omega1 + u, omega2])


def transition(state: np.ndarray, action: np.ndarray) -> np.ndarray:
    """(state, action) -> next_state: impulse, then one RK4 step.

    Angles are never wrapped (worlds §4.2: "stored unwrapped, not mod
    2*pi") -- distance and the flip task need the winding.
    """
    u = float(action[0])
    if not (ACTION_RANGE[0] <= u <= ACTION_RANGE[1]):
        raise ActionRangeError(
            f"pendulum action {u!r} outside the world's declared range "
            f"{ACTION_RANGE} (worlds spec §7)"
        )
    perturbed = _apply_action(state, action)
    return rk4_step(_deriv, perturbed, DT)


def conserved(state: np.ndarray) -> float:
    """Total mechanical energy E(theta1, theta2, omega1, omega2)."""
    theta1, theta2, omega1, omega2 = state
    delta = theta1 - theta2
    energy = (
        M * L**2 * omega1**2
        + 0.5 * M * L**2 * omega2**2
        + M * L**2 * omega1 * omega2 * math.cos(delta)
        - (2.0 * M) * G * L * math.cos(theta1)
        - M * G * L * math.cos(theta2)
    )
    return float(energy)


def _build_region_spec() -> RegionSpec:
    return RegionSpec(
        training_state_box=np.array(
            [[-0.3, 0.3], [-0.3, 0.3], [-0.5, 0.5], [-0.5, 0.5]]
        ),
        training_action_interval=np.array([[-1.0, 1.0]]),
        out_regions=(
            OutRegion(
                region_name="out-near-inverted",
                axis="state",
                state_box=np.array(
                    [[2.5, math.pi], [-0.3, 0.3], [-0.5, 0.5], [-0.5, 0.5]]
                ),
                action_box=np.array([[-1.0, 1.0]]),
            ),
        ),
    )


def _validate_region_spec(region_spec: RegionSpec) -> None:
    """Worlds spec §7: out-region disjoint from training box on at
    least one axis (no state-floor concept applies to this world)."""
    for out_region in region_spec.out_regions:
        disjoint = np.any(
            (out_region.state_box[:, 0] > region_spec.training_state_box[:, 1])
            | (out_region.state_box[:, 1] < region_spec.training_state_box[:, 0])
        )
        if not disjoint:
            raise RegionSpecError(
                f"pendulum out-region {out_region.region_name!r} is not "
                f"disjoint from the training box on any axis (worlds spec §7)"
            )


_REGION_SPEC = _build_region_spec()
_validate_region_spec(_REGION_SPEC)


def regions() -> RegionSpec:
    return _REGION_SPEC


def tasks() -> tuple[Task, ...]:
    return (
        Task(name="dp-control", kind="control", tolerance=0.05, horizon=HORIZON),
        Task(name="dp-planning", kind="planning", tolerance=0.30, horizon=HORIZON),
    )


class PendulumWorld:
    """Satisfies the World protocol (worlds spec §4.3) as one object."""

    d = 4
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


WORLD = PendulumWorld()
