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
from wmj.worlds.errors import ActionRangeError, WorldInputShapeError
from wmj.worlds.integrator import rk4_step
from wmj.worlds.regionspec import validate_out_regions

M = 1.0
L = 1.0
G = 9.81

DT = 0.002
HORIZON = 5000
SCALE = np.array([math.pi, math.pi, 2 * math.pi, 2 * math.pi])

ACTION_RANGE = (-2.0, 2.0)  # the world's full declared action range, worlds §4.2

# How often the lever is pulled: 1.0 kick per second of world time, so a
# kick on any given step with chance 1.0 × 0.002 = 0.002 (worlds §4.2, ADR-W2).
KICK_RATE_PER_S = 1.0
TRAINED_ACTION_MAX = 1.0  # trained kicks are in [-1, 1]; out-of-range kicks (1, 2]


def _deriv(state: np.ndarray) -> np.ndarray:
    """The double-pendulum equations, for one state `[4]` or a batch `[n, 4]`.

    The one derivative both `transition` and `transition_batch` use. It
    uses NumPy's element-wise `np.sin`/`np.cos` (it used Python's scalar
    `math.sin`/`math.cos` before design-review-010, which cannot take a
    batch); measured bit-identical on 2,000,000 values here, and TC-WD1-01's
    pinned reference values plus TC-WD3-04 are what guarantee it holds.
    """
    theta1 = state[..., 0]
    theta2 = state[..., 1]
    omega1 = state[..., 2]
    omega2 = state[..., 3]
    delta = theta1 - theta2
    denom = L * (3.0 * M - M * np.cos(2.0 * delta))

    theta1_ddot = (
        -G * (3.0 * M) * np.sin(theta1)
        - M * G * np.sin(theta1 - 2.0 * theta2)
        - 2.0
        * np.sin(delta)
        * M
        * (omega2**2 * L + omega1**2 * L * np.cos(delta))
    ) / denom
    theta2_ddot = (
        2.0
        * np.sin(delta)
        * (
            omega1**2 * L * (2.0 * M)
            + G * (2.0 * M) * np.cos(theta1)
            + omega2**2 * L * M * np.cos(delta)
        )
    ) / denom

    return np.stack([omega1, omega2, theta1_ddot, theta2_ddot], axis=-1)


def _apply_action(state: np.ndarray, action: np.ndarray) -> np.ndarray:
    """The lever: an instantaneous kick to the first joint's angular
    velocity (worlds ADR-W2). One state `[4]` with action `[1]`, or a
    batch `[n, 4]` with actions `[n, 1]`."""
    kicked = np.array(state, dtype=float, copy=True)
    kicked[..., 2] = kicked[..., 2] + action[..., 0]
    return kicked


def _first_bad_row(mask: np.ndarray) -> int:
    """Index of the first failing row (0 for a single state)."""
    rows = np.atleast_2d(mask).any(axis=-1)
    return int(np.argmax(rows))


def _check_inputs_finite(states: np.ndarray, actions: np.ndarray) -> None:
    # A NaN compares False against every bound, so it would slip past the
    # range check and spread silently; the pendulum has no floor to catch
    # it later (independent review, P3-C09 pass 1). Refuse it here.
    for what, values in (("state", states), ("action", actions)):
        bad = ~np.isfinite(values)
        if np.any(bad):
            row = _first_bad_row(bad)
            raise WorldInputShapeError(
                f"pendulum {what} in row {row} is not a finite number: "
                f"{np.atleast_2d(values)[row].tolist()} (WD-2, worlds §7)"
            )


def _check_action_range(actions: np.ndarray) -> None:
    bad = (actions < ACTION_RANGE[0]) | (actions > ACTION_RANGE[1])
    if np.any(bad):
        row = _first_bad_row(bad)
        raise ActionRangeError(
            f"pendulum action {np.atleast_2d(actions)[row].tolist()} in row {row} is "
            f"outside the world's declared range {ACTION_RANGE} (worlds spec §7)"
        )


def transition_batch(states: np.ndarray, actions: np.ndarray) -> np.ndarray:
    """Advance `n` states one step at once: `[n, 4]` states, `[n, 1]` actions.

    Row `i` is bit-identical to `transition(states[i], actions[i])` — the
    same derivative and the same `rk4_step`, only fed a batch (worlds
    ADR-W1, TC-WD3-04). An out-of-range action in any row aborts the
    whole call (worlds §7).
    """
    states = np.asarray(states, dtype=float)
    actions = np.asarray(actions, dtype=float)
    if states.ndim != 2 or states.shape[1] != 4 or actions.shape != (states.shape[0], 1):
        raise WorldInputShapeError(
            f"pendulum transition_batch expects states [n, 4] and actions [n, 1]; got "
            f"{states.shape} and {actions.shape} (WD-2, worlds §4.3)"
        )
    _check_inputs_finite(states, actions)
    _check_action_range(actions)
    return rk4_step(_deriv, _apply_action(states, actions), DT)


def transition(state: np.ndarray, action: np.ndarray) -> np.ndarray:
    """(state, action) -> next_state: impulse, then one RK4 step.

    Angles are never wrapped (worlds §4.2: "stored unwrapped, not mod
    2*pi") -- distance and the flip task need the winding.
    """
    state = np.asarray(state, dtype=float)
    action = np.asarray(action, dtype=float)
    if state.shape != (4,) or action.shape != (1,):
        raise WorldInputShapeError(
            f"pendulum transition expects one state [4] and one action [1]; got "
            f"{state.shape} and {action.shape} — use transition_batch for many "
            f"(WD-2, worlds §4.3)"
        )
    _check_inputs_finite(state, action)
    _check_action_range(action)
    return rk4_step(_deriv, _apply_action(state, action), DT)


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
        training_action_interval=np.array([[-TRAINED_ACTION_MAX, TRAINED_ACTION_MAX]]),
        out_regions=(
            OutRegion(
                region_name="out-near-inverted",
                axis="state",
                state_box=np.array(
                    [[2.5, math.pi], [-0.3, 0.3], [-0.5, 0.5], [-0.5, 0.5]]
                ),
                action_box=np.array([[-TRAINED_ACTION_MAX, TRAINED_ACTION_MAX]]),
            ),
            # design-review-010: familiar starts, kicks bigger than any seen
            # in training — the action axis of WD-5 (worlds ADR-W4).
            OutRegion(
                region_name="out-large-action",
                axis="action",
                state_box=np.array(
                    [[-0.3, 0.3], [-0.3, 0.3], [-0.5, 0.5], [-0.5, 0.5]]
                ),
                action_box=np.array([[-2 * TRAINED_ACTION_MAX, 2 * TRAINED_ACTION_MAX]]),
            ),
        ),
    )


def _validate_region_spec(region_spec: RegionSpec) -> None:
    """Worlds spec §7: every out-region leaves the training territory on
    the axis it declares (no state-floor concept applies to this world)."""
    validate_out_regions("pendulum", region_spec)


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
    kick_rate_per_s = KICK_RATE_PER_S

    def transition(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        return transition(state, action)

    def transition_batch(self, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
        return transition_batch(states, actions)

    def conserved(self, state: np.ndarray) -> float:
        return conserved(state)

    def regions(self) -> RegionSpec:
        return regions()

    def tasks(self) -> tuple[Task, ...]:
        return tasks()


WORLD = PendulumWorld()
