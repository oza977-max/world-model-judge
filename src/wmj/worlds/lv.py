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
from wmj.worlds.errors import (
    ActionRangeError,
    RegionSpecError,
    StateFloorClampError,
    WorldInputShapeError,
)
from wmj.worlds.integrator import rk4_step
from wmj.worlds.regionspec import validate_out_regions

ALPHA = 1.0
BETA = 0.4
GAMMA = 0.8
DELTA = 0.2

DT = 0.02
HORIZON = 700
SCALE = np.array([4.0, 2.5])

STATE_FLOOR = 0.05
ACTION_RANGE = (-1.0, 1.0)  # the world's full declared action range, worlds §4.1

# How often the lever is pulled: 0.5 kicks per second of world time, so a
# kick on any given step with chance 0.5 × 0.02 = 0.01 (worlds §4.1, ADR-W2;
# design-review-010 — larger or more frequent kicks were measured to crash
# the prey population through its floor on full-length runs).
KICK_RATE_PER_S = 0.5
TRAINED_ACTION_MAX = 0.1  # trained kicks are in [-0.1, 0.1]; out-of-range kicks (0.1, 0.2]


def _deriv(state: np.ndarray) -> np.ndarray:
    """The predator–prey equations, for one state `[2]` or a batch `[n, 2]`.

    The one derivative both `transition` and `transition_batch` use —
    element-wise arithmetic only, so each row's result is the same bits
    whether it is computed alone or in a batch (worlds ADR-W1, TC-WD3-04).
    """
    x = state[..., 0]
    y = state[..., 1]
    dx = ALPHA * x - BETA * x * y
    dy = DELTA * x * y - GAMMA * y
    return np.stack([dx, dy], axis=-1)


def _apply_action(state: np.ndarray, action: np.ndarray) -> np.ndarray:
    """The lever: an instantaneous prey impulse (worlds ADR-W2).

    Works for one state `[2]` with action `[1]`, or a batch `[n, 2]` with
    actions `[n, 1]`. Does not clamp — a floor violation is caught by the
    caller's own check, which aborts loudly per worlds §7 rather than
    silently keeping an unphysical excursion.
    """
    perturbed = np.array(state, dtype=float, copy=True)
    perturbed[..., 0] = perturbed[..., 0] + action[..., 0]
    return perturbed


def _first_bad_row(mask: np.ndarray) -> int:
    """Index of the first failing row (0 for a single state)."""
    rows = np.atleast_2d(mask).any(axis=-1)
    return int(np.argmax(rows))


def _check_inputs_finite(states: np.ndarray, actions: np.ndarray) -> None:
    # A NaN compares False against every bound, so it would slip past the
    # range and floor checks below and spread silently (independent
    # review, P3-C09 pass 1). Refuse it here.
    for what, values in (("state", states), ("action", actions)):
        bad = ~np.isfinite(values)
        if np.any(bad):
            row = _first_bad_row(bad)
            raise WorldInputShapeError(
                f"lv {what} in row {row} is not a finite number: "
                f"{np.atleast_2d(values)[row].tolist()} (WD-2, worlds §7)"
            )


def _check_action_range(actions: np.ndarray) -> None:
    bad = (actions < ACTION_RANGE[0]) | (actions > ACTION_RANGE[1])
    if np.any(bad):
        row = _first_bad_row(bad)
        raise ActionRangeError(
            f"lv action {np.atleast_2d(actions)[row].tolist()} in row {row} is outside "
            f"the world's declared range {ACTION_RANGE} (worlds spec §7)"
        )


def _check_floor(states: np.ndarray, what: str) -> None:
    bad = states < STATE_FLOOR
    if np.any(bad):
        row = _first_bad_row(bad)
        raise StateFloorClampError(
            f"lv {what} drove row {row} to {np.atleast_2d(states)[row].tolist()}, "
            f"below the floor {STATE_FLOOR} (worlds spec §7)"
        )


def transition_batch(states: np.ndarray, actions: np.ndarray) -> np.ndarray:
    """Advance `n` states one step at once: `[n, 2]` states, `[n, 1]` actions.

    Row `i` is bit-identical to `transition(states[i], actions[i])` — the
    same derivative and the same `rk4_step`, only fed a batch (worlds
    ADR-W1, TC-WD3-04). The same checks apply row by row; any failing row
    aborts the whole call (worlds §7 — never a silently dropped row).
    """
    states = np.asarray(states, dtype=float)
    actions = np.asarray(actions, dtype=float)
    if states.ndim != 2 or states.shape[1] != 2 or actions.shape != (states.shape[0], 1):
        raise WorldInputShapeError(
            f"lv transition_batch expects states [n, 2] and actions [n, 1]; got "
            f"{states.shape} and {actions.shape} (WD-2, worlds §4.3)"
        )
    _check_inputs_finite(states, actions)
    _check_action_range(actions)
    perturbed = _apply_action(states, actions)
    _check_floor(perturbed, "prey impulse")
    next_states = rk4_step(_deriv, perturbed, DT)
    _check_floor(next_states, "RK4 step")
    return next_states


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

    One state `[2]` and one action `[1]` only — a batch goes through
    `transition_batch` (design-review-010: a batch passed here used to
    come back as a wrong-shaped array with no error).
    """
    state = np.asarray(state, dtype=float)
    action = np.asarray(action, dtype=float)
    if state.shape != (2,) or action.shape != (1,):
        raise WorldInputShapeError(
            f"lv transition expects one state [2] and one action [1]; got "
            f"{state.shape} and {action.shape} — use transition_batch for many "
            f"(WD-2, worlds §4.3)"
        )
    _check_inputs_finite(state, action)
    _check_action_range(action)
    perturbed = _apply_action(state, action)
    _check_floor(perturbed, "prey impulse")
    next_state = rk4_step(_deriv, perturbed, DT)
    _check_floor(next_state, "RK4 step")
    return next_state


def conserved(state: np.ndarray) -> float:
    """V(x,y) = delta*x - gamma*ln(x) + beta*y - alpha*ln(y)."""
    x, y = state
    return float(DELTA * x - GAMMA * math.log(x) + BETA * y - ALPHA * math.log(y))


def _build_region_spec() -> RegionSpec:
    return RegionSpec(
        training_state_box=np.array([[2.0, 6.0], [1.0, 4.0]]),
        training_action_interval=np.array([[-TRAINED_ACTION_MAX, TRAINED_ACTION_MAX]]),
        out_regions=(
            OutRegion(
                region_name="out-high-amplitude",
                axis="state",
                state_box=np.array([[8.0, 12.0], [4.0, 6.0]]),
                action_box=np.array([[-TRAINED_ACTION_MAX, TRAINED_ACTION_MAX]]),
            ),
            # design-review-010: familiar starts, kicks bigger than any seen
            # in training — the action axis of WD-5 (worlds ADR-W4).
            OutRegion(
                region_name="out-large-action",
                axis="action",
                state_box=np.array([[2.0, 6.0], [1.0, 4.0]]),
                action_box=np.array([[-2 * TRAINED_ACTION_MAX, 2 * TRAINED_ACTION_MAX]]),
            ),
        ),
    )


def _validate_region_spec(region_spec: RegionSpec) -> None:
    """Worlds spec §7: regions are validated at construction.

    Training box strictly inside the state-floor-safe domain; every
    out-region leaves the training territory on the axis it declares
    (`wmj.worlds.regionspec.validate_out_regions`).
    """
    if np.any(region_spec.training_state_box[:, 0] <= STATE_FLOOR):
        raise RegionSpecError(
            f"lv training_state_box {region_spec.training_state_box!r} is not "
            f"strictly above the state floor {STATE_FLOOR} (worlds spec §7)"
        )
    validate_out_regions("lv", region_spec)


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


WORLD = LVWorld()
