"""wmj.harness.action_response — does the model actually use its action? (MU-3, TC-MU3-04)

In plain words: a simulator is only a simulator if what you *do* changes what
happens. The world has a lever (the action); a model that ignores the lever is a
time-series forecaster wearing a simulator's name, and it can look perfectly fine
on trials where nobody pushes anything. So every model faces one extra check: hold
the starting state still, hand the model two different actions from the range it was
trained on, and see whether its answer moves. If, across all the probes, its answer
never moves by more than floating-point noise, the model is flagged **action-blind**.

The rule is the one MU-3 fixed before any model existed: *the two predictions must
differ by more than floating-point noise for at least one (state, action-pair)
probe; a model that never distinguishes any action pair fails.* MU-3 left one number
open — how big "floating-point noise" is — and this module pins it in one place:
`ACTION_RESPONSE_TOLERANCE`, a billionth of the world's own scale (float64 rounding is
about ten million times smaller; every real effect of an action is far larger), recorded
in the spec-corrections backlog (A26).

How the probes are chosen (the same for every model, so no model can be favoured):
16 starting states drawn from the world's training box by the harness's own seeded
stream, and three action levels — the low end, the middle and the high end of the
trained interval (the ends included) — which gives three action pairs per state, 48
probes. A model "responds" to a probe if its guess *or its error bar* moves by more than
the tolerance (a model that only changes how sure it is still uses the action).

Two safeguards: a model with memory between steps (`linear`) is reset before every
single prediction, so its memory cannot pass for action response; and a prediction
that is not a finite number is an error, never "identical". This check scores nothing:
it takes the model, the world's context and a seed — no evaluation trials, no skill
scores, no matching margin (MU-6/JU-11).

Known limits (backlog A26, to be stated alongside any verdict that uses it): the three fixed
action levels can miss a response that is zero at both ends and the middle of the range and
non-zero only in between (no such model is in the roster; every model that uses the action responds on all
probes); and a model whose answers are random but ignore the action is not flagged, because
the check does not predict the same input twice (every model in the roster is deterministic).

The check is built and tested here; the run command that calls it is P6-C01.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from wmj.errors import WmjError
from wmj.models.base import SeedSource, WorldContext

ACTION_RESPONSE_TOLERANCE = 1e-9  # in units of the world's own scale (backlog A26)
PROBE_STATES = 16  # start states per world; three action pairs each


class ActionResponseError(WmjError):
    """Raised when the action-response check is given something it cannot judge."""


@dataclass(frozen=True)
class ActionResponseResult:
    """What the check found. `action_blind` is the flag; the rest is the evidence."""

    action_blind: bool
    n_probes: int
    n_responding: int  # probes where the guess or the error bar moved by more than the tolerance
    largest_change: float  # the biggest such movement, in units of the world's scale
    tolerance: float


def action_response_probes(
    ctx: WorldContext, run_seed: int, n_states: int = PROBE_STATES
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """`(states[n, d], actions[3, a], pairs[3, 2])`: the same probes for every model.

    The states are uniform in the training box (seeded, never model-dependent); the
    three action levels are the low end, the middle and the high end of the trained
    interval; `pairs` lists which two levels are compared: (low, high), (low, middle),
    (middle, high).
    """
    if isinstance(n_states, bool) or not isinstance(n_states, int) or n_states < 1:
        raise ActionResponseError(f"n_states must be a positive Python int, got {n_states!r}")
    box = ctx.training_state_box
    interval = ctx.training_action_interval
    rng = SeedSource(run_seed, None).rng_for("action-response", ctx.world_name, "states")
    states = box[:, 0] + (box[:, 1] - box[:, 0]) * rng.random((n_states, box.shape[0]))
    levels = np.stack([interval[:, 0], interval.mean(axis=1), interval[:, 1]])
    pairs = np.array([[0, 2], [0, 1], [1, 2]])
    return states, levels, pairs


def _forecast(model: Any, state: np.ndarray, action: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    model.reset()  # a model with memory must not let that memory pass for action response
    prediction = model.predict(state, action)
    mean = np.asarray(prediction.mean, dtype=float)
    spread = np.asarray(prediction.spread, dtype=float)
    if mean.shape != state.shape or spread.shape != state.shape:
        raise ActionResponseError(
            f"the model returned a guess of shape {tuple(mean.shape)} and an error bar of shape "
            f"{tuple(spread.shape)} for a state of shape {tuple(state.shape)} — one number per quantity "
            "is required"
        )
    if not (np.all(np.isfinite(mean)) and np.all(np.isfinite(spread))):
        raise ActionResponseError(
            "the model returned a guess or an error bar that is not a finite number during the "
            "action-response check — that is a broken model, not an action-blind one"
        )
    return mean, spread


def check_action_response(
    model: Any,
    ctx: WorldContext,
    run_seed: int,
    *,
    n_states: int = PROBE_STATES,
    tolerance: float = ACTION_RESPONSE_TOLERANCE,
) -> ActionResponseResult:
    """Run the MU-3 action-response check on one model (TC-MU3-04).

    The model is action-blind when *no* probe's two predictions differ by more than
    `tolerance` of the world's scale in any dimension of the guess or the error bar.
    """
    if (
        isinstance(tolerance, bool)
        or not isinstance(tolerance, (int, float))
        or not math.isfinite(tolerance)
        or tolerance < 0.0
    ):
        raise ActionResponseError(f"tolerance must be a finite non-negative number, got {tolerance!r}")
    states, levels, pairs = action_response_probes(ctx, run_seed, n_states)
    scale = np.asarray(ctx.scale, dtype=float)
    if scale.shape != (ctx.state_dim,) or not (np.all(np.isfinite(scale)) and np.all(scale > 0.0)):
        raise ActionResponseError(f"the world's scale must be one finite positive number per quantity, got {scale!r}")
    responding, largest = 0, 0.0
    for state in states:
        forecasts = [_forecast(model, state, level) for level in levels]
        for first, second in pairs:
            (mean_a, spread_a), (mean_b, spread_b) = forecasts[first], forecasts[second]
            change = max(
                float(np.max(np.abs(mean_a - mean_b) / scale)),
                float(np.max(np.abs(spread_a - spread_b) / scale)),
            )
            largest = max(largest, change)
            if change > tolerance:
                responding += 1
    return ActionResponseResult(
        action_blind=responding == 0,
        n_probes=len(states) * len(pairs),
        n_responding=responding,
        largest_change=largest,
        tolerance=float(tolerance),
    )
