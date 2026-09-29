"""wmj.worlds.actions — the one kick generator every rollout uses.

In plain words: this decides when the lever is pulled and how hard. At
each step there is a fixed small chance of a kick; if there is one, its
size is drawn evenly from a named band — the "in" band (the kicks the
models were trained on) or the "out" band (bigger than anything seen in
training, used only for the `out-large-action` test region). Every other
step is "do nothing". Training trajectories, evaluation trials and the
drift benchmark all draw from this one function, each with its own seed,
so the kicks the models learn from and the kicks they are graded on are
the same kind of thing (worlds spec ADR-W2, design-review-010).

Why kicks are occasional rather than every step: measured before this
was written, kicking every step drove predator–prey through its
population floor in 16 of 20 runs and threw the pendulum to ~13× its
normal scale; an occasional kick keeps each run recognisably the world
while still exercising the lever.

Pure: the caller passes a seeded `numpy.random.Generator`; this module
never creates one (cross-cutting ADR-002 rule 2 — the harness owns seeds).
"""

from __future__ import annotations

import math
import numbers
from typing import Any

import numpy as np

from wmj.errors import WmjError

_BANDS = ("in", "out")
_AXES = (None, "state", "action", "both")


class KickSpecError(WmjError):
    """Raised when a kick sequence is requested with impossible settings.

    A band that does not exist, a chance outside [0, 1], or a kick size
    that is not a positive finite number would each produce a sequence
    that quietly means something other than the spec says — so they
    fail loudly instead (worlds ADR-W2; cross-cutting Error-Handling
    rule 1).
    """


def step_probability(kick_rate_per_s: float, dt: float) -> float:
    """The chance of a kick on one step: `kick_rate_per_s × dt` (ADR-W2).

    LV: 0.5 kicks per second × 0.02 s = 0.01. Pendulum: 1.0 × 0.002 = 0.002.
    """
    p = kick_rate_per_s * dt
    if not (math.isfinite(p) and 0.0 <= p <= 1.0):
        raise KickSpecError(
            f"kick rate {kick_rate_per_s!r}/s at dt={dt!r} gives a per-step chance "
            f"{p!r}, outside [0, 1] (worlds ADR-W2)"
        )
    return p


def band_for_axis(axis: str | None) -> str:
    """Which kick band a region's trials use, from its declared axis (ADR-W4).

    Only regions that are out of the trained territory *on the action axis*
    (`"action"` or `"both"`) get the bigger `"out"` kicks; the training
    region (`None`) and state-axis regions get the trained `"in"` kicks.
    """
    if axis not in _AXES:
        raise KickSpecError(
            f"region axis {axis!r} is not one of {list(_AXES)} (worlds ADR-W4)"
        )
    return "out" if axis in ("action", "both") else "in"


def kick_sequence(rng: Any, horizon: int, p_step: float, band: str, umax: float) -> np.ndarray:
    """One rollout's actions, `float64[horizon, 1]` (worlds ADR-W2).

    Draws `r = rng.random((horizon, 3))` — three numbers per step, always,
    so the generator's state afterwards never depends on how many kicks
    happened. Step `t` gets a kick iff `r[t, 0] < p_step`:

    - `band="in"`:  `u = umax · (2·r[t,1] − 1)`, uniform on `[−umax, umax]`;
    - `band="out"`: `u = s · umax · (2 − r[t,1])` with `s = +1` if
      `r[t,2] < 0.5` else `−1`, so `|u|` is uniform on `(umax, 2·umax]`.

    Every other step is the null action `0.0`. `umax` is the world's
    trained half-width (its `training_action_interval` upper edge).
    """
    if band not in _BANDS:
        raise KickSpecError(f"kick band {band!r} is not one of {list(_BANDS)} (worlds ADR-W2)")
    if not (
        isinstance(horizon, numbers.Integral) and not isinstance(horizon, bool) and horizon >= 0
    ):
        raise KickSpecError(f"horizon {horizon!r} must be a non-negative int (worlds ADR-W2)")
    if not (math.isfinite(p_step) and 0.0 <= p_step <= 1.0):
        raise KickSpecError(f"per-step kick chance {p_step!r} is outside [0, 1] (worlds ADR-W2)")
    if not (math.isfinite(umax) and umax > 0.0):
        raise KickSpecError(f"kick half-width {umax!r} must be positive and finite (worlds ADR-W2)")

    r = rng.random((horizon, 3))
    kicked = r[:, 0] < p_step
    if band == "in":
        size = umax * (2.0 * r[:, 1] - 1.0)
    else:
        sign = np.where(r[:, 2] < 0.5, 1.0, -1.0)
        size = sign * umax * (2.0 - r[:, 1])
    return np.where(kicked, size, 0.0).reshape(horizon, 1)
