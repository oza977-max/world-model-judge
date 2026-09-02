"""wmj.worlds.divergence — how fast nearby trajectories separate (WD-4).

In plain words: start two copies of a world a hair apart, let both
run with nothing pushing on them, and watch the gap between them at
every step. For the predator-prey world the gap stays about the same
size (orbits are stable, they just slip out of phase very slowly).
For the pendulum, from a gentle start the gap stays small, but from a
near-inverted start it explodes — that is what "chaotic" means, and
it is the world's own fault, not any model's. The judge uses this
curve to know how far ahead *anyone* could be expected to predict
(worlds spec ADR-W3), and it separately checks that the integrator
itself isn't quietly leaking energy (ADR-W1, TC-WD3-03).

Pure functions only: every routine takes a transition function and
arrays in, returns arrays out. Seeding and sampling live in the
harness (`wmj.harness.benchmarks`), which is this module's consumer.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from wmj.errors import WmjError
from wmj.worlds.base import distance

Transition = Callable[[np.ndarray, np.ndarray], np.ndarray]
Conserved = Callable[[np.ndarray], float]


class DriftBoundError(WmjError):
    """Raised when the integrator's drift in a world's conserved
    quantity exceeds the declared bound (worlds ADR-W1, TC-WD3-03).

    The run fails loudly rather than producing a climatology reference
    the integrator has already corrupted.
    """


def perturb(state: np.ndarray, delta0: float) -> np.ndarray:
    """ADR-W3's declared perturbation: relative size delta0 applied to
    every state dimension, sign-alternating (+, -, +, -, ...).

    Returns a new array; the input is not mutated.
    """
    signs = np.where(np.arange(state.shape[0]) % 2 == 0, 1.0, -1.0)
    return state * (1.0 + delta0 * signs)


def separation_curve(
    transition: Transition,
    state0: np.ndarray,
    horizon: int,
    scale: np.ndarray,
    delta0: float,
    null_action: np.ndarray | None = None,
) -> np.ndarray:
    """Normalised distance between a trajectory and its perturbed twin
    at every step 0..horizon inclusive (`horizon + 1` entries).

    Null actions throughout: the benchmark measures the world's own
    drift, not a policy's (ADR-W3).
    """
    if null_action is None:
        null_action = np.zeros(1)
    base = np.array(state0, dtype=float, copy=True)
    twin = perturb(base, delta0)
    curve = np.empty(horizon + 1)
    curve[0] = distance(base, twin, scale)
    for step in range(horizon):
        base = transition(base, null_action)
        twin = transition(twin, null_action)
        curve[step + 1] = distance(base, twin, scale)
    return curve


def median_separation_curve(curves: np.ndarray) -> np.ndarray:
    """Per-step median across starts — median, not mean, because chaotic
    separations are heavy-tailed and one saturated trajectory would
    swamp a mean (ADR-W3)."""
    return np.median(curves, axis=0)


def conserved_drift(
    transition: Transition,
    conserved: Conserved,
    state0: np.ndarray,
    horizon: int,
    null_action: np.ndarray | None = None,
) -> tuple[float, float]:
    """Max absolute drift of the conserved quantity over the horizon
    under null action, plus its initial value.

    Returns `(max_abs_drift, initial_value)`. Normalising is the
    caller's decision — see `wmj.harness.benchmarks` for why the
    benchmark normalises by the invariant's range over a region rather
    than by the initial value alone.
    """
    if null_action is None:
        null_action = np.zeros(1)
    state = np.array(state0, dtype=float, copy=True)
    initial = conserved(state)
    worst = 0.0
    for _ in range(horizon):
        state = transition(state, null_action)
        worst = max(worst, abs(conserved(state) - initial))
    return worst, initial


def assert_drift_within_bound(rel_drift_max: float, bound: float, world_name: str) -> None:
    """ADR-W1's bound, enforced loudly (TC-WD3-03)."""
    if not (rel_drift_max < bound):
        raise DriftBoundError(
            f"WD-3 drift gate: {world_name} conserved-quantity drift "
            f"{rel_drift_max:.3e} (relative) is not below the declared bound "
            f"{bound:.1e} — the integrator would corrupt the climatology "
            f"reference; refusing rather than producing it (worlds ADR-W1, "
            f"TC-WD3-03)"
        )
