"""wmj.worlds.divergence — how fast nearby trajectories separate (WD-4).

In plain words: start two copies of a world a hair apart, give both
exactly the same kicks at exactly the same moments (the same kind of
kicks the graded test runs get — design-review-010), and watch the gap
between them at every step. Because the kicks are identical, any gap is
still the world's own doing. For the predator-prey world the gap stays about the same
size (orbits are stable, they just slip out of phase very slowly).
For the pendulum, from a gentle start the gap stays small, but from a
near-inverted start it explodes — that is what "chaotic" means, and
it is the world's own fault, not any model's. The judge uses this
curve to know how far ahead *anyone* could be expected to predict
(worlds spec ADR-W3), and it separately checks that the integrator
itself isn't quietly leaking energy (ADR-W1, TC-WD3-03) — that check
runs with no kicks at all, because the energy (or orbit) is only
conserved when nothing pushes on the world.

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


class DivergenceInputError(WmjError):
    """Raised when a separation curve is asked for with a kick sequence of
    the wrong shape — one kick per step, `[horizon, a]` for one start or
    `[n, horizon, a]` for a batch. A mismatched sequence would silently
    pair the wrong kick with the wrong step (worlds ADR-W3)."""


class DriftBoundError(WmjError):
    """Raised when the integrator's drift in a world's conserved
    quantity exceeds the declared bound (worlds ADR-W1, TC-WD3-03).

    The run fails loudly rather than producing a climatology reference
    the integrator has already corrupted.
    """


class DegenerateInvariantRangeError(WmjError):
    """Raised when a region's conserved quantity has no range to
    normalise drift by.

    ADR-W1's bound is *relative to the invariant's range over the
    region*. If that range is zero or not finite, "relative drift" has
    no meaning, and quietly falling back to an absolute figure would
    change what the 1e-6 bound measures without saying so
    (code-review-001 I8). A world whose declared box gives its
    invariant no range is a spec defect, reported as one.
    """


def perturb(state: np.ndarray, delta0: float) -> np.ndarray:
    """ADR-W3's declared perturbation: relative size delta0 applied to
    every state dimension, sign-alternating (+, -, +, -, ...).

    Works on one state `[d]` or a batch `[n, d]` (the signs run along the
    last axis). Returns a new array; the input is not mutated.
    """
    signs = np.where(np.arange(state.shape[-1]) % 2 == 0, 1.0, -1.0)
    return state * (1.0 + delta0 * signs)


def separation_curve(
    transition: Transition,
    state0: np.ndarray,
    horizon: int,
    scale: np.ndarray,
    delta0: float,
    null_action: np.ndarray | None = None,
    actions: np.ndarray | None = None,
) -> np.ndarray:
    """Normalised distance between a trajectory and its perturbed twin
    at every step 0..horizon inclusive (`horizon + 1` entries).

    With `actions` (`[horizon, a]`), step `t` applies `actions[t]` to the
    trajectory **and** to its twin — the same kick to both, so any
    separation is still the world's own doing (ADR-W3, design-review-010).
    Without it, every step is the null action, as before.
    """
    if null_action is None:
        null_action = np.zeros(1)
    if actions is not None:
        actions = np.asarray(actions, dtype=float)
        if actions.ndim != 2 or actions.shape[0] != horizon:
            raise DivergenceInputError(
                f"separation_curve expects one kick per step, [{horizon}, a]; got "
                f"{actions.shape} (worlds ADR-W3)"
            )
    base = np.array(state0, dtype=float, copy=True)
    twin = perturb(base, delta0)
    curve = np.empty(horizon + 1)
    curve[0] = distance(base, twin, scale)
    for step in range(horizon):
        action = null_action if actions is None else actions[step]
        base = transition(base, action)
        twin = transition(twin, action)
        curve[step + 1] = distance(base, twin, scale)
    return curve


def separation_curves_batch(
    transition_batch: Transition,
    starts: np.ndarray,
    horizon: int,
    scale: np.ndarray,
    delta0: float,
    actions: np.ndarray,
) -> np.ndarray:
    """`separation_curve` for `n` starts at once: `[n, horizon + 1]`.

    Every start and every twin advance together through the world's
    `transition_batch`, with start `i`'s kicks (`actions[i]`, shape
    `[n, horizon, a]` overall) applied to both it and its twin. Row `i`
    equals `separation_curve(transition, starts[i], ..., actions=actions[i])`
    exactly — the batch changes the speed, not the numbers (design-review-010).
    """
    starts = np.asarray(starts, dtype=float)
    actions = np.asarray(actions, dtype=float)
    n = starts.shape[0]
    if starts.ndim != 2 or actions.ndim != 3 or actions.shape[:2] != (n, horizon):
        raise DivergenceInputError(
            f"separation_curves_batch expects starts [n, d] and kicks [n, {horizon}, a]; "
            f"got {starts.shape} and {actions.shape} (worlds ADR-W3)"
        )
    base = starts.copy()
    twin = perturb(base, delta0)
    curves = np.empty((n, horizon + 1))
    curves[:, 0] = _row_distances(base, twin, scale)
    for step in range(horizon):
        both = transition_batch(np.concatenate([base, twin]), np.concatenate([actions[:, step]] * 2))
        base, twin = both[:n], both[n:]
        curves[:, step + 1] = _row_distances(base, twin, scale)
    return curves


def _row_distances(a: np.ndarray, b: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """`distance` (worlds.base) for each row of two `[n, d]` arrays."""
    return np.array([distance(a[i], b[i], scale) for i in range(a.shape[0])])


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


def conserved_drift_batch(
    transition_batch: Transition,
    conserved: Conserved,
    starts: np.ndarray,
    horizon: int,
    null_action: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """`conserved_drift` for `n` starts at once: `(max_abs_drift[n], initial[n])`.

    The states advance together through the world's `transition_batch`
    (bit-identical to one at a time — TC-WD3-04) under null actions, and
    the conserved quantity is evaluated per state exactly as the
    one-start version does, so every figure equals `conserved_drift`'s
    for that start (design-review-010: the benchmark's third region made
    the one-at-a-time loop the slowest part of the full run).
    """
    if null_action is None:
        null_action = np.zeros(1)
    states = np.array(starts, dtype=float, copy=True)
    n = states.shape[0]
    actions = np.tile(np.asarray(null_action, dtype=float), (n, 1))
    initial = np.array([conserved(state) for state in states])
    worst = [0.0] * n
    for _ in range(horizon):
        states = transition_batch(states, actions)
        for i in range(n):
            worst[i] = max(worst[i], abs(conserved(states[i]) - initial[i]))
    return np.array(worst), initial


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


def conserved_quantity_range(
    conserved: Conserved, box: np.ndarray, target_points: int = 20_000
) -> tuple[float, float]:
    """The conserved quantity's `(min, max)` over one declared region's box.

    Deterministic and independent of any RNG or `n_starts` (design-review-009
    I1: the benchmark's per-start Monte Carlo sample is the wrong source for
    a normaliser — it entangles a measurement gate with a sampling parameter,
    and pooling several regions' samples into one range corrupts the very
    thing the gate protects, worlds ADR-W1). A fixed grid over the box —
    `target_points` spread evenly across the box's dimensions, corners
    included via `linspace`'s endpoints — needs no assumption that
    `conserved` is convex (LV's V provably is; a future world's need not be),
    at a resolution the runtime budget (NF-2) can always afford: two
    dimensions get roughly 141 points per axis, four roughly 12, and either
    is worlds below the ~1e-6 bound's own precision.
    """
    d = box.shape[0]
    per_dim = max(2, round(target_points ** (1.0 / d)))
    axes = [np.linspace(box[i, 0], box[i, 1], per_dim) for i in range(d)]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, d)
    values = np.array([conserved(point) for point in grid])
    return float(values.min()), float(values.max())
