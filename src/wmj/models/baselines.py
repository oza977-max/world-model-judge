"""wmj.models.baselines — persistence and linear-extrapolation baselines.

In plain words: these are the two "dumbest reasonable guesses" every
real model has to beat. Persistence says "nothing changes"; linear
says "whatever just happened, keep happening." Both are honest about
how wrong they usually are — their spread is fitted from the same
training data a real model would use, not picked to look good (models
spec ADR-M2).

**Spread fits:** both baselines' fits are public, guarded, and share
one implementation (`_fit_spread`, P2-C05 + design-review-009 C1): the
*sample* standard deviation (`ddof=1`) of one-step training changes
(persistence) or of the linear rule's own training residuals (linear)
— the training set is a sample of the world, and the same Bessel
correction is what ADR-M3's ensemble spread uses. ADR-M2 does not pin
`ddof`; the choice is recorded as backlog candidate A8 because it
changes bytes (an earlier version of this module pinned it for
persistence only, leaving linear on NumPy's `ddof=0` population
default with no guard — design-review-009 caught the asymmetry; both
now go through the same fit). A dimension that never changes in
training would fit a spread of exactly zero, which the CRPS rightly
refuses — so the fit refuses first, loudly (`DegenerateSpreadError`),
rather than emitting a model that fails at judging time.
"""

from __future__ import annotations

import numpy as np

from wmj.errors import WmjError
from wmj.models.base import Prediction, SeedSource, TrainingData, WorldContext

SPREAD_DDOF = 1  # sample std — see module docstring and backlog A8


class DegenerateSpreadError(WmjError):
    """Raised when a spread fit would be zero, non-finite, or unsupported.

    Fails loudly at fit time (models ADR-M2 / MU-1: every model states
    a usable uncertainty) instead of producing a spread the CRPS will
    reject on the first trial.
    """


class PersistenceModel:
    """mean = current state; spread = std of one-step training changes."""

    name = "persistence"
    is_fixture = False
    is_baseline = True

    def __init__(self, spread: np.ndarray) -> None:
        self._spread = spread

    def reset(self) -> None:
        pass  # no rollout-local state: persistence never remembers anything

    def predict(self, state: np.ndarray, action: np.ndarray) -> Prediction:
        return Prediction(mean=np.array(state, copy=True), spread=self._spread)


class LinearModel:
    """mean = current + (current - previous); persistence on first call."""

    name = "linear"
    is_fixture = False
    is_baseline = True

    def __init__(self, spread: np.ndarray) -> None:
        self._spread = spread
        self._previous: np.ndarray | None = None

    def reset(self) -> None:
        self._previous = None

    def predict(self, state: np.ndarray, action: np.ndarray) -> Prediction:
        if self._previous is None:
            mean = np.array(state, copy=True)
        else:
            mean = state + (state - self._previous)
        self._previous = np.array(state, copy=True)
        return Prediction(mean=mean, spread=self._spread)


def _fit_spread(residual: np.ndarray, *, rule: str) -> np.ndarray:
    """Shared per-dimension sample std (`ddof=1`) fit + fail-loud guard.

    One implementation for both baselines (design-review-009 C1: an
    earlier version pinned `ddof=1` for persistence only, leaving linear
    on NumPy's population default with no degeneracy guard — two
    baselines under one ADR silently using two different statistics).
    Refuses a fit that has fewer than two residuals to estimate a
    sample std from, or any dimension whose spread is not strictly
    positive and finite (ADR-M2; the CRPS cannot score a zero-width
    forecast, judge ADR-J1).
    """
    flat = residual.reshape(-1, residual.shape[-1])
    if flat.shape[0] < 2:
        raise DegenerateSpreadError(
            f"ADR-M2 {rule} spread: need at least 2 residuals for a "
            f"sample std (ddof={SPREAD_DDOF}), got {flat.shape[0]}"
        )
    spread = np.std(flat, axis=0, ddof=SPREAD_DDOF)
    usable = np.isfinite(spread) & (spread > 0.0)
    if not np.all(usable):
        bad = np.flatnonzero(~usable).tolist()
        raise DegenerateSpreadError(
            f"ADR-M2 {rule} spread: dimension(s) {bad} have a zero or non-finite "
            f"spread {spread.tolist()} — the CRPS cannot score a zero-width forecast "
            f"(judge ADR-J1), so the fit refuses rather than the judge"
        )
    return spread


def fit_persistence_spread(training: TrainingData) -> np.ndarray:
    """Per-dimension sample std (`ddof=1`) of one-step training changes.

    ADR-M2: "give or take how much things usually change."
    """
    changes = training.states[:, 1:, :] - training.states[:, :-1, :]
    return _fit_spread(changes, rule="persistence")


def fit_linear_spread(training: TrainingData) -> np.ndarray:
    """Per-dimension sample std (`ddof=1`) of the linear rule's own
    training residuals — the same fit and the same fail-loud guard as
    `fit_persistence_spread` (design-review-009 C1)."""
    previous = training.states[:, :-2, :]
    current = training.states[:, 1:-1, :]
    actual_next = training.states[:, 2:, :]
    predicted_next = current + (current - previous)
    residual = actual_next - predicted_next
    return _fit_spread(residual, rule="linear")


def persistence_factory(
    ctx: WorldContext, seeds: SeedSource, training: TrainingData
) -> PersistenceModel:
    """factory(ctx, seeds, training) -> Model (models spec ADR-M1)."""
    return PersistenceModel(spread=fit_persistence_spread(training))


def linear_factory(
    ctx: WorldContext, seeds: SeedSource, training: TrainingData
) -> LinearModel:
    """factory(ctx, seeds, training) -> Model (models spec ADR-M1)."""
    return LinearModel(spread=fit_linear_spread(training))
