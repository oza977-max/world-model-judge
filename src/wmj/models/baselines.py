"""wmj.models.baselines — persistence and linear-extrapolation baselines.

In plain words: these are the two "dumbest reasonable guesses" every
real model has to beat. Persistence says "nothing changes"; linear
says "whatever just happened, keep happening." Both are honest about
how wrong they usually are — their spread is fitted from the same
training data a real model would use, not picked to look good (models
spec ADR-M2).

**Minimal spread only (this chunk's own scoped decision, build/prompts/
P1-C02.md):** the spread fitted here is real (computed from training
residuals, not a placeholder), but P2-C05/P3-C02 extend/refine it —
this module is expected to grow, not to be replaced.
"""

from __future__ import annotations

import numpy as np

from wmj.models.base import Prediction, SeedSource, TrainingData, WorldContext


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


def _fit_persistence_spread(training: TrainingData) -> np.ndarray:
    """Per-dimension std of one-step state changes over training data."""
    changes = training.states[:, 1:, :] - training.states[:, :-1, :]
    return np.std(changes.reshape(-1, changes.shape[-1]), axis=0)


def _fit_linear_spread(training: TrainingData) -> np.ndarray:
    """Per-dimension std of the linear rule's own training residuals."""
    previous = training.states[:, :-2, :]
    current = training.states[:, 1:-1, :]
    actual_next = training.states[:, 2:, :]
    predicted_next = current + (current - previous)
    residual = actual_next - predicted_next
    return np.std(residual.reshape(-1, residual.shape[-1]), axis=0)


def persistence_factory(
    ctx: WorldContext, seeds: SeedSource, training: TrainingData
) -> PersistenceModel:
    """factory(ctx, seeds, training) -> Model (models spec ADR-M1)."""
    return PersistenceModel(spread=_fit_persistence_spread(training))


def linear_factory(
    ctx: WorldContext, seeds: SeedSource, training: TrainingData
) -> LinearModel:
    """factory(ctx, seeds, training) -> Model (models spec ADR-M1)."""
    return LinearModel(spread=_fit_linear_spread(training))
