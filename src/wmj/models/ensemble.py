"""wmj.models.ensemble — Model B, "ensemble": five networks whose disagreement is the error bar.

In plain words: five copies of the same small neural network, each started from
a different random point and shown the training examples in a different order.
Their average is the model's best guess; how much they disagree — corrected for
having only five — is its error bar. Where the five copies agree the model is
confident, where they scatter it is not. This is the other way of getting an
error bar from Model A (which predicts its own); the experiment asks whether the
two ways can be told apart by the judge even when the guesses are equally good
(models spec ADR-M3, MU-5).

**The pre-registered rules (fixed in advance, repeated in `prereg/recipe.md`).**
Point prediction: the plain average of the five members' forecasts. Error bar,
per quantity: `sqrt(1 + 1/K) × std(member forecasts, ddof = 1)` — the sample
standard deviation (Bessel's correction, `ddof = 1`), inflated by the standard
small-ensemble factor so five members' natural under-spread does not make the
model look over-confident (TC-MU5-02, TC-MU5-03).

**How each member learns.** The same architecture as Model A minus the variance
head (it outputs only the change in the state), from the shared homework the
harness hands every model (`TrainingData.train_pairs` — never a subsample of its
own, which is what keeps the five members on identical data), on mean squared
error, with the same optimiser, learning-rate decay, epochs and batch size as
Model A. Member `k` draws its starting weights from `seeds.rng("member", str(k),
"weights")` and each epoch's shuffle from `seeds.rng("member", str(k),
"shuffle", str(epoch))`, so members differ only in initialisation and ordering.
Before any learning, each member's backprop is checked once against finite
differences on 64 fixed rows (the check validates backprop under MSE).

**Units of the loss.** The spec says "mean-squared error of the state change"
without units. The network outputs the change in the state's own units (as Model
A does); the loss is the mean over examples and quantities of `((prediction −
truth) / scale)²` — exactly the held-out error metric of ADR-M3, so each
quantity counts in proportion to how much it matters in the scale the project
judges by (backlog A24).

**Prediction path.** `predict` and `predict_batch` use `MLP.forward_invariant`
and plain element-wise arithmetic, so a batch gives exactly the numbers each row
would alone (TC-MU1-04). A forecast with no width (all members identical) or a
non-finite one is refused rather than returned.
"""

from __future__ import annotations

import math

import numpy as np

from wmj.errors import WmjError
from wmj.models.base import Prediction, SeedSource, TrainingData, WorldContext
from wmj.models.direct import (
    BATCH_SIZE,
    EPOCHS,
    GRADIENT_TOLERANCE,
    HIDDEN_UNITS,
    DirectTrainingError,
    learning_rates,
    normalise_inputs,
    require_training_pairs,
)
from wmj.models.mlp import MLP, Adam, gradient_check
from wmj.models.registry import register

# Mirror of prereg/recipe.md `ensemble_members` (drift-tested; this package may not read files).
ENSEMBLE_MEMBERS = 5
SPREAD_CORRECTION = math.sqrt(1.0 + 1.0 / ENSEMBLE_MEMBERS)


class EnsembleTrainingError(WmjError):
    """Raised when Model B cannot be trained or asked to predict as specified."""


def _positive_int(name: str, value: object, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise EnsembleTrainingError(f"{name} must be an int >= {minimum}, got {value!r}")
    return value


def mse_loss_and_grad(
    Y: np.ndarray, target: np.ndarray, scale: np.ndarray
) -> tuple[float, np.ndarray]:
    """Mean squared error of the scaled change and its hand-written gradient.

    `L = mean over examples and quantities of ((Y − target)/scale)²`; the gradient
    over the outputs is `2 (Y − target) / scale² / (n d)`.
    """
    Y = np.asarray(Y, dtype=float)
    target = np.asarray(target, dtype=float)
    scale = np.asarray(scale, dtype=float)
    if Y.ndim != 2 or Y.shape != target.shape or scale.shape != (Y.shape[1],):
        raise EnsembleTrainingError(
            f"loss shape mismatch: outputs {tuple(Y.shape)}, targets {tuple(target.shape)}, "
            f"scale {tuple(scale.shape)} (need [n, d], [n, d], [d])"
        )
    if not (np.all(np.isfinite(Y)) and np.all(np.isfinite(target))):
        raise EnsembleTrainingError("loss inputs must be finite (the outputs diverged or the data is bad)")
    n, d = Y.shape
    r = (Y - target) / scale
    return float(np.mean(r**2)), 2.0 * r / scale / (n * d)


def train_member(
    ctx: WorldContext,
    seeds: SeedSource,
    training: TrainingData,
    k: int,
    *,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
) -> MLP:
    """Train ensemble member `k` (models ADR-M3) and return its network."""
    pairs = require_training_pairs(ctx, training, EnsembleTrainingError, "ensemble")
    _positive_int("epochs", epochs)
    _positive_int("batch_size", batch_size)
    if pairs.state.shape[0] == 0:
        raise EnsembleTrainingError("train_pairs is empty — there is nothing to train on")
    if training.gradcheck_index is None:
        raise EnsembleTrainingError(
            "training.gradcheck_index is missing — the one-time backprop check must run "
            "before each training run (models ADR-M3); the harness always provides it"
        )
    d = ctx.state_dim
    scale = np.asarray(ctx.scale, dtype=float)
    net = MLP(
        (d + ctx.action_dim, HIDDEN_UNITS, HIDDEN_UNITS, d), seeds.rng("member", str(k), "weights")
    )
    try:
        X = normalise_inputs(ctx, pairs.state, pairs.action)
    except DirectTrainingError as exc:
        raise EnsembleTrainingError(str(exc)) from exc
    T = pairs.next_state - pairs.state
    idx = training.gradcheck_index
    check_target = T[idx]
    gradient_check(
        net,
        X[idx],
        lambda Y: mse_loss_and_grad(Y, check_target, scale),
        tolerance=GRADIENT_TOLERANCE,
    )

    rates = learning_rates(epochs)
    adam = Adam(net.param_shapes(), lr=rates[0])
    m = X.shape[0]
    for epoch in range(epochs):
        adam.lr = rates[epoch]
        order = seeds.rng("member", str(k), "shuffle", str(epoch)).permutation(m)
        for start in range(0, m, batch_size):
            rows = order[start : start + batch_size]
            Y, cache = net.forward(X[rows])
            _loss, d_output = mse_loss_and_grad(Y, T[rows], scale)
            adam.step(net.layers, net.backward(cache, d_output))
        if not all(np.all(np.isfinite(W)) and np.all(np.isfinite(b)) for W, b in net.layers):
            raise EnsembleTrainingError(
                f"member {k}'s weights are not finite after epoch {epoch} — training diverged "
                f"(models ADR-M3)"
            )
    net.freeze()
    return net


def train_ensemble(
    ctx: WorldContext,
    seeds: SeedSource,
    training: TrainingData,
    *,
    members: int = ENSEMBLE_MEMBERS,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
) -> list[MLP]:
    """Train all `members` networks on the same data, each from its own streams."""
    _positive_int("members", members, minimum=2)  # a sample std (ddof=1) needs at least two
    require_training_pairs(ctx, training, EnsembleTrainingError, "ensemble")
    return [
        train_member(ctx, seeds, training, k, epochs=epochs, batch_size=batch_size)
        for k in range(members)
    ]


class EnsembleModel:
    """Model B: mean of the member forecasts; spread = sqrt(1 + 1/K) × their sample std."""

    name = "ensemble"
    is_fixture = False
    is_baseline = False
    stateless = True

    def __init__(self, ctx: WorldContext, nets: list[MLP]) -> None:
        if len(nets) < 2:
            raise EnsembleTrainingError(f"an ensemble needs at least 2 members, got {len(nets)}")
        self._ctx = ctx
        self._nets = list(nets)
        self._correction = math.sqrt(1.0 + 1.0 / len(self._nets))

    def reset(self) -> None:
        """Stateless: nothing to clear between rollouts."""

    def predict_batch(
        self, states: np.ndarray, actions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Row `i` is bit-identical to `predict(states[i], actions[i])` (TC-MU1-04)."""
        try:
            x = normalise_inputs(self._ctx, states, actions)
        except DirectTrainingError as exc:
            raise EnsembleTrainingError(str(exc)) from exc
        states = np.asarray(states, dtype=float)
        forecasts = [states + net.forward_invariant(x) for net in self._nets]
        k = len(forecasts)
        total = forecasts[0]
        for forecast in forecasts[1:]:
            total = total + forecast
        mean = total / k
        squares = (forecasts[0] - mean) ** 2
        for forecast in forecasts[1:]:
            squares = squares + (forecast - mean) ** 2
        spread = self._correction * np.sqrt(squares / (k - 1))
        if not (np.all(np.isfinite(mean)) and np.all(np.isfinite(spread)) and np.all(spread > 0.0)):
            raise EnsembleTrainingError(
                "the ensemble's spread is zero or not finite (all members agree exactly, or a "
                "forecast is not a number) — a forecast with no width cannot be scored "
                "(MU-1, judge ADR-J1)"
            )
        return mean, spread

    def predict(self, state: np.ndarray, action: np.ndarray) -> Prediction:
        state = np.asarray(state, dtype=float)
        action = np.asarray(action, dtype=float)
        if state.ndim != 1 or action.ndim != 1:
            raise EnsembleTrainingError(
                f"predict takes one state [d] and one action [a]; got shapes "
                f"{tuple(state.shape)} and {tuple(action.shape)} — use predict_batch for many"
            )
        means, spreads = self.predict_batch(state[None, :], action[None, :])
        return Prediction(mean=means[0].copy(), spread=spreads[0].copy())


def ensemble_factory(ctx: WorldContext, seeds: SeedSource, training: TrainingData) -> EnsembleModel:
    """factory(ctx, seeds, training) -> Model (models spec ADR-M1)."""
    return EnsembleModel(ctx, train_ensemble(ctx, seeds, training))


register("ensemble", ensemble_factory)
