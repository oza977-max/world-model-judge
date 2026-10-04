"""wmj.models.direct — Model A, "direct": a network that predicts its own error bar.

In plain words: one small neural network looks at where the world is now and
what push is applied, and says two things for each quantity: how much it
expects the quantity to change, and how wrong it expects that guess to be.
That second number — the model's own error bar — is the whole point of this
model: the experiment asks whether a model that *states* its uncertainty can
be trusted more or less than a model whose uncertainty is read off the
disagreement of several copies (Model B, the ensemble). The two are built to
be equally accurate; they differ only in how they get the error bar
(models spec ADR-M3, MU-5).

**How it learns.** From the shared homework the harness hands every model
(`TrainingData.train_pairs`, never a subsample of its own), with the
*β-NLL* loss (Seitzer et al., 2022, β = 0.5): the ordinary "Gaussian
negative log-likelihood" lets a stubbornly large error bar excuse a bad
guess, so each example's loss is weighted by `σ^(2β)`, held fixed while the
gradient is taken. The gradient is written out by hand (this project has no
automatic differentiation) and is tested against brute-force numbers. Before
any learning, the network's backprop is checked once against finite
differences on 64 fixed training rows (plain Gaussian loss — the check
validates backprop, not the loss). Learning is mini-batch (256 rows), Adam,
a fixed number of epochs with a cosine learning-rate decay (D18), no early stopping (early stopping would peek at
validation loss, a tuning channel the pre-registration exists to close).

**Units.** The network's inputs are the state divided by the world's scale
and the push divided by the half-width of the trained push range (both come
in through `WorldContext`; models never import a world). Its outputs are the
change in the state and `log σ`, in the state's own units: `mean = state +
Δmean`, `spread = exp(log σ)`. (The spec names no output scaling; recorded
as backlog A20.)

**Why the three numbers below are copied here.** `EPOCHS`, `BATCH_SIZE` and
`BETA_NLL` are the frozen recipe's `epochs`, `batch_size` and `beta_nll`.
This package may not read files (its import allowlist, cross-cutting
ADR-003), so a test compares them with `prereg/recipe.md` on every run — the
same pattern the worlds' kick settings use — and a drift fails loudly.

**Prediction path.** `predict` and `predict_batch` use `MLP.forward_invariant`,
so predicting a whole batch gives exactly the numbers predicting each row
alone would (TC-MU1-04). A spread that is zero or not finite is refused at
predict time — a forecast with no width cannot be scored.
"""

from __future__ import annotations

import math

import numpy as np

from wmj.errors import WmjError
from wmj.models.base import Prediction, SeedSource, TrainingData, WorldContext
from wmj.models.mlp import ADAM_LR, MLP, Adam, gradient_check
from wmj.models.registry import register

# Mirrors of prereg/recipe.md (drift-tested; see the module docstring).
EPOCHS = 100
BATCH_SIZE = 256
BETA_NLL = 0.5

HIDDEN_UNITS = 64  # 2 hidden layers x 64, tanh (models ADR-M3 shared architecture)
LEARNING_RATE = ADAM_LR  # the learning rate of the first epoch
LR_FINAL = 1e-5  # prereg/recipe.md `lr_final`: where the cosine decay ends (D18)
GRADIENT_TOLERANCE = 1e-5  # models ADR-M3, A10 ratified


def learning_rates(epochs: int) -> list[float]:
    """The learning rate of each epoch: a cosine decay from `LEARNING_RATE` to `LR_FINAL`.

    In plain words: big steps early, tiny steps at the end. Measured (D18): with
    a constant rate the final error of the *same* network on the *same* data
    swung by a factor of 20 from the random seed alone; the decay settles every
    run into about the same answer. Constant within an epoch; the first epoch
    uses `LEARNING_RATE` exactly and the last uses `LR_FINAL` exactly.
    """
    if epochs == 1:
        return [LEARNING_RATE]
    span = LEARNING_RATE - LR_FINAL
    return [
        LR_FINAL + 0.5 * span * (1.0 + math.cos(math.pi * e / (epochs - 1))) for e in range(epochs)
    ]


class DirectTrainingError(WmjError):
    """Raised when Model A cannot be trained or asked to predict as specified.

    In plain words: the build or the prediction stops, saying which rule was
    broken (missing training pairs, mismatched widths, a loss that stopped
    being a finite number, a forecast with no width) rather than carrying on
    with numbers nobody can trust.
    """


def _require_finite_2d(name: str, array: np.ndarray, width: int | None = None) -> np.ndarray:
    array = np.asarray(array, dtype=float)
    if array.ndim != 2 or (width is not None and array.shape[1] != width):
        raise DirectTrainingError(
            f"{name} must have shape [n, {width if width is not None else 'k'}], got "
            f"{tuple(array.shape)}"
        )
    if not np.all(np.isfinite(array)):
        raise DirectTrainingError(f"{name} must hold only finite numbers")
    return array


def normalise_inputs(ctx: WorldContext, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
    """`[state / scale, action / half-width]` — the network's input (models ADR-M3)."""
    states = _require_finite_2d("states", states, ctx.state_dim)
    actions = _require_finite_2d("actions", actions, ctx.action_dim)
    if states.shape[0] != actions.shape[0]:
        raise DirectTrainingError(
            f"states and actions disagree on row count: shape {states.shape} vs {actions.shape}"
        )
    interval = ctx.training_action_interval
    half_width = (interval[:, 1] - interval[:, 0]) / 2.0
    return np.hstack([states / ctx.scale, actions / half_width])


def beta_nll_loss_and_grad(
    Y: np.ndarray, target: np.ndarray, *, beta: float
) -> tuple[float, np.ndarray]:
    """β-NLL and its hand-written gradient over the network outputs (models ADR-M3).

    `Y = [μ | s]` with `s = log σ`, `target` the observed change `y`. With
    `r = y − μ` and the weight `w = σ^(2β)` computed from this forward pass
    and held constant: `L = w·[0.5·log 2π + s + r²/(2σ²)]`, summed over the
    dimensions and averaged over the batch; `∂L/∂μ = −w·r/σ²`,
    `∂L/∂s = w·(1 − r²/σ²)`, each divided by the batch size. `beta = 0` is
    the plain Gaussian NLL (`w = 1`).
    """
    Y = np.asarray(Y, dtype=float)
    target = np.asarray(target, dtype=float)
    if target.ndim != 2 or Y.shape != (target.shape[0], 2 * target.shape[1]):
        raise DirectTrainingError(
            f"outputs must have shape [n, 2d] for targets [n, d]; got outputs "
            f"{tuple(Y.shape)} and targets {tuple(target.shape)}"
        )
    if not np.all(np.isfinite(Y)) or not np.all(np.isfinite(target)):
        raise DirectTrainingError("loss inputs must be finite (the outputs diverged or the data is bad)")
    n, d = target.shape
    mu, s = Y[:, :d], Y[:, d:]
    r = target - mu
    with np.errstate(over="ignore", invalid="ignore"):  # non-finite results are refused below
        inv_var = np.exp(-2.0 * s)
        w = np.exp(2.0 * beta * s)  # sigma ** (2 beta), held constant below
        sq = r**2 * inv_var
        loss = float((w * (0.5 * math.log(2.0 * math.pi) + s + 0.5 * sq)).sum(axis=1).mean())
        d_mu = -w * r * inv_var / n
        d_s = w * (1.0 - sq) / n
    if not (math.isfinite(loss) and np.all(np.isfinite(d_mu)) and np.all(np.isfinite(d_s))):
        raise DirectTrainingError("the loss or its gradient is not finite (training diverged)")
    return loss, np.hstack([d_mu, d_s])


def gaussian_nll_loss_and_grad(Y: np.ndarray, target: np.ndarray) -> tuple[float, np.ndarray]:
    """Plain Gaussian NLL (`w = 1`): what the one-time gradient check differentiates."""
    return beta_nll_loss_and_grad(Y, target, beta=0.0)


def require_training_pairs(
    ctx: WorldContext,
    training: TrainingData,
    error: type[WmjError] = DirectTrainingError,
    who: str = "direct",
):
    """The harness's shared training pairs, or a named refusal (shared with the ensemble)."""
    pairs = training.train_pairs
    if pairs is None:
        raise error(
            f"{who} needs training.train_pairs (the harness's shared subsample); this "
            f"TrainingData has none — the MLP models never draw a subsample of their own "
            f"(models ADR-M1, ADR-M3)"
        )
    if pairs.state.shape[1] != ctx.state_dim or pairs.action.shape[1] != ctx.action_dim:
        raise error(
            f"width mismatch: the world context says state {ctx.state_dim} / action "
            f"{ctx.action_dim} but train_pairs has state {pairs.state.shape[1]} / action "
            f"{pairs.action.shape[1]}"
        )
    return pairs


def train_direct(
    ctx: WorldContext,
    seeds: SeedSource,
    training: TrainingData,
    *,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    beta: float = BETA_NLL,
) -> MLP:
    """Train Model A's network (models ADR-M3) and return it.

    Order: draw the weights from `seeds.rng("weights")`; check backprop once
    on the 64 `gradcheck_index` rows under plain NLL (before any update);
    then `epochs` passes of mini-batch Adam, each epoch shuffled from
    `seeds.rng("shuffle", str(epoch))`. Weights that stop being finite abort
    the run.
    """
    pairs = require_training_pairs(ctx, training)
    if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs < 1:
        raise DirectTrainingError(f"epochs must be a positive int, got {epochs!r}")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise DirectTrainingError(f"batch_size must be a positive int, got {batch_size!r}")
    if not (isinstance(beta, (int, float)) and math.isfinite(beta) and beta >= 0.0):
        raise DirectTrainingError(f"beta must be a finite number >= 0, got {beta!r}")
    d, a = ctx.state_dim, ctx.action_dim
    net = MLP((d + a, HIDDEN_UNITS, HIDDEN_UNITS, 2 * d), seeds.rng("weights"))

    if pairs.state.shape[0] == 0:
        raise DirectTrainingError("train_pairs is empty — there is nothing to train on")
    if training.gradcheck_index is None:
        raise DirectTrainingError(
            "training.gradcheck_index is missing — the one-time backprop check must run "
            "before each training run (models ADR-M3); the harness always provides it"
        )
    X = normalise_inputs(ctx, pairs.state, pairs.action)
    T = pairs.next_state - pairs.state
    idx = training.gradcheck_index
    check_target = T[idx]
    gradient_check(
        net,
        X[idx],
        lambda Y: gaussian_nll_loss_and_grad(Y, check_target),
        tolerance=GRADIENT_TOLERANCE,
    )

    rates = learning_rates(epochs)
    adam = Adam(net.param_shapes(), lr=rates[0])
    m = X.shape[0]
    for epoch in range(epochs):
        adam.lr = rates[epoch]
        order = seeds.rng("shuffle", str(epoch)).permutation(m)
        for start in range(0, m, batch_size):
            rows = order[start : start + batch_size]
            Y, cache = net.forward(X[rows])
            _loss, d_output = beta_nll_loss_and_grad(Y, T[rows], beta=beta)
            adam.step(net.layers, net.backward(cache, d_output))
        if not all(np.all(np.isfinite(W)) and np.all(np.isfinite(b)) for W, b in net.layers):
            raise DirectTrainingError(
                f"the network's weights are not finite after epoch {epoch} — training diverged "
                f"(models ADR-M3)"
            )
    return net


class DirectModel:
    """Model A: mean = state + predicted change; spread = exp(predicted log σ)."""

    name = "direct"
    is_fixture = False
    is_baseline = False
    stateless = True

    def __init__(self, ctx: WorldContext, net: MLP) -> None:
        self._ctx = ctx
        self._net = net

    def reset(self) -> None:
        """Stateless: nothing to clear between rollouts."""

    def predict_batch(
        self, states: np.ndarray, actions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Row `i` is bit-identical to `predict(states[i], actions[i])` (TC-MU1-04)."""
        d = self._ctx.state_dim
        states = _require_finite_2d("states", states, d)
        out = self._net.forward_invariant(normalise_inputs(self._ctx, states, actions))
        means = states + out[:, :d]
        spreads = np.exp(out[:, d:])
        if not (np.all(np.isfinite(means)) and np.all(np.isfinite(spreads)) and np.all(spreads > 0.0)):
            raise DirectTrainingError(
                "the predicted mean or spread is not finite, or the spread is zero — a forecast with no width cannot "
                "be scored (MU-1, judge ADR-J1)"
            )
        return means, spreads

    def predict(self, state: np.ndarray, action: np.ndarray) -> Prediction:
        state = np.asarray(state, dtype=float)
        action = np.asarray(action, dtype=float)
        if state.ndim != 1 or action.ndim != 1:
            raise DirectTrainingError(
                f"predict takes one state [d] and one action [a]; got shapes "
                f"{tuple(state.shape)} and {tuple(action.shape)} — use predict_batch for many"
            )
        means, spreads = self.predict_batch(state[None, :], action[None, :])
        return Prediction(mean=means[0].copy(), spread=spreads[0].copy())


_LAST_CORE: tuple | None = None  # (training, run_seed, context key, trained network)


def _context_key(ctx: WorldContext) -> tuple:
    return (
        ctx.world_name, ctx.state_dim, ctx.action_dim, ctx.scale.tobytes(),
        ctx.training_state_box.tobytes(), ctx.training_action_interval.tobytes(),
    )


def shared_direct_core(ctx: WorldContext, seeds: SeedSource, training: TrainingData) -> MLP:
    """The trained network that `direct` and every fixture built on it share.

    In plain words: the three deliberately broken models are each "Model A with
    one thing broken" (models ADR-M4), so each needs Model A's trained network.
    Training it four times over would cost minutes and prove nothing — it is the
    same computation from the same seed on the same data. This trains it once
    per (run seed, world, data object) and hands the same network to everyone
    (the harness gives every factory the very same `TrainingData` object, ADR-M1).
    It is keyed on the name `"direct"` — whatever the caller is called — so a
    fixture's core is bit-identical to the registered `direct`'s (TC-MU4-02); the
    network is never modified after training, only read. One slot: the most
    recent combination, so nothing accumulates.
    """
    global _LAST_CORE
    key = (seeds.run_seed, _context_key(ctx))
    if _LAST_CORE is not None and _LAST_CORE[0] is training and _LAST_CORE[1] == key:
        return _LAST_CORE[2]
    net = train_direct(ctx, SeedSource(seeds.run_seed, "direct"), training)
    _LAST_CORE = (training, key, net)
    return net


def direct_factory(ctx: WorldContext, seeds: SeedSource, training: TrainingData) -> DirectModel:
    """factory(ctx, seeds, training) -> Model (models ADR-M1)."""
    return DirectModel(ctx, shared_direct_core(ctx, seeds, training))


register("direct", direct_factory)
