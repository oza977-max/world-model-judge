"""wmj.models.mlp — the hand-rolled MLP both unrigged models share (ADR-M3).

In plain words: a small neural network written out by hand in NumPy —
no deep-learning library — so every number it produces is ours to
explain and ours to reproduce exactly. Two of the models under test
(the "direct" one and the five-member "ensemble") are built on this
same core; they differ only in what sits on top of it, which is the
whole point of the experiment (models ADR-M3).

The load-bearing piece is the **gradient check**: a hand-written
backprop is only trustworthy if it agrees with finite differences, so
this module ships a checker that compares the two and refuses (loudly)
if they disagree. That check is this file's own first test — written
before the backprop it validates (models §8, Beck's failing-test-first).

This is the core only: the architecture, the optimiser, and the check.
The task-specific heads and losses — direct's error-bar output and its
Gaussian-NLL training, the ensemble members' mean-only training — are
built on top of this at P3-C03/P3-C04.

Determinism (cross-cutting ADR-002): every weight is drawn from a
`Generator` the caller passes in, never from NumPy's global RNG, so a
network is byte-identical run to run. Unlike the frozen value types in
`models.base`, an MLP's weights are deliberately *mutable* — the
optimiser writes new values into them in place every step.
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

from wmj.errors import WmjError

# A loss the check/optimiser differentiate: output -> (scalar loss, d/d output).
LossAndGrad = Callable[[np.ndarray], "tuple[float, np.ndarray]"]

# ADR-M3's pinned Adam hyperparameters (two implementers picking different
# conventional defaults would get bit-different trained weights under NF-1).
ADAM_LR = 1e-3
ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.999
ADAM_EPS = 1e-8

# Gradient-check numerics (spec correction A10 — see gradient_check's docstring
# and build/spec-corrections-backlog.md). A relative error is only meaningful
# where the gradient is large enough that float64 cancellation in
# f(x+eps) - f(x-eps) does not dominate. Parameters whose gradient is below
# GRADIENT_SCALE_FLOOR times the network's largest gradient are numerically
# indistinguishable from zero for this check; they are held to absolute
# agreement at that floor rather than to their own (near-zero) magnitude,
# which the naive relative-error formula divides by and so misreports.
GRADIENT_SCALE_FLOOR = 1e-3


class MLPArchitectureError(WmjError):
    """Raised for a layer specification the MLP cannot build or run.

    Fewer than two layer sizes (no weight layer at all), or a forward
    pass whose input width does not match the first layer — a wrong
    shape caught at the boundary, not as a downstream broadcast error
    (McConnell, models ADR-M3).
    """


class GradientCheckError(WmjError):
    """Raised when hand-written backprop disagrees with finite differences.

    The gradient check exists to catch exactly this; if the maximum
    relative error over all parameters exceeds the tolerance, the
    backprop is wrong and the network refuses to be trusted (models §8,
    pre-TC discipline).
    """


class MLP:
    """A tanh MLP with a linear output layer, weights from a seeded RNG.

    `layer_sizes = [in, h1, ..., out]` — `len - 1` weight layers. Every
    hidden layer applies tanh; the output layer is linear (its outputs
    are unbounded quantities: a state change, a log-spread, a mean).
    """

    def __init__(self, layer_sizes: Sequence[int], rng: np.random.Generator) -> None:
        sizes = list(layer_sizes)
        if len(sizes) < 2:
            raise MLPArchitectureError(
                f"MLP needs at least an input and an output size (>= 2 entries), "
                f"got {sizes} (models ADR-M3)"
            )
        if any(n < 1 for n in sizes):
            # A zero- or negative-width layer has no units; caught here with
            # a clear message rather than surfacing later as an opaque NumPy
            # `1/sqrt(0) = inf` / `uniform(-inf, inf)` OverflowError (McConnell).
            raise MLPArchitectureError(
                f"MLP layer sizes must all be >= 1 (a layer must have at least "
                f"one unit), got {sizes} (models ADR-M3)"
            )
        self.layer_sizes = tuple(sizes)
        # W ~ U(-1/sqrt(fan_in), 1/sqrt(fan_in)); biases 0 (ADR-M3).
        self.layers: list[tuple[np.ndarray, np.ndarray]] = []
        for fan_in, fan_out in zip(sizes[:-1], sizes[1:]):
            bound = 1.0 / np.sqrt(fan_in)
            W = rng.uniform(-bound, bound, size=(fan_in, fan_out))
            b = np.zeros(fan_out)
            self.layers.append((W, b))

    @property
    def input_dim(self) -> int:
        return self.layer_sizes[0]

    def param_shapes(self) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
        """The `(W.shape, b.shape)` per layer — the optimiser's moment shapes."""
        return [(W.shape, b.shape) for W, b in self.layers]

    def forward(self, X: np.ndarray) -> tuple[np.ndarray, dict]:
        """`X: float64[batch, in]` -> `(output[batch, out], cache)`.

        The cache holds each layer's input and, for the hidden layers, the
        tanh activation — everything `backward` needs to form the tanh
        derivative `1 - tanh^2` without recomputing it from the weights.
        (The reverse pass still uses each layer's *current* weight matrix
        to propagate the upstream gradient — `d_pre @ W.T` — so `backward`
        must run before any `Adam.step` updates those weights. That
        ordering is inherent to backprop, not removable by caching; see
        `backward`.)
        """
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.input_dim:
            raise MLPArchitectureError(
                f"MLP forward expected input of width {self.input_dim}, got array of "
                f"shape {X.shape} (models ADR-M3)"
            )
        inputs: list[np.ndarray] = []
        # Tanh activation per hidden layer; None in the output layer's slot
        # (linear, no activation to differentiate).
        hidden_outputs: list[np.ndarray | None] = []
        activation = X
        last = len(self.layers) - 1
        for index, (W, b) in enumerate(self.layers):
            inputs.append(activation)
            pre = activation @ W + b
            if index == last:
                activation = pre
                hidden_outputs.append(None)
            else:
                activation = np.tanh(pre)
                hidden_outputs.append(activation)
        return activation, {"inputs": inputs, "hidden_outputs": hidden_outputs}

    def backward(self, cache: dict, d_output: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
        """Reverse-mode gradients given the upstream gradient `d_output`.

        Returns `[(dW, db), ...]` in forward layer order. Loss-agnostic:
        `d_output` is dL/d(output) for whatever loss sits on top.

        **Ordering contract:** call `forward` then `backward` for the same
        inputs *before* any `Adam.step` updates the weights — the gradient
        of an early layer flows through later layers' current `W`
        (`d_pre @ W.T`), so a weight updated between forward and backward
        would corrupt the gradient silently. This is standard backprop
        discipline; the training loop (P3-C03/C04) honours it per iteration.
        """
        inputs = cache["inputs"]
        hidden_outputs = cache["hidden_outputs"]
        d_activation = np.asarray(d_output, dtype=float)
        grads: list[tuple[np.ndarray, np.ndarray]] = [None] * len(self.layers)  # type: ignore[list-item]
        last = len(self.layers) - 1
        for index in reversed(range(len(self.layers))):
            W, _b = self.layers[index]
            layer_input = inputs[index]
            if index != last:
                # Undo tanh: d/d(pre) = d_activation * (1 - tanh(pre)^2).
                # tanh(pre) is this layer's cached output — no recompute,
                # so this does not depend on the weights' current value.
                tanh_out = hidden_outputs[index]
                d_pre = d_activation * (1.0 - tanh_out**2)
            else:
                d_pre = d_activation
            dW = layer_input.T @ d_pre
            db = d_pre.sum(axis=0)
            grads[index] = (dW, db)
            d_activation = d_pre @ W.T
        return grads


class Adam:
    """The pinned ADR-M3 Adam optimiser; `step` updates weights in place."""

    def __init__(
        self,
        param_shapes: Sequence[tuple[tuple[int, ...], tuple[int, ...]]],
        lr: float = ADAM_LR,
        beta1: float = ADAM_BETA1,
        beta2: float = ADAM_BETA2,
        eps: float = ADAM_EPS,
    ) -> None:
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        self._m = [(np.zeros(Ws), np.zeros(bs)) for Ws, bs in param_shapes]
        self._v = [(np.zeros(Ws), np.zeros(bs)) for Ws, bs in param_shapes]

    def step(
        self,
        params: list[tuple[np.ndarray, np.ndarray]],
        grads: Sequence[tuple[np.ndarray, np.ndarray]],
    ) -> None:
        """One Adam update of every `(W, b)` in `params` in place."""
        self.t += 1
        bias1 = 1.0 - self.beta1**self.t
        bias2 = 1.0 - self.beta2**self.t
        for layer, (W, b) in enumerate(params):
            for slot, param, grad in (
                (0, W, grads[layer][0]),
                (1, b, grads[layer][1]),
            ):
                m = self._m[layer][slot]
                v = self._v[layer][slot]
                m *= self.beta1
                m += (1.0 - self.beta1) * grad
                v *= self.beta2
                v += (1.0 - self.beta2) * grad**2
                m_hat = m / bias1
                v_hat = v / bias2
                param -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


def gradient_check(
    mlp: MLP,
    X: np.ndarray,
    loss_and_grad: LossAndGrad,
    *,
    epsilon: float = 1e-5,
    tolerance: float = 1e-5,
) -> float:
    """Max relative error between analytic backprop and finite differences.

    For every weight and bias element, perturb it by ±`epsilon`, recompute
    the scalar loss through `forward` + `loss_and_grad`, form the central
    finite difference, and compare to the analytic gradient. Returns the
    maximum relative error; raises `GradientCheckError` if it exceeds
    `tolerance` (models §8). The relative error is floored per
    `GRADIENT_SCALE_FLOOR` so a near-zero gradient — where relative error
    is mathematically ill-conditioned — is judged by absolute agreement,
    not misreported (see below).

    **The perturbation step is `1e-5`, not the tolerance (Goldberg).** A
    central finite difference has two competing errors — truncation,
    which grows as `epsilon^2`, and float64 rounding, which grows as
    `~1e-16 / epsilon`. Their sum is minimised near `1e-5` for float64;
    a smaller step pushes into the rounding-dominated regime and inflates
    the measured error (executed sweep, P3-C01: a 4-16-16-3 net reads
    `4.2e-8` at `1e-5`, `3.1e-7` at `1e-6`, `2.6e-6` at `1e-7` — a clean
    U-curve with its minimum at `1e-5`). The step is a numerical choice;
    the tolerance is the accept threshold.

    **Two spec corrections, both recorded as A10 in
    `build/spec-corrections-backlog.md` for Round 10 design review:**

    1. **The metric is floored against a near-zero-gradient artifact.**
       The naive `|a - num| / (|a| + |num|)` relative error (Karpathy,
       Stanford CS231n) blows up for a parameter whose true gradient is
       near zero: on the real ADR-M3 net (2 hidden × 64) a weight whose
       gradient is `2.6e-8` — with analytic and numeric agreeing to four
       significant figures — reads `1.8e-4` relative, purely because a
       negligible `~1e-11` absolute difference is divided by a near-zero
       denominator. That is a defect in the *measurement*, not the
       backprop. The fix (standard practice): floor the denominator at
       `GRADIENT_SCALE_FLOOR` (`1e-3`) times the network's largest
       gradient, so a parameter below 0.1% of the peak gradient is held
       to absolute agreement at the peak scale. The floor is scale-free
       — multiply the loss by any constant and every reading is unchanged
       (the `1e-12` ultimate guard aside, which only bites on a genuinely
       all-zero-gradient loss) — so it carries to the NLL/mean losses of
       P3-C03/C04 unchanged. **The tradeoff, disclosed:** the floor buys
       robustness by giving up sensitivity to a bug that is *localised to
       a sub-0.1%-of-peak parameter* — a 5× error on a `1e-9` gradient
       reads `~4e-6` and passes. This is acceptable because such a
       parameter contributes negligibly to the actual optimisation step,
       and a *structural* backprop bug (wrong formula) corrupts the peak
       parameters too and is caught there; but it is a genuine blind spot
       for near-dead units, and Round 10 should weigh it against a
       numpy-`allclose`-style combined criterion (A10).

    2. **The tolerance default is `1e-5`, not the spec's `1e-6`.** Even
       with the metric floored, a *correct* hand-rolled central-difference
       check on a 3-weight-layer net floors at a few `1e-7` (240-net
       sweep across both worlds' real I/O shapes: worst `3.6e-7`). `1e-6`
       leaves only ~3x margin — too tight for a gate that must never
       flake, especially on trained nets downstream whose gradients drive
       closer to zero. `1e-5` gives ~30x margin over correct-backprop
       noise and still sits four orders of magnitude below any real bug:
       the phantom-gate ×1.5 corruption reads `0.2`. `1e-6` is a spec
       number falsified by execution, the same class as A7/A8; Round 10
       owns the spec text, the code uses `1e-5` until then.
    """
    if epsilon <= 0:
        raise GradientCheckError(
            f"MLP gradient check: epsilon (the finite-difference step) must be "
            f"positive, got {epsilon} (models ADR-M3 §8)"
        )

    output, cache = mlp.forward(X)
    _loss, d_output = loss_and_grad(output)
    analytic = mlp.backward(cache, d_output)

    # Characteristic gradient scale from the analytic gradients (known
    # before any perturbation): the largest gradient magnitude anywhere in
    # the network. A non-finite analytic gradient is itself a definitive
    # backprop failure, caught explicitly here — a bare `max()` would
    # silently DROP a NaN that is not its first argument (Python compares
    # with `>`, and every comparison against NaN is False), reporting a
    # broken backprop as verified. That silent NaN is the exact failure
    # this check exists to prevent (McConnell).
    scale = 0.0
    for layer_index, (dW, db) in enumerate(analytic):
        for kind, grad in (("weight", dW), ("bias", db)):
            if not np.all(np.isfinite(grad)):
                raise GradientCheckError(
                    f"MLP gradient check: backprop produced a non-finite (NaN/inf) "
                    f"{kind} gradient at layer {layer_index} — a broken backprop, "
                    f"refusing to trust it (models ADR-M3, §8)"
                )
            scale = max(scale, float(np.max(np.abs(grad))))
    denom_floor = GRADIENT_SCALE_FLOOR * scale

    def scalar_loss() -> float:
        out, _ = mlp.forward(X)
        return loss_and_grad(out)[0]

    max_rel_error = 0.0
    for layer_index, params in enumerate(mlp.layers):
        for slot, param in enumerate(params):
            analytic_param = analytic[layer_index][slot]
            # Index the live parameter array directly (np.ndindex over its
            # shape), never a raveled view: an in-place `param[idx] = ...`
            # always writes back to the network's own weights, with no
            # dependence on the array being C-contiguous.
            for idx in np.ndindex(param.shape):
                original = float(param[idx])
                try:
                    param[idx] = original + epsilon
                    loss_plus = scalar_loss()
                    param[idx] = original - epsilon
                    loss_minus = scalar_loss()
                finally:
                    # Always restore, even if loss_and_grad raised — never
                    # leave the caller's live weight perturbed.
                    param[idx] = original
                numeric = (loss_plus - loss_minus) / (2.0 * epsilon)
                a = float(analytic_param[idx])
                # denom_floor handles the near-zero-gradient artifact; the
                # 1e-12 term is the ultimate guard against a 0/0 on a
                # degenerate all-zero-gradient loss (scale == 0).
                denom = max(denom_floor, abs(a) + abs(numeric), 1e-12)
                rel = abs(a - numeric) / denom
                if not np.isfinite(rel):
                    # analytic was checked finite and denom >= 1e-12, so a
                    # non-finite rel means the finite-difference loss itself
                    # blew up — again a silent NaN this check must not pass.
                    raise GradientCheckError(
                        f"MLP gradient check: non-finite (NaN/inf) comparison at "
                        f"layer {layer_index} slot {slot} index {idx} — the "
                        f"finite-difference loss is non-finite (models ADR-M3, §8)"
                    )
                max_rel_error = max(max_rel_error, rel)

    if max_rel_error > tolerance:
        raise GradientCheckError(
            f"MLP gradient check: max relative error {max_rel_error:.3e} exceeds "
            f"tolerance {tolerance:.1e} — the hand-rolled backprop disagrees with "
            f"finite differences; refusing to trust it (models ADR-M3, §8)"
        )
    return max_rel_error
