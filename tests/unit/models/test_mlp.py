"""Tests for wmj.models.mlp — the hand-rolled MLP core (models ADR-M3).

The gradient check is the MLP's *first* test (models §8's pre-TC
discipline, Beck): the hand-rolled backprop is worthless if it doesn't
match finite differences, so that check is written and Red before
`backward` exists. A phantom-gate pairing proves the check can actually
fail — a gradient checker that never rejects a wrong gradient is no
check at all.
"""

from __future__ import annotations

import numpy as np
import pytest

from wmj.models.mlp import (
    MLP,
    Adam,
    GradientCheckError,
    MLPArchitectureError,
    gradient_check,
)


def _mse_loss_and_grad(target: np.ndarray):
    """Return a `loss_and_grad(output) -> (loss, d_output)` for MSE.

    L = 0.5 * mean over batch of sum_d (output - target)^2; its gradient
    w.r.t. the output is (output - target) / batch. A plain, convex
    scalar loss — enough to validate backprop through the linear+tanh
    layers independent of any task-specific head.
    """

    def loss_and_grad(output: np.ndarray):
        batch = output.shape[0]
        diff = output - target
        loss = 0.5 * float(np.sum(diff**2)) / batch
        return loss, diff / batch

    return loss_and_grad


# --- The gradient check comes first (models §8) ---


def test_gradient_check_passes_on_correct_backprop():
    # Asserts the check's own contract (the spec-corrected 1e-5 accept
    # threshold — see the gradient_check docstring / backlog A10), not a
    # tighter number coupled to this net's float64 finite-difference noise.
    # This small net in fact reads ~1.6e-8, far inside; the phantom gate
    # below is what proves the check discriminates.
    rng = np.random.default_rng(0)
    mlp = MLP([3, 8, 8, 2], rng)
    X = rng.standard_normal((5, 3))
    target = rng.standard_normal((5, 2))
    max_rel_error = gradient_check(mlp, X, _mse_loss_and_grad(target))
    assert max_rel_error < 1e-5


def test_gradient_check_raises_when_the_analytic_gradient_is_wrong():
    """Phantom-gate: a corrupted analytic gradient must be rejected, or
    the check proves nothing (models §8 / Beck)."""
    rng = np.random.default_rng(1)
    mlp = MLP([3, 8, 8, 2], rng)
    X = rng.standard_normal((5, 3))
    target = rng.standard_normal((5, 2))

    original_backward = mlp.backward

    def corrupted_backward(cache, d_output):
        grads = original_backward(cache, d_output)
        # Scale the first layer's weight gradient — now analytic != numeric.
        (dW0, db0), *rest = grads
        return [(dW0 * 1.5, db0), *rest]

    mlp.backward = corrupted_backward  # type: ignore[method-assign]
    with pytest.raises(GradientCheckError):
        gradient_check(mlp, X, _mse_loss_and_grad(target))


def test_gradient_check_raises_on_a_nan_analytic_gradient():
    """A NaN in the analytic gradient must be caught, not silently dropped.

    Python's builtin `max()` discards a NaN that is not its first argument
    (every `>` comparison against NaN is False), so a naive max-reduction
    could report a NaN-corrupted backprop as verified. The check guards
    the analytic gradient's finiteness explicitly — this proves it fires
    (the silent-NaN failure the gradient check exists to prevent)."""
    rng = np.random.default_rng(11)
    mlp = MLP([3, 8, 8, 2], rng)
    X = rng.standard_normal((5, 3))
    target = rng.standard_normal((5, 2))

    original_backward = mlp.backward

    def nan_backward(cache, d_output):
        grads = original_backward(cache, d_output)
        (dW0, db0), *rest = grads
        dW0 = dW0.copy()
        dW0.flat[0] = np.nan  # a single NaN, not in the reduction's first slot
        return [(dW0, db0), *rest]

    mlp.backward = nan_backward  # type: ignore[method-assign]
    with pytest.raises(GradientCheckError):
        gradient_check(mlp, X, _mse_loss_and_grad(target))


def test_gradient_check_raises_when_the_finite_difference_loss_is_non_finite():
    """A non-finite finite-difference loss (the regime the upcoming
    Gaussian-NLL head can enter on an untrained net) must raise, not pass.

    The analytic gradient is finite here — the loss *value* blows up — so
    this exercises the in-loop non-finite guard, distinct from the
    NaN-analytic path above."""
    rng = np.random.default_rng(12)
    mlp = MLP([3, 8, 8, 2], rng)
    X = rng.standard_normal((5, 3))
    target = rng.standard_normal((5, 2))
    base = _mse_loss_and_grad(target)

    def inf_loss_and_grad(output):
        _loss, grad = base(output)
        return float("inf"), grad  # finite gradient, non-finite loss value

    with pytest.raises(GradientCheckError):
        gradient_check(mlp, X, inf_loss_and_grad)


def test_gradient_check_restores_weights_when_the_loss_raises():
    """If `loss_and_grad` raises mid-sweep, the perturbed weight must be
    restored (try/finally) — the caller's live network is never left with
    a stray one-epsilon offset."""
    rng = np.random.default_rng(13)
    mlp = MLP([3, 8, 2], rng)
    X = rng.standard_normal((4, 3))
    before = [(W.copy(), b.copy()) for W, b in mlp.layers]

    calls = {"n": 0}

    def exploding_loss_and_grad(output):
        calls["n"] += 1
        if calls["n"] == 3:  # blow up partway through the FD sweep
            raise RuntimeError("boom")
        batch = output.shape[0]
        diff = output
        return 0.5 * float(np.sum(diff**2)) / batch, diff / batch

    with pytest.raises(RuntimeError):
        gradient_check(mlp, X, exploding_loss_and_grad)

    for (W_after, b_after), (W_before, b_before) in zip(mlp.layers, before):
        assert np.array_equal(W_after, W_before)
        assert np.array_equal(b_after, b_before)


# --- Architecture, init, determinism ---


def test_weights_are_fan_in_scaled_and_biases_zero():
    rng = np.random.default_rng(2)
    mlp = MLP([4, 64, 64, 3], rng)
    layers = mlp.layers
    assert [(W.shape, b.shape) for W, b in layers] == [
        ((4, 64), (64,)),
        ((64, 64), (64,)),
        ((64, 3), (3,)),
    ]
    for (W, b), fan_in in zip(layers, (4, 64, 64)):
        bound = 1.0 / np.sqrt(fan_in)
        assert np.all(np.abs(W) <= bound)
        assert np.array_equal(b, np.zeros_like(b))


def test_two_mlps_from_the_same_seed_are_weight_identical():
    a = MLP([3, 8, 2], np.random.default_rng(7))
    b = MLP([3, 8, 2], np.random.default_rng(7))
    for (Wa, ba), (Wb, bb) in zip(a.layers, b.layers):
        assert np.array_equal(Wa, Wb) and np.array_equal(ba, bb)


def test_two_mlps_from_different_seeds_differ():
    a = MLP([3, 8, 2], np.random.default_rng(7))
    b = MLP([3, 8, 2], np.random.default_rng(8))
    assert not np.array_equal(a.layers[0][0], b.layers[0][0])


def test_architecture_needs_at_least_one_layer():
    with pytest.raises(MLPArchitectureError):
        MLP([5], np.random.default_rng(0))


def test_architecture_rejects_a_zero_width_layer():
    # A config typo giving a 0-unit layer must raise the module's own error,
    # not an opaque NumPy OverflowError from 1/sqrt(0).
    with pytest.raises(MLPArchitectureError):
        MLP([3, 0, 2], np.random.default_rng(0))


def test_forward_output_shape_is_batch_by_output_dim():
    rng = np.random.default_rng(3)
    mlp = MLP([2, 16, 16, 4], rng)
    output, _cache = mlp.forward(rng.standard_normal((7, 2)))
    assert output.shape == (7, 4)


def test_forward_rejects_wrong_input_width():
    rng = np.random.default_rng(3)
    mlp = MLP([2, 16, 4], rng)
    with pytest.raises(MLPArchitectureError):
        mlp.forward(rng.standard_normal((7, 3)))  # width 3, expected 2


# --- Adam ---


def test_adam_step_reduces_a_convex_quadratic_loss():
    """One optimiser wired end-to-end against the MLP's own backprop:
    a few Adam steps on MSE must strictly reduce the loss."""
    rng = np.random.default_rng(5)
    mlp = MLP([3, 8, 8, 2], rng)
    X = rng.standard_normal((16, 3))
    target = rng.standard_normal((16, 2))
    loss_and_grad = _mse_loss_and_grad(target)
    adam = Adam(mlp.param_shapes())

    def current_loss() -> float:
        output, _ = mlp.forward(X)
        return loss_and_grad(output)[0]

    before = current_loss()
    for _ in range(20):
        output, cache = mlp.forward(X)
        _, d_output = loss_and_grad(output)
        grads = mlp.backward(cache, d_output)
        adam.step(mlp.layers, grads)
    after = current_loss()
    assert after < before


def test_adam_hyperparameters_match_adr_m3():
    adam = Adam([((2, 2), (2,))])
    assert (adam.lr, adam.beta1, adam.beta2, adam.eps) == (1e-3, 0.9, 0.999, 1e-8)


# --- Realistic-fixture variant (TDD-3, numerical domain) ---


def test_gradient_check_holds_on_a_single_example_batch():
    """A batch of exactly one row — the shape a happy-path multi-row
    batch would miss (does the batch-reduction axis behave with one
    example?). TDD-3 practitioner-named shape; not one of the six
    catalogue domains.

    Note (recorded in the handover): the finite-difference *reference*
    is only accurate at the data's own scale. Inputs to this MLP are
    always normalised to O(1) (state/scale, action/half-width — ADR-M3),
    so this variant uses a mild ±3 range within that regime, not a
    pathological magnitude that would break the finite-difference
    reference itself rather than the backprop. The training chunks
    (P3-C03/C04) gradient-check on real, normalised training inputs for
    the same reason.
    """
    rng = np.random.default_rng(9)
    mlp = MLP([3, 8, 8, 2], rng)
    X = rng.uniform(-3.0, 3.0, size=(1, 3))  # single row, mild dynamic range
    target = rng.uniform(-3.0, 3.0, size=(1, 2))
    max_rel_error = gradient_check(mlp, X, _mse_loss_and_grad(target))
    assert max_rel_error < 1e-5


def test_gradient_check_holds_on_the_real_adr_m3_architecture():
    """The net this check actually guards: 2 hidden layers × 64 units
    (ADR-M3), realistic I/O widths, normalised O(1) inputs, batch 32.

    This is the regression guard behind backlog A10. Two things it proves,
    both of which toy 8-unit nets miss:

    - With the naive relative-error metric this exact net read 1.8e-4 —
      not a backprop bug, but a near-zero-gradient artifact (one weight's
      true gradient is ~2.6e-8; analytic and numeric agree to four sig
      figs). The floored metric reports it correctly.
    - A *correct* central-difference check on a 3-weight-layer net floors
      at a few 1e-7 on float64 noise alone (this seed reads well under
      1e-6; the worst over a 240-net sweep read 3.6e-7). It is comfortably
      inside the corrected 1e-5 threshold — the margin the spec's 1e-6 did
      not leave.

    Without this test the suite would only ever exercise toy nets and the
    A10 finding would carry no positive regression evidence.
    """
    rng = np.random.default_rng(24)
    mlp = MLP([3, 64, 64, 4], rng)  # LV-direct-shaped: state+action -> Δstate+logσ
    X = rng.standard_normal((32, 3))  # normalised O(1), full training batch
    target = rng.standard_normal((32, 4))
    max_rel_error = gradient_check(mlp, X, _mse_loss_and_grad(target))
    assert max_rel_error < 1e-5
