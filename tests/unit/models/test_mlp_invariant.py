"""Tests for `MLP.forward_invariant` — the prediction forward pass (models ADR-M1).

In plain words: predicting many runs at once must give *exactly* the numbers
you get predicting them one at a time. NumPy's ordinary matrix multiply can
differ in the last bit depending on how many rows are in the batch; the
prediction path therefore uses a different arithmetic (einsum) whose rows
never depend on the batch size. These tests pin that.
"""

from __future__ import annotations

import numpy as np
import pytest

from wmj.models.mlp import MLP, MLPArchitectureError


def _net(seed=0, sizes=(3, 64, 64, 4)):
    return MLP(sizes, np.random.default_rng(seed))


@pytest.mark.parametrize("n", [1, 2, 7, 64, 200])
def test_each_row_of_a_batch_equals_that_row_predicted_alone_bit_for_bit(n):
    net = _net()
    X = np.random.default_rng(1).normal(size=(n, 3))
    batch = net.forward_invariant(X)
    for i in range(n):
        assert np.array_equal(batch[i], net.forward_invariant(X[i : i + 1])[0])


def test_it_agrees_with_the_training_forward_to_rounding():
    net = _net()
    X = np.random.default_rng(2).normal(size=(50, 3))
    assert np.allclose(net.forward_invariant(X), net.forward(X)[0], rtol=0, atol=1e-12)


def test_it_is_a_tanh_network_with_a_linear_output():
    net = MLP((2, 3, 1), np.random.default_rng(0))
    (W0, b0), (W1, b1) = net.layers
    x = np.array([[0.3, -0.7]])
    expected = np.tanh(x @ W0 + b0) @ W1 + b1
    assert np.allclose(net.forward_invariant(x), expected, rtol=0, atol=1e-12)


def test_wrong_width_or_rank_is_refused_like_forward():
    net = _net()
    with pytest.raises(MLPArchitectureError):
        net.forward_invariant(np.zeros((4, 5)))
    with pytest.raises(MLPArchitectureError):
        net.forward_invariant(np.zeros(3))


def test_it_does_not_change_the_weights():
    net = _net()
    before = [(W.copy(), b.copy()) for W, b in net.layers]
    net.forward_invariant(np.ones((5, 3)))
    for (W, b), (W0, b0) in zip(net.layers, before):
        assert np.array_equal(W, W0) and np.array_equal(b, b0)
