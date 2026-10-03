"""Tests for wmj.models.direct — Model A (models ADR-M3, TC-MU1-01/02/04, TC-MU5-04).

In plain words: Model A is a small network that predicts how each quantity
will change and how wrong it expects to be. These tests check the exact
arithmetic of the loss it learns from (against brute-force numbers), that it
learns something real, that the same seed gives the same network, that
predicting a batch equals predicting one at a time, and that the numbers it
was told to use are the numbers the frozen recipe pins.
"""

from __future__ import annotations

import itertools
import re
from pathlib import Path

import numpy as np
import pytest

from wmj.harness.training import TrainingRecipe, build_training_data
from wmj.models import direct
from wmj.models.base import SeedSource, TrainingData, WorldContext
from wmj.models.direct import (
    BATCH_SIZE,
    BETA_NLL,
    EPOCHS,
    DirectModel,
    DirectTrainingError,
    beta_nll_loss_and_grad,
    direct_factory,
    gaussian_nll_loss_and_grad,
    normalise_inputs,
    train_direct,
)
from wmj.models.registry import all_models
from wmj.worlds import lv

SEED = 20260825
RECIPE_PATH = Path(__file__).resolve().parents[3] / "prereg" / "recipe.md"


def _ctx() -> WorldContext:
    spec = lv.regions()
    return WorldContext(
        world_name="lv", state_dim=2, action_dim=1,
        training_state_box=spec.training_state_box,
        training_action_interval=spec.training_action_interval, scale=lv.WORLD.scale,
    )


@pytest.fixture(scope="module")
def data() -> TrainingData:
    recipe = TrainingRecipe(
        training_trajectories=100, subsample_pairs=2000, kick_pairs=50,
        heldout_pairs=500, gradcheck_pairs=16,
    )
    return build_training_data("lv", lv.WORLD, SeedSource(SEED, None), recipe, horizon=100)


def _seeds():
    return SeedSource(SEED, "direct")


# --- the pinned numbers are the recipe's numbers (the models gate forbids reading files) ---


def _recipe_value(key):
    found = re.findall(rf"^{key}:[ \t]*([^#\s]*)", RECIPE_PATH.read_text(), re.MULTILINE)
    assert len(found) == 1
    return float(found[0])


def test_the_code_constants_equal_the_frozen_recipe():
    assert EPOCHS == _recipe_value("epochs")
    assert BATCH_SIZE == _recipe_value("batch_size")
    assert BETA_NLL == _recipe_value("beta_nll")


def test_the_recipe_values_are_the_spec_values():
    assert (EPOCHS, BATCH_SIZE, BETA_NLL) == (100, 256, 0.5)


# --- TC-MU5-04: the beta-NLL closure against finite differences over the outputs ---


def _reference_loss(Y, target, w):
    """L with the weight `w` supplied from outside (held constant)."""
    d = target.shape[1]
    mu, s = Y[:, :d], Y[:, d:]
    r = target - mu
    sigma2 = np.exp(2 * s)
    per = w * (0.5 * np.log(2 * np.pi) + s + r**2 / (2 * sigma2))
    return float(per.sum(axis=1).mean())


def _fd_grad(Y, target, w, eps=1e-6):
    grad = np.zeros_like(Y)
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            hi, lo = Y.copy(), Y.copy()
            hi[i, j] += eps
            lo[i, j] -= eps
            grad[i, j] = (_reference_loss(hi, target, w) - _reference_loss(lo, target, w)) / (2 * eps)
    return grad


def _case(seed=3, n=6, d=3):
    rng = np.random.default_rng(seed)
    Y = rng.normal(0.0, 0.6, size=(n, 2 * d))
    target = rng.normal(0.0, 1.0, size=(n, d))
    return Y, target


def test_tc_mu5_04_the_closure_gradient_matches_finite_differences_with_w_held_fixed():
    Y, target = _case()
    d = target.shape[1]
    sigma = np.exp(Y[:, d:])
    w = sigma ** (2 * BETA_NLL)  # computed once from the unperturbed outputs
    loss, grad = beta_nll_loss_and_grad(Y, target, beta=BETA_NLL)
    assert loss == pytest.approx(_reference_loss(Y, target, w), rel=1e-12)
    numeric = _fd_grad(Y, target, w)
    scale = np.abs(numeric).max(axis=0)  # per output column
    assert np.all(np.abs(grad - numeric).max(axis=0) <= 1e-7 * np.maximum(scale, 1e-12))


def test_tc_mu5_04_a_closure_that_differentiates_w_fails():
    Y, target = _case()
    d = target.shape[1]
    w = np.exp(Y[:, d:]) ** (2 * BETA_NLL)
    numeric = _fd_grad(Y, target, w)

    def differentiated_w(Y, target):
        n = Y.shape[0]
        mu, s = Y[:, :d], Y[:, d:]
        r = target - mu
        sigma2 = np.exp(2 * s)
        w_ = np.exp(s)
        dmu = -w_ * r / sigma2
        # d/ds of w*[...] when w = exp(s) is NOT held constant: product rule
        ds = w_ * (1 - r**2 / sigma2) + w_ * (0.5 * np.log(2 * np.pi) + s + r**2 / (2 * sigma2))
        return np.hstack([dmu, ds]) / n

    wrong = differentiated_w(Y, target)
    assert not np.all(np.abs(wrong - numeric).max(axis=0) <= 1e-7 * np.abs(numeric).max(axis=0))


def test_tc_mu5_04_a_closure_missing_the_batch_division_fails():
    Y, target = _case()
    d = target.shape[1]
    w = np.exp(Y[:, d:]) ** (2 * BETA_NLL)
    numeric = _fd_grad(Y, target, w)
    _, grad = beta_nll_loss_and_grad(Y, target, beta=BETA_NLL)
    undivided = grad * Y.shape[0]
    assert not np.allclose(undivided, numeric, rtol=1e-3, atol=0)


@pytest.mark.parametrize("beta", [0.0, 0.25, 0.5, 1.0])
def test_the_closure_holds_for_other_beta_too(beta):
    Y, target = _case(seed=5)
    d = target.shape[1]
    w = np.exp(Y[:, d:]) ** (2 * beta)
    _, grad = beta_nll_loss_and_grad(Y, target, beta=beta)
    numeric = _fd_grad(Y, target, w)
    assert np.allclose(grad, numeric, rtol=1e-6, atol=1e-9)


def test_beta_zero_is_plain_gaussian_nll_and_matches_the_plain_closure():
    Y, target = _case(seed=7)
    loss_b, grad_b = beta_nll_loss_and_grad(Y, target, beta=0.0)
    loss_p, grad_p = gaussian_nll_loss_and_grad(Y, target)
    assert loss_b == pytest.approx(loss_p, rel=1e-12)
    assert np.allclose(grad_b, grad_p, rtol=1e-12, atol=0)


def test_the_plain_nll_closure_is_the_exact_gradient_of_the_plain_nll():
    Y, target = _case(seed=9)
    ones = np.ones_like(target)
    _, grad = gaussian_nll_loss_and_grad(Y, target)
    assert np.allclose(grad, _fd_grad(Y, target, ones), rtol=1e-6, atol=1e-9)


def test_closure_shape_and_finite_guards():
    Y, target = _case()
    with pytest.raises(DirectTrainingError, match="shape"):
        beta_nll_loss_and_grad(Y[:, :-1], target, beta=0.5)
    bad = Y.copy()
    bad[0, 0] = np.nan
    with pytest.raises(DirectTrainingError, match="finite"):
        beta_nll_loss_and_grad(bad, target, beta=0.5)


# --- inputs are normalised through the context ---


def test_inputs_are_state_over_scale_and_action_over_half_width():
    ctx = _ctx()
    states = np.array([[4.0, 2.5], [2.0, 1.0]])
    actions = np.array([[0.1], [-0.05]])
    x = normalise_inputs(ctx, states, actions)
    half = (ctx.training_action_interval[0, 1] - ctx.training_action_interval[0, 0]) / 2
    assert np.array_equal(x[:, :2], states / ctx.scale)
    assert np.array_equal(x[:, 2:], actions / half)


# --- training ---


def _trained(data, *, epochs=12, **kw):
    return train_direct(_ctx(), _seeds(), data, epochs=epochs, **kw)


@pytest.fixture(scope="module")
def trained_net(data):
    return _trained(data)


def _heldout_error(model, pairs, scale):
    means, _ = model.predict_batch(pairs.state, pairs.action)
    return float(np.mean(((means - pairs.next_state) / scale) ** 2))


def test_it_learns_something_real(data, trained_net):
    model = DirectModel(_ctx(), trained_net)
    persistence = float(np.mean(((data.heldout_pairs.state - data.heldout_pairs.next_state) / lv.WORLD.scale) ** 2))
    assert _heldout_error(model, data.heldout_pairs, lv.WORLD.scale) < 0.5 * persistence


def test_training_twice_from_the_same_seed_gives_identical_weights(data, trained_net):
    again = _trained(data)
    for (W, b), (W2, b2) in zip(trained_net.layers, again.layers):
        assert W.tobytes() == W2.tobytes() and b.tobytes() == b2.tobytes()


def test_a_different_run_seed_gives_different_weights(data, trained_net):
    other = train_direct(_ctx(), SeedSource(SEED + 1, "direct"), data, epochs=12)
    assert other.layers[0][0].tobytes() != trained_net.layers[0][0].tobytes()


def test_the_network_has_the_pinned_shape(trained_net):
    assert trained_net.layer_sizes == (3, 64, 64, 4)  # (state 2 + action 1) -> 2 x 64 -> (d mean, d log-sigma)


def test_initial_weights_come_from_the_weights_stream(data, monkeypatch):
    seen = []
    original = direct.MLP

    def spy(sizes, rng):
        seen.append(rng.bit_generator.state["state"]["state"])
        return original(sizes, rng)

    monkeypatch.setattr(direct, "MLP", spy)
    _trained(data, epochs=1)
    expected = _seeds().rng("weights").bit_generator.state["state"]["state"]
    assert seen == [expected]


def test_the_gradient_check_runs_first_on_the_gradcheck_rows_under_plain_nll(data, monkeypatch):
    calls = []
    original = direct.gradient_check

    def spy(mlp, X, loss_and_grad, **kw):
        calls.append((X.copy(), [W.copy() for W, _ in mlp.layers], kw))
        return original(mlp, X, loss_and_grad, **kw)

    monkeypatch.setattr(direct, "gradient_check", spy)
    net = _trained(data, epochs=1)
    assert len(calls) == 1
    X, weights_then, kw = calls[0]
    idx = data.gradcheck_index
    expected_X = normalise_inputs(_ctx(), data.train_pairs.state[idx], data.train_pairs.action[idx])
    assert np.array_equal(X, expected_X)
    fresh = direct.MLP((3, 64, 64, 4), _seeds().rng("weights"))
    for (W, _), W0 in zip(fresh.layers, weights_then):
        assert np.array_equal(W, W0), "the check must run before any weight update"
    assert any(not np.array_equal(W, W0) for (W, _), W0 in zip(net.layers, weights_then))
    assert kw.get("tolerance", 1e-5) == 1e-5


def test_a_failing_gradient_check_stops_training(data, monkeypatch):
    from wmj.models.mlp import GradientCheckError

    def broken(*_a, **_k):
        raise GradientCheckError("backprop disagrees")

    monkeypatch.setattr(direct, "gradient_check", broken)
    with pytest.raises(GradientCheckError):
        _trained(data, epochs=1)


def test_training_data_without_pairs_is_refused():
    bare = TrainingData(states=np.zeros((2, 4, 2)), actions=np.zeros((2, 3, 1)))
    with pytest.raises(DirectTrainingError, match="train_pairs"):
        train_direct(_ctx(), _seeds(), bare, epochs=1)
    with pytest.raises(DirectTrainingError, match="train_pairs"):
        direct_factory(_ctx(), _seeds(), bare)


def test_a_context_that_does_not_match_the_data_is_refused(data):
    ctx = WorldContext(
        world_name="lv", state_dim=3, action_dim=1,
        training_state_box=np.zeros((3, 2)), training_action_interval=np.array([[-0.1, 0.1]]),
        scale=np.ones(3),
    )
    with pytest.raises(DirectTrainingError, match="width"):
        train_direct(ctx, _seeds(), data, epochs=1)


def test_diverging_training_is_a_loud_error_not_a_silent_nan(data, monkeypatch):
    monkeypatch.setattr(direct, "LEARNING_RATE", 1e12)
    with pytest.raises(DirectTrainingError, match="finite|diverg"):
        _trained(data, epochs=3)


def test_batches_are_the_pinned_size_and_cover_every_row_each_epoch(data, monkeypatch):
    sizes = []
    original = direct.beta_nll_loss_and_grad

    def spy(Y, target, *, beta):
        if beta == BETA_NLL:  # training batches only (the gradient check uses beta 0)
            sizes.append(Y.shape[0])
        return original(Y, target, beta=beta)

    monkeypatch.setattr(direct, "beta_nll_loss_and_grad", spy)
    _trained(data, epochs=1, batch_size=256)
    m = data.train_pairs.state.shape[0]
    assert sum(sizes) == m and max(sizes) == 256 and sizes[:-1] == [256] * (len(sizes) - 1)


def test_each_epoch_shuffles_from_its_own_named_stream(data, monkeypatch):
    asked = []
    original = SeedSource.rng

    def spy(self, *purpose):
        asked.append(purpose)
        return original(self, *purpose)

    monkeypatch.setattr(SeedSource, "rng", spy)
    _trained(data, epochs=3)
    assert asked == [("weights",), ("shuffle", "0"), ("shuffle", "1"), ("shuffle", "2")]


# --- prediction: TC-MU1-01/02/04 ---


@pytest.fixture(scope="module")
def model(trained_net):
    return DirectModel(_ctx(), trained_net)


def test_flags_and_name(model):
    assert (model.name, model.is_fixture, model.is_baseline, model.stateless) == (
        "direct", False, False, True)
    model.reset()  # a no-op, must not raise


def test_prediction_format_is_mean_and_positive_spread_per_dimension(model, data):
    p = model.predict(data.heldout_pairs.state[0], data.heldout_pairs.action[0])
    assert p.mean.shape == (2,) and p.spread.shape == (2,)
    assert np.all(p.spread > 0) and np.all(np.isfinite(p.mean)) and np.all(np.isfinite(p.spread))


def test_mean_is_state_plus_the_predicted_change_and_spread_is_exp_log_sigma(model, data, trained_net):
    s, a = data.heldout_pairs.state[:5], data.heldout_pairs.action[:5]
    out = trained_net.forward_invariant(normalise_inputs(_ctx(), s, a))
    means, spreads = model.predict_batch(s, a)
    assert np.array_equal(means, s + out[:, :2])
    assert np.array_equal(spreads, np.exp(out[:, 2:]))


@pytest.mark.parametrize("n", [1, 2, 7, 64, 200])
def test_tc_mu1_04_batch_rows_are_bit_identical_to_one_at_a_time(model, data, n):
    s = np.resize(data.heldout_pairs.state, (n, 2))
    a = np.resize(data.heldout_pairs.action, (n, 1))
    means, spreads = model.predict_batch(s, a)
    for i in range(n):
        p = model.predict(s[i], a[i])
        assert np.array_equal(means[i], p.mean) and np.array_equal(spreads[i], p.spread)


def test_predict_refuses_wrong_shapes_and_non_finite_input(model):
    with pytest.raises(DirectTrainingError, match="shape"):
        model.predict_batch(np.zeros((3, 5)), np.zeros((3, 1)))
    with pytest.raises(DirectTrainingError, match="shape"):
        model.predict_batch(np.zeros((3, 2)), np.zeros((4, 1)))
    with pytest.raises(DirectTrainingError, match="finite"):
        model.predict(np.array([np.nan, 1.0]), np.zeros(1))


def test_a_degenerate_spread_is_refused_at_predict_time(trained_net):
    net = trained_net
    saved = net.layers[-1][1].copy()
    try:
        net.layers[-1][1][2:] = -1e4  # log sigma -> exp underflows to 0
        with pytest.raises(DirectTrainingError, match="spread"):
            DirectModel(_ctx(), net).predict_batch(np.full((2, 2), 3.0), np.zeros((2, 1)))
    finally:
        net.layers[-1][1][:] = saved


def test_returned_arrays_are_independent_of_the_network(model, data):
    p = model.predict(data.heldout_pairs.state[0], data.heldout_pairs.action[0])
    with pytest.raises(ValueError):
        p.mean[0] = 0.0


# --- registry ---


def test_direct_is_registered_under_its_name_with_the_uniform_factory(data):
    assert "direct" in all_models()
    assert all_models()["direct"] is direct_factory
    built = direct_factory(_ctx(), _seeds(), data)
    assert isinstance(built, DirectModel)


# --- independent review, P3-C03 pass 1: an independent reference trainer pins the whole recipe ---


def _reference_train(ctx, seeds, pairs, *, epochs, batch_size, beta):
    """Model A trained straight from the spec text, sharing no code with `train_direct`."""
    d = ctx.state_dim
    hw = (ctx.training_action_interval[:, 1] - ctx.training_action_interval[:, 0]) / 2.0
    X = np.hstack([pairs.state / ctx.scale, pairs.action / hw])
    T = pairs.next_state - pairs.state
    sizes = [X.shape[1], 64, 64, 2 * d]
    rng = seeds.rng("weights")
    Ws = [rng.uniform(-1 / np.sqrt(i), 1 / np.sqrt(i), size=(i, o)) for i, o in itertools.pairwise(sizes)]
    bs = [np.zeros(o) for o in sizes[1:]]
    m_w = [np.zeros_like(w) for w in Ws]; v_w = [np.zeros_like(w) for w in Ws]
    m_b = [np.zeros_like(b) for b in bs]; v_b = [np.zeros_like(b) for b in bs]
    t = 0
    for epoch in range(epochs):
        order = seeds.rng("shuffle", str(epoch)).permutation(X.shape[0])
        for start in range(0, X.shape[0], batch_size):
            rows = order[start : start + batch_size]
            x, y = X[rows], T[rows]
            acts = [x]
            for k in range(3):
                pre = acts[-1] @ Ws[k] + bs[k]
                acts.append(np.tanh(pre) if k < 2 else pre)
            out = acts[-1]
            mu, s = out[:, :d], out[:, d:]
            r = y - mu
            sigma = np.exp(s)
            w = sigma ** (2 * beta)
            g = np.hstack([-w * r / sigma**2, w * (1 - r**2 / sigma**2)]) / x.shape[0]
            dW, db = [None] * 3, [None] * 3
            for k in (2, 1, 0):
                dW[k] = acts[k].T @ g
                db[k] = g.sum(axis=0)
                if k:
                    g = (g @ Ws[k].T) * (1 - acts[k] ** 2)
            t += 1
            for k in range(3):
                for p, gr, mm, vv in ((Ws[k], dW[k], m_w[k], v_w[k]), (bs[k], db[k], m_b[k], v_b[k])):
                    mm *= 0.9; mm += 0.1 * gr
                    vv *= 0.999; vv += 0.001 * gr**2
                    p -= 1e-3 * (mm / (1 - 0.9**t)) / (np.sqrt(vv / (1 - 0.999**t)) + 1e-8)
    return list(zip(Ws, bs))


def _close(net, reference, rtol=1e-7, atol=1e-10):
    for (W, b), (Wr, br) in zip(net.layers, reference):
        assert np.allclose(W, Wr, rtol=rtol, atol=atol) and np.allclose(b, br, rtol=rtol, atol=atol)


@pytest.mark.parametrize("batch_size", [64, 256])
def test_train_direct_equals_an_independent_trainer_written_from_the_spec(data, batch_size):
    net = train_direct(_ctx(), _seeds(), data, epochs=3, batch_size=batch_size)
    ref = _reference_train(_ctx(), _seeds(), data.train_pairs, epochs=3, batch_size=batch_size, beta=0.5)
    _close(net, ref)


def test_the_registered_factory_trains_with_the_recipes_epochs_batch_size_and_beta(data):
    """I2 / pass-2 I1: the factory (what the registry and P6-C01 call) must train exactly as
    `train_direct` does at the recipe's 100 / 256 / 0.5. Training is deterministic, so the
    comparison is bit-exact — an off-by-one epoch, a batch of 255 or a beta of 0.49 all
    change the bytes (the training logic itself is pinned by the reference-trainer tests)."""
    built = direct_factory(_ctx(), _seeds(), data)
    expected = train_direct(
        _ctx(), _seeds(), data, epochs=EPOCHS, batch_size=BATCH_SIZE, beta=BETA_NLL
    )
    for (W, b), (We, be) in zip(built._net.layers, expected.layers):
        assert W.tobytes() == We.tobytes() and b.tobytes() == be.tobytes()
    assert (EPOCHS, BATCH_SIZE, BETA_NLL) == (100, 256, 0.5)


def test_the_factory_uses_the_seed_it_is_given(data):
    other = direct_factory(_ctx(), SeedSource(SEED + 5, "direct"), data)
    same = direct_factory(_ctx(), _seeds(), data)
    assert other._net.layers[0][0].tobytes() != same._net.layers[0][0].tobytes()


def test_a_missing_gradcheck_index_or_empty_training_set_is_refused():
    from wmj.models.base import Pairs

    pairs = Pairs(np.zeros((5, 2)), np.zeros((5, 1)), np.ones((5, 2)), np.zeros(5, dtype=bool))
    states, actions = np.zeros((2, 4, 2)), np.zeros((2, 3, 1))
    no_check = TrainingData(states=states, actions=actions, train_pairs=pairs)
    with pytest.raises(DirectTrainingError, match="gradcheck_index"):
        train_direct(_ctx(), _seeds(), no_check, epochs=1)
    empty = Pairs(np.zeros((0, 2)), np.zeros((0, 1)), np.zeros((0, 2)), np.zeros(0, dtype=bool))
    with pytest.raises(DirectTrainingError, match="empty|no training"):
        train_direct(_ctx(), _seeds(), TrainingData(states=states, actions=actions, train_pairs=empty), epochs=1)


# --- independent review, P3-C03 pass 2 ---


def test_a_non_default_beta_is_the_one_used(data):
    net = train_direct(_ctx(), _seeds(), data, epochs=2, batch_size=64, beta=0.25)
    ref = _reference_train(_ctx(), _seeds(), data.train_pairs, epochs=2, batch_size=64, beta=0.25)
    _close(net, ref)
    default = train_direct(_ctx(), _seeds(), data, epochs=2, batch_size=64)
    assert net.layers[0][0].tobytes() != default.layers[0][0].tobytes()


@pytest.mark.parametrize("kw", [{"epochs": 0}, {"epochs": -1}, {"epochs": 2.0}, {"batch_size": 0},
                                {"batch_size": -5}, {"batch_size": True}, {"beta": -0.1},
                                {"beta": float("nan")}, {"beta": None}])
def test_nonsense_training_arguments_are_refused(data, kw):
    with pytest.raises(DirectTrainingError):
        train_direct(_ctx(), _seeds(), data, **{"epochs": 1, **kw})


def test_a_last_step_blow_up_is_caught_by_the_epoch_end_check_not_left_to_predict(data, monkeypatch):
    monkeypatch.setattr(direct, "LEARNING_RATE", float("inf"))
    with pytest.raises(DirectTrainingError, match="not finite after epoch"):
        train_direct(_ctx(), _seeds(), data, epochs=1, batch_size=10**6)


def test_the_gradient_check_target_is_the_change_of_the_gradcheck_rows(data, monkeypatch):
    seen = {}

    def spy(mlp, X, loss_and_grad, **kw):
        # With outputs mu = 0 and s = 0 the plain-NLL gradient w.r.t. mu is -(target)/n per row.
        Y = np.zeros((X.shape[0], 4))
        _, grad = loss_and_grad(Y)
        seen["target"] = -grad[:, :2] * X.shape[0]

    monkeypatch.setattr(direct, "gradient_check", spy)
    train_direct(_ctx(), _seeds(), data, epochs=1)
    idx = data.gradcheck_index
    expected = data.train_pairs.next_state[idx] - data.train_pairs.state[idx]
    assert np.allclose(seen["target"], expected, rtol=0, atol=1e-12)


def test_an_infinite_spread_and_a_nan_action_are_each_refused_by_name(trained_net):
    model = DirectModel(_ctx(), trained_net)
    saved = trained_net.layers[-1][1].copy()
    try:
        trained_net.layers[-1][1][2:] = 1e4  # log sigma huge -> exp overflows to inf
        with np.errstate(over="ignore"), pytest.raises(DirectTrainingError, match="spread"):
            model.predict_batch(np.full((2, 2), 3.0), np.zeros((2, 1)))
    finally:
        trained_net.layers[-1][1][:] = saved
    with pytest.raises(DirectTrainingError, match="finite"):
        model.predict_batch(np.full((2, 2), 3.0), np.array([[np.nan], [0.0]]))
    with pytest.raises(DirectTrainingError, match="one state"):
        model.predict(np.zeros((2, 2)), np.zeros(1))
    with pytest.raises(DirectTrainingError, match="one state"):
        model.predict(np.zeros(2), np.zeros((1, 1)))
