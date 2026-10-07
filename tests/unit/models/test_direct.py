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
    assert direct.LR_FINAL == _recipe_value("lr_final")
    assert direct.LEARNING_RATE == _recipe_value("lr_initial")
    text = RECIPE_PATH.read_text()
    assert re.findall(r"^lr_schedule:[ \t]*([^#\s]*)", text, re.MULTILINE) == ["cosine"]
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
    import copy

    net = copy.deepcopy(trained_net)  # a trained network is read-only; edit a copy
    net.layers[-1][1][2:] = -1e4  # log sigma -> exp underflows to 0
    with pytest.raises(DirectTrainingError, match="spread"):
        DirectModel(_ctx(), net).predict_batch(np.full((2, 2), 3.0), np.zeros((2, 1)))


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


def _reference_train(ctx, seeds, pairs, *, epochs, batch_size, beta, log_sigma_bias0=0.0):
    """Model A trained straight from the spec text, sharing no code with `train_direct`."""
    d = ctx.state_dim
    hw = (ctx.training_action_interval[:, 1] - ctx.training_action_interval[:, 0]) / 2.0
    X = np.hstack([pairs.state / ctx.scale, pairs.action / hw])
    T = pairs.next_state - pairs.state
    sizes = [X.shape[1], 64, 64, 2 * d]
    rng = seeds.rng("weights")
    Ws = [rng.uniform(-1 / np.sqrt(i), 1 / np.sqrt(i), size=(i, o)) for i, o in itertools.pairwise(sizes)]
    bs = [np.zeros(o) for o in sizes[1:]]
    bs[2][d:] = log_sigma_bias0  # 0 in training; tests start it tiny to reach the small-sigma regime
    m_w = [np.zeros_like(w) for w in Ws]; v_w = [np.zeros_like(w) for w in Ws]
    m_b = [np.zeros_like(b) for b in bs]; v_b = [np.zeros_like(b) for b in bs]
    t = 0
    # cosine decay 1e-3 -> 1e-5 over the epochs (D18), written out here independently
    lrs = [1e-3] if epochs == 1 else [
        1e-5 + 0.5 * (1e-3 - 1e-5) * (1 + np.cos(np.pi * e / (epochs - 1))) for e in range(epochs)
    ]
    for epoch in range(epochs):
        lr = lrs[epoch]
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
                    p -= lr * (mm / (1 - 0.9**t)) / (np.sqrt(vv / (1 - 0.999**t)) + 1e-8)
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
    import copy

    model = DirectModel(_ctx(), trained_net)
    huge = copy.deepcopy(trained_net)
    huge.layers[-1][1][2:] = 1e4  # log sigma huge -> exp overflows to inf
    with np.errstate(over="ignore"), pytest.raises(DirectTrainingError, match="spread"):
        DirectModel(_ctx(), huge).predict_batch(np.full((2, 2), 3.0), np.zeros((2, 1)))
    with pytest.raises(DirectTrainingError, match="finite"):
        model.predict_batch(np.full((2, 2), 3.0), np.array([[np.nan], [0.0]]))
    with pytest.raises(DirectTrainingError, match="one state"):
        model.predict(np.zeros((2, 2)), np.zeros(1))
    with pytest.raises(DirectTrainingError, match="one state"):
        model.predict(np.zeros(2), np.zeros((1, 1)))


# --- independent review, P3-C03 pass 3: the 4-dimensional pendulum, and an asymmetric interval ---

from wmj.harness.training import make_world_context
from wmj.worlds import pendulum

PEND_RECIPE = TrainingRecipe(
    training_trajectories=300, subsample_pairs=1000, kick_pairs=40, heldout_pairs=300,
    gradcheck_pairs=16,
)


@pytest.fixture(scope="module")
def pend():
    data = build_training_data("pendulum", pendulum.WORLD, SeedSource(SEED, None), PEND_RECIPE, horizon=300)
    ctx = make_world_context("pendulum", pendulum.WORLD)
    net = train_direct(ctx, _seeds(), data, epochs=2)
    return ctx, data, net


def test_pendulum_network_shape_and_output_columns(pend):
    ctx, data, net = pend
    assert net.layer_sizes == (5, 64, 64, 8)  # (4 states + 1 action) -> 2 x 64 -> (4 means, 4 log-sigmas)
    s, a = data.heldout_pairs.state[:10], data.heldout_pairs.action[:10]
    out = net.forward_invariant(normalise_inputs(ctx, s, a))
    means, spreads = DirectModel(ctx, net).predict_batch(s, a)
    assert means.shape == (10, 4) and spreads.shape == (10, 4)
    assert np.array_equal(means, s + out[:, :4])
    assert np.array_equal(spreads, np.exp(out[:, 4:]))


@pytest.mark.parametrize("n", [1, 7, 200])
def test_pendulum_batch_rows_are_bit_identical_to_one_at_a_time(pend, n):
    ctx, data, net = pend
    model = DirectModel(ctx, net)
    s = np.resize(data.heldout_pairs.state, (n, 4))
    a = np.resize(data.heldout_pairs.action, (n, 1))
    means, spreads = model.predict_batch(s, a)
    for i in range(n):
        p = model.predict(s[i], a[i])
        assert np.array_equal(means[i], p.mean) and np.array_equal(spreads[i], p.spread)


def test_pendulum_training_equals_the_independent_reference_trainer(pend):
    ctx, data, _ = pend
    net = train_direct(ctx, _seeds(), data, epochs=2, batch_size=64)
    ref = _reference_train(ctx, _seeds(), data.train_pairs, epochs=2, batch_size=64, beta=0.5)
    _close(net, ref)


def test_the_half_width_is_half_the_interval_for_an_asymmetric_range():
    ctx = WorldContext(
        world_name="x", state_dim=2, action_dim=1, training_state_box=np.zeros((2, 2)),
        training_action_interval=np.array([[0.0, 0.4]]), scale=np.array([1.0, 1.0]),
    )
    x = normalise_inputs(ctx, np.zeros((1, 2)), np.array([[0.4]]))
    assert x[0, 2] == pytest.approx(2.0)  # 0.4 / ((0.4 - 0.0) / 2), not 0.4 / 0.4 or 0.4 / 0.2 by accident
    x2 = normalise_inputs(ctx, np.zeros((1, 2)), np.array([[-0.2]]))
    assert x2[0, 2] == pytest.approx(-1.0)


def test_predict_returns_arrays_that_do_not_alias_the_batch_buffers(model, data):
    p = model.predict(data.heldout_pairs.state[0], data.heldout_pairs.action[0])
    for array in (p.mean, p.spread):
        assert array.base is None or not array.base.flags.writeable


# --- independent review, P3-C03 pass 5: the push must reach the network ---


def test_the_action_is_used_and_not_dropped_negated_or_scaled(model, trained_net):
    """An action-blind Model A is exactly what the project exists to catch: with explicit
    non-zero actions the prediction must equal the network's output on the true inputs, and
    different actions must give different predictions."""
    ctx = _ctx()
    s = np.array([[3.0, 2.0], [3.0, 2.0], [4.5, 2.0], [3.0, 2.0]])
    a = np.array([[0.09], [-0.09], [0.06], [0.0]])
    out = trained_net.forward_invariant(normalise_inputs(ctx, s, a))
    means, spreads = model.predict_batch(s, a)
    assert np.array_equal(means, s + out[:, :2])
    assert np.array_equal(spreads, np.exp(out[:, 2:]))
    assert not np.array_equal(means[0], means[1])  # +0.09 and -0.09 differ
    assert not np.array_equal(means[0], means[3])  # a push differs from no push
    for i in range(4):
        p = model.predict(s[i], a[i])
        assert np.array_equal(p.mean, means[i]) and np.array_equal(p.spread, spreads[i])
    # and a halved / negated / zeroed action would give a different network input
    half = normalise_inputs(ctx, s, a * 0.5)
    assert not np.array_equal(half, normalise_inputs(ctx, s, a))


def test_the_error_classes_are_wmj_errors():
    from wmj.errors import WmjError
    from wmj.harness.sufficiency import SufficiencyError

    assert issubclass(DirectTrainingError, WmjError) and issubclass(SufficiencyError, WmjError)


# --- independent review, P3-C03 pass 6: every held-out row, the unfamiliar regions, and 100 epochs ---


def _expected_predictions(ctx, net, s, a):
    hw = (ctx.training_action_interval[:, 1] - ctx.training_action_interval[:, 0]) / 2.0
    x = np.hstack([s / ctx.scale, a / hw])  # written out here, not via normalise_inputs
    out = net.forward_invariant(x)
    d = ctx.state_dim
    return s + out[:, :d], np.exp(out[:, d:])


def test_predictions_equal_the_network_on_every_heldout_row_and_the_unfamiliar_corners(data, trained_net):
    """The model must not clip, rescale or special-case anything: all held-out rows plus corner
    rows from the out-high-amplitude region (states 8-12) and out-large-action (|push| 0.2)."""
    ctx = _ctx()
    h = data.heldout_pairs
    corner_s = np.array([[8.0, 4.0], [12.0, 6.0], [12.0, 4.0], [8.0, 6.0], [3.0, 2.0], [3.0, 2.0]])
    corner_a = np.array([[0.2], [-0.2], [0.15], [-0.15], [0.2], [-0.2]])
    s = np.vstack([h.state, corner_s])
    a = np.vstack([h.action, corner_a])
    means, spreads = DirectModel(ctx, trained_net).predict_batch(s, a)
    exp_means, exp_spreads = _expected_predictions(ctx, trained_net, s, a)
    assert np.array_equal(means, exp_means) and np.array_equal(spreads, exp_spreads)


def test_pendulum_predictions_equal_the_network_on_heldout_rows_and_large_pushes(pend):
    ctx, data, net = pend
    h = data.heldout_pairs
    corner_s = np.array([[3.0, 0.2, 0.4, -0.4], [2.6, -0.3, -0.5, 0.5], [0.0, 0.0, 0.0, 0.0]])
    corner_a = np.array([[2.0], [-2.0], [1.5]])
    s = np.vstack([h.state, corner_s])
    a = np.vstack([h.action, corner_a])
    means, spreads = DirectModel(ctx, net).predict_batch(s, a)
    exp_means, exp_spreads = _expected_predictions(ctx, net, s, a)
    assert np.array_equal(means, exp_means) and np.array_equal(spreads, exp_spreads)


def test_a_hundred_epochs_equal_the_independent_trainer_on_a_small_slice(data):
    """Epoch-dependent bugs (a schedule, a stream or a reset that only bites after epoch 3,
    a log-sigma floor) are invisible to the 3-epoch comparisons. With 600 rows a 100-epoch
    run is quick and the two trainers still agree to ~1e-11 (measured), so a tight tolerance
    holds; at full scale chaos amplifies rounding to O(0.1) after ~20,000 steps."""
    from wmj.models.base import Pairs

    p = data.train_pairs
    rows = np.r_[0:40, 1000:1560]  # 40 kick pairs then 560 non-kick pairs
    small = Pairs(p.state[rows], p.action[rows], p.next_state[rows], p.is_kick[rows])
    td = TrainingData(
        states=data.states, actions=data.actions, train_pairs=small,
        gradcheck_index=np.arange(16, dtype=np.int64),
    )
    net = train_direct(_ctx(), _seeds(), td, epochs=EPOCHS, batch_size=BATCH_SIZE, beta=BETA_NLL)
    ref = _reference_train(_ctx(), _seeds(), small, epochs=EPOCHS, batch_size=BATCH_SIZE, beta=BETA_NLL)
    _close(net, ref, rtol=1e-6, atol=1e-8)


# --- independent review, P3-C03 pass 7: reachable extremes, the small-sigma regime ---


def test_predictions_are_exact_in_the_states_the_judged_rollouts_actually_reach(data, trained_net):
    """True-world zero-push rollouts of out-high-amplitude reach LV states up to (15.5, 8.6)
    (3.9 in network-input units) — beyond the pass-6 corners; no clip or wrap may touch them."""
    ctx = _ctx()
    s = np.array([[15.5, 8.6], [15.5, 1.0], [1.0, 8.6], [14.0, 7.0]])
    a = np.array([[0.2], [-0.2], [0.0], [0.1]])
    means, spreads = DirectModel(ctx, trained_net).predict_batch(s, a)
    exp_means, exp_spreads = _expected_predictions(ctx, trained_net, s, a)
    assert np.array_equal(means, exp_means) and np.array_equal(spreads, exp_spreads)


def test_pendulum_predictions_are_exact_for_angles_and_speeds_the_inverted_region_reaches(pend):
    """out-near-inverted rollouts reach theta2 ~ 39 rad and speeds ~ 12 rad/s; angles are
    not wrapped (the network sees the raw state / scale), and nothing is clipped."""
    ctx, _data, net = pend
    s = np.array([[8.8, 39.1, 8.9, 12.6], [-8.8, -39.1, -8.9, -12.6], [3.5, 6.5, 0.0, 0.0],
                  [-3.5, 2.0, 0.0, 0.0], [2.0, 60.0, 0.0, 0.0], [1.0, -60.0, 0.0, 0.0]])
    a = np.array([[0.0], [1.0], [-1.0], [0.0], [0.0], [2.0]])
    means, spreads = DirectModel(ctx, net).predict_batch(s, a)
    exp_means, exp_spreads = _expected_predictions(ctx, net, s, a)
    assert np.array_equal(means, exp_means) and np.array_equal(spreads, exp_spreads)


def test_training_that_drives_the_error_bar_very_small_matches_the_independent_trainer(data, monkeypatch):
    """Real training pushes log-sigma below -6 for nearly every row (min about -6.9); a
    'stability' floor on log-sigma would change what is learned. Start the error-bar output
    far below any plausible floor (-8) so the closure and the update are exercised there."""
    original = direct.MLP

    class TinyBar(original):
        def __init__(self, sizes, rng):
            super().__init__(sizes, rng)
            bias = self.layers[-1][1]
            bias[len(bias) // 2 :] = -8.0

    monkeypatch.setattr(direct, "MLP", TinyBar)
    net = train_direct(_ctx(), _seeds(), data, epochs=3)
    ref = _reference_train(
        _ctx(), _seeds(), data.train_pairs, epochs=3, batch_size=256, beta=0.5, log_sigma_bias0=-8.0
    )
    _close(net, ref)


def test_the_closure_matches_the_formulas_at_very_small_sigma():
    rng = np.random.default_rng(11)
    Y = np.hstack([rng.normal(0, 1e-4, (5, 2)), rng.uniform(-9.0, -7.0, (5, 2))])
    target = rng.normal(0, 1e-4, (5, 2))
    loss, grad = beta_nll_loss_and_grad(Y, target, beta=0.5)
    mu, s = Y[:, :2], Y[:, 2:]
    sigma = np.exp(s)
    w = sigma  # beta = 0.5
    r = target - mu
    expected_loss = float((w * (0.5 * np.log(2 * np.pi) + s + r**2 / (2 * sigma**2))).sum(axis=1).mean())
    assert loss == pytest.approx(expected_loss, rel=1e-12)
    assert np.allclose(grad[:, :2], -w * r / sigma**2 / 5, rtol=1e-12, atol=0)
    assert np.allclose(grad[:, 2:], w * (1 - r**2 / sigma**2) / 5, rtol=1e-12, atol=0)


# --- independent review, P3-C03 pass 8: faithful at the error-bar sizes real training produces ---


@pytest.mark.parametrize("target_median_log_sigma", [-6.5, 2.7])
def test_predictions_are_faithful_at_the_error_bar_sizes_the_real_recipe_reaches(
    data, trained_net, target_median_log_sigma
):
    """The fixture net is trained for 12 epochs and has error bars of ~0.01-0.06; the real
    100-epoch net has ~1e-3 (log sigma about -6.5) and, at extreme states, ~15 (log sigma
    +2.7). A 'minimum std' floor or a cap in the prediction path — a common idiom — would
    change what the judge scores in exactly that regime, so shift the error-bar output to
    those sizes and require predict_batch to equal the network's own exp(out), bit for bit."""
    import copy

    ctx = _ctx()
    net = copy.deepcopy(trained_net)
    h = data.heldout_pairs
    out = net.forward_invariant(normalise_inputs(ctx, h.state, h.action))
    shift = target_median_log_sigma - float(np.median(out[:, 2:]))
    net.layers[-1][1][2:] += shift
    means, spreads = DirectModel(ctx, net).predict_batch(h.state, h.action)
    exp_means, exp_spreads = _expected_predictions(ctx, net, h.state, h.action)
    assert np.array_equal(means, exp_means) and np.array_equal(spreads, exp_spreads)
    assert abs(float(np.median(np.log(spreads))) - target_median_log_sigma) < 1e-6


# --- independent review, P3-C03 pass 9: the sizes the pendulum's kicks produce ---


@pytest.mark.parametrize("shift", [1.0, -1.5, 2.0])
def test_predictions_are_faithful_at_the_change_sizes_a_kicked_pendulum_reaches(pend, shift):
    """The real pendulum net predicts changes of ~1.0 for a push of 1 and ~2.0 for a push of 2
    (out-large-action); a clip on the predicted change would hide that. Shift the change
    outputs to those sizes and require bit-equality with the network's own output."""
    import copy

    ctx, data, net0 = pend
    net = copy.deepcopy(net0)
    net.layers[-1][1][:4] += shift
    h = data.heldout_pairs
    means, spreads = DirectModel(ctx, net).predict_batch(h.state, h.action)
    exp_means, exp_spreads = _expected_predictions(ctx, net, h.state, h.action)
    assert np.array_equal(means, exp_means) and np.array_equal(spreads, exp_spreads)
    assert np.abs(means - h.state).max() > 0.9


def test_training_targets_are_not_clipped_at_the_size_of_a_kick():
    """Train on pairs whose change is up to 2.0 in one dimension: a clip on the targets would
    stop the network ever learning the size of a large push."""
    from wmj.models.base import Pairs

    rng = np.random.default_rng(0)
    n = 300
    ctx = make_world_context("pendulum", pendulum.WORLD)
    s = rng.normal(0, 0.3, (n, 4))
    a = rng.uniform(-1, 1, (n, 1))
    ns = s.copy()
    ns[:, 2] += 2.0 * a[:, 0]
    pairs = Pairs(s, a, ns, np.abs(a[:, 0]) > 0)
    td = TrainingData(
        states=np.zeros((2, 4, 4)), actions=np.zeros((2, 3, 1)), train_pairs=pairs,
        gradcheck_index=np.arange(16, dtype=np.int64),
    )
    net = train_direct(ctx, _seeds(), td, epochs=60, batch_size=64)
    means, _ = DirectModel(ctx, net).predict_batch(s, a)
    assert np.abs(means[:, 2] - ns[:, 2]).max() < 0.5  # it learned the lever...
    assert np.abs(means[:, 2] - s[:, 2]).max() > 1.5  # ...including changes near 2.0


# --- D18 (owner-approved 2026-10-04): cosine learning-rate decay ---


def test_the_learning_rate_schedule_is_a_cosine_from_start_to_final():
    rates = direct.learning_rates(100)
    assert len(rates) == 100 and rates[0] == direct.LEARNING_RATE == 1e-3
    assert rates[-1] == pytest.approx(direct.LR_FINAL, rel=1e-12, abs=0) and direct.LR_FINAL == 1e-5
    assert all(a > b for a, b in itertools.pairwise(rates))  # strictly decreasing
    assert rates[49] == pytest.approx(1e-5 + 0.5 * (1e-3 - 1e-5) * (1 + np.cos(np.pi * 49 / 99)), rel=1e-12)
    mid = direct.learning_rates(3)
    assert mid[1] == pytest.approx((1e-3 + 1e-5) / 2, rel=1e-12)  # halfway through a 3-epoch run


def test_a_one_epoch_run_uses_the_initial_rate():
    assert direct.learning_rates(1) == [direct.LEARNING_RATE]


def test_each_epoch_trains_with_its_own_rate(data, monkeypatch):
    seen = []
    original = direct.Adam

    class Spy(original):
        def step(self, params, grads):
            seen.append(self.lr)
            return super().step(params, grads)

    monkeypatch.setattr(direct, "Adam", Spy)
    train_direct(_ctx(), _seeds(), data, epochs=4, batch_size=1000)
    per_epoch = [len(range(0, data.train_pairs.state.shape[0], 1000))] * 4
    expected = [r for r, n in zip(direct.learning_rates(4), per_epoch) for _ in range(n)]
    assert seen == expected


def test_the_seed_to_seed_spread_is_small_with_the_decay(data):
    """Why the decay exists (D18): different seeds on identical data should land close together."""
    errors = []
    for seed in (1, 2, 3):
        net = train_direct(_ctx(), SeedSource(seed, "direct"), data, epochs=30, batch_size=256)
        h = data.heldout_pairs
        means, _ = DirectModel(_ctx(), net).predict_batch(h.state, h.action)
        errors.append(float(np.mean(((means - h.next_state) / lv.WORLD.scale) ** 2)))
    assert max(errors) < 5 * min(errors)


# --- review code-review-002: a trained network is read-only ---------------------------------------


def test_a_trained_network_is_read_only_so_one_stray_write_cannot_change_every_model_built_on_it(trained_net):
    import copy

    for W, b in trained_net.layers:
        assert W.flags.writeable is False and b.flags.writeable is False
        with pytest.raises(ValueError):
            W[0, 0] = 0.0
        with pytest.raises(ValueError):
            b[:] = 0.0
    clone = copy.deepcopy(trained_net)  # a copy is a fresh, editable object (tests use this)
    clone.layers[-1][1][:] += 1.0
    assert not np.array_equal(clone.layers[-1][1], trained_net.layers[-1][1])


def test_the_network_freeze_locks_every_weight_and_bias_and_only_those():
    from wmj.models.mlp import MLP

    net = MLP([3, 4, 2], np.random.default_rng(0))
    assert all(W.flags.writeable and b.flags.writeable for W, b in net.layers)
    net.freeze()
    assert all(not W.flags.writeable and not b.flags.writeable for W, b in net.layers)
