"""Tests for wmj.models.ensemble — Model B (models ADR-M3, TC-MU5-02/03, TC-MU1-04).

In plain words: Model B is five copies of one small network, each started from a
different random point and shown the examples in a different order. Their average is
its best guess; how much they disagree (corrected for having only five) is its error
bar. These tests check the loss's exact arithmetic, that each copy trains exactly as
an independent reading of the spec says, that the average and the error bar follow the
pre-registered rule, that predicting a batch equals predicting one at a time, and —
because the real model lives in a regime the small tests never reach (tiny error
bars, pushes up to 2, four dimensions, extreme states) — that nothing floors, caps,
clips or rescales there.
"""

from __future__ import annotations

import copy
import itertools
import math
import re
from pathlib import Path

import numpy as np
import pytest

from wmj.harness.training import TrainingRecipe, build_training_data, make_world_context
from wmj.models import direct, ensemble
from wmj.models.base import Pairs, SeedSource, TrainingData, WorldContext
from wmj.models.ensemble import (
    ENSEMBLE_MEMBERS,
    EnsembleModel,
    EnsembleTrainingError,
    ensemble_factory,
    mse_loss_and_grad,
    train_ensemble,
    train_member,
)
from wmj.models.registry import all_models
from wmj.worlds import lv, pendulum

SEED = 20260825
RECIPE_PATH = Path(__file__).resolve().parents[3] / "prereg" / "recipe.md"
LV_RECIPE = TrainingRecipe(
    training_trajectories=100, subsample_pairs=2000, kick_pairs=50, heldout_pairs=500,
    gradcheck_pairs=16,
)
PEND_RECIPE = TrainingRecipe(
    training_trajectories=300, subsample_pairs=1000, kick_pairs=40, heldout_pairs=300,
    gradcheck_pairs=16,
)


def _lv_ctx():
    return make_world_context("lv", lv.WORLD)


def _seeds():
    return SeedSource(SEED, "ensemble")


@pytest.fixture(scope="module")
def data():
    return build_training_data("lv", lv.WORLD, SeedSource(SEED, None), LV_RECIPE, horizon=100)


@pytest.fixture(scope="module")
def pend():
    d = build_training_data("pendulum", pendulum.WORLD, SeedSource(SEED, None), PEND_RECIPE, horizon=300)
    ctx = make_world_context("pendulum", pendulum.WORLD)
    return ctx, d, train_ensemble(ctx, _seeds(), d, epochs=2)


@pytest.fixture(scope="module")
def nets(data):
    return train_ensemble(_lv_ctx(), _seeds(), data, epochs=6)


@pytest.fixture(scope="module")
def model(nets):
    return EnsembleModel(_lv_ctx(), nets)


def _recipe_value(key):
    found = re.findall(rf"^{key}:[ \t]*([^#\s]*)", RECIPE_PATH.read_text(), re.MULTILINE)
    assert len(found) == 1
    return float(found[0])


# --- the pinned numbers --------------------------------------------------------------


def test_the_member_count_is_the_recipes_and_the_correction_is_sqrt_one_plus_one_over_k():
    assert ENSEMBLE_MEMBERS == _recipe_value("ensemble_members") == 5
    assert ensemble.SPREAD_CORRECTION == math.sqrt(1 + 1 / 5)


def test_the_shared_training_numbers_are_model_as_and_the_recipes():
    assert (direct.EPOCHS, direct.BATCH_SIZE) == (_recipe_value("epochs"), _recipe_value("batch_size"))


# --- the loss: exact arithmetic (finite differences) ---------------------------------


def _fd(Y, T, scale, eps=1e-6):
    grad = np.zeros_like(Y)
    for i, j in itertools.product(range(Y.shape[0]), range(Y.shape[1])):
        hi, lo = Y.copy(), Y.copy()
        hi[i, j] += eps
        lo[i, j] -= eps
        grad[i, j] = (mse_loss_and_grad(hi, T, scale)[0] - mse_loss_and_grad(lo, T, scale)[0]) / (2 * eps)
    return grad


def _case(seed=2, n=6, d=3):
    rng = np.random.default_rng(seed)
    return rng.normal(0, 0.5, (n, d)), rng.normal(0, 0.5, (n, d)), np.array([1.0, 2.0, 4.0])[:d]


def test_the_loss_is_the_mean_over_examples_and_dimensions_of_the_scaled_squared_error():
    Y, T, scale = _case()
    loss, _ = mse_loss_and_grad(Y, T, scale)
    assert loss == pytest.approx(float(np.mean(((Y - T) / scale) ** 2)), rel=1e-13)
    # and it is exactly the held-out error metric of ADR-M3 (mean over dimensions of the scaled square)
    assert loss == pytest.approx(float(np.mean(np.mean(((Y - T) / scale) ** 2, axis=1))), rel=1e-13)


def test_the_gradient_matches_finite_differences_and_the_formula():
    Y, T, scale = _case()
    _, grad = mse_loss_and_grad(Y, T, scale)
    assert np.allclose(grad, _fd(Y, T, scale), rtol=1e-6, atol=1e-10)
    n, d = Y.shape
    assert np.allclose(grad, 2 * (Y - T) / scale**2 / (n * d), rtol=1e-13, atol=0)


def test_a_gradient_missing_the_batch_division_or_the_scale_is_detected():
    Y, T, scale = _case()
    _, grad = mse_loss_and_grad(Y, T, scale)
    numeric = _fd(Y, T, scale)
    assert not np.allclose(grad * Y.shape[0], numeric, rtol=1e-3, atol=0)
    assert not np.allclose(2 * (Y - T) / (Y.shape[0] * Y.shape[1]), numeric, rtol=1e-3, atol=0)


def test_the_loss_refuses_bad_shapes_and_non_finite_values():
    Y, T, scale = _case()
    with pytest.raises(EnsembleTrainingError, match="shape"):
        mse_loss_and_grad(Y[:, :-1], T, scale)
    with pytest.raises(EnsembleTrainingError, match="shape"):
        mse_loss_and_grad(Y, T, scale[:-1])
    bad = Y.copy()
    bad[0, 0] = np.nan
    with pytest.raises(EnsembleTrainingError, match="finite"):
        mse_loss_and_grad(bad, T, scale)


# --- independent reference trainer (written from the spec; shares no code) ------------


def _reference_member(ctx, seeds, pairs, k, *, epochs, batch_size, bias0=None):
    d = ctx.state_dim
    hw = (ctx.training_action_interval[:, 1] - ctx.training_action_interval[:, 0]) / 2.0
    X = np.hstack([pairs.state / ctx.scale, pairs.action / hw])
    T = pairs.next_state - pairs.state
    sizes = [X.shape[1], 64, 64, d]
    rng = seeds.rng("member", str(k), "weights")
    Ws = [rng.uniform(-1 / np.sqrt(i), 1 / np.sqrt(i), size=(i, o)) for i, o in itertools.pairwise(sizes)]
    bs = [np.zeros(o) for o in sizes[1:]]
    if bias0 is not None:
        bs[2][:] = bias0
    mw = [np.zeros_like(w) for w in Ws]; vw = [np.zeros_like(w) for w in Ws]
    mb = [np.zeros_like(b) for b in bs]; vb = [np.zeros_like(b) for b in bs]
    t = 0
    lrs = [1e-3] if epochs == 1 else [
        1e-5 + 0.5 * (1e-3 - 1e-5) * (1 + np.cos(np.pi * e / (epochs - 1))) for e in range(epochs)
    ]
    for epoch in range(epochs):
        lr = lrs[epoch]
        order = seeds.rng("member", str(k), "shuffle", str(epoch)).permutation(X.shape[0])
        for start in range(0, X.shape[0], batch_size):
            rows = order[start : start + batch_size]
            x, y = X[rows], T[rows]
            acts = [x]
            for layer in range(3):
                pre = acts[-1] @ Ws[layer] + bs[layer]
                acts.append(np.tanh(pre) if layer < 2 else pre)
            g = 2 * (acts[-1] - y) / ctx.scale**2 / (x.shape[0] * d)
            dW, db = [None] * 3, [None] * 3
            for layer in (2, 1, 0):
                dW[layer] = acts[layer].T @ g
                db[layer] = g.sum(axis=0)
                if layer:
                    g = (g @ Ws[layer].T) * (1 - acts[layer] ** 2)
            t += 1
            for layer in range(3):
                for p, gr, m, v in ((Ws[layer], dW[layer], mw[layer], vw[layer]),
                                    (bs[layer], db[layer], mb[layer], vb[layer])):
                    m *= 0.9; m += 0.1 * gr
                    v *= 0.999; v += 0.001 * gr**2
                    p -= lr * (m / (1 - 0.9**t)) / (np.sqrt(v / (1 - 0.999**t)) + 1e-8)
    return list(zip(Ws, bs))


def _close(net, reference, rtol=1e-7, atol=1e-10):
    for (W, b), (Wr, br) in zip(net.layers, reference):
        assert np.allclose(W, Wr, rtol=rtol, atol=atol) and np.allclose(b, br, rtol=rtol, atol=atol)


@pytest.mark.parametrize("k", range(5))
def test_every_member_equals_the_independent_trainer_at_three_epochs(data, k):
    net = train_member(_lv_ctx(), _seeds(), data, k, epochs=3, batch_size=64)
    _close(net, _reference_member(_lv_ctx(), _seeds(), data.train_pairs, k, epochs=3, batch_size=64))


def test_the_default_batch_size_is_256(data):
    net = train_member(_lv_ctx(), _seeds(), data, 1, epochs=2)
    _close(net, _reference_member(_lv_ctx(), _seeds(), data.train_pairs, 1, epochs=2, batch_size=256))


@pytest.mark.parametrize("k", [0, 3])
def test_a_hundred_epochs_equal_the_independent_trainer_on_a_small_slice(data, k):
    p = data.train_pairs
    rows = np.r_[0:40, 1000:1560]
    small = Pairs(p.state[rows], p.action[rows], p.next_state[rows], p.is_kick[rows])
    td = TrainingData(states=data.states, actions=data.actions, train_pairs=small,
                      gradcheck_index=np.arange(16, dtype=np.int64))
    net = train_member(_lv_ctx(), _seeds(), td, k, epochs=direct.EPOCHS, batch_size=direct.BATCH_SIZE)
    _close(net, _reference_member(_lv_ctx(), _seeds(), small, k, epochs=100, batch_size=256),
           rtol=1e-6, atol=1e-8)


def test_training_from_a_very_small_starting_output_matches_the_independent_trainer(data):
    """The change outputs start far from where real training ends up; a floor or clip on the
    output or the gradient would show here (and at the pendulum's change of ~2 below)."""
    ctx = _lv_ctx()
    original = ensemble.MLP

    class Shifted(original):
        def __init__(self, sizes, rng):
            super().__init__(sizes, rng)
            self.layers[-1][1][:] = 1.5

    ensemble.MLP = Shifted
    try:
        net = train_member(ctx, _seeds(), data, 0, epochs=3, batch_size=256)
    finally:
        ensemble.MLP = original
    _close(net, _reference_member(ctx, _seeds(), data.train_pairs, 0, epochs=3, batch_size=256, bias0=1.5))


# --- members, streams, determinism ----------------------------------------------------


def test_there_are_k_members_each_with_its_own_weights(nets):
    assert len(nets) == ENSEMBLE_MEMBERS == 5
    firsts = [n.layers[0][0].tobytes() for n in nets]
    assert len(set(firsts)) == 5
    assert all(n.layer_sizes == (3, 64, 64, 2) for n in nets)  # no variance head: d outputs


def test_each_member_draws_from_its_own_named_streams(data, monkeypatch):
    asked = []
    original = SeedSource.rng

    def spy(self, *purpose):
        asked.append(purpose)
        return original(self, *purpose)

    monkeypatch.setattr(SeedSource, "rng", spy)
    train_ensemble(_lv_ctx(), _seeds(), data, epochs=2)
    expected = []
    for k in range(5):
        expected += [("member", str(k), "weights"), ("member", str(k), "shuffle", "0"),
                     ("member", str(k), "shuffle", "1")]
    assert asked == expected


def test_training_twice_from_one_seed_is_identical_and_another_seed_differs(data, nets):
    again = train_ensemble(_lv_ctx(), _seeds(), data, epochs=6)
    for a, b in zip(nets, again):
        assert all(W.tobytes() == W2.tobytes() for (W, _), (W2, _) in zip(a.layers, b.layers))
    other = train_ensemble(_lv_ctx(), SeedSource(SEED + 1, "ensemble"), data, epochs=1)
    assert other[0].layers[0][0].tobytes() != nets[0].layers[0][0].tobytes()


def test_the_gradient_check_runs_first_for_every_member_under_mse(data, monkeypatch):
    calls = []
    original = ensemble.gradient_check

    def spy(net, X, loss_and_grad, **kw):
        _, grad = loss_and_grad(np.zeros((X.shape[0], 2)))
        calls.append((X.copy(), grad.copy(), [W.copy() for W, _ in net.layers], kw))
        return original(net, X, loss_and_grad, **kw)

    monkeypatch.setattr(ensemble, "gradient_check", spy)
    nets_ = train_ensemble(_lv_ctx(), _seeds(), data, epochs=1)
    assert len(calls) == 5
    idx = data.gradcheck_index
    ctx = _lv_ctx()
    hw = (ctx.training_action_interval[:, 1] - ctx.training_action_interval[:, 0]) / 2.0
    expected_X = np.hstack([data.train_pairs.state[idx] / ctx.scale, data.train_pairs.action[idx] / hw])
    target = data.train_pairs.next_state[idx] - data.train_pairs.state[idx]
    for k, (X, grad, weights_then, kw) in enumerate(calls):
        assert np.array_equal(X, expected_X)
        assert np.allclose(grad, -2 * target / ctx.scale**2 / (len(idx) * 2), rtol=1e-12, atol=0)  # MSE at output 0
        fresh = ensemble.MLP((3, 64, 64, 2), _seeds().rng("member", str(k), "weights"))
        assert all(np.array_equal(W, W0) for (W, _), W0 in zip(fresh.layers, weights_then))
        assert kw.get("tolerance") == 1e-5
    assert nets_[0].layers[0][0].tobytes() != calls[0][2][0].tobytes()


def test_a_failing_gradient_check_stops_training(data, monkeypatch):
    from wmj.models.mlp import GradientCheckError

    def broken(*_a, **_k):
        raise GradientCheckError("backprop disagrees")

    monkeypatch.setattr(ensemble, "gradient_check", broken)
    with pytest.raises(GradientCheckError):
        train_ensemble(_lv_ctx(), _seeds(), data, epochs=1)


def test_each_epoch_trains_with_the_decayed_rate_and_the_pinned_batches(data, monkeypatch):
    rates, sizes = [], []
    original_adam, original_loss = ensemble.Adam, ensemble.mse_loss_and_grad

    class Spy(original_adam):
        def step(self, params, grads):
            rates.append(self.lr)
            return super().step(params, grads)

    def loss_spy(Y, T, scale):
        sizes.append(Y.shape[0])
        return original_loss(Y, T, scale)

    monkeypatch.setattr(ensemble, "Adam", Spy)
    monkeypatch.setattr(ensemble, "mse_loss_and_grad", loss_spy)
    train_member(_lv_ctx(), _seeds(), data, 0, epochs=3, batch_size=256)
    m = data.train_pairs.state.shape[0]
    per_epoch = len(range(0, m, 256))
    expected = [r for r in direct.learning_rates(3) for _ in range(per_epoch)]
    assert rates == expected
    # the training batches (the gradient check also calls the loss, 64 rows at a time)
    train_batches = [s for s in sizes if s != 16]
    assert sum(train_batches) == 3 * m and max(train_batches) == 256


# --- refusals ---------------------------------------------------------------------------


def test_training_data_without_pairs_is_refused():
    bare = TrainingData(states=np.zeros((2, 4, 2)), actions=np.zeros((2, 3, 1)))
    with pytest.raises(EnsembleTrainingError, match="train_pairs"):
        train_ensemble(_lv_ctx(), _seeds(), bare, epochs=1)
    with pytest.raises(EnsembleTrainingError, match="train_pairs"):
        ensemble_factory(_lv_ctx(), _seeds(), bare)


def test_a_missing_gradcheck_index_or_empty_training_set_is_refused():
    pairs = Pairs(np.zeros((5, 2)), np.zeros((5, 1)), np.ones((5, 2)), np.zeros(5, dtype=bool))
    states, actions = np.zeros((2, 4, 2)), np.zeros((2, 3, 1))
    with pytest.raises(EnsembleTrainingError, match="gradcheck_index"):
        train_ensemble(_lv_ctx(), _seeds(), TrainingData(states=states, actions=actions, train_pairs=pairs), epochs=1)
    empty = Pairs(np.zeros((0, 2)), np.zeros((0, 1)), np.zeros((0, 2)), np.zeros(0, dtype=bool))
    with pytest.raises(EnsembleTrainingError, match="empty|no training"):
        train_ensemble(_lv_ctx(), _seeds(), TrainingData(states=states, actions=actions, train_pairs=empty), epochs=1)


def test_a_context_that_does_not_match_the_data_is_refused(data):
    ctx = WorldContext(world_name="lv", state_dim=3, action_dim=1, training_state_box=np.zeros((3, 2)),
                       training_action_interval=np.array([[-0.1, 0.1]]), scale=np.ones(3))
    with pytest.raises(EnsembleTrainingError, match="width"):
        train_ensemble(ctx, _seeds(), data, epochs=1)


@pytest.mark.parametrize("kw", [{"epochs": 0}, {"epochs": 2.0}, {"batch_size": 0}, {"batch_size": True},
                                {"members": 1}, {"members": 0}, {"members": 2.5}, {"members": True}])
def test_nonsense_training_arguments_are_refused(data, kw):
    with pytest.raises(EnsembleTrainingError):
        train_ensemble(_lv_ctx(), _seeds(), data, **{"epochs": 1, **kw})


def test_diverging_training_is_a_loud_error(data, monkeypatch):
    monkeypatch.setattr(direct, "LEARNING_RATE", float("inf"))
    with pytest.raises(EnsembleTrainingError, match="finite|diverg"):
        train_member(_lv_ctx(), _seeds(), data, 0, epochs=3, batch_size=64)


def test_a_last_step_blow_up_is_caught_at_the_epoch_end(data, monkeypatch):
    monkeypatch.setattr(direct, "LEARNING_RATE", float("inf"))
    with pytest.raises(EnsembleTrainingError, match="not finite after epoch"):
        train_member(_lv_ctx(), _seeds(), data, 0, epochs=1, batch_size=10**6)


# --- prediction: TC-MU5-02 (point), TC-MU5-03 (spread), TC-MU1-01/02 (format) -----------


def _member_means(ctx, nets_, s, a):
    hw = (ctx.training_action_interval[:, 1] - ctx.training_action_interval[:, 0]) / 2.0
    x = np.hstack([s / ctx.scale, a / hw])
    return np.stack([s + net.forward_invariant(x) for net in nets_])


def _expected(ctx, nets_, s, a):
    stack = _member_means(ctx, nets_, s, a)
    k = len(nets_)
    return stack.mean(axis=0), np.sqrt(1 + 1 / k) * stack.std(axis=0, ddof=1)


def test_tc_mu5_02_the_point_prediction_is_the_mean_of_the_member_means(model, nets, data):
    h = data.heldout_pairs
    means, _ = model.predict_batch(h.state, h.action)
    exp_means, _ = _expected(_lv_ctx(), nets, h.state, h.action)
    assert np.allclose(means, exp_means, rtol=1e-13, atol=0)
    other = _member_means(_lv_ctx(), nets, h.state, h.action)[0]  # a single member is NOT the answer
    assert not np.allclose(means, other, rtol=1e-6, atol=0)


def test_tc_mu5_03_the_spread_is_the_corrected_sample_standard_deviation(model, nets, data):
    h = data.heldout_pairs
    _, spreads = model.predict_batch(h.state, h.action)
    _, exp_spreads = _expected(_lv_ctx(), nets, h.state, h.action)
    assert np.allclose(spreads, exp_spreads, rtol=1e-12, atol=0)
    stack = _member_means(_lv_ctx(), nets, h.state, h.action)
    assert not np.allclose(spreads, stack.std(axis=0, ddof=1), rtol=1e-3)  # the correction is applied
    assert not np.allclose(spreads, np.sqrt(1 + 1 / 5) * stack.std(axis=0, ddof=0), rtol=1e-3)  # sample, not population


def test_the_correction_follows_the_member_count(data):
    ctx = _lv_ctx()
    three = train_ensemble(ctx, _seeds(), data, epochs=1, members=3)
    h = data.heldout_pairs
    _means, sp = EnsembleModel(ctx, three).predict_batch(h.state[:50], h.action[:50])
    _exp_means, esp = _expected(ctx, three, h.state[:50], h.action[:50])
    assert np.allclose(sp, esp, rtol=1e-12, atol=0) and np.sqrt(1 + 1 / 3) != np.sqrt(1 + 1 / 5)


def test_flags_name_and_registry(model, data):
    assert (model.name, model.is_fixture, model.is_baseline, model.stateless) == ("ensemble", False, False, True)
    model.reset()
    assert all_models()["ensemble"] is ensemble_factory


def test_the_prediction_format_is_the_same_as_every_other_model(model, data):
    p = model.predict(data.heldout_pairs.state[0], data.heldout_pairs.action[0])
    assert p.mean.shape == (2,) and p.spread.shape == (2,) and p.mean.dtype == p.spread.dtype == np.float64
    assert np.all(p.spread > 0) and np.all(np.isfinite(p.mean))
    with pytest.raises(ValueError):
        p.mean[0] = 0.0
    assert p.mean.base is None or not p.mean.base.flags.writeable


@pytest.mark.parametrize("n", [1, 2, 7, 64, 200])
def test_tc_mu1_04_batch_rows_are_bit_identical_to_one_at_a_time(model, data, n):
    s = np.resize(data.heldout_pairs.state, (n, 2))
    a = np.resize(data.heldout_pairs.action, (n, 1))
    means, spreads = model.predict_batch(s, a)
    for i in range(n):
        p = model.predict(s[i], a[i])
        assert np.array_equal(means[i], p.mean) and np.array_equal(spreads[i], p.spread)


@pytest.mark.parametrize("n", [1, 7, 200])
def test_pendulum_batch_rows_are_bit_identical_to_one_at_a_time(pend, n):
    ctx, d, nets_ = pend
    model = EnsembleModel(ctx, nets_)
    s = np.resize(d.heldout_pairs.state, (n, 4))
    a = np.resize(d.heldout_pairs.action, (n, 1))
    means, spreads = model.predict_batch(s, a)
    for i in range(n):
        p = model.predict(s[i], a[i])
        assert np.array_equal(means[i], p.mean) and np.array_equal(spreads[i], p.spread)


# --- the real regime: nothing floors, caps, clips, rescales or ignores the push --------


def test_predictions_follow_the_members_on_every_heldout_row_and_the_unfamiliar_corners(model, nets, data):
    h = data.heldout_pairs
    corner_s = np.array([[8.0, 4.0], [12.0, 6.0], [15.5, 8.6], [15.5, 1.0], [1.0, 8.6], [3.0, 2.0], [3.0, 2.0]])
    corner_a = np.array([[0.2], [-0.2], [0.15], [-0.15], [0.2], [0.2], [-0.2]])
    s, a = np.vstack([h.state, corner_s]), np.vstack([h.action, corner_a])
    means, spreads = model.predict_batch(s, a)
    exp_means, exp_spreads = _expected(_lv_ctx(), nets, s, a)
    assert np.allclose(means, exp_means, rtol=1e-13, atol=0) and np.allclose(spreads, exp_spreads, rtol=1e-12, atol=0)


def test_pendulum_predictions_follow_the_members_at_extreme_states_and_large_pushes(pend):
    ctx, d, nets_ = pend
    h = d.heldout_pairs
    cs = np.array([[8.8, 39.1, 8.9, 12.6], [-8.8, -39.1, -8.9, -12.6], [3.5, 6.5, 0.0, 0.0],
                   [2.0, 60.0, 0.0, 0.0], [1.0, -60.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
    ca = np.array([[0.0], [1.0], [-1.0], [0.0], [2.0], [2.0]])
    s, a = np.vstack([h.state, cs]), np.vstack([h.action, ca])
    means, spreads = EnsembleModel(ctx, nets_).predict_batch(s, a)
    exp_means, exp_spreads = _expected(ctx, nets_, s, a)
    assert np.allclose(means, exp_means, rtol=1e-13, atol=0) and np.allclose(spreads, exp_spreads, rtol=1e-12, atol=0)


def test_the_push_reaches_the_network_it_is_not_dropped_negated_or_scaled(model, nets):
    ctx = _lv_ctx()
    s = np.array([[3.0, 2.0], [3.0, 2.0], [4.5, 2.0], [3.0, 2.0]])
    a = np.array([[0.09], [-0.09], [0.06], [0.0]])
    means, spreads = model.predict_batch(s, a)
    exp_means, exp_spreads = _expected(ctx, nets, s, a)
    assert np.allclose(means, exp_means, rtol=1e-13, atol=0) and np.allclose(spreads, exp_spreads, rtol=1e-12, atol=0)
    assert not np.array_equal(means[0], means[1]) and not np.array_equal(means[0], means[3])
    for i in range(4):
        p = model.predict(s[i], a[i])
        assert np.array_equal(p.mean, means[i]) and np.array_equal(p.spread, spreads[i])


@pytest.mark.parametrize("disagreement", [1e-6, 1e-5, 1e-4, 1e-3, 15.0])
def test_the_spread_is_faithful_at_the_disagreement_sizes_real_training_produces(nets, data, disagreement):
    """Real members agree to ~1e-3 (error bars ~1e-3); at extreme states they differ by ~10. A
    'minimum std' floor or a cap would change what the judge scores there."""
    ctx = _lv_ctx()
    shifted = copy.deepcopy(nets)
    for k, net in enumerate(shifted):
        net.layers[-1][1][:] += (k - 2) * disagreement  # members offset symmetrically
    h = data.heldout_pairs
    means, spreads = EnsembleModel(ctx, shifted).predict_batch(h.state, h.action)
    exp_means, exp_spreads = _expected(ctx, shifted, h.state, h.action)
    assert np.allclose(means, exp_means, rtol=1e-13, atol=0) and np.allclose(spreads, exp_spreads, rtol=1e-5, atol=0)
    assert float(np.median(spreads)) > 0.5 * disagreement  # roughly sqrt(1.2) * std of k*disagreement offsets (~1.58x)


@pytest.mark.parametrize("shift", [1.0, -1.5, 2.0])
def test_predicted_changes_of_the_size_a_kicked_pendulum_reaches_are_not_clipped(pend, shift):
    ctx, d, nets_ = pend
    shifted = copy.deepcopy(nets_)
    for net in shifted:
        net.layers[-1][1][:] += shift
    h = d.heldout_pairs
    means, spreads = EnsembleModel(ctx, shifted).predict_batch(h.state, h.action)
    exp_means, exp_spreads = _expected(ctx, shifted, h.state, h.action)
    assert np.allclose(means, exp_means, rtol=1e-13, atol=0) and np.allclose(spreads, exp_spreads, rtol=1e-9, atol=0)
    assert np.abs(means - h.state).max() > 0.9


def test_zero_disagreement_is_refused_not_returned_as_a_zero_width_forecast(nets, data):
    clones = [copy.deepcopy(nets[0]) for _ in range(5)]
    h = data.heldout_pairs
    with pytest.raises(EnsembleTrainingError, match="spread"):
        EnsembleModel(_lv_ctx(), clones).predict_batch(h.state[:3], h.action[:3])


def test_input_guards(model):
    with pytest.raises(EnsembleTrainingError, match="shape"):
        model.predict_batch(np.zeros((3, 5)), np.zeros((3, 1)))
    with pytest.raises(EnsembleTrainingError, match="shape"):
        model.predict_batch(np.zeros((3, 2)), np.zeros((4, 1)))
    with pytest.raises(EnsembleTrainingError, match="finite"):
        model.predict(np.array([np.nan, 1.0]), np.zeros(1))
    with pytest.raises(EnsembleTrainingError, match="one state"):
        model.predict(np.zeros((2, 2)), np.zeros(1))


def test_training_targets_are_not_clipped_at_the_size_of_a_kick():
    rng = np.random.default_rng(0)
    n = 300
    ctx = make_world_context("pendulum", pendulum.WORLD)
    s, a = rng.normal(0, 0.3, (n, 4)), rng.uniform(-1, 1, (n, 1))
    ns = s.copy()
    ns[:, 2] += 2.0 * a[:, 0]
    td = TrainingData(states=np.zeros((2, 4, 4)), actions=np.zeros((2, 3, 1)),
                      train_pairs=Pairs(s, a, ns, np.abs(a[:, 0]) > 0),
                      gradcheck_index=np.arange(16, dtype=np.int64))
    net = train_member(ctx, _seeds(), td, 0, epochs=80, batch_size=64)
    hw = (ctx.training_action_interval[:, 1] - ctx.training_action_interval[:, 0]) / 2.0
    out = net.forward_invariant(np.hstack([s / ctx.scale, a / hw]))
    assert np.abs(s[:, 2] + out[:, 2] - ns[:, 2]).max() < 0.5 and np.abs(out[:, 2]).max() > 1.5


def test_it_learns_something_real_and_is_a_different_model_from_direct(model, data):
    h = data.heldout_pairs
    means, _ = model.predict_batch(h.state, h.action)
    err = float(np.mean(((means - h.next_state) / lv.WORLD.scale) ** 2))
    persistence = float(np.mean(((h.state - h.next_state) / lv.WORLD.scale) ** 2))
    assert err < 0.5 * persistence


# --- author's mutation check: defaults and the single-member entry point ---


def test_a_member_trained_directly_without_pairs_is_refused_by_name():
    bare = TrainingData(states=np.zeros((2, 4, 2)), actions=np.zeros((2, 3, 1)))
    with pytest.raises(EnsembleTrainingError, match="train_pairs"):
        train_member(_lv_ctx(), _seeds(), bare, 0, epochs=1)


def test_the_factory_and_the_defaults_train_exactly_the_recipes_100_epochs_of_batches_of_256(data):
    """What the registry and P6-C01 call must be the recipe: compared bit-exactly with an explicit
    run, so an off-by-one epoch or a batch of 255 anywhere in the defaults changes the bytes."""
    built = ensemble_factory(_lv_ctx(), _seeds(), data)
    explicit = train_ensemble(
        _lv_ctx(), _seeds(), data, members=5, epochs=direct.EPOCHS, batch_size=direct.BATCH_SIZE
    )
    assert len(built._nets) == 5
    for a, b in zip(built._nets, explicit):
        assert all(W.tobytes() == W2.tobytes() and x.tobytes() == x2.tobytes()
                   for (W, x), (W2, x2) in zip(a.layers, b.layers))
    single = train_member(_lv_ctx(), _seeds(), data, 2)
    assert all(W.tobytes() == W2.tobytes() for (W, _), (W2, _) in zip(single.layers, explicit[2].layers))
    assert (direct.EPOCHS, direct.BATCH_SIZE) == (100, 256)
