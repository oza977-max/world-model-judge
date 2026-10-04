"""P3-C03 real-scale gate: Model A trained on the frozen recipe, on both worlds.

In plain words: the unit tests train Model A for a few epochs on a couple of thousand
examples. The real model — 100 epochs on 50,000 examples — lives in a different regime:
error bars around a thousandth, steps of size 2 for the pendulum's biggest push, errors
around 1e-8. A bug that only acts in that regime (a floor, a cap, a clip) would pass every
small test. This trains the real thing (~17 s per world) and checks the promises that only
show at full size: predictions equal the network's own output exactly on every held-out row
and in the unfamiliar regions, the error is in the measured band, and the pendulum's big
pushes are really learned. Numbers are bands around build/measurements/p3-c03-model-a-real-run.md,
not goldens (training at this length is chaotic; a different machine may land elsewhere in
the band).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from wmj.harness.training import (
    build_training_data,
    make_world_context,
    read_training_recipe,
)
from wmj.models.base import SeedSource
from wmj.models.direct import direct_factory
from wmj.worlds import lv, pendulum

SEED = 20260825
RECIPE = Path(__file__).resolve().parents[2] / "prereg" / "recipe.md"


def _trained(name, module):
    ctx = make_world_context(name, module.WORLD)
    data = build_training_data(name, module.WORLD, SeedSource(SEED, None), read_training_recipe(RECIPE))
    model = direct_factory(ctx, SeedSource(SEED, "direct"), data)
    return ctx, data, model


def _expected(ctx, net, s, a):
    hw = (ctx.training_action_interval[:, 1] - ctx.training_action_interval[:, 0]) / 2.0
    out = net.forward_invariant(np.hstack([s / ctx.scale, a / hw]))
    return s + out[:, : ctx.state_dim], np.exp(out[:, ctx.state_dim :])


@pytest.mark.slow
@pytest.mark.parametrize(("name", "module", "band"), [("lv", lv, (5e-9, 5e-7)), ("pendulum", pendulum, (5e-9, 5e-7))])
def test_the_real_model_predicts_exactly_what_its_network_says_and_is_in_the_measured_band(name, module, band):
    ctx, data, model = _trained(name, module)
    h = data.heldout_pairs
    means, spreads = model.predict_batch(h.state, h.action)
    exp_means, exp_spreads = _expected(ctx, model._net, h.state, h.action)
    assert np.array_equal(means, exp_means) and np.array_equal(spreads, exp_spreads)
    assert np.all(spreads > 0) and np.all(spreads < 1.0)  # error bars are small in the familiar region
    error = float(np.mean(((means - h.next_state) / module.WORLD.scale) ** 2))
    assert band[0] < error < band[1], f"{name}: held-out error {error:.3e} outside the measured band"
    # spreads sit in the measured regime (~1e-3 to ~1e-2), not floored or inflated
    assert 3e-4 < float(np.median(spreads)) < 1e-2


@pytest.mark.slow
def test_the_real_pendulum_model_learned_the_size_of_its_pushes_including_the_large_ones():
    _ctx, _data, model = _trained("pendulum", pendulum)
    rng = np.random.default_rng(1)
    s = rng.uniform(-0.2, 0.2, (200, 4))
    for push, tolerance in ((1.0, 0.25), (-1.0, 0.25), (2.0, 0.6), (-2.0, 0.6)):
        a = np.full((200, 1), push)
        means, _ = model.predict_batch(s, a)
        truth = pendulum.transition_batch(s, a)
        change = np.abs(truth - s).max(axis=0)  # per dimension, the biggest change over the 200 states
        dim = int(np.argmax(change))  # the dimension the push moves (the angular speed)
        got = float(np.abs(means[:, dim] - s[:, dim]).max())
        assert abs(got - change[dim]) <= tolerance * change[dim], (
            f"push {push}: predicted change {got:.3f} vs true {change[dim]:.3f} in dimension {dim}"
        )


@pytest.mark.slow
def test_the_real_lv_model_responds_to_a_push_in_the_right_direction():
    _ctx, _data, model = _trained("lv", lv)
    s = np.tile(np.array([[4.0, 2.5]]), (3, 1))
    a = np.array([[0.1], [0.0], [-0.1]])
    means, _ = model.predict_batch(s, a)
    truth = lv.transition_batch(s, a)
    assert not np.array_equal(means[0], means[2])
    assert np.sign(means[0] - means[1]).tolist() == np.sign(truth[0] - truth[1]).tolist()
    assert np.sign(means[2] - means[1]).tolist() == np.sign(truth[2] - truth[1]).tolist()
