"""P3-C03 real-scale gate: Model A trained on the frozen recipe, on both worlds.

In plain words: the unit tests train Model A for a few epochs on a couple of thousand
examples. The real model — 100 epochs on 50,000 examples — lives in a different regime:
error bars around a thousandth, steps of size 2 for the pendulum's biggest push, errors
around 1e-8. A bug that only acts in that regime (a floor, a cap, a clip, a shortcut in how
much training is done) would pass every small test. This trains the real thing (~17 s per
world) and checks only what is true on every machine: the real run consumed exactly 100
epochs of 50,000 rows in batches of 256; predictions equal the network's own output exactly on
every held-out row; and the model is clearly better than "nothing changes" and has learned
the pendulum's push sizes. Training at this length is chaotic — a different CPU kernel lands
elsewhere in the measured range (measured: LV held-out error 5e-8 to 1.3e-5 across kernels) —
so the numeric checks are deliberately loose; the exact numbers for the reference machine are
in build/measurements/p3-c03-model-a-real-run.md. The unfamiliar-region predictions are
pinned exactly by the unit tests (tests/unit/models/test_direct.py).
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
@pytest.mark.parametrize(("name", "module"), [("lv", lv), ("pendulum", pendulum)])
def test_the_real_run_consumes_100_epochs_of_50000_rows_in_batches_of_256(name, module, monkeypatch):
    from wmj.models import direct

    batches = []
    original = direct.beta_nll_loss_and_grad

    def spy(Y, target, *, beta):
        if beta == direct.BETA_NLL:  # training batches only (the gradient check uses beta 0)
            batches.append(Y.shape[0])
        return original(Y, target, beta=beta)

    monkeypatch.setattr(direct, "beta_nll_loss_and_grad", spy)
    _trained(name, module)
    per_epoch, last = divmod(50000, 256)
    assert len(batches) == 100 * (per_epoch + 1)  # 195 full batches + one of 80, 100 times
    assert sum(batches) == 100 * 50000 and max(batches) == 256 and last == 80


@pytest.mark.slow
@pytest.mark.parametrize(("name", "module"), [("lv", lv), ("pendulum", pendulum)])
def test_the_real_model_predicts_what_its_network_says_and_beats_nothing_changes(name, module):
    ctx, data, model = _trained(name, module)
    h = data.heldout_pairs
    means, spreads = model.predict_batch(h.state, h.action)
    exp_means, exp_spreads = _expected(ctx, model._net, h.state, h.action)
    assert np.array_equal(means, exp_means) and np.array_equal(spreads, exp_spreads)
    assert np.all(spreads > 0) and np.all(spreads < 1.0)
    scale = module.WORLD.scale
    error = float(np.mean(((means - h.next_state) / scale) ** 2))
    persistence = float(np.mean(((h.state - h.next_state) / scale) ** 2))
    assert error < 0.5 * persistence, f"{name}: error {error:.3e} vs persistence {persistence:.3e}"
    assert 1e-4 < float(np.median(spreads)) < 1e-1


@pytest.mark.slow
def test_the_real_pendulum_model_learned_the_size_of_its_pushes_including_the_large_ones():
    _ctx, _data, model = _trained("pendulum", pendulum)
    rng = np.random.default_rng(1)
    s = rng.uniform(-0.2, 0.2, (200, 4))
    for push, tolerance in ((1.0, 0.5), (-1.0, 0.5), (2.0, 0.7), (-2.0, 0.7)):
        a = np.full((200, 1), push)
        means, _ = model.predict_batch(s, a)
        truth = pendulum.transition_batch(s, a)
        change = np.abs(truth - s).max(axis=0)  # per dimension, the biggest change over the 200 states
        dim = int(np.argmax(change))  # the dimension the push moves (the angular speed)
        got = float(np.abs(means[:, dim] - s[:, dim]).max())
        assert abs(got - change[dim]) <= tolerance * change[dim], (
            f"push {push}: predicted change {got:.3f} vs true {change[dim]:.3f} in dimension {dim}"
        )
