"""P3-C04 real-scale gate: Model B trained on the frozen recipe, on both worlds.

In plain words: the unit tests train five tiny members for a few epochs. The real model
— five networks, 100 epochs each on 100,000 examples — lives in a regime the small tests
never reach (tiny members' disagreement, pushes up to 2, error ~1e-8). This trains the
real thing once per world (~3 min each) and checks only what is true on every machine:
the run consumed exactly 5 members × 100 epochs × 100,000 rows in batches of 256; the
point prediction and error bar equal an independent assembly from the members on every
held-out row; the model is clearly better than "nothing changes"; its error bars are
positive and small; and the pendulum's push sizes (±1, ±2) are learned. Training at
this length is chaotic across CPU kernels, so numeric checks are deliberately loose;
reference-machine numbers are in build/measurements/p3-c04-model-b-real-run.md.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from wmj.harness.training import build_training_data, make_world_context, read_training_recipe
from wmj.models import ensemble
from wmj.models.base import SeedSource
from wmj.models.ensemble import ensemble_factory
from wmj.worlds import lv, pendulum

SEED = 20260825
RECIPE = Path(__file__).resolve().parents[2] / "prereg" / "recipe.md"


@pytest.fixture(scope="module", params=[("lv", lv), ("pendulum", pendulum)], ids=["lv", "pendulum"])
def real(request):
    name, module = request.param
    recipe = read_training_recipe(RECIPE)
    ctx = make_world_context(name, module.WORLD)
    data = build_training_data(name, module.WORLD, SeedSource(SEED, None), recipe)
    batches = []
    original = ensemble.mse_loss_and_grad

    def spy(Y, target, scale):
        batches.append(Y.shape[0])
        return original(Y, target, scale)

    ensemble.mse_loss_and_grad = spy
    try:
        model = ensemble_factory(ctx, SeedSource(SEED, "ensemble"), data)
    finally:
        ensemble.mse_loss_and_grad = original
    return name, module, ctx, data, model, batches, recipe


@pytest.mark.slow
def test_the_real_run_consumes_5_members_of_100_epochs_of_100000_rows_in_batches_of_256(real):
    _name, _module, _ctx, _data, _model, batches, recipe = real
    gradient_check_calls = [b for b in batches if b == recipe.gradcheck_pairs]
    training_batches = batches[len(gradient_check_calls) :]  # the check's calls come first, member by member
    # per member: the gradient-check calls (many 64-row calls) then 100 epochs of 390 full + one of 160 rows
    per_member = 100 * (100000 // 256 + 1)
    assert sum(1 for b in batches if b == 256) == 5 * 100 * (100000 // 256)
    assert sum(1 for b in batches if b == 160) == 5 * 100
    assert sum(b for b in batches if b in (256, 160)) == 5 * 100 * 100000
    assert max(training_batches) == 256 and per_member == 39100


@pytest.mark.slow
def test_the_real_ensemble_follows_its_members_and_beats_nothing_changes(real):
    name, module, ctx, data, model, _batches, _recipe = real
    h = data.heldout_pairs
    means, spreads = model.predict_batch(h.state, h.action)
    hw = (ctx.training_action_interval[:, 1] - ctx.training_action_interval[:, 0]) / 2.0
    x = np.hstack([h.state / ctx.scale, h.action / hw])
    stack = np.stack([h.state + net.forward_invariant(x) for net in model._nets])
    assert np.allclose(means, stack.mean(axis=0), rtol=1e-12, atol=0)
    assert np.allclose(spreads, np.sqrt(1 + 1 / 5) * stack.std(axis=0, ddof=1), rtol=1e-9, atol=0)
    assert np.all(spreads > 0) and np.all(spreads < 1.0)
    scale = module.WORLD.scale
    error = float(np.mean(((means - h.next_state) / scale) ** 2))
    persistence = float(np.mean(((h.state - h.next_state) / scale) ** 2))
    assert error < 0.5 * persistence, f"{name}: error {error:.3e} vs persistence {persistence:.3e}"
    assert len({net.layers[0][0].tobytes() for net in model._nets}) == 5


@pytest.mark.slow
def test_the_real_pendulum_ensemble_learned_the_size_of_its_pushes(real):
    name, module, _ctx, _data, model, _batches, _recipe = real
    if name != "pendulum":
        pytest.skip("pendulum only")
    rng = np.random.default_rng(1)
    s = rng.uniform(-0.2, 0.2, (200, 4))
    for push, tolerance in ((1.0, 0.5), (-1.0, 0.5), (2.0, 0.7), (-2.0, 0.7)):
        a = np.full((200, 1), push)
        means, _ = model.predict_batch(s, a)
        truth = module.transition_batch(s, a)
        change = np.abs(truth - s).max(axis=0)
        dim = int(np.argmax(change))
        got = float(np.abs(means[:, dim] - s[:, dim]).max())
        assert abs(got - change[dim]) <= tolerance * change[dim], f"push {push}: {got:.3f} vs {change[dim]:.3f}"
