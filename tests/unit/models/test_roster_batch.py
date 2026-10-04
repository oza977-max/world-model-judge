"""TC-MU1-04 over the whole registered roster (models ADR-M1, design-review-010).

In plain words: every model that says it has no memory between steps must give, row
for row, exactly the numbers it gives one at a time — for n = 1, 2, 7, 64 and 200 —
and every model that remembers (linear) must say so and never be batched. This walks
the real registry, so a new model that forgets the rule is caught without anyone
remembering to add a test for it.
"""

from __future__ import annotations

import numpy as np
import pytest

import wmj.models.baselines  # noqa: F401  (importing registers the model)
import wmj.models.direct  # noqa: F401
import wmj.models.ensemble  # noqa: F401
from wmj.harness.training import TrainingRecipe, build_training_data, make_world_context
from wmj.models.base import SeedSource
from wmj.models.registry import all_models
from wmj.worlds import lv

SEED = 20260825
RECIPE = TrainingRecipe(
    training_trajectories=100, subsample_pairs=1500, kick_pairs=40, heldout_pairs=300,
    gradcheck_pairs=16,
)


@pytest.fixture(scope="module")
def roster():
    data = build_training_data("lv", lv.WORLD, SeedSource(SEED, None), RECIPE, horizon=100)
    ctx = make_world_context("lv", lv.WORLD)
    built = {name: factory(ctx, SeedSource(SEED, name), data) for name, factory in all_models().items()}
    return data, built


def test_every_expected_model_is_registered():
    assert {"persistence", "linear", "direct", "ensemble"} <= set(all_models())


def test_every_registered_model_declares_whether_it_is_stateless(roster):
    _, built = roster
    for name, model in built.items():
        assert isinstance(model.stateless, bool), name
    assert built["linear"].stateless is False
    assert all(built[n].stateless for n in ("persistence", "direct", "ensemble"))


@pytest.mark.parametrize("n", [1, 2, 7, 64, 200])
def test_tc_mu1_04_every_stateless_model_batches_bit_identically(roster, n):
    data, built = roster
    held = data.heldout_pairs
    s, a = np.resize(held.state, (n, 2)), np.resize(held.action, (n, 1))
    checked = []
    for name, model in built.items():
        if not model.stateless:
            assert not hasattr(model, "predict_batch"), f"{name} remembers state but offers predict_batch"
            continue
        means, spreads = model.predict_batch(s, a)
        for i in range(n):
            p = model.predict(s[i], a[i])
            assert np.array_equal(means[i], p.mean) and np.array_equal(spreads[i], p.spread), name
        checked.append(name)
    assert set(checked) >= {"persistence", "direct", "ensemble"}


def test_the_stateful_baseline_is_never_batchable_and_resets_cleanly(roster):
    _, built = roster
    linear = built["linear"]
    first = linear.predict(np.array([3.0, 2.0]), np.zeros(1))
    second = linear.predict(np.array([3.1, 2.1]), np.zeros(1))
    assert not np.array_equal(first.mean, second.mean)
    linear.reset()
    again = linear.predict(np.array([3.0, 2.0]), np.zeros(1))
    assert np.array_equal(first.mean, again.mean)
