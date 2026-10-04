"""P3-C06 full-scale gate: the real recipe, the real worlds, the real sizes.

In plain words: the unit tests build small homework sets. This builds the
real ones — 2,000 full-length histories per world and the 50,000 / 12,500 /
10,000 / 64 picks the frozen recipe names — and checks the promises that
only show at full size: the worlds really supply enough kicked examples
(the build refuses otherwise), the numbers are exact, nothing breaches a
floor, and the same seed gives the same bytes. It also reports how many
kicked examples each world actually has, so the margin over 12,500 is
visible rather than assumed (models spec §4 expects LV ≈ 14,000 and the
pendulum ≈ 20,000).
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path

import numpy as np
import pytest

from wmj.harness.benchmarks import declared_regions, sample_region_starts
from wmj.harness.training import (
    assert_eval_starts_disjoint,
    build_training_data,
    read_training_recipe,
)
from wmj.models.base import SeedSource
from wmj.worlds import lv, pendulum

SEED = 20260825
RECIPE = Path(__file__).resolve().parents[2] / "prereg" / "recipe.md"
WORLDS = [("lv", lv), ("pendulum", pendulum)]


def _digest(data) -> str:
    h = hashlib.sha256()
    for array in (
        data.states, data.actions, data.train_pairs.state, data.train_pairs.action,
        data.train_pairs.next_state, data.train_pairs.is_kick, data.heldout_pairs.state,
        data.heldout_pairs.action, data.heldout_pairs.next_state, data.heldout_pairs.is_kick,
        data.gradcheck_index,
    ):
        h.update(np.ascontiguousarray(array).tobytes())
    return h.hexdigest()


# Known answers for seed 20260825 and the frozen recipe (numpy 1.26.4 on glibc;
# the pendulum half also rests on the platform's sin/cos, so a different libm
# could differ in the last bit — if so, say that, don't call it a recipe revision).
# If one of these fails the DATA changed: that is a recipe/seed revision to be
# made openly and re-reported (REMEMBER.md D17, backlog A19), never a number to
# update quietly. Pass 3 of the P3-C06 review: every other determinism test
# compares a build with another build of the same code, so none would notice
# `rng_for` switching generator or entropy order.
GOLDEN = {
    "lv": {"states": "510476bc873b", "available": 13874, "held_kicks": 1000},
    "pendulum": {"states": "0aed9d76c286", "available": 19900, "held_kicks": 1000},
}


@pytest.mark.parametrize(("name", "module"), WORLDS)
def test_the_real_recipe_builds_the_real_training_set(name, module, capsys):
    recipe = read_training_recipe(RECIPE)
    seeds = SeedSource(SEED, None)
    started = time.perf_counter()
    data = build_training_data(name, module.WORLD, seeds, recipe)  # real generator, real horizon
    seconds = time.perf_counter() - started

    n, h = recipe.training_trajectories, module.HORIZON
    assert data.states.shape == (n, h + 1, module.WORLD.d) and data.actions.shape == (n, h, 1)
    assert np.all(np.isfinite(data.states))
    assert data.train_pairs.state.shape[0] == recipe.subsample_pairs == 50000
    assert int(data.train_pairs.is_kick.sum()) == recipe.kick_pairs == 12500
    assert data.heldout_pairs.state.shape[0] == recipe.heldout_pairs == 10000
    assert data.gradcheck_index.shape == (64,)

    available = int(np.count_nonzero(np.any(data.actions != 0.0, axis=2)))
    held_kicks = int(data.heldout_pairs.is_kick.sum())
    with capsys.disabled():
        print(
            f"\n[P3-C06 full scale] {name}: built in {seconds:.1f} s; kick pairs available "
            f"{available} (need {recipe.kick_pairs}); held-out kick share {held_kicks}/"
            f"{recipe.heldout_pairs}"
        )
    assert available >= recipe.kick_pairs
    golden = GOLDEN[name]
    assert hashlib.sha256(np.ascontiguousarray(data.states).tobytes()).hexdigest()[:12] == golden["states"]
    assert available == golden["available"] and held_kicks == golden["held_kicks"]

    for region, box, _band in declared_regions(module.WORLD):
        eval_starts = sample_region_starts(seeds.rng_for(name, region, "eval-starts"), box, 200)
        assert_eval_starts_disjoint(data, eval_starts)


@pytest.mark.parametrize(("name", "module"), WORLDS)
def test_the_real_training_set_is_byte_identical_when_built_twice(name, module):
    recipe = read_training_recipe(RECIPE)
    first = build_training_data(name, module.WORLD, SeedSource(SEED, None), recipe)
    second = build_training_data(name, module.WORLD, SeedSource(SEED, None), recipe)
    assert _digest(first) == _digest(second)
