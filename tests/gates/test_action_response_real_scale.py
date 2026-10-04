"""P3-C08 real-scale gate: the action-response check on the real, frozen-recipe models.

In plain words: the unit tests use small, quickly trained networks. The real Model A
(100 epochs on 100,000 examples) is the one that has to be told apart from its
action-blind twin — and the Round 10 recipe made the predator-prey lever *smaller*
(kicks of at most 0.1), so the real question is whether the action still changes the
real model's answer by more than floating-point noise. This trains the real thing for
both worlds (≈ 33 s each) and checks, machine-independently: the real Model A passes
the check with a very large margin over the tolerance; `fx-action-blind`, built on the
very same network, is flagged with a change of exactly zero; and the other three models
that wrap Model A pass. The measured numbers are written up in
`build/measurements/p3-c08-action-response.md`.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import wmj.models.fixtures  # noqa: F401  (registers the fx-* models)
from wmj.harness.action_response import ACTION_RESPONSE_TOLERANCE, check_action_response
from wmj.harness.training import (
    build_training_data,
    make_world_context,
    read_training_recipe,
)
from wmj.models.base import SeedSource
from wmj.models.registry import all_models
from wmj.worlds import lv, pendulum

SEED = 20260825
RECIPE = Path(__file__).resolve().parents[2] / "prereg" / "recipe.md"
NAMES = ("direct", "fx-overconfident", "fx-honest-rough", "fx-brittle", "fx-action-blind")


@pytest.fixture(scope="module", params=[("lv", lv), ("pendulum", pendulum)], ids=["lv", "pendulum"])
def real(request):
    name, module = request.param
    ctx = make_world_context(name, module.WORLD)
    data = build_training_data(name, module.WORLD, SeedSource(SEED, None), read_training_recipe(RECIPE))
    models = {n: all_models()[n](ctx, SeedSource(SEED, n), data) for n in NAMES}
    return ctx, {n: check_action_response(m, ctx, SEED) for n, m in models.items()}


@pytest.mark.slow
def test_the_real_model_a_uses_its_action_by_a_wide_margin_even_with_the_small_lv_lever(real):
    _, results = real
    direct = results["direct"]
    assert direct.action_blind is False
    assert direct.n_responding == direct.n_probes  # every probe, not just one
    assert direct.largest_change > 1e5 * ACTION_RESPONSE_TOLERANCE


@pytest.mark.slow
def test_the_real_action_blind_fixture_is_flagged_with_a_change_of_exactly_zero(real):
    _, results = real
    blind = results["fx-action-blind"]
    assert blind.action_blind is True and blind.n_responding == 0 and blind.largest_change == 0.0


@pytest.mark.slow
def test_the_other_real_fixtures_that_wrap_model_a_pass_the_check(real):
    _, results = real
    for name in ("fx-overconfident", "fx-honest-rough", "fx-brittle"):
        assert results[name].action_blind is False, name
