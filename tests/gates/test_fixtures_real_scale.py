"""P3-C05 real-scale gate: the three broken models built on the real, frozen-recipe Model A.

In plain words: the unit tests build the broken models on a small, quickly trained network.
The real ones sit on the real network (100 epochs, 100,000 examples) where error bars are
about a thousandth and a jitter of "twice the error bar" is tiny in absolute terms. A rule
that only misbehaves there (an added floor, a clip, a fixed-size noise) would pass the small
tests. This builds `direct` and all three fixtures from one real training run and checks, on
every real held-out example and on far-out-of-range inputs: the network is trained exactly
once and shared; `fx-overconfident` differs from Model A only in an error bar a quarter the
size; `fx-honest-rough`'s jitter is a standard normal in units of two error bars and its bar is
the exact root-sum-of-squares; `fx-brittle` equals Model A at home and "nothing changes" away.
Only machine-independent relations are checked (training at this length is chaotic across
CPU kernels), never the trained numbers themselves.
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
from wmj.models import direct
from wmj.models.base import SeedSource
from wmj.models.registry import all_models
from wmj.worlds import lv, pendulum

SEED = 20260825
RECIPE = Path(__file__).resolve().parents[2] / "prereg" / "recipe.md"
NAMES = ("direct", "fx-overconfident", "fx-honest-rough", "fx-brittle", "fx-action-blind")


@pytest.fixture(scope="module", params=[("lv", lv), ("pendulum", pendulum)], ids=["lv", "pendulum"])
def real(request):
    name, module = request.param
    import wmj.models.fixtures  # noqa: F401  (registers the fx-* models)

    ctx = make_world_context(name, module.WORLD)
    data = build_training_data(name, module.WORLD, SeedSource(SEED, None), read_training_recipe(RECIPE))
    calls = []
    original = direct.train_direct

    def counting(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    direct.train_direct = counting
    try:
        models = {n: all_models()[n](ctx, SeedSource(SEED, n), data) for n in NAMES}
    finally:
        direct.train_direct = original
    return ctx, data, models, len(calls)


@pytest.mark.slow
def test_the_real_network_is_trained_once_and_shared_by_direct_and_the_three_fixtures(real):
    _, _, models, trainings = real
    assert trainings == 1
    assert models["fx-overconfident"]._inner._net is models["direct"]._net


@pytest.mark.slow
def test_overconfident_is_model_a_with_exactly_a_quarter_error_bar_on_every_real_row(real):
    _, data, models, _ = real
    h = data.heldout_pairs
    m_a, sp_a = models["direct"].predict_batch(h.state, h.action)
    m, sp = models["fx-overconfident"].predict_batch(h.state, h.action)
    assert np.array_equal(m, m_a) and np.array_equal(sp, sp_a * 0.25)
    assert np.all(sp > 0) and np.all(sp < 0.25)


@pytest.mark.slow
def test_honest_rough_jitter_is_standard_normal_in_units_of_two_real_error_bars(real):
    _, data, models, _ = real
    h = data.heldout_pairs
    m_a, sp_a = models["direct"].predict_batch(h.state, h.action)
    m, sp = models["fx-honest-rough"].predict_batch(h.state, h.action)
    assert np.array_equal(sp, np.sqrt(sp_a**2 + (2.0 * sp_a) ** 2))
    z = (m - m_a) / (2.0 * sp_a)
    assert z.shape[0] >= 5000
    assert np.all(np.abs(z.mean(axis=0)) < 0.05) and np.all(np.abs(z.std(axis=0) - 1.0) < 0.05)
    assert np.max(np.abs(z)) <= 6.0 + 1e-6


@pytest.mark.slow
def test_brittle_is_model_a_at_home_and_nothing_changes_away_on_real_and_far_inputs(real):
    ctx, data, models, _ = real
    h = data.heldout_pairs
    box, act = ctx.training_state_box, ctx.training_action_interval
    rng = np.random.default_rng(3)
    mid, half = box.mean(axis=1), (box[:, 1] - box[:, 0]) / 2
    far_s = mid + 4.0 * half * rng.uniform(-1, 1, (2000, box.shape[0]))
    far_a = act.mean(axis=1) + 3.0 * ((act[:, 1] - act[:, 0]) / 2) * rng.uniform(-1, 1, (2000, act.shape[0]))
    for s, a in ((h.state, h.action), (far_s, far_a)):
        m_a, sp_a = models["direct"].predict_batch(s, a)
        m, sp = models["fx-brittle"].predict_batch(s, a)
        home = (
            np.all((s >= box[:, 0]) & (s <= box[:, 1]), axis=1)
            & np.all((a >= act[:, 0]) & (a <= act[:, 1]), axis=1)
        )
        assert np.array_equal(m[home], m_a[home]) and np.array_equal(m[~home], s[~home])
        assert np.array_equal(sp, sp_a)
    assert (~home).sum() > 1500  # the far inputs really are away
