"""Tests for wmj.models.fixtures — the three deliberately broken models (models ADR-M4,
TC-MU3-01/02/03 behavioural halves, TC-MU4-01/02, TC-MU1-04).

In plain words: each broken model must be Model A plus exactly one change, and that
change must be exactly the one written in the spec — a quarter-size error bar; a
jittered guess with an error bar widened by the exact amount; "nothing changes"
outside the home region (edges count as home). They must be labelled as test equipment,
use the very same trained network as the registered `direct`, and give the same numbers
for a row whether it is asked alone or inside a batch of 200. These tests run on both
worlds (2-D and 4-D), on real held-out rows and on extreme states, not on toy numbers.
"""

from __future__ import annotations

import numpy as np
import pytest

from wmj.harness.training import TrainingRecipe, build_training_data, make_world_context
from wmj.models import direct, fixtures
from wmj.models.base import SeedSource, WorldContext
from wmj.models.direct import DirectModel, shared_direct_core, train_direct
from wmj.models.fixtures import (
    NOISE_SIGMA_FACTOR,
    OVERCONFIDENCE_FACTOR,
    FixtureError,
    FxBrittle,
    FxHonestRough,
    FxOverconfident,
    hashed_standard_normal,
)
from wmj.models.registry import all_models
from wmj.worlds import lv, pendulum

SEED = 20260825
LV_RECIPE = TrainingRecipe(
    training_trajectories=100, subsample_pairs=2000, kick_pairs=50, heldout_pairs=500,
    gradcheck_pairs=16,
)
PEND_RECIPE = TrainingRecipe(
    training_trajectories=300, subsample_pairs=2000, kick_pairs=60, heldout_pairs=500,
    gradcheck_pairs=16,
)
NAMES = ("fx-overconfident", "fx-honest-rough", "fx-brittle")


def _build(world_name, world, recipe, horizon):
    data = build_training_data(world_name, world, SeedSource(SEED, None), recipe, horizon=horizon)
    ctx = make_world_context(world_name, world)
    models = {n: all_models()[n](ctx, SeedSource(SEED, n), data) for n in NAMES}
    inner = DirectModel(ctx, shared_direct_core(ctx, SeedSource(SEED, "direct"), data))
    return ctx, data, models, inner


@pytest.fixture(scope="module")
def lv_world():
    return _build("lv", lv.WORLD, LV_RECIPE, 100)


@pytest.fixture(scope="module")
def pend_world():
    return _build("pendulum", pendulum.WORLD, PEND_RECIPE, 300)


@pytest.fixture(params=["lv", "pendulum"])
def world(request, lv_world, pend_world):
    return lv_world if request.param == "lv" else pend_world


def _random_rows(ctx, n, seed, *, widen=1.0):
    """States/actions drawn over the training box widened by `widen` (>1 reaches outside)."""
    rng = np.random.default_rng(seed)
    box, act = ctx.training_state_box, ctx.training_action_interval
    mid, half = box.mean(axis=1), (box[:, 1] - box[:, 0]) / 2
    amid, ahalf = act.mean(axis=1), (act[:, 1] - act[:, 0]) / 2
    s = mid + widen * half * rng.uniform(-1, 1, (n, box.shape[0]))
    a = amid + widen * ahalf * rng.uniform(-1, 1, (n, act.shape[0]))
    return s, a


# --- labels (TC-MU4-01) and registration -------------------------------------------------


def test_the_three_fixtures_are_registered_and_labelled_as_fixtures(world):
    _, _, models, _ = world
    assert set(NAMES) <= set(all_models())
    for name, model in models.items():
        assert model.is_fixture is True
        assert model.is_baseline is False
        assert model.name == name and name.startswith("fx-")
        assert model.stateless is True


def test_the_registered_direct_is_not_a_fixture(world):
    _, _, _, inner = world
    assert inner.is_fixture is False


@pytest.mark.parametrize("cls", [FxOverconfident, FxHonestRough, FxBrittle])
def test_every_fixture_says_so_in_its_own_words(cls):
    assert "FIXTURE" in (cls.__doc__ or "")
    assert cls.is_fixture is True
    assert fixtures.__doc__ and "FIXTURES, NEVER FINDINGS" in fixtures.__doc__


# --- TC-MU4-02: the inner network is Model A, bit for bit --------------------------------


def test_the_inner_network_is_bit_identical_to_an_independently_trained_direct(lv_world):
    ctx, data, models, _ = lv_world
    independent = DirectModel(ctx, train_direct(ctx, SeedSource(SEED, "direct"), data))
    s, a = _random_rows(ctx, 300, 1)
    mean_i, spread_i = independent.predict_batch(s, a)
    mean_o, spread_o = models["fx-overconfident"].predict_batch(s, a)
    assert np.array_equal(mean_o, mean_i)
    assert np.array_equal(spread_o, spread_i * OVERCONFIDENCE_FACTOR)
    mean_b, spread_b = models["fx-brittle"].predict_batch(s, a)
    inside = np.all(mean_b == mean_i, axis=1)
    assert inside.any() and np.array_equal(spread_b, spread_i)


def test_direct_and_all_three_fixtures_share_one_trained_core(lv_world):
    ctx, data, _, _ = lv_world
    first = shared_direct_core(ctx, SeedSource(SEED, "whatever-name"), data)
    again = shared_direct_core(ctx, SeedSource(SEED, "direct"), data)
    assert first is again
    registered = all_models()["direct"](ctx, SeedSource(SEED, "direct"), data)
    assert registered._net is first


def test_the_cache_retrains_for_a_different_training_object_or_seed(lv_world):
    ctx, data, _, _ = lv_world
    net = shared_direct_core(ctx, SeedSource(SEED, "direct"), data)
    other_seed = shared_direct_core(ctx, SeedSource(SEED + 1, "direct"), data)
    assert other_seed is not net
    s, a = _random_rows(ctx, 20, 2)
    pred = lambda n: DirectModel(ctx, n).predict_batch(s, a)[0]
    assert not np.array_equal(pred(net), pred(other_seed))
    # the one slot now holds the other seed; asking for the first again must retrain, equal bits
    rebuilt = shared_direct_core(ctx, SeedSource(SEED, "direct"), data)
    assert rebuilt is not net
    assert np.array_equal(pred(rebuilt), pred(net))


def test_a_rebuilt_but_equal_training_object_is_retrained_not_trusted(lv_world):
    ctx, _, _, _ = lv_world
    d1 = build_training_data("lv", lv.WORLD, SeedSource(SEED, None), LV_RECIPE, horizon=100)
    n1 = shared_direct_core(ctx, SeedSource(SEED, "direct"), d1)
    d2 = build_training_data("lv", lv.WORLD, SeedSource(SEED, None), LV_RECIPE, horizon=100)
    n2 = shared_direct_core(ctx, SeedSource(SEED, "direct"), d2)
    assert n2 is not n1 and direct._LAST_CORE[0] is d2


def test_the_core_is_keyed_on_the_world_too(lv_world, pend_world):
    ctx_l, data_l, _, _ = lv_world
    ctx_p, data_p, _, _ = pend_world
    a = shared_direct_core(ctx_l, SeedSource(SEED, "direct"), data_l)
    b = shared_direct_core(ctx_p, SeedSource(SEED, "direct"), data_p)
    assert a is not b
    assert direct._LAST_CORE[1][1][0] == "pendulum"


# --- fx-overconfident --------------------------------------------------------------------


def test_overconfident_changes_only_the_error_bar_by_exactly_a_quarter(world):
    ctx, data, models, inner = world
    held = data.heldout_pairs
    extra_s, extra_a = _random_rows(ctx, 400, 3, widen=3.0)
    for s, a in ((held.state, held.action), (extra_s, extra_a)):
        m_in, sp_in = inner.predict_batch(s, a)
        m, sp = models["fx-overconfident"].predict_batch(s, a)
        assert np.array_equal(m, m_in)
        assert np.array_equal(sp, sp_in * 0.25)
        assert OVERCONFIDENCE_FACTOR == 0.25
        assert np.all(sp > 0)


# --- fx-honest-rough ---------------------------------------------------------------------


def test_honest_rough_error_bar_is_the_exact_root_sum_of_squares(world):
    ctx, _data, models, inner = world
    s, a = _random_rows(ctx, 500, 4, widen=2.0)
    _, base = inner.predict_batch(s, a)
    _, sp = models["fx-honest-rough"].predict_batch(s, a)
    assert NOISE_SIGMA_FACTOR == 2.0
    assert np.array_equal(sp, np.sqrt(base**2 + (2.0 * base) ** 2))
    assert np.allclose(sp, base * np.sqrt(5.0), rtol=1e-12)


def test_honest_rough_noise_is_standard_normal_in_units_of_two_error_bars(world):
    ctx, _data, models, inner = world
    s, a = _random_rows(ctx, 20000, 5, widen=1.0)
    m_in, base = inner.predict_batch(s, a)
    m, _ = models["fx-honest-rough"].predict_batch(s, a)
    z = (m - m_in) / (2.0 * base)
    d = z.shape[1]
    assert np.all(np.abs(z.mean(axis=0)) < 0.03)
    assert np.all(np.abs(z.std(axis=0) - 1.0) < 0.03)
    corr = np.corrcoef(z.T)
    assert np.all(np.abs(corr[~np.eye(d, dtype=bool)]) < 0.03)
    # the guess really changed (this is not a disguised copy of Model A)
    assert np.all(np.any(m != m_in, axis=1))
    # a forecast with this error bar is honest: the jittered guess is (Gaussian-ish) within the stated bar
    coverage = np.mean(np.abs(m - m_in) <= 1.96 * np.sqrt(base**2 + (2 * base) ** 2))
    assert 0.95 < coverage <= 1.0


def test_honest_rough_noise_scales_with_the_error_bar_not_with_a_fixed_size(lv_world):
    """Real-regime lesson: error bars here are ~1e-3; a noise of fixed size would swamp them."""
    ctx, _data, models, inner = lv_world
    s, a = _random_rows(ctx, 2000, 6)
    m_in, base = inner.predict_batch(s, a)
    m, _ = models["fx-honest-rough"].predict_batch(s, a)
    assert base.max() < 1.0
    assert np.all(np.abs(m - m_in) <= 12.0 * base + 1e-12)  # sum of 12 uniforms is within ±6 σ_noise


def test_honest_rough_is_a_function_of_the_row_and_the_seed_only(lv_world):
    ctx, data, models, _ = lv_world
    s, a = _random_rows(ctx, 50, 7)
    m1, sp1 = models["fx-honest-rough"].predict_batch(s, a)
    m2, sp2 = models["fx-honest-rough"].predict_batch(s, a)
    assert np.array_equal(m1, m2) and np.array_equal(sp1, sp2)
    # reversed order -> the same row gets the same answer
    m3, _ = models["fx-honest-rough"].predict_batch(s[::-1], a[::-1])
    assert np.array_equal(m3[::-1], m1)
    # a different run seed -> different noise (same network: same data and seed 'direct')
    other = all_models()["fx-honest-rough"](ctx, SeedSource(SEED + 1, "fx-honest-rough"), data)
    m4, _ = other.predict_batch(s, a)
    assert not np.array_equal(m4, m1)
    # ...and the *noise itself* differs, not just the network under it
    inner_a = shared_direct_core(ctx, SeedSource(SEED + 1, "direct"), data)
    z_other = (m4 - DirectModel(ctx, inner_a).predict_batch(s, a)[0]) / (
        2.0 * DirectModel(ctx, inner_a).predict_batch(s, a)[1]
    )
    inner_b = shared_direct_core(ctx, SeedSource(SEED, "direct"), data)
    z_here = (m1 - DirectModel(ctx, inner_b).predict_batch(s, a)[0]) / (
        2.0 * DirectModel(ctx, inner_b).predict_batch(s, a)[1]
    )
    assert not np.allclose(z_other, z_here, atol=1e-6)
    assert abs(np.corrcoef(z_other.ravel(), z_here.ravel())[0, 1]) < 0.3
    # same seed, rebuilt -> identical
    again = all_models()["fx-honest-rough"](ctx, SeedSource(SEED, "fx-honest-rough"), data)
    assert np.array_equal(again.predict_batch(s, a)[0], m1)


def test_hashed_noise_is_batch_size_invariant_and_input_layout_invariant():
    rng = np.random.default_rng(0)
    rows = rng.normal(size=(200, 5))
    full = hashed_standard_normal(rows, 12345, 4)
    for i in (0, 1, 7, 63, 199):
        assert np.array_equal(hashed_standard_normal(rows[i : i + 1], 12345, 4)[0], full[i])
    # memory layouts that hold the same numbers
    assert np.array_equal(hashed_standard_normal(np.asfortranarray(rows), 12345, 4), full)
    wide = np.zeros((200, 10))
    wide[:, ::2] = rows  # a column-strided view of the same numbers
    strided = wide[:, ::2]
    assert not strided.flags["C_CONTIGUOUS"] and np.array_equal(strided, rows)
    assert np.array_equal(hashed_standard_normal(strided, 12345, 4), full)
    reversed_rows = rows[::-1]  # a negative-stride view: row i of it is row 199-i
    assert np.array_equal(hashed_standard_normal(reversed_rows, 12345, 4), full[::-1])
    # other number types give the noise of the same values, not of their raw bytes
    ints = np.arange(30).reshape(10, 3)
    assert np.array_equal(
        hashed_standard_normal(ints, 5, 2), hashed_standard_normal(ints.astype(np.float64), 5, 2)
    )
    f32 = rows.astype(np.float32)
    assert np.array_equal(
        hashed_standard_normal(f32, 5, 2), hashed_standard_normal(f32.astype(np.float64), 5, 2)
    )


def test_hashed_noise_known_answers_pin_the_exact_bits():
    """Changing any step of the hash would silently change every recorded fx-honest-rough
    result (MU-8); these values were produced once and are pinned exactly."""
    rows = np.array([[1.0, 2.0, -0.5], [0.0, 0.0, 0.0], [3.25, -7.5, 1e-3]])
    assert hashed_standard_normal(rows, 1, 2).tolist() == [
        [-0.9428008877453253, -2.143575420712338],
        [1.6829023589716687, 0.6735365380239111],
        [0.950612124876125, 0.9502725667178957],
    ]
    assert hashed_standard_normal(rows, 123456789, 2).tolist() == [
        [0.27356271815904964, -1.3172804733270818],
        [-0.45637518812011635, -0.8420024094152208],
        [0.2907030509169406, 1.4062934494323711],
    ]


def test_hashed_noise_treats_negative_zero_and_zero_alike():
    assert np.array_equal(
        hashed_standard_normal(np.array([[0.0, 1.0]]), 1, 3),
        hashed_standard_normal(np.array([[-0.0, 1.0]]), 1, 3),
    )


def test_noise_reads_the_whole_row_every_state_column_and_every_action_column(lv_world):
    _ctx, _data, models, inner = lv_world
    model = models["fx-honest-rough"]
    s = np.array([[3.0, 2.0]] * 4)
    a = np.zeros((4, 1))
    s[1, 1] += 1e-9  # differs only in the last state column
    s[2, 0] += 1e-9  # differs only in the first state column
    a[3, 0] = 1e-9  # differs only in the action
    m, _ = model.predict_batch(s, a)
    m_in, sp = inner.predict_batch(s, a)
    z = (m - m_in) / (2.0 * sp)
    for i in range(4):
        for j in range(i + 1, 4):  # the rows are 1e-9 apart, the noise must be unrelated
            assert np.max(np.abs(z[i] - z[j])) > 1e-2, (i, j)


def test_hashed_noise_depends_on_the_key_the_row_and_the_column():
    rows = np.array([[1.0, 2.0], [1.0, 2.0000000000000004], [2.0, 1.0]])
    a = hashed_standard_normal(rows, 1, 3)
    b = hashed_standard_normal(rows, 2, 3)
    assert not np.array_equal(a, b)
    assert not np.array_equal(a[0], a[1]) and not np.array_equal(a[0], a[2])
    assert len(set(a[0])) == 3  # the three outputs for one row differ
    assert hashed_standard_normal(rows[:0], 1, 3).shape == (0, 3)


def test_hashed_noise_is_standard_normal_at_scale():
    rows = np.random.default_rng(1).normal(size=(60000, 3))
    z = hashed_standard_normal(rows, 99, 2)
    assert np.all(np.abs(z.mean(axis=0)) < 0.02)
    assert np.all(np.abs(z.std(axis=0) - 1.0) < 0.02)
    assert np.abs(np.corrcoef(z.T)[0, 1]) < 0.02
    assert np.max(np.abs(z)) <= 6.0
    # the tails are approximately normal: ≈4.55% beyond two sigma
    assert abs(np.mean(np.abs(z) > 2.0) - 0.0455) < 0.006


def test_hashed_noise_refuses_non_2d_rows():
    with pytest.raises(FixtureError, match="2-D"):
        hashed_standard_normal(np.zeros(4), 1, 2)


# --- fx-brittle: the four region cases and closed edges ----------------------------------


def _home_mid(ctx):
    return ctx.training_state_box.mean(axis=1), ctx.training_action_interval.mean(axis=1)


def test_brittle_four_cases_inside_both_state_out_action_out_both_out(world):
    ctx, _, models, inner = world
    box, act = ctx.training_state_box, ctx.training_action_interval
    s_in, a_in = _home_mid(ctx)
    s_out = s_in.copy()
    s_out[0] = box[0, 1] + 0.5 * (box[0, 1] - box[0, 0])
    a_out = a_in.copy()
    a_out[0] = act[0, 1] + 0.5 * (act[0, 1] - act[0, 0] or 1.0)
    cases = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])  # (state_out, action_out)
    S = np.stack([s_out if c[0] else s_in for c in cases])
    A = np.stack([a_out if c[1] else a_in for c in cases])
    m_in, sp_in = inner.predict_batch(S, A)
    m, sp = models["fx-brittle"].predict_batch(S, A)
    assert np.array_equal(m[0], m_in[0])  # home: Model A
    for i in (1, 2, 3):  # any one outside: nothing changes
        assert np.array_equal(m[i], S[i])
    assert np.array_equal(sp, sp_in)  # error bar untouched, even where it is wrong
    assert not np.array_equal(m_in[1], S[1])  # (and Model A would have predicted a change)


def test_brittle_edges_are_home_on_both_worlds_and_every_dimension(world):
    ctx, _, models, inner = world
    box, act = ctx.training_state_box, ctx.training_action_interval
    s_mid, a_mid = _home_mid(ctx)
    rows_s, rows_a, expect_home = [], [], []
    for d in range(box.shape[0]):
        for edge in (box[d, 0], box[d, 1]):
            s = s_mid.copy()
            s[d] = edge
            rows_s.append(s); rows_a.append(a_mid); expect_home.append(True)
        for beyond in (np.nextafter(box[d, 0], -np.inf), np.nextafter(box[d, 1], np.inf)):
            s = s_mid.copy()
            s[d] = beyond
            rows_s.append(s); rows_a.append(a_mid); expect_home.append(False)
    for d in range(act.shape[0]):
        for edge in (act[d, 0], act[d, 1]):
            a = a_mid.copy()
            a[d] = edge
            rows_s.append(s_mid); rows_a.append(a); expect_home.append(True)
        for beyond in (np.nextafter(act[d, 0], -np.inf), np.nextafter(act[d, 1], np.inf)):
            a = a_mid.copy()
            a[d] = beyond
            rows_s.append(s_mid); rows_a.append(a); expect_home.append(False)
    S, A, home = np.array(rows_s), np.array(rows_a), np.array(expect_home)
    m_in, _ = inner.predict_batch(S, A)
    m, _ = models["fx-brittle"].predict_batch(S, A)
    assert np.array_equal(m[home], m_in[home])
    assert np.array_equal(m[~home], S[~home])


def test_brittle_at_extreme_states_far_outside_says_nothing_changes(world):
    ctx, _, models, _ = world
    s, a = _random_rows(ctx, 300, 8, widen=6.0)
    m, _ = models["fx-brittle"].predict_batch(s, a)
    box, act = ctx.training_state_box, ctx.training_action_interval
    home = (
        np.all((s >= box[:, 0]) & (s <= box[:, 1]), axis=1)
        & np.all((a >= act[:, 0]) & (a <= act[:, 1]), axis=1)
    )
    assert (~home).sum() > 250 and np.array_equal(m[~home], s[~home])


# --- batch bit-identity (TC-MU1-04) and input refusals -----------------------------------


@pytest.mark.parametrize("n", [1, 2, 7, 64, 200])
def test_every_fixture_batches_bit_identically_on_both_worlds(world, n):
    ctx, _, models, _ = world
    s, a = _random_rows(ctx, n, 9, widen=1.5)
    for name, model in models.items():
        means, spreads = model.predict_batch(s, a)
        for i in range(n):
            p = model.predict(s[i], a[i])
            assert np.array_equal(means[i], p.mean), name
            assert np.array_equal(spreads[i], p.spread), name


def test_predict_returns_copies_and_refuses_batches(lv_world):
    ctx, _, models, _ = lv_world
    s, a = _home_mid(ctx)
    p = models["fx-overconfident"].predict(s, a)
    q = models["fx-overconfident"].predict(s, a)
    assert not np.shares_memory(p.mean, q.mean) and not np.shares_memory(p.spread, q.spread)
    assert np.array_equal(p.mean, q.mean)
    for model in models.values():
        with pytest.raises(FixtureError, match="predict_batch"):
            model.predict(s[None, :], a[None, :])
        model.reset()


def test_a_non_finite_input_is_refused_not_passed_through(lv_world):
    ctx, _, models, _ = lv_world
    s, a = _random_rows(ctx, 3, 10)
    s[1, 0] = np.nan
    for model in models.values():
        with pytest.raises(Exception, match="finite|not finite"):
            model.predict_batch(s, a)


# --- the fixture's own safety checks and refusals (with a stand-in for Model A) ----------


class _Stub:
    """Stands in for Model A: returns whatever mean and spread it was loaded with."""

    def __init__(self, mean, spread):
        self.mean, self.spread = np.asarray(mean, float), np.asarray(spread, float)

    def predict_batch(self, states, actions):
        n = len(states)
        return np.tile(self.mean, (n, 1)), np.tile(self.spread, (n, 1))


def _stub_ctx(action_dim=1):
    return WorldContext(
        world_name="stub", state_dim=2, action_dim=action_dim,
        training_state_box=np.array([[0.0, 1.0], [0.0, 1.0]]),
        training_action_interval=np.array([[-1.0, 1.0]] * action_dim),
        scale=np.ones(2),
    )


@pytest.mark.parametrize(
    ("cls", "mean", "spread"),
    [
        (FxOverconfident, [0.0, 0.0], [5e-324, 1.0]),  # a quarter of the tiniest number is zero
        (FxOverconfident, [np.nan, 0.0], [1.0, 1.0]),
        (FxOverconfident, [0.0, np.inf], [1.0, 1.0]),
        (FxOverconfident, [0.0, 0.0], [np.inf, 1.0]),
        (FxOverconfident, [0.0, 0.0], [-1.0, 1.0]),
        (FxBrittle, [np.nan, 0.0], [1.0, 1.0]),
        (FxBrittle, [0.0, 0.0], [0.0, 1.0]),
    ],
)
def test_a_fixture_refuses_to_emit_a_non_finite_forecast_or_a_spread_with_no_width(cls, mean, spread):
    model = cls(_stub_ctx(), _Stub(mean, spread))
    with pytest.raises(FixtureError, match="non-finite|no width"):
        model.predict_batch(np.full((3, 2), 0.5), np.zeros((3, 1)))


def test_honest_rough_refuses_a_spread_so_large_its_square_overflows():
    model = FxHonestRough(_stub_ctx(), _Stub([0.0, 0.0], [1e308, 1.0]), key=1)
    with pytest.raises(FixtureError, match="non-finite|no width"), np.errstate(all="ignore"):
        model.predict_batch(np.full((2, 2), 0.5), np.zeros((2, 1)))


@pytest.mark.parametrize(
    ("state", "action"),
    [
        (np.zeros((1, 2)), np.zeros((1,))),  # batched state, single action
        (np.zeros(2), np.zeros((1, 1))),  # single state, batched action
        (np.zeros((1, 2)), np.zeros((1, 1))),
        (np.zeros((1, 1, 2)), np.zeros(1)),
    ],
)
def test_predict_refuses_anything_but_one_state_and_one_action(state, action):
    model = FxOverconfident(_stub_ctx(), _Stub([0.0, 0.0], [1.0, 1.0]))
    with pytest.raises(FixtureError, match="predict_batch"):
        model.predict(state, action)


def test_brittle_is_away_if_any_one_of_several_action_dimensions_is_outside():
    ctx = _stub_ctx(action_dim=2)
    model = FxBrittle(ctx, _Stub([9.0, 9.0], [1.0, 1.0]))
    s = np.full((4, 2), 0.5)
    a = np.array([[0.0, 0.0], [1.0, -1.0], [0.0, 1.5], [-1.5, 0.0]])
    mean, spread = model.predict_batch(s, a)
    assert mean[0].tolist() == [9.0, 9.0] and mean[1].tolist() == [9.0, 9.0]  # home (edges count)
    assert mean[2].tolist() == [0.5, 0.5] and mean[3].tolist() == [0.5, 0.5]  # away
    assert np.array_equal(spread, np.ones((4, 2)))


def test_every_fixture_holds_the_very_same_network_as_direct(world):
    _, _, models, inner = world
    for name in NAMES:
        assert models[name]._inner._net is inner._net, name
