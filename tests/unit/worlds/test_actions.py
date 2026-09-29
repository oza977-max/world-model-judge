"""Tests for wmj.worlds.actions — the one kick generator (worlds ADR-W2).

In plain words: these check that the lever is pulled exactly the way the
spec pins it — how often (a fixed chance per step), how hard (a uniform
draw from a named band), and that the same seed always gives the same
kicks. The band edges and odd inputs are tested too, because a generator
that quietly produced a kick one hair outside its band would silently
move a trial into the wrong region label.
"""

from __future__ import annotations

import numpy as np
import pytest

from wmj.worlds.actions import (
    KickSpecError,
    band_for_axis,
    kick_sequence,
    step_probability,
)


def _rng(seed: int = 7) -> np.random.Generator:
    return np.random.default_rng(seed)


def test_kick_sequence_has_shape_horizon_by_one_and_float64():
    kicks = kick_sequence(_rng(), horizon=50, p_step=0.1, band="in", umax=0.1)
    assert kicks.shape == (50, 1)
    assert kicks.dtype == np.float64


def test_kick_sequence_follows_the_pinned_draw_rule_exactly_for_the_in_band():
    # worlds ADR-W2: r = rng.random((H, 3)); kick iff r[t,0] < p_step;
    # "in": u = umax * (2 * r[t,1] - 1).
    horizon, p_step, umax = 400, 0.3, 0.1
    r = _rng(11).random((horizon, 3))
    expected = np.where(r[:, 0] < p_step, umax * (2.0 * r[:, 1] - 1.0), 0.0)
    kicks = kick_sequence(_rng(11), horizon=horizon, p_step=p_step, band="in", umax=umax)
    assert np.array_equal(kicks[:, 0], expected)


def test_kick_sequence_follows_the_pinned_draw_rule_exactly_for_the_out_band():
    # "out": u = s * umax * (2 - r[t,1]), s = +1 if r[t,2] < 0.5 else -1.
    horizon, p_step, umax = 400, 0.3, 1.0
    r = _rng(12).random((horizon, 3))
    sign = np.where(r[:, 2] < 0.5, 1.0, -1.0)
    expected = np.where(r[:, 0] < p_step, sign * umax * (2.0 - r[:, 1]), 0.0)
    kicks = kick_sequence(_rng(12), horizon=horizon, p_step=p_step, band="out", umax=umax)
    assert np.array_equal(kicks[:, 0], expected)


def test_in_band_kicks_stay_inside_the_trained_interval():
    kicks = kick_sequence(_rng(3), horizon=20_000, p_step=0.5, band="in", umax=0.1)
    assert np.all(np.abs(kicks) <= 0.1)


def test_out_band_kicks_are_strictly_outside_the_trained_interval_and_within_twice_it():
    kicks = kick_sequence(_rng(4), horizon=20_000, p_step=0.5, band="out", umax=0.1)[:, 0]
    kicked = kicks[kicks != 0.0]
    assert kicked.size > 0
    assert np.all(np.abs(kicked) > 0.1)
    assert np.all(np.abs(kicked) <= 0.2)
    # both directions occur — "remove rabbits" and "add rabbits"
    assert np.any(kicked > 0.0) and np.any(kicked < 0.0)


def test_band_edges_realistic_fixture_extreme_uniform_draws():
    # TDD-3 realistic fixture: the extreme uniform values a real stream
    # can produce. r = 0.0 gives u = -umax ("in") and |u| = 2*umax
    # ("out") — both still inside their band's closed/half-open edges.
    class _EdgeRng:
        def __init__(self, value: float) -> None:
            self.value = value

        def random(self, size):
            out = np.full(size, self.value)
            out[:, 0] = 0.0  # always kick
            return out

    in_kicks = kick_sequence(_EdgeRng(0.0), horizon=3, p_step=0.5, band="in", umax=0.1)
    assert np.all(in_kicks == -0.1)
    out_kicks = kick_sequence(_EdgeRng(0.0), horizon=3, p_step=0.5, band="out", umax=0.1)
    assert np.all(out_kicks == 0.2)  # s=+1 (0.0 < 0.5), |u| = 2*umax


def test_kick_frequency_matches_p_step_over_a_long_sequence():
    kicks = kick_sequence(_rng(5), horizon=200_000, p_step=0.01, band="in", umax=0.1)
    rate = np.count_nonzero(kicks) / kicks.shape[0]
    # binomial sd = sqrt(0.01*0.99/200000) ~ 2.2e-4; allow ~5 sd
    assert abs(rate - 0.01) < 1.2e-3


def test_same_seed_same_kicks_different_seed_different_kicks():
    a = kick_sequence(_rng(9), horizon=1000, p_step=0.1, band="in", umax=0.1)
    b = kick_sequence(_rng(9), horizon=1000, p_step=0.1, band="in", umax=0.1)
    c = kick_sequence(_rng(10), horizon=1000, p_step=0.1, band="in", umax=0.1)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_draws_are_unconditional_so_the_stream_layout_never_depends_on_outcomes():
    # Three draws per step regardless of p_step: after generating a
    # sequence, the generator is in the same state whether it kicked
    # every step or never.
    never, always = _rng(21), _rng(21)
    kick_sequence(never, horizon=100, p_step=0.0, band="in", umax=0.1)
    kick_sequence(always, horizon=100, p_step=1.0, band="in", umax=0.1)
    assert never.random() == always.random()


def test_p_step_zero_gives_only_null_actions():
    kicks = kick_sequence(_rng(), horizon=500, p_step=0.0, band="out", umax=1.0)
    assert np.all(kicks == 0.0)


def test_zero_horizon_gives_an_empty_sequence():
    assert kick_sequence(_rng(), horizon=0, p_step=0.5, band="in", umax=0.1).shape == (0, 1)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"band": "sideways"},
        {"p_step": -0.01},
        {"p_step": 1.01},
        {"p_step": float("nan")},
        {"umax": 0.0},
        {"umax": -0.1},
        {"umax": float("inf")},
        {"horizon": -1},
    ],
)
def test_kick_sequence_refuses_bad_parameters_loudly(kwargs):
    params = {"horizon": 10, "p_step": 0.1, "band": "in", "umax": 0.1}
    params.update(kwargs)
    with pytest.raises(KickSpecError, match="ADR-W2"):
        kick_sequence(_rng(), **params)


def test_step_probability_is_rate_times_dt_for_both_worlds():
    # worlds ADR-W2: p_step = kick_rate_per_s * dt.
    assert step_probability(0.5, 0.02) == 0.5 * 0.02
    assert step_probability(1.0, 0.002) == 1.0 * 0.002


def test_step_probability_refuses_a_rate_that_would_exceed_one_kick_per_step():
    with pytest.raises(KickSpecError, match="ADR-W2"):
        step_probability(100.0, 0.02)
    with pytest.raises(KickSpecError, match="ADR-W2"):
        step_probability(-1.0, 0.02)


@pytest.mark.parametrize(
    ("axis", "band"), [(None, "in"), ("state", "in"), ("action", "out"), ("both", "out")]
)
def test_band_for_axis_uses_the_out_band_only_where_the_action_axis_is_out(axis, band):
    assert band_for_axis(axis) == band


def test_band_for_axis_refuses_an_unknown_axis():
    with pytest.raises(KickSpecError, match="ADR-W4"):
        band_for_axis("diagonal")


def test_numpy_integer_horizons_are_accepted_and_bool_is_refused():
    # A caller may pass kicks.shape[1] or a NumPy task horizon.
    assert kick_sequence(_rng(), horizon=np.int64(4), p_step=0.5, band="in", umax=0.1).shape == (4, 1)
    with pytest.raises(KickSpecError, match="ADR-W2"):
        kick_sequence(_rng(), horizon=True, p_step=0.5, band="in", umax=0.1)
