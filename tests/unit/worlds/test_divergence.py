"""Tests for wmj.worlds.divergence (worlds spec ADR-W1 drift bound, ADR-W3).

TC-WD4-01: separation is a curve, not a scalar; LV's is sub-exponential.
TC-WD4-02: the pendulum's low-energy and near-inverted curves differ.
TC-WD3-03: drift stays within bound — or the run fails loudly.

Unit tests run the real code at reduced scale (few starts, short
horizon) so the suite stays fast; the full-scale numbers live in
tests/gates/test_benchmark_full_scale.py (marked slow).
"""

from __future__ import annotations

import numpy as np
import pytest

from wmj.worlds import lv, pendulum
from wmj.worlds.base import distance
from wmj.worlds.divergence import (
    DriftBoundError,
    assert_drift_within_bound,
    conserved_drift,
    median_separation_curve,
    perturb,
    separation_curve,
)


# --- perturb (ADR-W3: relative, sign-alternating per dimension) ---


def test_perturb_is_relative_and_sign_alternating():
    state = np.array([4.0, 2.0, 1.0, 0.5])
    out = perturb(state, delta0=1e-6)
    expected = state * (1.0 + 1e-6 * np.array([1.0, -1.0, 1.0, -1.0]))
    assert np.array_equal(out, expected)


def test_perturb_with_zero_delta_is_identity():
    state = np.array([4.5, 2.0])
    assert np.array_equal(perturb(state, delta0=0.0), state)


def test_perturb_does_not_mutate_its_input():
    state = np.array([4.5, 2.0])
    perturb(state, delta0=1e-6)
    assert np.array_equal(state, np.array([4.5, 2.0]))


# --- separation_curve ---


def test_separation_curve_has_horizon_plus_one_entries_starting_at_step_zero():
    state0 = np.array([4.5, 2.0])
    curve = separation_curve(lv.transition, state0, horizon=20, scale=lv.SCALE, delta0=1e-6)
    assert curve.shape == (21,)
    assert curve[0] == pytest.approx(distance(state0, perturb(state0, 1e-6), lv.SCALE))


def test_separation_curve_is_non_negative_and_deterministic():
    state0 = np.array([4.5, 2.0])
    first = separation_curve(lv.transition, state0, 30, lv.SCALE, 1e-6)
    second = separation_curve(lv.transition, state0, 30, lv.SCALE, 1e-6)
    assert np.all(first >= 0.0)
    assert np.array_equal(first, second)


# --- median_separation_curve (ADR-W3: median, not mean) ---


def test_median_separation_curve_takes_the_per_step_median_across_starts():
    curves = np.array([[0.0, 1.0, 10.0], [0.0, 3.0, 20.0], [0.0, 2.0, 1000.0]])
    med = median_separation_curve(curves)
    assert np.array_equal(med, np.array([0.0, 2.0, 20.0]))


# --- TC-WD4-01: LV — a curve, sub-exponential ---


def test_tc_wd4_01_lv_separation_is_a_curve_not_a_single_rate():
    state0 = np.array([4.5, 2.0])
    curve = separation_curve(lv.transition, state0, horizon=200, scale=lv.SCALE, delta0=1e-6)
    assert curve.ndim == 1 and curve.size == 201


def test_tc_wd4_01_lv_separation_grows_sub_exponentially():
    """Executed before writing (build/prompts/P2-C03.md): LV orbits are
    neutrally stable, so over the declared horizon the separation is
    bounded and oscillates; over many cycles it grows linearly (phase
    drift), never exponentially. Assert the bound, not a slope."""
    rng = np.random.default_rng(0)
    box = lv.regions().training_state_box
    starts = rng.uniform(box[:, 0], box[:, 1], size=(8, 2))
    curves = np.stack(
        [separation_curve(lv.transition, s, 700, lv.SCALE, 1e-6) for s in starts]
    )
    med = median_separation_curve(curves)
    # Two-sided (review pass 1): executed, the ratio sits in a 0.6-1.4
    # band. An exponential runaway (1e-6 * exp(0.02 t) reaches ~1e6 by
    # step 700) fails the upper bound; a broken integrator that pulls
    # the twin trajectories together fails the lower one.
    ratio = med[-1] / med[0]
    assert 0.1 < ratio < 10.0, f"TC-WD4-01: LV separation ratio {ratio:.3g} outside (0.1, 10)"


# --- TC-WD4-02: pendulum — regime-dependent ---


def test_tc_wd4_02_pendulum_near_inverted_diverges_far_faster_than_training():
    spec = pendulum.regions()
    rng = np.random.default_rng(0)
    horizon = 600
    half = horizon // 2

    def med_curve(box):
        starts = rng.uniform(box[:, 0], box[:, 1], size=(8, 4))
        return median_separation_curve(
            np.stack(
                [separation_curve(pendulum.transition, s, horizon, pendulum.SCALE, 1e-6) for s in starts]
            )
        )

    training = med_curve(spec.training_state_box)
    inverted = med_curve(spec.out_regions[0].state_box)
    assert inverted[half] / training[half] >= 5.0


# --- TC-WD3-03: drift within bound, or fails loudly ---


def test_conserved_drift_is_zero_for_an_identity_transition():
    max_abs, initial = conserved_drift(lambda s, a: s, lv.conserved, np.array([4.5, 2.0]), 10)
    assert max_abs == 0.0
    assert initial == pytest.approx(lv.conserved(np.array([4.5, 2.0])))


def test_tc_wd3_03_lv_drift_is_tiny_over_a_short_horizon():
    max_abs, _ = conserved_drift(lv.transition, lv.conserved, np.array([4.5, 2.0]), 100)
    assert max_abs < 1e-8


def test_tc_wd3_03_assert_drift_within_bound_passes_when_inside():
    assert_drift_within_bound(rel_drift_max=1e-9, bound=1e-6, world_name="lv")


def test_tc_wd3_03_negative_drift_over_bound_fails_loudly():
    with pytest.raises(DriftBoundError):
        assert_drift_within_bound(rel_drift_max=2e-6, bound=1e-6, world_name="lv")
