"""Tests for wmj.judge.skill — CRPS closed form and skill (judge ADR-J1).

CRPS(mu, sigma; y) = sigma * [z*(2*Phi(z)-1) + 2*phi(z) - 1/sqrt(pi)],
z = (y-mu)/sigma. skill = 1 - CRPS_model / CRPS_baseline.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from wmj.judge.skill import NonPositiveSpreadError, crps_gaussian, skill_score


def test_crps_at_the_mean_matches_the_known_closed_form():
    # CRPS(0, 1; 0): z=0, so CRPS = 2*phi(0) - 1/sqrt(pi) -- a
    # well-known standard-normal CRPS value, independent of this
    # project's own Phi/phi implementation.
    mean = np.array([0.0])
    spread = np.array([1.0])
    outcome = np.array([0.0])

    result = crps_gaussian(mean, spread, outcome)

    expected = 2.0 / math.sqrt(2.0 * math.pi) - 1.0 / math.sqrt(math.pi)
    assert result[0] == pytest.approx(expected, rel=1e-9)


def test_crps_is_zero_only_in_the_zero_spread_limit_not_negative():
    # CRPS is a proper scoring rule: it is always >= 0.
    mean = np.array([0.0, 1.0, -2.0])
    spread = np.array([0.5, 2.0, 0.1])
    outcome = np.array([0.3, 1.0, -1.9])

    result = crps_gaussian(mean, spread, outcome)
    assert np.all(result >= 0.0)


def test_crps_scales_with_spread_at_fixed_standardised_distance():
    # holding z = (y-mu)/sigma fixed, CRPS scales linearly with sigma
    mean = np.array([0.0])
    outcome = np.array([1.0])

    small = crps_gaussian(mean, np.array([1.0]), outcome)
    large = crps_gaussian(mean, np.array([2.0]), outcome * 2.0)
    assert large[0] == pytest.approx(small[0] * 2.0, rel=1e-9)


def test_crps_gaussian_broadcasts_over_a_multi_trial_multi_dimension_array():
    # shape [n_trials, d] -- the real shape crps_gaussian is called with
    # (harness/skeleton.py rolls out N_EVAL trials over d=2 dimensions)
    mean = np.array([[0.0, 1.0], [2.0, -1.0]])
    spread = np.array([[1.0, 0.5], [0.2, 2.0]])
    outcome = np.array([[0.1, 1.2], [2.3, -0.5]])

    result = crps_gaussian(mean, spread, outcome)

    assert result.shape == (2, 2)
    for i in range(2):
        for j in range(2):
            scalar = crps_gaussian(
                mean[i : i + 1, j], spread[i : i + 1, j], outcome[i : i + 1, j]
            )
            assert result[i, j] == pytest.approx(scalar[0])


def test_crps_rejects_non_positive_spread():
    with pytest.raises(NonPositiveSpreadError):
        crps_gaussian(np.array([0.0]), np.array([0.0]), np.array([0.0]))

    with pytest.raises(NonPositiveSpreadError):
        crps_gaussian(np.array([0.0]), np.array([-1.0]), np.array([0.0]))


def test_skill_score_zero_when_model_equals_baseline():
    assert skill_score(1.0, 1.0) == pytest.approx(0.0)


def test_skill_score_positive_when_model_beats_baseline():
    assert skill_score(0.5, 1.0) == pytest.approx(0.5)


def test_skill_score_negative_when_model_loses_to_baseline():
    assert skill_score(1.5, 1.0) == pytest.approx(-0.5)


def test_skill_score_refuses_a_non_positive_baseline():
    """code-review-001 (Panel B): the ratio's precondition is enforced at
    the division, not discovered later as a NaN the serializer rejects."""
    from wmj.judge.skill import NonPositiveBaselineError

    with pytest.raises(NonPositiveBaselineError):
        skill_score(0.1, 0.0)
    with pytest.raises(NonPositiveBaselineError):
        skill_score(0.1, -1.0)
