"""Tests for wmj.judge.distance and wmj.judge.regions — the shared yardstick and region helpers.

In plain words: the distance is the root-mean-square gap over the quantities of each row; it must be
exact for small, huge and all-zero rows, never square a tiny gap to zero or overflow a huge one, and
refuse (not report) a result that cannot be represented. The region helper hands each region its own
trials and curve, and refuses arrays that hold infinity.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tests.unit.judge._builders import make_input
from wmj.judge.distance import rms_distance, rms_of_sizes
from wmj.judge.errors import JudgeInputError
from wmj.judge.regions import region_curve, region_rows, require_finite


def test_one_quantity_is_its_absolute_gap_and_several_are_root_mean_square_not_sum_or_max():
    assert rms_of_sizes(np.array([[3.0]])).tolist() == [3.0]
    assert rms_of_sizes(np.array([[3.0, 4.0]])).tolist() == [pytest.approx(math.sqrt(12.5), rel=1e-15)]
    assert rms_of_sizes(np.array([[0.0, 0.0], [6.0, 8.0]])).tolist() == [0.0, pytest.approx(math.sqrt(50.0), rel=1e-15)]


def test_tiny_huge_zero_and_mixed_rows_are_exact():
    rows = np.array([[1e-170, 1e-170], [1e200, 1e200], [0.0, 0.0], [1e200, 1e-170], [1e308, 1e308]])
    got = rms_of_sizes(rows)
    assert got.tolist() == [pytest.approx(1e-170, rel=1e-12), pytest.approx(1e200, rel=1e-12), 0.0,
                            pytest.approx(1e200 / math.sqrt(2), rel=1e-12), pytest.approx(1e308, rel=1e-12)]


def test_the_distance_between_two_arrays_is_over_the_last_axis_and_keeps_the_leading_shape():
    a = np.zeros((2, 3, 2))
    b = np.zeros((2, 3, 2))
    b[1, 2] = [3.0, 4.0]
    got = rms_distance(a, b)
    assert got.shape == (2, 3) and got[1, 2] == pytest.approx(math.sqrt(12.5)) and got.sum() == got[1, 2]


def test_an_unrepresentable_gap_is_refused():
    with pytest.raises(JudgeInputError, match="overflow"):
        rms_distance(np.array([[-1.5e308]]), np.array([[1.5e308]]))


def test_require_finite_refuses_infinity_and_nan_even_when_only_one_element_is_bad():
    for bad in (np.array([1.0, np.inf]), np.array([np.nan, 1.0]), np.array([-np.inf])):
        with pytest.raises(JudgeInputError, match="overflow"):
            require_finite(bad, "x")
    ok = np.array([1.0, 2.0])
    assert require_finite(ok, "x") is ok


def test_region_rows_and_curves_belong_to_their_own_region_in_sorted_order():
    inp = make_input()
    rows = region_rows(inp)
    assert list(rows) == ["out-of-range", "training"]
    assert rows["training"].tolist() == [0, 1, 2] and rows["out-of-range"].tolist() == [3, 4, 5]
    assert region_curve(inp, "training").tolist() == pytest.approx(np.linspace(0.0, 1.0, 6).tolist())
    assert region_curve(inp, "out-of-range").tolist() == pytest.approx(np.linspace(0.0, 2.0, 6).tolist())
