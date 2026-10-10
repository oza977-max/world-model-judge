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
    assert got.tolist() == [pytest.approx(1e-170, rel=1e-12, abs=0), pytest.approx(1e200, rel=1e-12, abs=0), 0.0,
                            pytest.approx(1e200 / math.sqrt(2), rel=1e-12, abs=0), pytest.approx(1e308, rel=1e-12, abs=0)]


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


def test_huge_gaps_of_either_sign_are_measured_not_falsely_refused():
    assert rms_distance(np.array([[0.0, 0.0]]), np.array([[1e200, 1e200]])).tolist() == [pytest.approx(1e200, rel=1e-12, abs=0)]
    assert rms_distance(np.array([[0.0, 0.0]]), np.array([[-1e200, -1e200]])).tolist() == [pytest.approx(1e200, rel=1e-12, abs=0)]
    assert rms_distance(np.array([[1e200, -1e200]]), np.array([[-1e200, 1e200]])).tolist() == [pytest.approx(2e200, rel=1e-12, abs=0)]


def test_a_tie_is_exact_for_ordinary_rows_even_with_unequal_gaps():
    assert rms_of_sizes(np.array([[1.0, 1.0, 5.0]])).tolist() == [3.0]  # sqrt((1 + 1 + 25) / 3) is exactly 3


def test_region_names_that_share_a_prefix_stay_separate():
    from tests.unit.judge._builders import judge_input_kwargs
    from wmj.judge.types import JudgeInput, RegionClimatology, RegionCurve, RegionLabel

    kwargs = judge_input_kwargs()
    kwargs["region_labels"] = tuple(RegionLabel("a" if i < 3 else "ab", None) for i in range(6))
    kwargs["divergence_curves"] = (RegionCurve("ab", np.linspace(0, 2, 6)), RegionCurve("a", np.linspace(0, 1, 6)))  # the longer name first
    kwargs["climatology"] = (RegionClimatology("a", kwargs["climatology"][0].bins), RegionClimatology("ab", kwargs["climatology"][1].bins))
    inp = JudgeInput(**kwargs)
    rows = region_rows(inp)
    assert list(rows) == ["a", "ab"] and rows["a"].tolist() == [0, 1, 2] and rows["ab"].tolist() == [3, 4, 5]
    assert region_curve(inp, "a")[-1] == 1.0 and region_curve(inp, "ab")[-1] == 2.0


def test_tiny_rows_are_never_squared_to_zero_and_huge_rows_never_overflow_across_the_whole_range():
    from decimal import Decimal, getcontext

    getcontext().prec = 60
    for exponent in (-300, -170, -155, -151, -150, -149, -100, 0, 100, 149, 150, 151, 153, 154, 155, 200, 300):
        for mantissa in (1.0, 1.7):
            sizes = np.array([[mantissa * 10.0**exponent, mantissa * 10.0**exponent * 0.5]])
            exact = (sum(Decimal(float(v)) ** 2 for v in sizes[0]) / 2).sqrt()
            got = float(rms_of_sizes(sizes)[0])
            assert got > 0.0 and abs(Decimal(got) - exact) / exact < Decimal("4e-16"), (exponent, mantissa)


def test_rows_straddling_the_hand_over_between_the_plain_and_scaled_forms_agree_on_both_sides():
    from decimal import Decimal, getcontext

    getcontext().prec = 60
    for edge in (1e-150, 1e150):
        for factor in (0.99, 0.9999999, 1.0, 1.0000001, 1.01):
            sizes = np.array([[edge * factor, edge * factor * 3.0]])
            exact = (sum(Decimal(float(v)) ** 2 for v in sizes[0]) / 2).sqrt()
            assert abs(Decimal(float(rms_of_sizes(sizes)[0])) - exact) / exact < Decimal("4e-16"), (edge, factor)


def test_a_tie_stays_exact_in_the_scaled_range_too():
    big = 2.0**500
    assert rms_of_sizes(np.array([[5.0 * big, 13.0 * big, 35.0 * big, 5.0 * big]])).tolist() == [pytest.approx(19.0 * big, rel=0, abs=0)]
    small = 2.0**-515
    assert rms_of_sizes(np.array([[5.0 * small, 13.0 * small, 35.0 * small, 5.0 * small]])).tolist() == [pytest.approx(19.0 * small, rel=0, abs=0)]


def test_equal_huge_gaps_near_the_top_of_the_plain_range_are_not_falsely_refused():
    for value, d in ((1e154, 2), (9.9e153, 4), (1.3e154, 4)):  # a plain sum of the squares would overflow
        assert rms_of_sizes(np.full((1, d), value)).tolist() == [pytest.approx(value, rel=1e-12, abs=0)]


def test_the_smallest_and_largest_representable_gaps_are_exact():
    tiny, big = 5e-324, 1.7976931348623157e308
    assert rms_of_sizes(np.array([[tiny]])).tolist() == [tiny]
    assert rms_of_sizes(np.array([[tiny, tiny]])).tolist() == [tiny]
    assert rms_of_sizes(np.array([[1e-323, tiny]])).tolist() == [pytest.approx(math.sqrt((1e-323**2 + tiny**2) / 2), rel=0.5)]
    for d in (1, 2, 4):
        assert rms_of_sizes(np.full((1, d), big)).tolist() == [big]


def test_the_distance_is_computed_row_by_row_for_arrays_with_leading_axes():
    gaps = np.full((2, 3, 2), 1e200)
    gaps[1, 2] = [3e200, 4e200]
    got = rms_of_sizes(gaps)
    assert got.shape == (2, 3)
    assert got[1, 2] == pytest.approx(math.sqrt(12.5) * 1e200, rel=1e-14, abs=0)
    assert got[0, 0] == pytest.approx(1e200, rel=1e-14, abs=0) and got[1, 1] == pytest.approx(1e200, rel=1e-14, abs=0)


def test_rms_of_sizes_itself_refuses_infinity_and_nan():
    for bad in (np.inf, np.nan):
        with pytest.raises(JudgeInputError, match="overflow"):
            rms_of_sizes(np.array([[bad, 1.0]]))
