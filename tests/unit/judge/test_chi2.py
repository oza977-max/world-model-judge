"""Tests for wmj.judge._chi2 — the joint 50/80/90/95% regions of d quantities (ADR-J2/J4, code-review-003).

In plain words: the judge's "90% sure" claim is about all of a model's numbers together. Its test for
that needs one number per (how many quantities, which level): how far, in the model's own stated sizes,
a combined miss may reach. These tests pin that number against exact formulas (for 2 and 4 quantities
it has a closed form), against well-known printed table values, against the familiar single-quantity
multipliers 0.6745 / 1.2816 / 1.6449 / 1.96, and check it behaves as it must: it grows with the level and
with the number of quantities, refuses nonsense, and is the same every time.
"""

from __future__ import annotations

import math

import pytest

from wmj.judge._chi2 import chi2_quantile, gamma_p, joint_limit
from wmj.judge._normal import LEVELS, Z_VALUES


def test_two_quantities_have_the_exact_closed_form_minus_two_log_of_the_tail():
    for p in (0.5, 0.8, 0.9, 0.95, 0.999, 0.01):
        assert chi2_quantile(2, p) == pytest.approx(-2.0 * math.log(1.0 - p), rel=1e-12)


def test_four_quantities_satisfy_their_closed_form_cdf():
    for p in (0.5, 0.8, 0.9, 0.95):
        x = chi2_quantile(4, p)
        assert 1.0 - math.exp(-x / 2.0) * (1.0 + x / 2.0) == pytest.approx(p, rel=1e-12)


@pytest.mark.parametrize(
    "d, p, printed",
    [(1, 0.9, 2.7055), (1, 0.95, 3.8415), (2, 0.9, 4.6052), (3, 0.9, 6.2514), (4, 0.9, 7.7794), (4, 0.5, 3.3567),
     (5, 0.9, 9.2364), (10, 0.95, 18.3070), (30, 0.9, 40.2560), (3, 0.5, 2.3660)],
)
def test_it_matches_well_known_printed_table_values(d, p, printed):
    assert chi2_quantile(d, p) == pytest.approx(printed, abs=6e-5)


def test_one_quantity_reproduces_the_specs_four_fixed_multipliers_to_four_decimals():
    assert [round(joint_limit(1, level), 4) for level in LEVELS] == list(Z_VALUES)


def test_the_limit_grows_with_the_level_and_with_the_number_of_quantities():
    for d in range(1, 9):
        limits = [joint_limit(d, level) for level in LEVELS]
        assert limits == sorted(limits) and len(set(limits)) == 4
    for level in LEVELS:
        by_d = [joint_limit(d, level) for d in range(1, 9)]
        assert by_d == sorted(by_d) and len(set(by_d)) == 8


def test_gamma_p_is_a_cdf_known_values_and_both_computation_branches_agree_at_the_switch():
    assert gamma_p(1.0, 1.0) == pytest.approx(1.0 - math.exp(-1.0), rel=1e-14)  # series branch
    assert gamma_p(1.0, 5.0) == pytest.approx(1.0 - math.exp(-5.0), rel=1e-14)  # continued-fraction branch
    assert gamma_p(0.5, 2.0) == pytest.approx(math.erf(math.sqrt(2.0)), rel=1e-13)
    a = 3.0
    below, above = gamma_p(a, a + 1.0 - 1e-9), gamma_p(a, a + 1.0 + 1e-9)
    assert below == pytest.approx(above, abs=1e-8) and below < above
    assert gamma_p(2.0, 0.0) == 0.0 and gamma_p(2.0, -1.0) == 0.0
    assert gamma_p(2.0, 1e6) == pytest.approx(1.0, abs=1e-15)


def test_the_quantile_inverts_gamma_p_to_machine_precision():
    for d in (1, 2, 3, 4, 7, 20):
        for p in (0.5, 0.8, 0.9, 0.95):
            assert gamma_p(d / 2.0, chi2_quantile(d, p) / 2.0) == pytest.approx(p, abs=1e-13)


def test_nonsense_is_refused_and_the_answer_is_the_same_every_time():
    for d in (0, -1, 1.5, True, None, "2"):
        with pytest.raises(ValueError):
            chi2_quantile(d, 0.9)
    for p in (0.0, 1.0, -0.1, 1.1, float("nan")):
        with pytest.raises(ValueError):
            chi2_quantile(2, p)
    assert chi2_quantile(4, 0.9) == chi2_quantile(4, 0.9) == 7.779440339734856
