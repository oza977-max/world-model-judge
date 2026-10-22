"""wmj.judge._chi2 — the 90% (and 50/80/95%) region of d quantities at once (ADR-J2/J4, code-review-003).

In plain words: a model states, for each of its d numbers (4 for the pendulum, 2 for predator-prey),
a typical size of its own mistake. "I am 90% sure the answer is in my range" is a claim about the
*whole answer*, all d numbers together — not about each number separately. If every number
separately has a 90% interval, the chance that at least one of d numbers falls outside is
1 − 0.9^d: 19% with 2 numbers, 34% with 4. A perfectly honest model would then look wildly
overconfident. So the judge asks the joint question: scale each number's miss by the model's own
stated size for it, square and add the d values; an honest model's total follows a known curve (the
chi-square curve with d degrees of freedom), and the joint 90% region is "total at or below the 90%
point of that curve". An honest model then falls outside exactly 10% of the time whatever d is.
With one number this is the same as the familiar ± 1.6449 sizes.

This module computes that point from nothing but `math`: the regularised lower incomplete gamma
function (a series for small arguments, a continued fraction for large) and a fixed-step bisection.
It is deterministic and exact to better than 1e-12.
"""

from __future__ import annotations

import math

_TINY = 1e-300
_EPS = 1e-16


def gamma_p(a: float, x: float) -> float:
    """Regularised lower incomplete gamma function P(a, x): the chance a gamma(a) variable is at most x."""
    if x <= 0.0:
        return 0.0
    if x < a + 1.0:  # series
        term = total = 1.0 / a
        n = a
        for _ in range(10_000):
            n += 1.0
            term *= x / n
            total += term
            if abs(term) < abs(total) * _EPS:
                break
        return total * math.exp(-x + a * math.log(x) - math.lgamma(a))
    b = x + 1.0 - a  # continued fraction for Q(a, x) (modified Lentz)
    c = 1.0 / _TINY
    d = 1.0 / b
    h = d
    for i in range(1, 10_000):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        d = _TINY if abs(d) < _TINY else d
        c = b + an / c
        c = _TINY if abs(c) < _TINY else c
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < _EPS:
            break
    return 1.0 - math.exp(-x + a * math.log(x) - math.lgamma(a)) * h


def chi2_quantile(d: int, p: float) -> float:
    """The point `q` such that a chi-square variable with `d` degrees of freedom is at most `q` with chance `p`."""
    if type(d) is not int or d < 1:
        raise ValueError(f"degrees of freedom must be a whole number >= 1, got {d!r}")
    if not 0.0 < p < 1.0:
        raise ValueError(f"p must lie strictly between 0 and 1, got {p!r}")
    half = d / 2.0
    low, high = 0.0, float(d) + 10.0
    while gamma_p(half, high / 2.0) < p:
        low, high = high, high * 2.0
    for _ in range(200):  # fixed step count: the same answer on every machine
        mid = 0.5 * (low + high)
        if gamma_p(half, mid / 2.0) < p:
            low = mid
        else:
            high = mid
    return 0.5 * (low + high)


def joint_limit(d: int, level: float) -> float:
    """How far (in the model's own stated sizes, all `d` numbers together) a miss may reach and still be inside the `level` joint region."""
    return math.sqrt(chi2_quantile(d, level))
