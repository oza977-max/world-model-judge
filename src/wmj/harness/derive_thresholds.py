"""wmj.harness.derive_thresholds — JU-11's bands, derived not chosen by eye.

In plain words: the judge counts how often a model's 90% interval misses the
truth, and compares that count to a band. A model that banking would call
"green" is inside the band you'd expect from chance alone; "amber" is the
warning zone; "red" is a real miss. The point of JU-11 (and of SR 11-7 /
SS1/23 in public banking practice) is that those bands are **derived from a
declared test and sample size and fixed in advance**, not picked after
seeing results. This module computes them once, from the exact binomial CDF
for Bin(200, 0.10), and writes them to `prereg/thresholds.json` — the file
the judge reads (as data; the judge never reads files, JU-12).

The derivation (judge ADR-J4): a two-sided exact binomial test against
p = 0.10 at n = 200 (expected 20 exceptions). Each band is the widest
central acceptance region whose **each tail** is at most α/2:
  - green: α = 5%  → tails ≤ 2.5%  → [12, 29]  (outside-prob 3.31% ≤ 5%)
  - amber: α = 0.1% → tails ≤ 0.05% → [8, 35]  (outside-prob 0.087% ≤ 0.1%)
red is anything beyond the amber region. These integers are *computed* from
the CDF, not hardcoded; the tests assert they equal the spec's pinned values.

This is a build-time script (run once, committed, never re-run after judging
begins — JU-11). It lives in the harness, not the judge, because it reads the
worlds' scale vectors and writes a file — neither of which the pure judge may
do (JU-12).
"""

from __future__ import annotations

from itertools import accumulate
from math import comb
from pathlib import Path

import numpy as np

from wmj.harness.serialize import canonical_serialize
from wmj.worlds import lv, pendulum

# Climatology agreement holds when the standardised deviation |z| <= 1
# (judge spec, conditioned-climatology section). A constant, not world-specific.
AGREEMENT_THRESHOLD = 1.0

# The pinned evaluation test (judge ADR-J4): exact two-sided binomial at these.
_N = 200
_P = 0.10
_GREEN_ALPHA = 0.05    # green band: each tail <= alpha/2 = 2.5%
_AMBER_ALPHA = 0.001   # amber outer: each tail <= alpha/2 = 0.05%

# The two worlds are a fixed set the harness names directly (cross-cutting
# ADR-003's disclosed asymmetry: there is no all_worlds() registry).
_WORLDS = {"lv": lv.WORLD, "pendulum": pendulum.WORLD}


def _binomial_cdf(n: int, p: float) -> list[float]:
    pmf = [comb(n, k) * p**k * (1.0 - p) ** (n - k) for k in range(n + 1)]
    return list(accumulate(pmf))


def _tail_bounded_region(cdf: list[float], n: int, tail: float) -> tuple[int, int]:
    """Widest central [lo, hi] whose each tail probability is <= `tail`.

    `lo` = largest k with P(X < k) <= tail; `hi` = smallest k with
    P(X > k) <= tail. (Not equal-tailed-by-nearest: this is the acceptance
    region whose *each* tail is bounded, which is what reproduces the spec's
    [12, 29] / [8, 35] — an equal-tailed-rounding rule gives [13, 28].)
    """

    def below(k: int) -> float:  # P(X < k) = P(X <= k-1)
        return cdf[k - 1] if k > 0 else 0.0

    def above(k: int) -> float:  # P(X > k)
        return 1.0 - cdf[k]

    lo = max(k for k in range(n + 1) if below(k) <= tail)
    hi = min(k for k in range(n + 1) if above(k) <= tail)
    return lo, hi


def binomial_bands(n: int = _N, p: float = _P) -> dict:
    """The green/amber integer boundaries and their realised outside-probs."""
    cdf = _binomial_cdf(n, p)
    g_lo, g_hi = _tail_bounded_region(cdf, n, _GREEN_ALPHA / 2.0)
    a_lo, a_hi = _tail_bounded_region(cdf, n, _AMBER_ALPHA / 2.0)

    def outside(lo: int, hi: int) -> float:
        below_lo = cdf[lo - 1] if lo > 0 else 0.0
        above_hi = 1.0 - cdf[hi]
        return below_lo + above_hi

    return {
        "n": n,
        "p": p,
        "expected_exceptions": int(round(n * p)),
        "green": [g_lo, g_hi],
        "amber_outer": [a_lo, a_hi],
        "green_outside_prob": outside(g_lo, g_hi),
        "amber_outside_prob": outside(a_lo, a_hi),
    }


def sharpness_hedge_threshold(scale: np.ndarray) -> np.ndarray:
    """Per-dimension 90%-interval width bound = the world's own scale vector.

    JU-5's anti-hedging cross-flag (judge ADR-J4) fires when a region/task's
    mean 90%-interval width exceeds this. Judge ADR-J4 pins only "from the
    world's scale vector"; the chosen rule (pre-registered, fixed in advance)
    is the scale itself: a 90% interval as wide as the state's own
    characteristic magnitude has said nothing, so anything wider is hedging.
    """
    return np.asarray(scale, dtype=float)


def build_thresholds() -> dict:
    """The full, judge-readable thresholds object for `prereg/thresholds.json`."""
    return {
        "bands": binomial_bands(),
        "sharpness_hedge_threshold": {
            name: sharpness_hedge_threshold(world.scale).tolist()
            for name, world in _WORLDS.items()
        },
        "agreement_threshold": AGREEMENT_THRESHOLD,
    }


def write_thresholds(path: str | Path) -> None:
    """Write `prereg/thresholds.json` byte-reproducibly (NF-1) via the
    canonical serializer (sorted keys, `allow_nan=False`)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_serialize(build_thresholds()))


if __name__ == "__main__":  # run once at pre-registration, then commit
    out = Path(__file__).resolve().parents[3] / "prereg" / "thresholds.json"
    write_thresholds(out)
    print(f"wrote {out}")
