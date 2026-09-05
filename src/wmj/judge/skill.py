"""wmj.judge.skill — CRPS closed form and skill relative to a baseline.

In plain words: CRPS scores a whole predicted distribution against
what actually happened, not just "how far off was the mean" — a model
that is both close AND honestly confident scores better than one that
is close but overconfident (judge spec ADR-J1). "Skill" turns that raw
score into a ranking anyone can read: 0 means "no better than the
baseline", 1 means "essentially perfect", negative means "worse than
just guessing the baseline's answer".

This module imports only numpy and math — nothing else, per this
project's own rule that the judge cannot import any other wmj package
(cross-cutting ADR-003), so it defines its own small exception rather
than reaching for the shared WmjError base.
"""

from __future__ import annotations

import numpy as np

from wmj.judge._normal import Phi, phi


class NonPositiveSpreadError(ValueError):
    """Raised when a stated spread is zero or negative.

    CRPS is undefined for a non-positive spread; the judge refuses
    rather than clamping it to something plausible-looking (judge spec
    §7: "the judge additionally guards CRPS/coverage against sigma<=0
    with a hard error, never a clamp").
    """


def crps_gaussian(
    mean: np.ndarray, spread: np.ndarray, outcome: np.ndarray
) -> np.ndarray:
    """Closed-form CRPS for a Gaussian prediction, elementwise.

    CRPS(mu, sigma; y) = sigma * [z*(2*Phi(z)-1) + 2*phi(z) - 1/sqrt(pi)],
    z = (y - mu) / sigma (judge spec ADR-J1). Inputs are expected to
    already be in normalised units (divided by the world's scale
    vector) — this function has no notion of a world to normalise
    against.
    """
    if np.any(spread <= 0.0):
        raise NonPositiveSpreadError(
            f"crps_gaussian requires spread > 0 everywhere, got {spread!r} "
            f"(judge spec §7 sigma<=0 guard)"
        )
    z = (outcome - mean) / spread
    return spread * (
        z * (2.0 * Phi(z) - 1.0) + 2.0 * phi(z) - 1.0 / np.sqrt(np.pi)
    )


class NonPositiveBaselineError(ValueError):
    """Raised when the baseline CRPS a skill score divides by is not > 0.

    A Gaussian CRPS is strictly positive for any finite spread > 0, so
    anything routed through `crps_gaussian` cannot reach this; the guard
    exists so a caller that computes its baseline some other way gets
    the refusal here, at the division, not later as a NaN the canonical
    serializer rejects (code-review-001, Panel B).
    """


def skill_score(crps_model: float, crps_baseline: float) -> float:
    """skill = 1 - CRPS_model / CRPS_baseline (judge spec ADR-J1).

    0 means no better than the baseline; 1 means essentially perfect;
    negative means worse than the baseline. Requires `crps_baseline > 0`.
    """
    if not crps_baseline > 0.0:
        raise NonPositiveBaselineError(
            f"skill_score requires crps_baseline > 0, got {crps_baseline!r} "
            f"(judge spec ADR-J1: skill is a ratio to the baseline's CRPS)"
        )
    return float(1.0 - crps_model / crps_baseline)
