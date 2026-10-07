"""wmj.judge.skill — CRPS closed form and skill relative to a baseline.

In plain words: CRPS scores a whole predicted distribution against
what actually happened, not just "how far off was the mean" — a model
that is both close AND honestly confident scores better than one that
is close but overconfident (judge spec ADR-J1). "Skill" turns that raw
score into a ranking anyone can read: 0 means "no better than the
baseline", 1 means "essentially perfect", negative means "worse than
just guessing the baseline's answer".

This module imports only numpy and the judge's own modules — nothing
else, per this project's own rule that the judge cannot import any other
wmj package (cross-cutting ADR-003). Its refusals are the judge's own
`JudgeError` family (`wmj.judge.errors`), never a bare Python error and
never a silently wrong number.
"""

from __future__ import annotations

import math

import numpy as np

from wmj.judge._normal import Phi, phi
from wmj.judge.errors import JudgeInputError
from wmj.judge.types import JudgeInput


class NonPositiveSpreadError(JudgeInputError):
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
    with np.errstate(over="ignore", invalid="ignore"):
        score = spread * (z * (2.0 * Phi(z) - 1.0) + 2.0 * phi(z) - 1.0 / np.sqrt(np.pi))
    if not np.all(np.isfinite(score)):
        raise NonFiniteScoreError(
            "the CRPS overflowed or is not a number (a spread so small that (outcome − mean)/spread cannot be "
            "represented); the judge refuses rather than report a skill built on it"
        )
    return score


class NonFiniteScoreError(JudgeInputError):
    """Raised when a CRPS comes out infinite or not-a-number (an overflow), never reported as a number."""


class NonPositiveBaselineError(JudgeInputError):
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
    if not (math.isfinite(crps_model) and math.isfinite(crps_baseline)):
        raise NonFiniteScoreError(f"skill_score needs finite CRPS values, got {crps_model!r} and {crps_baseline!r}")
    if not crps_baseline > 0.0:
        raise NonPositiveBaselineError(
            f"skill_score requires crps_baseline > 0, got {crps_baseline!r} "
            f"(judge spec ADR-J1: skill is a ratio to the baseline's CRPS)"
        )
    return float(1.0 - crps_model / crps_baseline)


def _region_crps(forecasts, outcomes: np.ndarray, rows: np.ndarray) -> float:
    """Mean CRPS over the given trials, at one step ahead (array index 0), averaged over quantities."""
    per_quantity = crps_gaussian(forecasts.mean[rows, 0, :], forecasts.spread[rows, 0, :], outcomes[rows, 0, :])
    return float(np.mean(np.mean(per_quantity, axis=1)))


def compute_skill(inp: JudgeInput) -> dict:
    """The verdict's `skill` block: one-step skill against both baselines, per task and region.

    In plain words: for each region of the world, score the model's whole forecast (guess and
    stated uncertainty together) one step ahead with the unfoolable CRPS rule, score the two
    reference forecasts the same way, and report how much better the model is — 0 is "no better
    than the reference", 1 is "essentially perfect", negative is "worse". The score is pinned to
    the first step ahead whatever the task, so every task of a region carries the same numbers
    (the spec still wants the task named on each entry). Only forecasts and outcomes are used,
    in the units they arrive in (the harness hands over normalised ones), and only that region's
    own trials. The entry names exactly `vs_persistence` and `vs_linear` — the judge cannot see
    any other model, so no third comparator can appear (TC-MU6-05(b)).
    """
    if type(inp) is not JudgeInput:
        raise JudgeInputError(f"compute_skill needs a JudgeInput, got {type(inp).__name__}")
    labels = np.array([label.region_name for label in inp.region_labels], dtype=object)
    scored: dict[str, tuple[float, float, float]] = {}
    for region in sorted(set(labels.tolist())):
        rows = np.flatnonzero(labels == region)
        model = _region_crps(inp.predictions, inp.outcomes, rows)
        scored[region] = (
            model,
            skill_score(model, _region_crps(inp.persistence, inp.outcomes, rows)),
            skill_score(model, _region_crps(inp.linear, inp.outcomes, rows)),
        )
    return {
        "per_task_region": [
            {
                "task": task.name,
                "region": region,
                "vs_persistence": scored[region][1],
                "vs_linear": scored[region][2],
                "crps": scored[region][0],
            }
            for task in inp.tasks
            for region in sorted(scored)
        ]
    }
