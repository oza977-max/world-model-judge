"""wmj.judge.errors — the judge's own refusals.

In plain words: the judge refuses loudly rather than guess. Each kind of refusal has its
own name so a caller (and a test) can tell *why* it said no: a baseline was missing, the
input was malformed, or a verdict would have been incomplete. The judge package may not
import anything from the rest of the project (cross-cutting ADR-003), so it defines these
here instead of using the shared project base class.
"""

from __future__ import annotations


def short_repr(value) -> str:
    """A safe, short `repr` for refusal messages (an absurdly long integer must not crash the message)."""
    try:
        text = repr(value)
    except Exception:  # noqa: BLE001 - an integer of 4,300+ digits, a deeply nested or hostile object
        return "<value too large or odd to display>"
    return text if len(text) <= 80 else text[:77] + "..."


class JudgeError(Exception):
    """Base of every deliberate refusal by the judge."""


class MissingBaselineError(JudgeError):
    """No verdict without comparing: the persistence or the linear baseline is absent (MU-2, TC-MU2-01)."""


class JudgeInputError(JudgeError, ValueError):
    """The input handed to the judge is malformed or inconsistent (judge spec §7)."""


class VerdictIncompleteError(JudgeError):
    """A verdict would be missing, null, non-finite or mis-keyed in a required group (JU-9, TC-JU9-02)."""


class NonPositiveSpreadError(JudgeInputError):
    """Raised when a stated spread is zero or negative.

    CRPS is undefined for a non-positive spread; the judge refuses
    rather than clamping it to something plausible-looking (judge spec
    §7: "the judge additionally guards CRPS/coverage against sigma<=0
    with a hard error, never a clamp").
    """


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
