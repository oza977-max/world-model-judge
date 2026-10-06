"""wmj.judge.errors — the judge's own refusals.

In plain words: the judge refuses loudly rather than guess. Each kind of refusal has its
own name so a caller (and a test) can tell *why* it said no: a baseline was missing, the
input was malformed, or a verdict would have been incomplete. The judge package may not
import anything from the rest of the project (cross-cutting ADR-003), so it defines these
here instead of using the shared project base class.
"""

from __future__ import annotations


class JudgeError(Exception):
    """Base of every deliberate refusal by the judge."""


class MissingBaselineError(JudgeError):
    """No verdict without comparing: the persistence or the linear baseline is absent (MU-2, TC-MU2-01)."""


class JudgeInputError(JudgeError, ValueError):
    """The input handed to the judge is malformed or inconsistent (judge spec §7)."""


class VerdictIncompleteError(JudgeError):
    """A verdict would be missing, null, non-finite or mis-keyed in a required group (JU-9, TC-JU9-02)."""
