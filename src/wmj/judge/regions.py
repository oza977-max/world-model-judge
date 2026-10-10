"""wmj.judge.regions — small shared helpers: which trials belong to which region, and the overflow refusal.

In plain words: every block of the verdict works region by region, so the question "which trials
are in this region, and which drift curve is theirs?" is answered in exactly one place. The
same place holds the refusal used whenever arithmetic overflows: a number that has become
infinity must never be compared as if it were a real size (at absurd magnitudes `inf <= inf`
would call a miss "covered").
"""

from __future__ import annotations

import numpy as np

from wmj.judge.errors import JudgeInputError
from wmj.judge.types import JudgeInput


def require_finite(array: np.ndarray, what: str) -> np.ndarray:
    """Refuse an array holding infinity or not-a-number (an overflow), instead of counting with it.

    At absurd magnitudes `inf <= inf` is true, which would report a miss of 3e308 as covered."""
    if not np.all(np.isfinite(array)):
        raise JudgeInputError(f"{what} overflowed (a gap or width too large to represent); the judge refuses rather than count with it")
    return array


def region_rows(inp: JudgeInput) -> dict[str, np.ndarray]:
    """For each region present, the indices of its trials (regions in sorted-name order)."""
    names = np.array([label.region_name for label in inp.region_labels], dtype=object)  # not fixed-width text: NumPy would strip trailing NULs
    return {region: np.flatnonzero(names == region) for region in sorted(set(names.tolist()))}


def region_curve(inp: JudgeInput, region: str) -> np.ndarray:
    """The world's drift curve (indexed from step 0) for one region."""
    return next(c.curve for c in inp.divergence_curves if c.region_name == region)
