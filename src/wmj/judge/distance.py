"""wmj.judge.distance — the one "how far apart are these?" yardstick (ADR-J5).

In plain words: one number for "how far is the prediction from the truth" when there are several
quantities: square each quantity's gap, average them, take the square root (the root-mean-square, so
adding a quantity does not by itself make the distance bigger). The arrays already arrive divided by
each quantity's own scale, so every quantity counts equally. The same yardstick measures the
error-versus-horizon curve, the tolerances of the tasks and the plotted distances of the main chart.
Rows with gaps too tiny or too huge to square (below 1e-150 or above 1e150) are scaled by the row's
largest gap first, so gaps of 1e-170 or 1e200 are neither squared to zero nor overflowed; a result that still cannot be represented is refused.
"""

from __future__ import annotations

import numpy as np

from wmj.judge.regions import require_finite

SAFE_LOW, SAFE_HIGH = 1e-150, 1e150  # inside this range squaring a size can neither vanish nor overflow


def rms_of_sizes(sizes: np.ndarray) -> np.ndarray:
    """Root-mean-square over the last axis of non-negative sizes (callers pass absolute values).

    Rows of ordinary size use the plain formula, so a tie such as sizes (1, 1, 5) against a tolerance of 3
    is exactly 3; only rows too tiny or too huge to square safely use the largest-size-scaled form."""
    largest = np.max(sizes, axis=-1, keepdims=True)
    with np.errstate(over="ignore", invalid="ignore"):
        plain = np.sqrt(np.mean(sizes**2, axis=-1))
        # a power of two near the largest size: dividing by it is exact, so scaling cannot disturb an exact tie
        scale = np.where(largest > 0.0, np.ldexp(1.0, np.frexp(largest)[1] - 1), 1.0)
        scaled = scale[..., 0] * np.sqrt(np.mean((sizes / scale) ** 2, axis=-1))
    ordinary = (largest[..., 0] == 0.0) | ((largest[..., 0] >= SAFE_LOW) & (largest[..., 0] <= SAFE_HIGH))
    return require_finite(np.where(ordinary, plain, scaled), "a distance")


def rms_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Distance between two arrays of quantities, over the last axis (see the module note).

    The final overflow refusals here and in `rms_of_sizes` overlap on purpose (defence in depth: a result
    that cannot be represented must never be returned, whichever step produced it). Exact ties at a
    tolerance hold for the worlds' own quantity counts (1, 2, 4) and for any gaps within the ordinary range."""
    with np.errstate(over="ignore", invalid="ignore"):
        gap = require_finite(np.abs(a - b), "a gap between prediction and outcome")
    return rms_of_sizes(gap)
