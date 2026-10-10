"""wmj.judge.distance — the one "how far apart are these?" yardstick (ADR-J5).

In plain words: one number for "how far is the prediction from the truth" when there are several
quantities: square each quantity's gap, average them, take the square root (the root-mean-square, so
adding a quantity does not by itself make the distance bigger). The arrays already arrive divided by
each quantity's own scale, so every quantity counts equally. The same yardstick measures the
error-versus-horizon curve, the tolerances of the tasks and the plotted distances of the main chart.
It is scaled by each row's largest gap before squaring, so gaps of 1e-170 or 1e200 are neither
squared to zero nor overflowed; a result that still cannot be represented is refused.
"""

from __future__ import annotations

import numpy as np

from wmj.judge.regions import require_finite


def rms_of_sizes(sizes: np.ndarray) -> np.ndarray:
    """Root-mean-square over the last axis of non-negative sizes (callers pass absolute values)."""
    largest = np.max(sizes, axis=-1, keepdims=True)
    scale = np.where(largest > 0.0, largest, 1.0)
    with np.errstate(over="ignore", invalid="ignore"):
        rms = scale[..., 0] * np.sqrt(np.mean((sizes / scale) ** 2, axis=-1))
    return require_finite(rms, "a distance")


def rms_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Distance between two arrays of quantities, over the last axis (see the module note)."""
    with np.errstate(over="ignore", invalid="ignore"):
        gap = require_finite(np.abs(a - b), "a gap between prediction and outcome")
    return rms_of_sizes(gap)
