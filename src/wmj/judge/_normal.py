"""wmj.judge._normal — the standard normal CDF/PDF, vectorised.

In plain words: CRPS and the calibration bands both need the normal
distribution's cumulative curve (Phi) and its bell curve (phi). Python
only gives us `math.erf` as a scalar function, so this module wraps it
once to work on whole NumPy arrays at once (judge spec ADR-J1) — no
SciPy dependency needed for something this small (NF-3).
"""

from __future__ import annotations

import math

import numpy as np

_erf = np.frompyfunc(math.erf, 1, 1)

SQRT_2 = math.sqrt(2.0)
SQRT_2_PI = math.sqrt(2.0 * math.pi)

# The four ADR-J2 central-interval multipliers, pinned exactly (no
# general inverse-CDF routine exists in this project's dependency
# budget) — Phi^-1((1+p)/2) for p in {0.50, 0.80, 0.90, 0.95}.
Z_50 = 0.6745
Z_80 = 1.2816
Z_90 = 1.6449
Z_95 = 1.9600


def Phi(z: np.ndarray) -> np.ndarray:
    """The standard normal CDF: 0.5 * (1 + erf(z / sqrt(2)))."""
    return 0.5 * (1.0 + _erf(z / SQRT_2).astype(np.float64))


def phi(z: np.ndarray) -> np.ndarray:
    """The standard normal PDF: (1 / sqrt(2*pi)) * exp(-z^2 / 2)."""
    return (1.0 / SQRT_2_PI) * np.exp(-(z**2) / 2.0)
