"""Tests for wmj.judge._normal — Phi/phi and the CRPS building blocks.

Covers judge spec ADR-J1's exact formulas: Phi(z) = 0.5*(1+erf(z/sqrt2)),
phi(z) = (1/sqrt(2*pi))*exp(-z^2/2), vectorised over NumPy arrays.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from wmj.judge._normal import Phi, phi


def test_phi_at_zero_is_one_half():
    assert Phi(np.array([0.0]))[0] == pytest.approx(0.5)


def test_phi_matches_scalar_erf_definition():
    z = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    expected = np.array([0.5 * (1.0 + math.erf(zi / math.sqrt(2.0))) for zi in z])
    assert Phi(z) == pytest.approx(expected)


def test_lowercase_phi_matches_scalar_definition():
    z = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    expected = np.array(
        [(1.0 / math.sqrt(2.0 * math.pi)) * math.exp(-(zi**2) / 2.0) for zi in z]
    )
    assert phi(z) == pytest.approx(expected)


def test_phi_is_symmetric_around_zero():
    z = np.array([0.3, 1.7])
    assert phi(z) == pytest.approx(phi(-z))
