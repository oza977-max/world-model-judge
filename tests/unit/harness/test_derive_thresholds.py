"""Tests for wmj.harness.derive_thresholds — JU-11 bands, derived not chosen.

The bands come from the exact binomial CDF for Bin(200, 0.10) (judge ADR-J4):
green = the widest central region whose each tail is <= 2.5% -> [12, 29];
amber-outer = each tail <= 0.05% -> [8, 35]; red beyond. These integers are
*computed* here and asserted against the spec's pinned values, so the
derivation — not a hardcoded table — is what is tested (JU-11).
"""

from __future__ import annotations

import json

import numpy as np

from wmj.harness.derive_thresholds import (
    AGREEMENT_THRESHOLD,
    binomial_bands,
    build_thresholds,
    sharpness_hedge_threshold,
    write_thresholds,
)


def test_binomial_bands_reproduce_the_spec_pinned_boundaries():
    bands = binomial_bands(n=200, p=0.10)
    assert bands["green"] == [12, 29]
    assert bands["amber_outer"] == [8, 35]
    assert bands["n"] == 200 and bands["p"] == 0.10


def test_binomial_band_outside_probabilities_meet_the_declared_bounds():
    bands = binomial_bands(n=200, p=0.10)
    # JU-8 per-band false-alarm bounds: green <= 5%, amber <= 0.1%.
    assert bands["green_outside_prob"] < 0.05
    assert bands["amber_outside_prob"] < 0.001
    # And the spec's stated figures, to 3 sig figs.
    assert round(bands["green_outside_prob"], 4) == 0.0331
    assert round(bands["amber_outside_prob"], 4) == 0.0009


def test_green_is_strictly_inside_amber():
    bands = binomial_bands(n=200, p=0.10)
    g_lo, g_hi = bands["green"]
    a_lo, a_hi = bands["amber_outer"]
    assert a_lo < g_lo <= g_hi < a_hi


def test_sharpness_hedge_threshold_is_the_world_scale_vector():
    scale = np.array([4.0, 2.5])
    assert np.array_equal(sharpness_hedge_threshold(scale), scale)


def test_agreement_threshold_is_one():
    # Climatology agreement holds when standardised deviation |z| <= 1.
    assert AGREEMENT_THRESHOLD == 1.0


def test_build_thresholds_has_both_worlds_and_the_bands():
    t = build_thresholds()
    assert t["bands"]["green"] == [12, 29]
    assert set(t["sharpness_hedge_threshold"]) == {"lv", "pendulum"}
    assert t["sharpness_hedge_threshold"]["lv"] == [4.0, 2.5]
    assert t["agreement_threshold"] == 1.0


def test_write_thresholds_is_byte_reproducible(tmp_path):
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    write_thresholds(a)
    write_thresholds(b)
    assert a.read_bytes() == b.read_bytes()
    # and it is valid JSON with the bands
    loaded = json.loads(a.read_text())
    assert loaded["bands"]["amber_outer"] == [8, 35]
