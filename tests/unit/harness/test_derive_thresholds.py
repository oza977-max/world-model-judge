"""Tests for wmj.harness.derive_thresholds — JU-11 bands, derived not chosen.

The bands come from the exact binomial CDF for Bin(200, 0.10) (judge ADR-J4):
green = the widest central region whose each tail is <= 2.5% -> [12, 29];
amber-outer = each tail <= 0.05% -> [8, 35]; red beyond. These integers are
*computed* here and asserted against the spec's pinned values, so the
derivation — not a hardcoded table — is what is tested (JU-11).

In plain words: these tests check that the pass/fail bands are computed from the exact binomial arithmetic rather than chosen by eye, that the committed file matches the code, and that writing it can never leave a half-written file.
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


# --- review code-review-002 ---------------------------------------------------------------------


def test_the_committed_thresholds_file_is_exactly_what_the_code_derives():
    """prereg/thresholds.json is committed before judging; if it drifts from the derivation
    (a hand edit, or a change to a world's scale vector) the pre-registration would certify
    numbers the code no longer produces."""
    from pathlib import Path

    from wmj.harness.serialize import canonical_serialize

    committed = Path(__file__).resolve().parents[3] / "prereg" / "thresholds.json"
    assert canonical_serialize(build_thresholds()) == committed.read_bytes()


def test_writing_thresholds_is_atomic_a_failed_write_leaves_the_old_file_and_no_scratch(tmp_path, monkeypatch):
    from wmj.harness import derive_thresholds as dt

    target = tmp_path / "prereg" / "thresholds.json"
    write_thresholds(target)
    good = target.read_bytes()
    assert not list(target.parent.glob("*.tmp"))  # no scratch file is left behind on success

    def boom():
        raise RuntimeError("crash while computing")

    monkeypatch.setattr(dt, "build_thresholds", boom)
    try:
        write_thresholds(target)
    except RuntimeError:
        pass
    assert target.read_bytes() == good and not list(target.parent.glob("*.tmp"))
    monkeypatch.undo()

    def crash_on_swap(self, other):
        raise OSError("disk full")

    monkeypatch.setattr(type(target), "replace", crash_on_swap)
    try:
        write_thresholds(target)
    except OSError:
        pass
    monkeypatch.undo()
    assert target.read_bytes() == good  # the live file was never half-written
