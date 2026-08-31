"""Tests for wmj.harness.serialize.canonical_serialize.

Covers cross-cutting ADR-002 rule 4: sorted keys, UTF-8, \\n newlines,
no trailing whitespace, arrays via .tolist(), json.dumps as the sole
float-rendering call, non-finite values rejected.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from wmj.harness.serialize import NonFiniteValueError, canonical_serialize


def test_adr002_rule4_canonical_serialize_sorted_keys():
    obj = {"zebra": 1, "apple": 2, "mango": 3}
    out = canonical_serialize(obj)
    # keys must appear in sorted order in the byte output
    text = out.decode("utf-8")
    assert text.index('"apple"') < text.index('"mango"') < text.index('"zebra"')


def test_adr002_rule4_canonical_serialize_roundtrips_to_equal_value():
    obj = {"a": [1, 2, 3], "b": {"c": 1.5}}
    out = canonical_serialize(obj)
    assert json.loads(out.decode("utf-8")) == obj


def test_adr002_rule4_canonical_serialize_is_deterministic_across_calls():
    obj = {"seed": 20260825, "values": [0.1, 0.2, 0.3], "name": "direct"}
    first = canonical_serialize(obj)
    second = canonical_serialize(obj)
    assert first == second


def test_adr002_rule4_canonical_serialize_converts_numpy_array_to_nested_list():
    obj = {"predictions": np.array([[1.0, 2.0], [3.0, 4.0]])}
    out = canonical_serialize(obj)
    parsed = json.loads(out.decode("utf-8"))
    assert parsed["predictions"] == [[1.0, 2.0], [3.0, 4.0]]


def test_adr002_rule4_canonical_serialize_converts_numpy_scalar_to_native_float():
    obj = {"mean": np.float64(3.14)}
    out = canonical_serialize(obj)
    parsed = json.loads(out.decode("utf-8"))
    assert parsed["mean"] == pytest.approx(3.14)
    # confirm no numpy repr ever leaked into the text (NEP 51 guard)
    assert "np.float64" not in out.decode("utf-8")


def test_adr002_rule4_canonical_serialize_no_trailing_whitespace_on_any_line():
    obj = {"a": 1, "b": {"c": 2, "d": [1, 2, 3]}}
    out = canonical_serialize(obj)
    for line in out.decode("utf-8").split("\n"):
        assert line == line.rstrip(), f"trailing whitespace on line: {line!r}"


def test_adr002_rule4_canonical_serialize_uses_unix_newlines():
    obj = {"a": 1, "b": 2}
    out = canonical_serialize(obj)
    assert b"\r\n" not in out


def test_adr002_rule4_canonical_serialize_output_is_valid_utf8():
    obj = {"note": "no ambient reads — pure function"}
    out = canonical_serialize(obj)
    # must not raise
    out.decode("utf-8")


def test_adr002_rule4_canonical_serialize_rejects_nan():
    with pytest.raises(NonFiniteValueError):
        canonical_serialize({"value": float("nan")})


def test_adr002_rule4_canonical_serialize_rejects_infinity():
    with pytest.raises(NonFiniteValueError):
        canonical_serialize({"value": float("inf")})


def test_adr002_rule4_canonical_serialize_rejects_negative_infinity():
    with pytest.raises(NonFiniteValueError):
        canonical_serialize({"value": float("-inf")})


def test_adr002_rule4_canonical_serialize_rejects_numpy_nan_inside_array():
    """Realistic-fixture variant (TDD-3): a NaN buried inside an
    otherwise-normal numpy array — the shape a happy-path synthetic
    fixture (all-finite arrays) would never include, but a real
    training run producing a divergent rollout genuinely could."""
    arr = np.array([1.0, 2.0, np.nan, 4.0])
    with pytest.raises(NonFiniteValueError):
        canonical_serialize({"outcome_distance": arr})
