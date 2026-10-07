"""Tests for wmj.harness.fingerprint — the record of what machine and library build made the numbers.

In plain words: results are only byte-identical on the same kind of CPU and library build, so
every run writes down what it ran on. These tests check the record is complete, plain, stable
within a run, and never the reason a run fails.
"""

from __future__ import annotations

import json
import platform
import sys
import types

import numpy as np

from wmj.harness import fingerprint
from wmj.harness.fingerprint import runtime_fingerprint
from wmj.harness.serialize import canonical_serialize


def test_the_fingerprint_has_exactly_the_documented_fields_with_plain_values():
    fp = runtime_fingerprint()
    assert list(fp) == ["python", "numpy", "blas", "platform", "cpu_features", "openblas_coretype_override"]
    assert fp["numpy"] == np.__version__ and fp["python"] == platform.python_version()
    assert fp["platform"] == f"{sys.platform}-{platform.machine()}"  # the same coarse composite as the envelope's meta
    assert isinstance(fp["blas"], str) and fp["blas"]
    assert isinstance(fp["cpu_features"], list) and fp["cpu_features"] == sorted(fp["cpu_features"])
    assert all(isinstance(x, str) for x in fp["cpu_features"])


def test_the_fingerprint_serialises_canonically_and_is_stable_within_a_run():
    assert canonical_serialize(runtime_fingerprint()) == canonical_serialize(runtime_fingerprint())
    json.loads(canonical_serialize(runtime_fingerprint()))


def test_cpu_features_are_the_enabled_ones_sorted(monkeypatch):
    fake = types.SimpleNamespace(__cpu_features__={"SSE2": True, "AVX512F": False, "AVX2": True, "FMA3": True})
    monkeypatch.setattr(np.core, "_multiarray_umath", fake, raising=False)
    assert fingerprint._cpu_features() == ["AVX2", "FMA3", "SSE2"]


def test_a_numpy_that_does_not_report_features_or_blas_gives_unknowns_not_a_crash(monkeypatch):
    monkeypatch.setattr(np.core, "_multiarray_umath", types.SimpleNamespace(), raising=False)
    assert fingerprint._cpu_features() == []

    def broken(*args, **kwargs):
        raise RuntimeError("no build info")

    monkeypatch.setattr(np, "show_config", broken)
    assert fingerprint._blas_description() == "unknown"
    assert runtime_fingerprint()["blas"] == "unknown"


def test_a_core_type_override_in_the_environment_is_recorded(monkeypatch):
    monkeypatch.setenv("OPENBLAS_CORETYPE", "Haswell")
    assert runtime_fingerprint()["openblas_coretype_override"] == "Haswell"
    monkeypatch.delenv("OPENBLAS_CORETYPE")
    assert runtime_fingerprint()["openblas_coretype_override"] == ""
