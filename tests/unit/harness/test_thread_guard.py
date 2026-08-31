"""Tests for wmj.harness.thread_guard.

Covers cross-cutting ADR-002 rule 1: the startup gate asserts
OMP_NUM_THREADS/OPENBLAS_NUM_THREADS/MKL_NUM_THREADS are set to "1" in
os.environ — the control that actually determines BLAS threading,
since NumPy exposes no runtime thread-count introspection API.
"""

from __future__ import annotations

import os

import pytest

from wmj.harness.thread_guard import (
    THREAD_ENV_VARS,
    ThreadGuardError,
    assert_single_threaded,
    ensure_single_threaded,
)


def test_ensure_single_threaded_sets_all_three_vars(monkeypatch):
    for name in THREAD_ENV_VARS:
        monkeypatch.delenv(name, raising=False)

    ensure_single_threaded()

    for name in THREAD_ENV_VARS:
        assert os.environ[name] == "1"


def test_assert_single_threaded_passes_when_all_set(monkeypatch):
    for name in THREAD_ENV_VARS:
        monkeypatch.setenv(name, "1")
    assert_single_threaded()  # must not raise


@pytest.mark.parametrize("missing_var", THREAD_ENV_VARS)
def test_assert_single_threaded_raises_when_one_var_wrong(monkeypatch, missing_var):
    for name in THREAD_ENV_VARS:
        monkeypatch.setenv(name, "1")
    monkeypatch.setenv(missing_var, "4")

    with pytest.raises(ThreadGuardError):
        assert_single_threaded()


def test_assert_single_threaded_raises_when_var_unset(monkeypatch):
    for name in THREAD_ENV_VARS:
        monkeypatch.setenv(name, "1")
    monkeypatch.delenv(THREAD_ENV_VARS[0], raising=False)

    with pytest.raises(ThreadGuardError):
        assert_single_threaded()
