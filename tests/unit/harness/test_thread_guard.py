"""Tests for wmj.harness.thread_guard.

Covers cross-cutting ADR-002 rule 1: the startup gate asserts
OMP_NUM_THREADS/OPENBLAS_NUM_THREADS/MKL_NUM_THREADS are set to "1" in
os.environ — the control that actually determines BLAS threading,
since NumPy exposes no runtime thread-count introspection API.

In plain words: these tests check the start-up guard that tells the maths library to use one thread so rounding cannot vary from run to run, and that it refuses to pretend if it is too late.
"""

from __future__ import annotations

import os

import pytest

from wmj.harness import thread_guard
from wmj.harness.thread_guard import (
    THREAD_ENV_VARS,
    ThreadGuardError,
    assert_single_threaded,
    ensure_single_threaded,
)


def test_adr002_rule1_ensure_single_threaded_sets_all_three_vars(monkeypatch):
    monkeypatch.setattr(thread_guard, "_numpy_already_imported", lambda: False)
    for name in THREAD_ENV_VARS:
        monkeypatch.delenv(name, raising=False)

    ensure_single_threaded()

    for name in THREAD_ENV_VARS:
        assert os.environ[name] == "1"


def test_adr002_rule1_assert_single_threaded_passes_when_all_set(monkeypatch):
    for name in THREAD_ENV_VARS:
        monkeypatch.setenv(name, "1")
    assert_single_threaded()  # must not raise


@pytest.mark.parametrize("missing_var", THREAD_ENV_VARS)
def test_adr002_rule1_assert_single_threaded_raises_when_one_var_wrong(monkeypatch, missing_var):
    for name in THREAD_ENV_VARS:
        monkeypatch.setenv(name, "1")
    monkeypatch.setenv(missing_var, "4")

    with pytest.raises(ThreadGuardError):
        assert_single_threaded()


def test_adr002_rule1_assert_single_threaded_raises_when_var_unset(monkeypatch):
    for name in THREAD_ENV_VARS:
        monkeypatch.setenv(name, "1")
    monkeypatch.delenv(THREAD_ENV_VARS[0], raising=False)

    with pytest.raises(ThreadGuardError):
        assert_single_threaded()


def test_ensure_single_threaded_refuses_when_numpy_is_already_loaded_and_a_variable_is_not_one(monkeypatch):
    monkeypatch.setattr(thread_guard, "_numpy_already_imported", lambda: True)
    for name in THREAD_ENV_VARS:
        monkeypatch.setenv(name, "1")
    monkeypatch.setenv(THREAD_ENV_VARS[1], "4")
    with pytest.raises(ThreadGuardError, match="already imported"):
        ensure_single_threaded()
    assert os.environ[THREAD_ENV_VARS[1]] == "4"  # it did not pretend to fix it
    monkeypatch.delenv(THREAD_ENV_VARS[0], raising=False)
    with pytest.raises(ThreadGuardError, match="already imported"):
        ensure_single_threaded()


def test_ensure_single_threaded_is_a_no_op_success_when_numpy_is_loaded_but_all_variables_are_already_one(monkeypatch):
    monkeypatch.setattr(thread_guard, "_numpy_already_imported", lambda: True)
    for name in THREAD_ENV_VARS:
        monkeypatch.setenv(name, "1")
    ensure_single_threaded()


def test_the_real_numpy_import_check_reports_the_truth():
    import sys

    assert thread_guard._numpy_already_imported() is ("numpy" in sys.modules)
