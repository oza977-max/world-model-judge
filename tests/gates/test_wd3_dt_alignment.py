"""WD-3 dt-alignment gate (worlds spec ADR-W1).

TC-WD3-01 (skeleton scope): at this build stage there is exactly one
ground-truth call site — wmj.worlds.lv.transition, called from both of
wmj/harness/skeleton.py's functions (_generate_training_data's step
loop and _run_one_step_trials's outcome computation). This checks,
structurally, that both call sites reach the one shared
wmj.worlds.integrator.rk4_step function object with the world's own
dt constant — not a private reimplementation or a re-derived value.
Full three-generator identity (once wmj.harness.benchmarks and a
separate training-data module exist) is re-checked at P6-C01.

TC-WD3-02: the negative — a mismatched dt must fail the check.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

from tests.gates._dt_alignment import DtMismatchError, assert_dt_alignment
from wmj.harness import skeleton
from wmj.worlds import integrator, lv


def test_tc_wd3_01_lv_transition_uses_the_shared_rk4_step_object():
    """Static identity: lv.transition references the one shared
    rk4_step function object, not a private reimplementation."""
    assert lv.transition.__globals__["rk4_step"] is integrator.rk4_step


def _functions_calling_transition(tree: ast.Module) -> set[str]:
    """Which top-level function definitions contain a call to
    something named `.transition(...)`, checked per-function so this
    can actually fail if only ONE of two expected call sites still
    calls it (independent-review finding, P1-C03 pass 1: a single
    module-wide set check cannot distinguish "one site calls
    transition" from "both do")."""
    callers: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        for inner in ast.walk(node):
            if (
                isinstance(inner, ast.Call)
                and isinstance(inner.func, ast.Attribute)
                and inner.func.attr == "transition"
            ):
                callers.add(node.name)
                break
    return callers


def test_tc_wd3_01_skeletons_two_ground_truth_call_sites_both_call_lv_transition():
    """The chunk's only two ground-truth call sites (training-data
    generation and eval-truth generation) are AST-confirmed to call
    lv.transition, not some other stepping function — checked per
    function, so either site silently switching away from
    lv.transition would fail this test."""
    tree = ast.parse(inspect.getsource(skeleton))
    callers = _functions_calling_transition(tree)
    assert "_generate_training_data" in callers
    assert "_run_one_step_trials" in callers


def test_tc_wd3_01_transition_passes_the_module_constant_dt_to_rk4_step():
    """dt-alignment, non-circular: inspects transition()'s own source
    to confirm the third argument to rk4_step is the DT *name*, not a
    literal that merely happens to equal it today (independent-review
    finding, P1-C03 pass 1: comparing lv.WORLD.dt == lv.DT alone is
    guaranteed true by construction and proves nothing about what
    transition() actually passes to the integrator)."""
    tree = ast.parse(inspect.getsource(lv.transition))
    rk4_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "rk4_step"
    ]
    assert len(rk4_calls) == 1, "expected exactly one call to rk4_step"
    dt_argument = rk4_calls[0].args[2]
    assert isinstance(dt_argument, ast.Name) and dt_argument.id == "DT"


def test_tc_wd3_01_ground_truth_dt_is_the_worlds_own_constant():
    """dt-alignment: the world's declared dt is what every ground-truth
    step actually uses (no re-derived or hardcoded local dt)."""
    assert lv.WORLD.dt == lv.DT
    assert_dt_alignment(world_dt=lv.WORLD.dt, model_dt=lv.DT)


def test_tc_wd3_02_mismatched_dt_fails_the_alignment_check():
    """Negative/phantom-gate: a model configured with a different step
    size must make the check fail — proving it actually inspects
    something (worlds spec ADR-W1)."""
    with pytest.raises(DtMismatchError):
        assert_dt_alignment(world_dt=lv.DT, model_dt=lv.DT * 2.0)
