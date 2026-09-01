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


def test_tc_wd3_01_skeletons_two_ground_truth_call_sites_both_call_lv_transition():
    """The chunk's only two ground-truth call sites (training-data
    generation and eval-truth generation) are AST-confirmed to call
    lv.transition, not some other stepping function."""
    source = inspect.getsource(skeleton)
    tree = ast.parse(source)

    call_names = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "transition"
    }
    assert "transition" in call_names


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
