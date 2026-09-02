"""TC-WD6-01/02: task distinctness and boundary determinism.

TC-WD6-01: at least two tasks exist, one control one planning, with
quantitatively different tolerances (not two tasks relabeled).

TC-WD6-02: a distance value exactly equal to a task's tolerance
classifies as within-tolerance, deterministically — worlds.md §4.1:
"band-edge classification uses <= (closed), so exact-boundary values
pass deterministically."
"""

from __future__ import annotations

import numpy as np
import pytest

from wmj.worlds import lv
from wmj.worlds.base import distance, within_tolerance


def test_tc_wd6_01_lv_declares_one_control_and_one_planning_task():
    tasks = {task.kind: task for task in lv.tasks()}
    assert set(tasks) == {"control", "planning"}
    assert tasks["control"].tolerance != tasks["planning"].tolerance


def test_tc_wd6_01_lv_tolerances_are_not_a_relabeled_duplicate():
    control = next(t for t in lv.tasks() if t.kind == "control")
    planning = next(t for t in lv.tasks() if t.kind == "planning")
    assert control.tolerance == pytest.approx(0.10)
    assert planning.tolerance == pytest.approx(0.40)


def test_tc_wd6_02_distance_exactly_at_tolerance_is_within_tolerance():
    control = next(t for t in lv.tasks() if t.kind == "control")
    assert within_tolerance(control.tolerance, control.tolerance) is True


def test_tc_wd6_02_distance_just_over_tolerance_is_not_within():
    control = next(t for t in lv.tasks() if t.kind == "control")
    assert within_tolerance(control.tolerance + 1e-9, control.tolerance) is False


def test_tc_wd6_02_distance_just_under_tolerance_is_within():
    control = next(t for t in lv.tasks() if t.kind == "control")
    assert within_tolerance(control.tolerance - 1e-9, control.tolerance) is True


def test_distance_matches_the_rms_normalised_formula():
    a = np.array([4.0, 2.5])
    b = np.array([8.0, 5.0])
    scale = np.array([4.0, 2.5])
    # per-dim normalised diffs: (4/4, 2.5/2.5) = (1.0, 1.0); RMS = 1.0
    assert distance(a, b, scale) == pytest.approx(1.0)


def test_distance_is_zero_for_identical_states():
    a = np.array([4.0, 2.5])
    assert distance(a, a, lv.WORLD.scale) == pytest.approx(0.0)


def test_distance_is_symmetric():
    a = np.array([4.0, 2.5])
    b = np.array([5.0, 3.0])
    assert distance(a, b, lv.WORLD.scale) == pytest.approx(
        distance(b, a, lv.WORLD.scale)
    )
