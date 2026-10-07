"""wmj.judge.climatology — where a task stops being predictable (the "switch step"), ADR-J5.

In plain words: every task has a tolerance, and the world has a curve showing how fast two
slightly different histories drift apart. The *switch step* is the first step at which that
drift exceeds the task's tolerance — before it, a model can be graded on getting the actual
path right; after it, only on getting the statistics right (nobody can promise an exact
path in a chaotic world). Calibration and sharpness are measured at that step. If the drift
never exceeds the tolerance within the task's horizon there is no switch step, and the task
is graded on its whole horizon, so its evaluation step is the horizon's last step.

This module will also hold the conditioned-climatology agreement (P4-C05); for now it holds
the switch step both calibration and sharpness need.
"""

from __future__ import annotations

import numpy as np


def switch_step(curve: np.ndarray, tolerance: float, horizon: int) -> int | None:
    """First step `s` in `1..horizon` where `curve[s] > tolerance`, else `None`.

    `curve` is the world's drift curve for one region, indexed from step 0 (length `H + 1`).
    A distance exactly equal to the tolerance does not exceed it (the band edge is closed,
    worlds spec §4.1). Step 0 is never a switch step: no forecast is made at step 0.
    """
    if horizon < 1 or curve.shape[0] < horizon + 1:
        raise ValueError(f"the curve ({curve.shape[0]} points) does not reach the horizon {horizon}")
    over = np.flatnonzero(curve[1 : horizon + 1] > tolerance)
    return int(over[0]) + 1 if over.size else None


def evaluation_step(switch: int | None, horizon: int) -> int:
    """The single step a task is judged at: its switch step, or its last step if it has none."""
    return horizon if switch is None else switch
