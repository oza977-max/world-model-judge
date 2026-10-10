"""wmj.judge.climatology — where a task stops being predictable (the "switch step"), ADR-J5.

In plain words: every task has a tolerance, and the world has a curve showing how fast two
slightly different histories drift apart. The *switch step* is the first step at which that
drift exceeds the task's tolerance — before it, a model can be graded on getting the actual
path right; after it, only on getting the statistics right (nobody can promise an exact
path in a chaotic world). Calibration and sharpness are measured at that step. If the drift
never exceeds the tolerance within the task's horizon there is no switch step, and the task
is graded on its whole horizon, so its evaluation step is the horizon's last step.

Past the switch step the judge asks a different question: does the model's predicted state look
like the world's long-run behaviour *for the same amount of the conserved quantity* (energy)? The
world's reference table says, for each range of that quantity (a "bin"), the typical state and its
spread. At every step from the switch to the task's horizon, for every trial, the bin is the one
*measured* from the true trajectory at that step (not frozen at the start), and the model's
predicted mean is compared with that bin's typical state in units of its spread. The average
absolute gap, over trials, steps and quantities, is the agreement score; at or below the
pre-registered threshold the model "agrees".
"""

from __future__ import annotations

import math

import numpy as np

from wmj.judge.errors import JudgeInputError
from wmj.judge.regions import region_curve, region_rows, require_finite
from wmj.judge.types import JudgeInput


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


def compute_climatology(inp: JudgeInput) -> dict:
    """The verdict's `climatology` block: switch step and agreement with the conditioned climatology."""
    if type(inp) is not JudgeInput:
        raise JudgeInputError(f"compute_climatology needs a JudgeInput, got {type(inp).__name__}")
    tables = {t.region_name: t for t in inp.climatology}
    entries = []
    for task in inp.tasks:
        for region, rows in region_rows(inp).items():
            switch = switch_step(region_curve(inp, region), task.tolerance, task.horizon)
            entry = {"task": task.name, "region": region, "switch_step": switch, "agreement_mean_abs_z": None, "agrees": None}
            if switch is not None:
                bins = tables[region].bins
                bin_mean = np.stack([b.mean for b in bins])  # [bins, d]
                bin_sd = np.stack([b.sd for b in bins])
                window = slice(switch - 1, task.horizon)  # steps switch..horizon (array index s - 1)
                chosen = inp.invariant_bins[rows, window]  # [trials, steps] the bin measured at each trial and step
                with np.errstate(over="ignore", invalid="ignore"):
                    gap = np.abs(inp.predictions.mean[rows, window, :] - bin_mean[chosen])
                    z = require_finite(gap / bin_sd[chosen], "a standardised gap from the climatology")
                    score = float(np.mean(z))  # the plain mean keeps an exact tie at the threshold exact
                    if not math.isfinite(score):  # only a sum too large to hold falls back to the scaled mean
                        largest = float(np.max(z))
                        score = largest * float(np.mean(z / largest))
                    require_finite(np.float64(score), "the mean standardised gap")
                entry["agreement_mean_abs_z"] = score
                entry["agrees"] = bool(score <= inp.thresholds.agreement_threshold)
            entries.append(entry)
    return {"per_task": entries}
