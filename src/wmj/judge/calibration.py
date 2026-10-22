"""wmj.judge.calibration — is "90% sure" really right 90% of the time? (ADR-J2, JU-4)

In plain words: a model states, for every number it predicts, a typical size of its own
mistake. If it is honest, then about half the time the true value should land inside the
range "prediction ± 0.67 of that size", about 80% of the time inside ± 1.28, 90% inside
± 1.64, and 95% inside ± 1.96. This module counts how often that actually happens, at those
four levels, for each task and region, at the task's own judging step. The claim is about the
*whole answer*, all the quantities together, so a trial counts as "covered" when its misses,
measured in the model's own stated sizes and combined over all the quantities, fall inside the
joint region of that level (the chi-square region, `_chi2.py`: an honest model is covered 50%,
80%, 90%, 95% of the time whatever the number of quantities — checking each quantity separately
would call a perfectly honest 4-quantity model overconfident, code-review-003). The per-quantity
rates are kept too, for diagnosis. It never combines this with the CRPS skill score — they
answer different questions and stay separate (TC-JU4-01).
"""

from __future__ import annotations

import numpy as np

from wmj.judge._chi2 import rms_limit
from wmj.judge._normal import LEVELS, Z_VALUES
from wmj.judge.climatology import evaluation_step, switch_step
from wmj.judge.distance import miss_in_stated_sizes
from wmj.judge.errors import JudgeInputError
from wmj.judge.regions import region_curve, region_rows, require_finite
from wmj.judge.types import JudgeInput


def task_region_step(inp: JudgeInput, task, region: str) -> int:
    """The step (1-based) at which this task is judged in this region."""
    curve = region_curve(inp, region)
    return evaluation_step(switch_step(curve, task.tolerance, task.horizon), task.horizon)


def compute_calibration(inp: JudgeInput) -> dict:
    """The verdict's `calibration` block: coverage at four levels, per task and region."""
    if type(inp) is not JudgeInput:
        raise JudgeInputError(f"compute_calibration needs a JudgeInput, got {type(inp).__name__}")
    rows_by_region = region_rows(inp)
    entries = []
    for task in inp.tasks:
        for region, rows in rows_by_region.items():
            index = task_region_step(inp, task, region) - 1
            mean = inp.predictions.mean[rows, index, :]
            spread = inp.predictions.spread[rows, index, :]
            with np.errstate(over="ignore", invalid="ignore"):
                miss = require_finite(np.abs(inp.outcomes[rows, index, :] - mean), "outcome minus predicted mean")
                reach = [require_finite(z * spread, "a stated interval half-width") for z in Z_VALUES]
            whole = miss_in_stated_sizes(miss, spread)  # all the quantities together, in the model's own stated sizes
            joint, per_dimension = [], []
            for level, reach_z in zip(LEVELS, reach, strict=True):
                joint.append(float(np.mean(whole <= rms_limit(miss.shape[1], level))))  # the joint region; the edge is inside
                per_dimension.append([float(v) for v in np.mean(miss <= reach_z, axis=0)])  # each quantity on its own, for diagnosis
            entries.append(
                {
                    "task": task.name,
                    "region": region,
                    "levels": list(LEVELS),
                    "coverage": joint,
                    "n_trials": int(rows.size),
                    "per_dimension": per_dimension,
                }
            )
    return {"per_task": entries}
