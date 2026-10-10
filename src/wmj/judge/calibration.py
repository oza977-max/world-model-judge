"""wmj.judge.calibration — is "90% sure" really right 90% of the time? (ADR-J2, JU-4)

In plain words: a model states, for every number it predicts, a typical size of its own
mistake. If it is honest, then about half the time the true value should land inside the
range "prediction ± 0.67 of that size", about 80% of the time inside ± 1.28, 90% inside
± 1.64, and 95% inside ± 1.96. This module counts how often that actually happens, at those
four levels, for each task and region, at the task's own judging step. A trial counts as
"covered" only if *all* its quantities are inside (that is what a user of the prediction
experiences); the per-quantity rates are kept too, for diagnosis. It never combines this with
the CRPS skill score — they answer different questions and stay separate (TC-JU4-01).
"""

from __future__ import annotations

import numpy as np

from wmj.judge._normal import Z_50, Z_80, Z_90, Z_95
from wmj.judge.climatology import evaluation_step, switch_step
from wmj.judge.errors import JudgeInputError
from wmj.judge.types import JudgeInput

LEVELS = (0.5, 0.8, 0.9, 0.95)
Z_VALUES = (Z_50, Z_80, Z_90, Z_95)


def require_finite(array: np.ndarray, what: str) -> np.ndarray:
    """Refuse an array holding infinity or not-a-number (an overflow), instead of counting with it.

    At absurd magnitudes `inf <= inf` is true, which would report a miss of 3e308 as covered."""
    if not np.all(np.isfinite(array)):
        raise JudgeInputError(f"{what} overflowed (a gap or width too large to represent); the judge refuses rather than count with it")
    return array


def region_rows(inp: JudgeInput) -> dict[str, np.ndarray]:
    """For each region present, the indices of its trials (regions in sorted-name order)."""
    names = np.array([label.region_name for label in inp.region_labels], dtype=object)  # not fixed-width text: NumPy would strip trailing NULs
    return {region: np.flatnonzero(names == region) for region in sorted(set(names.tolist()))}


def task_region_step(inp: JudgeInput, task, region: str) -> int:
    """The step (1-based) at which this task is judged in this region."""
    curve = next(c.curve for c in inp.divergence_curves if c.region_name == region)
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
            joint, per_dimension = [], []
            for reach_z in reach:
                inside = miss <= reach_z  # closed: exactly z sigma away is still inside
                joint.append(float(np.mean(np.all(inside, axis=1))))
                per_dimension.append([float(v) for v in np.mean(inside, axis=0)])
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
