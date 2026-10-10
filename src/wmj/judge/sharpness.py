"""wmj.judge.sharpness — how narrow are the stated ranges? (ADR-J3, JU-5)

In plain words: a model can be perfectly "calibrated" by saying nothing useful — a range wide
enough to catch everything. So whenever calibration is reported, so is how wide the model's 90%
ranges were: the average width, over trials and quantities, at the same judging step as the
calibration. Smaller is sharper. Width is exactly 2 × 1.6449 × the stated spread, so it can only
shrink if the model states smaller spreads (TC-JU5-02).
"""

from __future__ import annotations

import numpy as np

from wmj.judge._normal import Z_90
from wmj.judge.calibration import region_rows, require_finite, task_region_step
from wmj.judge.errors import JudgeInputError
from wmj.judge.types import JudgeInput


def compute_sharpness(inp: JudgeInput) -> dict:
    """The verdict's `sharpness` block: mean 90% interval width, per task and region."""
    if type(inp) is not JudgeInput:
        raise JudgeInputError(f"compute_sharpness needs a JudgeInput, got {type(inp).__name__}")
    rows_by_region = region_rows(inp)
    entries = []
    for task in inp.tasks:
        for region, rows in rows_by_region.items():
            index = task_region_step(inp, task, region) - 1
            with np.errstate(over="ignore", invalid="ignore"):
                width = require_finite(2.0 * Z_90 * inp.predictions.spread[rows, index, :], "a stated 90% interval width")
                mean_width = require_finite(np.mean(width), "the mean 90% interval width")
            entries.append({"task": task.name, "region": region, "mean_width_90": float(mean_width)})
    return {"per_task": entries}
