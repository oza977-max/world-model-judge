"""wmj.judge.horizon — how far ahead can the model be believed? (ADR-J5, JU-3, JU-7)

In plain words: two things built from the same curve. First, the *error curve*: for each region,
the typical (median over the region's independent trials) distance between the model's best guess
and what really happened, after 1, 2, 3 … steps, drawn beside the world's own drift curve — the
yardstick for "how much error is unavoidable here". Second, the *trust horizon* for each task and
region: the last step up to which the typical error stayed within the task's tolerance at every
step, never counted past the step where the world itself has drifted beyond that tolerance (after
that, exact-path trust is not a claim anyone can earn). Failing at step 1 is an explicit 0. The
answer is given in steps, in world time, and — where the world has a natural cycle — as a fraction
of it, and always names its task, its region and its tolerance.
"""

from __future__ import annotations

import numpy as np

from wmj.judge.climatology import switch_step
from wmj.judge.distance import rms_distance
from wmj.judge.errors import JudgeInputError
from wmj.judge.regions import region_curve, region_rows, require_finite
from wmj.judge.types import JudgeInput


def _median_error_curves(inp: JudgeInput) -> dict[str, np.ndarray]:
    """Per region, the median distance at steps 1..H (array index s - 1)."""
    curves = {}
    for region, rows in region_rows(inp).items():
        distance = rms_distance(inp.predictions.mean[rows], inp.outcomes[rows])  # [trials, H]
        with np.errstate(over="ignore", invalid="ignore"):
            median = np.median(distance, axis=0)
            if not np.all(np.isfinite(median)):  # the average of two huge middle values overflowed: halve before adding
                ordered = np.sort(distance, axis=0)
                half = distance.shape[0] // 2
                median = ordered[half] if distance.shape[0] % 2 else ordered[half - 1] / 2.0 + ordered[half] / 2.0
            curves[region] = require_finite(median, "the median error")
    return curves


def compute_error_vs_horizon(inp: JudgeInput) -> dict:
    """The verdict's `error_vs_horizon` block: median error per step beside the world's drift, per region."""
    if type(inp) is not JudgeInput:
        raise JudgeInputError(f"compute_error_vs_horizon needs a JudgeInput, got {type(inp).__name__}")
    medians = _median_error_curves(inp)
    entries = []
    for region in medians:
        curve = region_curve(inp, region)
        entries.append(
            {
                "region": region,
                "steps": list(range(curve.size)),
                "median_error": [0.0] + [float(v) for v in medians[region]],  # step 0: the known start, no error
                "divergence_reference": [float(v) for v in curve],
            }
        )
    return {"dt": inp.dt, "per_region": entries}


def compute_trust_horizons(inp: JudgeInput) -> dict:
    """The verdict's `trust_horizons` block: per task and region, steps, world time and natural-cycle units."""
    if type(inp) is not JudgeInput:
        raise JudgeInputError(f"compute_trust_horizons needs a JudgeInput, got {type(inp).__name__}")
    medians = _median_error_curves(inp)
    entries = []
    for task in inp.tasks:
        for region in medians:
            switch = switch_step(region_curve(inp, region), task.tolerance, task.horizon)
            cap = task.horizon if switch is None else switch  # no switch: graded on the task's whole horizon (A31)
            within = medians[region][:cap] <= task.tolerance  # closed: exactly on the tolerance is within it
            steps = int(np.argmin(within)) if not within.all() else cap  # the first failing step s sits at index s - 1
            with np.errstate(over="ignore", invalid="ignore"):
                world_time = float(require_finite(np.float64(steps) * np.float64(inp.dt), "the trust horizon in world time"))
            cycle = inp.natural_cycle_length
            with np.errstate(over="ignore", invalid="ignore"):
                fraction = None if cycle is None else require_finite(np.float64(world_time) / cycle, "the trust horizon in natural cycles")
            entries.append(
                {
                    "task": task.name,
                    "region": region,
                    "tolerance": float(task.tolerance),
                    "steps": steps,
                    "world_time": world_time,
                    "natural_units": None if fraction is None else f"{float(fraction):.2f} cycles",
                }
            )
    return {"per_task": entries}
