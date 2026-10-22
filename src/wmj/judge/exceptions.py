"""wmj.judge.exceptions — the backtest: count the misses and band them (ADR-J4, JU-8).

In plain words: every independent test start is a bet that "the truth lands inside my stated 90%
region". The bet is about the whole answer, all the quantities together: a miss (an "exception") is
a trial whose misses, measured in the model's own stated sizes and combined over all the quantities,
fall outside the joint 90% region (`_chi2.py`). Landing exactly on the edge is inside. Checking each
quantity separately would make a perfectly honest 4-quantity model miss 34% of the time instead of
10% (code-review-003), so an honest model now misses 10% whatever the number of quantities. The misses are counted for each task and region at
two pre-declared steps — step 1 and the step where that task is judged — and the count is placed
in the colour band fixed before judging (green is what honest luck looks like, amber is a
warning, red is out). A count that is *too low* is not rewarded: if it is at or below the top of green and the
ranges were very wide, the entry is flagged as possible padding. The same per-trial list of
hit/miss flags that the count is made from is returned for the main chart, and the chart plots the
very numbers the decision compares: each trial's whole-answer miss in the model's own stated sizes (the
root-mean-square over the quantities of miss / stated spread) against the fixed line of the 90% region
on that scale. A dot above the line is a miss and a dot on or below it is not, always (fix-check of
code-review-003: a chart in plain distance against an averaged spread disagreed with the flag about one
trial in ten when the stated spreads differ between quantities). How wide the ranges were is shown by
sharpness, not by this chart.
"""

from __future__ import annotations

import numpy as np

from wmj.judge._chi2 import rms_limit
from wmj.judge.calibration import task_region_step
from wmj.judge.distance import miss_in_stated_sizes
from wmj.judge.errors import JudgeInputError
from wmj.judge.regions import region_rows, require_finite
from wmj.judge.sharpness import mean_width_90
from wmj.judge.types import JudgeInput

DISTANCE_UNIT = "stated-size: root-mean-square over the quantities of each miss divided by the model's own stated spread"


def _band_for(observed: int, green: tuple[int, int], amber_outer: tuple[int, int]) -> str:
    """Green inside its range, amber between green and the outer amber edges, red beyond."""
    if green[0] <= observed <= green[1]:
        return "green"
    if amber_outer[0] <= observed <= amber_outer[1]:
        return "amber"
    return "red"


def compute_exceptions_and_trials(inp: JudgeInput) -> tuple[dict, dict]:
    """The verdict's `exceptions` and `trials` blocks, built from the one per-trial miss list."""
    if type(inp) is not JudgeInput:
        raise JudgeInputError(f"compute_exceptions_and_trials needs a JudgeInput, got {type(inp).__name__}")
    bands = inp.thresholds.bands
    g_lo, g_hi = bands.green
    a_lo, a_hi = bands.amber_outer
    if a_lo >= g_lo or a_hi <= g_hi:
        raise JudgeInputError(
            f"the pre-registered bands {bands.green}/{bands.amber_outer} leave no amber range on one side of green "
            "— the judge will not invent one (ADR-J4)"
        )
    expected_count = float(bands.n) * float(bands.p)
    with np.errstate(over="ignore", invalid="ignore"):
        hedge_threshold = float(require_finite(np.mean(inp.thresholds.sharpness_hedge_threshold), "the hedging threshold"))
    exception_entries, trial_entries = [], []
    for task in inp.tasks:
        for region, rows in region_rows(inp).items():
            judged = task_region_step(inp, task, region)
            for step in sorted({1, judged}):
                mean = inp.predictions.mean[rows, step - 1, :]
                spread = inp.predictions.spread[rows, step - 1, :]
                with np.errstate(over="ignore", invalid="ignore"):
                    miss = require_finite(np.abs(inp.outcomes[rows, step - 1, :] - mean), "outcome minus predicted mean")
                line = rms_limit(miss.shape[1], 0.9)  # the joint 90% region of all the quantities together, on the chart's scale
                whole = miss_in_stated_sizes(miss, spread)
                is_exception = whole > line  # the edge itself is inside; the chart plots exactly `whole` against `line`
                observed = int(np.sum(is_exception))
                band = _band_for(observed, bands.green, bands.amber_outer)
                key = {"task": task.name, "region": region, "horizon_step": step}
                exception_entries.append(
                    {
                        **key,
                        "n_trials": int(rows.size),
                        "expected": expected_count,
                        "observed": observed,
                        "band": band,
                        "low_side_sharpness_flag": bool(
                            observed <= g_hi and mean_width_90(inp, rows, step) > hedge_threshold
                        ),
                        "bands": {"green": [g_lo, g_hi], "amber": [[a_lo, g_lo - 1], [g_hi + 1, a_hi]], "red": "outside"},  # a fresh record per entry
                    }
                )
                trial_entries.append(
                    {
                        **key,
                        "distance_unit": DISTANCE_UNIT,
                        "outcome_distance": [float(v) for v in whole],
                        "band_lo": [0.0] * int(rows.size),
                        "band_hi": [float(line)] * int(rows.size),
                        "is_exception": [bool(v) for v in is_exception],
                    }
                )
    return {"per_task": exception_entries}, {"per_task": trial_entries}
