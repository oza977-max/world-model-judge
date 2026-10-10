"""wmj.judge.exceptions — the backtest: count the misses and band them (ADR-J4, JU-8).

In plain words: every independent test start is a bet that "the truth lands inside my stated 90%
range". A miss (an "exception") is any trial where even one of its quantities lands outside that
range; landing exactly on the edge is inside. The misses are counted for each task and region at
two pre-declared steps — step 1 and the step where that task is judged — and the count is placed
in the colour band fixed before judging (green is what honest luck looks like, amber is a
warning, red is out). A count that is *too low* is not rewarded: if it is green-or-better and the
ranges were very wide, the entry is flagged as possible padding. The same per-trial list of
hit/miss flags that the count is made from is returned for the main chart, so the chart and the
count cannot disagree.
"""

from __future__ import annotations

import numpy as np

from wmj.judge._normal import Z_90
from wmj.judge.calibration import region_rows, require_finite, task_region_step
from wmj.judge.errors import JudgeInputError
from wmj.judge.sharpness import compute_sharpness
from wmj.judge.types import JudgeInput

DISTANCE_UNIT = "rms-normalised: root-mean-square over the quantities, in the harness's normalised units"


def _rms(values: np.ndarray) -> np.ndarray:
    """Root-mean-square over the quantities of each trial; an overflow while squaring is refused, never charted."""
    with np.errstate(over="ignore", invalid="ignore"):
        return require_finite(np.sqrt(np.mean(values**2, axis=1)), "a plotted distance")


def _band_for(observed: int, green: tuple[int, int], amber: tuple[int, int, int, int]) -> str:
    if green[0] <= observed <= green[1]:
        return "green"
    if amber[0] <= observed <= green[0] - 1 or green[1] + 1 <= observed <= amber[3]:
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
    band_record = {"green": [g_lo, g_hi], "amber": [[a_lo, g_lo - 1], [g_hi + 1, a_hi]], "red": "outside"}
    hedge_threshold = float(np.mean(inp.thresholds.sharpness_hedge_threshold))
    widths = {(e["task"], e["region"]): e["mean_width_90"] for e in compute_sharpness(inp)["per_task"]}
    exception_entries, trial_entries = [], []
    for task in inp.tasks:
        for region, rows in region_rows(inp).items():
            judged = task_region_step(inp, task, region)
            for step in sorted({1, judged}):
                mean = inp.predictions.mean[rows, step - 1, :]
                with np.errstate(over="ignore", invalid="ignore"):
                    radius = require_finite(Z_90 * inp.predictions.spread[rows, step - 1, :], "a stated interval half-width")
                    miss = require_finite(np.abs(inp.outcomes[rows, step - 1, :] - mean), "outcome minus predicted mean")
                is_exception = np.any(miss > radius, axis=1)  # the edge itself is inside
                observed = int(np.sum(is_exception))
                band = _band_for(observed, bands.green, (a_lo, g_lo - 1, g_hi + 1, a_hi))
                key = {"task": task.name, "region": region, "horizon_step": step}
                exception_entries.append(
                    {
                        **key,
                        "n_trials": int(rows.size),
                        "expected": float(bands.n * bands.p),
                        "observed": observed,
                        "band": band,
                        "low_side_sharpness_flag": bool(
                            observed <= g_hi and widths[(task.name, region)] > hedge_threshold
                        ),
                        "bands": band_record,
                    }
                )
                trial_entries.append(
                    {
                        **key,
                        "distance_unit": DISTANCE_UNIT,
                        "outcome_distance": [float(v) for v in _rms(miss)],
                        "band_lo": [0.0] * int(rows.size),
                        "band_hi": [float(v) for v in _rms(radius)],
                        "is_exception": [bool(v) for v in is_exception],
                    }
                )
    return {"per_task": exception_entries}, {"per_task": trial_entries}
