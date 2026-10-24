"""Verdict-shaped envelopes made by the real judge, for the reporting tests.

In plain words: the charts must be tested on what the judge actually produces, not on hand-typed
dicts that agree with the chart's own assumptions. This builds envelopes from the judge's own
`compute_*` functions on the judge test builder's input (two regions, two tasks with switch steps).
"""

from __future__ import annotations

from tests.unit.judge._builders import judge_input_kwargs, make_input
from wmj.judge.calibration import compute_calibration
from wmj.judge.climatology import compute_climatology
from wmj.judge.horizon import compute_error_vs_horizon
from wmj.judge.sharpness import compute_sharpness
from wmj.judge.types import Forecasts, JudgeInput


def envelope(*, name="direct", is_fixture=False, inp=None, world="lv", predictions=None):
    if predictions is not None:
        kw = judge_input_kwargs()
        kw["predictions"] = predictions
        inp = JudgeInput(**kw)
    inp = inp if inp is not None else make_input()
    return {
        "model_ref": "opaque-0",
        "model_name": name,
        "is_fixture": is_fixture,
        "verdict": {
            "world": world,
            "error_vs_horizon": compute_error_vs_horizon(inp),
            "climatology": compute_climatology(inp),
            "calibration": compute_calibration(inp),
            "sharpness": compute_sharpness(inp),
        },
        "meta": {},
    }


def baseline_envelopes(seed_offset: int = 10) -> list[dict]:
    """Two baseline envelopes on the same world and trials as `envelope()` (different predictions)."""
    from tests.unit.judge._builders import forecasts

    return [
        envelope(name="persistence", predictions=forecasts(seed_offset)),
        envelope(name="linear", predictions=forecasts(seed_offset + 1)),
    ]


__all__ = ["Forecasts", "baseline_envelopes", "envelope"]
