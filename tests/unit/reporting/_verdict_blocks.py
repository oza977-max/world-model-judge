"""Verdict-shaped envelopes made by the real judge, for the reporting tests.

In plain words: the charts must be tested on what the judge actually produces, not on hand-typed
dicts that agree with the chart's own assumptions. This builds envelopes from the judge's own
`compute_*` functions on the judge test builder's input (two regions, two tasks with switch steps).
"""

from __future__ import annotations

from tests.unit.judge._builders import forecasts, judge_input_kwargs, make_input
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


META = {"seed": 20260825, "platform": "linux-x86_64", "prereg_commit": "0" * 40, "thresholds_file": "prereg/thresholds.json"}


def full_envelope(*, name="direct", is_fixture=False, seed=1, world="lv", ref="opaque-0"):
    """A complete envelope: every verdict group computed by the real judge and assembled through its door."""
    from wmj.judge.calibration import compute_calibration as calibration
    from wmj.judge.exceptions import compute_exceptions_and_trials
    from wmj.judge.horizon import compute_trust_horizons
    from wmj.judge.skill import compute_skill
    from wmj.judge.verdict import assemble_verdict

    kw = judge_input_kwargs()
    kw.update(world=world, predictions=forecasts(seed))
    inp = JudgeInput(**kw)
    exceptions, trials = compute_exceptions_and_trials(inp)
    verdict = assemble_verdict(
        world=world, skill=compute_skill(inp), error_vs_horizon=compute_error_vs_horizon(inp),
        calibration=calibration(inp), sharpness=compute_sharpness(inp), exceptions=exceptions, trials=trials,
        climatology=compute_climatology(inp), trust_horizons=compute_trust_horizons(inp),
    )
    return {"model_ref": ref, "model_name": name, "is_fixture": is_fixture, "verdict": verdict.to_dict(), "meta": dict(META)}


def set_crps(env, crps, task="lv-control", region="training"):
    """Give one skill entry a chosen ordinary error (a test needs rankings it can predict)."""
    for entry in env["verdict"]["skill"]["per_task_region"]:
        if (entry["task"], entry["region"]) == (task, region):
            entry["crps"] = crps
    return env


def set_horizon(env, steps, task="lv-control", region="training", cycle=3.5):
    """Give one trust-horizon entry a chosen length, with its world time and cycle fraction to match."""
    dt = env["verdict"]["error_vs_horizon"]["dt"]
    for entry in env["verdict"]["trust_horizons"]["per_task"]:
        if (entry["task"], entry["region"]) == (task, region):
            entry.update(steps=steps, world_time=steps * dt, natural_units=f"{steps * dt / cycle:.2f} cycles")
    return env


def run_envelopes():
    """One world's run: two baselines, two unrigged models and a fixture, every one judged by the real judge."""
    return [
        full_envelope(name="persistence", seed=2, ref="opaque-0"),
        full_envelope(name="linear", seed=3, ref="opaque-1"),
        full_envelope(name="direct", seed=1, ref="opaque-2"),
        full_envelope(name="ensemble", seed=4, ref="opaque-3"),
        full_envelope(name="fx-brittle", is_fixture=True, seed=5, ref="opaque-4"),
    ]


__all__ = ["META", "Forecasts", "baseline_envelopes", "envelope", "full_envelope", "run_envelopes", "set_crps", "set_horizon"]
