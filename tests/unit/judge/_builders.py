"""Small builders for judge tests: a valid JudgeInput and a valid set of verdict blocks.

In plain words: tests need a complete, correct input and a complete, correct set of
verdict blocks to start from; each test then breaks exactly one thing and checks the
judge refuses (or accepts) it.
"""

from __future__ import annotations

import copy

import numpy as np

from wmj.judge.types import (
    Bands,
    ClimatologyBin,
    Forecasts,
    JudgeInput,
    RegionClimatology,
    RegionCurve,
    RegionLabel,
    TaskSpec,
    Thresholds,
)

N, H, D = 6, 5, 2


def forecasts(seed: int = 0, n: int = N, h: int = H, d: int = D) -> Forecasts:
    rng = np.random.default_rng(seed)
    return Forecasts(mean=rng.normal(size=(n, h, d)), spread=rng.uniform(0.1, 1.0, size=(n, h, d)))


def judge_input_kwargs() -> dict:
    rng = np.random.default_rng(7)
    bins = tuple(
        ClimatologyBin(
            invariant_lo=-np.inf if i == 0 else float(i),
            invariant_hi=np.inf if i == 3 else float(i + 1),
            mean=np.array([1.0, 2.0]) * (i + 1),
            sd=np.array([0.1, 0.2]),
            n_samples=100,
        )
        for i in range(4)
    )
    return {
        "world": "lv",
        "dt": 0.02,
        "natural_cycle_length": 3.5,
        "predictions": forecasts(1),
        "outcomes": rng.normal(size=(N, H, D)),
        "persistence": forecasts(2),
        "linear": forecasts(3),
        "region_labels": tuple(
            RegionLabel("training", None) if i < 4 else RegionLabel("out-of-range", "state") for i in range(N)
        ),
        "divergence_curves": (
            RegionCurve("training", np.linspace(0.0, 1.0, H + 1)),
            RegionCurve("out-of-range", np.linspace(0.0, 2.0, H + 1)),
        ),
        "climatology": (
            RegionClimatology("training", bins),
            RegionClimatology("out-of-range", bins),
        ),
        "invariant_bins": rng.integers(0, 4, size=(N, H)),
        "tasks": (
            TaskSpec("lv-control", "control", 0.1, 3),
            TaskSpec("lv-planning", "planning", 0.3, 5),
        ),
        "thresholds": Thresholds(
            bands=Bands(n=200, p=0.1, green=(12, 29), amber_outer=(8, 35)),
            sharpness_hedge_threshold=np.array([4.0, 2.5]),
            agreement_threshold=1.0,
        ),
    }


def make_input(**overrides) -> JudgeInput:
    kwargs = judge_input_kwargs()
    kwargs.update(overrides)
    return JudgeInput(**kwargs)


def good_blocks() -> dict:
    """Eight valid metric blocks (everything the judge computes except the two constant groups).

    Two tasks in one region ("training"); every group covers both, as the spec requires.
    """
    keys = [("lv-control", "training"), ("lv-planning", "training")]
    flags = {"lv-control": [True, False, True, False], "lv-planning": [False, False, True, False]}
    return {
        "skill": {"per_task_region": [
            {"task": t, "region": r, "vs_persistence": 0.4, "vs_linear": 0.3, "crps": 0.03} for t, r in keys]},
        "error_vs_horizon": {"dt": 0.02, "per_region": [{
            "region": "training", "steps": [0, 1, 2], "median_error": [0.0, 0.1, 0.2],
            "divergence_reference": [0.0, 0.2, 0.4]}]},
        "calibration": {"per_task": [{
            "task": t, "region": r, "levels": [0.5, 0.8, 0.9, 0.95], "coverage": [0.5, 0.8, 0.9, 0.94],
            "n_trials": 200, "per_dimension": [[0.5, 0.5], [0.8, 0.8], [0.9, 0.9], [0.95, 0.94]]} for t, r in keys]},
        "sharpness": {"per_task": [{"task": t, "region": r, "mean_width_90": 0.18} for t, r in keys]},
        "exceptions": {"per_task": [
            {"task": t, "region": r, "horizon_step": 1, "n_trials": 4, "expected": 0.4,
             "observed": sum(flags[t]), "band": "green", "low_side_sharpness_flag": False} for t, r in keys]},
        "trials": {"per_task": [
            {"task": t, "region": r, "horizon_step": 1, "distance_unit": "rms-normalised",
             "outcome_distance": [0.1, 0.2, 0.3, 0.4], "band_lo": [0.0] * 4, "band_hi": [0.2] * 4,
             "is_exception": flags[t]} for t, r in keys]},
        "climatology": {"per_task": [
            {"task": "lv-control", "region": "training", "switch_step": None,
             "agreement_mean_abs_z": None, "agrees": None},
            {"task": "lv-planning", "region": "training", "switch_step": 4,
             "agreement_mean_abs_z": 0.6, "agrees": True}]},
        "trust_horizons": {"per_task": [
            {"task": "lv-control", "region": "training", "tolerance": 0.1, "steps": 118, "world_time": 2.36,
             "natural_units": "0.34 cycles"},
            {"task": "lv-planning", "region": "training", "tolerance": 0.3, "steps": 41, "world_time": 0.82,
             "natural_units": None}]},
    }


def blocks_copy() -> dict:
    return copy.deepcopy(good_blocks())
