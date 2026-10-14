"""Input variants shared by the blindness and purity tests: every branch of the judge that a real run can take.

In plain words: a judge test that only feeds one shape of input only proves the judge on that shape. These
variants cover ordinary random forecasts, a near-perfect model (trust horizons above zero, agreement holding),
a world that never drifts past a tolerance (no switch step: the predator-prey case), a world with no natural
cycle (the pendulum), out-of-region axes, and a third task. Tasks are given in a non-alphabetical order so an
output that quietly sorts them is noticed.
"""

from __future__ import annotations

import numpy as np

from tests.unit.judge._builders import judge_input_kwargs
from wmj.judge.types import (
    ClimatologyBin,
    Forecasts,
    RegionClimatology,
    RegionCurve,
    RegionLabel,
    TaskSpec,
)


def random_kwargs(seed: int, *, near_perfect: bool = False) -> dict:
    """Random forecasts with a different curve and climatology table per region (so a mix-up of regions changes the numbers).

    `near_perfect` puts the forecasts close to the outcomes and the outcomes on the climatology, so trust horizons are
    non-zero and the agreement test holds — the branches a random input never reaches."""
    kwargs = judge_input_kwargs()
    rng = np.random.default_rng(seed)
    n, h, d = kwargs["predictions"].mean.shape
    kwargs["predictions"] = Forecasts(rng.normal(size=(n, h, d)), rng.uniform(0.3, 2.0, size=(n, h, d)))
    kwargs["outcomes"] = rng.normal(size=(n, h, d))
    kwargs["divergence_curves"] = (
        RegionCurve("training", np.array([0.0, 0.01, 0.02, 0.2, 0.5, 0.6])),
        RegionCurve("out-of-range", np.array([0.0, 0.05, 0.3, 0.4, 0.45, 0.5])),
    )
    kwargs["climatology"] = tuple(
        RegionClimatology(
            name,
            tuple(
                ClimatologyBin(-np.inf if i == 0 else float(i), np.inf if i == 3 else float(i + 1),
                               np.array([1.0, 2.0]) * (i + 1) * scale, np.array([0.1, 0.2]) * (i + 1) * scale, 100)
                for i in range(4)
            ),
        )
        for name, scale in (("training", 1.0), ("out-of-range", 7.0))
    )
    kwargs["tasks"] = (TaskSpec("lv-planning", "planning", 0.3, 5), TaskSpec("lv-control", "control", 0.1, 3))
    if near_perfect:
        bins = np.zeros((n, h), dtype=np.int64)
        means = np.stack([kwargs["climatology"][0 if x.region_name == "training" else 1].bins[0].mean for x in kwargs["region_labels"]])
        target = np.broadcast_to(means[:, None, :], (n, h, d)).copy()
        target += rng.normal(scale=1e-4, size=target.shape)
        kwargs.update(outcomes=target, predictions=Forecasts(target + rng.normal(scale=1e-3, size=target.shape), np.full((n, h, d), 0.5)),
                      invariant_bins=bins)
    return kwargs


def all_variants() -> list[dict]:
    """Random inputs, near-perfect ones, and the structural variants (see the module note)."""
    variants = [random_kwargs(seed) for seed in range(8)] + [random_kwargs(100 + seed, near_perfect=True) for seed in range(4)]
    no_cycle = random_kwargs(200, near_perfect=True)
    no_cycle["natural_cycle_length"] = None
    axes = random_kwargs(201)
    axes["region_labels"] = tuple(
        RegionLabel(x.region_name, ("action", "both", "state")[i % 3] if x.region_name != "training" else None)
        for i, x in enumerate(axes["region_labels"])
    )
    three_tasks = random_kwargs(202, near_perfect=True)
    three_tasks["tasks"] = (*three_tasks["tasks"], TaskSpec("lv-extra", "planning", 0.05, 4))
    for seed in (300, 301):  # the world never drifts past a tolerance: no switch step, tasks graded on their whole horizon
        quiet = random_kwargs(seed, near_perfect=seed == 301)
        quiet["divergence_curves"] = tuple(RegionCurve(c.region_name, np.zeros(c.curve.size)) for c in quiet["divergence_curves"])
        variants.append(quiet)
    # error that grows with the step, so a task stops being within tolerance part-way through its horizon (the ordinary real case)
    mid = random_kwargs(400, near_perfect=True)
    h = mid["predictions"].mean.shape[1]
    growth = (0.08 * np.arange(1, h + 1))[None, :, None]
    mid["predictions"] = Forecasts(mid["outcomes"] + growth, mid["predictions"].spread)
    # ... and a world where one region has a switch step at step 1 and the other has none
    mixed = random_kwargs(401, near_perfect=True)
    mixed["predictions"] = Forecasts(mixed["outcomes"] + growth, mixed["predictions"].spread)
    mixed["divergence_curves"] = (
        RegionCurve("training", np.zeros(h + 1)),
        RegionCurve("out-of-range", np.array([0.0, 0.5, 0.6, 0.7, 0.8, 0.9])),
    )
    return [*variants, no_cycle, axes, three_tasks, mid, mixed]


def mid_horizon_entries(kwargs: dict, record: dict) -> list[dict]:
    """Trust-horizon entries that stopped part-way: 0 < steps < the cap (the switch step, or the task's horizon when there is none)."""
    horizons = {t.name: t.horizon for t in kwargs["tasks"]}
    switches = {(e["task"], e["region"]): e["switch_step"] for e in record["climatology"]["per_task"]}
    found = []
    for e in record["trust_horizons"]["per_task"]:
        switch = switches[(e["task"], e["region"])]
        cap = horizons[e["task"]] if switch is None else switch
        if 0 < e["steps"] < cap:
            found.append(e)
    return found
