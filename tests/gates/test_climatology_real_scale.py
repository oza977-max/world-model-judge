"""P4-C05b real-scale gate: the climatology table on the two built worlds, and the D15 figures.

In plain words: the unit tests use a made-up world. This builds the real table for the real
predator-prey and double-pendulum worlds at the real size (200 starts x 1,000 steps), and checks
the table the judge will actually receive: sixteen ranges, every one with at least 50 samples and a
real spread, open at both ends, accepted by the judge's own input type, and that the true world's
own score against its table (what a perfect model would get, the D15 question) is a finite number and
that kicks do move trials between ranges. How *well* the true world agrees is recorded in
`build/measurements/p4-c05b-d15.md`, not asserted here: in some regions it does not. It does not pin
any trained or measured number, only relations that must hold.
"""

from __future__ import annotations

import numpy as np
import pytest

from wmj.harness.benchmarks import (
    benchmark_kicks,
    declared_regions,
    sample_region_starts,
)
from wmj.harness.climatology import (
    build_climatology_table,
    judge_tables,
    measure_invariant_bins,
    truth_mean_abs_z,
)
from wmj.models.base import SeedSource
from wmj.worlds import lv, pendulum

SEED = 20260825


@pytest.fixture(scope="module", params=[("lv", lv), ("pendulum", pendulum)], ids=["lv", "pendulum"])
def built(request):
    name, module = request.param
    seeds = SeedSource(run_seed=SEED, my_name=None)
    return name, module.WORLD, seeds, build_climatology_table(name, module.WORLD, seeds)


@pytest.mark.slow
def test_the_real_table_has_sixteen_well_populated_open_ended_ranges_the_judge_accepts(built):
    _, world, _, table = built
    bins = table["bins"]
    assert len(bins) == 16 and table["reference_run_steps"] == 1_000_000
    assert bins[0]["invariant_range"][0] == -np.inf and bins[-1]["invariant_range"][1] == np.inf
    assert min(b["n_samples"] for b in bins) >= 50 and sum(b["n_samples"] for b in bins) == 1_000_000
    assert all(min(b["sd"]) > 0.0 for b in bins)
    names = ("training", *(o.region_name for o in world.regions().out_regions))
    assert [t.region_name for t in judge_tables(table, names, world.scale)] == list(names)


@pytest.mark.slow
def test_the_true_world_agrees_with_its_own_table_kicked_or_not_and_kicks_do_move_trials_between_ranges(built):
    name, world, seeds, table = built
    horizon = 300
    for region_name, box, band in declared_regions(world):
        starts = sample_region_starts(seeds.rng_for(name, region_name, "d15-starts"), box, 40)
        kicks = benchmark_kicks(name, world, region_name, band, seeds, 40, horizon)
        results = {}
        for label, actions in (("unkicked", np.zeros_like(kicks)), ("kicked", kicks)):
            states = starts.copy()
            history = np.empty((40, horizon, starts.shape[1]))
            for step in range(horizon):
                states = world.transition_batch(states, actions[:, step, :])
                history[:, step, :] = states
            bins = measure_invariant_bins(world, history, table)
            results[label] = (history, bins, truth_mean_abs_z(table, history, bins))
        assert np.isfinite(results["kicked"][2]) and np.isfinite(results["unkicked"][2])
        # the invariant is re-measured at each step: with kicks some trials change range during the run
        moved_kicked = np.mean(results["kicked"][1] != results["kicked"][1][:, :1])
        moved_unkicked = np.mean(results["unkicked"][1] != results["unkicked"][1][:, :1])
        assert moved_kicked >= moved_unkicked
        if region_name in {"training", "out-large-action"}:
            assert moved_kicked > 0.0 and moved_unkicked == 0.0  # kicks, and only kicks, move these trials between ranges


@pytest.mark.slow
def test_the_real_table_and_measured_bins_feed_the_judges_climatology_block(built):
    from tests.unit.judge._builders import judge_input_kwargs
    from wmj.judge.climatology import compute_climatology
    from wmj.judge.types import (
        Forecasts,
        JudgeInput,
        RegionCurve,
        RegionLabel,
        TaskSpec,
    )

    name, world, seeds, table = built
    n, steps, d = 3, 40, world.regions().training_state_box.shape[0]
    box = world.regions().training_state_box
    states = sample_region_starts(seeds.rng_for(name, "training", "gate-starts"), box, n)
    history = np.empty((n, steps, d))
    for step in range(steps):
        states = world.transition_batch(states, np.zeros((n, world.a)))
        history[:, step, :] = states
    bins = measure_invariant_bins(world, history, table)  # raw states: the conserved quantity is a function of the world's own units
    normalised = history / world.scale  # the judge is handed forecasts and outcomes divided by the world's scale vector (A30)
    kwargs = judge_input_kwargs()
    kwargs.update(
        predictions=Forecasts(normalised, np.ones((n, steps, d))), outcomes=normalised,
        persistence=Forecasts(normalised, np.ones((n, steps, d))), linear=Forecasts(normalised, np.ones((n, steps, d))),
        region_labels=tuple(RegionLabel("training", None) for _ in range(n)),
        divergence_curves=(RegionCurve("training", np.linspace(0.0, 1.0, steps + 1)),),
        climatology=judge_tables(table, ("training",), world.scale), invariant_bins=bins,
        tasks=(TaskSpec("t", "control", 0.1, steps),),
    )
    kwargs["thresholds"] = type(kwargs["thresholds"])(
        type(kwargs["thresholds"].bands)(n=n, p=0.1, green=(1, 1), amber_outer=(0, 2)), np.ones(d), 1.0)
    entry = compute_climatology(JudgeInput(**kwargs))["per_task"][0]
    assert entry["switch_step"] is not None and np.isfinite(entry["agreement_mean_abs_z"])


@pytest.mark.slow
def test_the_truth_score_over_the_judges_window_equals_the_judges_own_agreement_score(built):
    from tests.unit.judge._builders import judge_input_kwargs
    from wmj.judge.climatology import compute_climatology
    from wmj.judge.types import (
        Bands,
        Forecasts,
        JudgeInput,
        RegionCurve,
        RegionLabel,
        TaskSpec,
        Thresholds,
    )

    name, world, seeds, table = built
    n, steps, switch = 4, 60, 20
    d = world.regions().training_state_box.shape[0]
    starts = sample_region_starts(seeds.rng_for(name, "training", "gate-starts-2"), world.regions().training_state_box, n)
    kicks = benchmark_kicks(name, world, "training", "in", seeds, n, steps)
    history = np.empty((n, steps, d))
    for step in range(steps):
        starts = world.transition_batch(starts, kicks[:, step, :])
        history[:, step, :] = starts
    bins = measure_invariant_bins(world, history, table)
    curve = np.zeros(steps + 1)
    curve[switch:] = 1.0  # the world's drift first exceeds the tolerance at step `switch`
    normalised = history / world.scale  # the judge works in the world's normalised units (A30)
    kwargs = judge_input_kwargs()
    kwargs.update(
        predictions=Forecasts(normalised, np.ones((n, steps, d))), outcomes=normalised,
        persistence=Forecasts(normalised, np.ones((n, steps, d))), linear=Forecasts(normalised, np.ones((n, steps, d))),
        region_labels=tuple(RegionLabel("training", None) for _ in range(n)),
        divergence_curves=(RegionCurve("training", curve),), climatology=judge_tables(table, ("training",), world.scale),
        invariant_bins=bins, tasks=(TaskSpec("t", "control", 0.5, steps),),
        thresholds=Thresholds(Bands(n=n, p=0.1, green=(1, 1), amber_outer=(0, 2)), np.ones(d), 1.0),
    )
    entry = compute_climatology(JudgeInput(**kwargs))["per_task"][0]
    assert entry["switch_step"] == switch
    assert entry["agreement_mean_abs_z"] == pytest.approx(truth_mean_abs_z(table, history, bins, first_step=switch), rel=1e-12)
