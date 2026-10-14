"""Reproduces build/measurements/p4-c05b-d15.md's window table and seven-seed spread.

In plain words: builds the pendulum's reference table in several ways, runs 64 seeded trials of the
`out-near-inverted` region (unkicked and kicked) for the full 5,000-step horizon, and prints how well
the TRUE world agrees with each table over the judge's window (steps 4,750-5,000, the control task's
switch step to its horizon). No model is involved. Run:  python build/measurements/p4-c05b-reproduce.py
(about two minutes). Trials: run seed 20260825, purpose key 'd15-starts'. Table seeds: 20260825, 1-6.
"""

from __future__ import annotations

import numpy as np

from wmj.harness.benchmarks import benchmark_kicks, build_divergence_artefact, declared_regions, sample_region_starts
from wmj.harness.climatology import build_climatology_table, measure_invariant_bins, truth_mean_abs_z
from wmj.judge.climatology import switch_step
from wmj.models.base import SeedSource
from wmj.worlds import pendulum

WORLD, NAME, REGION, TRIAL_SEED = pendulum.WORLD, "pendulum", "out-near-inverted", 20260825
TABLE_SEEDS = (20260825, 1, 2, 3, 4, 5, 6)
RECIPES = ((200, 1000), (200, 5000), (50, 20000))


def main() -> None:
    trial_seeds = SeedSource(run_seed=TRIAL_SEED, my_name=None)
    tasks = {t.name: t for t in WORLD.tasks()}
    horizon = max(t.horizon for t in tasks.values())
    curve = np.array(build_divergence_artefact(NAME, WORLD, trial_seeds)["regions"][REGION]["median_separation"])
    switch = switch_step(curve, tasks["dp-control"].tolerance, tasks["dp-control"].horizon)
    print(f"dp-control switch step in {REGION}: {switch}")
    box, band = next((b, bd) for n, b, bd in declared_regions(WORLD) if n == REGION)
    starts = sample_region_starts(trial_seeds.rng_for(NAME, REGION, "d15-starts"), box, 64)
    kicks = benchmark_kicks(NAME, WORLD, REGION, band, trial_seeds, 64, horizon)
    runs = {}
    for label, actions in (("unkicked", np.zeros_like(kicks)), ("kicked", kicks)):
        state, history = starts.copy(), np.empty((64, horizon, starts.shape[1]))
        for k in range(horizon):
            state = WORLD.transition_batch(state, actions[:, k, :])
            history[:, k, :] = state
        runs[label] = history
    for source in ("training", REGION):
        for n_starts, steps in RECIPES:
            for label in ("unkicked", "kicked"):
                scores = []
                for seed in TABLE_SEEDS if source == REGION else TABLE_SEEDS[:1]:
                    table = build_climatology_table(NAME, WORLD, SeedSource(run_seed=seed, my_name=None), region_name=source, n_starts=n_starts, steps_per_start=steps)
                    bins = measure_invariant_bins(WORLD, runs[label], table)
                    scores.append((truth_mean_abs_z(table, runs[label], bins), truth_mean_abs_z(table, runs[label], bins, first_step=switch)))
                full = " ".join(f"{a:.2f}" for a, _ in scores)
                window = " ".join(f"{b:.2f}" for _, b in scores)
                print(f"table from {source:18s} {n_starts:3d}x{steps:5d} {label:8s} full: {full} | window: {window}")


if __name__ == "__main__":
    main()
