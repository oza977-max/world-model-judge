"""Measures the D21 pass mark exactly as `build/measurements/d21-pass-mark.md` pins it.

In plain words: for every place the judge's climatology-agreement test runs, this builds 20 reference
tables and 5 sets of 200 kicked true-world trials, scores the true world (a perfect model) against
each table over the judge's own window, and reports the largest score. The mark is 1.10 times that
score, rounded up. It then scores two plainly wrong forecasts built from the truth alone (persistence,
and the true path with 10% too much energy) to see whether the test can tell them apart. No model is
involved. Run:  python build/measurements/d21-reproduce.py
"""

from __future__ import annotations

import math
import sys
import time

import numpy as np

from wmj.harness.benchmarks import benchmark_kicks, build_divergence_artefact, declared_regions, sample_region_starts
from wmj.harness.climatology import build_climatology_table, measure_invariant_bins, truth_mean_abs_z
from wmj.judge.climatology import switch_step
from wmj.models.base import SeedSource
from wmj.worlds import lv, pendulum

RUN_SEED = 20260825
TABLE_SEEDS = tuple(range(1, 21))
TRIAL_SEEDS = tuple(range(101, 106))
N_TRIALS, N_STARTS, STEPS_PER_START = 200, 50, 20000
HEADROOM = 1.10
WORLDS = {"lv": lv.WORLD, "pendulum": pendulum.WORLD}


def cells() -> list[tuple[str, object, str, np.ndarray, str, object, int]]:
    """(world name, world, region, box, kick band, task, switch step) for every cell where the test runs."""
    found = []
    for name, world in WORLDS.items():
        artefact = build_divergence_artefact(name, world, SeedSource(run_seed=RUN_SEED, my_name=None))
        for region, box, band in declared_regions(world):
            curve = np.array(artefact["regions"][region]["median_separation"])
            for task in world.tasks():
                switch = switch_step(curve, task.tolerance, task.horizon)
                print(f"{name:9s} {region:20s} {task.name:12s} switch: {switch}", flush=True)
                if switch is not None:
                    found.append((name, world, region, box, band, task, switch))
    return found


def trial_set(name, world, region, box, band, horizon, seed) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(starts, true states [n, H, d], conserved values [n, H]) for one seeded trial set."""
    seeds = SeedSource(run_seed=seed, my_name=None)
    starts = sample_region_starts(seeds.rng_for(name, region, "d21-starts"), box, N_TRIALS)
    kicks = benchmark_kicks(name, world, region, band, seeds, N_TRIALS, horizon)
    state, history = starts.copy(), np.empty((N_TRIALS, horizon, starts.shape[1]))
    for k in range(horizon):
        state = world.transition_batch(state, kicks[:, k, :])
        history[:, k, :] = state
    values = np.array([[world.conserved(history[i, s]) for s in range(horizon)] for i in range(N_TRIALS)])
    return starts, history, values


def bins_for(table: dict, values: np.ndarray) -> np.ndarray:
    """The same lookup as `measure_invariant_bins`, from conserved values computed once."""
    inner = np.array([entry["invariant_range"][1] for entry in table["bins"][:-1]])
    return np.searchsorted(inner, values, side="right").astype(np.int64)


def score(table: dict, forecast: np.ndarray, bins: np.ndarray, switch: int) -> float:
    """The judge's arithmetic: mean |forecast - bin mean| / bin sd over steps switch..H (index s - 1)."""
    mean = np.array([entry["mean"] for entry in table["bins"]])
    sd = np.array([entry["sd"] for entry in table["bins"]])
    window = slice(switch - 1, None)
    return float(np.mean(np.abs(forecast[:, window] - mean[bins[:, window]]) / sd[bins[:, window]]))


def energy_inflated(world, history: np.ndarray) -> np.ndarray:
    """The true path with each state's angular velocities scaled so its energy is E + 10% of |E| (pendulum)."""
    out = history.copy()
    n, steps, _ = history.shape
    for i in range(n):
        for s in range(steps):
            th1, th2, w1, w2 = history[i, s]
            total = world.conserved(history[i, s])
            potential = world.conserved(np.array([th1, th2, 0.0, 0.0]))
            kinetic = total - potential
            if kinetic > 0.0:
                factor = math.sqrt((total + 0.1 * abs(total) - potential) / kinetic)
                out[i, s, 2:] = history[i, s, 2:] * factor
    return out


def main() -> None:
    t0 = time.time()
    found = cells()
    print(f"cells where the test runs: {[(c[0], c[2], c[5].name, c[6]) for c in found]}", flush=True)
    marks: dict[str, float] = {}
    for name, world, region, box, band, task, switch in found:
        horizon = task.horizon
        tables = [build_climatology_table(name, world, SeedSource(run_seed=s, my_name=None), region_name=region,
                                          n_starts=N_STARTS, steps_per_start=STEPS_PER_START) for s in TABLE_SEEDS]
        print(f"[{time.time() - t0:.0f}s] built {len(tables)} tables for {name}/{region}", flush=True)
        truth, persist, inflated = [], [], []
        for seed in TRIAL_SEEDS:
            starts, history, values = trial_set(name, world, region, box, band, horizon, seed)
            frozen = np.broadcast_to(starts[:, None, :], history.shape)
            bumped = energy_inflated(world, history) if name == "pendulum" else None
            for ti, table in enumerate(tables):
                bins = bins_for(table, values)
                if seed == TRIAL_SEEDS[0] and ti == 0:  # the shortcut agrees with the harness's own functions
                    assert np.array_equal(bins, measure_invariant_bins(world, history, table))
                    assert score(table, history, bins, switch) == truth_mean_abs_z(table, history, bins, first_step=switch)
                truth.append(score(table, history, bins, switch))
                persist.append(score(table, frozen, bins, switch))
                if bumped is not None:
                    inflated.append(score(table, bumped, bins, switch))
            print(f"[{time.time() - t0:.0f}s] trial set {seed}: truth {min(truth[-20:]):.3f}..{max(truth[-20:]):.3f}", flush=True)
        truth_arr = np.array(truth)
        print(f"\n{name}/{region}/{task.name} (switch {switch}, window {switch}..{horizon}), 100 perfect-model scores:")
        print(f"  min {truth_arr.min():.4f}  median {np.median(truth_arr):.4f}  95th {np.quantile(truth_arr, 0.95):.4f}  max {truth_arr.max():.4f}")
        for label, values in (("persistence", persist), ("energy +10%", inflated)):
            if values:
                v = np.array(values)
                print(f"  {label:12s} min {v.min():.4f}  median {np.median(v):.4f}  max {v.max():.4f}")
        marks[name] = max(marks.get(name, 0.0), float(truth_arr.max()))
        np.save(f"/tmp/d21_{name}_{region}_{task.name}.npy", np.stack([truth_arr, np.array(persist),
                np.array(inflated) if inflated else np.full(len(truth), np.nan)]))
    for name in WORLDS:
        if name in marks:
            mark = math.ceil(HEADROOM * marks[name] * 100.0) / 100.0
            print(f"\nMARK {name}: 1.10 x {marks[name]:.4f} = {HEADROOM * marks[name]:.4f} -> rounded up {mark:.2f}")
        else:
            print(f"\nMARK {name}: the test runs in no cell -> 1.0, inert")
    print(f"total {time.time() - t0:.0f}s")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
