"""wmj.harness.climatology — the reference table and the measured bins (worlds §5, judge ADR-J5).

In plain words: the judge asks, past the point where nobody can promise the exact path, "does the
model's guess still look like the world?". That needs a reference. This module builds it: it
starts the true world from 200 seeded points spread across the training region, lets each run 1,000
steps with no actions (200,000 samples in all), sorts every sample by the world's conserved
quantity (energy, or the predator-prey "orbit number") and cuts them into sixteen ranges holding
equally many samples. For each range it records the typical state and how widely the states
vary. The outermost ranges are open-ended, so every possible value has a range.

(The spec's recipe was one long trajectory from the training region's centre. In both worlds that
centre is a resting point, so the "trajectory" is a single point with a single conserved value and
no ranges at all — owner ruling 2026-10-14, backlog A36: use many starts instead.)

It also works out, for each test trial at each step, which range the *true* state is in — measured
afresh at every step, because a kick changes the conserved quantity and so moves the trial into a
different range — and measures how closely the true world agrees with its own table, which is what
a perfect model would score on the judge's agreement test (owner question D15). It refuses a
degenerate reference (too few distinct conserved values, a thin range, a range with no spread)
instead of handing the judge a table it would have to refuse.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from wmj.errors import WmjError
from wmj.harness.benchmarks import declared_regions, sample_region_starts
from wmj.judge.types import ClimatologyBin, RegionClimatology
from wmj.models.base import SeedSource

DEFAULT_N_STARTS = 200
DEFAULT_STEPS_PER_START = 1_000
N_BINS = 16
MIN_SAMPLES_PER_BIN = 50
BIN_METHOD = "equal-population"


class ClimatologyError(WmjError):
    """Raised when the reference run cannot make a usable table (degenerate invariant, thin or flat bin)."""


def _invariant_values(world: Any, states: np.ndarray) -> np.ndarray:
    return np.array([world.conserved(state) for state in states], dtype=float)


def build_climatology_table(
    world_name: str,
    world: Any,
    seeds: SeedSource,
    *,
    n_starts: int = DEFAULT_N_STARTS,
    steps_per_start: int = DEFAULT_STEPS_PER_START,
    n_bins: int = N_BINS,
    min_samples: int = MIN_SAMPLES_PER_BIN,
    region_name: str = "training",
) -> dict[str, Any]:
    """The worlds §5 climatology table for one world (see the module note).

    By default the starts are spread over the training region, as the spec says. `region_name` builds the
    same kind of table from another declared region's starts instead (open owner question D21)."""
    boxes = {name: box for name, box, _ in declared_regions(world)}
    if region_name not in boxes:
        raise ClimatologyError(f"{world_name}: no declared region {region_name!r} (declared: {sorted(boxes)})")
    box = boxes[region_name]
    rng = seeds.rng_for(world_name, region_name, "climatology-starts")
    states = sample_region_starts(rng, box, n_starts)
    null_actions = np.zeros((n_starts, world.a))
    sample_states, sample_values = [], []
    for _ in range(steps_per_start):
        states = world.transition_batch(states, null_actions)
        sample_states.append(states.copy())
        sample_values.append(_invariant_values(world, states))
    pooled_states = np.concatenate(sample_states)
    pooled_values = np.concatenate(sample_values)
    if not np.all(np.isfinite(pooled_values)):
        raise ClimatologyError(f"{world_name}: the reference run produced a non-finite conserved quantity")
    if np.unique(pooled_values).size < n_bins:
        raise ClimatologyError(
            f"{world_name}: the reference run has fewer than {n_bins} distinct conserved-quantity values "
            f"({np.unique(pooled_values).size}) — it cannot be cut into {n_bins} ranges"
        )
    inner = np.quantile(pooled_values, [k / n_bins for k in range(1, n_bins)])
    if not np.all(np.diff(inner) > 0.0):
        raise ClimatologyError(f"{world_name}: the {n_bins} equal-population range edges are not distinct (too many ties)")
    edges = np.concatenate([[-np.inf], inner, [np.inf]])
    which = np.searchsorted(inner, pooled_values, side="right")
    bins = []
    for k in range(n_bins):
        members = pooled_states[which == k]
        if members.shape[0] < min_samples:
            raise ClimatologyError(f"{world_name}: climatology bin {k} holds {members.shape[0]} samples, fewer than the required {min_samples}")
        sd = members.std(axis=0)
        if not np.all(sd > 0.0):
            raise ClimatologyError(f"{world_name}: climatology bin {k} has no spread in some quantity (sd = {sd.tolist()})")
        bins.append(
            {
                "invariant_range": [float(edges[k]), float(edges[k + 1])],
                "mean": [float(v) for v in members.mean(axis=0)],
                "sd": [float(v) for v in sd],
                "n_samples": int(members.shape[0]),
            }
        )
    return {
        "world": world_name,
        "reference_region": region_name,
        "reference_starts": n_starts,
        "steps_per_start": steps_per_start,
        "reference_run_steps": n_starts * steps_per_start,
        "bin_method": BIN_METHOD,
        "n_bins": n_bins,
        "bins": bins,
    }


def judge_tables(table: dict[str, Any], region_names: tuple[str, ...]) -> tuple[RegionClimatology, ...]:
    """The judge's input type: the one table, under each region's name."""
    bins = tuple(
        ClimatologyBin(
            invariant_lo=entry["invariant_range"][0],
            invariant_hi=entry["invariant_range"][1],
            mean=np.array(entry["mean"]),
            sd=np.array(entry["sd"]),
            n_samples=entry["n_samples"],
        )
        for entry in table["bins"]
    )
    return tuple(RegionClimatology(name, bins) for name in region_names)


def measure_invariant_bins(world: Any, states: np.ndarray, table: dict[str, Any]) -> np.ndarray:
    """`int64[n, H]`: the table range holding the conserved quantity of each true state, step by step.

    `states` is `[n_trials, H, d]`, the true state after each step. A value exactly on an edge belongs to
    the range above it. Measured at every step, never carried forward from the start (TC-JU6-02)."""
    inner = np.array([entry["invariant_range"][1] for entry in table["bins"][:-1]])
    n, steps, _ = states.shape
    values = np.array([[world.conserved(states[i, s]) for s in range(steps)] for i in range(n)], dtype=float)
    if not np.all(np.isfinite(values)):
        raise ClimatologyError("a true state has a non-finite conserved quantity")
    return np.searchsorted(inner, values, side="right").astype(np.int64)


def truth_mean_abs_z(table: dict[str, Any], states: np.ndarray, bins: np.ndarray) -> float:
    """Mean |state − bin mean| / bin sd over trials, steps and quantities (owner question D15).

    With true states and their own measured bins this is what a perfect model would score on the
    judge's climatology-agreement test."""
    mean = np.array([entry["mean"] for entry in table["bins"]])
    sd = np.array([entry["sd"] for entry in table["bins"]])
    return float(np.mean(np.abs(states - mean[bins]) / sd[bins]))
