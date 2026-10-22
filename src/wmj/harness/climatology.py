"""wmj.harness.climatology — the reference table and the measured bins (worlds §5, judge ADR-J5).

In plain words: the judge asks, past the point where nobody can promise the exact path, "does the
model's guess still look like the world?". That needs a reference. This module builds it: it
starts the true world from 50 seeded points spread across a region (by default the training region;
the judged run builds one table per region), lets each run 20,000 steps with no actions (1,000,000
samples in all — owner decision D21: the spec's 200 x 1,000 was too short for even the true world to
agree with it at a 5,000-step task's late steps), sorts every sample by the world's conserved
quantity (energy, or the predator-prey "orbit number") and cuts them into sixteen bins holding
equally many samples. For each range it records the typical state and how widely the states
vary. The outermost bins are open-ended, so every possible value has a bin.

(The spec's recipe was one long trajectory from the training region's centre. In both worlds that
centre is a resting point, so the "trajectory" is a single point with a single conserved value and
no bins at all — owner ruling 2026-10-14, backlog A36: use many starts instead.)

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

# D21 (2026-10-22): long runs, so the table reaches the late states a 5,000-step task visits; the spec's
# 200 x 1,000 made even the true world fail the agreement test (build/measurements/d21-pass-mark.md)
DEFAULT_N_STARTS = 50
DEFAULT_STEPS_PER_START = 20_000
N_BINS = 16
MIN_SAMPLES_PER_BIN = 50
BIN_METHOD = "equal-population"
RELATIVE_SPREAD_FLOOR = 1e-9  # a bin's spread must exceed this fraction of the quantity's size, or it is flat


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
    for label, value, smallest in (("n_starts", n_starts, 1), ("steps_per_start", steps_per_start, 1), ("n_bins", n_bins, 2), ("min_samples", min_samples, 1)):
        if type(value) is not int or value < smallest:
            raise ClimatologyError(f"{world_name}: {label} must be a whole number >= {smallest}, got {value!r}")
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
            f"({np.unique(pooled_values).size}) — it cannot be cut into {n_bins} bins"
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
        mean = members.mean(axis=0)
        floor = RELATIVE_SPREAD_FLOOR * np.maximum(np.abs(mean), np.asarray(world.scale, dtype=float))
        if not (np.all(np.isfinite(sd)) and np.all(sd > floor)):
            raise ClimatologyError(
                f"{world_name}: climatology bin {k} has no real spread in some quantity (sd = {sd.tolist()}, "
                f"smallest acceptable {floor.tolist()}) — a flat quantity would make every standardised gap enormous"
            )
        bins.append(
            {
                "invariant_range": [float(edges[k]), float(edges[k + 1])],
                "mean": [float(v) for v in mean],
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


def serialisable_table(table: dict[str, Any]) -> dict[str, Any]:
    """The same table with the two open ends written as `null` (the project's one serializer refuses infinity).

    `judge_tables` reads either form: `null` as the lower end means "no lower limit", as the upper end "no upper limit"."""
    bins = [
        dict(entry, invariant_range=[None if not np.isfinite(v) else v for v in entry["invariant_range"]], mean=list(entry["mean"]), sd=list(entry["sd"]))
        for entry in table["bins"]
    ]
    return dict(table, bins=bins)


def judge_tables(table: dict[str, Any], region_names: tuple[str, ...], scale: np.ndarray) -> tuple[RegionClimatology, ...]:
    """The judge's input type: the one table, under each region's name, **in the judge's units**.

    The table is built in the world's own units (rabbit counts, radians). The judge is handed forecasts and
    outcomes already divided by the world's scale vector (backlog A30), so the table's typical state and spread
    must be divided by the same vector or the agreement score compares numbers in different units. `scale` is
    required for that reason: a caller cannot forget it. (`measure_invariant_bins` still takes RAW states —
    the conserved quantity is a function of the world's own units.)"""
    scale = np.asarray(scale, dtype=float)
    width = len(table["bins"][0]["mean"])
    if scale.shape != (width,) or not np.all(np.isfinite(scale)) or not np.all(scale > 0.0):
        raise ClimatologyError(f"scale must be {width} positive finite numbers (one per quantity), got {scale.tolist()}")
    bins = tuple(
        ClimatologyBin(
            invariant_lo=-np.inf if entry["invariant_range"][0] is None else entry["invariant_range"][0],
            invariant_hi=np.inf if entry["invariant_range"][1] is None else entry["invariant_range"][1],
            mean=np.array(entry["mean"]) / scale,
            sd=np.array(entry["sd"]) / scale,
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
    if states.ndim != 3:
        raise ClimatologyError(f"states must be [n_trials, H, d], got shape {states.shape}")
    n, steps, _ = states.shape
    values = np.array([[world.conserved(states[i, s]) for s in range(steps)] for i in range(n)], dtype=float)
    if not np.all(np.isfinite(values)):
        raise ClimatologyError("a true state has a non-finite conserved quantity")
    return np.searchsorted(inner, values, side="right").astype(np.int64)


def truth_mean_abs_z(table: dict[str, Any], states: np.ndarray, bins: np.ndarray, first_step: int = 1) -> float:
    """Mean |state − bin mean| / bin sd over trials, steps `first_step..H` and quantities (owner question D15).

    With true states and their own measured bins this is what a perfect model would score on the judge's
    climatology-agreement test *if scored over the same window* — the judge averages only from the task's
    switch step to its horizon, so pass the switch step as `first_step` to reproduce that, and cut `states` to the task's horizon first (the judge
    never looks past it). Step `s` is array index `s − 1`."""
    if not 1 <= first_step <= states.shape[1]:
        raise ClimatologyError(f"first_step must be in 1..{states.shape[1]}, got {first_step}")
    if np.any(bins < 0) or np.any(bins >= len(table["bins"])):
        raise ClimatologyError("a bin index lies outside the table")
    mean = np.array([entry["mean"] for entry in table["bins"]])
    sd = np.array([entry["sd"] for entry in table["bins"]])
    window = slice(first_step - 1, None)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        score = float(np.mean(np.abs(states[:, window] - mean[bins[:, window]]) / sd[bins[:, window]]))
    if not np.isfinite(score):
        raise ClimatologyError("the truth-agreement score is not a finite number (a non-finite state, or a bin with no spread)")
    return score
