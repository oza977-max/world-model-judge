"""wmj.harness.benchmarks — builds the WD-4 divergence artefact.

In plain words: this is the harness command that runs the "how fast do
nearby trajectories separate" experiment for a world, exactly the way
the spec declares it — 64 seeded starting points in each declared
region, a hair of perturbation, the same seeded kicks applied to each
start and its twin (design-review-010: the graded trials carry kicks,
so the reference line must too), watch the gap at every step — and packages the answer into the fixed JSON shape the judge and
the charts read (worlds spec §5). It also measures how much the
integrator itself leaks the world's conserved quantity over the run,
and refuses to produce the artefact if that leak is over the declared
bound (ADR-W1, TC-WD3-03) — a corrupted reference is worse than none.

This module RETURNS the artefact dict. Writing it under `out/` is
reporting's job, the sole writer of every artefact there
(cross-cutting ADR-002 rule 4, design-review-008 C8).

**Drift normalisation, stated (executed before writing — see
build/prompts/P2-C03.md; corrected per design-review-009 I1/I2):** the
spec pins the drift bound as "1e-6 (relative)". LV's invariant V(x, y)
passes through zero inside the training box, so "relative to the
initial value" divides by numbers as small as 8e-4 and reports 1.6e-6
for an absolute drift of 2.2e-9 — an artefact of the normaliser, not a
measurement. The bound exists so integrator error cannot corrupt the
climatology's *binning* of the invariant (ADR-W1), whose natural unit
is the invariant's dynamic range **over the region the climatology is
built from** — not pooled across regions. An earlier version of this
module pooled every declared region's benchmark starts into one span,
which design-review-009's blind panel executed and found dilutes the
figure for the training region specifically (3.4e-8 reported where the
training region alone gives 6.7e-9 — the wrong direction for a safety
gate) and entangles the normaliser with `n_starts` and the benchmark's
RNG. `conserved_rel_drift_max` is now computed **per region**, from a
deterministic grid over that region's own box
(`wmj.worlds.divergence.conserved_quantity_range` — no RNG, no
`n_starts`, never pooled with another region):

    max over that region's starts of |ΔV|  /  (grid max V − grid min V, that region's box)

and the literal initial-value-relative figure is reported alongside,
per region, as `conserved_rel_drift_max_vs_initial`, so nothing is
hidden. `drift.bound`/`within_bound` at the top level summarise across
`drift.per_region`. Recorded as a spec correction candidate
(build/spec-corrections-backlog.md, A7).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from wmj.harness.kicks import seeded_kick_sequences
from wmj.models.base import SeedSource
from wmj.worlds.actions import band_for_axis
from wmj.worlds.divergence import (
    DegenerateInvariantRangeError,
    assert_drift_within_bound,
    conserved_drift_batch,
    conserved_quantity_range,
    median_separation_curve,
    separation_curves_batch,
)

DEFAULT_N_STARTS = 64
DEFAULT_DELTA0 = 1e-6
DEFAULT_DRIFT_BOUND = 1e-6
DISTANCE_NAME = "rms-normalised"


def sample_region_starts(rng: np.random.Generator, box: np.ndarray, n: int) -> np.ndarray:
    """`n` starting states uniform inside an axis-aligned `[d, 2]` box.

    The generator is the caller's — the harness derives it from
    `seeds.rng_for(world, region, "benchmark-starts")` so benchmark
    starts never collide with training or evaluation starts for the
    same (world, region) (cross-cutting ADR-002 rule 2).
    """
    return rng.uniform(box[:, 0], box[:, 1], size=(n, box.shape[0]))


def declared_regions(world: Any) -> list[tuple[str, np.ndarray, str]]:
    """`(region_name, state_box, kick_band)` for every declared region, in order.

    The band is the region's own (worlds ADR-W4): `"out"` only for a region
    that is out of training on the action axis (`out-large-action`),
    `"in"` for the training region and state-axis regions.
    """
    spec = world.regions()
    regions = [("training", spec.training_state_box, band_for_axis(None))]
    regions.extend(
        (out.region_name, out.state_box, band_for_axis(out.axis)) for out in spec.out_regions
    )
    return regions


def benchmark_kicks(
    world_name: str,
    world: Any,
    region_name: str,
    band: str,
    seeds: SeedSource,
    n_starts: int,
    horizon: int,
) -> np.ndarray:
    """`float64[n_starts, horizon, 1]` — start `i`'s kicks, from
    `seeds.rng_for(world, region, "benchmark-kicks", str(i))` (worlds ADR-W3)."""
    return seeded_kick_sequences(
        seeds, world_name, world, region_name, band, "benchmark-kicks", n_starts, horizon
    )


def build_divergence_artefact(
    world_name: str,
    world: Any,
    seeds: SeedSource,
    n_starts: int = DEFAULT_N_STARTS,
    horizon: int | None = None,
    delta0: float = DEFAULT_DELTA0,
    drift_bound: float = DEFAULT_DRIFT_BOUND,
) -> dict[str, Any]:
    """The worlds spec §5 divergence artefact for one world.

    Raises `DriftBoundError` — loudly, producing nothing — if the
    integrator's conserved-quantity drift over the horizon is not below
    `drift_bound` (TC-WD3-03).
    """
    if horizon is None:
        horizon = max(task.horizon for task in world.tasks())

    regions_out: dict[str, dict[str, list[float]]] = {}
    drift_per_region: list[dict[str, Any]] = []
    steps = list(range(horizon + 1))
    # The null action is the world's own zero vector, not a hardcoded
    # shape (code-review-001, Panel C): both built worlds have a = 1,
    # but this function is written for any world.
    null_action = np.zeros(world.a)

    for region_name, box, band in declared_regions(world):
        rng = seeds.rng_for(world_name, region_name, "benchmark-starts")
        starts = sample_region_starts(rng, box, n_starts)

        # Kicked, batched: every start and its twin advance together, each
        # pair under the start's own seeded kicks (worlds ADR-W3,
        # design-review-010). The drift check below stays null-action.
        kicks = benchmark_kicks(world_name, world, region_name, band, seeds, n_starts, horizon)
        curves = separation_curves_batch(
            world.transition_batch, starts, horizon, world.scale, delta0, actions=kicks
        )
        regions_out[region_name] = {
            "steps": steps,
            "median_separation": median_separation_curve(curves).tolist(),
        }

        # Null actions: the conserved quantity is only conserved without
        # kicks (ADR-W1). Batched, bit-identical to one start at a time.
        drift_array, initial_array = conserved_drift_batch(
            world.transition_batch, world.conserved, starts, horizon, null_action=null_action
        )
        abs_drifts: list[float] = drift_array.tolist()
        initial_values: list[float] = initial_array.tolist()

        # The normaliser is this region's own invariant range — a
        # deterministic grid over its box, never pooled with another
        # region's starts (design-review-009 I1: pooling in a wider,
        # out-of-region span dilutes the figure for the region the
        # bound is meant to protect).
        range_min, range_max = conserved_quantity_range(world.conserved, box)
        span = range_max - range_min
        if not (np.isfinite(span) and span > 0.0):
            raise DegenerateInvariantRangeError(
                f"WD-3 drift gate: {world_name}/{region_name} conserved quantity has range "
                f"{span!r} over its declared box — 'relative drift' is undefined and the "
                f"1e-6 bound cannot be applied; the region's box or invariant is a spec "
                f"defect (worlds ADR-W1)"
            )
        max_abs_drift = max(abs_drifts)
        rel_drift_max = max_abs_drift / span
        # The literal "relative to the initial value" figure, reported
        # for transparency only. A start whose invariant is exactly 0.0
        # has no such ratio; it is skipped rather than emitted as inf,
        # which the canonical serializer rightly refuses (ADR-002 rule 4).
        ratios_vs_initial = [
            abs_drift / abs(initial)
            for abs_drift, initial in zip(abs_drifts, initial_values)
            if initial != 0.0
        ]
        # A zero invariant is a measure-zero event for the declared
        # boxes, but an empty max() would crash the benchmark on a
        # transparency-only figure; report NaN-free None instead.
        rel_vs_initial = max(ratios_vs_initial) if ratios_vs_initial else None

        # `within_bound` is derived from the same comparison the gate
        # makes, not written as a literal. In any artefact this function
        # RETURNS it is necessarily True — a False raises on the next
        # line and nothing is returned — but the field is kept so the
        # JSON is self-describing to a reader who has the file and not
        # this code (code-review-001 I4).
        within_bound = bool(rel_drift_max < drift_bound)
        drift_per_region.append(
            {
                "region": region_name,
                "conserved_rel_drift_max": rel_drift_max,
                "conserved_rel_drift_max_vs_initial": rel_vs_initial,
                "within_bound": within_bound,
            }
        )
        assert_drift_within_bound(rel_drift_max, drift_bound, f"{world_name}/{region_name}")

    return {
        "world": world_name,
        "perturbation": delta0,
        "distance": DISTANCE_NAME,
        "regions": regions_out,
        "drift": {
            "per_region": drift_per_region,
            "bound": drift_bound,
            "within_bound": all(r["within_bound"] for r in drift_per_region),
        },
        "seed": seeds.run_seed,
        "n_starts": n_starts,
    }
