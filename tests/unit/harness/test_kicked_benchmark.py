"""The benchmark under kicks: seeds, bands, and the null-action drift check.

In plain words: each benchmark start gets its own seeded kick sequence
(`"benchmark-kicks"`, indexed by the start), drawn from its region's band
— the trained band for the training and state-axis regions, the larger
out-of-range band for `out-large-action` — and the same sequence is
applied to the start and its twin. The energy/orbit leak check stays
kick-free, because the quantity is only conserved without kicks
(worlds ADR-W1/W3, cross-cutting ADR-002 rule 2, design-review-010).
"""

from __future__ import annotations

import numpy as np
import pytest

from wmj.harness.benchmarks import (
    benchmark_kicks,
    build_divergence_artefact,
    declared_regions,
    sample_region_starts,
)
from wmj.models.base import SeedSource
from wmj.worlds import lv, pendulum
from wmj.worlds.actions import kick_sequence, step_probability
from wmj.worlds.divergence import median_separation_curve, separation_curve

SEED = 20260825


def _seeds() -> SeedSource:
    return SeedSource(run_seed=SEED, my_name=None)


def test_declared_regions_carry_each_regions_kick_band():
    assert [(name, band) for name, _box, band in declared_regions(lv.WORLD)] == [
        ("training", "in"),
        ("out-high-amplitude", "in"),
        ("out-large-action", "out"),
    ]
    assert [band for _n, _b, band in declared_regions(pendulum.WORLD)] == ["in", "in", "out"]


def test_benchmark_kicks_use_one_seeded_stream_per_start_with_the_pinned_rule():
    horizon = 120
    kicks = benchmark_kicks("lv", lv.WORLD, "out-large-action", "out", _seeds(), 3, horizon)
    assert kicks.shape == (3, horizon, 1)
    p = step_probability(lv.KICK_RATE_PER_S, lv.DT)
    for i in range(3):
        rng = _seeds().rng_for("lv", "out-large-action", "benchmark-kicks", str(i))
        expected = kick_sequence(rng, horizon=horizon, p_step=p, band="out", umax=0.1)
        assert np.array_equal(kicks[i], expected)


def test_tc_nf1_10_kick_purposes_differ_from_each_other_and_from_start_purposes():
    # TC-NF1-10 (kick half; the training-subsample half lands at P3-C06).
    purposes = [
        ("training", "benchmark-kicks", "0"),
        ("training", "eval-kicks", "0"),
        ("training", "train-kicks", "0"),
        ("training", "benchmark-starts"),
        ("training", "eval-starts"),
        ("training", "train-starts"),
    ]
    first = [_seeds().rng_for("lv", *p).random(4) for p in purposes]
    for i in range(len(first)):
        for j in range(i + 1, len(first)):
            assert not np.array_equal(first[i], first[j]), (purposes[i], purposes[j])


def test_kick_streams_differ_between_starts_and_between_regions():
    a = _seeds().rng_for("lv", "training", "benchmark-kicks", "0").random(4)
    b = _seeds().rng_for("lv", "training", "benchmark-kicks", "1").random(4)
    c = _seeds().rng_for("lv", "out-high-amplitude", "benchmark-kicks", "0").random(4)
    assert not np.array_equal(a, b) and not np.array_equal(a, c)


@pytest.fixture(scope="module")
def lv_artefact() -> dict:
    return build_divergence_artefact("lv", lv.WORLD, _seeds(), n_starts=6, horizon=120)


def test_artefact_covers_all_three_regions(lv_artefact):
    assert set(lv_artefact["regions"]) == {"training", "out-high-amplitude", "out-large-action"}
    assert [r["region"] for r in lv_artefact["drift"]["per_region"]] == [
        "training",
        "out-high-amplitude",
        "out-large-action",
    ]


def test_artefact_curve_is_the_median_of_kicked_per_start_curves(lv_artefact):
    # Recompute one region the slow, obvious way and compare exactly.
    region, band, n, horizon = "out-large-action", "out", 6, 120
    box = lv.regions().training_state_box
    starts = sample_region_starts(_seeds().rng_for("lv", region, "benchmark-starts"), box, n)
    kicks = benchmark_kicks("lv", lv.WORLD, region, band, _seeds(), n, horizon)
    curves = np.stack(
        [separation_curve(lv.transition, starts[i], horizon, lv.SCALE, 1e-6, actions=kicks[i])
         for i in range(n)]
    )
    assert lv_artefact["regions"][region]["median_separation"] == \
        median_separation_curve(curves).tolist()


def test_drift_check_stays_null_action_even_though_curves_are_kicked(lv_artefact):
    # The conserved-quantity drift under kicks would be large (kicks
    # change the orbit); a within-bound figure proves it is measured
    # kick-free (ADR-W1, TC-WD3-03).
    for entry in lv_artefact["drift"]["per_region"]:
        assert entry["conserved_rel_drift_max"] < 1e-6
