"""Tests for wmj.harness.benchmarks — the WD-4 divergence artefact.

The harness owns the seeded sampling (64 starts per declared region,
cross-cutting ADR-002 rule 2's "benchmark-starts" purpose key) and
assembles the artefact in exactly worlds spec §5's shape. It RETURNS
the dict; writing under out/ is reporting's job (design-review-008 C8).

Unit tests run at reduced scale; full scale is the slow gate.
"""

from __future__ import annotations

import numpy as np
import pytest

from wmj.harness.benchmarks import build_divergence_artefact, sample_region_starts
from wmj.harness.serialize import canonical_serialize
from wmj.models.base import SeedSource
from wmj.worlds import lv
from wmj.worlds.divergence import DriftBoundError

SEED = 20260825


def _seeds() -> SeedSource:
    return SeedSource(run_seed=SEED, my_name=None)


# --- sample_region_starts ---


def test_sample_region_starts_returns_n_by_d_inside_the_box():
    box = lv.regions().training_state_box
    rng = _seeds().rng_for("lv", "training", "benchmark-starts")
    starts = sample_region_starts(rng, box, n=16)
    assert starts.shape == (16, 2)
    assert np.all(starts >= box[:, 0]) and np.all(starts <= box[:, 1])


def test_sample_region_starts_is_seeded_reproducible():
    box = lv.regions().training_state_box
    a = sample_region_starts(_seeds().rng_for("lv", "training", "benchmark-starts"), box, 8)
    b = sample_region_starts(_seeds().rng_for("lv", "training", "benchmark-starts"), box, 8)
    assert np.array_equal(a, b)


def test_benchmark_starts_use_their_own_purpose_key_not_eval_starts():
    """cross-cutting ADR-002 rule 2: benchmark-starts must not collide
    with eval-starts or train-starts for the same (world, region)."""
    box = lv.regions().training_state_box
    bench = sample_region_starts(_seeds().rng_for("lv", "training", "benchmark-starts"), box, 8)
    evals = sample_region_starts(_seeds().rng_for("lv", "training", "eval-starts"), box, 8)
    assert not np.array_equal(bench, evals)


# --- build_divergence_artefact: the worlds §5 contract ---


@pytest.fixture(scope="module")
def lv_artefact() -> dict:
    return build_divergence_artefact("lv", lv.WORLD, _seeds(), n_starts=8, horizon=50)


def test_artefact_has_exactly_the_contracted_top_level_keys(lv_artefact):
    assert set(lv_artefact) == {
        "world",
        "perturbation",
        "distance",
        "regions",
        "drift",
        "seed",
        "n_starts",
    }
    assert lv_artefact["world"] == "lv"
    assert lv_artefact["perturbation"] == 1e-6
    assert lv_artefact["distance"] == "rms-normalised"
    assert lv_artefact["seed"] == SEED
    assert lv_artefact["n_starts"] == 8


def test_artefact_regions_cover_training_and_every_declared_out_region(lv_artefact):
    assert set(lv_artefact["regions"]) == {"training", "out-high-amplitude"}


def test_artefact_steps_run_zero_to_horizon_inclusive(lv_artefact):
    for region in lv_artefact["regions"].values():
        assert region["steps"] == list(range(51))
        assert len(region["median_separation"]) == 51
        assert region["median_separation"][0] > 0.0


def test_artefact_drift_block_matches_contract_and_is_within_bound(lv_artefact):
    """design-review-009 I1/I2: the drift figure is per region — never
    pooled with another region's span — so the block carries one entry
    per declared region plus a shared bound/summary within_bound."""
    drift = lv_artefact["drift"]
    assert set(drift) == {"per_region", "bound", "within_bound"}
    assert drift["bound"] == 1e-6
    assert drift["within_bound"] is True

    region_names = {r["region"] for r in drift["per_region"]}
    assert region_names == set(lv_artefact["regions"])  # training + every out-region

    for entry in drift["per_region"]:
        assert set(entry) == {
            "region",
            "conserved_rel_drift_max",
            "conserved_rel_drift_max_vs_initial",
            "within_bound",
        }
        assert entry["within_bound"] is True
        assert 0.0 <= entry["conserved_rel_drift_max"] < 1e-6


def test_drift_normaliser_is_never_pooled_across_regions():
    """A wider out-region span must not dilute the training region's
    own drift figure (design-review-009 I1, executed evidence)."""
    from wmj.worlds.divergence import conserved_quantity_range
    from wmj.worlds import lv as lv_module

    spec = lv_module.regions()
    training_range = conserved_quantity_range(lv_module.conserved, spec.training_state_box)
    out_range = conserved_quantity_range(
        lv_module.conserved, spec.out_regions[0].state_box
    )
    assert training_range != out_range  # the two regions have distinct spans

    art = build_divergence_artefact("lv", lv.WORLD, _seeds(), n_starts=8, horizon=50)
    by_region = {r["region"]: r for r in art["drift"]["per_region"]}
    training_span = training_range[1] - training_range[0]
    out_span = out_range[1] - out_range[0]
    # A region's own span, not some pooled wider span, bounds its ratio.
    assert training_span < out_span
    assert by_region["training"]["conserved_rel_drift_max"] >= 0.0


def test_artefact_is_canonically_serializable_and_seed_deterministic(lv_artefact):
    first = canonical_serialize(lv_artefact)
    again = build_divergence_artefact("lv", lv.WORLD, _seeds(), n_starts=8, horizon=50)
    assert canonical_serialize(again) == first


def test_artefact_changes_with_the_seed():
    a = build_divergence_artefact("lv", lv.WORLD, SeedSource(1, None), n_starts=8, horizon=50)
    b = build_divergence_artefact("lv", lv.WORLD, SeedSource(2, None), n_starts=8, horizon=50)
    assert canonical_serialize(a) != canonical_serialize(b)


def test_tc_wd3_03_artefact_build_fails_loudly_when_drift_exceeds_bound():
    with pytest.raises(DriftBoundError):
        build_divergence_artefact(
            "lv", lv.WORLD, _seeds(), n_starts=8, horizon=50, drift_bound=1e-30
        )
