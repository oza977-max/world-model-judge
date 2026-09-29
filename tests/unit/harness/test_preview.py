"""Tests for wmj.harness.preview — the scoped stand-in producer for Chart 2.

The real `error_vs_horizon` block is the judge's (JU-3, P4-C05). Until
it exists, the harness computes a stand-in with exactly the judge's
block shape (judge §5) so the renderer never changes when the producer
swaps. Reduced scale here; the CLI's defaults are the spec's numbers.
"""

from __future__ import annotations

import numpy as np

from wmj.harness.preview import (
    build_lv_persistence_error_vs_horizon,
    one_step_skill_persistence_vs_linear,
)
from wmj.harness.serialize import canonical_serialize
from wmj.models.base import SeedSource
from wmj.worlds import lv

SEED = 20260825


def _seeds() -> SeedSource:
    return SeedSource(run_seed=SEED, my_name=None)


def test_block_has_exactly_the_judge_shape():
    block = build_lv_persistence_error_vs_horizon(_seeds(), n_starts=4, n_trials=5, horizon=30)
    assert set(block) == {"dt", "per_region"}
    assert block["dt"] == lv.DT
    names = [entry["region"] for entry in block["per_region"]]
    assert names == ["training", *(r.region_name for r in lv.regions().out_regions)]
    for entry in block["per_region"]:
        assert set(entry) == {"region", "steps", "median_error", "divergence_reference"}
        assert entry["steps"] == list(range(31))
        assert len(entry["median_error"]) == 31
        assert len(entry["divergence_reference"]) == 31


def test_step_zero_error_is_zero_and_later_errors_positive():
    block = build_lv_persistence_error_vs_horizon(_seeds(), n_starts=4, n_trials=5, horizon=30)
    for entry in block["per_region"]:
        assert entry["median_error"][0] == 0.0
        assert all(e > 0.0 for e in entry["median_error"][1:])


def test_persistence_error_is_the_distance_from_the_held_start_under_the_trials_own_kicks():
    """With one trial the median IS that trial: recompute it independently.

    The truth is rolled out under that trial's own seeded kicks — purpose
    "eval-kicks", the same kind the divergence reference uses
    (design-review-010). The horizon is long enough that the trial is
    actually kicked, so reverting to kick-free truth, or drawing from the
    wrong purpose, fails here (independent review, P3-C09 pass 1).
    """
    from wmj.harness.benchmarks import sample_region_starts
    from wmj.worlds.actions import kick_sequence, step_probability
    from wmj.worlds.base import distance

    horizon = 400
    start = sample_region_starts(
        _seeds().rng_for("lv", "training", "eval-starts"), lv.regions().training_state_box, 1
    )[0]
    kicks = kick_sequence(
        _seeds().rng_for("lv", "training", "eval-kicks", "0"),
        horizon=horizon,
        p_step=step_probability(lv.KICK_RATE_PER_S, lv.DT),
        band="in",
        umax=0.1,
    )
    assert np.any(kicks != 0.0), "choose a horizon that includes at least one kick"
    state, expected = start, [0.0]
    for t in range(horizon):
        state = lv.transition(state, kicks[t])
        expected.append(distance(state, start, lv.SCALE))
    block = build_lv_persistence_error_vs_horizon(_seeds(), n_starts=4, n_trials=1, horizon=horizon)
    assert block["per_region"][0]["median_error"] == expected


def test_block_is_canonically_serializable_and_seed_deterministic():
    a = build_lv_persistence_error_vs_horizon(_seeds(), n_starts=4, n_trials=5, horizon=20)
    b = build_lv_persistence_error_vs_horizon(_seeds(), n_starts=4, n_trials=5, horizon=20)
    assert canonical_serialize(a) == canonical_serialize(b)


def test_one_step_skill_runs_over_real_trajectories():
    result = one_step_skill_persistence_vs_linear(_seeds(), n_trials=12)
    assert result.crps_persistence > 0.0 and result.crps_linear > 0.0
    assert result.skill == 1.0 - result.crps_persistence / result.crps_linear


def test_chart_preview_refuses_non_positive_sizes_loudly():
    """code-review-001 I6: a bare CLI integer of 0 used to reach a
    division and surface as a contextless ZeroDivisionError; every
    refusal now names what failed (cross-cutting Error-Handling rule 2)."""
    import pytest

    from wmj.harness.preview import PreviewArgumentError

    with pytest.raises(PreviewArgumentError):
        build_lv_persistence_error_vs_horizon(_seeds(), n_starts=4, n_trials=0, horizon=10)
    with pytest.raises(PreviewArgumentError):
        build_lv_persistence_error_vs_horizon(_seeds(), n_starts=0, n_trials=2, horizon=10)
    with pytest.raises(PreviewArgumentError):
        build_lv_persistence_error_vs_horizon(_seeds(), n_starts=2, n_trials=2, horizon=0)
    with pytest.raises(PreviewArgumentError):
        one_step_skill_persistence_vs_linear(_seeds(), n_trials=0)
