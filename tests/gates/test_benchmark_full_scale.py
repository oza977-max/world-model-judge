"""Full-scale divergence benchmark gate — the spec's actual numbers.

worlds spec ADR-W3: 64 seeded starts per region, full JU-6 horizons
(LV 700, pendulum 5000), delta0 = 1e-6, null actions. ADR-W1: drift
within 1e-6. TC-WD3-03 / TC-WD4-01 / TC-WD4-02 at real scale.

Marked `slow` (~90 s): `pytest -m slow tests/gates/test_benchmark_full_scale.py`.
The unit suite exercises the same code at reduced scale.
"""

from __future__ import annotations

import pytest

from wmj.harness.benchmarks import build_divergence_artefact
from wmj.harness.serialize import canonical_serialize
from wmj.models.base import SeedSource
from wmj.worlds import lv, pendulum

SEED = 20260825


@pytest.mark.slow
def test_tc_wd3_03_wd4_01_lv_full_scale_benchmark():
    artefact = build_divergence_artefact("lv", lv.WORLD, SeedSource(SEED, None))
    assert artefact["n_starts"] == 64
    assert artefact["drift"]["within_bound"] is True
    for region in artefact["regions"].values():
        assert region["steps"] == list(range(lv.HORIZON + 1))
    training = artefact["regions"]["training"]["median_separation"]
    # TC-WD4-01, asserted two-sided at full scale exactly as at reduced
    # scale (code-review-001 I1): bounded/sub-exponential over the
    # declared horizon — executed, the ratio is flat/oscillating (see
    # build/prompts/P2-C03.md). The upper bound catches an exponential
    # runaway; the lower bound catches a broken integrator collapsing
    # the twin trajectories together.
    ratio = training[-1] / training[0]
    assert 0.1 < ratio < 10.0, f"TC-WD4-01 (full scale): ratio {ratio:.3g} outside (0.1, 10)"


@pytest.mark.slow
def test_tc_wd3_03_wd4_02_pendulum_full_scale_benchmark():
    artefact = build_divergence_artefact("pendulum", pendulum.WORLD, SeedSource(SEED, None))
    assert artefact["n_starts"] == 64
    assert artefact["drift"]["within_bound"] is True
    half = pendulum.HORIZON // 2
    training = artefact["regions"]["training"]["median_separation"]
    inverted = artefact["regions"]["out-near-inverted"]["median_separation"]
    # TC-WD4-02 / ADR-W3: >= 5x separation at half-horizon between regimes
    assert inverted[half] / training[half] >= 5.0


@pytest.mark.slow
def test_full_scale_lv_artefact_is_seed_deterministic():
    a = build_divergence_artefact("lv", lv.WORLD, SeedSource(SEED, None))
    b = build_divergence_artefact("lv", lv.WORLD, SeedSource(SEED, None))
    assert canonical_serialize(a) == canonical_serialize(b)
