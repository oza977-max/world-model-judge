"""TC-WD2-02 — kicks never breach the floor, at full scale, in every region.

In plain words: Round 10 measured that the originally proposed
predator–prey kicks (up to 0.5, about once a second) crash the prey
population through its floor on full-length runs — 7 of 2,000 training
runs, 477 of 2,000 high-amplitude runs. The pinned kicks (up to 0.1, about
once every two seconds) crashed none. Because every kick sequence is
seeded, the real workloads either always pass or always fail; this gate
runs them at their real size with the project's seed and fails loudly if
any step would cross the floor (or, for the pendulum, stop being finite).
A failure is a spec bug to fix openly before the freeze — never a run to
filter out (worlds §7, ADR-W2, design-review-010).

Workloads, per world, using the pinned seed purposes (cross-cutting
ADR-002 rule 2): 2,000 full-horizon training trajectories
("train-starts" / "train-kicks"), and for every declared region 200
evaluation trials ("eval-starts" / "eval-kicks") and 64 benchmark starts
with their perturbed twins ("benchmark-starts" / "benchmark-kicks").

**Note for P3-C06:** the training half below re-creates the generation
rule from the spec. When P3-C06 builds the real training-data generator,
this gate must call it instead, so the gate checks the code that runs
rather than a copy of it.
"""

from __future__ import annotations

import numpy as np
import pytest

from wmj.harness.benchmarks import (
    benchmark_kicks,
    declared_regions,
    sample_region_starts,
)
from wmj.models.base import SeedSource
from wmj.worlds import lv, pendulum
from wmj.worlds.actions import kick_sequence, step_probability
from wmj.worlds.divergence import perturb

SEED = 20260825
N_TRAIN = 2000
N_EVAL = 200
N_BENCH = 64

WORLDS = [("lv", lv), ("pendulum", pendulum)]


def _roll(module, starts: np.ndarray, kicks: np.ndarray) -> None:
    """Advance every row for the full horizon; transition_batch raises on
    any floor breach (worlds §7). The pendulum has no floor, so finiteness
    is asserted instead."""
    states = starts.copy()
    for t in range(kicks.shape[1]):
        states = module.transition_batch(states, kicks[:, t, :])
    assert np.all(np.isfinite(states))


def _kicks(seeds: SeedSource, world: str, region: str, purpose: str, n: int, module, band: str):
    p = step_probability(module.KICK_RATE_PER_S, module.DT)
    umax = float(module.regions().training_action_interval[0, 1])
    return np.stack(
        [
            kick_sequence(seeds.rng_for(world, region, purpose, str(i)),
                          horizon=module.HORIZON, p_step=p, band=band, umax=umax)
            for i in range(n)
        ]
    )


@pytest.mark.parametrize(("name", "module"), WORLDS)
def test_tc_wd2_02_training_trajectories_never_breach_the_floor(name, module):
    seeds = SeedSource(SEED, None)
    box = module.regions().training_state_box
    starts = sample_region_starts(seeds.rng_for(name, "training", "train-starts"), box, N_TRAIN)
    kicks = _kicks(seeds, name, "training", "train-kicks", N_TRAIN, module, "in")
    _roll(module, starts, kicks)


@pytest.mark.parametrize(("name", "module"), WORLDS)
def test_tc_wd2_02_evaluation_trials_never_breach_the_floor_in_any_region(name, module):
    seeds = SeedSource(SEED, None)
    for region, box, band in declared_regions(module.WORLD):
        starts = sample_region_starts(seeds.rng_for(name, region, "eval-starts"), box, N_EVAL)
        kicks = _kicks(seeds, name, region, "eval-kicks", N_EVAL, module, band)
        _roll(module, starts, kicks)


@pytest.mark.parametrize(("name", "module"), WORLDS)
def test_tc_wd2_02_benchmark_starts_and_twins_never_breach_the_floor(name, module):
    seeds = SeedSource(SEED, None)
    for region, box, band in declared_regions(module.WORLD):
        starts = sample_region_starts(seeds.rng_for(name, region, "benchmark-starts"), box, N_BENCH)
        kicks = benchmark_kicks(name, module.WORLD, region, band, seeds, N_BENCH, module.HORIZON)
        twins = np.stack([perturb(s, 1e-6) for s in starts])
        _roll(module, np.concatenate([starts, twins]), np.concatenate([kicks, kicks]))
