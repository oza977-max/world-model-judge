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

The training half calls the real training-data generator (P3-C06), so the gate
checks the code that runs rather than a copy of it.
"""

from __future__ import annotations

import numpy as np
import pytest

from wmj.harness.benchmarks import (
    benchmark_kicks,
    declared_regions,
    sample_region_starts,
)
from wmj.harness.kicks import seeded_kick_sequences
from wmj.models.base import SeedSource
from wmj.worlds import lv, pendulum
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


@pytest.mark.parametrize(("name", "module"), WORLDS)
def test_tc_wd2_02_training_trajectories_never_breach_the_floor(name, module):
    # Calls the real training-data generator (P3-C06), not a copy of its
    # rule: `transition_batch` aborts on any floor breach, so a successful
    # build is the proof (and finiteness is asserted for the pendulum).
    from pathlib import Path

    from wmj.harness.training import build_training_data, read_training_recipe

    recipe = read_training_recipe(Path(__file__).resolve().parents[2] / "prereg" / "recipe.md")
    assert recipe.training_trajectories == N_TRAIN
    data = build_training_data(name, module.WORLD, SeedSource(SEED, None), recipe)
    assert np.all(np.isfinite(data.states))


@pytest.mark.parametrize(("name", "module"), WORLDS)
def test_tc_wd2_02_evaluation_trials_never_breach_the_floor_in_any_region(name, module):
    # Uses the harness's own kick helper (not a copy), so this checks the
    # code the pipeline runs (independent review, P3-C09 pass 1).
    seeds = SeedSource(SEED, None)
    for region, box, band in declared_regions(module.WORLD):
        starts = sample_region_starts(seeds.rng_for(name, region, "eval-starts"), box, N_EVAL)
        kicks = seeded_kick_sequences(
            seeds, name, module.WORLD, region, band, "eval-kicks", N_EVAL, module.HORIZON
        )
        _roll(module, starts, kicks)


@pytest.mark.parametrize(("name", "module"), WORLDS)
def test_every_out_large_action_trial_is_kicked_at_least_once(name, module):
    # A trial in the action-axis region with no kick at all would be
    # labelled fully in-region (axis None) despite its region name. At the
    # pinned rates that is rare (LV ~0.09% per trial), and with the pinned
    # seed it does not happen; this asserts it rather than assuming it
    # (independent review, P3-C09 pass 1; backlog A17).
    seeds = SeedSource(SEED, None)
    kicks = seeded_kick_sequences(
        seeds, name, module.WORLD, "out-large-action", "out", "eval-kicks", N_EVAL, module.HORIZON
    )
    per_trial = np.count_nonzero(kicks[:, :, 0], axis=1)
    assert per_trial.min() >= 1, f"{name}: trials {np.flatnonzero(per_trial == 0)} carry no kick"


@pytest.mark.parametrize(("name", "module"), WORLDS)
def test_tc_wd2_02_benchmark_starts_and_twins_never_breach_the_floor(name, module):
    seeds = SeedSource(SEED, None)
    for region, box, band in declared_regions(module.WORLD):
        starts = sample_region_starts(seeds.rng_for(name, region, "benchmark-starts"), box, N_BENCH)
        kicks = benchmark_kicks(name, module.WORLD, region, band, seeds, N_BENCH, module.HORIZON)
        twins = np.stack([perturb(s, 1e-6) for s in starts])
        _roll(module, np.concatenate([starts, twins]), np.concatenate([kicks, kicks]))
