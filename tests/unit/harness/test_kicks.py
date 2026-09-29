"""Tests for wmj.harness.kicks — seeded kicks per world, region and purpose.

In plain words: every part of the pipeline gets its kicks through one
helper. These tests call it directly for each of the three purposes and
check it draws each rollout from its own named seed stream, at the
world's own kick rate and trained kick size, and refuses a purpose it
does not know (cross-cutting ADR-002 rule 2, worlds ADR-W2).
"""

from __future__ import annotations

import numpy as np
import pytest

from wmj.harness.kicks import KICK_PURPOSES, KickPurposeError, seeded_kick_sequences
from wmj.models.base import SeedSource
from wmj.worlds import lv, pendulum
from wmj.worlds.actions import kick_sequence, step_probability

SEEDS = SeedSource(run_seed=20260825, my_name=None)


@pytest.mark.parametrize("purpose", KICK_PURPOSES)
@pytest.mark.parametrize(("name", "module", "band"), [("lv", lv, "in"), ("pendulum", pendulum, "out")])
def test_each_rollout_draws_from_its_own_named_stream(purpose, name, module, band):
    kicks = seeded_kick_sequences(SEEDS, name, module.WORLD, "training", band, purpose, 3, 50)
    assert kicks.shape == (3, 50, 1)
    p = step_probability(module.KICK_RATE_PER_S, module.DT)
    umax = float(module.regions().training_action_interval[0, 1])
    for i in range(3):
        expected = kick_sequence(
            SEEDS.rng_for(name, "training", purpose, str(i)), horizon=50, p_step=p, band=band, umax=umax
        )
        assert np.array_equal(kicks[i], expected)


def test_the_three_purposes_give_different_kicks():
    a, b, c = (
        seeded_kick_sequences(SEEDS, "lv", lv.WORLD, "training", "in", p, 2, 2000)
        for p in KICK_PURPOSES
    )
    assert not np.array_equal(a, b) and not np.array_equal(b, c) and not np.array_equal(a, c)


def test_an_unknown_purpose_is_refused():
    with pytest.raises(KickPurposeError, match="ADR-002"):
        seeded_kick_sequences(SEEDS, "lv", lv.WORLD, "training", "in", "train-starts", 1, 10)


def test_zero_rollouts_give_an_empty_array_of_the_right_shape():
    assert seeded_kick_sequences(SEEDS, "lv", lv.WORLD, "training", "in", "eval-kicks", 0, 10).shape == (0, 10, 1)


@pytest.mark.parametrize(("n", "horizon"), [(-1, 10), (2, -1), (True, 10), (0, -5)])
def test_impossible_counts_or_horizons_are_refused(n, horizon):
    with pytest.raises(KickPurposeError, match="ADR-W2"):
        seeded_kick_sequences(SEEDS, "lv", lv.WORLD, "training", "in", "eval-kicks", n, horizon)
