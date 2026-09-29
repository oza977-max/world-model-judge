"""wmj.harness.kicks — seeded kick sequences for a named region and purpose.

In plain words: the worlds know *how* to draw a kick sequence
(`wmj.worlds.actions.kick_sequence`); the harness owns the seeds. This
helper joins the two: for rollout `i` of a given world, region and
purpose ("train-kicks", "eval-kicks" or "benchmark-kicks"), it draws that
rollout's kicks from its own named seed stream, at the world's pinned
kick rate and trained kick size. Every part of the pipeline that needs
kicks — training data, evaluation trials, the drift benchmark — comes
through here, so they cannot drift apart (worlds ADR-W2, cross-cutting
ADR-002 rule 2, design-review-010).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from wmj.errors import WmjError
from wmj.models.base import SeedSource
from wmj.worlds.actions import kick_sequence, step_probability

KICK_PURPOSES = ("train-kicks", "eval-kicks", "benchmark-kicks")


class KickPurposeError(WmjError):
    """Raised for a kick purpose outside the three pinned ones.

    A new purpose string would open a new seed stream no test knows
    about — the collision class cross-cutting ADR-002 rule 2 exists to
    prevent (TC-NF1-10).
    """


def seeded_kick_sequences(
    seeds: SeedSource,
    world_name: str,
    world: Any,
    region_name: str,
    band: str,
    purpose: str,
    n: int,
    horizon: int,
) -> np.ndarray:
    """`float64[n, horizon, 1]`: rollout `i` from `seeds.rng_for(world, region, purpose, str(i))`."""
    if purpose not in KICK_PURPOSES:
        raise KickPurposeError(
            f"kick purpose {purpose!r} is not one of {list(KICK_PURPOSES)} "
            f"(cross-cutting ADR-002 rule 2, TC-NF1-10)"
        )
    p_step = step_probability(world.kick_rate_per_s, world.dt)
    umax = float(world.regions().training_action_interval[0, 1])
    if n == 0:
        return np.zeros((0, horizon, 1))
    return np.stack(
        [
            kick_sequence(
                seeds.rng_for(world_name, region_name, purpose, str(i)),
                horizon=horizon,
                p_step=p_step,
                band=band,
                umax=umax,
            )
            for i in range(n)
        ]
    ).reshape(n, horizon, 1)
