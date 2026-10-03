"""wmj.harness.sufficiency — "is 50,000 training examples enough?", asked once, blind.

In plain words: the recipe trains each practice model on 50,000 examples. Is
that enough, or would 100,000 make it noticeably better? This is checked
**once, at build time, before the lock**, by training each model twice — on
50,000 and on 100,000 examples — and comparing the error on examples neither
version trained on. If the 50,000-example model is within 10% of the
100,000-example model for **both** practice models, 50,000 stands; otherwise
the recipe becomes 100,000 for both — the only fallback, with no further
iteration (models ADR-M3, design-review-010).

**What it must never do.** It must never look at the comparison that the
whole experiment is about — how close the two practice models are to each
other (the "matching margin") — nor at evaluation trials or skill scores.
Choosing the amount of data by peeking at that comparison would tune the
setup against the very result it is meant to leave open. So nothing here
accepts or reads any of those: the scoring functions take held-out examples
and predictions only, and `check_world` takes only the world, a model factory,
the recipe counts and a seed (TC-MU5-05).

**Which held-out set.** Both versions of a model are scored on the held-out
set of the *larger* (100,000-example) build. That set is disjoint from both
training sets (the 50,000 set is the first part of the 100,000 set); scoring
the larger model on the smaller build's held-out set would let it see some of
those examples in training and tilt the test toward "50,000 is not enough"
(found in the independent review of the training-data chunk; recorded in
backlog A21 as a clarification of "each model's own held-out pairs").

**Kick split.** The same step reports each model's held-out error separately
for kicked and un-kicked examples, because training is 25% kicks while the
evaluation steps are about 1% (predator–prey) and 0.2% (pendulum) kicks. The
number of kicked held-out examples is reported with it: it is small (backlog
A19 / D17), and a kick-split figure resting on a handful of examples must not
be read as more than that.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, Callable

import numpy as np

from wmj.errors import WmjError
from wmj.harness.training import TrainingRecipe, build_training_data, make_world_context
from wmj.models.base import Pairs, SeedSource, TrainingData

SUFFICIENCY_TOLERANCE = 0.10  # prereg/recipe.md `sufficiency_tolerance`


class SufficiencyError(WmjError):
    """Raised when the sufficiency check is given something it cannot judge."""


def held_out_error(means: np.ndarray, pairs: Pairs, scale: np.ndarray) -> float:
    """Mean over examples of mean over dimensions of `((mean − y)/scale)²` (ADR-M3)."""
    means = np.asarray(means, dtype=float)
    if means.shape != pairs.next_state.shape:
        raise SufficiencyError(
            f"predicted means have shape {tuple(means.shape)} but the held-out targets have "
            f"{tuple(pairs.next_state.shape)}"
        )
    if pairs.state.shape[0] == 0:
        raise SufficiencyError("the held-out set is empty")
    error = float(np.mean(((means - pairs.next_state) / scale) ** 2))
    if not math.isfinite(error):
        raise SufficiencyError("the held-out error is not finite (the model's predictions are)")
    return error


@dataclass(frozen=True)
class KickSplit:
    """Held-out error separately for kicked and un-kicked examples, with the counts."""

    n_kick: int
    n_plain: int
    error_kick: float | None  # None when there are no kicked held-out examples
    error_plain: float | None


def kick_split_error(means: np.ndarray, pairs: Pairs, scale: np.ndarray) -> KickSplit:
    means = np.asarray(means, dtype=float)
    if means.shape != pairs.next_state.shape:
        raise SufficiencyError(
            f"predicted means have shape {tuple(means.shape)} but the held-out targets have "
            f"{tuple(pairs.next_state.shape)}"
        )
    per_example = np.mean(((means - pairs.next_state) / scale) ** 2, axis=1)
    kick = pairs.is_kick
    return KickSplit(
        n_kick=int(kick.sum()),
        n_plain=int((~kick).sum()),
        error_kick=float(per_example[kick].mean()) if kick.any() else None,
        error_plain=float(per_example[~kick].mean()) if (~kick).any() else None,
    )


def m_is_sufficient(err_m: float, err_2m: float, tolerance: float = SUFFICIENCY_TOLERANCE) -> bool:
    """`err(M) ≤ (1 + tolerance) × err(2M)` (ADR-M3: tolerance 0.10 ⇒ 1.10×)."""
    for name, value in (("err(M)", err_m), ("err(2M)", err_2m), ("tolerance", tolerance)):
        if not (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
            and value >= 0.0
        ):
            raise SufficiencyError(f"{name} must be a finite non-negative number, got {value!r}")
    return bool(err_m <= (1.0 + tolerance) * err_2m)


def decide_subsample_pairs(recipe_m: int, fallback: int, sufficient: list[bool]) -> int:
    """`recipe_m` if every model says M is enough, else `fallback` — nothing else, ever."""
    if not sufficient:
        raise SufficiencyError("no models were checked — the decision needs every unrigged model")
    return recipe_m if all(sufficient) else fallback


@dataclass(frozen=True)
class SufficiencyResult:
    world: str
    err_m: float
    err_2m: float
    sufficient: bool
    split_m: KickSplit
    split_2m: KickSplit


Factory = Callable[[Any, SeedSource, TrainingData], Any]


def check_world(
    world_name: str,
    world: Any,
    factory: Factory,
    recipe: TrainingRecipe,
    run_seed: int,
    model_name: str,
    *,
    tolerance: float = SUFFICIENCY_TOLERANCE,
    horizon: int | None = None,
) -> SufficiencyResult:
    """Train one model at M and at 2M and compare held-out error (models ADR-M3).

    The 2M build is the same recipe with `subsample_pairs` doubled (the same
    kick pairs plus more non-kick pairs of the same permutation — the M set is
    an exact prefix). Both models are scored on the 2M build's held-out set.
    Inputs are the world, the factory, the recipe counts and the seed; no
    evaluation trials, skill scores or matching margin exist in this
    function's world (TC-MU5-05).
    """
    ctx = make_world_context(world_name, world)
    data_seeds = SeedSource(run_seed, None)
    big_recipe = replace(recipe, subsample_pairs=2 * recipe.subsample_pairs)
    data_big = build_training_data(world_name, world, data_seeds, big_recipe, horizon=horizon)
    data_small = build_training_data(world_name, world, data_seeds, recipe, horizon=horizon)
    heldout = data_big.heldout_pairs
    scale = np.asarray(world.scale, dtype=float)

    model_small = factory(ctx, SeedSource(run_seed, model_name), data_small)
    means_small, _ = model_small.predict_batch(heldout.state, heldout.action)
    model_big = factory(ctx, SeedSource(run_seed, model_name), data_big)
    means_big, _ = model_big.predict_batch(heldout.state, heldout.action)

    err_m = held_out_error(means_small, heldout, scale)
    err_2m = held_out_error(means_big, heldout, scale)
    return SufficiencyResult(
        world=world_name,
        err_m=err_m,
        err_2m=err_2m,
        sufficient=m_is_sufficient(err_m, err_2m, tolerance),
        split_m=kick_split_error(means_small, heldout, scale),
        split_2m=kick_split_error(means_big, heldout, scale),
    )
