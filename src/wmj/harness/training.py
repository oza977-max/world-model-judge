"""wmj.harness.training — the homework every model is handed.

In plain words: before any practice model can learn, the harness makes the
homework once, for each world, and gives every model *exactly the same copy*.
It simulates 2,000 histories of the world with the occasional kick pushed in,
then makes one fixed, seeded pick from all the one-step examples in them:
a training set (50,000 examples, exactly 12,500 of them kicked ones, so the
models see the lever often enough to learn it), a held-back set that is
never trained on (10,000 examples, kept for checking), and 64 training rows
used once to sanity-check the learning arithmetic. No model picks its own
examples — that is what keeps the five ensemble members on identical
homework and each fixture a bit-identical copy of `direct`. If the world
cannot supply enough kicked (or un-kicked) examples the build refuses; it
never quietly shrinks the number (models spec ADR-M1 and §4,
design-review-010).

Every count comes from `prereg/recipe.md`'s `key: value` lines, so the
numbers that were frozen are the numbers used. Every random choice has its
own named stream (`train-starts`, `train-kicks`, `subsample-kick`,
`subsample-nonkick`, `heldout`, `gradcheck-batch`), so changing one never
shifts another and the same seed always gives the same bytes
(cross-cutting ADR-002 rule 2, TC-MU8-01, TC-NF1-10).

Layout contract (relied on by the build-time sufficiency test, ADR-M3):
`train_pairs` is the kick pairs first, in permutation order, then the
non-kick pairs in permutation order. A set of size `2M` therefore begins
with the whole of the set of size `M`.

Evaluation never starts from a training start: the two draw from separate
seed purposes, and `assert_eval_starts_disjoint` checks it at run time too
(MU-7, TC-MU7-01).
"""

from __future__ import annotations

import numbers
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from wmj.errors import WmjError
from wmj.harness.benchmarks import sample_region_starts
from wmj.harness.kicks import seeded_kick_sequences
from wmj.models.base import Pairs, SeedSource, TrainingData

TRAINING_REGION = "training"
TRAINING_PURPOSES = (
    "train-starts",
    "train-kicks",
    "subsample-kick",
    "subsample-nonkick",
    "heldout",
    "gradcheck-batch",
)

_RECIPE_KEYS = (
    "training_trajectories",
    "subsample_pairs",
    "kick_pairs",
    "heldout_pairs",
    "gradcheck_pairs",
)
_PLAIN_POSITIVE_INT = re.compile(r"[1-9][0-9]*")
_RECIPE_LINE = re.compile(r"[ \t]*([^#\s]*)[ \t]*(?:#.*)?")


class TrainingDataError(WmjError):
    """Raised when the training data cannot be built exactly as pinned.

    In plain words: the build stops, naming the key or count at fault,
    rather than shrinking a number or using a different recipe (models
    spec §4; Code Complete ch. 8 — fail loudly, say which rule).
    """


@dataclass(frozen=True)
class TrainingRecipe:
    """The five counts the harness reads from `prereg/recipe.md`."""

    training_trajectories: int
    subsample_pairs: int
    kick_pairs: int
    heldout_pairs: int
    gradcheck_pairs: int

    def __post_init__(self) -> None:
        for key in _RECIPE_KEYS:
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < 1:
                raise TrainingDataError(f"recipe key {key!r} must be a positive integer, got {value!r}")
        if self.kick_pairs > self.subsample_pairs:
            raise TrainingDataError(
                f"kick_pairs ({self.kick_pairs}) cannot exceed subsample_pairs "
                f"({self.subsample_pairs}) — the training set is made of the kick pairs plus the rest"
            )
        if self.gradcheck_pairs > self.subsample_pairs:
            raise TrainingDataError(
                f"gradcheck_pairs ({self.gradcheck_pairs}) cannot exceed subsample_pairs "
                f"({self.subsample_pairs}) — the check batch is drawn from the training set"
            )


def read_training_recipe(recipe_path: str | Path) -> TrainingRecipe:
    """Read the five pinned counts from `prereg/recipe.md`.

    In plain words: the numbers that were locked in the recipe are the numbers
    the training data is built from — read here, never typed in code.

    Each key must appear exactly once, at the start of a line, with a plain
    positive integer (an indented or bullet-quoted mention in the prose is
    not a key); a trailing `# comment` is ignored. Anything else is refused,
    naming the key.
    """
    path = Path(recipe_path)
    try:
        text = path.read_bytes().decode("utf-8")
    except FileNotFoundError as exc:
        raise TrainingDataError(f"the recipe {path} does not exist — the counts live there") from exc
    except UnicodeDecodeError as exc:
        raise TrainingDataError(f"the recipe {path} is not valid UTF-8 text") from exc
    except OSError as exc:
        raise TrainingDataError(f"the recipe {path} cannot be read: {exc}") from exc
    values: dict[str, int] = {}
    for key in _RECIPE_KEYS:
        lines = [
            line.rstrip("\r")
            for line in re.findall(rf"^{re.escape(key)}:.*$", text, re.MULTILINE)
        ]
        if not lines:
            raise TrainingDataError(f"the recipe has no '{key}:' line at the left margin ({path})")
        if len(lines) > 1:
            raise TrainingDataError(
                f"the recipe pins '{key}' more than once ({lines}) — a later line could "
                f"disagree with the one used"
            )
        match = _RECIPE_LINE.fullmatch(lines[0][len(key) + 1 :])
        if match is None or not _PLAIN_POSITIVE_INT.fullmatch(match.group(1)):
            raise TrainingDataError(
                f"the recipe's line {lines[0]!r} is not '{key}: <positive plain integer>' "
                f"(optionally followed by a '# comment')"
            )
        values[key] = int(match.group(1))
    return TrainingRecipe(**values)


def _default_horizon(world: Any) -> int:
    tasks = world.tasks()
    if not tasks:
        raise TrainingDataError("the world declares no tasks, so it has no default horizon")
    return max(int(task.horizon) for task in tasks)


def _gather_pairs(
    states: np.ndarray, actions: np.ndarray, flat_index: np.ndarray, horizon: int
) -> Pairs:
    """Rows `(i, t)` for flat index `i * horizon + t`, copied out of the trajectories."""
    traj, step = np.divmod(flat_index, horizon)
    action = actions[traj, step]
    return Pairs(
        state=states[traj, step],
        action=action,
        next_state=states[traj, step + 1],
        is_kick=np.any(action != 0.0, axis=1),
    )


def build_training_data(
    world_name: str,
    world: Any,
    seeds: SeedSource,
    recipe: TrainingRecipe,
    *,
    horizon: int | None = None,
) -> TrainingData:
    """Make one world's homework (models spec ADR-M1, §4).

    In plain words: simulate the world's histories with occasional kicks, then
    pick, once and by seeded shuffle, the training examples (with the pinned
    number of kicked ones), the held-back examples and the few used to check
    the learning arithmetic. Consumers must shuffle the training rows
    themselves each epoch (the kicked ones come first), with their own stream.

    `horizon` defaults to the world's declared evaluation horizon (the
    longest task horizon); tests pass a shorter one to run quickly.
    `world_name` must be the name of `world` — it keys every seed stream, and
    nothing can check it against the world object (the harness owns the pairing).
    """
    if horizon is None:
        horizon = _default_horizon(world)
    if isinstance(horizon, bool) or not isinstance(horizon, numbers.Integral) or horizon < 1:
        raise TrainingDataError(f"horizon must be a positive int of steps, got {horizon!r}")
    horizon = int(horizon)
    if world.a != 1:
        raise TrainingDataError(
            f"{world_name!r} has {world.a} action dimensions — the kick generator draws one "
            f"(worlds ADR-W2); training data for a wider lever is not defined"
        )

    n = recipe.training_trajectories
    box = world.regions().training_state_box
    starts = sample_region_starts(seeds.rng_for(world_name, TRAINING_REGION, "train-starts"), box, n)
    kicks = seeded_kick_sequences(
        seeds, world_name, world, TRAINING_REGION, "in", "train-kicks", n, horizon
    )
    states = np.zeros((n, horizon + 1, world.d))
    states[:, 0] = starts
    for step in range(horizon):
        states[:, step + 1] = world.transition_batch(states[:, step], kicks[:, step])
    actions = kicks.copy()  # owns its memory: freezing it must not leave a writable base

    is_kick = np.any(actions != 0.0, axis=2).reshape(-1)
    kick_index = np.flatnonzero(is_kick)
    plain_index = np.flatnonzero(~is_kick)
    plain_wanted = recipe.subsample_pairs - recipe.kick_pairs
    if kick_index.size < recipe.kick_pairs:
        raise TrainingDataError(
            f"only {kick_index.size} kick pairs exist in {world_name}'s {n} trajectories but "
            f"kick_pairs is {recipe.kick_pairs} — refusing to shrink the count (models spec §4)"
        )
    if plain_index.size < plain_wanted:
        raise TrainingDataError(
            f"only {plain_index.size} non-kick pairs exist in {world_name}'s {n} trajectories "
            f"but {plain_wanted} are needed (subsample_pairs {recipe.subsample_pairs} minus "
            f"kick_pairs {recipe.kick_pairs}) — refusing to shrink the count"
        )

    kick_order = seeds.rng_for(world_name, TRAINING_REGION, "subsample-kick").permutation(kick_index.size)
    plain_order = seeds.rng_for(world_name, TRAINING_REGION, "subsample-nonkick").permutation(
        plain_index.size
    )
    chosen = np.concatenate(
        [kick_index[kick_order[: recipe.kick_pairs]], plain_index[plain_order[:plain_wanted]]]
    )

    unused = np.ones(is_kick.size, dtype=bool)
    unused[chosen] = False
    pool = np.flatnonzero(unused)
    if pool.size < recipe.heldout_pairs:
        raise TrainingDataError(
            f"only {pool.size} pairs are left after the training set but heldout_pairs is "
            f"{recipe.heldout_pairs} — the held-out set must not overlap training and the "
            f"count is never shrunk"
        )
    held = pool[seeds.rng_for(world_name, TRAINING_REGION, "heldout").permutation(pool.size)][
        : recipe.heldout_pairs
    ]
    gradcheck = (
        seeds.rng_for(world_name, TRAINING_REGION, "gradcheck-batch")
        .permutation(chosen.size)[: recipe.gradcheck_pairs]
        .astype(np.int64)
    )

    return TrainingData(
        states=states,
        actions=actions,
        train_pairs=_gather_pairs(states, actions, chosen, horizon),
        heldout_pairs=_gather_pairs(states, actions, held, horizon),
        gradcheck_index=gradcheck,
    )


def assert_eval_starts_disjoint(training: TrainingData, eval_starts: np.ndarray) -> None:
    """Refuse if any evaluation start is exactly a training start (MU-7, TC-MU7-01).

    In plain words: the exam must never begin from a question the model
    practised on.

    Only *starts* are compared: later states of a training trajectory may
    legitimately resemble evaluation states (the systems cycle) — MU-7
    forbids literal start reuse.
    """
    eval_starts = np.asarray(eval_starts)
    d = training.states.shape[2]
    if eval_starts.ndim != 2 or eval_starts.shape[1] != d:
        raise TrainingDataError(
            f"evaluation starts must have shape [n, {d}], got {tuple(eval_starts.shape)}"
        )
    used = {row.tobytes() for row in training.states[:, 0]}
    for i, row in enumerate(np.ascontiguousarray(eval_starts, dtype=np.float64)):
        if row.tobytes() in used:
            raise TrainingDataError(
                f"evaluation start {i} is also a training start — evaluation must never "
                f"start from a training initial condition (MU-7, TC-MU7-01)"
            )
