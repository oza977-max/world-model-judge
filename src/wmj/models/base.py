"""wmj.models.base — the shared types every model uses: seeding, predictions,
world context and training data.

In plain words (types): `Prediction` is what every model says back (a best
guess and an error bar); `WorldContext` is the world facts a model is handed;
`Pairs`/`TrainingData` are the shared homework (see their docstrings).

Seeding, in plain words: every trained component (a model, a fixture, a member
of an ensemble) gets its own random-number stream, derived from its
*name* rather than from where it happens to sit in a list. That means
adding a new model never quietly reshuffles another model's training —
a change that would otherwise be invisible in the code and show up
only as a different verdict (cross-cutting ADR-002 rule 2).

Lives here, not in the harness, so a fixture can rebuild another
model's exact stream (`seeds.rng_for("direct", ...)`) without models
importing the harness (ADR-003's no `models -> harness` import rule).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from wmj.errors import WmjError


def _freeze_arrays(*arrays: object) -> None:
    """Mark numpy arrays read-only in place.

    `frozen=True` on a dataclass stops its *fields* being rebound; it
    says nothing about the contents of an array a field holds. These
    types are handed around by reference (a model returns the same
    fitted spread from every `predict()`; a world's context is built
    once per run), so a stray in-place write anywhere downstream would
    silently corrupt every later reader. Read-only flags turn that into
    an immediate error instead (code-review-001 I7).
    """
    for array in arrays:
        if isinstance(array, np.ndarray):
            array.setflags(write=False)


class SeedKeyError(WmjError):
    """Raised when a component_key part is not a bare, colon-free str."""


def component_key(*parts: str) -> tuple[int]:
    """Derive a stable seed key from named parts, content-addressed.

    In plain words: turns names like ("direct", "lv", "weights") into
    one fixed number, the same number every time, no matter what else
    exists elsewhere in the roster — that's what "content-addressed"
    means (as opposed to "the 3rd model gets seed 3").

    Every part must already be a str (the caller does its own str()
    conversion, e.g. str(k) for a member index) and must not contain
    ':', the join delimiter — both are rejected here so the join stays
    injective (cross-cutting ADR-002 rule 2, TC-NF1-07, TC-NF1-08).
    """
    for part in parts:
        if not isinstance(part, str):
            raise SeedKeyError(
                f"seed key part {part!r} is not a str "
                f"(type {type(part).__name__}); the caller "
                f"must pass its own str() form explicitly (TC-NF1-08)"
            )
        if ":" in part:
            raise SeedKeyError(
                f"seed key part {part!r} contains ':' (reserved delimiter); "
                f"model/world/region/purpose names must not contain ':' (TC-NF1-07)"
            )
    joined = ":".join(parts).encode("utf-8")
    digest = hashlib.blake2b(joined, digest_size=8).digest()
    return (int.from_bytes(digest, "big"),)


@dataclass(frozen=True)
class Prediction:
    """One model's prediction at one step (models spec ADR-M1).

    In plain words: every model, no matter how it works inside, says
    the same two things back — a per-dimension best guess (`mean`) and
    a per-dimension one-standard-deviation error bar (`spread`). That
    fixed shape is what lets the judge grade every model the same way.
    """

    mean: np.ndarray  # float64[d]
    spread: np.ndarray  # float64[d], one standard deviation

    def __post_init__(self) -> None:
        _freeze_arrays(self.mean, self.spread)


@dataclass(frozen=True)
class WorldContext:
    """World geometry handed to every model factory (models spec ADR-M1).

    In plain words: a model is never allowed to import the world it's
    being trained for (that coupling is banned, ADR-003) — so whatever
    it needs to know about that world's shape and scale arrives here
    instead, as plain data, the same way for every model.
    """

    world_name: str
    state_dim: int
    action_dim: int
    training_state_box: np.ndarray  # float64[d, 2]
    training_action_interval: np.ndarray  # float64[a, 2]
    scale: np.ndarray  # float64[d]

    def __post_init__(self) -> None:
        _freeze_arrays(self.training_state_box, self.training_action_interval, self.scale)


class TrainingDataShapeError(WmjError):
    """Raised when training data (or a `Pairs` bundle) is malformed.

    In plain words: the homework handed to every model must be well
    formed — matching shapes, finite numbers, indices that point at real
    rows — or it is refused on the spot, not discovered halfway through
    training (models spec ADR-M1, design-review-010).
    """


@dataclass(frozen=True)
class Pairs:
    """One-step examples: (state, action) -> next state, plus a kick flag.

    In plain words: each row says "from this state, with this push, the
    world moved to that state"; `is_kick` marks the rows where the push
    was not zero. Built once by the harness, never by a model (models
    spec ADR-M1, §4).
    """

    state: np.ndarray  # float64[m, d]
    action: np.ndarray  # float64[m, a]
    next_state: np.ndarray  # float64[m, d]
    is_kick: np.ndarray  # bool[m]

    def __post_init__(self) -> None:
        for name in ("state", "action", "next_state"):
            array = getattr(self, name)
            if not isinstance(array, np.ndarray) or array.ndim != 2:
                raise TrainingDataShapeError(f"Pairs.{name} must be a 2-D numpy array")
            if array.dtype != np.float64:
                raise TrainingDataShapeError(
                    f"Pairs.{name} must be float64, got {array.dtype} (models spec ADR-M1)"
                )
            if not np.all(np.isfinite(array)):
                raise TrainingDataShapeError(f"Pairs.{name} must hold only finite numbers")
        m = self.state.shape[0]
        if self.action.shape[0] != m or self.next_state.shape[0] != m:
            raise TrainingDataShapeError(
                f"Pairs rows disagree: state {self.state.shape[0]}, action "
                f"{self.action.shape[0]}, next_state {self.next_state.shape[0]}"
            )
        if self.next_state.shape[1] != self.state.shape[1]:
            raise TrainingDataShapeError(
                f"Pairs.next_state width {self.next_state.shape[1]} differs from "
                f"state width {self.state.shape[1]}"
            )
        if (
            not isinstance(self.is_kick, np.ndarray)
            or self.is_kick.dtype != np.bool_
            or self.is_kick.shape != (m,)
        ):
            raise TrainingDataShapeError(f"Pairs.is_kick must be a bool array of shape ({m},)")
        if not np.array_equal(self.is_kick, np.any(self.action != 0.0, axis=1)):
            raise TrainingDataShapeError(
                "Pairs.is_kick disagrees with the actions: a kick pair is exactly one whose "
                "action is not zero (the exact kick quota rests on this flag)"
            )
        _freeze_arrays(self.state, self.action, self.next_state, self.is_kick)


@dataclass(frozen=True)
class TrainingData:
    """The seeded training trajectories every factory fits against.

    In plain words: the shared homework. `states`/`actions` are the
    simulated histories; `train_pairs` is the fixed set of one-step examples
    every network trains on; `heldout_pairs` is held back for checking
    only; `gradcheck_index` picks the few training rows used once to check
    the learning arithmetic. Built once per world by the harness and handed
    identically to every registered factory — one producer, one
    construction site (models spec ADR-M1, design-review-010).

    The three pair fields are optional only so the earlier skeleton and
    preview builders, which fit the baselines on a handful of trajectories,
    still construct; the MLP factories (P3-C03/C04) must refuse `None` when
    they are built (backlog A19, REMEMBER.md §3).
    """

    states: np.ndarray  # float64[N, H+1, d]
    actions: np.ndarray  # float64[N, H, a]
    train_pairs: Pairs | None = None
    heldout_pairs: Pairs | None = None
    gradcheck_index: np.ndarray | None = None  # int64[g], indices into train_pairs

    def __post_init__(self) -> None:
        for name in ("states", "actions"):
            array = getattr(self, name)
            if not isinstance(array, np.ndarray) or array.ndim != 3:
                raise TrainingDataShapeError(
                    f"{name} must be a 3-D numpy array [trajectories, steps, dims]"
                )
            if array.dtype != np.float64:
                raise TrainingDataShapeError(f"{name} must be float64, got {array.dtype}")
            if not np.all(np.isfinite(array)):
                raise TrainingDataShapeError(f"{name} must hold only finite numbers")
        if self.states.shape[0] != self.actions.shape[0]:
            raise TrainingDataShapeError(
                f"{self.states.shape[0]} state trajectories but {self.actions.shape[0]} "
                f"action trajectories"
            )
        if self.states.shape[1] != self.actions.shape[1] + 1:
            raise TrainingDataShapeError(
                f"states must have one more step than actions (H+1 vs H): got "
                f"{self.states.shape[1]} and {self.actions.shape[1]}"
            )
        d, a = self.states.shape[2], self.actions.shape[2]
        for name in ("train_pairs", "heldout_pairs"):
            pairs = getattr(self, name)
            if pairs is not None and not isinstance(pairs, Pairs):
                raise TrainingDataShapeError(f"{name} must be a Pairs, got {type(pairs).__name__}")
            if pairs is not None and (pairs.state.shape[1] != d or pairs.action.shape[1] != a):
                raise TrainingDataShapeError(
                    f"{name} width (state {pairs.state.shape[1]}, action "
                    f"{pairs.action.shape[1]}) differs from the trajectories' (state {d}, action {a})"
                )
        if self.gradcheck_index is not None:
            index = self.gradcheck_index
            if self.train_pairs is None:
                raise TrainingDataShapeError("a gradcheck_index needs train_pairs to index into")
            m = self.train_pairs.state.shape[0]
            if (
                not isinstance(index, np.ndarray)
                or index.ndim != 1
                or index.dtype != np.int64
                or index.size == 0
                or index.min() < 0
                or index.max() >= m
                or np.unique(index).size != index.size
            ):
                raise TrainingDataShapeError(
                    f"gradcheck_index must be a non-empty 1-D int64 array of distinct rows in "
                    f"[0, {m}) of train_pairs (the first g of a permutation of the training set)"
                )
        _freeze_arrays(self.states, self.actions, self.gradcheck_index)


@dataclass(frozen=True)
class SeedSource:
    """Handed to every model factory as data (models spec ADR-M1).

    In plain words: this is the one door a factory has to randomness.
    `rng(*purpose)` gives the factory its own stream; `rng_for(*parts)`
    lets a fixture reach another named component's stream on purpose
    (e.g. to rebuild `direct`'s weights bit-for-bit).
    """

    run_seed: int
    my_name: str | None

    def rng(self, *purpose: str) -> np.random.Generator:
        if self.my_name is None:
            raise SeedKeyError(
                "SeedSource.rng() requires my_name to be set; "
                "use rng_for(*parts) to derive another component's stream "
                "(cross-cutting ADR-002 rule 2)"
            )
        return self.rng_for(self.my_name, *purpose)

    def rng_for(self, *parts: str) -> np.random.Generator:
        key = component_key(*parts)
        seed_sequence = np.random.SeedSequence(entropy=[self.run_seed, *key])
        return np.random.Generator(np.random.PCG64(seed_sequence))
