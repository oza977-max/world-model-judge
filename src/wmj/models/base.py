"""wmj.models.base — content-addressed seeding, shared by every model.

In plain words: every trained component (a model, a fixture, a member
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


@dataclass(frozen=True)
class TrainingData:
    """The seeded training trajectories every factory fits against.

    Built once per world by the harness and hand identically to every
    registered factory — one producer, one construction site (models
    spec ADR-M1).
    """

    states: np.ndarray  # float64[N, H+1, d]
    actions: np.ndarray  # float64[N, H, a]


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
