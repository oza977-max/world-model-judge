"""wmj.worlds.base — shared value types every world's interface returns.

In plain words: a world doesn't just simulate physics, it also
declares what counts as "the training region" versus "somewhere the
model was never trained" (`RegionSpec`), and what counts as a task
worth grading ("hold the population steady" vs "will it crash?",
`Task`). These are plain data, not behaviour, so the harness can build
a `WorldContext` (models spec ADR-M1) from them without either package
importing the other's internals.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _freeze_arrays(*arrays: object) -> None:
    """Mark numpy arrays read-only in place.

    A world's region boxes are built once at import and returned by
    reference from every `regions()` call for the rest of the process;
    `frozen=True` protects the field bindings, not the array contents.
    Read-only flags make a stray in-place write fail at once instead
    of silently changing the region for every later caller
    (code-review-001 I7).
    """
    for array in arrays:
        if isinstance(array, np.ndarray):
            array.setflags(write=False)


@dataclass(frozen=True)
class OutRegion:
    """One declared out-of-training region (worlds spec ADR-W4)."""

    region_name: str
    axis: str  # "state" | "action" | "both"
    state_box: np.ndarray  # float64[d, 2]
    action_box: np.ndarray  # float64[a, 2]

    def __post_init__(self) -> None:
        _freeze_arrays(self.state_box, self.action_box)


@dataclass(frozen=True)
class RegionSpec:
    """A world's declared training and out-of-training regions."""

    training_state_box: np.ndarray  # float64[d, 2]
    training_action_interval: np.ndarray  # float64[a, 2]
    out_regions: tuple[OutRegion, ...]

    def __post_init__(self) -> None:
        _freeze_arrays(self.training_state_box, self.training_action_interval)


@dataclass(frozen=True)
class Task:
    """One evaluation task a world declares (worlds spec §4)."""

    name: str
    kind: str  # "control" | "planning"
    tolerance: float
    horizon: int


def distance(a: np.ndarray, b: np.ndarray, scale: np.ndarray) -> float:
    """The one shared normalised-distance metric (worlds spec ADR-W3).

    In plain words: to compare two states fairly when they mix
    different units (a rabbit count, a radian), divide each dimension
    by the world's own typical scale first, then take the RMS. This is
    the same metric used everywhere a "how far off" number is needed —
    the divergence curves, the judge's trajectory grading, every
    task's tolerance.

    RMS over state dimensions each divided by the world's declared
    scale vector: distance(a, b) = sqrt(mean_d(((a_d - b_d) / scale_d)^2)).
    """
    normalised_diff = (a - b) / scale
    return float(np.sqrt(np.mean(normalised_diff**2)))


def within_tolerance(distance_value: float, tolerance: float) -> bool:
    """Band-edge classification is closed (worlds spec §4.1): a
    distance exactly equal to the tolerance passes, deterministically —
    never an off-by-one ambiguity at the boundary."""
    return distance_value <= tolerance
