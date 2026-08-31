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


@dataclass(frozen=True)
class OutRegion:
    """One declared out-of-training region (worlds spec ADR-W4)."""

    region_name: str
    axis: str  # "state" | "action" | "both"
    state_box: np.ndarray  # float64[d, 2]
    action_box: np.ndarray  # float64[a, 2]


@dataclass(frozen=True)
class RegionSpec:
    """A world's declared training and out-of-training regions."""

    training_state_box: np.ndarray  # float64[d, 2]
    training_action_interval: np.ndarray  # float64[a, 2]
    out_regions: tuple[OutRegion, ...]


@dataclass(frozen=True)
class Task:
    """One evaluation task a world declares (worlds spec §4)."""

    name: str
    kind: str  # "control" | "planning"
    tolerance: float
    horizon: int
