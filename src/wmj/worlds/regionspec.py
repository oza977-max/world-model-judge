"""wmj.worlds.regionspec — the one check that a world's regions make sense.

In plain words: every "out-of-training" region a world declares has to
actually be out of training territory, in the way it says it is. A
state-axis region must start somewhere the training box does not reach;
an action-axis region must use kicks bigger than any the models were
trained on. A region that fails this would quietly test nothing new, so
the world refuses to load (worlds spec §7, ADR-W4). Both worlds call this
one function, so the rule cannot drift apart between them.
"""

from __future__ import annotations

import numpy as np

from wmj.worlds.base import OutRegion, RegionSpec
from wmj.worlds.errors import RegionSpecError

_AXES = ("state", "action", "both")


def _state_disjoint(out_region: OutRegion, training_box: np.ndarray) -> bool:
    """True iff the region's state box misses the training box on some axis."""
    return bool(
        np.any(
            (out_region.state_box[:, 0] > training_box[:, 1])
            | (out_region.state_box[:, 1] < training_box[:, 0])
        )
    )


def _action_wider(out_region: OutRegion, training_interval: np.ndarray) -> bool:
    """True iff the region's action range reaches past the trained interval."""
    return bool(
        np.any(
            (out_region.action_box[:, 0] < training_interval[:, 0])
            | (out_region.action_box[:, 1] > training_interval[:, 1])
        )
    )


def validate_out_regions(world_name: str, region_spec: RegionSpec) -> None:
    """Raise `RegionSpecError` unless every out-region is out on its declared axis.

    - `"state"`: its state box is disjoint from the training box on at
      least one dimension, and its action box *is* the trained interval
      (its trials get the trained "in" kicks, so a wider declared box
      would describe kicks it never receives);
    - `"action"`: its state box *is* the training box (otherwise its
      trials would be labelled `"both"`), and its action box is exactly
      twice the trained interval — the hull of the "out" kicks
      (`wmj.worlds.actions`, `|u| ∈ (umax, 2·umax]`);
    - `"both"`: a disjoint state box and the doubled action box.

    Tying the declared boxes to the kicks the region really gets means an
    edit to a region's declaration cannot silently disagree with what its
    trials do (independent review, P3-C09 pass 1).
    """
    for out_region in region_spec.out_regions:
        if out_region.axis not in _AXES:
            raise RegionSpecError(
                f"{world_name} out-region {out_region.region_name!r} declares axis "
                f"{out_region.axis!r}, not one of {list(_AXES)} (worlds ADR-W4)"
            )
        needs_state = out_region.axis in ("state", "both")
        needs_action = out_region.axis in ("action", "both")
        if needs_state and not _state_disjoint(out_region, region_spec.training_state_box):
            raise RegionSpecError(
                f"{world_name} out-region {out_region.region_name!r} is not disjoint "
                f"from the training box on any axis (worlds spec §7)"
            )
        trained = region_spec.training_action_interval
        expected_action_box = 2.0 * trained if needs_action else trained
        if not np.array_equal(out_region.action_box, expected_action_box):
            raise RegionSpecError(
                f"{world_name} out-region {out_region.region_name!r} (axis "
                f"{out_region.axis!r}) declares action box {out_region.action_box.tolist()}, "
                f"but its trials get kicks spanning {expected_action_box.tolist()} "
                f"(worlds spec §7, ADR-W4)"
            )
        if out_region.axis == "action" and not np.array_equal(
            out_region.state_box, region_spec.training_state_box
        ):
            raise RegionSpecError(
                f"{world_name} out-region {out_region.region_name!r} is out on the action "
                f"axis only, so its state box must be the training box "
                f"{region_spec.training_state_box.tolist()} (worlds spec §7, ADR-W4)"
            )
        # Defence in depth: after the equality check above this can only
        # fire for a degenerate trained interval (e.g. [0, 0], whose double
        # is itself) — a region that would then test no new kicks at all.
        if needs_action and not _action_wider(out_region, trained):
            raise RegionSpecError(
                f"{world_name} out-region {out_region.region_name!r} declares the action "
                f"axis but its action range {out_region.action_box.tolist()} does not reach "
                f"past the trained interval "
                f"{region_spec.training_action_interval.tolist()} (worlds spec §7)"
            )
