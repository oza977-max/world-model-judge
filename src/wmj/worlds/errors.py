"""wmj.worlds.errors — exceptions raised by the worlds package.

In plain words: a world's declared regions and action range are its
contract. An excursion outside that contract — a state driven below
the floor, an action outside the declared range — is not data to
train on or grade against; it means the region/action declarations
allowed an unphysical excursion, which is a spec bug to fix, never
something to quietly paper over (worlds spec §7).
"""

from __future__ import annotations

from wmj.errors import WmjError


class StateFloorClampError(WmjError):
    """Raised when a step would drive a state dimension below the floor.

    Worlds spec §7: "any LV state-floor clamp activation... aborts the
    run it occurs in — loudly... whether that run is benchmark
    generation, training-data generation, or a judged evaluation."
    One rule, no scope exceptions — this is never silently clamped.
    """


class ActionRangeError(WmjError):
    """Raised when an action falls outside the world's declared range.

    Worlds spec §7: the declared range is the world's contract;
    out-of-*trained*-range is legitimate and labelled, out-of-
    *declared*-range is a caller bug.
    """


class RegionSpecError(WmjError):
    """Raised when a world's own declared regions violate worlds §7.

    Checked once, at module import time, against the world's hardcoded
    constants: the training box must be strictly inside the
    state-floor-safe domain, and every out-region must be disjoint
    from the training box on at least one axis.
    """
