"""wmj.harness.regions — labels each trial in-region or out, and says which axis.

In plain words: every evaluation trial gets a small tag that says two
things — which declared region its starting point was drawn from
(`region_name`, the key used to pick the right divergence curve and
climatology table), and whether anything about the trial left the
territory the models were trained on (`axis`): the *state* wandered
outside the training box, an *action* fell outside the trained range,
*both*, or neither (`null`). Without the axis, "it failed out of
region" would hide whether the model met an unfamiliar situation or
an unfamiliar command — WD-5's "say which" clause (worlds spec ADR-W4).

Membership is closed: a value exactly on a boundary is inside. This
is the same `<=` discipline as `within_tolerance` (P2-C01), and for
the same reason — a boundary case must classify the same way every
time, on every machine, with no epsilon to argue about.

The region a trial belongs to is known by construction: the harness
draws starts per declared region (cross-cutting ADR-004), so this
module takes `region_name` as an input and validates it, rather than
re-deriving it from the state. A trial drawn for `"training"` whose
rollout uses an out-of-range action is therefore
`{"region_name": "training", "axis": "action"}` — compared against the
training curve (where it started) but flagged on the action axis,
exactly TC-WD5-02's case.
"""

from __future__ import annotations

import numpy as np

from wmj.errors import WmjError
from wmj.worlds.base import RegionSpec


class UndeclaredRegionError(WmjError):
    """Raised when a label would name a region the world never declared.

    The name is the join key every downstream lookup uses; a typo here
    would silently detach the trial from its divergence curve and
    climatology table, so it fails loudly instead (WD-5, worlds ADR-W4;
    no numbered test case covers this path — it is a fail-loudly
    convention, cross-cutting Error-Handling rule 1).
    """


def state_in_training_box(state: np.ndarray, box: np.ndarray) -> bool:
    """True iff every state dimension lies in `[low, high]` (closed)."""
    return bool(np.all(state >= box[:, 0]) and np.all(state <= box[:, 1]))


def actions_in_trained_interval(actions: np.ndarray, interval: np.ndarray) -> bool:
    """True iff every action at every step lies in `[low, high]` (closed).

    `actions` is `float64[H, a]`; `interval` is `float64[a, 2]`. One
    out-of-range action anywhere in the rollout makes the whole trial
    out-of-region on the action axis (ADR-W4: "*any* action").
    """
    return bool(np.all(actions >= interval[:, 0]) and np.all(actions <= interval[:, 1]))


def attribute_axis(state_in: bool, actions_in: bool) -> str | None:
    """ADR-W4's four-value convention.

    `None` when fully in-region; `"state"`, `"action"`, or `"both"`
    naming which axis left the trained territory.
    """
    if state_in and actions_in:
        return None
    if not state_in and not actions_in:
        return "both"
    return "state" if not state_in else "action"


def declared_region_names(region_spec: RegionSpec) -> tuple[str, ...]:
    """`"training"` followed by every declared out-region, in declared order."""
    return ("training", *(out.region_name for out in region_spec.out_regions))


def label_trial(
    region_spec: RegionSpec,
    region_name: str,
    start_state: np.ndarray,
    actions: np.ndarray,
) -> dict[str, str | None]:
    """The canonical per-trial label `{"region_name", "axis"}` (worlds §5).

    `region_name` must be one the world declared; it is the region the
    start was drawn for (ADR-004), not re-derived here.
    """
    declared = declared_region_names(region_spec)
    if region_name not in declared:
        raise UndeclaredRegionError(
            f"WD-5 label: region {region_name!r} is not one this world declares "
            f"{list(declared)} — refusing to emit a label that no divergence "
            f"curve or climatology table could be joined to (WD-5, worlds "
            f"ADR-W4; fail-loudly convention, cross-cutting Error-Handling rule 1)"
        )
    state_in = state_in_training_box(start_state, region_spec.training_state_box)
    actions_in = actions_in_trained_interval(actions, region_spec.training_action_interval)
    return {"region_name": region_name, "axis": attribute_axis(state_in, actions_in)}
