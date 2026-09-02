"""Tests for wmj.harness.regions — in/out labelling with axis attribution.

TC-WD5-01 (boundary value analysis): a start just inside the training
box labels in-region, just outside labels out-of-region, and exactly
on the boundary is in-region — deterministically (worlds ADR-W4:
membership is closed on the training region).

TC-WD5-02 (equivalence partitioning): a state inside the training box
with an action outside the trained action interval is flagged
out-of-region on the *action* axis — never silently in-region because
the state looked familiar.

Label shape is the one canonical `{"region_name", "axis"}` pair
(worlds §5, judge §4), with `axis` in {"state", "action", "both", None}.
"""

from __future__ import annotations

import numpy as np
import pytest

from wmj.harness.regions import (
    UndeclaredRegionError,
    actions_in_trained_interval,
    attribute_axis,
    label_trial,
    state_in_training_box,
)
from wmj.worlds import lv, pendulum

LV_SPEC = lv.regions()  # training x in [2,6], y in [1,4]; actions [-0.5, 0.5]
PD_SPEC = pendulum.regions()

NULL_ACTIONS = np.zeros((10, 1))


# --- state_in_training_box: closed on the boundary (TC-WD5-01) ---


def test_tc_wd5_01_state_just_inside_the_box_is_in():
    assert state_in_training_box(np.array([2.0 + 1e-9, 1.0 + 1e-9]), LV_SPEC.training_state_box)


def test_tc_wd5_01_state_just_outside_the_box_is_out():
    assert not state_in_training_box(np.array([2.0 - 1e-9, 2.0]), LV_SPEC.training_state_box)
    assert not state_in_training_box(np.array([4.0, 4.0 + 1e-9]), LV_SPEC.training_state_box)


def test_tc_wd5_01_state_exactly_on_every_boundary_is_in():
    box = LV_SPEC.training_state_box
    for corner in (
        np.array([2.0, 1.0]),
        np.array([6.0, 4.0]),
        np.array([2.0, 4.0]),
        np.array([6.0, 1.0]),
    ):
        assert state_in_training_box(corner, box)


def test_state_in_training_box_is_deterministic_on_repeat():
    edge = np.array([6.0, 2.5])
    results = {state_in_training_box(edge, LV_SPEC.training_state_box) for _ in range(5)}
    assert results == {True}


# --- actions_in_trained_interval: closed, any-step violation counts ---


def test_actions_all_inside_the_trained_interval_are_in():
    actions = np.full((10, 1), 0.5)  # exactly on the edge, every step
    assert actions_in_trained_interval(actions, LV_SPEC.training_action_interval)


def test_a_single_out_of_range_action_anywhere_in_the_rollout_is_out():
    actions = np.zeros((10, 1))
    actions[7, 0] = 0.5 + 1e-9
    assert not actions_in_trained_interval(actions, LV_SPEC.training_action_interval)


# --- attribute_axis: ADR-W4's four values ---


@pytest.mark.parametrize(
    "state_in, actions_in, expected",
    [
        (True, True, None),
        (False, True, "state"),
        (True, False, "action"),
        (False, False, "both"),
    ],
)
def test_attribute_axis_covers_all_four_adr_w4_values(state_in, actions_in, expected):
    assert attribute_axis(state_in, actions_in) == expected


# --- label_trial: the canonical shape ---


def test_label_trial_returns_exactly_the_canonical_two_key_shape():
    label = label_trial(LV_SPEC, "training", np.array([4.0, 2.5]), NULL_ACTIONS)
    assert label == {"region_name": "training", "axis": None}


def test_tc_wd5_02_state_inside_but_action_outside_is_flagged_on_the_action_axis():
    actions = np.zeros((10, 1))
    actions[3, 0] = 0.9  # inside the world's declared range, outside the TRAINED interval
    label = label_trial(LV_SPEC, "training", np.array([4.0, 2.5]), actions)
    assert label == {"region_name": "training", "axis": "action"}


def test_out_region_start_is_flagged_on_the_state_axis_and_keeps_its_region_name():
    start = np.array([10.0, 5.0])  # inside out-high-amplitude, outside training
    label = label_trial(LV_SPEC, "out-high-amplitude", start, NULL_ACTIONS)
    assert label == {"region_name": "out-high-amplitude", "axis": "state"}


def test_out_region_start_with_out_of_range_action_is_both():
    actions = np.zeros((10, 1))
    actions[0, 0] = -0.75
    label = label_trial(LV_SPEC, "out-high-amplitude", np.array([10.0, 5.0]), actions)
    assert label["axis"] == "both"


def test_pendulum_labels_follow_the_same_rules():
    inside = np.array([0.1, -0.2, 0.3, 0.0])
    near_inverted = np.array([2.8, 0.0, 0.0, 0.0])
    ok_actions = np.zeros((10, 1))
    hot_actions = np.zeros((10, 1))
    hot_actions[5, 0] = 1.5  # trained interval is [-1, 1]
    assert label_trial(PD_SPEC, "training", inside, ok_actions)["axis"] is None
    assert label_trial(PD_SPEC, "training", inside, hot_actions)["axis"] == "action"
    assert label_trial(PD_SPEC, "out-near-inverted", near_inverted, ok_actions)["axis"] == "state"


def test_label_trial_fails_loudly_on_an_undeclared_region_name():
    with pytest.raises(UndeclaredRegionError):
        label_trial(LV_SPEC, "out-of-thin-air", np.array([4.0, 2.5]), NULL_ACTIONS)
