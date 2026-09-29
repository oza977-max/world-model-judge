"""The action lever's declared settings and the action-axis region.

In plain words: Round 10 measured that predator–prey kicks of up to 0.5
crash the prey population through its floor on full-length runs, so the
trained kicks are now at most 0.1, pulled about once every two seconds of
world time; the pendulum keeps kicks up to 1.0, about once a second. Each
world also declares a third region, `out-large-action`: familiar
starting states, but kicks bigger than anything seen in training — the
only way a trial can be "out of region" on the action axis (WD-5,
TC-WD5-02). These tests pin those declarations (worlds §4.1/§4.2,
ADR-W4, design-review-010).
"""

from __future__ import annotations

import numpy as np
import pytest

from wmj.harness.regions import declared_region_names, label_trial
from wmj.worlds import lv, pendulum
from wmj.worlds.actions import band_for_axis, kick_sequence, step_probability
from wmj.worlds.base import OutRegion, RegionSpec
from wmj.worlds.errors import RegionSpecError


def test_lv_trained_kicks_are_at_most_point_one_at_half_a_kick_per_second():
    spec = lv.regions()
    assert np.array_equal(spec.training_action_interval, np.array([[-0.1, 0.1]]))
    assert lv.KICK_RATE_PER_S == 0.5
    assert step_probability(lv.KICK_RATE_PER_S, lv.DT) == 0.5 * 0.02


def test_pendulum_trained_kicks_are_at_most_one_at_one_kick_per_second():
    spec = pendulum.regions()
    assert np.array_equal(spec.training_action_interval, np.array([[-1.0, 1.0]]))
    assert pendulum.KICK_RATE_PER_S == 1.0
    assert step_probability(pendulum.KICK_RATE_PER_S, pendulum.DT) == 1.0 * 0.002


@pytest.mark.parametrize("module", [lv, pendulum])
def test_each_world_declares_three_regions_in_a_fixed_order(module):
    names = declared_region_names(module.regions())
    state_region = "out-high-amplitude" if module is lv else "out-near-inverted"
    assert names == ("training", state_region, "out-large-action")


@pytest.mark.parametrize(
    ("module", "outer"), [(lv, 0.2), (pendulum, 2.0)]
)
def test_out_large_action_region_is_the_training_box_with_twice_the_kick_width(module, outer):
    spec = module.regions()
    region = {r.region_name: r for r in spec.out_regions}["out-large-action"]
    assert region.axis == "action"
    assert np.array_equal(region.state_box, spec.training_state_box)
    assert np.array_equal(region.action_box, np.array([[-outer, outer]]))
    assert band_for_axis(region.axis) == "out"


def test_lv_state_axis_region_uses_the_trained_kick_interval():
    region = {r.region_name: r for r in lv.regions().out_regions}["out-high-amplitude"]
    assert np.array_equal(region.action_box, np.array([[-0.1, 0.1]]))
    assert band_for_axis(region.axis) == "in"


@pytest.mark.parametrize("module", [lv, pendulum])
def test_out_large_action_kicks_stay_inside_the_declared_full_action_range(module):
    region = {r.region_name: r for r in module.regions().out_regions}["out-large-action"]
    assert module.ACTION_RANGE[0] <= region.action_box[0, 0]
    assert region.action_box[0, 1] <= module.ACTION_RANGE[1]


@pytest.mark.parametrize("module", [lv, pendulum])
def test_validator_rejects_an_action_region_with_no_room_outside_the_trained_interval(module):
    spec = module.regions()
    bad = RegionSpec(
        training_state_box=spec.training_state_box.copy(),
        training_action_interval=spec.training_action_interval.copy(),
        out_regions=(
            OutRegion(
                region_name="out-large-action",
                axis="action",
                state_box=spec.training_state_box.copy(),
                action_box=spec.training_action_interval.copy(),  # no wider
            ),
        ),
    )
    with pytest.raises(RegionSpecError, match="§7"):
        module._validate_region_spec(bad)


@pytest.mark.parametrize("module", [lv, pendulum])
def test_validator_rejects_an_unknown_axis(module):
    spec = module.regions()
    bad = RegionSpec(
        training_state_box=spec.training_state_box.copy(),
        training_action_interval=spec.training_action_interval.copy(),
        out_regions=(
            OutRegion(
                region_name="out-sideways",
                axis="sideways",
                state_box=spec.training_state_box.copy(),
                action_box=spec.training_action_interval.copy(),
            ),
        ),
    )
    with pytest.raises(RegionSpecError, match="ADR-W4"):
        module._validate_region_spec(bad)


@pytest.mark.parametrize("module", [lv, pendulum])
def test_tc_wd5_02_an_out_large_action_trial_is_labelled_on_the_action_axis(module):
    # TC-WD5-02 made producible (design-review-010): start inside the
    # training box, kicks drawn from the out band → axis "action".
    spec = module.regions()
    rng = np.random.default_rng(5)
    start = rng.uniform(spec.training_state_box[:, 0], spec.training_state_box[:, 1])
    umax = float(spec.training_action_interval[0, 1])
    kicks = kick_sequence(rng, horizon=500, p_step=0.05, band="out", umax=umax)
    assert np.any(kicks != 0.0)
    label = label_trial(spec, "out-large-action", start, kicks)
    assert label == {"region_name": "out-large-action", "axis": "action"}


@pytest.mark.parametrize("module", [lv, pendulum])
def test_a_training_trial_with_in_band_kicks_stays_fully_in_region(module):
    spec = module.regions()
    rng = np.random.default_rng(6)
    start = rng.uniform(spec.training_state_box[:, 0], spec.training_state_box[:, 1])
    umax = float(spec.training_action_interval[0, 1])
    kicks = kick_sequence(rng, horizon=500, p_step=0.05, band="in", umax=umax)
    assert label_trial(spec, "training", start, kicks) == {"region_name": "training", "axis": None}


@pytest.mark.parametrize("module", [lv, pendulum])
def test_world_object_exposes_its_kick_rate(module):
    assert module.WORLD.kick_rate_per_s == module.KICK_RATE_PER_S


def _spec_with(module, out_region):
    spec = module.regions()
    return RegionSpec(
        training_state_box=spec.training_state_box.copy(),
        training_action_interval=spec.training_action_interval.copy(),
        out_regions=(out_region,),
    )


@pytest.mark.parametrize("module", [lv, pendulum])
def test_validator_rejects_a_state_region_declaring_kicks_it_never_gets(module):
    # Its trials get the trained "in" kicks, so a wider declared action
    # box would misdescribe them (independent review, P3-C09 pass 1).
    real = module.regions().out_regions[0]
    bad = OutRegion(real.region_name, "state", real.state_box.copy(), 2.0 * real.action_box)
    with pytest.raises(RegionSpecError, match="ADR-W4"):
        module._validate_region_spec(_spec_with(module, bad))


@pytest.mark.parametrize("module", [lv, pendulum])
def test_validator_rejects_an_action_region_whose_box_is_not_the_out_kick_hull(module):
    spec = module.regions()
    bad = OutRegion("out-large-action", "action", spec.training_state_box.copy(),
                    1.5 * spec.training_action_interval)
    with pytest.raises(RegionSpecError, match="ADR-W4"):
        module._validate_region_spec(_spec_with(module, bad))


@pytest.mark.parametrize("module", [lv, pendulum])
def test_validator_rejects_an_action_only_region_that_starts_off_the_training_box(module):
    spec = module.regions()
    shifted = spec.training_state_box.copy()
    shifted[0] += 0.05  # overlaps, but is not the training box
    bad = OutRegion("out-large-action", "action", shifted, 2.0 * spec.training_action_interval)
    with pytest.raises(RegionSpecError, match="ADR-W4"):
        module._validate_region_spec(_spec_with(module, bad))


@pytest.mark.parametrize("module", [lv, pendulum])
def test_validator_accepts_a_both_axis_region_and_rejects_one_with_trained_kicks(module):
    real = module.regions().out_regions[0]  # the state-axis region: a disjoint box
    trained = module.regions().training_action_interval
    good = OutRegion("out-both", "both", real.state_box.copy(), 2.0 * trained)
    module._validate_region_spec(_spec_with(module, good))
    bad = OutRegion("out-both", "both", real.state_box.copy(), trained.copy())
    with pytest.raises(RegionSpecError, match="§7"):
        module._validate_region_spec(_spec_with(module, bad))
