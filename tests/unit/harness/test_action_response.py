"""Tests for wmj.harness.action_response — the MU-3 action-response check (TC-MU3-04).

In plain words: the check hands a model one starting state and two different actions
from the range it was trained on and asks "did your answer move?". These tests give it
stand-in models whose behaviour is known exactly — one that ignores the action, one that
uses it a lot, one that uses it by a hair, one that only changes its error bar, one that
reacts at a single probe only, one with memory that *looks* responsive but isn't — and
check each is judged correctly. Then the real thing: Model A passes, the action-blind
fixture is flagged, on both worlds, with the probes identical for every model.
"""

from __future__ import annotations

import numpy as np
import pytest

from wmj.harness.action_response import (
    ACTION_RESPONSE_TOLERANCE,
    PROBE_STATES,
    ActionResponseError,
    action_response_probes,
    check_action_response,
)
from wmj.harness.training import TrainingRecipe, build_training_data, make_world_context
from wmj.models.base import Prediction, SeedSource, WorldContext
from wmj.models.registry import all_models
from wmj.worlds import lv, pendulum

SEED = 20260825


def _ctx(action_dim=1, scale=(2.0, 5.0)):
    return WorldContext(
        world_name="stub", state_dim=2, action_dim=action_dim,
        training_state_box=np.array([[1.0, 3.0], [-4.0, 4.0]]),
        training_action_interval=np.array([[-0.5, 0.5]] * action_dim),
        scale=np.array(scale),
    )


class Stub:
    """A model with a chosen response: mean = state + gain * action[0]; spread constant."""

    def __init__(self, gain=0.0, spread_gain=0.0, only_if=None):
        self.gain, self.spread_gain, self.only_if = gain, spread_gain, only_if

    def reset(self):
        pass

    def predict(self, state, action):
        responds = True if self.only_if is None else self.only_if(state, action)
        a = float(action[0]) if responds else 0.0
        return Prediction(
            mean=state + self.gain * a * np.ones_like(state),
            spread=np.ones_like(state) * (1.0 + self.spread_gain * a),
        )


# --- the probes -------------------------------------------------------------------------


def test_the_probes_are_in_the_training_region_with_the_ends_included_and_seeded():
    ctx = _ctx()
    states, levels, pairs = action_response_probes(ctx, SEED)
    box, interval = ctx.training_state_box, ctx.training_action_interval
    assert states.shape == (PROBE_STATES, 2)
    assert np.all(states >= box[:, 0]) and np.all(states <= box[:, 1])
    assert levels.tolist() == [[-0.5], [0.0], [0.5]]  # low end, middle, high end — closed ends
    assert pairs.tolist() == [[0, 2], [0, 1], [1, 2]]
    again = action_response_probes(ctx, SEED)
    assert np.array_equal(states, again[0])
    other = action_response_probes(ctx, SEED + 1)[0]
    assert not np.array_equal(states, other)
    assert action_response_probes(ctx, SEED, 5)[0].shape == (5, 2)
    assert np.array_equal(action_response_probes(ctx, SEED, 5)[0], states[:5])  # a prefix: stable stream
    assert interval.shape == (1, 2)
    # known answers: the stream name and the way states are drawn are part of the recorded probes
    assert states[:3].tolist() == [
        [1.660340344706178, -0.11388843145846383],
        [1.971374566056359, -0.7516906576621993],
        [2.928286599785886, 2.1018048056809615],
    ]


@pytest.mark.parametrize("bad", [0, -1, 2.5, True, None])
def test_the_number_of_probe_states_must_be_a_positive_integer(bad):
    with pytest.raises(ActionResponseError, match="n_states"):
        action_response_probes(_ctx(), SEED, bad)


def test_the_probes_do_not_depend_on_any_model():
    ctx = _ctx()
    a = action_response_probes(ctx, SEED)
    check_action_response(Stub(gain=1.0), ctx, SEED)
    b = action_response_probes(ctx, SEED)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))


# --- the rule: at least one probe, more than floating-point noise -------------------------


def test_a_model_that_ignores_the_action_is_flagged_action_blind():
    result = check_action_response(Stub(gain=0.0), _ctx(), SEED)
    assert result.action_blind is True
    assert result.n_responding == 0 and result.largest_change == 0.0
    assert result.n_probes == 3 * PROBE_STATES


def test_a_model_that_uses_the_action_passes_and_reports_how_much():
    result = check_action_response(Stub(gain=1.0), _ctx(), SEED)
    assert result.action_blind is False and result.n_responding == result.n_probes
    # largest move: (low, high) pair, action gap 1.0, divided by the smallest scale 2.0
    assert result.largest_change == pytest.approx(1.0 / 2.0)
    assert result.tolerance == ACTION_RESPONSE_TOLERANCE == 1e-9


def test_a_change_below_floating_point_noise_does_not_count_but_a_small_real_one_does():
    below = check_action_response(Stub(gain=1e-12), _ctx(), SEED)  # 5e-13 of the scale
    assert below.action_blind is True and below.largest_change > 0.0
    above = check_action_response(Stub(gain=1e-7), _ctx(), SEED)  # 5e-8 of the scale
    assert above.action_blind is False


def test_the_tolerance_is_in_units_of_the_worlds_own_scale():
    # the same absolute movement is noise in a big-scale world and real in a small-scale one
    gain = 2e-9  # action gap 1.0 -> a 2e-9 movement
    assert check_action_response(Stub(gain=gain), _ctx(scale=(1000.0, 1000.0)), SEED).action_blind is True
    assert check_action_response(Stub(gain=gain), _ctx(scale=(0.1, 0.1)), SEED).action_blind is False


def test_a_model_that_only_changes_its_error_bar_still_uses_the_action():
    assert check_action_response(Stub(spread_gain=1.0), _ctx(), SEED).action_blind is False


def test_one_responding_probe_out_of_many_is_enough():
    ctx = _ctx()
    states, _, _ = action_response_probes(ctx, SEED)
    only_here = states[7]

    def at_one_state(state, action):
        return bool(np.array_equal(state, only_here))

    result = check_action_response(Stub(gain=1.0, only_if=at_one_state), ctx, SEED)
    assert result.action_blind is False and result.n_responding == 3  # that state's three pairs


def test_a_single_responding_probe_is_enough_to_pass():
    """Exactly one of the 48 probes moves by more than the tolerance (the other two pairs of
    that state move by less) — the model is *not* action-blind: the rule is 'at least one'."""
    ctx = _ctx(scale=(2.0, 2.0))
    states, _, _ = action_response_probes(ctx, SEED)
    here = states[4]
    nudge = {-0.5: 0.0, 0.0: 0.8e-9 * 2.0, 0.5: 1.5e-9 * 2.0}  # pair gaps: 1.5, 0.8, 0.7 (x 1e-9 scale)

    class Straddle:
        def reset(self):
            pass

        def predict(self, state, action):
            move = nudge[float(action[0])] if np.array_equal(state, here) else 0.0
            return Prediction(mean=state + move, spread=np.ones_like(state))

    result = check_action_response(Straddle(), ctx, SEED)
    assert result.n_responding == 1 and result.action_blind is False
    assert result.largest_change == pytest.approx(1.5e-9, rel=1e-6)


def test_a_response_confined_to_one_end_of_the_range_is_still_seen():
    only_high = lambda state, action: bool(action[0] >= 0.5)
    result = check_action_response(Stub(gain=1.0, only_if=only_high), _ctx(), SEED)
    assert result.action_blind is False and result.n_responding == 2 * PROBE_STATES  # (low,high),(mid,high)


def test_a_model_that_depends_on_the_state_but_not_the_action_is_still_blind():
    class StateOnly:
        def reset(self):
            pass

        def predict(self, state, action):
            return Prediction(mean=np.sin(state) * 3.0, spread=np.abs(state) + 1.0)

    assert check_action_response(StateOnly(), _ctx(), SEED).action_blind is True


def test_memory_cannot_pass_for_action_response_because_the_model_is_reset_every_time():
    class Drifting:
        """Ignores its action but changes its answer on every call, until reset."""

        def __init__(self):
            self.calls = 0

        def reset(self):
            self.calls = 0

        def predict(self, state, action):
            self.calls += 1
            return Prediction(mean=state + 0.01 * self.calls, spread=np.ones_like(state))

    model = Drifting()
    assert check_action_response(model, _ctx(), SEED).action_blind is True

    class NoReset(Drifting):
        def reset(self):  # a model whose reset does nothing would be caught only by this being called
            raise AssertionError("never reached")

    with pytest.raises(AssertionError):
        check_action_response(NoReset(), _ctx(), SEED)  # proves the check really calls reset()


def test_a_non_finite_prediction_is_an_error_not_identical():
    class Broken:
        def reset(self):
            pass

        def predict(self, state, action):
            return Prediction(mean=np.full_like(state, np.nan), spread=np.ones_like(state))

    with pytest.raises(ActionResponseError, match="not a finite number"):
        check_action_response(Broken(), _ctx(), SEED)

    class BrokenSpread(Broken):
        def predict(self, state, action):
            return Prediction(mean=state, spread=np.full_like(state, np.inf))

    with pytest.raises(ActionResponseError, match="not a finite number"):
        check_action_response(BrokenSpread(), _ctx(), SEED)


@pytest.mark.parametrize("bad", [-1e-9, float("nan"), float("inf"), True, "1e-9", None])
def test_the_tolerance_must_be_a_finite_non_negative_number(bad):
    with pytest.raises(ActionResponseError, match="tolerance"):
        check_action_response(Stub(gain=1.0), _ctx(), SEED, tolerance=bad)


def test_a_tolerance_of_zero_flags_only_exactly_identical_predictions():
    assert check_action_response(Stub(spread_gain=1e-12), _ctx(), SEED, tolerance=0.0).action_blind is False
    assert check_action_response(Stub(gain=0.0), _ctx(), SEED, tolerance=0.0).action_blind is True


def test_a_two_dimensional_action_where_only_the_second_component_matters_is_seen():
    class SecondOnly:
        def reset(self):
            pass

        def predict(self, state, action):
            return Prediction(mean=state + action[1], spread=np.ones_like(state))

    ctx = _ctx(action_dim=2)
    assert check_action_response(SecondOnly(), ctx, SEED).action_blind is False


# --- the real thing: Model A passes, the action-blind fixture is flagged -------------------

LV_RECIPE = TrainingRecipe(
    training_trajectories=100, subsample_pairs=2000, kick_pairs=50, heldout_pairs=500,
    gradcheck_pairs=16,
)
PEND_RECIPE = TrainingRecipe(
    training_trajectories=300, subsample_pairs=2000, kick_pairs=60, heldout_pairs=500,
    gradcheck_pairs=16,
)


@pytest.fixture(scope="module", params=["lv", "pendulum"])
def trained(request):
    import wmj.models.baselines
    import wmj.models.direct
    import wmj.models.ensemble
    import wmj.models.fixtures  # noqa: F401  (register every model)

    name, world, recipe, horizon = (
        ("lv", lv.WORLD, LV_RECIPE, 100) if request.param == "lv" else ("pendulum", pendulum.WORLD, PEND_RECIPE, 300)
    )
    ctx = make_world_context(name, world)
    data = build_training_data(name, world, SeedSource(SEED, None), recipe, horizon=horizon)
    return ctx, {n: f(ctx, SeedSource(SEED, n), data) for n, f in all_models().items()}


def test_tc_mu3_04_the_action_blind_fixture_is_flagged_and_the_responsive_model_passes(trained):
    ctx, models = trained
    blind = check_action_response(models["fx-action-blind"], ctx, SEED)
    assert blind.action_blind is True and blind.n_responding == 0 and blind.largest_change == 0.0
    responsive = check_action_response(models["direct"], ctx, SEED)
    assert responsive.action_blind is False
    assert responsive.largest_change > 1e3 * ACTION_RESPONSE_TOLERANCE


def test_the_other_fixtures_that_wrap_model_a_respond_to_the_action_too(trained):
    ctx, models = trained
    for name in ("fx-overconfident", "fx-honest-rough", "fx-brittle", "ensemble"):
        assert check_action_response(models[name], ctx, SEED).action_blind is False, name


def test_the_persistence_baseline_is_flagged_because_it_ignores_the_action_by_construction(trained):
    ctx, models = trained
    assert check_action_response(models["persistence"], ctx, SEED).action_blind is True


# --- pinned details (review pass 1) -----------------------------------------------------


def test_sixteen_states_and_forty_eight_probes_are_the_recorded_design():
    assert PROBE_STATES == 16
    assert check_action_response(Stub(gain=1.0), _ctx(), SEED).n_probes == 48


def test_a_model_already_carrying_memory_is_reset_before_it_is_asked_anything():
    from wmj.models.baselines import LinearModel

    linear = LinearModel(np.ones(2))
    linear.predict(np.array([1.0, 1.0]), np.zeros(1))
    linear.predict(np.array([2.0, 5.0]), np.zeros(1))  # memory left over from an earlier rollout
    result = check_action_response(linear, _ctx(), SEED)
    assert result.action_blind is True and result.largest_change == 0.0


def test_each_dimension_is_judged_against_its_own_scale_and_the_largest_change_is_reported():
    class SecondQuantity:
        def reset(self):
            pass

        def predict(self, state, action):
            move = np.array([0.0, 1e-6 * action[0]])  # only the second quantity (scale 5.0) reacts
            return Prediction(mean=state + move, spread=np.ones_like(state))

    r = check_action_response(SecondQuantity(), _ctx(scale=(2.0, 5.0)), SEED)
    assert r.largest_change == pytest.approx(1e-6 / 5.0) and r.action_blind is False


def test_the_error_bar_movement_is_measured_per_dimension_against_scale_and_takes_the_largest():
    class SpreadOnly:
        def reset(self):
            pass

        def predict(self, state, action):
            return Prediction(mean=state, spread=np.array([1.0 + 2e-5 * action[0], 1.0 + 1e-5 * action[0]]))

    r = check_action_response(SpreadOnly(), _ctx(scale=(2.0, 5.0)), SEED)
    assert r.largest_change == pytest.approx(2e-5 / 2.0)  # the larger of 1e-5 and 2e-6, not their sum


def test_a_custom_tolerance_is_reported_back_and_applied():
    r = check_action_response(Stub(gain=1e-3), _ctx(), SEED, tolerance=0.5)
    assert r.tolerance == 0.5 and r.action_blind is True  # 1e-3/2 is below 0.5 of scale


def test_a_prediction_with_one_non_finite_entry_is_an_error():
    for bad_mean, bad_spread in ((np.array([1.0, np.nan]), np.ones(2)), (np.zeros(2), np.array([np.inf, 1.0]))):
        class OneBad:
            def reset(self):
                pass

            def predict(self, state, action, _m=bad_mean, _s=bad_spread):
                return Prediction(mean=_m, spread=_s)

        with pytest.raises(ActionResponseError, match="not a finite number"):
            check_action_response(OneBad(), _ctx(), SEED)


def test_a_prediction_of_the_wrong_shape_is_an_error_not_a_quiet_verdict():
    class Scalar:
        def reset(self):
            pass

        def predict(self, state, action):
            return Prediction(mean=np.array(0.0), spread=np.array(1.0))

    with pytest.raises(ActionResponseError, match="one number per quantity"):
        check_action_response(Scalar(), _ctx(), SEED)

    class WrongSpreadOnly(Scalar):  # a right-shaped guess but a wrong-shaped error bar
        def predict(self, state, action):
            return Prediction(mean=state, spread=np.array(1.0))

    class WrongMeanOnly(Scalar):
        def predict(self, state, action):
            return Prediction(mean=np.array(0.0), spread=np.ones_like(state))

    for cls in (WrongSpreadOnly, WrongMeanOnly):
        with pytest.raises(ActionResponseError, match="one number per quantity"):
            check_action_response(cls(), _ctx(), SEED)


@pytest.mark.parametrize("scale", [(np.nan, 1.0), (-1.0, 1.0), (0.0, 1.0), (np.inf, 1.0)])
def test_a_world_scale_that_is_not_finite_and_positive_is_refused(scale):
    with pytest.raises(ActionResponseError, match="scale"):
        check_action_response(Stub(gain=1.0), _ctx(scale=scale), SEED)


def test_an_even_response_is_not_action_blind():
    """a response like action² is identical at the two ends but differs at the middle: it uses the action."""

    class Even:
        def reset(self):
            pass

        def predict(self, state, action):
            return Prediction(mean=state + action[0] ** 2, spread=np.ones_like(state))

    assert check_action_response(Even(), _ctx(), SEED).action_blind is False
