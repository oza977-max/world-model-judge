"""The divergence curve under kicks, and its batched form (worlds ADR-W3).

In plain words: the "how fast does the world drift away from itself"
line is now measured with the same kind of kicks the graded trials get,
applied identically to a trajectory and its hair-perturbed twin — so any
separation is still the world's own doing, not the kick's. And because
the benchmark now steps all its starts at once, the batched curve must
equal the one-start-at-a-time curve exactly (design-review-010).
"""

from __future__ import annotations

import numpy as np
import pytest

from wmj.worlds import lv, pendulum
from wmj.worlds.actions import kick_sequence
from wmj.worlds.base import distance
from wmj.worlds.divergence import (
    DivergenceInputError,
    median_separation_curve,
    perturb,
    separation_curve,
    separation_curves_batch,
)


def _kicks(module, horizon: int, n: int, seed: int, band: str = "in", p_step: float = 0.05):
    rng = np.random.default_rng(seed)
    umax = float(module.regions().training_action_interval[0, 1])
    return np.stack(
        [kick_sequence(rng, horizon=horizon, p_step=p_step, band=band, umax=umax) for _ in range(n)]
    )  # [n, horizon, 1]


def test_kicked_separation_curve_applies_the_same_kick_to_both_twins():
    start = np.array([4.0, 2.0])
    horizon = 40
    kicks = _kicks(lv, horizon, 1, seed=3, p_step=0.5)[0]
    curve = separation_curve(lv.transition, start, horizon, lv.SCALE, 1e-6, actions=kicks)
    base, twin = start.copy(), perturb(start, 1e-6)
    expected = [distance(base, twin, lv.SCALE)]
    for t in range(horizon):
        base = lv.transition(base, kicks[t])
        twin = lv.transition(twin, kicks[t])
        expected.append(distance(base, twin, lv.SCALE))
    assert np.array_equal(curve, np.array(expected))


def test_separation_curve_without_actions_is_the_null_action_curve():
    start = np.array([4.0, 2.0])
    null = np.zeros((30, 1))
    assert np.array_equal(
        separation_curve(lv.transition, start, 30, lv.SCALE, 1e-6),
        separation_curve(lv.transition, start, 30, lv.SCALE, 1e-6, actions=null),
    )


def test_kicks_change_the_trajectory_but_the_twins_still_start_a_hair_apart():
    start = np.array([4.0, 2.0])
    kicks = _kicks(lv, 60, 1, seed=4, p_step=0.5)[0]
    kicked = separation_curve(lv.transition, start, 60, lv.SCALE, 1e-6, actions=kicks)
    null = separation_curve(lv.transition, start, 60, lv.SCALE, 1e-6)
    assert kicked[0] == null[0]
    assert not np.array_equal(kicked, null)


@pytest.mark.parametrize(("module", "n"), [(lv, 1), (lv, 7), (pendulum, 2), (pendulum, 7)])
def test_batched_separation_curves_equal_the_per_start_curves_exactly(module, n):
    spec = module.regions()
    rng = np.random.default_rng(10 + n)
    starts = rng.uniform(spec.training_state_box[:, 0], spec.training_state_box[:, 1],
                         size=(n, spec.training_state_box.shape[0]))
    horizon = 80
    kicks = _kicks(module, horizon, n, seed=20 + n, p_step=0.2)
    batched = separation_curves_batch(module.transition_batch, starts, horizon, module.SCALE,
                                      1e-6, actions=kicks)
    assert batched.shape == (n, horizon + 1)
    for i in range(n):
        single = separation_curve(module.transition, starts[i], horizon, module.SCALE, 1e-6,
                                  actions=kicks[i])
        assert np.array_equal(batched[i], single), f"start {i} of {n} differs"


def test_batched_separation_curves_refuse_mismatched_action_shapes():
    starts = np.array([[4.0, 2.0], [3.0, 2.0]])
    with pytest.raises(DivergenceInputError, match="ADR-W3"):
        separation_curves_batch(lv.transition_batch, starts, 10, lv.SCALE, 1e-6,
                                actions=np.zeros((2, 9, 1)))
    with pytest.raises(DivergenceInputError, match="ADR-W3"):
        separation_curve(lv.transition, starts[0], 10, lv.SCALE, 1e-6, actions=np.zeros((9, 1)))


def test_tc_wd4_01_lv_curve_stays_bounded_under_kicks():
    # TC-WD4-01 re-run with kicks (design-review-010): the two-sided band
    # still holds. Executed before pinning at full scale: ratio 0.92.
    spec = lv.regions()
    rng = np.random.default_rng(0)
    starts = rng.uniform(spec.training_state_box[:, 0], spec.training_state_box[:, 1], size=(8, 2))
    kicks = _kicks(lv, 700, 8, seed=1, p_step=0.01)
    med = median_separation_curve(
        separation_curves_batch(lv.transition_batch, starts, 700, lv.SCALE, 1e-6, actions=kicks)
    )
    ratio = med[-1] / med[0]
    assert 0.1 < ratio < 10.0, f"TC-WD4-01 (kicked): ratio {ratio:.3g} outside (0.1, 10)"


def test_tc_wd4_02_pendulum_regime_difference_survives_kicks():
    spec = pendulum.regions()
    rng = np.random.default_rng(0)
    horizon, half = 600, 300
    inverted_box = {r.region_name: r for r in spec.out_regions}["out-near-inverted"].state_box

    def med(box, seed):
        starts = rng.uniform(box[:, 0], box[:, 1], size=(8, 4))
        kicks = _kicks(pendulum, horizon, 8, seed=seed, p_step=0.002)
        return median_separation_curve(
            separation_curves_batch(pendulum.transition_batch, starts, horizon, pendulum.SCALE,
                                    1e-6, actions=kicks)
        )

    assert med(inverted_box, 2)[half] / med(spec.training_state_box, 3)[half] >= 5.0


@pytest.mark.parametrize(("module", "n"), [(lv, 1), (lv, 7), (pendulum, 7)])
def test_batched_conserved_drift_equals_the_per_start_drift_exactly(module, n):
    from wmj.worlds.divergence import conserved_drift, conserved_drift_batch

    spec = module.regions()
    rng = np.random.default_rng(40 + n)
    starts = rng.uniform(spec.training_state_box[:, 0], spec.training_state_box[:, 1],
                         size=(n, spec.training_state_box.shape[0]))
    worst, initial = conserved_drift_batch(module.transition_batch, module.conserved, starts, 60)
    for i in range(n):
        w, init = conserved_drift(module.transition, module.conserved, starts[i], 60)
        assert worst[i] == w and initial[i] == init
