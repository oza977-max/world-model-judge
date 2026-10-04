"""P3-C02 amendment (design-review-010): `stateless` and `predict_batch` on the baselines (TC-MU1-04).

In plain words: the harness predicts many runs at once for every model that has no
memory between steps. Persistence ("nothing changes") has none, so it can; linear
("whatever just happened keeps happening") remembers the previous state, so it must
never be batched. Every model that says it is stateless must give, row for row, exactly
the numbers it gives one at a time.
"""

from __future__ import annotations

import numpy as np
import pytest

from wmj.models.base import SeedSource, TrainingData, WorldContext
from wmj.models.baselines import linear_factory, persistence_factory


def _data():
    rng = np.random.default_rng(3)
    states = np.cumsum(rng.normal(0, 0.05, (4, 30, 2)), axis=1) + 3.0
    return TrainingData(states=states, actions=np.zeros((4, 29, 1)))


def _ctx():
    return WorldContext(
        world_name="x", state_dim=2, action_dim=1, training_state_box=np.zeros((2, 2)),
        training_action_interval=np.array([[-0.1, 0.1]]), scale=np.ones(2),
    )


def test_persistence_declares_itself_stateless_and_linear_does_not():
    persistence = persistence_factory(_ctx(), SeedSource(1, "persistence"), _data())
    linear = linear_factory(_ctx(), SeedSource(1, "linear"), _data())
    assert persistence.stateless is True
    assert linear.stateless is False
    assert not hasattr(linear, "predict_batch")  # a stateful model is never batched


@pytest.mark.parametrize("n", [1, 2, 7, 64, 200])
def test_persistence_batch_rows_are_bit_identical_to_one_at_a_time(n):
    model = persistence_factory(_ctx(), SeedSource(1, "persistence"), _data())
    rng = np.random.default_rng(n)
    s, a = rng.normal(3.0, 1.0, (n, 2)), rng.normal(0, 0.05, (n, 1))
    means, spreads = model.predict_batch(s, a)
    assert means.shape == spreads.shape == (n, 2)
    for i in range(n):
        p = model.predict(s[i], a[i])
        assert np.array_equal(means[i], p.mean) and np.array_equal(spreads[i], p.spread)


def test_persistence_batch_is_the_state_with_the_fitted_spread_on_every_row():
    model = persistence_factory(_ctx(), SeedSource(1, "persistence"), _data())
    s = np.array([[1.0, 2.0], [3.0, 4.0]])
    means, spreads = model.predict_batch(s, np.zeros((2, 1)))
    assert np.array_equal(means, s)
    assert np.array_equal(spreads, np.tile(model.predict(s[0], np.zeros(1)).spread, (2, 1)))
    means[0, 0] = 99.0  # the returned arrays are copies: the caller's states are untouched
    assert s[0, 0] == 1.0


def test_persistence_batch_refuses_bad_shapes_and_non_finite_states():
    model = persistence_factory(_ctx(), SeedSource(1, "persistence"), _data())
    from wmj.errors import WmjError

    with pytest.raises(WmjError):
        model.predict_batch(np.zeros((3, 5)), np.zeros((3, 1)))
    with pytest.raises(WmjError):
        model.predict_batch(np.zeros((3, 2)), np.zeros((4, 1)))
    with pytest.raises(WmjError):
        model.predict_batch(np.array([[np.nan, 1.0]]), np.zeros((1, 1)))
