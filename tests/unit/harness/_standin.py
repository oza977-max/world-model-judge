"""A stand-in roster and small run sizes for tests of the orchestration loop (P6-C01a).

In plain words: the two real learned models (`direct`, `ensemble`) may not be measured before the
pre-registration is frozen, and the four fixtures are built on `direct`'s trained core, so they may not be
either (backlog A42 item 2). These tests therefore run the real loop on a roster that contains:
- the two real baselines;
- the real fixture classes, wrapped round a stand-in core that is not a trained network;
- one small test-only "honest" model.
Nothing here describes the real `direct` model.
"""

from __future__ import annotations

import numpy as np

from wmj.harness import derive_thresholds
from wmj.harness.training import TrainingRecipe
from wmj.models.base import Prediction
from wmj.models.baselines import linear_factory, persistence_factory
from wmj.models.fixtures import FxActionBlind, FxBrittle, FxOverconfident

N_SMALL = 8
H_SMALL = 12
META = {"seed": 11, "platform": "test-platform", "prereg_commit": "stand-in-no-freeze", "thresholds_file": "stand-in"}
RECIPE = TrainingRecipe(training_trajectories=40, subsample_pairs=400, kick_pairs=1, heldout_pairs=50, gradcheck_pairs=4)


def small_thresholds(n: int = N_SMALL) -> dict:
    """The real thresholds file's contents, with bands derived for `n` trials instead of 200."""
    raw = derive_thresholds.build_thresholds()
    raw["bands"] = derive_thresholds.binomial_bands(n=n)
    return raw


class StandInCore:
    """Not a trained network: the next state is the current one, nudged by the action. Same call shape as
    Model A's core (`predict_batch`), so the real fixture wrappers can be built round it."""

    def __init__(self, ctx) -> None:
        self._scale = np.asarray(ctx.scale, dtype=float)

    def predict_batch(self, states, actions):
        states = np.asarray(states, dtype=float)
        actions = np.asarray(actions, dtype=float)
        mean = states + 0.05 * self._scale * actions[:, :1]
        spread = np.broadcast_to(0.02 * self._scale, states.shape).copy()
        return mean, spread


class StandInModel:
    """A test-only model that is neither a baseline nor a fixture."""

    is_fixture = False
    is_baseline = False
    stateless = True

    def __init__(self, ctx) -> None:
        self._core = StandInCore(ctx)

    def reset(self) -> None:
        """Stateless."""

    def predict_batch(self, states, actions):
        return self._core.predict_batch(states, actions)

    def predict(self, state, action) -> Prediction:
        mean, spread = self.predict_batch(np.asarray(state)[None, :], np.asarray(action)[None, :])
        return Prediction(mean=mean[0].copy(), spread=spread[0].copy())


def standin_roster() -> dict:
    return {
        "fx-action-blind": lambda ctx, seeds, training: FxActionBlind(ctx, StandInCore(ctx)),
        "fx-brittle": lambda ctx, seeds, training: FxBrittle(ctx, StandInCore(ctx)),
        "fx-overconfident": lambda ctx, seeds, training: FxOverconfident(ctx, StandInCore(ctx)),
        "linear": linear_factory,
        "persistence": persistence_factory,
        "standin": lambda ctx, seeds, training: StandInModel(ctx),
    }


def small_config(**overrides):
    from wmj.harness.trials import RunConfig

    values = dict(
        run_seed=META["seed"], recipe=RECIPE, thresholds=small_thresholds(), n_trials=N_SMALL, horizon=H_SMALL,
        training_horizon=60, divergence_starts=4, climatology_starts=4, climatology_steps=400, climatology_bins=4,
        climatology_min_samples=20, probe_states=4,
    )
    values.update(overrides)
    return RunConfig(**values)
