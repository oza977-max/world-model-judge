"""wmj.harness.skeleton — the smallest honest end-to-end slice.

In plain words: this proves the whole pipe is connected — a world
that can be simulated, two baselines that can be trained from seeded
data, and a judge score computed from real predictions against real
outcomes — before any of the real pipeline's numbers (training set
size, evaluation trial count) exist. It is deliberately smaller than
the real run and writes a distinct, smaller report shape
(`wmj-skeleton/0`, not the pinned `wmj-verdict/1` Verdict) — see
build/prompts/P1-C02.md's "Scope decisions" for why every constant
below is what it is.

This module is throwaway by design: `wmj/harness/trials.py` (built at
P6-C01, cross-cutting ADR-004) is the real orchestration loop this one
previews, not extends.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from wmj.harness.serialize import canonical_serialize
from wmj.judge.skill import crps_gaussian, skill_score
from wmj.models.base import SeedSource, TrainingData, WorldContext
from wmj.models.baselines import linear_factory, persistence_factory
from wmj.worlds import lv

SKELETON_SEED = 20260825
N_TRAIN = 8
H_TRAIN = 50
N_EVAL = 20
REPORT_PATH = Path("out") / "wmj-skeleton" / "0.json"


def _build_context() -> WorldContext:
    region_spec = lv.regions()
    return WorldContext(
        world_name="lv",
        state_dim=lv.WORLD.d,
        action_dim=lv.WORLD.a,
        training_state_box=region_spec.training_state_box,
        training_action_interval=region_spec.training_action_interval,
        scale=lv.WORLD.scale,
    )


def _generate_training_data(seeds: SeedSource) -> TrainingData:
    """A small set of seeded, in-training-region trajectories."""
    rng = seeds.rng_for("lv", "training", "skeleton-train-starts")
    region_spec = lv.regions()
    low, high = region_spec.training_state_box[:, 0], region_spec.training_state_box[:, 1]

    states = np.zeros((N_TRAIN, H_TRAIN + 1, lv.WORLD.d))
    actions = np.zeros((N_TRAIN, H_TRAIN, lv.WORLD.a))
    states[:, 0, :] = rng.uniform(low, high, size=(N_TRAIN, lv.WORLD.d))
    for trajectory in range(N_TRAIN):
        state = states[trajectory, 0, :]
        for step in range(H_TRAIN):
            action = np.zeros(lv.WORLD.a)
            state = lv.transition(state, action)
            states[trajectory, step + 1, :] = state
            actions[trajectory, step, :] = action
    return TrainingData(states=states, actions=actions)


def _run_one_step_trials(model, seeds: SeedSource) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """N_EVAL one-step trials from seeded training-region starts.

    Returns (mean, spread, outcome), each shaped [N_EVAL, d], all
    already normalised by the world's scale vector (judge ADR-J1).
    """
    rng = seeds.rng_for("lv", "training", "skeleton-eval-starts")
    region_spec = lv.regions()
    low, high = region_spec.training_state_box[:, 0], region_spec.training_state_box[:, 1]
    starts = rng.uniform(low, high, size=(N_EVAL, lv.WORLD.d))

    means = np.zeros((N_EVAL, lv.WORLD.d))
    spreads = np.zeros((N_EVAL, lv.WORLD.d))
    outcomes = np.zeros((N_EVAL, lv.WORLD.d))
    null_action = np.zeros(lv.WORLD.a)
    for trial in range(N_EVAL):
        model.reset()
        state = starts[trial]
        prediction = model.predict(state, null_action)
        outcome = lv.transition(state, null_action)
        means[trial] = prediction.mean / lv.WORLD.scale
        spreads[trial] = prediction.spread / lv.WORLD.scale
        outcomes[trial] = outcome / lv.WORLD.scale
    return means, spreads, outcomes


def build_skeleton_report(seed: int = SKELETON_SEED) -> dict:
    """Build the skeleton report dict — pure, no file I/O."""
    seeds = SeedSource(run_seed=seed, my_name=None)
    ctx = _build_context()
    training = _generate_training_data(seeds)

    model = persistence_factory(ctx, seeds, training)
    baseline = linear_factory(ctx, seeds, training)

    model_mean, model_spread, model_outcome = _run_one_step_trials(model, seeds)
    baseline_mean, baseline_spread, baseline_outcome = _run_one_step_trials(
        baseline, seeds
    )

    crps_model = float(
        np.mean(crps_gaussian(model_mean, model_spread, model_outcome))
    )
    crps_baseline = float(
        np.mean(crps_gaussian(baseline_mean, baseline_spread, baseline_outcome))
    )

    return {
        "schema": "wmj-skeleton/0",
        "world": "lv",
        "seed": seed,
        "model": model.name,
        "baseline": baseline.name,
        "n_trials": N_EVAL,
        "crps_model": crps_model,
        "crps_baseline": crps_baseline,
        "skill_vs_linear": skill_score(crps_model, crps_baseline),
    }


def write_skeleton_report(seed: int = SKELETON_SEED) -> Path:
    """Build the report and write it via the canonical serializer."""
    report = build_skeleton_report(seed)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    # Write-then-replace so the report is never observed half-written
    # (code-review-001, Panel E).
    tmp = REPORT_PATH.with_name(REPORT_PATH.name + ".tmp")
    tmp.write_bytes(canonical_serialize(report))
    tmp.replace(REPORT_PATH)
    return REPORT_PATH
