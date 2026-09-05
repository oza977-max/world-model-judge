"""wmj.harness.preview — the stand-in producer behind `wmj chart-preview`.

In plain words: the real error-against-horizon numbers are the judge's
to compute (JU-3, built at P4-C05). This module exists so the first
real chart can be drawn twelve chunks earlier: it computes the same
block, in exactly the judge's shape (judge spec §5), for one model
(persistence) on one world (the predator-prey world), from real
trajectories. When the judge's block exists, the renderer does not
change — only the producer is swapped out.

What it does, per declared region of the LV world: draw seeded start
states, roll the truth forward under null actions, hold the start
constant as persistence's forecast ("nothing changes"), measure the
normalised distance between them at every step (the shared metric,
worlds ADR-W3), and take the median across trials. The world's own
divergence curve (P2-C03) is attached as the reference line.

It also runs the CRPS/skill path (judge ADR-J1) over real one-step
trials — persistence against linear — and hands the number back for
the command's summary line. It writes nothing itself: the files under
`out/` are written by `wmj.reporting` (design-review-008 C8).

`wmj chart-preview` is internal-only (reporting ADR-R5): not part of
the public `run`/`verify` pair, and not in the README.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from wmj.errors import WmjError
from wmj.harness.benchmarks import (
    DEFAULT_N_STARTS,
    build_divergence_artefact,
    declared_regions,
    sample_region_starts,
)
from wmj.judge.skill import crps_gaussian, skill_score
from wmj.models.base import SeedSource, TrainingData, WorldContext
from wmj.models.baselines import linear_factory, persistence_factory
from wmj.reporting.captions import CHART2_SCOPED, write_caption
from wmj.reporting.horizon_plot import render_horizon_chart
from wmj.worlds import lv
from wmj.worlds.base import distance

DEFAULT_SEED = 20260825
DEFAULT_N_TRIALS = 200  # evaluation trials per region (judge ADR-J4)
DEFAULT_HORIZON = lv.HORIZON
N_TRAIN = 8  # training trajectories for the baselines' spread fits
H_TRAIN = 50
CHART_STEM = "lv-persistence-horizon"
MODEL_LABEL = "persistence"


class PreviewArgumentError(WmjError):
    """Raised when a chart-preview size argument cannot produce a chart.

    A bare CLI integer reaches a division and a median here; `0` would
    surface as a contextless ZeroDivisionError far from its cause. The
    project's convention is that every refusal names what failed and
    why (cross-cutting Error-Handling rule 2; code-review-001 I6).
    """


def _require_positive(name: str, value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise PreviewArgumentError(
            f"chart-preview: --{name.replace('_', '-')} must be an integer >= 1, got {value!r} "
            f"(a run with no trials, no starts, or no steps has nothing to plot)"
        )


def _build_context() -> WorldContext:
    spec = lv.regions()
    return WorldContext(
        world_name="lv",
        state_dim=lv.WORLD.d,
        action_dim=lv.WORLD.a,
        training_state_box=spec.training_state_box,
        training_action_interval=spec.training_action_interval,
        scale=lv.WORLD.scale,
    )


def _rollout_truth(start: np.ndarray, horizon: int) -> np.ndarray:
    """`float64[H+1, d]`: the true null-action trajectory from `start`."""
    null_action = np.zeros(lv.WORLD.a)
    states = np.zeros((horizon + 1, lv.WORLD.d))
    states[0] = start
    for step in range(horizon):
        states[step + 1] = lv.transition(states[step], null_action)
    return states


def _generate_training_data(seeds: SeedSource) -> TrainingData:
    """Seeded in-training-region null-action trajectories (ADR-002 "train-starts")."""
    rng = seeds.rng_for("lv", "training", "train-starts")
    starts = sample_region_starts(rng, lv.regions().training_state_box, N_TRAIN)
    states = np.stack([_rollout_truth(start, H_TRAIN) for start in starts])
    actions = np.zeros((N_TRAIN, H_TRAIN, lv.WORLD.a))
    return TrainingData(states=states, actions=actions)


def build_lv_persistence_error_vs_horizon(
    seeds: SeedSource, n_starts: int, n_trials: int, horizon: int
) -> dict:
    """The judge-shaped `error_vs_horizon` block for persistence on LV.

    `median_error[0]` is exactly 0.0 by construction (the start *is*
    the forecast at step 0), matching judge §5's shared step-zero
    origin with the divergence artefact.
    """
    _require_positive("n_starts", n_starts)
    _require_positive("n_trials", n_trials)
    _require_positive("horizon", horizon)
    artefact = build_divergence_artefact("lv", lv.WORLD, seeds, n_starts=n_starts, horizon=horizon)
    per_region = []
    for region_name, box in declared_regions(lv.WORLD):
        rng = seeds.rng_for("lv", region_name, "eval-starts")
        starts = sample_region_starts(rng, box, n_trials)
        errors = np.zeros((n_trials, horizon + 1))
        for trial, start in enumerate(starts):
            truth = _rollout_truth(start, horizon)
            errors[trial] = [distance(truth[step], start, lv.SCALE) for step in range(horizon + 1)]
        per_region.append(
            {
                "region": region_name,
                "steps": list(range(horizon + 1)),
                "median_error": np.median(errors, axis=0).tolist(),
                "divergence_reference": list(artefact["regions"][region_name]["median_separation"]),
            }
        )
    return {"dt": lv.DT, "per_region": per_region}


@dataclass(frozen=True)
class OneStepSkill:
    crps_persistence: float
    crps_linear: float
    skill: float  # persistence's skill measured against linear (judge ADR-J1)


def one_step_skill_persistence_vs_linear(seeds: SeedSource, n_trials: int) -> OneStepSkill:
    """Mean one-step CRPS of both baselines on shared seeded trials, and the skill."""
    _require_positive("n_trials", n_trials)
    ctx = _build_context()
    training = _generate_training_data(seeds)
    persistence = persistence_factory(ctx, seeds, training)
    linear = linear_factory(ctx, seeds, training)

    rng = seeds.rng_for("lv", "training", "one-step-eval-starts")
    starts = sample_region_starts(rng, lv.regions().training_state_box, n_trials)
    null_action = np.zeros(lv.WORLD.a)
    scale = lv.WORLD.scale

    def mean_crps(model) -> float:
        total = 0.0
        for start in starts:
            model.reset()
            prediction = model.predict(start, null_action)
            outcome = lv.transition(start, null_action)
            total += float(
                np.mean(
                    crps_gaussian(prediction.mean / scale, prediction.spread / scale, outcome / scale)
                )
            )
        return total / len(starts)

    crps_p = mean_crps(persistence)
    crps_l = mean_crps(linear)
    return OneStepSkill(crps_persistence=crps_p, crps_linear=crps_l, skill=skill_score(crps_p, crps_l))


@dataclass(frozen=True)
class PreviewResult:
    png: Path
    svg: Path
    caption: Path
    one_step: OneStepSkill


def run_chart_preview(
    n_starts: int = DEFAULT_N_STARTS,
    n_trials: int = DEFAULT_N_TRIALS,
    horizon: int = DEFAULT_HORIZON,
    seed: int = DEFAULT_SEED,
    out_dir: Path = Path("out"),
) -> PreviewResult:
    """Compute the block, hand it to reporting, return where things went."""
    seeds = SeedSource(run_seed=seed, my_name=None)
    block = build_lv_persistence_error_vs_horizon(seeds, n_starts, n_trials, horizon)
    out_dir = Path(out_dir)
    png = out_dir / "charts" / f"{CHART_STEM}.png"
    svg = out_dir / "charts" / f"{CHART_STEM}.svg"
    caption = out_dir / "captions" / f"{CHART_STEM}.txt"
    render_horizon_chart(block, MODEL_LABEL, png, svg, is_fixture=False)
    write_caption(caption, CHART2_SCOPED)
    return PreviewResult(png=png, svg=svg, caption=caption,
                         one_step=one_step_skill_persistence_vs_linear(seeds, n_trials))
