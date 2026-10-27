"""wmj.harness.trials — the orchestration loop that runs every model on every world (cross-cutting ADR-004).

In plain words: this is the exam. For each world, in a fixed order, it:
1. makes the world's training data once, and hands the same data to every model;
2. draws one set of test starting points per region, the same for every model, so models are compared on
   the very same trials (judge ADR-J4), and steps the true world once from each, with no kick: grading
   starts after that step, and every model is first shown it, so a model that extrapolates has a direction
   to go on (the warm start, owner decision 2026-10-27). No drawn point or start may be a training start (MU-7);
3. runs the true world forward from those starts, with each trial's own seeded kicks;
4. runs the two simple baselines first, because every other model is scored against them (MU-2);
5. runs each model from the same starts with the same kicks. A model is given the true start once; after
   that it is fed its own previous guess, so its small errors pile up, which is what the judge measures (the
   requirements: "feeds its own guesses back into itself"). It is advanced exactly once per true step (WD-3);
6. converts everything into the judge's units (divided by the world's scale, backlog A30) and calls the
   judge once per model and world, with every region inside that one call;
7. wraps each verdict in its envelope: the model's name, its fixture label, the run facts (`meta`, passed
   in), and the result of the action-lever check (MU-3).

It writes nothing: the writer (`wmj.reporting.writer`) does, and `wmj run` joins the two.

Models are found by scanning the `wmj.models` package (owner decision D7: discovery lives here, so the models
package keeps its narrow import rules). The worlds are a fixed pair named here, a disclosed asymmetry
(ADR-004): adding a world is not a one-file change.
"""

from __future__ import annotations

import importlib
import pkgutil
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

import wmj.models
from wmj.errors import WmjError
from wmj.harness.action_response import PROBE_STATES, check_action_response
from wmj.harness.benchmarks import (
    DEFAULT_N_STARTS,
    build_divergence_artefact,
    declared_regions,
    sample_region_starts,
)
from wmj.harness.climatology import (
    DEFAULT_N_STARTS as CLIMATOLOGY_STARTS,
)
from wmj.harness.climatology import (
    DEFAULT_STEPS_PER_START,
    MIN_SAMPLES_PER_BIN,
    N_BINS,
    build_climatology_table,
    judge_tables,
    measure_invariant_bins,
)
from wmj.harness.kicks import seeded_kick_sequences
from wmj.harness.regions import label_trial
from wmj.harness.training import (
    TrainingRecipe,
    assert_eval_starts_disjoint,
    build_training_data,
    make_world_context,
)
from wmj.judge.calibration import compute_calibration
from wmj.judge.climatology import compute_climatology
from wmj.judge.exceptions import compute_exceptions_and_trials
from wmj.judge.horizon import compute_error_vs_horizon, compute_trust_horizons
from wmj.judge.sharpness import compute_sharpness
from wmj.judge.skill import compute_skill
from wmj.judge.types import (
    Bands,
    Forecasts,
    JudgeInput,
    RegionCurve,
    RegionLabel,
    TaskSpec,
    Thresholds,
)
from wmj.judge.verdict import assemble_verdict
from wmj.models import registry
from wmj.models.base import SeedSource
from wmj.reporting.writer import META_KEYS
from wmj.worlds import lv, pendulum

WORLDS = {"lv": lv.WORLD, "pendulum": pendulum.WORLD}  # fixed, not discovered (ADR-004's disclosed asymmetry)
BASELINES = ("persistence", "linear")  # the two pinned reference baselines (MU-2, models ADR-M1)
N_TRIALS = 200  # per (world, region), paired across models (judge ADR-J4)
MODEL_REF = "opaque-index-{}"  # judge §5: the envelope's reference is the model's place in the sorted roster


class TrialsError(WmjError):
    """The run cannot go ahead as specified (cross-cutting ADR-004); nothing is returned."""


def roster() -> dict[str, Any]:
    """Every registered model, found by importing each module of the `wmj.models` package (D7, MU-9).

    A module registers its model when it is imported, so adding a model is one new file there. The result is
    the registry's sorted-name table."""
    importlib.invalidate_caches()
    for info in pkgutil.iter_modules(wmj.models.__path__):
        importlib.import_module(f"{wmj.models.__name__}.{info.name}")
    return registry.all_models()


@dataclass(frozen=True)
class RunConfig:
    """The sizes of one run. The defaults are the production values; tests pass small ones.

    `thresholds` is the parsed contents of `prereg/thresholds.json`. `horizon` and `training_horizon` default
    to the world's own declared horizon (its longest task)."""

    run_seed: int
    recipe: TrainingRecipe
    thresholds: dict
    n_trials: int = N_TRIALS
    horizon: int | None = None
    training_horizon: int | None = None
    divergence_starts: int = DEFAULT_N_STARTS
    climatology_starts: int = CLIMATOLOGY_STARTS
    climatology_steps: int = DEFAULT_STEPS_PER_START
    climatology_bins: int = N_BINS
    climatology_min_samples: int = MIN_SAMPLES_PER_BIN
    probe_states: int = PROBE_STATES

    def __post_init__(self) -> None:
        if type(self.run_seed) is not int:
            raise TrialsError(f"run_seed must be a whole number, got {self.run_seed!r}")
        if type(self.recipe) is not TrainingRecipe:
            raise TrialsError("recipe must be the TrainingRecipe read from prereg/recipe.md")
        if type(self.thresholds) is not dict:
            raise TrialsError("thresholds must be the parsed contents of prereg/thresholds.json")
        for name in ("n_trials", "divergence_starts", "climatology_starts", "climatology_steps",
                     "climatology_bins", "climatology_min_samples", "probe_states"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise TrialsError(f"{name} must be a whole number of at least 1, got {value!r}")
        for name in ("horizon", "training_horizon"):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value < 1):
                raise TrialsError(f"{name} must be a whole number of steps of at least 1, or None, got {value!r}")


def judge_thresholds(raw: dict, world_name: str, scale: np.ndarray) -> Thresholds:
    """The pre-registered thresholds for one world, in the judge's units.

    The hedge threshold is written in the world's own units, so it is divided by the scale vector like every
    forecast (backlog A30). The bands and the agreement mark have no units."""
    try:
        bands = raw["bands"]
        hedge = np.asarray(raw["sharpness_hedge_threshold"][world_name], dtype=float)
        agreement = raw["agreement_threshold"][world_name]
        if hedge.shape != np.shape(scale):  # one entry per quantity: a shorter list must not be stretched to fit
            raise ValueError(f"hedge threshold has {hedge.shape} entries, the world has {np.shape(scale)} quantities")
        return Thresholds(
            Bands(n=bands["n"], p=bands["p"], green=tuple(bands["green"]), amber_outer=tuple(bands["amber_outer"])),
            hedge / np.asarray(scale, dtype=float),
            agreement,
        )
    except (KeyError, TypeError, AttributeError, ValueError) as error:
        raise TrialsError(f"prereg/thresholds.json lacks the {world_name!r} thresholds the judge needs: {error!r}") from error


@dataclass(frozen=True)
class WorldTrials:
    """One world's shared exam: everything every model faces identically, per region, in declared order."""

    world_name: str
    ctx: Any
    training: Any
    region_names: tuple[str, ...]
    warmups: dict[str, np.ndarray]  # [N, d] per region: the drawn points, one true step before each start
    starts: dict[str, np.ndarray]  # [N, d] per region: the true state after that step, where grading begins
    kicks: dict[str, np.ndarray]  # [N, H, a] per region
    truth: dict[str, np.ndarray]  # [N, H, d] per region, raw units, the true state after steps 1..H
    labels: dict[str, tuple[RegionLabel, ...]]
    curves: dict[str, np.ndarray]  # the world's own divergence curve, H+1 points, normalised distance
    tables: dict[str, dict]  # one climatology table per region (D21), raw units
    bins: dict[str, np.ndarray]  # [N, H] climatology range of each true state, measured every step


def _true_rollout(world: Any, starts: np.ndarray, kicks: np.ndarray) -> np.ndarray:
    """The truth: the world's own step, applied once per step to every trial at once (WD-3: no integrator here)."""
    n, h = kicks.shape[:2]
    out = np.empty((n, h, starts.shape[1]))
    state = starts
    for step in range(h):
        state = world.transition_batch(state, kicks[:, step])
        out[:, step] = state
    return out


def build_world_trials(world_name: str, world: Any, config: RunConfig) -> WorldTrials:
    """Draw and run everything shared by every model on one world."""
    seeds = SeedSource(config.run_seed, None)
    horizon = config.horizon if config.horizon is not None else max(int(t.horizon) for t in world.tasks())
    ctx = make_world_context(world_name, world)
    training = build_training_data(world_name, world, seeds, config.recipe, horizon=config.training_horizon)
    spec = world.regions()
    divergence = build_divergence_artefact(world_name, world, seeds, n_starts=config.divergence_starts, horizon=horizon)
    names, warmups, starts, kicks, truth, labels, curves, tables, bins = [], {}, {}, {}, {}, {}, {}, {}, {}
    for region, box, band in declared_regions(world):
        names.append(region)
        # The warm start (owner decision 2026-10-27, backlog A43 item 6): draw a point in the region, step the
        # true world once with no kick, and grade from there; every model is first shown that true step, so
        # "carry on in a straight line" has a direction to carry on.
        warmups[region] = sample_region_starts(seeds.rng_for(world_name, region, "eval-starts"), box, config.n_trials)
        starts[region] = world.transition_batch(warmups[region], np.zeros((config.n_trials, world.a)))
        assert_eval_starts_disjoint(training, warmups[region])  # MU-7, at run time: neither the drawn point
        assert_eval_starts_disjoint(training, starts[region])  # nor the start may be a training start
        kicks[region] = seeded_kick_sequences(seeds, world_name, world, region, band, "eval-kicks", config.n_trials, horizon)
        truth[region] = _true_rollout(world, starts[region], kicks[region])
        labels[region] = tuple(  # the state axis is judged at the drawn point: the point the region's box was drawn for
            RegionLabel(**label_trial(spec, region, warmups[region][i], kicks[region][i])) for i in range(config.n_trials)
        )
        curves[region] = np.asarray(divergence["regions"][region]["median_separation"], dtype=float)
        tables[region] = build_climatology_table(
            world_name, world, seeds, n_starts=config.climatology_starts, steps_per_start=config.climatology_steps,
            n_bins=config.climatology_bins, min_samples=config.climatology_min_samples, region_name=region,
        )
        bins[region] = measure_invariant_bins(world, truth[region], tables[region])
        for shared in (warmups[region], starts[region], kicks[region], truth[region], bins[region]):
            # every model faces these same arrays: a model that wrote into its input (a kick, a state) would
            # silently change every later model's trials, so writing into them fails instead (review of P6-C01a)
            shared.setflags(write=False)
    return WorldTrials(world_name, ctx, training, tuple(names), warmups, starts, kicks, truth, labels, curves, tables, bins)


def _checked(name: str, mean: Any, spread: Any, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    mean = np.asarray(mean, dtype=float)
    spread = np.asarray(spread, dtype=float)
    if mean.shape != shape or spread.shape != shape:
        raise TrialsError(f"{name}: a forecast must be {shape} guesses and error bars, got {mean.shape} and {spread.shape}")
    if not (np.all(np.isfinite(mean)) and np.all(np.isfinite(spread)) and np.all(spread > 0.0)):
        raise TrialsError(f"{name}: a forecast must be finite with an error bar wider than zero (models ADR-M1)")
    return mean, spread


def _one(name: str, forecast: Any, d: int) -> tuple[np.ndarray, np.ndarray]:
    """One `Prediction` as a checked `(1, d)` guess and error bar."""
    try:
        mean, spread = np.asarray(forecast.mean)[None, :], np.asarray(forecast.spread)[None, :]
    except (AttributeError, IndexError, TypeError, ValueError) as error:
        raise TrialsError(f"{name}: predict must return a Prediction with a guess and an error bar (models ADR-M1)") from error
    return _checked(name, mean, spread, (1, d))


def roll_out(
    model: Any, warmups: np.ndarray, starts: np.ndarray, kicks: np.ndarray, name: str = "the model"
) -> tuple[np.ndarray, np.ndarray]:
    """`(mean, spread)`, each `[N, H, d]` in raw units: the model's own rollout from each start.

    First the model is shown the true step it has just watched: one call from the drawn point with no kick,
    whose answer is not scored (the warm start). Step 1 is then predicted from the true start; each later
    step from the model's previous guess, with the trial's true kick for that step. A model that keeps no memory runs all trials together (`predict_batch`);
    one that does runs trial by trial after `reset()`. Both give the same numbers bit for bit (TC-MU1-04).
    `name` is the model's roster name, for error messages."""
    starts = np.asarray(starts, dtype=float)
    n, d = starts.shape
    h = kicks.shape[1]
    means, spreads = np.empty((n, h, d)), np.empty((n, h, d))
    warmups = np.asarray(warmups, dtype=float)
    if warmups.shape != (n, d):
        raise TrialsError(f"{name}: one warm-up point per start is needed, got {warmups.shape} for {n} starts")
    quiet = np.zeros((n, kicks.shape[2]))
    if model.stateless:
        model.reset()
        _checked(name, *model.predict_batch(warmups.copy(), quiet), (n, d))  # the warm-up: shown, not scored
        state = starts.copy()
        for step in range(h):
            means[:, step], spreads[:, step] = _checked(name, *model.predict_batch(state, kicks[:, step]), (n, d))
            state = means[:, step].copy()
        return means, spreads
    for i in range(n):
        model.reset()
        _one(name, model.predict(warmups[i].copy(), quiet[i]), d)  # the warm-up: shown, not scored
        state = starts[i].copy()
        for step in range(h):
            mean, spread = _one(name, model.predict(state, kicks[i, step]), d)
            means[i, step], spreads[i, step] = mean[0], spread[0]
            state = means[i, step].copy()
    return means, spreads


def _stack(per_region: dict[str, Any], names: tuple[str, ...]) -> np.ndarray:
    return np.concatenate([np.asarray(per_region[r]) for r in names])


def assemble_input(world: Any, shared: WorldTrials, thresholds: Thresholds, forecast: dict, baselines: dict) -> JudgeInput:
    """One `JudgeInput`: every region of the world in one call, in the judge's units (backlog A30)."""
    names = shared.region_names
    scale = np.asarray(world.scale, dtype=float)
    horizon = shared.truth[names[0]].shape[1]

    def scaled(preds: dict) -> Forecasts:
        return Forecasts(mean=_stack({r: preds[r][0] for r in names}, names) / scale,
                         spread=_stack({r: preds[r][1] for r in names}, names) / scale)

    return JudgeInput(
        world=shared.world_name,
        dt=world.dt,
        natural_cycle_length=world.natural_cycle_length,
        predictions=scaled(forecast),
        outcomes=_stack(shared.truth, names) / scale,
        persistence=scaled(baselines["persistence"]),
        linear=scaled(baselines["linear"]),
        region_labels=tuple(label for r in names for label in shared.labels[r]),
        divergence_curves=tuple(RegionCurve(r, shared.curves[r]) for r in names),
        climatology=tuple(judge_tables(shared.tables[r], (r,), scale)[0] for r in names),
        invariant_bins=_stack(shared.bins, names),
        tasks=tuple(TaskSpec(t.name, t.kind, t.tolerance, min(int(t.horizon), horizon)) for t in world.tasks()),
        thresholds=thresholds,
    )


def judge_one(inp: JudgeInput) -> dict:
    """The judge's whole verdict on one input, as plain data. The judge is pure: this only composes it."""
    exceptions, trial_block = compute_exceptions_and_trials(inp)
    return assemble_verdict(
        world=inp.world,
        skill=compute_skill(inp),
        error_vs_horizon=compute_error_vs_horizon(inp),
        calibration=compute_calibration(inp),
        sharpness=compute_sharpness(inp),
        exceptions=exceptions,
        trials=trial_block,
        climatology=compute_climatology(inp),
        trust_horizons=compute_trust_horizons(inp),
    ).to_dict()


@dataclass(frozen=True)
class RunResult:
    envelopes: list[dict]  # one per (model, world): world by world in sorted order, models in sorted order
    baseline_names: frozenset[str]  # every model that declares itself a baseline (for the page's action sentence)


def _flags(name: str, model: Any) -> dict[str, bool]:
    """The three flags every model declares (models ADR-M1), each True or False, or the run refuses."""
    flags = {}
    for attribute in ("stateless", "is_baseline", "is_fixture"):
        value = getattr(model, attribute, None)
        if type(value) is not bool:
            raise TrialsError(f"{name}: a model must declare {attribute} as True or False (models ADR-M1)")
        flags[attribute] = value
    if name in BASELINES and not flags["is_baseline"]:
        raise TrialsError(f"{name} is a pinned reference baseline but does not declare is_baseline (MU-2, models ADR-M1)")
    return flags


def _run_model(factory: Any, name: str, shared: WorldTrials, run_seed: int) -> tuple[Any, dict]:
    """A fresh instance of one model for this world, and its rollout from every region's shared starts."""
    model = factory(shared.ctx, SeedSource(run_seed, name), shared.training)
    _flags(name, model)  # checked before the long rollout, not after it
    return model, {
        r: roll_out(model, shared.warmups[r], shared.starts[r], shared.kicks[r], name) for r in shared.region_names
    }


def run_trials(config: RunConfig, meta: dict, *, models: dict | None = None) -> RunResult:
    """Run every model on every world and return the envelopes. `models` replaces the discovered roster (tests).

    **This does not check the pre-registration.** With the discovered roster it trains and evaluates the two
    unrigged models, which must never happen before the freeze (models ADR-M5; backlog A42 item 2). The only
    production caller, `wmj run`, calls `check_prereg` first; tests pass a stand-in roster."""
    table = roster() if models is None else dict(sorted(models.items()))
    missing = [b for b in BASELINES if b not in table]
    if missing:
        raise TrialsError(f"the roster lacks the reference baseline(s) {missing}: every model is scored against both (MU-2)")
    if type(meta) is not dict or set(meta) != META_KEYS:
        raise TrialsError(f"meta must be a record of exactly {sorted(META_KEYS)} (judge §5)")
    if type(meta["seed"]) is not int or meta["seed"] != config.run_seed:
        raise TrialsError(
            f"meta records seed {meta['seed']!r} but the run uses {config.run_seed}: the verdict files would name a "
            "seed that did not produce them (NF-1)"
        )
    thresholds = {name: judge_thresholds(config.thresholds, name, WORLDS[name].scale) for name in sorted(WORLDS)}
    wrong = {name: t.bands.n for name, t in thresholds.items() if t.bands.n != config.n_trials}
    if wrong:
        raise TrialsError(
            f"the pre-registered bands are for n = {wrong} trials per region, but this run has "
            f"{config.n_trials}: they would grade the wrong number (judge ADR-J4)"
        )
    names = sorted(table)
    envelopes, baseline_names = [], set()
    for world_name in sorted(WORLDS):
        world = WORLDS[world_name]
        shared = build_world_trials(world_name, world, config)
        # the pre-pass: every input needs both baselines' forecasts, so they run first (ADR-004)
        runs = {name: _run_model(table[name], name, shared, config.run_seed) for name in BASELINES}
        baselines = {name: runs[name][1] for name in BASELINES}
        for ref, name in enumerate(names):
            model, forecast = runs[name] if name in runs else _run_model(table[name], name, shared, config.run_seed)
            flags = _flags(name, model)
            if flags["is_baseline"]:
                baseline_names.add(name)
            action = asdict(check_action_response(model, shared.ctx, config.run_seed, n_states=config.probe_states))
            verdict = judge_one(assemble_input(world, shared, thresholds[world_name], forecast, baselines))
            envelopes.append({
                "model_ref": MODEL_REF.format(ref),
                "model_name": name,
                "is_fixture": flags["is_fixture"],
                "verdict": verdict,
                "meta": dict(meta),
                "action_response": action,
            })
    return RunResult(envelopes, frozenset(baseline_names))

