"""wmj.judge.types — the judge's input door, built so it cannot carry identity (JU-1).

In plain words: the judge may be told what a model *predicted*, what *happened*, what the
two simple baselines predicted, which region of the world each trial started in, how fast
the world drifts from itself, what the world usually looks like, which tasks are being
graded, and the thresholds fixed in advance. That is all. There is no field for a model's
name, whether it is a deliberately broken test model, its architecture or its training
history — and the classes are frozen, slotted and exact-typed (a subclass with extra
attributes is refused), so a field cannot be attached afterwards by accident. This is
blindness by construction, not by promise (TC-JU1-01). What no type can stop is a person
writing a model's name into a free-text field such as the world's or a task's name; the
harness hands the judge only world, region and task names, and a test pins that these are
the only text fields. Without both baselines the
input cannot even be built (MU-2, TC-MU2-01). Every array is copied and locked read-only,
and every shape, count and range is checked here, once, so the arithmetic that follows can
trust it.

Imports only numpy, math, dataclasses and typing, plus this package (cross-cutting ADR-003).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from wmj.judge.errors import JudgeInputError, MissingBaselineError

AXES = ("state", "action", "both")
TASK_KINDS = ("control", "planning")


def _bad(message: str) -> JudgeInputError:
    return JudgeInputError(message)


def _frozen_array(name: str, value, *, dtype=float, ndim: int | None = None) -> np.ndarray:
    try:
        source = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise _bad(f"{name} must be an array of numbers: {exc}") from exc
    wanted = "iu" if np.dtype(dtype).kind in "iu" else "iuf"
    if source.dtype.kind not in wanted:
        raise _bad(f"{name} must hold {'whole numbers' if wanted == 'iu' else 'numbers'}, got {source.dtype}")
    if source.dtype.kind in "iu" and np.dtype(dtype).kind == "i" and source.size and (
        source.max() > np.iinfo(np.int64).max
    ):
        raise _bad(f"{name} holds a number too large for a 64-bit integer")
    arr = np.array(source, dtype=dtype, copy=True)
    if ndim is not None and arr.ndim != ndim:
        raise _bad(f"{name} must have {ndim} dimension(s), got shape {tuple(arr.shape)}")
    if arr.dtype.kind == "f" and not np.all(np.isfinite(arr)):
        raise _bad(f"{name} contains NaN or infinity")
    arr.setflags(write=False)
    return arr


def _exact_str(name: str, value) -> str:
    if type(value) is not str or not value.strip():
        raise _bad(f"{name} must be a plain non-blank string, got {value!r}")
    return value


def _whole(name: str, value) -> int:
    if type(value) is bool or not isinstance(value, (int, np.integer)):
        raise _bad(f"{name} must be a whole number, got {value!r}")
    try:
        return int(value)
    except (OverflowError, ValueError) as exc:  # pragma: no cover - ints never overflow
        raise _bad(f"{name} is not usable as a whole number: {value!r}") from exc


def _positive_finite(name: str, value) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise _bad(f"{name} must be a number, got {value!r}")
    try:
        value = float(value)
    except OverflowError as exc:
        raise _bad(f"{name} is too large to use, got a huge number") from exc
    if not (math.isfinite(value) and value > 0.0):
        raise _bad(f"{name} must be finite and positive, got {value!r}")
    return value


def _set(obj, name: str, value) -> None:
    object.__setattr__(obj, name, value)


@dataclass(frozen=True, slots=True)
class RegionLabel:
    """Which named region a trial started in, and which axis (if any) took it out of range."""

    region_name: str
    axis: str | None

    def __post_init__(self) -> None:
        _exact_str("region_name", self.region_name)
        if self.axis is not None and (type(self.axis) is not str or self.axis not in AXES):
            raise _bad(f"axis must be one of {AXES} or None, got {self.axis!r}")


@dataclass(frozen=True, slots=True)
class TaskSpec:
    """One graded task: its name, its kind, its tolerance and its horizon (worlds spec §4)."""

    name: str
    kind: str
    tolerance: float
    horizon: int

    def __post_init__(self) -> None:
        _exact_str("task name", self.name)
        if type(self.kind) is not str or self.kind not in TASK_KINDS:
            raise _bad(f"task kind must be one of {TASK_KINDS}, got {self.kind!r}")
        _set(self, "tolerance", _positive_finite("task tolerance", self.tolerance))
        horizon = _whole("task horizon", self.horizon)
        if horizon < 1:
            raise _bad(f"task horizon must be an int >= 1, got {self.horizon!r}")
        _set(self, "horizon", horizon)


@dataclass(frozen=True, slots=True, eq=False)
class Forecasts:
    """A set of Gaussian forecasts: a mean and a spread per trial, step and quantity."""

    mean: np.ndarray
    spread: np.ndarray

    def __post_init__(self) -> None:
        _set(self, "mean", _frozen_array("forecast mean", self.mean, ndim=3))
        _set(self, "spread", _frozen_array("forecast spread", self.spread, ndim=3))
        if self.mean.shape != self.spread.shape:
            raise _bad(f"mean {tuple(self.mean.shape)} and spread {tuple(self.spread.shape)} must have the same shape")
        if not np.all(self.spread > 0.0):
            raise _bad("every spread must be > 0 (a forecast with no width cannot be scored)")

    def __reduce__(self):  # copy / pickle go back through the checks and the read-only lock
        return (Forecasts, (self.mean, self.spread))


@dataclass(frozen=True, slots=True)
class Bands:
    """The pre-registered exception bands: green and the outer edge of amber (judge ADR-J4)."""

    n: int
    p: float
    green: tuple[int, int]
    amber_outer: tuple[int, int]

    def __post_init__(self) -> None:
        n = _whole("bands n", self.n)
        if n < 1:
            raise _bad(f"bands n must be an int >= 1, got {self.n!r}")
        if type(self.p) is not float and type(self.p) is not int or not (0.0 < self.p < 1.0):
            raise _bad(f"bands p must be strictly between 0 and 1, got {self.p!r}")
        for label, pair in (("green", self.green), ("amber_outer", self.amber_outer)):
            if not isinstance(pair, (tuple, list)) or len(pair) != 2:
                raise _bad(f"{label} must be a (low, high) pair, got {pair!r}")
        g_lo, g_hi = (_whole("green", v) for v in self.green)
        a_lo, a_hi = (_whole("amber_outer", v) for v in self.amber_outer)
        if not (0 <= a_lo <= g_lo <= g_hi <= a_hi <= n):
            raise _bad(f"bands must nest as 0 <= amber_lo <= green_lo <= green_hi <= amber_hi <= n, got {self.green}, {self.amber_outer}")
        _set(self, "n", n)
        _set(self, "green", (g_lo, g_hi))
        _set(self, "amber_outer", (a_lo, a_hi))


@dataclass(frozen=True, slots=True, eq=False)
class Thresholds:
    """Everything fixed in advance (JU-11): bands, the hedging threshold per quantity, the agreement threshold."""

    bands: Bands
    sharpness_hedge_threshold: np.ndarray
    agreement_threshold: float

    def __post_init__(self) -> None:
        if type(self.bands) is not Bands:
            raise _bad("thresholds.bands must be a Bands")
        hedge = _frozen_array("sharpness_hedge_threshold", self.sharpness_hedge_threshold, ndim=1)
        if hedge.size == 0 or not np.all(hedge > 0.0):
            raise _bad("sharpness_hedge_threshold must be a non-empty vector of positive numbers")
        _set(self, "sharpness_hedge_threshold", hedge)
        _set(self, "agreement_threshold", _positive_finite("agreement_threshold", self.agreement_threshold))

    def __reduce__(self):
        return (Thresholds, (self.bands, self.sharpness_hedge_threshold, self.agreement_threshold))


@dataclass(frozen=True, slots=True, eq=False)
class ClimatologyBin:
    """One bin of the conditioned climatology: what the world looks like for a range of its invariant."""

    invariant_lo: float
    invariant_hi: float
    mean: np.ndarray
    sd: np.ndarray
    n_samples: int

    def __post_init__(self) -> None:
        for label, v in (("invariant_lo", self.invariant_lo), ("invariant_hi", self.invariant_hi)):
            if type(v) is bool or not isinstance(v, (int, float, np.integer, np.floating)):
                raise _bad(f"{label} must be a number (infinite ends allowed), got {v!r}")
        lo, hi = float(self.invariant_lo), float(self.invariant_hi)
        if math.isnan(lo) or math.isnan(hi) or not lo < hi:
            raise _bad(f"a climatology bin needs invariant_lo < invariant_hi (infinite ends allowed), got {lo}, {hi}")
        _set(self, "invariant_lo", lo)
        _set(self, "invariant_hi", hi)
        _set(self, "mean", _frozen_array("climatology bin mean", self.mean, ndim=1))
        _set(self, "sd", _frozen_array("climatology bin sd", self.sd, ndim=1))
        if self.mean.shape != self.sd.shape or self.mean.size == 0:
            raise _bad("a climatology bin's mean and sd must be non-empty vectors of the same length")
        if not np.all(self.sd > 0.0):
            raise _bad("a climatology bin's sd must be > 0")
        n_samples = _whole("n_samples", self.n_samples)
        if n_samples < 1:
            raise _bad(f"n_samples must be an int >= 1, got {self.n_samples!r}")
        _set(self, "n_samples", n_samples)

    def __reduce__(self):
        return (ClimatologyBin, (self.invariant_lo, self.invariant_hi, self.mean, self.sd, self.n_samples))


@dataclass(frozen=True, slots=True, eq=False)
class RegionClimatology:
    """The climatology table for one named region."""

    region_name: str
    bins: tuple[ClimatologyBin, ...]

    def __post_init__(self) -> None:
        _exact_str("climatology region_name", self.region_name)
        bins = tuple(self.bins)
        if not bins or not all(type(b) is ClimatologyBin for b in bins):
            raise _bad("a region's climatology needs at least one ClimatologyBin")
        if len({b.mean.shape for b in bins}) != 1:
            raise _bad("every climatology bin of a region must cover the same number of quantities")
        _set(self, "bins", bins)


@dataclass(frozen=True, slots=True, eq=False)
class RegionCurve:
    """The world's drift-from-itself curve for one named region, indexed from step 0 (length H+1)."""

    region_name: str
    curve: np.ndarray

    def __post_init__(self) -> None:
        _exact_str("divergence curve region_name", self.region_name)
        curve = _frozen_array("divergence curve", self.curve, ndim=1)
        if curve.size < 2 or np.any(curve < 0.0):
            raise _bad("a divergence curve needs at least two points, none negative")
        _set(self, "curve", curve)

    def __reduce__(self):
        return (RegionCurve, (self.region_name, self.curve))


@dataclass(frozen=True, slots=True, eq=False)
class JudgeInput:
    """The whole of what the judge may know (JU-1) — and nothing it must not.

    `predictions`, `outcomes`, `persistence` and `linear` are `[n_trials, H, d]`; both
    baselines are required (a missing one raises `MissingBaselineError`). `region_labels`
    has one entry per trial; `divergence_curves` and `climatology` have one entry per
    region present; `invariant_bins` is the true invariant's bin index per trial and step.
    `world` is the world's name (the verdict records which world it is about).
    """

    world: str
    dt: float
    natural_cycle_length: float | None
    predictions: Forecasts
    outcomes: np.ndarray
    persistence: Forecasts | None
    linear: Forecasts | None
    region_labels: tuple[RegionLabel, ...]
    divergence_curves: tuple[RegionCurve, ...]
    climatology: tuple[RegionClimatology, ...]
    invariant_bins: np.ndarray
    tasks: tuple[TaskSpec, ...]
    thresholds: Thresholds

    def __post_init__(self) -> None:
        for label, value in (("persistence", self.persistence), ("linear", self.linear)):
            if value is None:
                raise MissingBaselineError(
                    f"no verdict without the {label} baseline: the judge compares every model against both "
                    "baselines (MU-2) and will not proceed without one"
                )
        for label, value, cls in (
            ("predictions", self.predictions, Forecasts),
            ("persistence", self.persistence, Forecasts),
            ("linear", self.linear, Forecasts),
            ("thresholds", self.thresholds, Thresholds),
        ):
            if type(value) is not cls:
                raise _bad(f"{label} must be a {cls.__name__}, got {type(value).__name__}")
        _exact_str("world", self.world)
        _set(self, "dt", _positive_finite("dt", self.dt))
        if self.natural_cycle_length is not None:
            _set(self, "natural_cycle_length", _positive_finite("natural_cycle_length", self.natural_cycle_length))

        outcomes = _frozen_array("outcomes", self.outcomes, ndim=3)
        _set(self, "outcomes", outcomes)
        n, steps, d = outcomes.shape
        if min(n, steps, d) < 1:
            raise _bad(f"outcomes must be non-empty in every dimension, got shape {tuple(outcomes.shape)}")
        for label, fc in (("predictions", self.predictions), ("persistence", self.persistence), ("linear", self.linear)):
            if fc.mean.shape != outcomes.shape:
                raise _bad(f"{label} shape {tuple(fc.mean.shape)} does not match outcomes {tuple(outcomes.shape)}")

        labels = tuple(self.region_labels)
        if len(labels) != n or not all(type(x) is RegionLabel for x in labels):
            raise _bad(f"region_labels needs exactly one RegionLabel per trial ({n}), got {len(labels)}")
        _set(self, "region_labels", labels)
        regions = sorted({x.region_name for x in labels})

        curves = tuple(self.divergence_curves)
        if not all(type(c) is RegionCurve for c in curves):
            raise _bad("divergence_curves must be RegionCurve entries")
        if sorted(c.region_name for c in curves) != regions:
            raise _bad(f"divergence_curves must cover exactly the regions present {regions}")
        if any(c.curve.size != steps + 1 for c in curves):
            raise _bad(f"every divergence curve must have H+1 = {steps + 1} points (indexed from step 0)")
        _set(self, "divergence_curves", curves)

        tables = tuple(self.climatology)
        if not all(type(t) is RegionClimatology for t in tables):
            raise _bad("climatology must be RegionClimatology entries")
        if sorted(t.region_name for t in tables) != regions:
            raise _bad(f"climatology must cover exactly the regions present {regions}")
        if any(t.bins[0].mean.shape != (d,) for t in tables):
            raise _bad(f"every climatology bin must cover the world's {d} quantities")
        _set(self, "climatology", tables)

        bins = _frozen_array("invariant_bins", self.invariant_bins, dtype=np.int64, ndim=2)
        if bins.shape != (n, steps):
            raise _bad(f"invariant_bins must be [n_trials, H] = {(n, steps)}, got {tuple(bins.shape)}")
        n_bins = {t.region_name: len(t.bins) for t in tables}
        limits = np.array([n_bins[x.region_name] for x in labels])
        if np.any(bins < 0) or np.any(bins >= limits[:, None]):
            raise _bad("invariant_bins holds a bin index outside its region's climatology table")
        _set(self, "invariant_bins", bins)

        tasks = tuple(self.tasks)
        if not tasks or not all(type(t) is TaskSpec for t in tasks):
            raise _bad("tasks must be a non-empty tuple of TaskSpec")
        if len({t.name for t in tasks}) != len(tasks):
            raise _bad("task names must be unique")
        if any(t.horizon > steps for t in tasks):
            raise _bad(f"a task's horizon exceeds the {steps} steps the rollouts cover")
        _set(self, "tasks", tasks)

        if self.thresholds.sharpness_hedge_threshold.shape != (d,):
            raise _bad(f"sharpness_hedge_threshold must have one entry per quantity ({d})")

    def __reduce__(self):  # copy / pickle re-run every check and re-lock every array
        return (JudgeInput, (
            self.world, self.dt, self.natural_cycle_length, self.predictions, self.outcomes, self.persistence,
            self.linear, self.region_labels, self.divergence_curves, self.climatology, self.invariant_bins,
            self.tasks, self.thresholds,
        ))
