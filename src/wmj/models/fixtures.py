"""wmj.models.fixtures — three deliberately broken test models. FIXTURES, NEVER FINDINGS.

In plain words: to show the judge catches what it should, we build models that are
broken on purpose, each in exactly one way, and check the judge notices exactly that
way. (1) `fx-overconfident` is right but cocky: its guesses are Model A's, its error
bars are a quarter the size. (2) `fx-honest-rough` is rougher but honest: its guesses
are jittered, and its error bar is widened by exactly the amount that makes it honest
again. (3) `fx-brittle` is excellent at home and catastrophic away: inside the region it
was trained on it is Model A; outside it just says "nothing changes". Each is Model A
itself — the very same trained network, bit for bit — plus that one change (models
spec ADR-M4).

**These are test equipment, not results (MU-4).** Detecting a failure we engineered is
a passing unit test, never a discovery; the label is `is_fixture = True` and a name starting
`fx-`; the code surface is built here, and the verdict record and every chart must carry
it too (built in later chunks, MU-4/RP-8). Nothing reported about them may be presented
as a finding.

**The noise is a function of the input, not of the order of calls.** A model with no
memory between steps must give the same numbers for a row whether it is predicted
alone or inside a batch of 200 (TC-MU1-04), so `fx-honest-rough`'s "seeded noise"
cannot be drawn from a stream consumed call by call. It is derived from the row's own
bytes and a seed-derived key by integer mixing, summed over twelve uniform numbers per
quantity (an approximately standard-normal draw built from integer arithmetic and
addition only, so the batch size cannot change a single bit).
"""

from __future__ import annotations

import numpy as np

from wmj.errors import WmjError
from wmj.models.base import Prediction, SeedSource, TrainingData, WorldContext
from wmj.models.direct import DirectModel, shared_direct_core
from wmj.models.registry import register

OVERCONFIDENCE_FACTOR = 0.25  # fx-overconfident: spread x 0.25 (ADR-M4 table)
NOISE_SIGMA_FACTOR = 2.0  # fx-honest-rough: noise sigma = 2 x the model's spread (ADR-M4 table)
# fx-honest-rough refuses error bars outside this range: beyond it `sqrt(spread² + σ²)` loses
# precision silently (denormals) or overflows. Real error bars are ~1e-4 to 1e-2.
HONEST_SPREAD_RANGE = (1e-100, 1e100)

_MASK = np.uint64(0xFFFFFFFFFFFFFFFF)
_GOLDEN = np.uint64(0x9E3779B97F4A7C15)


class FixtureError(WmjError):
    """Raised when a fixture is asked for something its one corruption cannot do."""


def _splitmix64(x: np.ndarray) -> np.ndarray:
    """One round of the SplitMix64 integer mixer (wraps modulo 2**64, element-wise)."""
    with np.errstate(over="ignore"):
        x = x + _GOLDEN
        x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return x ^ (x >> np.uint64(31))


def hashed_standard_normal(rows: np.ndarray, key: int, n_out: int) -> np.ndarray:
    """`float64[n, n_out]` approximately N(0, 1) draws, each a pure function of its row.

    The row's float bytes are folded into a 64-bit hash with `key`; each output is the
    sum of twelve uniforms (from further mixing) minus six — mean 0, variance 1, range
    ±6. Integer mixing and addition only: no transcendental function, so the result for
    a row does not depend on how many other rows are in the call. The noise is a function
    of the row's *value* (so `-0.0` and `0.0`, which the network cannot tell apart, get the
    same noise). Its tails are a little lighter than a true bell curve (measured: 3σ tail
    0.20% against 0.27%) — the judge discloses elsewhere that tails are not validated (JU-10).
    """
    rows = np.ascontiguousarray(rows, dtype=np.float64) + 0.0  # + 0.0 turns -0.0 into 0.0
    if rows.ndim != 2:
        raise FixtureError(f"noise rows must be 2-D, got shape {tuple(rows.shape)}")
    bits = rows.view(np.uint64)
    h = np.full(rows.shape[0], np.uint64(key) & _MASK, dtype=np.uint64)
    for column in range(bits.shape[1]):
        h = _splitmix64(h ^ bits[:, column])
    out = np.zeros((rows.shape[0], n_out))
    scale = 1.0 / float(1 << 53)
    for j in range(n_out):
        total = np.zeros(rows.shape[0])
        for t in range(12):
            u = _splitmix64(h + np.uint64(1 + 12 * j + t))
            total = total + ((u >> np.uint64(11)).astype(np.float64) + 0.5) * scale
        out[:, j] = total - 6.0
    return out


class _Fixture:
    """Common shape of a fixture: a labelled, stateless wrapper around Model A's core."""

    is_fixture = True  # MU-4: the label travels with every output
    is_baseline = False
    stateless = True
    name = "fx-"

    def __init__(self, ctx: WorldContext, inner: DirectModel) -> None:
        self._ctx = ctx
        self._inner = inner

    def reset(self) -> None:
        """Stateless: nothing to clear between rollouts."""

    def _corrupt(
        self, states: np.ndarray, actions: np.ndarray, mean: np.ndarray, spread: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def predict_batch(
        self, states: np.ndarray, actions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Row `i` is bit-identical to `predict(states[i], actions[i])` (TC-MU1-04)."""
        mean, spread = self._inner.predict_batch(states, actions)
        states = np.asarray(states, dtype=float)
        actions = np.asarray(actions, dtype=float)
        mean, spread = self._corrupt(states, actions, mean, spread)
        if not (np.all(np.isfinite(mean)) and np.all(np.isfinite(spread)) and np.all(spread > 0.0)):
            raise FixtureError(f"{self.name} produced a non-finite forecast or a spread with no width")
        return mean, spread

    def predict(self, state: np.ndarray, action: np.ndarray) -> Prediction:
        state = np.asarray(state, dtype=float)
        action = np.asarray(action, dtype=float)
        if state.ndim != 1 or action.ndim != 1:
            raise FixtureError(
                f"predict takes one state [d] and one action [a]; got shapes "
                f"{tuple(state.shape)} and {tuple(action.shape)} — use predict_batch for many"
            )
        means, spreads = self.predict_batch(state[None, :], action[None, :])
        return Prediction(mean=means[0].copy(), spread=spreads[0].copy())


class FxOverconfident(_Fixture):
    """FIXTURE — right but cocky: Model A's guess, a quarter of its error bar (MU-3, TC-MU3-01)."""

    name = "fx-overconfident"

    def _corrupt(self, states, actions, mean, spread):
        return mean, spread * OVERCONFIDENCE_FACTOR


class FxHonestRough(_Fixture):
    """FIXTURE — rougher but honest: Model A's guess plus seeded noise of σ = 2 × its error
    bar, with the error bar widened to `sqrt(spread² + σ²)` — the exact standard deviation
    of "guess + independent Gaussian noise", so it is honest by construction (MU-3, TC-MU3-02)."""

    name = "fx-honest-rough"

    def __init__(self, ctx: WorldContext, inner: DirectModel, key: int) -> None:
        super().__init__(ctx, inner)
        self._key = int(key)

    def _corrupt(self, states, actions, mean, spread):
        low, high = HONEST_SPREAD_RANGE
        if not np.all((spread >= low) & (spread <= high)):
            raise FixtureError(
                f"fx-honest-rough needs every inner error bar to be a finite number between {low:g} and "
                f"{high:g} (outside that the exact root-sum-of-squares widening loses precision silently, "
                f"or is not defined); got a range {float(np.nanmin(spread)):g} to {float(np.nanmax(spread)):g}"
                f"{' with a non-number in it' if not np.all(np.isfinite(spread)) else ''}"
            )
        sigma = NOISE_SIGMA_FACTOR * spread
        noise = sigma * hashed_standard_normal(
            np.hstack([states, actions]), self._key, states.shape[1]
        )
        return mean + noise, np.sqrt(spread**2 + sigma**2)


class FxBrittle(_Fixture):
    """FIXTURE — great at home, catastrophic away: Model A when both the state is inside the
    training box and the action inside the trained interval (closed on both ends); otherwise
    "nothing changes" (the mean is the state itself), error bar untouched (MU-3, TC-MU3-03)."""

    name = "fx-brittle"

    def _corrupt(self, states, actions, mean, spread):
        box, interval = self._ctx.training_state_box, self._ctx.training_action_interval
        at_home = (
            np.all(states >= box[:, 0], axis=1)
            & np.all(states <= box[:, 1], axis=1)
            & np.all(actions >= interval[:, 0], axis=1)
            & np.all(actions <= interval[:, 1], axis=1)
        )
        return np.where(at_home[:, None], mean, states), spread


def _core(ctx: WorldContext, seeds: SeedSource, training: TrainingData) -> DirectModel:
    return DirectModel(ctx, shared_direct_core(ctx, seeds, training))


def fx_overconfident_factory(ctx, seeds: SeedSource, training: TrainingData) -> FxOverconfident:
    """FIXTURE: Model A's shared trained core, error bar x 0.25."""
    return FxOverconfident(ctx, _core(ctx, seeds, training))


def fx_honest_rough_factory(ctx, seeds: SeedSource, training: TrainingData) -> FxHonestRough:
    """FIXTURE: Model A's shared core, seeded noise; the noise key comes from this model's own seed."""
    key = int(seeds.rng("noise").integers(0, 2**63 - 1))
    return FxHonestRough(ctx, _core(ctx, seeds, training), key)


def fx_brittle_factory(ctx, seeds: SeedSource, training: TrainingData) -> FxBrittle:
    """FIXTURE: Model A's shared core, "nothing changes" outside the training region."""
    return FxBrittle(ctx, _core(ctx, seeds, training))


register("fx-overconfident", fx_overconfident_factory)
register("fx-honest-rough", fx_honest_rough_factory)
register("fx-brittle", fx_brittle_factory)
