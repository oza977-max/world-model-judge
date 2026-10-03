"""Tests for wmj.harness.training — the homework every model is handed.

In plain words: these tests build small versions of the training set for the
real worlds and check every rule the spec states: exact counts, the kick
quota, the held-back set that never overlaps the training set, a refusal
(never a quiet shrink) when the world cannot supply enough kicks, the same
bytes every time from the same seed, and separate seed streams for
separate jobs. Counts are shrunk so the tests run in seconds; the
full-size run is a separate gate (tests/gates/test_training_full_scale.py).
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from wmj.harness.benchmarks import declared_regions, sample_region_starts
from wmj.harness.kicks import seeded_kick_sequences
from wmj.harness.training import (
    TRAINING_PURPOSES,
    TrainingDataError,
    TrainingRecipe,
    assert_eval_starts_disjoint,
    build_training_data,
    read_training_recipe,
)
from wmj.models.base import SeedSource, TrainingData
from wmj.worlds import lv, pendulum

SEED = 20260825
REAL_RECIPE = Path(__file__).resolve().parents[3] / "prereg" / "recipe.md"

# Small-scale recipes: enough kicks exist at these sizes for each world's rate.
SMALL = {
    "lv": (lv, TrainingRecipe(
        training_trajectories=200, subsample_pairs=2000, kick_pairs=100,
        heldout_pairs=500, gradcheck_pairs=16), 200),
    "pendulum": (pendulum, TrainingRecipe(
        training_trajectories=200, subsample_pairs=2000, kick_pairs=50,
        heldout_pairs=500, gradcheck_pairs=16), 600),
}
WORLDS = list(SMALL)


def _build(name, *, recipe=None, seed=SEED):
    module, small, horizon = SMALL[name]
    return build_training_data(
        name, module.WORLD, SeedSource(seed, None), recipe or small, horizon=horizon
    )


@pytest.fixture(scope="module", params=WORLDS)
def built(request):
    return request.param, _build(request.param)


# --- the recipe: every count comes from prereg/recipe.md ---


def test_the_real_recipe_gives_the_pinned_counts():
    r = read_training_recipe(REAL_RECIPE)
    assert r == TrainingRecipe(
        training_trajectories=2000, subsample_pairs=50000, kick_pairs=12500,
        heldout_pairs=10000, gradcheck_pairs=64,
    )


def _recipe_text(**overrides):
    values = {
        "training_trajectories": "2000", "subsample_pairs": "50000", "kick_pairs": "12500",
        "heldout_pairs": "10000", "gradcheck_pairs": "64",
    }
    values.update(overrides)
    lines = [f"{k}: {v}" for k, v in values.items() if v is not None]
    return "```\n" + "\n".join(lines) + "\n```\n"


def _read(tmp_path, text):
    path = tmp_path / "recipe.md"
    path.write_text(text)
    return read_training_recipe(path)


def test_a_missing_key_is_refused_and_named(tmp_path):
    with pytest.raises(TrainingDataError, match="kick_pairs"):
        _read(tmp_path, _recipe_text(kick_pairs=None))


def test_a_repeated_key_is_refused(tmp_path):
    with pytest.raises(TrainingDataError, match="more than once"):
        _read(tmp_path, _recipe_text() + "kick_pairs: 99\n")


@pytest.mark.parametrize("bad", ["12.5", "1e4", "-5", "0", "12_500", "", "abc", "０"])
def test_a_value_that_is_not_a_positive_plain_integer_is_refused(tmp_path, bad):
    with pytest.raises(TrainingDataError, match="kick_pairs"):
        _read(tmp_path, _recipe_text(kick_pairs=bad))


def test_an_indented_or_prose_mention_is_not_a_key(tmp_path):
    text = "- `kick_pairs: 7` — prose\n  kick_pairs: 8\n" + _recipe_text()
    assert _read(tmp_path, text).kick_pairs == 12500


def test_a_trailing_comment_is_not_part_of_the_number(tmp_path):
    assert _read(tmp_path, _recipe_text(kick_pairs="12500  # why")).kick_pairs == 12500


def test_inconsistent_counts_are_refused(tmp_path):
    with pytest.raises(TrainingDataError, match="kick_pairs"):
        _read(tmp_path, _recipe_text(kick_pairs="60000"))  # more kicks than pairs
    with pytest.raises(TrainingDataError, match="gradcheck_pairs"):
        _read(tmp_path, _recipe_text(gradcheck_pairs="60000"))


def test_a_recipe_that_is_not_utf8_is_refused(tmp_path):
    path = tmp_path / "recipe.md"
    path.write_bytes(b"kick_pairs: 1\xff\n")
    with pytest.raises(TrainingDataError, match="UTF-8"):
        read_training_recipe(path)


def test_a_missing_recipe_file_is_a_clear_refusal(tmp_path):
    with pytest.raises(TrainingDataError, match="recipe"):
        read_training_recipe(tmp_path / "nope.md")


# --- TC-MU7-03: exact counts, layout, held-out disjoint ---


def test_shapes_and_exact_counts(built):
    name, data = built
    module, recipe, horizon = SMALL[name]
    d = module.WORLD.d
    assert data.states.shape == (recipe.training_trajectories, horizon + 1, d)
    assert data.actions.shape == (recipe.training_trajectories, horizon, 1)
    assert data.train_pairs.state.shape == (recipe.subsample_pairs, d)
    assert data.heldout_pairs.state.shape == (recipe.heldout_pairs, d)
    assert data.gradcheck_index.shape == (recipe.gradcheck_pairs,)
    assert data.gradcheck_index.dtype == np.int64


def test_exactly_the_quota_of_kick_pairs_in_training(built):
    name, data = built
    recipe = SMALL[name][1]
    assert int(data.train_pairs.is_kick.sum()) == recipe.kick_pairs


def test_layout_is_kick_pairs_first_then_non_kick_pairs(built):
    name, data = built
    k = SMALL[name][1].kick_pairs
    flags = data.train_pairs.is_kick
    assert flags[:k].all() and not flags[k:].any()


def test_is_kick_means_a_non_zero_action(built):
    _, data = built
    for pairs in (data.train_pairs, data.heldout_pairs):
        assert np.array_equal(pairs.is_kick, np.any(pairs.action != 0.0, axis=1))


@pytest.mark.parametrize("name", WORLDS)
def test_every_pair_is_one_true_step_of_the_world(name):
    module, _, _ = SMALL[name]
    data = _build(name)
    for pairs in (data.train_pairs, data.heldout_pairs):
        for i in range(0, pairs.state.shape[0], 7):  # every 7th keeps it quick
            expected = module.transition(pairs.state[i], pairs.action[i])
            assert np.array_equal(pairs.next_state[i], expected)


def test_pairs_are_consecutive_steps_of_the_trajectories(built):
    """Each pair's (state, next_state) is found in the trajectories at adjacent steps."""
    name, data = built
    horizon = SMALL[name][2]
    flat = {}
    for i in range(data.states.shape[0]):
        for t in range(horizon):
            flat[data.states[i, t].tobytes()] = (i, t)
    for pairs in (data.train_pairs, data.heldout_pairs):
        for j in range(0, pairs.state.shape[0], 11):
            i, t = flat[pairs.state[j].tobytes()]
            assert np.array_equal(data.states[i, t + 1], pairs.next_state[j])
            assert np.array_equal(data.actions[i, t], pairs.action[j])


def test_heldout_pairs_share_no_pair_with_training(built):
    _, data = built
    train = {row.tobytes() + a.tobytes() for row, a in zip(data.train_pairs.state, data.train_pairs.action)}
    held = {row.tobytes() + a.tobytes() for row, a in zip(data.heldout_pairs.state, data.heldout_pairs.action)}
    assert len(train) == data.train_pairs.state.shape[0] and len(held) == data.heldout_pairs.state.shape[0]
    assert train.isdisjoint(held)


def test_gradcheck_index_is_the_first_of_a_seeded_permutation_of_the_training_set(built):
    name, data = built
    recipe = SMALL[name][1]
    perm = SeedSource(SEED, None).rng_for(name, "training", "gradcheck-batch").permutation(
        recipe.subsample_pairs
    )
    assert np.array_equal(data.gradcheck_index, perm[: recipe.gradcheck_pairs])
    assert len(set(data.gradcheck_index.tolist())) == recipe.gradcheck_pairs


def test_the_m_set_is_a_prefix_of_the_2m_set(built):
    """ADR-M3's sufficiency test needs 2M = same kick pairs + more non-kick pairs."""
    name, data = built
    module, recipe, horizon = SMALL[name]
    doubled = build_training_data(
        name, module.WORLD, SeedSource(SEED, None),
        replace(recipe, subsample_pairs=2 * recipe.subsample_pairs, heldout_pairs=300),
        horizon=horizon,
    )
    m = recipe.subsample_pairs
    for field in ("state", "action", "next_state", "is_kick"):
        assert np.array_equal(
            getattr(doubled.train_pairs, field)[:m], getattr(data.train_pairs, field)
        )


# --- TC-MU8-01 (data half) / determinism ---


@pytest.mark.parametrize("name", WORLDS)
def test_built_twice_from_the_same_seed_is_byte_identical(name):
    a, b = _build(name), _build(name)
    assert a.states.tobytes() == b.states.tobytes()
    assert a.actions.tobytes() == b.actions.tobytes()
    for field in ("state", "action", "next_state", "is_kick"):
        assert getattr(a.train_pairs, field).tobytes() == getattr(b.train_pairs, field).tobytes()
        assert getattr(a.heldout_pairs, field).tobytes() == getattr(b.heldout_pairs, field).tobytes()
    assert a.gradcheck_index.tobytes() == b.gradcheck_index.tobytes()


def test_a_different_run_seed_gives_different_data():
    a, b = _build("lv"), _build("lv", seed=SEED + 1)
    assert a.states.tobytes() != b.states.tobytes()
    assert a.train_pairs.state.tobytes() != b.train_pairs.state.tobytes()


# --- shortfall is refused, never shrunk (TC-MU7-03 e) ---


def test_too_few_kick_pairs_is_refused_not_shrunk():
    module, recipe, horizon = SMALL["lv"]
    greedy = replace(recipe, kick_pairs=recipe.subsample_pairs)  # asks for every pair to be a kick
    with pytest.raises(TrainingDataError, match="kick pairs"):
        build_training_data("lv", module.WORLD, SeedSource(SEED, None), greedy, horizon=horizon)


def test_a_lowered_kick_rate_refuses_with_the_counts_named(monkeypatch):
    module, recipe, horizon = SMALL["lv"]
    monkeypatch.setattr(module.WORLD.__class__, "kick_rate_per_s", 0.0005, raising=False)
    with pytest.raises(TrainingDataError, match=r"kick pairs.*\d+"):
        build_training_data("lv", module.WORLD, SeedSource(SEED, None), recipe, horizon=horizon)


def test_too_few_non_kick_pairs_is_refused():
    module, recipe, horizon = SMALL["lv"]
    total = recipe.training_trajectories * horizon
    huge = replace(recipe, subsample_pairs=total + 1, heldout_pairs=1)
    with pytest.raises(TrainingDataError, match="non-kick pairs"):
        build_training_data("lv", module.WORLD, SeedSource(SEED, None), huge, horizon=horizon)


def test_too_few_pairs_left_for_the_held_out_set_is_refused():
    module, recipe, horizon = SMALL["lv"]
    total = recipe.training_trajectories * horizon
    tight = replace(recipe, heldout_pairs=total - recipe.subsample_pairs + 1)
    with pytest.raises(TrainingDataError, match="held-out"):
        build_training_data("lv", module.WORLD, SeedSource(SEED, None), tight, horizon=horizon)


# --- seed purposes (TC-NF1-10, training side) and who draws what ---


class _RecordingSeeds(SeedSource):
    """A SeedSource that remembers every stream asked for."""

    def __new__(cls, *args, **kwargs):
        return object.__new__(cls)

    def rng_for(self, *parts):
        self.log.append(parts)  # type: ignore[attr-defined]
        return super().rng_for(*parts)


def _recorded(name):
    module, recipe, horizon = SMALL[name]
    seeds = _RecordingSeeds(SEED, None)
    object.__setattr__(seeds, "log", [])
    build_training_data(name, module.WORLD, seeds, recipe, horizon=horizon)
    return seeds.log


@pytest.mark.parametrize("name", WORLDS)
def test_only_the_pinned_training_purposes_are_drawn(name):
    log = _recorded(name)
    purposes = {parts[2] for parts in log}
    assert purposes == set(TRAINING_PURPOSES)
    assert all(parts[0] == name and parts[1] == "training" for parts in log)


def test_training_purposes_are_the_six_pinned_ones():
    assert set(TRAINING_PURPOSES) == {
        "train-starts", "train-kicks", "subsample-kick", "subsample-nonkick", "heldout",
        "gradcheck-batch",
    }


def test_the_first_draws_of_all_training_streams_differ_pairwise():
    seeds = SeedSource(SEED, None)
    draws = [
        tuple(seeds.rng_for("lv", "training", purpose, *extra).random(4))
        for purpose in TRAINING_PURPOSES
        for extra in (("0",) if purpose == "train-kicks" else ())
    ]
    assert len(set(draws)) == len(draws)


@pytest.mark.parametrize("name", WORLDS)
def test_training_starts_come_from_the_train_starts_stream_inside_the_training_box(name):
    module, recipe, _ = SMALL[name]
    data = _build(name)
    expected = sample_region_starts(
        SeedSource(SEED, None).rng_for(name, "training", "train-starts"),
        module.regions().training_state_box,
        recipe.training_trajectories,
    )
    assert np.array_equal(data.states[:, 0], expected)


@pytest.mark.parametrize("name", WORLDS)
def test_training_kicks_are_the_harness_in_band_kicks(name):
    module, recipe, horizon = SMALL[name]
    data = _build(name)
    expected = seeded_kick_sequences(
        SeedSource(SEED, None), name, module.WORLD, "training", "in", "train-kicks",
        recipe.training_trajectories, horizon,
    )
    assert np.array_equal(data.actions, expected)
    assert np.abs(data.actions).max() <= module.regions().training_action_interval[0, 1]


# --- TC-WD3-01, training half: the shared integrator ---


@pytest.mark.parametrize("name", WORLDS)
def test_trajectories_equal_the_worlds_own_single_steps_bit_for_bit(name):
    module, _, _ = SMALL[name]
    data = _build(name)
    for i in (0, 1, data.states.shape[0] - 1):
        state = data.states[i, 0]
        for t in range(60):
            state = module.transition(state, data.actions[i, t])
            assert np.array_equal(state, data.states[i, t + 1])


def test_the_default_horizon_is_the_worlds_declared_horizon():
    recipe = replace(SMALL["lv"][1], training_trajectories=4, subsample_pairs=200, kick_pairs=2,
                     heldout_pairs=50, gradcheck_pairs=4)
    data = build_training_data("lv", lv.WORLD, SeedSource(SEED, None), recipe)
    assert data.actions.shape[1] == lv.HORIZON == 700


# --- TC-MU7-01: start disjointness, with a can-fail proof ---


@pytest.mark.parametrize("name", WORLDS)
def test_eval_starts_never_coincide_with_training_starts(name):
    module, _, _ = SMALL[name]
    data = _build(name)
    seeds = SeedSource(SEED, None)
    for region, box, _band in declared_regions(module.WORLD):
        eval_starts = sample_region_starts(seeds.rng_for(name, region, "eval-starts"), box, 200)
        assert_eval_starts_disjoint(data, eval_starts)


def test_the_disjointness_check_can_fail():
    data = _build("lv")
    # Only the LAST row is a real training start, so a check that looks at
    # just the first rows (or stops early) would miss it.
    leaked = np.vstack([data.states[0, 1], data.states[1, 1], data.states[2, 1], data.states[3, 0]])
    with pytest.raises(TrainingDataError, match="start 3"):
        assert_eval_starts_disjoint(data, leaked)


def test_a_later_state_of_a_training_trajectory_is_not_a_start():
    data = _build("lv")
    assert_eval_starts_disjoint(data, data.states[:3, 1])  # same trajectories, step 1: allowed


def test_disjointness_check_validates_its_input_shape():
    data = _build("lv")
    with pytest.raises(TrainingDataError, match="shape"):
        assert_eval_starts_disjoint(data, np.zeros((5, 3)))


# --- what the builder refuses to be given ---


class _TwoActionWorld:
    d, a, dt, scale, kick_rate_per_s = 2, 2, 0.02, np.ones(2), 0.5

    def regions(self):
        return lv.regions()

    def tasks(self):
        return lv.tasks()

    def transition_batch(self, s, a):
        return s


def test_a_world_with_more_than_one_action_dimension_is_refused():
    with pytest.raises(TrainingDataError, match="action"):
        build_training_data("lv", _TwoActionWorld(), SeedSource(SEED, None), SMALL["lv"][1], horizon=5)


@pytest.mark.parametrize("horizon", [0, -1, 2.5, True])
def test_a_bad_horizon_is_refused(horizon):
    with pytest.raises(TrainingDataError, match="horizon"):
        build_training_data("lv", lv.WORLD, SeedSource(SEED, None), SMALL["lv"][1], horizon=horizon)


def test_the_result_is_a_training_data_with_read_only_arrays(built):
    _, data = built
    assert isinstance(data, TrainingData)
    with pytest.raises(ValueError):
        data.states[0, 0, 0] = 0.0
    with pytest.raises(ValueError):
        data.train_pairs.state[0, 0] = 0.0


@pytest.mark.parametrize("key", ["training_trajectories", "subsample_pairs", "kick_pairs",
                                 "heldout_pairs", "gradcheck_pairs"])
@pytest.mark.parametrize("bad", [0, -3, 2.5, True, "7"])
def test_a_recipe_object_built_directly_is_validated_too(key, bad):
    values = {"training_trajectories": 10, "subsample_pairs": 100, "kick_pairs": 5,
              "heldout_pairs": 20, "gradcheck_pairs": 4}
    values[key] = bad
    with pytest.raises(TrainingDataError, match=key):
        TrainingRecipe(**values)


class _TwoTaskWorld:
    """lv, but with two tasks of different horizons: the default is the longest."""

    d, a, dt, scale = lv.WORLD.d, lv.WORLD.a, lv.WORLD.dt, lv.WORLD.scale
    kick_rate_per_s = lv.WORLD.kick_rate_per_s

    def regions(self):
        return lv.regions()

    def tasks(self):
        from wmj.worlds.base import Task

        return (Task("short", "control", 0.1, 5), Task("long", "planning", 0.4, 9))

    def transition_batch(self, s, a):
        return lv.transition_batch(s, a)


def test_the_default_horizon_is_the_longest_task_horizon():
    recipe = TrainingRecipe(training_trajectories=6, subsample_pairs=30, kick_pairs=1,
                            heldout_pairs=5, gradcheck_pairs=3)
    seeds = SeedSource(SEED, None)
    rate_boost = _TwoTaskWorld()
    rate_boost.kick_rate_per_s = 20.0  # plenty of kicks in a tiny run
    data = build_training_data("lv", rate_boost, seeds, recipe)
    assert data.actions.shape[1] == 9


# --- independent review, P3-C06 pass 1 ---


def _flat_kick_mask(data):
    return np.any(data.actions != 0.0, axis=2).reshape(-1)


def _pairs_at(data, flat):
    h = data.actions.shape[1]
    i, t = np.divmod(flat, h)
    return data.states[i, t], data.actions[i, t], data.states[i, t + 1]


@pytest.mark.parametrize("name", WORLDS)
def test_the_chosen_pairs_are_exactly_the_first_of_the_named_seeded_permutations(name):
    """I-1: recompute the choice independently from the spec and compare every row."""
    _, recipe, _ = SMALL[name]
    data = _build(name)
    seeds = SeedSource(SEED, None)
    mask = _flat_kick_mask(data)
    kicks, plain = np.flatnonzero(mask), np.flatnonzero(~mask)
    k_perm = seeds.rng_for(name, "training", "subsample-kick").permutation(kicks.size)
    p_perm = seeds.rng_for(name, "training", "subsample-nonkick").permutation(plain.size)
    wanted_plain = recipe.subsample_pairs - recipe.kick_pairs
    chosen = np.concatenate([kicks[k_perm[: recipe.kick_pairs]], plain[p_perm[:wanted_plain]]])
    rest = np.ones(mask.size, dtype=bool)
    rest[chosen] = False
    pool = np.flatnonzero(rest)
    held = pool[seeds.rng_for(name, "training", "heldout").permutation(pool.size)][
        : recipe.heldout_pairs
    ]
    for pairs, flat in ((data.train_pairs, chosen), (data.heldout_pairs, held)):
        state, action, nxt = _pairs_at(data, flat)
        assert np.array_equal(pairs.state, state)
        assert np.array_equal(pairs.action, action)
        assert np.array_equal(pairs.next_state, nxt)


def test_the_first_training_row_is_a_kick_the_first_heldout_row_is_not_and_gradcheck_matches():
    """Layout sanity on one small build (kick pairs lead the training set; the
    held-out pool is almost all non-kick) and the gradient-check draw."""
    data = _build("lv")
    assert data.train_pairs.is_kick[0] and data.train_pairs.action[0, 0] != 0.0
    assert not data.heldout_pairs.is_kick[0]
    perm = SeedSource(SEED, None).rng_for("lv", "training", "gradcheck-batch").permutation(2000)
    assert data.gradcheck_index[:3].tolist() == perm[:3].tolist()


# I-2: one pair short refuses; exactly enough builds.


def _available(name):
    data = _build(name)
    n_kick = int(_flat_kick_mask(data).sum())
    return data, n_kick, data.actions.shape[0] * data.actions.shape[1] - n_kick


@pytest.mark.parametrize("name", WORLDS)
def test_exactly_the_available_kick_pairs_builds_and_one_more_is_refused(name):
    module, recipe, horizon = SMALL[name]
    _, n_kick, _ = _available(name)
    seeds = SeedSource(SEED, None)
    ok = replace(recipe, kick_pairs=n_kick, subsample_pairs=n_kick + 200, heldout_pairs=50)
    data = build_training_data(name, module.WORLD, seeds, ok, horizon=horizon)
    assert int(data.train_pairs.is_kick.sum()) == n_kick
    short = replace(ok, kick_pairs=n_kick + 1, subsample_pairs=n_kick + 201)
    with pytest.raises(TrainingDataError, match="kick pairs"):
        build_training_data(name, module.WORLD, seeds, short, horizon=horizon)


@pytest.mark.parametrize("name", WORLDS)
def test_exactly_the_available_non_kick_pairs_builds_and_one_more_is_refused(name):
    module, recipe, horizon = SMALL[name]
    _, n_kick, n_plain = _available(name)
    seeds = SeedSource(SEED, None)
    kp = min(recipe.kick_pairs, n_kick - 60)  # leaves kicked pairs for the held-out pool
    ok = replace(recipe, kick_pairs=kp, subsample_pairs=kp + n_plain, heldout_pairs=50,
                 gradcheck_pairs=8)
    data = build_training_data(name, module.WORLD, seeds, ok, horizon=horizon)
    assert data.train_pairs.state.shape[0] == kp + n_plain
    with pytest.raises(TrainingDataError, match="non-kick pairs"):
        build_training_data(
            name, module.WORLD, seeds, replace(ok, subsample_pairs=ok.subsample_pairs + 1),
            horizon=horizon,
        )


@pytest.mark.parametrize("name", WORLDS)
def test_a_heldout_request_equal_to_the_leftover_pool_builds_and_one_more_is_refused(name):
    module, recipe, horizon = SMALL[name]
    total = recipe.training_trajectories * horizon
    seeds = SeedSource(SEED, None)
    exact = replace(recipe, heldout_pairs=total - recipe.subsample_pairs)
    data = build_training_data(name, module.WORLD, seeds, exact, horizon=horizon)
    assert data.heldout_pairs.state.shape[0] == total - recipe.subsample_pairs
    with pytest.raises(TrainingDataError, match="held-out"):
        build_training_data(
            name, module.WORLD, seeds, replace(exact, heldout_pairs=exact.heldout_pairs + 1),
            horizon=horizon,
        )


# I-3: the disjointness check at its edges.


@pytest.mark.parametrize("eval_position", [0, 1, 4])
@pytest.mark.parametrize("train_row", [0, 1, 199])
def test_a_leak_is_found_at_any_position_and_for_any_training_trajectory(eval_position, train_row):
    data = _build("lv")
    eval_starts = data.states[:5, 1].copy()  # later states: legal
    eval_starts[eval_position] = data.states[train_row, 0]
    with pytest.raises(TrainingDataError, match=f"start {eval_position}"):
        assert_eval_starts_disjoint(data, eval_starts)


@pytest.mark.parametrize("shape", [(2,), (1, 2, 2), (0,)])
def test_eval_starts_of_the_wrong_rank_are_refused(shape):
    data = _build("lv")
    with pytest.raises(TrainingDataError, match="shape"):
        assert_eval_starts_disjoint(data, np.zeros(shape))


# I-4 / minors


@pytest.mark.parametrize("name", WORLDS)
def test_training_arrays_own_their_memory_so_freezing_leaves_no_writable_base(name):
    data = _build(name)
    for array in (data.states, data.actions, data.train_pairs.state, data.train_pairs.action,
                  data.heldout_pairs.state, data.gradcheck_index):
        base = array.base
        assert base is None or not base.flags.writeable, "a writable base can change a frozen array"


@pytest.mark.parametrize("line", ["kick_pairs: 12500 13000", "kick_pairs: 12500 (was 10000)",
                                  "kick_pairs: 012500", "kick_pairs:\\t0"])
def test_trailing_text_that_is_not_a_comment_is_refused(tmp_path, line):
    text = _recipe_text(kick_pairs=None) + line.replace("\\t", "\t") + "\n"
    with pytest.raises(TrainingDataError, match="kick_pairs"):
        _read(tmp_path, text)


@pytest.mark.parametrize("value", ["12500\t", "12500 # c", "12500# c", "\t12500", "12500  "])
def test_whitespace_and_comments_around_the_value_are_fine(tmp_path, value):
    assert _read(tmp_path, _recipe_text(kick_pairs=value)).kick_pairs == 12500


def test_a_crlf_recipe_is_read(tmp_path):
    path = tmp_path / "recipe.md"
    path.write_bytes(_recipe_text().replace("\n", "\r\n").encode())
    assert read_training_recipe(path).kick_pairs == 12500


def test_a_directory_in_place_of_the_recipe_is_a_clear_refusal(tmp_path):
    with pytest.raises(TrainingDataError, match="cannot be read"):
        read_training_recipe(tmp_path)


def test_a_world_with_no_tasks_has_no_default_horizon():
    class _NoTasks(_TwoTaskWorld):
        def tasks(self):
            return ()

    with pytest.raises(TrainingDataError, match="no tasks"):
        build_training_data("lv", _NoTasks(), SeedSource(SEED, None), SMALL["lv"][1])


def test_a_gradcheck_batch_as_large_as_the_training_set_is_a_legal_recipe():
    TrainingRecipe(training_trajectories=10, subsample_pairs=50, kick_pairs=5,
                   heldout_pairs=5, gradcheck_pairs=50)


# --- independent review, P3-C06 pass 3 ---


def test_mixed_numpy_integer_recipe_values_are_normalised_and_build():
    recipe = TrainingRecipe(np.int64(200), np.uint64(2000), np.int32(100), np.int16(500), np.uint8(16))
    assert all(type(getattr(recipe, k)) is int for k in
               ("training_trajectories", "subsample_pairs", "kick_pairs", "heldout_pairs",
                "gradcheck_pairs"))
    data = build_training_data("lv", lv.WORLD, SeedSource(SEED, None), recipe, horizon=200)
    plain = build_training_data("lv", lv.WORLD, SeedSource(SEED, None), SMALL["lv"][1], horizon=200)
    assert data.train_pairs.state.tobytes() == plain.train_pairs.state.tobytes()


def test_impossible_counts_are_refused_before_anything_is_simulated(monkeypatch):
    def boom(*_a, **_k):
        raise AssertionError("simulation started although the counts cannot be met")

    monkeypatch.setattr(lv.WORLD.__class__, "transition_batch", boom)
    recipe = TrainingRecipe(5, 10**6, 100, 50, 16)
    with pytest.raises(TrainingDataError, match="training pairs"):
        build_training_data("lv", lv.WORLD, SeedSource(SEED, None), recipe, horizon=10)
    recipe = TrainingRecipe(5, 30, 10, 30, 16)  # 30 + 30 > 50
    with pytest.raises(TrainingDataError, match="held-out"):
        build_training_data("lv", lv.WORLD, SeedSource(SEED, None), recipe, horizon=10)


class _BadStepWorld(_TwoTaskWorld):
    mode = "ok"

    def transition_batch(self, s, a):
        if self.mode == "scalar":
            return 0.0
        if self.mode == "row":
            return s[:1]
        if self.mode == "nan":
            out = lv.transition_batch(s, a)
            out[0, 0] = np.nan
            return out
        if self.mode == "mutate":
            s += 1.0  # edits the caller's array
        return lv.transition_batch(s, a)


@pytest.mark.parametrize("mode", ["scalar", "row", "nan"])
def test_a_world_step_with_a_wrong_shape_or_non_finite_result_is_refused(mode):
    world = _BadStepWorld()
    world.mode = mode
    world.kick_rate_per_s = 20.0
    recipe = TrainingRecipe(6, 30, 1, 5, 3)
    with pytest.raises(TrainingDataError, match="transition_batch"):
        build_training_data("lv", world, SeedSource(SEED, None), recipe, horizon=9)


def test_a_world_step_that_edits_its_input_cannot_corrupt_the_record():
    clean, noisy = _BadStepWorld(), _BadStepWorld()
    for w in (clean, noisy):
        w.kick_rate_per_s = 20.0
    noisy.mode = "mutate"
    recipe = TrainingRecipe(6, 30, 1, 5, 3)
    # The mutating world steps from start + 1 each time, so its result differs from
    # the clean world's; but whatever it returns, earlier recorded states must be
    # exactly the values that were passed in (no after-the-fact edits).
    data = build_training_data("lv", noisy, SeedSource(SEED, None), recipe, horizon=9)
    for t in range(9):
        assert np.array_equal(
            data.states[:, t + 1], lv.transition_batch(data.states[:, t] + 1.0, data.actions[:, t])
        )
    assert build_training_data("lv", clean, SeedSource(SEED, None), recipe, horizon=9).states.shape == data.states.shape


def test_a_world_whose_dimension_disagrees_with_its_training_box_is_refused():
    class _Wrong(_TwoTaskWorld):
        d = 3

    with pytest.raises(TrainingDataError, match="state dimensions"):
        build_training_data("lv", _Wrong(), SeedSource(SEED, None), TrainingRecipe(5, 30, 1, 5, 3), horizon=9)


def test_non_finite_eval_starts_are_refused_and_negative_zero_equals_zero():
    data = _build("lv")
    with pytest.raises(TrainingDataError, match="finite"):
        assert_eval_starts_disjoint(data, np.array([[np.nan, 1.0]]))
    # A training start with a 0.0 component, rebuilt by hand, must collide with its -0.0 twin.
    states = data.states.copy()
    states[0, 0] = [0.0, 2.0]
    patched = TrainingData(states=states, actions=data.actions)
    with pytest.raises(TrainingDataError, match="start 0"):
        assert_eval_starts_disjoint(patched, np.array([[-0.0, 2.0]]))
