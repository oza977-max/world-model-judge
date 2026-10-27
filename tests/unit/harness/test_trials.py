"""The orchestration loop (cross-cutting ADR-004), run end to end on a stand-in roster at small sizes.

In plain words: these tests run the real loop: training data, paired starts, the true world, the baselines
first, every model fed its own guesses, one judge call per model and world, the envelope. They check what
comes out is something the real writer accepts and writes. The roster is a stand-in (see `_standin.py`): the
two real learned models are never measured before the pre-registration freeze (backlog A42 item 2).
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.unit.harness import _standin
from tests.unit.harness._standin import (
    H_SMALL,
    META,
    N_SMALL,
    small_config,
    standin_roster,
)
from wmj.harness import trials
from wmj.harness.training import TrainingDataError
from wmj.reporting import writer

WORLDS = ("lv", "pendulum")
REGIONS = {"lv": 3, "pendulum": 3}


@pytest.fixture(scope="module")
def recorded():
    """One small run, with every judge input and every rollout's starts recorded on the way."""
    inputs, starts_seen, kicks_seen = [], [], []
    real_judge, real_roll = trials.judge_one, trials.roll_out

    def judge_spy(inp):
        inputs.append(inp)
        return real_judge(inp)

    def roll_spy(model, warmups, starts, kicks, name):
        starts_seen.append(np.ascontiguousarray(starts).tobytes())
        kicks_seen.append(np.ascontiguousarray(kicks).tobytes())
        return real_roll(model, warmups, starts, kicks, name)

    patch = pytest.MonkeyPatch()
    patch.setattr(trials, "judge_one", judge_spy)
    patch.setattr(trials, "roll_out", roll_spy)
    try:
        result = trials.run_trials(small_config(), dict(META), models=standin_roster())
    finally:
        patch.undo()
    return result, inputs, (starts_seen, kicks_seen)


# ---- outside in: the loop's output is a run the real writer accepts and writes ------------------------------


def test_a_stand_in_run_produces_envelopes_the_writer_accepts_and_writes(recorded, tmp_path):
    result, _inputs, _starts = recorded
    names = sorted(standin_roster())
    assert len(result.envelopes) == len(WORLDS) * len(names)
    assert result.baseline_names == frozenset({"linear", "persistence"})
    grouped = writer.check_run(result.envelopes, result.baseline_names)
    assert sorted(grouped) == list(WORLDS)
    assert [e["verdict"]["world"] for e in result.envelopes] == [w for w in WORLDS for _ in names]  # world by world
    for world, group in grouped.items():
        assert [e["model_name"] for e in group] == names  # sorted roster order, inside each world
        assert all(e["verdict"]["world"] == world for e in group)
        assert all(e["meta"] == META for e in group)
        assert {e["model_name"] for e in group if e["is_fixture"]} == {n for n in names if n.startswith("fx-")}
        assert [e["model_ref"] for e in group] == [f"opaque-index-{i}" for i in range(len(names))]
    # Drawing every chart of all twelve envelopes takes about a minute; one world's baselines, one fixture and
    # the stand-in exercise every kind of section the page has.
    some = [e for e in result.envelopes
            if e["verdict"]["world"] == "lv" and e["model_name"] in ("linear", "persistence", "fx-brittle", "standin")]
    writer.write_run(some, result.baseline_names, tmp_path / "out", 0.05, {"machine": "test"})
    page = (tmp_path / "out" / "results.html").read_text(encoding="utf-8")
    assert "FIXTURE: fx-brittle" in page and "World: lv" in page


def test_the_action_response_check_runs_on_every_model_and_catches_the_action_blind_fixture(recorded):
    result, _inputs, _starts = recorded
    for env in result.envelopes:
        assert set(env["action_response"]) == writer.ACTION_KEYS
        assert env["action_response"]["n_probes"] == 4 * 3  # probe_states x three action pairs
    blind = {(e["verdict"]["world"], e["model_name"]) for e in result.envelopes if e["action_response"]["action_blind"]}
    assert blind == {(w, n) for w in WORLDS for n in ("fx-action-blind", "persistence", "linear")}


# ---- TC-NF1-09: one JudgeInput per (model, world), every region inside -------------------------------------


def test_tc_nf1_09_each_judge_input_carries_every_region_of_its_world(recorded):
    _result, inputs, _starts = recorded
    assert len(inputs) == len(WORLDS) * len(standin_roster())  # once per (model, world), never per region
    for inp in inputs:
        regions = {label.region_name for label in inp.region_labels}
        assert len(regions) == REGIONS[inp.world]
        assert inp.outcomes.shape[:2] == (N_SMALL * REGIONS[inp.world], H_SMALL)
        assert inp.predictions.mean.shape == inp.outcomes.shape == inp.persistence.mean.shape == inp.linear.mean.shape


def test_judge_inputs_are_in_the_judges_units(recorded):
    _result, inputs, _starts = recorded
    for inp in inputs:
        assert np.array_equal(inp.thresholds.sharpness_hedge_threshold, np.ones(inp.outcomes.shape[2]))  # scale / scale
        assert inp.thresholds.bands.n == N_SMALL
    lv_inputs = [inp for inp in inputs if inp.world == "lv"]
    assert np.max(np.abs(lv_inputs[0].outcomes)) < 4.0  # prey counts reach ~12; divided by the scale 4.0 they stay small
    assert lv_inputs[0].natural_cycle_length == pytest.approx(2 * np.pi / np.sqrt(1.0 * 0.8))
    assert all(inp.natural_cycle_length is None for inp in inputs if inp.world == "pendulum")


# ---- TC-NF1-05: paired starts ---------------------------------------------------------------------------------


def test_tc_nf1_05_every_model_sees_byte_identical_starts_per_world_and_region(recorded):
    _result, _inputs, (starts_seen, kicks_seen) = recorded
    n_world_regions = sum(REGIONS.values())
    assert len(starts_seen) == len(kicks_seen) == n_world_regions * len(standin_roster())
    assert len(set(starts_seen)) == n_world_regions  # one set of starts per (world, region), shared by every model
    # and each set of starts always comes with the same kicks (kicks are sparse: two regions' can both be all zero)
    assert len(set(zip(starts_seen, kicks_seen))) == n_world_regions


def test_truth_and_labels_are_shared_across_models(recorded):
    _result, inputs, _starts = recorded
    for world in WORLDS:
        same = [inp for inp in inputs if inp.world == world]
        assert all(np.array_equal(inp.outcomes, same[0].outcomes) for inp in same)
        assert all(inp.region_labels == same[0].region_labels for inp in same)
        assert all(np.array_equal(inp.invariant_bins, same[0].invariant_bins) for inp in same)


# ---- free-running rollouts: a model is fed its own previous guess ---------------------------------------------


class _Recorder(_standin.StandInModel):
    def __init__(self, ctx, stateless=True):
        super().__init__(ctx)
        self.stateless = stateless
        self.seen, self.seen_actions, self.stated = [], [], []

    def predict_batch(self, states, actions):
        self.seen.append(np.array(states, copy=True))
        self.seen_actions.append(np.array(actions, copy=True))
        mean, _spread = super().predict_batch(states, actions)
        spread = 0.01 + 0.03 * np.abs(states)  # depends on the state, so it changes from step to step
        self.stated.append(spread.copy())
        return mean, spread


def _lv_trial_inputs(n=5, h=7):
    from wmj.harness.training import make_world_context
    from wmj.worlds import lv

    ctx = make_world_context("lv", lv.WORLD)
    rng = np.random.default_rng(3)
    warmups = np.column_stack([rng.uniform(2, 6, n), rng.uniform(1, 4, n)])
    starts = lv.WORLD.transition_batch(warmups, np.zeros((n, 1)))
    kicks = np.zeros((n, h, 1))
    kicks[:, 0, 0] = 0.05  # a kick on the first step, so a warm-up wrongly given it is visible
    kicks[:, 2, 0] = 0.05
    return ctx, warmups, starts, kicks


def test_a_model_is_shown_the_warm_up_step_then_gets_the_true_start_then_its_own_guesses_and_the_kicks():
    ctx, warmups, starts, kicks = _lv_trial_inputs()
    model = _Recorder(ctx)
    mean, spread = trials.roll_out(model, warmups, starts, kicks)
    assert mean.shape == spread.shape == (5, 7, 2)
    assert np.array_equal(model.seen[0], warmups) and not np.any(model.seen_actions[0])  # the warm-up, no kick
    assert np.array_equal(model.seen[1], starts)
    for step in range(1, 7):
        assert np.array_equal(model.seen[step + 1], mean[:, step - 1])  # its own guess, never the truth
    for step in range(7):
        assert np.array_equal(model.seen_actions[step + 1], kicks[:, step])  # the trial's own kick for that step
        assert np.array_equal(spread[:, step], model.stated[step + 1])  # the error bar it stated for that step
    assert len(model.seen) == 1 + 7  # one warm-up, then exactly once per truth step (WD-3, dt-alignment)


def test_one_trial_at_a_time_a_model_gets_each_trials_own_kicks():
    ctx, warmups, starts, kicks = _lv_trial_inputs()
    kicks[3, 5, 0] = -0.07
    model = _Recorder(ctx, stateless=False)
    trials.roll_out(model, warmups, starts, kicks)
    fed = np.array(model.seen_actions)[:, 0].reshape(5, 8, 1)  # trial by trial: the warm-up, then step by step
    assert not np.any(fed[:, 0]) and np.array_equal(fed[:, 1:], kicks)
    states = np.array(model.seen)[:, 0].reshape(5, 8, 2)
    assert np.array_equal(states[:, 0], warmups) and np.array_equal(states[:, 1], starts)


def test_batched_and_one_trial_at_a_time_rollouts_are_bit_identical():
    ctx, warmups, starts, kicks = _lv_trial_inputs()
    batched = trials.roll_out(_Recorder(ctx, stateless=True), warmups, starts, kicks)
    single = trials.roll_out(_Recorder(ctx, stateless=False), warmups, starts, kicks)
    assert all(np.array_equal(a, b) for a, b in zip(batched, single))


def test_a_stateful_model_is_reset_before_every_trial():
    ctx, warmups, starts, kicks = _lv_trial_inputs()
    model = _Recorder(ctx, stateless=False)
    resets = []
    model.reset = lambda: resets.append(len(model.seen))
    trials.roll_out(model, warmups, starts, kicks)
    assert resets == [0, 8, 16, 24, 32]  # before each trial's warm-up


@pytest.mark.parametrize(
    "bad",
    [lambda m, s: (m[:, :1], s), lambda m, s: (m * np.nan, s), lambda m, s: (m, -s), lambda m, s: (m, s * 0.0)],
    ids=["shape", "nan", "negative-spread", "zero-spread"],
)
def test_a_model_output_that_is_not_a_usable_forecast_is_refused(bad):
    ctx, warmups, starts, kicks = _lv_trial_inputs()

    class Broken(_standin.StandInModel):
        def predict_batch(self, states, actions):
            return bad(*super().predict_batch(states, actions))

    with pytest.raises(trials.TrialsError):
        trials.roll_out(Broken(ctx), warmups, starts, kicks)


# ---- MU-7, MU-2, bands, TC-MU6-05(b) --------------------------------------------------------------------------


def test_every_regions_starts_are_checked_against_the_training_starts(monkeypatch):
    calls = []
    real = trials.assert_eval_starts_disjoint
    monkeypatch.setattr(trials, "assert_eval_starts_disjoint", lambda tr, st: calls.append(st.shape) or real(tr, st))
    trials.run_trials(small_config(), dict(META), models={k: standin_roster()[k] for k in ("linear", "persistence")})
    assert calls == [(N_SMALL, 2)] * 6 + [(N_SMALL, 4)] * 6  # the drawn points and the starts, every region


def test_a_planted_training_start_among_the_evaluation_starts_is_refused(monkeypatch):
    real = trials.build_training_data
    planted = {}

    def remember(*args, **kwargs):
        data = real(*args, **kwargs)
        planted["start"] = np.array(data.states[0, 0])
        return data

    real_draw = trials.sample_region_starts

    def draw(rng, box, n):
        starts = real_draw(rng, box, n)
        starts[0] = planted["start"]
        return starts

    monkeypatch.setattr(trials, "build_training_data", remember)
    monkeypatch.setattr(trials, "sample_region_starts", draw)
    with pytest.raises(TrainingDataError, match="MU-7"):
        trials.run_trials(small_config(), dict(META), models={k: standin_roster()[k] for k in ("linear", "persistence")})


@pytest.mark.parametrize("missing", ["persistence", "linear"])
def test_a_roster_without_both_baselines_is_refused(missing):
    roster = standin_roster()
    del roster[missing]
    with pytest.raises(trials.TrialsError, match="MU-2"):
        trials.run_trials(small_config(), dict(META), models=roster)


def test_bands_for_a_different_trial_count_are_refused():
    with pytest.raises(trials.TrialsError, match="bands"):
        trials.run_trials(small_config(thresholds=_standin.small_thresholds(n=N_SMALL + 1)), dict(META), models=standin_roster())


def test_tc_mu6_05_b_a_new_baseline_is_judged_but_never_becomes_a_comparator():
    class ExtraBaseline(_standin.StandInModel):
        is_baseline = True

    roster = {k: standin_roster()[k] for k in ("linear", "persistence")}
    roster["zz-extra-baseline"] = lambda ctx, seeds, training: ExtraBaseline(ctx)
    result = trials.run_trials(small_config(), dict(META), models=roster)
    assert result.baseline_names == frozenset({"linear", "persistence", "zz-extra-baseline"})
    for env in result.envelopes:
        for entry in env["verdict"]["skill"]["per_task_region"]:
            assert {k for k in entry if k.startswith("vs_")} == {"vs_persistence", "vs_linear"}


def test_two_runs_of_the_loop_are_identical(recorded):
    result, _inputs, _starts = recorded
    again = trials.run_trials(small_config(), dict(META), models=standin_roster())
    assert again.envelopes == result.envelopes


# ---- the units rule (A30) and the shared arrays, checked against an independent rebuild ----------------------


@pytest.fixture(scope="module")
def rebuilt():
    return {w: trials.build_world_trials(w, trials.WORLDS[w], small_config()) for w in WORLDS}


def _by_model(inputs, world):
    return dict(zip(sorted(standin_roster()), [inp for inp in inputs if inp.world == world]))


@pytest.mark.parametrize("world", WORLDS)
def test_every_part_of_the_judge_input_is_the_shared_trials_in_judge_units(recorded, rebuilt, world):
    _result, inputs, _seen = recorded
    shared, scale = rebuilt[world], np.asarray(trials.WORLDS[world].scale)
    names = shared.region_names
    by_model = _by_model(inputs, world)
    for inp in by_model.values():
        assert inp.region_labels == tuple(label for r in names for label in shared.labels[r])
        blocks = [r for r in names for _ in range(N_SMALL)]  # row i of every array belongs to this region
        assert [label.region_name for label in inp.region_labels] == blocks
        assert np.array_equal(inp.outcomes, np.concatenate([shared.truth[r] for r in names]) / scale)
        assert np.array_equal(inp.invariant_bins, np.concatenate([shared.bins[r] for r in names]))
        for table, region in zip(inp.climatology, names):
            assert table.region_name == region
            for got, raw in zip(table.bins, shared.tables[region]["bins"]):
                assert np.array_equal(got.mean, np.array(raw["mean"]) / scale)
                assert np.array_equal(got.sd, np.array(raw["sd"]) / scale)
        assert [c.region_name for c in inp.divergence_curves] == list(names)
        # the baselines are the baselines, not swapped: each equals that model's own forecast
        for baseline in ("persistence", "linear"):
            assert np.array_equal(getattr(inp, baseline).mean, by_model[baseline].predictions.mean)
            assert np.array_equal(getattr(inp, baseline).spread, by_model[baseline].predictions.spread)
    starts = np.concatenate([shared.starts[r] for r in names]) / scale
    assert np.array_equal(by_model["persistence"].predictions.mean[:, -1], starts)  # "nothing changes", in judge units
    # the warm start (owner decision 2026-10-27): linear has seen one true step, so it carries on along it
    warm = np.concatenate([shared.warmups[r] for r in names]) / scale
    linear = by_model["linear"].predictions.mean
    assert np.allclose(linear[:, 0], starts + (starts - warm), rtol=1e-12, atol=0.0)
    assert np.allclose(linear[:, 1], linear[:, 0] + (linear[:, 0] - starts), rtol=1e-12, atol=0.0)
    assert not np.array_equal(linear, by_model["persistence"].predictions.mean)
    # the stand-in states an error bar of 0.02 x scale: divided by the scale, 0.02 in judge units
    assert np.allclose(by_model["standin"].predictions.spread, 0.02, rtol=1e-15, atol=0.0)


@pytest.mark.parametrize("world", WORLDS)
def test_the_shared_truth_is_the_world_stepped_with_each_trials_own_kicks(world):
    w = trials.WORLDS[world]
    shared = trials.build_world_trials(world, w, small_config(horizon=300))  # long enough for every region to be kicked
    for region in shared.region_names:
        kicks = shared.kicks[region]
        assert np.any(kicks != 0.0)  # the check below is about kicks, so there must be some
        state, expected = shared.starts[region], []
        for step in range(kicks.shape[1]):
            state = w.transition_batch(state, kicks[:, step])
            expected.append(state)
        assert np.array_equal(shared.truth[region], np.stack(expected, axis=1))


@pytest.mark.parametrize("world", WORLDS)
def test_starts_and_kicks_come_from_their_own_pinned_seed_purposes(world):
    from wmj.harness.benchmarks import declared_regions, sample_region_starts
    from wmj.harness.kicks import seeded_kick_sequences
    from wmj.models.base import SeedSource

    w = trials.WORLDS[world]
    shared = trials.build_world_trials(world, w, small_config(horizon=300))  # long enough for every region to be kicked
    seeds = SeedSource(META["seed"], None)
    for region, box, band in declared_regions(w):
        drawn = sample_region_starts(seeds.rng_for(world, region, "eval-starts"), box, N_SMALL)
        assert np.array_equal(shared.warmups[region], drawn)
        assert np.array_equal(shared.starts[region], w.transition_batch(drawn, np.zeros((N_SMALL, w.a))))  # one quiet true step
        assert np.array_equal(
            shared.kicks[region], seeded_kick_sequences(seeds, world, w, region, band, "eval-kicks", N_SMALL, 300)
        )
        train_kicks = seeded_kick_sequences(seeds, world, w, region, band, "train-kicks", N_SMALL, 300)
        assert np.any(shared.kicks[region] != 0.0)
        assert not np.array_equal(shared.kicks[region], train_kicks)  # never the training kicks (ADR-002 rule 2)


def test_each_model_gets_its_own_named_seed_stream_and_the_action_check_the_run_seed(monkeypatch):
    from wmj.models.base import SeedSource

    given, checked = {}, []
    def spying(name, factory):
        def build(ctx, seeds, training):
            given[(ctx.world_name, name)] = seeds
            return factory(ctx, seeds, training)
        return build

    roster = {k: spying(k, standin_roster()[k]) for k in ("linear", "persistence", "standin")}
    real = trials.check_action_response
    monkeypatch.setattr(trials, "check_action_response", lambda m, ctx, seed, **kw: checked.append(seed) or real(m, ctx, seed, **kw))
    trials.run_trials(small_config(), dict(META), models=roster)
    assert {key: (s.run_seed, s.my_name) for key, s in given.items()} == {
        (w, k): (META["seed"], k) for w in WORLDS for k in ("linear", "persistence", "standin")
    }
    assert all(type(s) is SeedSource for s in given.values())
    assert checked == [META["seed"]] * 6


def test_a_model_that_writes_into_its_inputs_fails_instead_of_changing_later_models_trials():
    class Mutator(_standin.StandInModel):
        def predict_batch(self, states, actions):
            actions *= 0.0  # in place: would rewrite the shared kicks every later model is fed
            return super().predict_batch(states, actions)

    roster = {k: standin_roster()[k] for k in ("linear", "persistence")}
    roster["aa-mutator"] = lambda ctx, seeds, training: Mutator(ctx)
    with pytest.raises(ValueError, match="read-only"):
        trials.run_trials(small_config(), dict(META), models=roster)


def test_the_shared_trial_arrays_are_read_only(rebuilt):
    for shared in rebuilt.values():
        for arrays in (shared.warmups, shared.starts, shared.kicks, shared.truth, shared.bins):
            assert all(not a.flags.writeable for a in arrays.values())


@pytest.mark.parametrize("value", [None, 1, "yes", np.bool_(True)], ids=["none", "int", "text", "numpy-bool"])
@pytest.mark.parametrize("attribute", ["stateless", "is_baseline", "is_fixture"])
def test_a_model_without_its_three_flags_is_refused_before_it_is_rolled_out(attribute, value):
    class Unflagged(_standin.StandInModel):
        pass

    setattr(Unflagged, attribute, value)
    roster = {k: standin_roster()[k] for k in ("linear", "persistence")}
    roster["unflagged"] = lambda ctx, seeds, training: Unflagged(ctx)
    with pytest.raises(trials.TrialsError, match=attribute):
        trials.run_trials(small_config(), dict(META), models=roster)


def test_a_pinned_baseline_that_does_not_declare_itself_one_is_refused():
    roster = {k: standin_roster()[k] for k in ("linear",)}
    roster["persistence"] = lambda ctx, seeds, training: _standin.StandInModel(ctx)
    with pytest.raises(trials.TrialsError, match="is_baseline"):
        trials.run_trials(small_config(), dict(META), models=roster)


@pytest.mark.parametrize("meta", [{}, {**META, "extra": "x"}, None])
def test_meta_without_exactly_its_four_fields_is_refused_before_anything_runs(meta):
    with pytest.raises(trials.TrialsError, match="meta"):
        trials.run_trials(small_config(), meta, models=standin_roster())


def test_a_meta_seed_that_is_not_the_run_seed_is_refused():
    roster = {k: standin_roster()[k] for k in ("linear", "persistence")}
    with pytest.raises(trials.TrialsError, match="seed"):
        trials.run_trials(small_config(), {**META, "seed": META["seed"] + 1}, models=roster)


@pytest.mark.parametrize("stateless", [True, False])
def test_a_warm_up_answer_that_is_not_a_usable_forecast_is_refused(stateless):
    ctx, warmups, starts, kicks = _lv_trial_inputs()

    class BadWarmUp(_standin.StandInModel):
        calls = 0

        def predict_batch(self, states, actions):
            BadWarmUp.calls += 1
            mean, spread = super().predict_batch(states, actions)
            return (mean * np.nan, spread) if BadWarmUp.calls == 1 else (mean, spread)

    model = BadWarmUp(ctx)
    model.stateless = stateless
    with pytest.raises(trials.TrialsError):
        trials.roll_out(model, warmups, starts, kicks)


def test_a_predict_that_returns_no_forecast_is_refused_with_a_clear_error():
    ctx, warmups, starts, kicks = _lv_trial_inputs()

    class NoForecast(_standin.StandInModel):
        stateless = False

        def predict(self, state, action):
            return (state, state)

    with pytest.raises(trials.TrialsError, match="Prediction"):
        trials.roll_out(NoForecast(ctx), warmups, starts, kicks)


def test_the_hedge_threshold_must_have_one_entry_per_quantity():
    raw = _standin.small_thresholds()
    raw["sharpness_hedge_threshold"]["pendulum"] = [3.0]
    with pytest.raises(trials.TrialsError, match="pendulum"):
        trials.judge_thresholds(raw, "pendulum", trials.WORLDS["pendulum"].scale)


def test_both_the_drawn_points_and_the_starts_are_the_arrays_checked_against_training(monkeypatch):
    checked = []
    real = trials.assert_eval_starts_disjoint
    monkeypatch.setattr(trials, "assert_eval_starts_disjoint", lambda tr, st: checked.append(np.array(st)) or real(tr, st))
    shared = trials.build_world_trials("lv", trials.WORLDS["lv"], small_config())
    expected = [a for r in shared.region_names for a in (shared.warmups[r], shared.starts[r])]
    assert len(checked) == len(expected) and all(np.array_equal(a, b) for a, b in zip(checked, expected))


# ---- the long-horizon checks: labels, climatology, curves, tasks, against independent sources ----------------


@pytest.fixture(scope="module")
def long_trials():
    config = small_config(horizon=300)
    return config, {w: trials.build_world_trials(w, trials.WORLDS[w], config) for w in WORLDS}


@pytest.mark.parametrize("world", WORLDS)
def test_every_label_is_judged_at_the_drawn_point_with_that_trials_own_kicks(long_trials, world):
    from wmj.harness.regions import label_trial

    _config, built = long_trials
    shared, spec = built[world], trials.WORLDS[world].regions()
    for region in shared.region_names:
        expected = tuple(
            trials.RegionLabel(**label_trial(spec, region, shared.warmups[region][i], shared.kicks[region][i]))
            for i in range(N_SMALL)
        )
        assert shared.labels[region] == expected
    assert any(label.axis == "action" for label in shared.labels["out-large-action"])  # the action axis does fire


def test_a_start_that_one_quiet_step_carries_out_of_the_box_is_still_labelled_by_its_drawn_point(monkeypatch):
    world = trials.WORLDS["lv"]
    box = world.regions().training_state_box
    corners = np.array([[x, y] for x in box[0] for y in box[1]])
    stepped = world.transition_batch(corners, np.zeros((4, 1)))
    outside = ~np.all((stepped >= box[:, 0]) & (stepped <= box[:, 1]), axis=1)
    assert outside.any()  # some corner of the training box is left by one step of the true world
    edge = corners[np.argmax(outside)]
    real = trials.sample_region_starts
    monkeypatch.setattr(trials, "sample_region_starts", lambda rng, b, n: np.vstack([edge, real(rng, b, n)[1:]]))
    shared = trials.build_world_trials("lv", world, small_config())
    assert not np.all((shared.starts["training"][0] >= box[:, 0]) & (shared.starts["training"][0] <= box[:, 1]))
    assert shared.labels["training"][0].axis is None  # in-region: judged at the drawn point (A43 item 6)


@pytest.mark.parametrize("world", WORLDS)
def test_each_regions_climatology_table_and_bins_come_from_that_region_in_raw_units(long_trials, world):
    from wmj.harness.climatology import build_climatology_table
    from wmj.models.base import SeedSource

    config, built = long_trials
    shared, w = built[world], trials.WORLDS[world]
    moved = 0
    for region in shared.region_names:
        table = shared.tables[region]
        assert table["reference_region"] == region
        again = build_climatology_table(
            world, w, SeedSource(META["seed"], None), n_starts=config.climatology_starts,
            steps_per_start=config.climatology_steps, n_bins=config.climatology_bins,
            min_samples=config.climatology_min_samples, region_name=region,
        )
        assert again == table
        inner = np.array([b["invariant_range"][1] for b in table["bins"][:-1]])
        raw = shared.truth[region]
        values = np.array([[w.conserved(raw[i, t]) for t in range(raw.shape[1])] for i in range(raw.shape[0])])
        expected = np.searchsorted(inner, values, side="right")
        assert np.array_equal(shared.bins[region], expected)  # measured from the raw true state, every step
        moved += int(np.sum(expected[:, 1:] != expected[:, :-1]))
    assert moved > 0  # some trial changes bin along the way, so "every step" is actually exercised
    assert len({str(shared.tables[r]["bins"]) for r in shared.region_names}) == len(shared.region_names)


@pytest.mark.parametrize("world", WORLDS)
def test_the_rest_of_the_judge_input_is_the_worlds_own_declaration(recorded, world):
    from wmj.harness.benchmarks import build_divergence_artefact
    from wmj.models.base import SeedSource

    _result, inputs, _seen = recorded
    w = trials.WORLDS[world]
    raw = _standin.small_thresholds()
    artefact = build_divergence_artefact(world, w, SeedSource(META["seed"], None), n_starts=4, horizon=H_SMALL)
    for inp in (i for i in inputs if i.world == world):
        assert inp.dt == w.dt
        assert inp.thresholds.agreement_threshold == raw["agreement_threshold"][world]
        assert inp.tasks == tuple(
            trials.TaskSpec(t.name, t.kind, t.tolerance, min(t.horizon, H_SMALL)) for t in w.tasks()
        )
        for curve in inp.divergence_curves:
            assert np.array_equal(curve.curve, np.array(artefact["regions"][curve.region_name]["median_separation"]))


# ---- the production configuration ------------------------------------------------------------------------------


def test_the_run_config_defaults_are_the_pinned_production_values():
    import dataclasses

    defaults = {f.name: f.default for f in dataclasses.fields(trials.RunConfig) if f.default is not dataclasses.MISSING}
    assert defaults == {
        "n_trials": 200,  # judge ADR-J4
        "horizon": None, "training_horizon": None,  # each world's longest task horizon
        "divergence_starts": 64,  # worlds ADR-W3
        "climatology_starts": 50, "climatology_steps": 20_000,  # D21
        "climatology_bins": 16, "climatology_min_samples": 50,  # worlds §5
        "probe_states": 16,  # MU-3 check (backlog A26)
    }
    assert {w: max(t.horizon for t in trials.WORLDS[w].tasks()) for w in WORLDS} == {"lv": 700, "pendulum": 5000}


class _Stop(Exception):
    pass


def test_with_no_horizon_given_every_generator_runs_to_the_worlds_longest_task_horizon(monkeypatch):
    seen = {}

    def stop_at_divergence(world_name, world, seeds, n_starts, horizon):
        seen["divergence"] = horizon
        raise _Stop

    monkeypatch.setattr(trials, "build_divergence_artefact", stop_at_divergence)
    with pytest.raises(_Stop):
        trials.build_world_trials("pendulum", trials.WORLDS["pendulum"], small_config(horizon=None))
    assert seen == {"divergence": 5000}


@pytest.mark.parametrize("training_horizon", [None, 60])
def test_the_training_horizon_is_passed_through_unchanged(monkeypatch, training_horizon):
    seen = []

    def stop_at_training(*args, horizon=None):
        seen.append(horizon)
        raise _Stop

    monkeypatch.setattr(trials, "build_training_data", stop_at_training)
    with pytest.raises(_Stop):
        trials.build_world_trials("lv", trials.WORLDS["lv"], small_config(training_horizon=training_horizon))
    assert seen == [training_horizon]


@pytest.mark.parametrize(
    "field, value",
    [("run_seed", "11"), ("run_seed", True), ("n_trials", 0), ("probe_states", 2.0), ("horizon", 0),
     ("training_horizon", -1), ("recipe", None), ("thresholds", [])],
)
def test_a_run_config_with_a_bad_size_is_refused(field, value):
    with pytest.raises(trials.TrialsError, match=field):
        small_config(**{field: value})


def test_thresholds_without_a_worlds_entry_are_refused_by_name():
    raw = _standin.small_thresholds()
    del raw["agreement_threshold"]["pendulum"]
    with pytest.raises(trials.TrialsError, match="pendulum"):
        trials.judge_thresholds(raw, "pendulum", trials.WORLDS["pendulum"].scale)


@pytest.mark.parametrize("stateless", [True, False])
def test_a_model_that_writes_into_the_state_it_is_given_cannot_change_its_recorded_forecast(stateless):
    ctx, warmups, starts, kicks = _lv_trial_inputs()

    class Scribbler(_standin.StandInModel):
        def predict_batch(self, states, actions):
            answer = super().predict_batch(states, actions)
            states *= 0.0  # in place, after answering
            return answer

    clean = _standin.StandInModel(ctx)
    clean.stateless = stateless
    model = Scribbler(ctx)
    model.stateless = stateless
    expected = trials.roll_out(clean, warmups, starts, kicks)
    got = trials.roll_out(model, warmups, starts, kicks)
    assert all(np.array_equal(a, b) for a, b in zip(got, expected))


def test_a_predict_batch_that_returns_no_forecast_is_refused_with_a_clear_error():
    ctx, warmups, starts, kicks = _lv_trial_inputs()

    class Nothing(_standin.StandInModel):
        def predict_batch(self, states, actions):
            return None

    with pytest.raises(trials.TrialsError, match="predict_batch"):
        trials.roll_out(Nothing(ctx), warmups, starts, kicks)
