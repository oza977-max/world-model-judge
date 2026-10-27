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
from tests.unit.harness._standin import META, N_SMALL, H_SMALL, small_config, standin_roster
from wmj.harness import trials
from wmj.reporting import writer

WORLDS = ("lv", "pendulum")
REGIONS = {"lv": 3, "pendulum": 3}


@pytest.fixture(scope="module")
def recorded():
    """One small run, with every judge input and every rollout's starts recorded on the way."""
    inputs, starts_seen = [], []
    real_judge, real_roll = trials.judge_one, trials.roll_out

    def judge_spy(inp):
        inputs.append(inp)
        return real_judge(inp)

    def roll_spy(model, world, starts, kicks):
        starts_seen.append(np.ascontiguousarray(starts).tobytes())
        return real_roll(model, world, starts, kicks)

    patch = pytest.MonkeyPatch()
    patch.setattr(trials, "judge_one", judge_spy)
    patch.setattr(trials, "roll_out", roll_spy)
    try:
        result = trials.run_trials(small_config(), dict(META), models=standin_roster())
    finally:
        patch.undo()
    return result, inputs, starts_seen


# ---- outside in: the loop's output is a run the real writer accepts and writes ------------------------------


def test_a_stand_in_run_produces_envelopes_the_writer_accepts_and_writes(recorded, tmp_path):
    result, _inputs, _starts = recorded
    names = sorted(standin_roster())
    assert len(result.envelopes) == len(WORLDS) * len(names)
    assert result.baseline_names == frozenset({"linear", "persistence"})
    grouped = writer.check_run(result.envelopes, result.baseline_names)
    assert sorted(grouped) == list(WORLDS)
    for world, group in grouped.items():
        assert [e["model_name"] for e in group] == names  # sorted roster order, inside each world
        assert all(e["verdict"]["world"] == world for e in group)
        assert all(e["meta"] == META for e in group)
        assert {e["model_name"] for e in group if e["is_fixture"]} == {n for n in names if n.startswith("fx-")}
        assert [e["model_ref"] for e in group] == [f"opaque-index-{i}" for i in range(len(names))]
    paths = writer.write_run(result.envelopes, result.baseline_names, tmp_path / "out", 0.05, {"machine": "test"})
    assert (tmp_path / "out" / "results.html").is_file() and paths is not None


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
    _result, _inputs, starts_seen = recorded
    n_world_regions = sum(REGIONS.values())
    assert len(starts_seen) == n_world_regions * len(standin_roster())
    assert len(set(starts_seen)) == n_world_regions  # one set of starts per (world, region), shared by every model


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
        self.seen = []

    def predict_batch(self, states, actions):
        self.seen.append(np.array(states, copy=True))
        return super().predict_batch(states, actions)


def _lv_trial_inputs(n=5, h=7):
    from wmj.harness.training import make_world_context
    from wmj.worlds import lv

    ctx = make_world_context("lv", lv.WORLD)
    rng = np.random.default_rng(3)
    starts = np.column_stack([rng.uniform(2, 6, n), rng.uniform(1, 4, n)])
    kicks = np.zeros((n, h, 1))
    kicks[:, 2, 0] = 0.05
    return ctx, lv.WORLD, starts, kicks


def test_a_model_gets_the_true_start_then_its_own_previous_guess_and_the_trials_kick():
    ctx, world, starts, kicks = _lv_trial_inputs()
    model = _Recorder(ctx)
    mean, spread = trials.roll_out(model, world, starts, kicks)
    assert mean.shape == spread.shape == (5, 7, 2)
    assert np.array_equal(model.seen[0], starts)
    for step in range(1, 7):
        assert np.array_equal(model.seen[step], mean[:, step - 1])  # its own guess, never the truth
    assert len(model.seen) == 7  # advanced exactly once per truth step (WD-3, dt-alignment)


def test_batched_and_one_trial_at_a_time_rollouts_are_bit_identical():
    ctx, world, starts, kicks = _lv_trial_inputs()
    batched = trials.roll_out(_Recorder(ctx, stateless=True), world, starts, kicks)
    single = trials.roll_out(_Recorder(ctx, stateless=False), world, starts, kicks)
    assert all(np.array_equal(a, b) for a, b in zip(batched, single))


def test_a_stateful_model_is_reset_before_every_trial():
    ctx, world, starts, kicks = _lv_trial_inputs()
    model = _Recorder(ctx, stateless=False)
    resets = []
    model.reset = lambda: resets.append(len(model.seen))
    trials.roll_out(model, world, starts, kicks)
    assert resets == [0, 7, 14, 21, 28]


@pytest.mark.parametrize(
    "bad",
    [lambda m, s: (m[:, :1], s), lambda m, s: (m * np.nan, s), lambda m, s: (m, -s), lambda m, s: (m, s * 0.0)],
    ids=["shape", "nan", "negative-spread", "zero-spread"],
)
def test_a_model_output_that_is_not_a_usable_forecast_is_refused(bad):
    ctx, world, starts, kicks = _lv_trial_inputs()

    class Broken(_standin.StandInModel):
        def predict_batch(self, states, actions):
            return bad(*super().predict_batch(states, actions))

    with pytest.raises(trials.TrialsError):
        trials.roll_out(Broken(ctx), world, starts, kicks)


# ---- MU-7, MU-2, bands, TC-MU6-05(b) --------------------------------------------------------------------------


def test_every_regions_starts_are_checked_against_the_training_starts(monkeypatch):
    calls = []
    real = trials.assert_eval_starts_disjoint
    monkeypatch.setattr(trials, "assert_eval_starts_disjoint", lambda tr, st: calls.append(st.shape) or real(tr, st))
    trials.run_trials(small_config(), dict(META), models={k: standin_roster()[k] for k in ("linear", "persistence")})
    assert calls == [(N_SMALL, 2)] * 3 + [(N_SMALL, 4)] * 3


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
    with pytest.raises(Exception, match="MU-7"):
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
