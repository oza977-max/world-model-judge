"""Tests for wmj.judge.verdict — the output door (JU-9, JU-10, TC-JU9-01/02/03, TC-JU10-01).

In plain words: a verdict must have all nine required groups, and the judge must refuse —
loudly — to produce one that is missing a group, has an empty or null group, carries a
number that is not finite, is keyed in a way the spec bans, or whose exception counts
disagree with the per-trial points they are made of. Each refusal is proved able to fail:
one test per rule, starting from a good verdict and breaking exactly one thing.
"""

from __future__ import annotations

import dataclasses
import json

import numpy as np
import pytest

from tests.unit.judge._builders import blocks_copy
from wmj.judge.errors import VerdictIncompleteError
from wmj.judge.limitations import JU10_DISCLOSURES, NOT_TESTED
from wmj.judge.verdict import BLOCK_KEYS, VERDICT_SCHEMA, Verdict, assemble_verdict

NINE_GROUPS = [
    "skill", "error_vs_horizon", "calibration", "sharpness", "exceptions", "trials", "climatology",
    "trust_horizons", "not_tested", "limitations",
]


def build(blocks=None, world="lv"):
    return assemble_verdict(world=world, **(blocks if blocks is not None else blocks_copy()))


# --- TC-JU9-01: the whole record, in order, with no identity -----------------------------------


def test_tc_ju9_01_a_complete_verdict_has_every_group_and_nothing_that_identifies_a_model():
    record = build().to_dict()
    assert list(record) == ["schema", "world"] + NINE_GROUPS
    assert record["schema"] == VERDICT_SCHEMA == "wmj-verdict/1"
    assert record["limitations"] == list(JU10_DISCLOSURES) and record["not_tested"] == list(NOT_TESTED)

    def keys(x):
        if isinstance(x, dict):
            for k, v in x.items():
                yield k
                yield from keys(v)
        elif isinstance(x, list):
            for v in x:
                yield from keys(v)

    assert not {"model_ref", "model_name", "is_fixture", "meta", "name"} & set(keys(record))


def test_the_verdict_has_exactly_the_documented_fields_and_is_frozen():
    assert [f.name for f in dataclasses.fields(Verdict)] == [
        "world", "skill", "error_vs_horizon", "calibration", "sharpness", "exceptions", "trials", "climatology",
        "trust_horizons", "not_tested", "limitations",
    ]
    v = build()
    with pytest.raises(dataclasses.FrozenInstanceError):
        v.world = "x"
    with pytest.raises((AttributeError, TypeError)):
        v.is_fixture = True


def test_the_record_is_json_serialisable_and_round_trips():
    record = build().to_dict()
    assert json.loads(json.dumps(record)) == record


# --- TC-JU9-02: refuse rather than emit a partial record -----------------------------------------


@pytest.mark.parametrize("group", list(BLOCK_KEYS))
def test_tc_ju9_02_a_missing_null_or_empty_group_aborts_the_run(group):
    for broken in (None, {}, {BLOCK_KEYS[group][0]: []}, {BLOCK_KEYS[group][0]: None}, [], "calibration"):
        blocks = blocks_copy()
        blocks[group] = broken
        with pytest.raises(VerdictIncompleteError):
            build(blocks)


@pytest.mark.parametrize("group", list(BLOCK_KEYS))
def test_tc_ju9_02_leaving_a_group_out_entirely_is_a_type_error_not_a_partial_verdict(group):
    blocks = blocks_copy()
    del blocks[group]
    with pytest.raises(TypeError):
        build(blocks)


def test_tc_ju9_02_the_guard_can_fail_a_verdict_with_one_uncomputable_field_is_not_emitted():
    blocks = blocks_copy()
    blocks["calibration"]["per_task"][0]["coverage"] = [float("nan")] * 4  # the calibration could not be computed
    with pytest.raises(VerdictIncompleteError, match="not finite"):
        build(blocks)


def test_the_world_name_is_required():
    for bad in ("", None, 3):
        with pytest.raises(VerdictIncompleteError, match="world"):
            build(world=bad)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf"), np.float64("nan"), np.inf])
def test_no_group_may_carry_a_number_that_is_not_finite_anywhere_in_it(bad):
    for group, path in (("skill", ("per_task_region", 0, "crps")), ("trust_horizons", ("per_task", 0, "world_time")),
                        ("error_vs_horizon", ("per_region", 0, "median_error")),
                        ("sharpness", ("per_task", 1, "mean_width_90"))):
        blocks = blocks_copy()
        node = blocks[group]
        for step in path[:-1]:
            node = node[step]
        node[path[-1]] = [0.0, bad, 0.1] if path[-1] == "median_error" else bad
        with pytest.raises(VerdictIncompleteError, match="not finite"):
            build(blocks)


def test_values_a_verdict_cannot_carry_are_refused_and_numpy_values_are_converted():
    blocks = blocks_copy()
    blocks["skill"]["per_task_region"][0]["crps"] = object()
    with pytest.raises(VerdictIncompleteError, match="cannot carry"):
        build(blocks)
    blocks = blocks_copy()
    blocks["skill"]["per_task_region"][0]["crps"] = np.float64(0.25)
    blocks["calibration"]["per_task"][0]["coverage"] = np.array([0.5, 0.8, 0.9, 0.95])
    blocks["exceptions"]["per_task"][0]["observed"] = np.int64(2)
    record = build(blocks).to_dict()
    assert type(record["skill"]["per_task_region"][0]["crps"]) is float
    assert record["calibration"]["per_task"][0]["coverage"] == [0.5, 0.8, 0.9, 0.95]
    assert type(record["exceptions"]["per_task"][0]["observed"]) is int


def test_dictionary_keys_must_be_clean_strings():
    blocks = blocks_copy()
    blocks["skill"]["per_task_region"][0][3] = 1.0
    with pytest.raises(VerdictIncompleteError, match="key"):
        build(blocks)
    blocks = blocks_copy()
    blocks["skill"]["per_task_region"][0][""] = 1.0
    with pytest.raises(VerdictIncompleteError, match="key"):
        build(blocks)


# --- the canonical keying rule -----------------------------------------------------------------


@pytest.mark.parametrize(("group", "field"), [
    (g, f) for g, (_, fields) in BLOCK_KEYS.items() for f in fields
])
def test_every_entry_must_carry_each_of_its_key_fields(group, field):
    for broken in ("delete", None, ""):
        blocks = blocks_copy()
        entry = blocks[group][BLOCK_KEYS[group][0]][0]
        if broken == "delete":
            del entry[field]
        else:
            entry[field] = broken
        with pytest.raises(VerdictIncompleteError, match=field):
            build(blocks)


def test_two_entries_for_the_same_key_are_refused():
    blocks = blocks_copy()
    blocks["skill"]["per_task_region"].append(dict(blocks["skill"]["per_task_region"][0]))
    with pytest.raises(VerdictIncompleteError, match="same key"):
        build(blocks)


def test_the_keying_table_is_the_specs():
    assert BLOCK_KEYS == {
        "skill": ("per_task_region", ("task", "region")),
        "error_vs_horizon": ("per_region", ("region",)),
        "calibration": ("per_task", ("task", "region")),
        "sharpness": ("per_task", ("task", "region")),
        "exceptions": ("per_task", ("task", "region", "horizon_step")),
        "trials": ("per_task", ("task", "region", "horizon_step")),
        "climatology": ("per_task", ("task", "region")),
        "trust_horizons": ("per_task", ("task", "region")),
    }


@pytest.mark.parametrize("suffix", ["_in", "_out", "_in_region", "_out_region"])
def test_suffix_encoded_axes_are_banned_anywhere_in_a_group(suffix):
    blocks = blocks_copy()
    blocks["calibration"]["per_task"][0]["coverage" + suffix] = [0.5]
    with pytest.raises(VerdictIncompleteError, match="suffix"):
        build(blocks)
    blocks = blocks_copy()
    blocks["trust_horizons"]["per_task"][0]["deeply"] = {"nested" + suffix: 1}
    with pytest.raises(VerdictIncompleteError, match="suffix"):
        build(blocks)


def test_a_key_that_merely_contains_in_or_out_is_fine():
    blocks = blocks_copy()
    blocks["calibration"]["per_task"][0]["input_count"] = 4
    blocks["calibration"]["per_task"][0]["outcome_rate"] = 0.1
    build(blocks)


# --- error_vs_horizon's own contract -----------------------------------------------------------


@pytest.mark.parametrize("dt", [None, 0, -0.02, "0.02", True, float("nan")])
def test_error_vs_horizon_needs_a_positive_dt_at_block_level(dt):
    blocks = blocks_copy()
    blocks["error_vs_horizon"]["dt"] = dt
    with pytest.raises(VerdictIncompleteError):
        build(blocks)
    blocks = blocks_copy()
    del blocks["error_vs_horizon"]["dt"]
    with pytest.raises(VerdictIncompleteError, match="dt"):
        build(blocks)


def test_error_vs_horizon_series_must_exist_agree_in_length_and_start_at_zero():
    for field in ("steps", "median_error", "divergence_reference"):
        for broken in (None, []):
            blocks = blocks_copy()
            blocks["error_vs_horizon"]["per_region"][0][field] = broken
            with pytest.raises(VerdictIncompleteError, match=field):
                build(blocks)
        blocks = blocks_copy()
        blocks["error_vs_horizon"]["per_region"][0][field] = [0.0, 0.1]  # shorter than the others
        if field == "steps":
            blocks["error_vs_horizon"]["per_region"][0][field] = [0, 1]
        with pytest.raises(VerdictIncompleteError, match="length"):
            build(blocks)
    blocks = blocks_copy()
    blocks["error_vs_horizon"]["per_region"][0]["steps"] = [1, 2, 3]
    with pytest.raises(VerdictIncompleteError, match="start at 0"):
        build(blocks)


# --- TC-JU9-03: exceptions are made of the per-trial points ------------------------------------


def test_tc_ju9_03_observed_must_equal_the_sum_of_the_trials_is_exception_flags():
    assert blocks_copy()["exceptions"]["per_task"][0]["observed"] == 2  # sanity: the good case sums to 2
    for wrong in (0, 1, 3, 4):
        blocks = blocks_copy()
        blocks["exceptions"]["per_task"][0]["observed"] = wrong
        with pytest.raises(VerdictIncompleteError, match="TC-JU9-03"):
            build(blocks)
    blocks = blocks_copy()
    blocks["trials"]["per_task"][0]["is_exception"] = [True, True, True, False]  # now sums to 3, observed is 2
    with pytest.raises(VerdictIncompleteError, match="TC-JU9-03"):
        build(blocks)


def test_tc_ju9_03_property_holds_over_many_random_verdicts():
    rng = np.random.default_rng(11)
    for _ in range(60):
        n = int(rng.integers(1, 40))
        flags = (rng.random(n) < rng.random()).tolist()
        blocks = blocks_copy()
        trial = blocks["trials"]["per_task"][0]
        trial.update(outcome_distance=[0.1] * n, band_lo=[0.0] * n, band_hi=[0.2] * n, is_exception=flags)
        blocks["exceptions"]["per_task"][0]["observed"] = sum(flags)
        record = build(blocks).to_dict()
        for e in record["exceptions"]["per_task"]:
            matching = [t for t in record["trials"]["per_task"]
                        if (t["task"], t["region"], t["horizon_step"]) == (e["task"], e["region"], e["horizon_step"])]
            assert e["observed"] == sum(matching[0]["is_exception"])


def test_exceptions_and_trials_must_pair_up_one_to_one():
    blocks = blocks_copy()
    blocks["exceptions"]["per_task"].append({**blocks["exceptions"]["per_task"][0], "horizon_step": 99})
    with pytest.raises(VerdictIncompleteError, match="no matching trials"):
        build(blocks)
    blocks = blocks_copy()
    blocks["trials"]["per_task"].append({**blocks["trials"]["per_task"][0], "horizon_step": 99})
    with pytest.raises(VerdictIncompleteError, match="matching exceptions"):
        build(blocks)


def test_observed_must_be_a_whole_number_and_is_exception_must_be_booleans():
    for bad in (2.0, "2", True, None):
        blocks = blocks_copy()
        blocks["exceptions"]["per_task"][0]["observed"] = bad
        with pytest.raises(VerdictIncompleteError, match="whole number"):
            build(blocks)
    blocks = blocks_copy()
    blocks["trials"]["per_task"][0]["is_exception"] = [1, 0, 1, 0]
    with pytest.raises(VerdictIncompleteError, match="true/false"):
        build(blocks)


@pytest.mark.parametrize("field", ["outcome_distance", "band_lo", "band_hi", "is_exception"])
def test_the_per_trial_arrays_must_exist_and_have_equal_lengths(field):
    for broken in (None, []):
        blocks = blocks_copy()
        blocks["trials"]["per_task"][0][field] = broken
        with pytest.raises(VerdictIncompleteError, match=field):
            build(blocks)
    blocks = blocks_copy()
    blocks["trials"]["per_task"][0][field] = blocks["trials"]["per_task"][0][field][:-1]
    with pytest.raises(VerdictIncompleteError, match="differ in length"):
        build(blocks)


# --- TC-JU10-01: the fixed text cannot be altered ------------------------------------------------


def test_tc_ju10_01_every_verdict_carries_the_seven_disclosures_verbatim():
    assert build().limitations == JU10_DISCLOSURES and len(build().limitations) == 7


def test_the_limitations_and_not_tested_text_is_checked_word_for_word_on_direct_construction():
    good = dict(world="lv", **blocks_copy())
    Verdict(**good, not_tested=NOT_TESTED, limitations=JU10_DISCLOSURES)
    for bad in (None, (), JU10_DISCLOSURES[:-1], JU10_DISCLOSURES[::-1],
                JU10_DISCLOSURES[:6] + (JU10_DISCLOSURES[6] + " ",),
                tuple(s.replace("middle", "tails") for s in JU10_DISCLOSURES)):
        with pytest.raises(VerdictIncompleteError, match="limitations"):
            Verdict(**{**good, **blocks_copy()}, not_tested=NOT_TESTED, limitations=bad)
    for bad in (None, (), NOT_TESTED[:-1]):
        with pytest.raises(VerdictIncompleteError, match="not_tested"):
            Verdict(**{**good, **blocks_copy()}, not_tested=bad, limitations=JU10_DISCLOSURES)


def test_assemble_verdict_cannot_be_given_its_own_limitations():
    with pytest.raises(TypeError):
        assemble_verdict(world="lv", limitations=("nothing to see",), **blocks_copy())


# --- independence of the result from the inputs ---------------------------------------------------


def test_the_verdict_holds_its_own_copy_and_to_dict_returns_fresh_copies():
    blocks = blocks_copy()
    verdict = build(blocks)
    blocks["skill"]["per_task_region"][0]["crps"] = 999.0  # the caller edits after the fact
    assert verdict.skill["per_task_region"][0]["crps"] == 0.03
    out = verdict.to_dict()
    out["skill"]["per_task_region"][0]["crps"] = 5.0
    out["limitations"].append("extra")
    assert verdict.skill["per_task_region"][0]["crps"] == 0.03 and len(verdict.limitations) == 7
    assert verdict.to_dict() != out
