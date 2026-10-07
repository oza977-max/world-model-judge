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

from tests.unit.judge._builders import GOOD_BANDS as BUILDER_BANDS
from tests.unit.judge._builders import band_for, blocks_copy
from wmj.judge.errors import VerdictIncompleteError
from wmj.judge.limitations import JU10_DISCLOSURES, NOT_TESTED
from wmj.judge.verdict import BLOCK_KEYS, VERDICT_SCHEMA, Verdict, assemble_verdict

TEN_FIELDS = [  # the nine JU-9 groups (calibration and sharpness count as one) as ten record fields
    "skill", "error_vs_horizon", "calibration", "sharpness", "exceptions", "trials", "climatology",
    "trust_horizons", "not_tested", "limitations",
]


def build(blocks=None, world="lv"):
    return assemble_verdict(world=world, **(blocks if blocks is not None else blocks_copy()))


# --- TC-JU9-01: the whole record, in order, with no identity -----------------------------------


def test_tc_ju9_01_a_complete_verdict_has_every_group_and_nothing_that_identifies_a_model():
    record = build().to_dict()
    assert list(record) == ["schema", "world"] + TEN_FIELDS
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


def test_a_key_that_merely_contains_in_or_out_is_fine_only_a_trailing_suffix_is_banned():
    from wmj.judge.verdict import _plain

    assert _plain({"input_count": 1, "outcome_rate": 0.1, "in_range": 2, "n_trials": 3}, "x")
    for bad in ("coverage_in", "coverage_out", "x_in_region", "x_out_region"):
        with pytest.raises(VerdictIncompleteError, match="suffix"):
            _plain({bad: 1}, "x")


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
        blocks["exceptions"]["per_task"][0]["band"] = band_for(sum(flags))
        blocks["exceptions"]["per_task"][0]["n_trials"] = n
        blocks["calibration"]["per_task"][0]["n_trials"] = n
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
    for bad in (2.0, "2", True, None, -1):
        blocks = blocks_copy()
        blocks["exceptions"]["per_task"][0]["observed"] = bad
        with pytest.raises(VerdictIncompleteError, match="observed must be nonnegint"):
            build(blocks)
    blocks = blocks_copy()
    blocks["trials"]["per_task"][0]["is_exception"] = [1, 0, 1, 0]
    with pytest.raises(VerdictIncompleteError, match="is_exception must be boollist"):
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


# --- review pass 1: a built verdict is read-only (I1) --------------------------------------------


def test_a_built_verdict_cannot_be_edited_into_an_incomplete_or_inconsistent_one():
    v = build()
    with pytest.raises(TypeError):
        v.skill["extra"] = 1
    with pytest.raises(TypeError):
        v.skill.clear()
    with pytest.raises(TypeError):
        del v.skill["per_task_region"]
    with pytest.raises(TypeError):
        v.skill.update({"x": 1})
    with pytest.raises(TypeError):
        v.skill.pop("per_task_region")
    with pytest.raises(TypeError):
        v.skill.popitem()
    with pytest.raises(TypeError):
        v.skill.setdefault("x", 1)
    with pytest.raises(TypeError):
        v.skill["per_task_region"][0]["crps"] = 5.0  # entries are read-only too
    with pytest.raises(TypeError):
        v.skill["per_task_region"][0] |= {"x": 1}
    with pytest.raises(TypeError):
        v.trials["per_task"][0]["is_exception"][0] = False  # lists became tuples
    with pytest.raises(AttributeError):
        v.calibration["per_task"].clear()
    with pytest.raises(AttributeError):
        v.trials["per_task"][0]["is_exception"].append(True)
    assert v.to_dict() == build().to_dict()  # nothing changed


def test_the_frozen_maps_survive_copying_and_pickling_still_read_only():
    import copy
    import pickle

    v = build()
    for clone in (copy.deepcopy(v.skill), pickle.loads(pickle.dumps(v.skill))):
        assert clone == v.skill
        with pytest.raises(TypeError):
            clone["x"] = 1


def test_a_verdict_can_be_rebuilt_from_its_own_groups_and_is_checked_again():
    v = build()
    again = Verdict(world=v.world, skill=v.skill, error_vs_horizon=v.error_vs_horizon, calibration=v.calibration,
                    sharpness=v.sharpness, exceptions=v.exceptions, trials=v.trials, climatology=v.climatology,
                    trust_horizons=v.trust_horizons, not_tested=v.not_tested, limitations=v.limitations)
    assert again.to_dict() == v.to_dict()
    with pytest.raises(VerdictIncompleteError):
        Verdict(world=v.world, skill={}, error_vs_horizon=v.error_vs_horizon, calibration=v.calibration,
                sharpness=v.sharpness, exceptions=v.exceptions, trials=v.trials, climatology=v.climatology,
                trust_horizons=v.trust_horizons, not_tested=v.not_tested, limitations=v.limitations)


def test_limitations_and_not_tested_are_stored_as_tuples_and_the_class_is_slotted():
    blocks = blocks_copy()
    v = Verdict(world="lv", **blocks, not_tested=list(NOT_TESTED), limitations=list(JU10_DISCLOSURES))
    assert type(v.limitations) is tuple and type(v.not_tested) is tuple
    assert not hasattr(v, "__dict__")


# --- review pass 1: every entry has exactly the spec's fields, nothing else (I2) ----------------


ENTRY_FIELDS_UNDER_TEST = {
    "skill": ("per_task_region", ["vs_persistence", "vs_linear", "crps"]),
    "error_vs_horizon": ("per_region", ["steps", "median_error", "divergence_reference"]),
    "calibration": ("per_task", ["levels", "coverage", "n_trials", "per_dimension"]),
    "sharpness": ("per_task", ["mean_width_90"]),
    "exceptions": ("per_task", ["n_trials", "expected", "observed", "band", "low_side_sharpness_flag"]),
    "trials": ("per_task", ["distance_unit", "outcome_distance", "band_lo", "band_hi", "is_exception"]),
    "climatology": ("per_task", ["switch_step", "agreement_mean_abs_z", "agrees"]),
    "trust_horizons": ("per_task", ["tolerance", "steps", "world_time", "natural_units"]),
}


@pytest.mark.parametrize(("group", "field"), [(g, f) for g, (_, fs) in ENTRY_FIELDS_UNDER_TEST.items() for f in fs])
def test_every_required_value_field_must_be_present(group, field):
    blocks = blocks_copy()
    del blocks[group][ENTRY_FIELDS_UNDER_TEST[group][0]][0][field]
    with pytest.raises(VerdictIncompleteError, match=f"missing its required field '{field}'"):
        build(blocks)


@pytest.mark.parametrize(("group", "field"), [
    (g, f) for g, (_, fs) in ENTRY_FIELDS_UNDER_TEST.items() for f in fs
    if (g, f) not in {("climatology", "switch_step"), ("climatology", "agreement_mean_abs_z"),
                      ("climatology", "agrees"), ("trust_horizons", "natural_units")}
])
def test_a_required_value_field_may_not_be_null_or_the_wrong_kind(group, field):
    for bad in (None, "n/a", object()):
        if field == "distance_unit" and bad == "n/a":
            continue  # a text field: "n/a" is a perfectly good string
        blocks = blocks_copy()
        blocks[group][ENTRY_FIELDS_UNDER_TEST[group][0]][0][field] = bad
        with pytest.raises(VerdictIncompleteError):
            build(blocks)


def test_only_the_spec_names_may_be_null_no_switch_step_no_natural_cycle():
    blocks = blocks_copy()
    blocks["climatology"]["per_task"][1].update(switch_step=None, agreement_mean_abs_z=None, agrees=None)
    blocks["trust_horizons"]["per_task"][0]["natural_units"] = None
    build(blocks)


@pytest.mark.parametrize("group", list(BLOCK_KEYS))
@pytest.mark.parametrize("extra", ["is_fixture", "model_name", "model_ref", "meta", "name", "anything_else"])
def test_a_field_the_spec_does_not_define_is_refused_so_no_identity_can_ride_along(group, extra):
    blocks = blocks_copy()
    blocks[group][BLOCK_KEYS[group][0]][0][extra] = "ensemble"
    with pytest.raises(VerdictIncompleteError, match="does not define"):
        build(blocks)
    blocks = blocks_copy()
    blocks[group][extra] = "ensemble"
    with pytest.raises(VerdictIncompleteError, match="does not define"):
        build(blocks)



def _set_bands(blocks, bands):
    """Give every exceptions entry the same `bands` (they must agree) and the band name its count earns."""
    for entry in blocks["exceptions"]["per_task"]:
        entry["bands"] = bands
        try:
            g_lo, g_hi = bands["green"]
            (a1, a2), (a3, a4) = bands["amber"]
            n = entry["observed"]
            entry["band"] = "green" if g_lo <= n <= g_hi else ("amber" if a1 <= n <= a2 or a3 <= n <= a4 else "red")
        except (TypeError, ValueError, KeyError):
            pass  # a malformed shape: leave the band name alone, the shape check is what must refuse it


GOOD_BANDS = {"green": [12, 29], "amber": [[8, 11], [30, 35]], "red": "outside"}


def test_the_bands_field_has_exactly_the_specs_shape_and_agrees_with_the_band_name():
    blocks = blocks_copy()
    entry = blocks["exceptions"]["per_task"][0]
    entry["bands"] = dict(GOOD_BANDS)
    entry["observed"], entry["band"] = 2, "red"  # 2 exceptions is below 8: red
    blocks["trials"]["per_task"][0]["is_exception"] = [True, True, False, False]
    build(blocks)
    for observed, flags, band in ((12, None, "green"), (29, None, "green"), (8, None, "amber"), (11, None, "amber"),
                                  (30, None, "amber"), (35, None, "amber"), (7, None, "red"), (36, None, "red")):
        b = blocks_copy()
        n = 40
        e = b["exceptions"]["per_task"][0]
        e.update(bands=dict(GOOD_BANDS), observed=observed, band=band, n_trials=n)
        b["calibration"]["per_task"][0]["n_trials"] = n
        b["trials"]["per_task"][0].update(
            outcome_distance=[0.1] * n, band_lo=[0.0] * n, band_hi=[0.2] * n,
            is_exception=[True] * observed + [False] * (n - observed) if observed <= n else None)
        if observed > n:
            continue
        build(b)  # the band name matches the count at every edge of every band
        wrong = {"green": "amber", "amber": "red", "red": "green"}[band]
        b["exceptions"]["per_task"][0]["band"] = wrong
        with pytest.raises(VerdictIncompleteError, match="band="):
            build(b)


@pytest.mark.parametrize("bad", [
    "green", {}, {"green": [12, 29]}, {**GOOD_BANDS, "extra": 1}, {**GOOD_BANDS, "model_name": "ens"},
    {**GOOD_BANDS, "is_fixture": True}, {**GOOD_BANDS, "green": [12]}, {**GOOD_BANDS, "green": [29, 12]},
    {**GOOD_BANDS, "green": ["a", "b"]}, {**GOOD_BANDS, "amber": [[8, 11]]}, {**GOOD_BANDS, "amber": [[8, 11], "x"]},
    {**GOOD_BANDS, "red": 3}, {**GOOD_BANDS, "red": " "}, {**GOOD_BANDS, "amber": [[8, 13], [30, 35]]},
    {**GOOD_BANDS, "amber": [[8, 11], [28, 35]]},
])
def test_a_malformed_or_identity_carrying_bands_field_is_refused(bad):
    blocks = blocks_copy()
    _set_bands(blocks, bad)
    with pytest.raises(VerdictIncompleteError):
        build(blocks)


def test_value_ranges_are_checked_where_the_spec_gives_them():
    cases = [
        ("exceptions", ("per_task", 0, "band"), "purple"),
        ("exceptions", ("per_task", 0, "n_trials"), 0),
        ("exceptions", ("per_task", 0, "observed"), -1),
        ("calibration", ("per_task", 0, "n_trials"), 0),
        ("trust_horizons", ("per_task", 0, "steps"), -1),
        ("climatology", ("per_task", 1, "switch_step"), -3),
        ("skill", ("per_task_region", 0, "crps"), "0.03"),
        ("skill", ("per_task_region", 0, "vs_linear"), True),
    ]
    for group, (lst, i, field), bad in cases:
        blocks = blocks_copy()
        blocks[group][lst][i][field] = bad
        with pytest.raises(VerdictIncompleteError):
            build(blocks)


def test_calibration_levels_and_coverage_agree_and_stay_inside_zero_one():
    for patch in ({"coverage": [0.5, 0.8, 0.9]}, {"coverage": [0.5, 0.8, 0.9, 1.2]}, {"coverage": [-0.1, 0.8, 0.9, 0.9]},
                  {"levels": [0.5, 0.8, 0.9, 1.0]}, {"levels": [0.0, 0.8, 0.9, 0.95]}):
        blocks = blocks_copy()
        blocks["calibration"]["per_task"][0].update(patch)
        with pytest.raises(VerdictIncompleteError, match="calibration"):
            build(blocks)


def test_exceptions_n_trials_must_equal_the_number_of_trial_points():
    for wrong in (1, 3, 5, 200):
        blocks = blocks_copy()
        blocks["exceptions"]["per_task"][0]["n_trials"] = wrong
        with pytest.raises(VerdictIncompleteError, match="n_trials"):
            build(blocks)


def test_trial_points_must_make_sense_non_negative_distances_and_ordered_bands():
    blocks = blocks_copy()
    blocks["trials"]["per_task"][0]["outcome_distance"][0] = -0.1
    with pytest.raises(VerdictIncompleteError, match="outcome_distance must be nonneglist"):
        build(blocks)
    blocks = blocks_copy()
    blocks["trials"]["per_task"][0]["band_lo"][1] = 0.5  # above band_hi (0.2)
    with pytest.raises(VerdictIncompleteError, match="band_lo"):
        build(blocks)


def test_error_vs_horizon_series_are_numbers_and_whole_increasing_steps():
    cases = [("steps", [0, "a", "b"]), ("steps", [0.0, 1.0, 2.0]), ("steps", [False, 1, 2]), ("steps", [0, 5, 3]),
             ("steps", [0, 1, 1]), ("median_error", ["a", "b", "c"]), ("divergence_reference", [0.0, None, 0.4])]
    for field, bad in cases:
        blocks = blocks_copy()
        blocks["error_vs_horizon"]["per_region"][0][field] = bad
        with pytest.raises(VerdictIncompleteError):
            build(blocks)


# --- review pass 1: the groups must cover the same (task, region) pairs (I3) --------------------


@pytest.mark.parametrize("group", ["skill", "calibration", "sharpness", "climatology", "trust_horizons", "exceptions", "trials"])
def test_dropping_a_task_from_any_one_group_is_a_partial_verdict_and_is_refused(group):
    blocks = blocks_copy()
    entries = blocks[group][BLOCK_KEYS[group][0]]
    blocks[group][BLOCK_KEYS[group][0]] = [e for e in entries if e["task"] != "lv-planning"]
    with pytest.raises(VerdictIncompleteError, match="different \\(task, region\\) pairs|no matching|matching"):
        build(blocks)


def test_an_extra_task_in_one_group_is_refused_and_so_is_a_region_only_one_group_knows():
    blocks = blocks_copy()
    extra = dict(blocks["sharpness"]["per_task"][0], task="lv-extra")
    blocks["sharpness"]["per_task"].append(extra)
    with pytest.raises(VerdictIncompleteError, match="different"):
        build(blocks)
    blocks = blocks_copy()
    blocks["error_vs_horizon"]["per_region"].append(dict(blocks["error_vs_horizon"]["per_region"][0], region="phantom"))
    with pytest.raises(VerdictIncompleteError, match="error_vs_horizon covers regions"):
        build(blocks)
    blocks = blocks_copy()
    blocks["error_vs_horizon"]["per_region"] = []
    with pytest.raises(VerdictIncompleteError):
        build(blocks)


def test_a_second_region_must_appear_in_every_group_or_none():
    blocks = blocks_copy()
    for group in TASK_REGION_GROUPS:
        entries = blocks[group][BLOCK_KEYS[group][0]]
        blocks[group][BLOCK_KEYS[group][0]] = entries + [dict(e, region="out-of-range") for e in entries]
    blocks["error_vs_horizon"]["per_region"].append(dict(blocks["error_vs_horizon"]["per_region"][0], region="out-of-range"))
    assert len(build(blocks).to_dict()["skill"]["per_task_region"]) == 4


TASK_REGION_GROUPS = ["skill", "calibration", "sharpness", "climatology", "trust_horizons", "exceptions", "trials"]


# --- review pass 1: typed key fields, no coercion, no raw crashes (M1, M2) ---------------------


@pytest.mark.parametrize("bad", [True, 1.0, "x", -5, None, [1]])
def test_horizon_step_must_be_a_whole_number_not_a_bool_float_or_string(bad):
    for group in ("exceptions", "trials"):
        blocks = blocks_copy()
        blocks[group]["per_task"][0]["horizon_step"] = bad
        with pytest.raises(VerdictIncompleteError):
            build(blocks)


@pytest.mark.parametrize("bad", [0, " ", ["lv"], {"a": 1}, 1.5, True])
def test_task_and_region_must_be_plain_non_blank_strings(bad):
    for field in ("task", "region"):
        blocks = blocks_copy()
        blocks["skill"]["per_task_region"][0][field] = bad
        with pytest.raises(VerdictIncompleteError):
            build(blocks)


def test_whitespace_only_world_is_refused_and_a_str_subclass_is_not_accepted():
    class S(str):
        pass

    for bad in ("  ", S("lv")):
        with pytest.raises(VerdictIncompleteError, match="world"):
            build(world=bad)


def test_exotic_numbers_and_self_referencing_data_are_refused_cleanly_not_with_a_crash():
    blocks = blocks_copy()
    blocks["skill"]["per_task_region"][0]["crps"] = np.longdouble(0.25)
    assert build(blocks).to_dict()["skill"]["per_task_region"][0]["crps"] == 0.25
    blocks = blocks_copy()
    blocks["skill"]["per_task_region"][0]["crps"] = np.longdouble("nan")
    with pytest.raises(VerdictIncompleteError, match="not finite"):
        build(blocks)
    blocks = blocks_copy()
    loop = []
    loop.append(loop)
    blocks["calibration"]["per_task"][0]["per_dimension"] = loop
    with pytest.raises(VerdictIncompleteError, match="nested"):
        build(blocks)
    blocks = blocks_copy()
    blocks["skill"]["per_task_region"][0]["crps"] = np.complex128(1)
    with pytest.raises(VerdictIncompleteError, match="cannot carry"):
        build(blocks)
    blocks = blocks_copy()
    blocks["skill"]["per_task_region"][0]["crps"] = np.str_("x")
    with pytest.raises(VerdictIncompleteError, match="cannot carry"):
        build(blocks)


def test_numpy_booleans_and_integers_are_converted_to_plain_python_values():
    blocks = blocks_copy()
    blocks["trials"]["per_task"][0]["is_exception"] = np.array([True, False, True, False])
    blocks["exceptions"]["per_task"][0]["n_trials"] = np.int64(4)
    record = build(blocks).to_dict()
    assert all(type(x) is bool for x in record["trials"]["per_task"][0]["is_exception"])
    assert type(record["exceptions"]["per_task"][0]["n_trials"]) is int


# --- boundary values and odd shapes (mutation survivors from pass 1 follow-up) --------------------


def test_zero_valued_edges_are_legitimate():
    blocks = blocks_copy()
    blocks["trials"]["per_task"][0]["outcome_distance"] = [0.0, 0.0, 0.3, 0.4]
    blocks["trials"]["per_task"][0]["band_lo"] = [0.0] * 4
    blocks["trials"]["per_task"][0]["band_hi"] = [0.0, 0.2, 0.2, 0.2]  # lo == hi is allowed
    blocks["calibration"]["per_task"][0]["coverage"] = [0.0, 0.0, 0.5, 1.0]
    blocks["trust_horizons"]["per_task"][0].update(steps=0, world_time=0.0)  # fails even at step one: 0 by convention
    build(blocks)


def test_a_numpy_boolean_scalar_is_converted_and_other_numpy_scalars_are_not_silently_taken():
    blocks = blocks_copy()
    blocks["trials"]["per_task"][0]["is_exception"] = [np.True_, np.False_, np.True_, np.False_]
    record = build(blocks).to_dict()
    assert record["trials"]["per_task"][0]["is_exception"] == [True, False, True, False]
    assert all(type(x) is bool for x in record["trials"]["per_task"][0]["is_exception"])


def test_entries_that_are_not_mappings_and_series_that_are_not_lists_are_refused():
    for bad_entry in (5, None, "lv-control", ["task"]):
        blocks = blocks_copy()
        blocks["sharpness"]["per_task"][0] = bad_entry
        with pytest.raises(VerdictIncompleteError):
            build(blocks)
    for field in ("outcome_distance", "band_lo", "band_hi", "is_exception"):
        blocks = blocks_copy()
        blocks["trials"]["per_task"][0][field] = "abcd"
        with pytest.raises(VerdictIncompleteError, match=field):
            build(blocks)


# --- review pass 2 -----------------------------------------------------------------------------


def test_iterators_cannot_smuggle_an_empty_limitations_or_not_tested_past_the_check():
    blocks = blocks_copy()
    v = Verdict(world="lv", **blocks, not_tested=iter(NOT_TESTED), limitations=iter(JU10_DISCLOSURES))
    assert v.limitations == JU10_DISCLOSURES and v.not_tested == NOT_TESTED
    v2 = Verdict(world="lv", **blocks_copy(), not_tested=(x for x in NOT_TESTED), limitations=(x for x in JU10_DISCLOSURES))
    assert len(v2.to_dict()["limitations"]) == 7
    with pytest.raises(VerdictIncompleteError, match="limitations"):
        Verdict(world="lv", **blocks_copy(), not_tested=NOT_TESTED, limitations=iter(()))


def test_per_dimension_coverage_has_one_numeric_row_per_level_of_equal_length():
    good = [[0.5, 0.5], [0.8, 0.8], [0.9, 0.9], [0.95, 0.94]]
    for bad in ([["a"]] * 4, ["a", "b", "c", "d"], [{"model_name": "ens"}] * 4, good[:3], [[0.5, 0.5]] * 3 + [[0.9]],
                [[0.5, 1.5]] * 4, [[]] * 4, [[True, 0.5]] * 4):
        blocks = blocks_copy()
        blocks["calibration"]["per_task"][0]["per_dimension"] = bad
        with pytest.raises(VerdictIncompleteError, match="per_dimension"):
            build(blocks)
    blocks = blocks_copy()
    blocks["calibration"]["per_task"][0]["per_dimension"] = [[0.5], [0.8], [0.9], [0.95]]  # one quantity
    build(blocks)


def test_calibration_levels_are_the_four_constants_in_order():
    for bad in ([0.5, 0.8, 0.9, 0.99], [0.95, 0.9, 0.8, 0.5], [0.5, 0.8, 0.9], [0.5, 0.8, 0.9, 0.95, 0.99]):
        blocks = blocks_copy()
        blocks["calibration"]["per_task"][0]["levels"] = bad
        blocks["calibration"]["per_task"][0]["coverage"] = [0.5] * len(bad)
        with pytest.raises(VerdictIncompleteError, match="levels"):
            build(blocks)


def test_a_verdict_cannot_be_subclassed_and_survives_copy_and_pickle_through_the_checks():
    import copy
    import pickle

    with pytest.raises(TypeError, match="cannot be subclassed"):
        class Tagged(Verdict):
            pass

    v = build()
    for clone in (copy.deepcopy(v), pickle.loads(pickle.dumps(v)), copy.copy(v)):
        assert clone.to_dict() == v.to_dict()
        with pytest.raises(TypeError):
            clone.skill["x"] = 1


def test_the_same_pair_dropped_from_exceptions_and_trials_together_is_still_refused():
    blocks = blocks_copy()
    for group in ("exceptions", "trials"):
        blocks[group]["per_task"] = [e for e in blocks[group]["per_task"] if e["task"] != "lv-planning"]
    with pytest.raises(VerdictIncompleteError, match="different"):
        build(blocks)


@pytest.mark.parametrize(("group", "field"), [("climatology", "agrees"), ("exceptions", "low_side_sharpness_flag")])
def test_flags_are_real_booleans_not_zero_and_one(group, field):
    blocks = blocks_copy()
    blocks[group]["per_task"][1 if group == "climatology" else 0][field] = 1
    with pytest.raises(VerdictIncompleteError, match=field):
        build(blocks)


def test_the_band_vocabulary_is_exactly_green_amber_red():
    for band in ("yellow", "Green", "", None, 1):
        blocks = blocks_copy()
        blocks["exceptions"]["per_task"][0]["band"] = band
        with pytest.raises(VerdictIncompleteError, match="band"):
            build(blocks)


def test_a_dictionary_that_contains_itself_is_refused_cleanly():
    blocks = blocks_copy()
    loop = {}
    loop["me"] = loop
    blocks["calibration"]["per_task"][0]["per_dimension"] = [loop]
    with pytest.raises(VerdictIncompleteError, match="nested"):
        build(blocks)


@pytest.mark.parametrize("bad", [[], (), 5, "x"])
def test_a_group_that_is_not_a_mapping_and_an_empty_entry_list_are_refused(bad):
    blocks = blocks_copy()
    blocks["skill"] = bad
    with pytest.raises(VerdictIncompleteError):
        build(blocks)
    blocks = blocks_copy()
    blocks["skill"]["per_task_region"] = bad
    with pytest.raises(VerdictIncompleteError):
        build(blocks)


# --- semantic consistency the spec implies ---------------------------------------------------------


def test_the_climatology_fields_are_all_null_or_all_present():
    for patch in ({"switch_step": None, "agrees": True, "agreement_mean_abs_z": 0.5},
                  {"switch_step": 4, "agrees": None, "agreement_mean_abs_z": 0.5},
                  {"switch_step": 4, "agrees": True, "agreement_mean_abs_z": None},
                  {"switch_step": None, "agrees": None, "agreement_mean_abs_z": 0.5}):
        blocks = blocks_copy()
        blocks["climatology"]["per_task"][1].update(patch)
        with pytest.raises(VerdictIncompleteError, match="all null"):
            build(blocks)


def test_world_time_is_steps_times_dt_and_negative_quantities_are_refused():
    blocks = blocks_copy()
    blocks["trust_horizons"]["per_task"][0]["world_time"] = 9.99
    with pytest.raises(VerdictIncompleteError, match="steps x dt"):
        build(blocks)
    for group, lst, i, field in (("skill", "per_task_region", 0, "crps"), ("sharpness", "per_task", 0, "mean_width_90"),
                                 ("trust_horizons", "per_task", 0, "tolerance"), ("exceptions", "per_task", 0, "expected"),
                                 ("climatology", "per_task", 1, "agreement_mean_abs_z")):
        blocks = blocks_copy()
        blocks[group][lst][i][field] = -0.01
        with pytest.raises(VerdictIncompleteError):
            build(blocks)
    blocks = blocks_copy()
    blocks["error_vs_horizon"]["per_region"][0]["median_error"][1] = -0.1
    with pytest.raises(VerdictIncompleteError):
        build(blocks)


def test_calibration_and_exceptions_describe_the_same_trial_set():
    blocks = blocks_copy()
    blocks["calibration"]["per_task"][0]["n_trials"] = 200
    with pytest.raises(VerdictIncompleteError, match="one shared trial set"):
        build(blocks)


# --- review pass 3 -----------------------------------------------------------------------------------------


def test_absurdly_large_numbers_are_refused_with_the_judges_own_error_never_a_raw_one():
    huge = 10**5000
    for group, lst, i, field in (("trust_horizons", "per_task", 0, "steps"), ("trust_horizons", "per_task", 0, "world_time"),
                                 ("exceptions", "per_task", 0, "observed"),
                                 ("calibration", "per_task", 0, "n_trials")):
        blocks = blocks_copy()
        blocks[group][lst][i][field] = huge
        with pytest.raises(VerdictIncompleteError):
            build(blocks)
    blocks = blocks_copy()
    blocks["error_vs_horizon"]["dt"] = huge
    with pytest.raises(VerdictIncompleteError):
        build(blocks)
    blocks = blocks_copy()
    blocks["skill"]["per_task_region"][0]["task"] = huge
    with pytest.raises(VerdictIncompleteError):
        build(blocks)
    with pytest.raises(VerdictIncompleteError):
        build(world=huge)


@pytest.mark.parametrize("bad", [5, 3.5, object(), True])
def test_limitations_and_not_tested_that_are_not_sequences_are_refused_cleanly(bad):
    with pytest.raises(VerdictIncompleteError):
        Verdict(world="lv", **blocks_copy(), not_tested=NOT_TESTED, limitations=bad)
    with pytest.raises(VerdictIncompleteError):
        Verdict(world="lv", **blocks_copy(), not_tested=bad, limitations=JU10_DISCLOSURES)


def test_the_stored_fixed_text_is_the_constants_themselves_even_if_a_lookalike_passes_the_comparison():
    class Liar(str):
        def __eq__(self, other):
            return True

        __hash__ = str.__hash__

    sneaky = (Liar("evil " + JU10_DISCLOSURES[0]), *JU10_DISCLOSURES[1:])
    v = Verdict(world="lv", **blocks_copy(), not_tested=NOT_TESTED, limitations=sneaky)
    assert v.limitations is JU10_DISCLOSURES and v.to_dict()["limitations"] == list(JU10_DISCLOSURES)


def test_band_ranges_must_not_overlap_and_must_be_contiguous_with_no_gap():
    base = {"green": [12, 29], "amber": [[8, 11], [30, 35]], "red": "outside"}
    ok = [base,
          {**base, "amber": [[0, 11], [30, 200]]},
          {"green": [1, 1], "amber": [[0, 0], [2, 2]], "red": "outside"}]  # degenerate one-count ranges are allowed
    for bands in ok:
        blocks = blocks_copy()
        _set_bands(blocks, bands)
        blocks["exceptions"]["per_task"][0]["band"] = "red"  # observed is 2: red for the first, see the next lines
        if bands is not base:
            blocks["exceptions"]["per_task"][0]["band"] = "red"
        try:
            build(blocks)
        except VerdictIncompleteError as err:
            assert "band=" in str(err), err  # only the band-name rule may object, not the ranges
    bad = [
        {**base, "green": [0, 10], "amber": [[0, 5], [8, 12]]},  # overlapping
        {**base, "amber": [[8, 12], [30, 35]]},  # amber-low runs into green
        {**base, "amber": [[8, 10], [30, 35]]},  # a gap between amber-low and green
        {**base, "amber": [[8, 11], [29, 35]]},  # amber-high runs into green
        {**base, "amber": [[8, 11], [31, 35]]},  # a gap between green and amber-high
        {**base, "amber": [[11, 8], [30, 35]]},  # inverted
        {**base, "amber": [[8, 11], [35, 30]]},
        {**base, "green": [29, 12]},
    ]
    for bands in bad:
        blocks = blocks_copy()
        _set_bands(blocks, bands)
        with pytest.raises(VerdictIncompleteError, match="bands"):
            build(blocks)


def test_bands_are_required_on_every_exception_entry_so_the_band_name_is_always_checked():
    blocks = blocks_copy()
    del blocks["exceptions"]["per_task"][0]["bands"]
    with pytest.raises(VerdictIncompleteError, match="missing its required field 'bands'"):
        build(blocks)


def test_the_header_trial_count_must_equal_the_number_of_points_in_both_blocks():
    for n in (3, 5, 200):
        blocks = blocks_copy()
        blocks["exceptions"]["per_task"][0]["n_trials"] = n
        blocks["calibration"]["per_task"][0]["n_trials"] = n  # keep calibration agreeing so only the points differ
        with pytest.raises(VerdictIncompleteError, match="holds 4 points|header count"):
            build(blocks)


def test_world_time_must_equal_steps_times_dt_to_a_billionth_and_exact_products_are_accepted():
    for off in (1e-6, 1e-3, 0.5, -0.5):
        blocks = blocks_copy()
        blocks["trust_horizons"]["per_task"][0]["world_time"] = 2.36 + off
        with pytest.raises(VerdictIncompleteError, match="steps x dt"):
            build(blocks)
    for dt, steps in ((0.002, 5000), (0.02, 118), (0.01, 3), (1 / 3, 7), (0.05, 0)):
        blocks = blocks_copy()
        blocks["error_vs_horizon"]["dt"] = dt
        blocks["trust_horizons"]["per_task"][0].update(steps=steps, world_time=steps * dt)
        blocks["trust_horizons"]["per_task"][1].update(steps=0, world_time=0.0)
        blocks["climatology"]["per_task"][0].update(switch_step=None, agreement_mean_abs_z=None, agrees=None)
        build(blocks)


def test_horizon_steps_are_one_based_and_trust_horizons_are_capped_at_the_switch_step():
    for group in ("exceptions", "trials"):
        blocks = blocks_copy()
        blocks[group]["per_task"][0]["horizon_step"] = 0
        with pytest.raises(VerdictIncompleteError, match="horizon_step"):
            build(blocks)
    blocks = blocks_copy()
    blocks["climatology"]["per_task"][1]["switch_step"] = 40  # the trust horizon is 41
    with pytest.raises(VerdictIncompleteError, match="switch step"):
        build(blocks)
    blocks["climatology"]["per_task"][1]["switch_step"] = 41  # equal is allowed (closed)
    build(blocks)


def test_divergence_references_and_per_dimension_coverage_edges():
    blocks = blocks_copy()
    blocks["error_vs_horizon"]["per_region"][0]["divergence_reference"][1] = -0.1
    with pytest.raises(VerdictIncompleteError):
        build(blocks)
    blocks = blocks_copy()
    blocks["calibration"]["per_task"][0]["per_dimension"] = [[0.0, 1.0]] * 4  # exactly 0 and exactly 1 are real values
    build(blocks)
    blocks = blocks_copy()
    blocks["calibration"]["per_task"][0]["per_dimension"] = [0.5, 0.8, 0.9, 0.95]  # flat: not one row per level
    with pytest.raises(VerdictIncompleteError, match="per_dimension"):
        build(blocks)
    assert BUILDER_BANDS["green"] == [12, 29]


# --- review pass 4 -------------------------------------------------------------------------------------------


def test_hostile_containers_inside_a_group_are_refused_cleanly_not_with_a_raw_error():
    class BadDict(dict):
        def items(self):
            raise RuntimeError("boom")

    class BadList(list):
        def __iter__(self):
            raise RuntimeError("boom")

    class BadRepr:
        def __repr__(self):
            raise RuntimeError("boom")

    blocks = blocks_copy()
    blocks["skill"]["per_task_region"][0] = BadDict(blocks["skill"]["per_task_region"][0])
    with pytest.raises(VerdictIncompleteError):
        build(blocks)
    blocks = blocks_copy()
    blocks["skill"]["per_task_region"] = BadList(blocks["skill"]["per_task_region"])
    with pytest.raises(VerdictIncompleteError):
        build(blocks)
    blocks = blocks_copy()
    blocks["skill"]["per_task_region"][0]["task"] = BadRepr()
    with pytest.raises(VerdictIncompleteError):
        build(blocks)
    deep = []
    for _ in range(3000):
        deep = [deep]
    blocks = blocks_copy()
    blocks["calibration"]["per_task"][0]["per_dimension"] = deep
    with pytest.raises(VerdictIncompleteError):
        build(blocks)


def test_calibration_coverage_cannot_fall_as_the_interval_widens():
    for coverage in ([1.0, 0.8, 0.9, 0.94], [0.5, 0.8, 0.9, 0.89], [0.5, 0.4, 0.9, 0.94]):
        blocks = blocks_copy()
        blocks["calibration"]["per_task"][0]["coverage"] = coverage
        with pytest.raises(VerdictIncompleteError, match="must not fall"):
            build(blocks)
    blocks = blocks_copy()
    blocks["calibration"]["per_task"][0]["coverage"] = [0.5, 0.5, 0.5, 0.5]  # flat is fine
    build(blocks)


def test_a_trust_horizon_tolerance_must_be_positive():
    for bad in (0, 0.0, -0.1):
        blocks = blocks_copy()
        blocks["trust_horizons"]["per_task"][0]["tolerance"] = bad
        with pytest.raises(VerdictIncompleteError, match="tolerance"):
            build(blocks)


def test_a_group_with_an_empty_entry_list_is_refused_by_itself_even_when_every_group_is_empty():
    blocks = blocks_copy()
    for group, (lst, _) in BLOCK_KEYS.items():
        blocks[group][lst] = []
    with pytest.raises(VerdictIncompleteError, match="missing or empty"):
        build(blocks)
    for group, (lst, _) in BLOCK_KEYS.items():
        blocks = blocks_copy()
        blocks[group][lst] = []
        with pytest.raises(VerdictIncompleteError, match="missing or empty"):
            build(blocks)


def test_dt_of_exactly_zero_is_refused_on_its_own_terms():
    blocks = blocks_copy()
    blocks["error_vs_horizon"]["dt"] = 0
    for entry in blocks["trust_horizons"]["per_task"]:
        entry.update(steps=0, world_time=0.0)  # so the steps x dt rule cannot be what refuses it
    for entry in blocks["climatology"]["per_task"]:
        entry.update(switch_step=None, agreement_mean_abs_z=None, agrees=None)
    with pytest.raises(VerdictIncompleteError, match="dt"):
        build(blocks)


@pytest.mark.parametrize("bad", [4.0, 0, -4, True, "4"])
def test_trial_counts_must_be_positive_whole_numbers_not_floats(bad):
    blocks = blocks_copy()
    blocks["calibration"]["per_task"][0]["n_trials"] = bad
    with pytest.raises(VerdictIncompleteError, match="n_trials"):
        build(blocks)


@pytest.mark.parametrize("bands", [
    {"green": [12, 29], "amber": [[8, 11], [30, 35]], "red": "outside"},
])
def test_each_band_range_is_a_pair_of_whole_non_negative_numbers_in_order(bands):
    for green in ([12.0, 29], [-1, 29], [12, 29, 30], [True, 29], "ab"):
        blocks = blocks_copy()
        _set_bands(blocks, {**bands, "green": green})
        with pytest.raises(VerdictIncompleteError, match="bands"):
            build(blocks)
    for amber in ([[8.0, 11], [30, 35]], [[-1, 11], [30, 35]], [[8, 11], [30]], [[8, 11], [30, 35], [40, 41]]):
        blocks = blocks_copy()
        _set_bands(blocks, {**bands, "amber": amber})
        with pytest.raises(VerdictIncompleteError, match="bands"):
            build(blocks)
    blocks = blocks_copy()  # a band range may start at zero (no lower exceptions possible)
    _set_bands(blocks, {"green": [1, 29], "amber": [[0, 0], [30, 35]], "red": "outside"})  # 2 and 1 against green [1, 29]
    build(blocks)


def test_world_time_precision_is_a_billionth_relative():
    for rel, ok in ((2e-9, False), (1e-10, True), (-2e-9, False), (-1e-10, True)):
        blocks = blocks_copy()
        blocks["trust_horizons"]["per_task"][0]["world_time"] = 2.36 * (1 + rel)
        if ok:
            build(blocks)
        else:
            with pytest.raises(VerdictIncompleteError, match="steps x dt"):
                build(blocks)


# --- review pass 5 hardening ----------------------------------------------------------------------------------


def test_two_exceptions_entries_with_different_bands_are_refused():
    blocks = blocks_copy()
    blocks["exceptions"]["per_task"][1]["bands"] = {"green": [12, 29], "amber": [[8, 11], [30, 36]], "red": "outside"}
    with pytest.raises(VerdictIncompleteError, match="different `bands`"):
        build(blocks)


def test_a_skill_above_one_is_impossible_and_refused_but_exactly_one_is_allowed():
    for field in ("vs_persistence", "vs_linear"):
        blocks = blocks_copy()
        blocks["skill"]["per_task_region"][0][field] = 17.0
        with pytest.raises(VerdictIncompleteError, match=field):
            build(blocks)
        blocks["skill"]["per_task_region"][0][field] = 1.0
        build(blocks)
        blocks["skill"]["per_task_region"][0][field] = -250.0  # terrible is allowed: skill has no floor
        build(blocks)


def test_the_lower_band_of_a_trial_plot_is_always_zero():
    blocks = blocks_copy()
    blocks["trials"]["per_task"][0]["band_lo"][1] = 0.05
    with pytest.raises(VerdictIncompleteError, match="band_lo is always 0"):
        build(blocks)


def test_integers_too_large_to_write_out_are_refused_in_any_number_field():
    for group, lst, i, field in (("skill", "per_task_region", 0, "crps"), ("sharpness", "per_task", 0, "mean_width_90"),
                                 ("trust_horizons", "per_task", 0, "tolerance")):
        blocks = blocks_copy()
        blocks[group][lst][i][field] = 10**5000
        with pytest.raises(VerdictIncompleteError):
            build(blocks)
        blocks[group][lst][i][field] = 2**63 + 1
        with pytest.raises(VerdictIncompleteError):
            build(blocks)
    blocks = blocks_copy()
    blocks["skill"]["per_task_region"][0]["crps"] = 2**63
    build(blocks)


def test_a_verdict_is_deliberately_unhashable_compare_with_to_dict():
    with pytest.raises(TypeError):
        hash(build())
    assert build() == build() and build().to_dict() == build().to_dict()
