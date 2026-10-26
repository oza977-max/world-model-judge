"""P5-C04 outside-in: one call writes the whole `out/` folder, or nothing.

In plain words: these tests run the writer on verdicts the real judge produced (two baselines, two
ordinary models, one fixture) and then read the folder back the way a reader or another tool would.
They check:
- the verdict files are there and pass the judge's own check;
- the manifest says which run it was;
- the page shows the model card near the top;
- the comparison caption says what the table shows.
A bad verdict anywhere writes nothing at all.
"""

from __future__ import annotations

import json

import pytest

from tests.unit.reporting._verdict_blocks import META, run_envelopes, set_crps, set_horizon
from wmj.judge.limitations import JU10_DISCLOSURES
from wmj.judge.verdict import NOT_TESTED, Verdict

BASELINES = frozenset({"persistence", "linear"})


def _write(tmp_path, envelopes=None):
    from wmj.reporting.writer import write_run

    return write_run(envelopes if envelopes is not None else run_envelopes(), BASELINES, tmp_path / "out")


def test_tc_rp6_01_each_verdict_is_a_schema_valid_file_beside_its_charts(tmp_path):
    _write(tmp_path)
    out = tmp_path / "out"
    for env in run_envelopes():
        path = out / "verdicts" / f"lv-{env['model_name']}.json"
        record = json.loads(path.read_text(encoding="utf-8"))
        assert record == env  # the whole envelope, nothing added, nothing dropped
        verdict = dict(record["verdict"])
        assert verdict.pop("schema")
        Verdict(**verdict)  # the judge's own door accepts it back
        assert (out / "charts" / f"lv-{env['model_name']}-horizon.svg").exists()
        assert (out / "charts" / f"lv-{env['model_name']}-calibration.svg").exists()
    assert (out / "charts" / "lv-comparison.svg").exists() and (out / "captions" / "lv-comparison.txt").exists()


def test_the_manifest_is_the_runs_four_meta_fields_plus_package_versions(tmp_path):
    from importlib.metadata import version

    _write(tmp_path)
    manifest = json.loads((tmp_path / "out" / "run-manifest.json").read_text(encoding="utf-8"))
    assert manifest == {**META, "package_versions": {"numpy": version("numpy"), "matplotlib": version("matplotlib")}}


def test_tc_rp_card_01_the_model_card_is_near_the_top_of_the_page_verbatim(tmp_path):
    _write(tmp_path)
    page = (tmp_path / "out" / "results.html").read_text(encoding="utf-8")
    card = page.index("What this verdict does not tell you")
    assert card < page.index("<img")  # before any chart: never below the fold
    never = page.index("Never tested")
    for sentence in JU10_DISCLOSURES:
        assert card < page.index(_escaped(sentence)) < never
    for item in NOT_TESTED:
        assert never < page.index(_escaped(item))


def test_tc_ju12_04_the_methods_note_states_the_purity_residual_as_measured(tmp_path):
    _write(tmp_path)
    page = (tmp_path / "out" / "results.html").read_text(encoding="utf-8")
    note = page[page.index("How the judge was kept honest"):]
    for phrase in ("best-effort", "ctypes", "ten", "separate processes", "same CPU family"):
        assert phrase in note
    assert "getrandom" not in page  # a closed route is not listed as open


def test_tc_rp4_01_a_disagreement_between_real_models_is_marked_and_stated(tmp_path):
    envelopes = run_envelopes()
    by_name = {e["model_name"]: e for e in envelopes}
    set_crps(by_name["direct"], 0.40)
    set_crps(by_name["ensemble"], 0.41)  # ordinary error: direct a hair ahead
    set_horizon(by_name["direct"], 1)
    set_horizon(by_name["ensemble"], 3)  # the judge: ensemble trusted for longer
    _write(tmp_path, envelopes)
    caption = (tmp_path / "out" / "captions" / "lv-comparison.txt").read_text(encoding="utf-8")
    assert "Ordinary error says direct and ensemble are equals; the judge does not." in caption
    svg = (tmp_path / "out" / "charts" / "lv-comparison.svg").read_text(encoding="utf-8")
    assert svg.count("disagree") == 3  # the column header and both rows


def test_tc_rp4_02_agreeing_rankings_mark_nothing(tmp_path):
    envelopes = run_envelopes()
    for rank, env in enumerate(envelopes):  # error order and every horizon order the same way
        set_crps(env, 0.1 * (rank + 1))
        for task in ("lv-control", "lv-planning"):
            for region in ("training", "out-of-range"):
                set_horizon(env, 5 - rank if not env["is_fixture"] else 0, task, region)
    _write(tmp_path, envelopes)
    caption = (tmp_path / "out" / "captions" / "lv-comparison.txt").read_text(encoding="utf-8")
    assert "the two rankings agree" in caption
    svg = (tmp_path / "out" / "charts" / "lv-comparison.svg").read_text(encoding="utf-8")
    assert svg.count("disagree") == 1  # the column header only


def test_a_verdict_the_judge_would_refuse_writes_nothing_at_all(tmp_path):
    from wmj.reporting.writer import VerdictSchemaError

    envelopes = run_envelopes()
    envelopes[3]["verdict"]["limitations"] = envelopes[3]["verdict"]["limitations"][:-1]
    with pytest.raises(VerdictSchemaError):
        _write(tmp_path, envelopes)
    assert list(tmp_path.iterdir()) == []


def _escaped(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;").replace("'", "&#x27;")
