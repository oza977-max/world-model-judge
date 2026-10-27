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

from tests.unit.reporting._verdict_blocks import (
    FINGERPRINT,
    META,
    run_envelopes,
    set_crps,
    set_horizon,
)
from wmj.judge.limitations import JU10_DISCLOSURES
from wmj.judge.verdict import NOT_TESTED, Verdict
from wmj.reporting.writer import VerdictSchemaError

BASELINES = frozenset({"persistence", "linear"})
MARGIN = 0.05  # prereg/recipe.md's matching_margin


@pytest.fixture(scope="module")
def plain_run(tmp_path_factory):
    """One ordinary run, written once and read by several tests (a run renders about seventy charts)."""
    root = tmp_path_factory.mktemp("run")
    _write(root)
    return root / "out"


def _write(tmp_path, envelopes=None):
    from wmj.reporting.writer import write_run

    return write_run(envelopes if envelopes is not None else run_envelopes(), BASELINES, tmp_path / "out", MARGIN, FINGERPRINT)


def test_tc_rp6_01_each_verdict_is_a_schema_valid_file_beside_its_charts(plain_run):
    out = plain_run
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


def test_the_manifest_is_the_runs_four_meta_fields_plus_package_versions(plain_run):
    from importlib.metadata import version

    manifest = json.loads((plain_run / "run-manifest.json").read_text(encoding="utf-8"))
    assert manifest == {**META, "package_versions": {"numpy": version("numpy"), "matplotlib": version("matplotlib")},
                        "runtime_fingerprint": FINGERPRINT}  # owner decision 2026-10-26: the fingerprint lives here


def test_tc_rp_card_01_the_model_card_is_near_the_top_of_the_page_verbatim(plain_run):
    page = (plain_run / "results.html").read_text(encoding="utf-8")
    card = page.index("What this verdict does not tell you")
    assert card < page.index("<img")  # before any chart: never below the fold
    never = page.index("Never tested")
    for sentence in JU10_DISCLOSURES:
        assert card < page.index(f"<li>{_escaped(sentence)}</li>") < never
    for item in NOT_TESTED:
        assert never < page.index(f"<li>{_escaped(item)}</li>")


def test_tc_ju12_04_the_methods_note_states_the_purity_residual_as_measured(plain_run):
    page = (plain_run / "results.html").read_text(encoding="utf-8")
    note = page[page.index("How the judge was kept honest"):]
    for phrase in ("best-effort", "ctypes", "ten", "separate processes", "same CPU family"):
        assert phrase in note
    assert "getrandom" not in page  # a closed route is not listed as open


def test_tc_rp4_01_a_disagreement_between_real_models_is_marked_and_stated(tmp_path):
    envelopes = run_envelopes()
    by_name = {e["model_name"]: e for e in envelopes}
    _flat(envelopes)
    set_crps(by_name["direct"], 0.40)
    set_crps(by_name["ensemble"], 0.41)  # ordinary error: direct a hair ahead
    set_horizon(by_name["direct"], 1, "lv-planning")
    set_horizon(by_name["ensemble"], 2, "lv-planning")  # the judge: ensemble trusted for longer
    _write(tmp_path, envelopes)
    caption = (tmp_path / "out" / "captions" / "lv-comparison.txt").read_text(encoding="utf-8")
    assert "Ordinary error says direct and ensemble are equals; the judge does not." in caption
    svg = (tmp_path / "out" / "charts" / "lv-comparison.svg").read_text(encoding="utf-8")
    assert svg.count("disagree") == 3  # the column header and both rows


def test_tc_rp4_02_agreeing_rankings_mark_nothing(tmp_path):
    envelopes = run_envelopes()
    _flat(envelopes)
    for rank, env in enumerate(envelopes):  # error order and the horizon order the same way (ties are not disagreement)
        set_crps(env, 0.1 * (rank + 1))
        set_horizon(env, max(0, 2 - rank), "lv-planning")
    _write(tmp_path, envelopes)
    caption = (tmp_path / "out" / "captions" / "lv-comparison.txt").read_text(encoding="utf-8")
    assert "the two rankings agree" in caption
    svg = (tmp_path / "out" / "charts" / "lv-comparison.svg").read_text(encoding="utf-8")
    assert svg.count("disagree") == 1  # the column header only


def test_a_verdict_the_judge_would_refuse_writes_nothing_at_all(tmp_path):
    envelopes = run_envelopes()
    envelopes[3]["verdict"]["limitations"] = envelopes[3]["verdict"]["limitations"][:-1]
    with pytest.raises(VerdictSchemaError):
        _write(tmp_path, envelopes)
    assert list(tmp_path.iterdir()) == []


def _escaped(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;").replace("'", "&#x27;")


def _flat(envelopes):
    """Every trust horizon at 0 steps, so a test sets exactly the ones it means (the judge caps them at the switch step)."""
    for env in envelopes:
        for entry in env["verdict"]["trust_horizons"]["per_task"]:
            set_horizon(env, 0, entry["task"], entry["region"])


# --- refusals: checked before anything is written ---------------------------------------------------------


def _spoiled(change):
    envelopes = run_envelopes()
    change(envelopes)
    return envelopes


@pytest.mark.parametrize(
    "change, match",
    [
        (lambda es: es[0].update(extra=1), "exactly"),
        (lambda es: es[0].pop("model_ref"), "exactly"),
        (lambda es: es[0].update(model_ref=" "), "model_ref"),
        (lambda es: es[1]["meta"].pop("seed"), "meta must have"),
        (lambda es: es[1]["meta"].update(hostname="x"), "meta must have"),
        (lambda es: es[1]["meta"].update(seed="1"), "whole number"),
        (lambda es: es[1]["meta"].update(platform=""), "non-empty"),
        (lambda es: es[2]["meta"].update(seed=1), "meta differs"),
        (lambda es: es[0]["verdict"].update(schema="wmj-verdict/0"), "schema"),
        (lambda es: es[0]["verdict"].pop("schema"), "schema"),
        (lambda es: es[0]["verdict"].update(is_fixture=False), "fields must be exactly"),
        (lambda es: es[0]["verdict"].pop("trials"), "fields must be exactly"),
        (lambda es: es[0]["verdict"].update(not_tested=tuple(es[0]["verdict"]["not_tested"])), "plain JSON"),
        (lambda es: es[0]["verdict"]["skill"]["per_task_region"][0].update(crps=-1.0), "judge's own check"),
        (lambda es: es.pop(0), "baseline"),
        (lambda es: es.append(dict(es[2], model_ref="opaque-9")), "appears twice"),
        (lambda es: es[0].update(model_name="a/b"), "file name"),
        (lambda es: es[0].update(is_fixture=1), "fixture flag"),
    ],
)
def test_an_envelope_that_does_not_match_the_schema_is_refused_and_nothing_written(tmp_path, change, match):
    with pytest.raises(VerdictSchemaError, match=match):
        _write(tmp_path, _spoiled(change))
    assert list(tmp_path.iterdir()) == []


def test_the_run_needs_envelopes_and_a_set_of_baseline_names(tmp_path):
    from wmj.reporting.writer import write_run

    for envelopes, baselines in (([], BASELINES), (run_envelopes(), ["persistence", "linear"])):
        with pytest.raises(VerdictSchemaError):
            write_run(envelopes, baselines, tmp_path / "out", MARGIN, FINGERPRINT)
    assert list(tmp_path.iterdir()) == []


def test_a_folder_holding_files_or_a_leftover_staging_folder_is_refused(tmp_path):
    from wmj.reporting.writer import OutputFolderError

    (tmp_path / "out").mkdir()
    (tmp_path / "out" / "old.json").write_text("{}")
    with pytest.raises(OutputFolderError, match="already holds files"):
        _write(tmp_path)
    (tmp_path / "out" / "old.json").unlink()
    (tmp_path / "out.partial").mkdir()
    with pytest.raises(OutputFolderError, match="in the way"):
        _write(tmp_path)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["out", "out.partial"] and not any((tmp_path / "out").iterdir())


def test_a_failure_part_way_leaves_neither_out_nor_its_staging_folder(tmp_path, monkeypatch):
    from wmj.reporting import writer

    def broken(*args, **kwargs):
        raise RuntimeError("a chart failed after others were written")

    monkeypatch.setattr(writer, "render_calibration_chart", broken)
    with pytest.raises(RuntimeError):
        _write(tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_two_worlds_into_an_empty_folder_each_get_their_files_and_page_section(tmp_path):
    from tests.unit.reporting._verdict_blocks import full_envelope
    from wmj.reporting.writer import write_run

    envelopes = [full_envelope(name=n, seed=s, world=w, ref=f"opaque-{i}")
                 for i, (n, s, w) in enumerate((("persistence", 2, "lv"), ("linear", 3, "lv"),
                                                ("persistence", 2, "pendulum"), ("linear", 3, "pendulum")))]
    (tmp_path / "out").mkdir()
    paths = write_run(envelopes, BASELINES, tmp_path / "out", MARGIN, FINGERPRINT)
    names = sorted(p.name for p in paths.verdicts)
    assert names == ["lv-linear.json", "lv-persistence.json", "pendulum-linear.json", "pendulum-persistence.json"]
    html = paths.page.read_text(encoding="utf-8")
    assert html.index("World: lv") < html.index("World: pendulum")
    assert all(p.png.exists() and p.svg.exists() and p.caption.exists() for p in paths.charts)
    assert not (tmp_path / "out.partial").exists()


# --- review of P5-C04, pass 1 ------------------------------------------------------------------------------


def test_two_world_model_pairs_that_make_the_same_file_names_are_refused(tmp_path):
    from tests.unit.reporting._verdict_blocks import full_envelope

    envelopes = [full_envelope(name=n, seed=s, world=w, ref=f"r{i}") for i, (n, s, w) in enumerate((
        ("persistence", 2, "a"), ("linear", 3, "a"), ("b-c", 1, "a"),
        ("persistence", 2, "a-b"), ("linear", 3, "a-b"), ("c", 1, "a-b")))]
    with pytest.raises(VerdictSchemaError, match="same file names"):
        _write(tmp_path, envelopes)
    assert list(tmp_path.iterdir()) == []


def test_an_interrupt_part_way_also_leaves_nothing(tmp_path, monkeypatch):
    from wmj.reporting import writer

    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(writer, "render_exception_charts", interrupted)
    with pytest.raises(KeyboardInterrupt):
        _write(tmp_path)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("value", ["np-float", "np-array", "tuple"])
def test_values_that_are_not_plain_json_are_refused_up_front(tmp_path, value):
    import numpy as np

    envelopes = run_envelopes()
    entry = envelopes[2]["verdict"]["skill"]["per_task_region"][1]
    entry["crps"] = {"np-float": np.float64(entry["crps"]), "np-array": np.array([1.0]), "tuple": (1.0,)}[value]
    with pytest.raises(VerdictSchemaError, match="not plain JSON"):
        _write(tmp_path, envelopes)
    assert list(tmp_path.iterdir()) == []


def test_a_symlinked_or_parentless_output_folder_is_refused(tmp_path):
    from wmj.reporting.writer import OutputFolderError, write_run

    (tmp_path / "real").mkdir()
    (tmp_path / "out").symlink_to(tmp_path / "real")
    with pytest.raises(OutputFolderError, match="plain folder"):
        _write(tmp_path)
    with pytest.raises(OutputFolderError, match="parent"):
        write_run(run_envelopes(), BASELINES, tmp_path / "missing" / "deeper" / "out", MARGIN, FINGERPRINT)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["out", "real"] and not any((tmp_path / "real").iterdir())


@pytest.mark.parametrize("margin", [0.0, 1.0, 0, "0.05", None])
def test_the_matching_margin_is_checked_before_anything_is_written(tmp_path, margin):
    from wmj.reporting.writer import write_run

    with pytest.raises(VerdictSchemaError, match="matching margin"):
        write_run(run_envelopes(), BASELINES, tmp_path / "out", margin, FINGERPRINT)
    assert list(tmp_path.iterdir()) == []


def test_a_file_written_twice_or_a_chart_that_does_not_match_its_entry_stops_the_run(tmp_path, monkeypatch):
    from wmj.reporting import writer

    monkeypatch.setattr(writer, "render_calibration_chart", lambda env, c, t: writer.render_horizon_charts(env, [], c, t))
    with pytest.raises(writer.OutputFolderError, match="written twice"):
        _write(tmp_path)
    monkeypatch.undo()
    real = writer.render_exception_charts
    monkeypatch.setattr(writer, "render_exception_charts", lambda env, c, t: list(reversed(real(env, c, t))))
    with pytest.raises(writer.OutputFolderError, match="does not match its verdict entry"):
        _write(tmp_path)
    assert list(tmp_path.iterdir()) == []



def test_a_failed_final_move_also_leaves_nothing(tmp_path, monkeypatch):
    from pathlib import Path

    real = Path.replace

    def failing(self, target):
        if self.name.endswith(".partial"):
            raise OSError("the move failed")
        return real(self, target)

    monkeypatch.setattr(Path, "replace", failing)
    with pytest.raises(OSError, match="the move failed"):
        _write(tmp_path)
    assert list(tmp_path.iterdir()) == []



def test_a_nameless_output_folder_is_refused(tmp_path, monkeypatch):
    from wmj.reporting.writer import OutputFolderError, write_run

    monkeypatch.chdir(tmp_path)
    for name in (".", "sub/.."):
        with pytest.raises(OutputFolderError, match="must name a folder"):
            write_run(run_envelopes(), BASELINES, name, MARGIN, FINGERPRINT)
    assert list(tmp_path.iterdir()) == []


def test_names_that_differ_only_in_case_are_refused_as_one_file_name(tmp_path):
    envelopes = run_envelopes()
    envelopes[3]["model_name"] = "Direct"  # beside "direct": one file on a case-insensitive file system
    with pytest.raises(VerdictSchemaError, match="same file names"):
        _write(tmp_path, envelopes)
    assert list(tmp_path.iterdir()) == []



# --- code-review-004 ---------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "change, match",
    [
        (lambda r: r.pop("tolerance"), "exactly"),
        (lambda r: r.update(extra=1), "exactly"),
        (lambda r: r.update(action_blind=1), "valid MU-3"),
        (lambda r: r.update(n_probes=0), "valid MU-3"),
        (lambda r: r.update(n_responding=49), "valid MU-3"),
        (lambda r: r.update(largest_change=float("nan")), "valid MU-3"),
        (lambda r: r.update(tolerance=0.0), "valid MU-3"),
        (lambda r: r.update(action_blind=True), "contradicts"),  # 48 of 48 responded, yet called blind
        (lambda r: r.update(largest_change=1e-12), "contradicts"),  # responded, yet nothing moved past the tolerance
    ],
)
def test_an_action_response_result_that_does_not_add_up_is_refused(tmp_path, change, match):
    envelopes = run_envelopes()
    change(envelopes[2]["action_response"])
    with pytest.raises(VerdictSchemaError, match=match):
        _write(tmp_path, envelopes)
    assert list(tmp_path.iterdir()) == []


def test_each_models_action_response_is_in_its_verdict_file_and_said_plainly_on_the_page(plain_run):
    from wmj.reporting import page

    record = json.loads((plain_run / "verdicts" / "lv-direct.json").read_text(encoding="utf-8"))
    assert record["action_response"]["action_blind"] is False
    html = (plain_run / "results.html").read_text(encoding="utf-8")
    assert html.count(page.escape(page.ACTION_BASELINE)) == 2  # persistence and linear, by construction
    assert page.escape(page.ACTION_RESPONDS.format(k=48, n=48)) in html
    assert "Flagged" not in html  # no real model is action-blind in this run


def test_an_action_blind_real_model_is_flagged_and_an_action_blind_fixture_is_not(tmp_path):
    from tests.unit.reporting._verdict_blocks import action_response
    from wmj.reporting import page

    envelopes = run_envelopes()
    by = {e["model_name"]: e for e in envelopes}
    by["ensemble"]["action_response"] = action_response(blind=True)
    by["fx-brittle"]["action_response"] = action_response(blind=True)
    _write(tmp_path, envelopes)
    html = (tmp_path / "out" / "results.html").read_text(encoding="utf-8")
    assert html.count(page.escape(page.ACTION_FLAGGED.format(n=48))) == 1
    assert html.count(page.escape(page.ACTION_FIXTURE)) == 1
    ensemble = html[html.index("<h3>ensemble</h3>") + len("<h3>ensemble</h3>"):]
    assert ensemble.index("Flagged (MU-3)") < ensemble.index("<h3")  # in ensemble's own section, before the next


@pytest.mark.parametrize("fingerprint", [None, {}, {"cpu": object()}, ["linux"]])
def test_the_runtime_fingerprint_is_required_and_plain(tmp_path, fingerprint):
    from wmj.reporting.writer import write_run

    with pytest.raises(VerdictSchemaError):
        write_run(run_envelopes(), BASELINES, tmp_path / "out", MARGIN, fingerprint)
    assert list(tmp_path.iterdir()) == []


def test_an_interrupt_just_after_the_final_move_is_reported_as_itself(tmp_path, monkeypatch):
    from pathlib import Path

    real = Path.replace

    def moved_then_interrupted(self, target):
        result = real(self, target)
        if self.name.endswith(".partial"):
            raise KeyboardInterrupt
        return result

    monkeypatch.setattr(Path, "replace", moved_then_interrupted)
    with pytest.raises(KeyboardInterrupt):  # not a FileNotFoundError from cleaning a folder already moved
        _write(tmp_path)
    assert (tmp_path / "out" / "results.html").exists() and not (tmp_path / "out.partial").exists()


def test_chart2_draws_exactly_the_named_baselines_in_grey(plain_run):
    # code-review-004: feeding every real model in as a "baseline" passed every test
    import re

    from wmj.reporting import style

    for model in ("direct", "ensemble", "fx-brittle", "persistence"):
        svg = (plain_run / "charts" / f"lv-{model}-horizon.svg").read_text(encoding="utf-8")
        grey = set(re.findall(r'<text[^>]*fill: ?' + style.BASELINE_COLOUR + r'[^>]*>\s*([^<]+?)\s*</text>', svg, re.IGNORECASE))
        assert grey == BASELINES - {model}, (model, grey)
