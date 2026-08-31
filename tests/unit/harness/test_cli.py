"""Outside-in acceptance test for `python -m wmj run --skeleton`.

TDD-1: this is the FIRST deliverable written for P1-C02 (Hard Gate 9's
MVP-1 user-facing slice) — every unit test in this chunk falls out from
this test's failures, not the other way around.

Covers the chunk's own acceptance criterion (build/prompts/P1-C02.md):
a clean run from checkout produces out/wmj-skeleton/0.json, and two
consecutive invocations are byte-identical.
"""

from __future__ import annotations

import json
from pathlib import Path

from wmj.harness.cli import main


def test_run_skeleton_writes_report_with_required_keys(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    exit_code = main(["run", "--skeleton"])

    assert exit_code == 0
    report_path = tmp_path / "out" / "wmj-skeleton" / "0.json"
    assert report_path.exists()

    report = json.loads(report_path.read_bytes().decode("utf-8"))
    for key in (
        "schema",
        "world",
        "seed",
        "model",
        "baseline",
        "n_trials",
        "crps_model",
        "crps_baseline",
        "skill_vs_linear",
    ):
        assert key in report, f"missing required key: {key}"

    assert report["schema"] == "wmj-skeleton/0"
    assert report["world"] == "lv"
    assert report["model"] == "persistence"
    assert report["baseline"] == "linear"
    assert report["n_trials"] > 0
    assert report["crps_model"] > 0.0
    assert report["crps_baseline"] > 0.0
    assert isinstance(report["skill_vs_linear"], float)


def test_run_skeleton_two_consecutive_runs_are_byte_identical(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    report_path = Path("out") / "wmj-skeleton" / "0.json"

    assert main(["run", "--skeleton"]) == 0
    first_bytes = (tmp_path / report_path).read_bytes()

    assert main(["run", "--skeleton"]) == 0
    second_bytes = (tmp_path / report_path).read_bytes()

    assert first_bytes == second_bytes
