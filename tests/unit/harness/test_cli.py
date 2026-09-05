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


# --- code-review-001: the two entry-point guards proved, not assumed ---


def test_run_without_skeleton_flag_errors_loudly(tmp_path, monkeypatch):
    """code-review-001 I5: `wmj run` with no `--skeleton` is the most
    obvious invocation a user can type today and reaches the argparse
    fallthrough; it must refuse (exit 2), not silently do nothing."""
    import pytest

    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit) as excinfo:
        main(["run"])
    assert excinfo.value.code == 2


def test_adr002_rule1_main_refuses_when_the_thread_guard_is_not_satisfied(monkeypatch):
    """code-review-001 I3: `main()` *asserts* the single-thread guard on
    every entry, so a caller that bypasses `python -m wmj` is checked."""
    import pytest

    from wmj.harness.thread_guard import THREAD_ENV_VARS, ThreadGuardError

    monkeypatch.setenv(THREAD_ENV_VARS[0], "4")
    with pytest.raises(ThreadGuardError):
        main(["run", "--skeleton"])
