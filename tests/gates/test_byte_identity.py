"""TC-NF1-01 (skeleton scope): ten separate-process runs of the
skeleton CLI produce byte-identical output.

design-review-008 pinned this to run across SEPARATE PROCESSES, not
ten in-process calls — an in-process cache filled from an unseeded
draw on first use would pass ten "consecutive runs" in one process
while varying freely across fresh ones, exactly the class of
accidental impurity this gate exists to catch (cross-cutting ADR-002
rule 3). The full pipeline's byte-identity gate (over out/verdicts/)
is a later chunk's concern; this one runs over P1-C02's
out/wmj-skeleton/0.json, the only deterministic artefact that exists
at this build stage.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
N_RUNS = 10


def _run_skeleton_in_fresh_process(cwd: Path) -> bytes:
    env = {"PATH": "/usr/bin:/bin", "PYTHONPATH": str(SRC_ROOT)}
    result = subprocess.run(
        [sys.executable, "-m", "wmj", "run", "--skeleton"],
        cwd=cwd,
        env=env,
        capture_output=True,
        check=True,
    )
    assert result.returncode == 0
    report_path = cwd / "out" / "wmj-skeleton" / "0.json"
    return report_path.read_bytes()


def test_tc_nf1_01_ten_separate_process_runs_are_byte_identical(tmp_path):
    outputs = []
    for run_index in range(N_RUNS):
        run_dir = tmp_path / f"run-{run_index}"
        run_dir.mkdir()
        outputs.append(_run_skeleton_in_fresh_process(run_dir))

    first = outputs[0]
    for run_index, output in enumerate(outputs[1:], start=1):
        assert output == first, f"run {run_index} differs from run 0"
