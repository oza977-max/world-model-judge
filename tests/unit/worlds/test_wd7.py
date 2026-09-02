"""TC-WD7-01/02: byte-identical trajectories across separate processes.

TC-WD7-01: a world's trajectory, generated twice in two separate
process invocations from the same seed/state, is byte-identical.

TC-WD7-02 (negative/phantom-gate): injecting a deliberately unseeded
numpy.random call into the step function must make the byte-identity
check fail — proving it inspects something real (same phantom-gate
proof as WD-3).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
PURE_SCRIPT = Path(__file__).resolve().parent / "_pendulum_rollout_pure.py"
IMPURE_SCRIPT = Path(__file__).resolve().parent / "_pendulum_rollout_impure.py"


def _run_rollout_script(script: Path) -> bytes:
    result = subprocess.run(
        [sys.executable, str(script), str(SRC_ROOT)],
        capture_output=True,
        check=True,
    )
    return result.stdout


def test_tc_wd7_01_pendulum_trajectory_is_byte_identical_across_processes():
    first = _run_rollout_script(PURE_SCRIPT)
    second = _run_rollout_script(PURE_SCRIPT)
    assert first == second


def test_tc_wd7_02_injected_unseeded_randomness_breaks_byte_identity():
    first = _run_rollout_script(IMPURE_SCRIPT)
    second = _run_rollout_script(IMPURE_SCRIPT)
    assert first != second
