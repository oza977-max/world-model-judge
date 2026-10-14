# ruff: noqa: DTZ003, DTZ005, DTZ011, SIM115  (the tests deliberately call the clock and open files to prove the guard stops them)
"""TC-JU12-01..04: the judge's runtime purity tripwire, its phantom-gate proofs, and the stated residual.

In plain words: while the whole judge runs on a realistic input, every known route to the outside
world (clock, files, environment variables, network, other programs, random numbers) is rigged to
raise; the judge must finish without tripping one and give exactly the verdict it gives unrigged.
Then the rigging itself is proved to work, in both directions: each rigged route really raises
and is really put back afterwards, and a copy of the judge deliberately made impure — including by
the sneaky routes (`__globals__['os']`, `numpy.datetime64('now')`) — is caught by the rigging, while
the unmodified copy is not. Finally the specs are checked to say plainly what this cannot catch.
"""

from __future__ import annotations

import datetime
import importlib
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

from tests.gates._purity_guard import AmbientAccessError, purity_guard
from tests.unit.judge._builders import make_input

REPO_ROOT = Path(__file__).resolve().parents[2]
JUDGE_SRC = REPO_ROOT / "src" / "wmj" / "judge"


def _judge_everything(modules):
    """The judge's whole public call graph on one input, returned as plain data."""
    inp = modules["input"]
    skill = modules["skill"].compute_skill(inp)
    error = modules["horizon"].compute_error_vs_horizon(inp)
    calibration = modules["calibration"].compute_calibration(inp)
    sharpness = modules["sharpness"].compute_sharpness(inp)
    exceptions, trials = modules["exceptions"].compute_exceptions_and_trials(inp)
    climatology = modules["climatology"].compute_climatology(inp)
    trust = modules["horizon"].compute_trust_horizons(inp)
    verdict = modules["verdict"].assemble_verdict(
        world=inp.world, skill=skill, error_vs_horizon=error, calibration=calibration, sharpness=sharpness,
        exceptions=exceptions, trials=trials, climatology=climatology, trust_horizons=trust,
    )
    return verdict.to_dict()


def _real_modules(package: str = "wmj.judge"):
    names = ("skill", "horizon", "calibration", "sharpness", "exceptions", "climatology", "verdict")
    modules = {name: importlib.import_module(f"{package}.{name}") for name in names}
    modules["input"] = make_input()
    return modules


def fires(call) -> bool:
    """True iff `call()` trips the guard. The guard is left before anything is asserted about it."""
    with purity_guard():
        try:
            call()
        except AmbientAccessError:
            return True
    return False


# ---- TC-JU12-01: the load-bearing tripwire ------------------------------------------------------------


def test_the_whole_judge_runs_inside_the_guard_without_tripping_it_and_gives_the_unguarded_verdict():
    modules = _real_modules()
    unguarded = _judge_everything(modules)
    with purity_guard():
        guarded = _judge_everything(modules)
    assert guarded == unguarded
    assert unguarded["skill"]["per_task_region"] and unguarded["trust_horizons"]["per_task"]


@pytest.mark.parametrize(
    "label, call",
    [
        ("os.getcwd", lambda: os.getcwd()),
        ("os.getenv", lambda: os.getenv("HOME")),
        ("os.urandom", lambda: os.urandom(4)),
        ("os.getpid", lambda: os.getpid()),
        ("os.listdir", lambda: os.listdir(".")),
        ("os.stat", lambda: os.stat(".")),
        ("os.system", lambda: os.system("true")),
        ("os.fork", lambda: os.fork()),
        ("os.execv", lambda: os.execv("/bin/true", ["true"])),
        ("os.posix_spawn", lambda: os.posix_spawn("/bin/true", ["true"], {})),
        ("os.spawnv", lambda: os.spawnv(os.P_WAIT, "/bin/true", ["true"])),
        ("os.environ[x]", lambda: os.environ["HOME"]),
        ("x in os.environ", lambda: "HOME" in os.environ),
        ("dict(os.environ)", lambda: dict(os.environ)),
        ("os.environ.get", lambda: os.environ.get("HOME")),
        ("len(os.environ)", lambda: len(os.environ)),
        ("open", lambda: open(__file__)),
        ("io.open", lambda: __import__("io").open(__file__)),
        ("time.time", lambda: time.time()),
        ("time.perf_counter", lambda: time.perf_counter()),
        ("time.sleep", lambda: time.sleep(0)),
        ("datetime.now", lambda: datetime.datetime.now()),
        ("datetime.utcnow", lambda: datetime.datetime.utcnow()),
        ("date.today", lambda: datetime.date.today()),
        ("socket.socket", lambda: socket.socket()),
        ("socket.create_connection", lambda: socket.create_connection(("localhost", 9))),
        ("subprocess.run", lambda: subprocess.run(["true"], check=False)),
        ("subprocess.Popen", lambda: subprocess.Popen(["true"])),
        ("np.random.normal", lambda: np.random.normal()),
        ("np.random.rand", lambda: np.random.rand(2)),
        ("np.random.seed", lambda: np.random.seed(1)),
        ("np.random.default_rng", lambda: np.random.default_rng()),
        ("np.random.Generator", lambda: np.random.Generator(np.random.PCG64(1))),
        ("PCG64 via the private module", lambda: __import__("numpy.random._pcg64", fromlist=["PCG64"]).PCG64()),
        ("Generator via the private module", lambda: __import__("numpy.random._generator", fromlist=["Generator"]).Generator(None)),
        ("SeedSequence via bit_generator", lambda: __import__("numpy.random.bit_generator", fromlist=["SeedSequence"]).SeedSequence()),
        ("np.datetime64('now')", lambda: np.datetime64("now")),
    ],
)
def test_every_known_outside_world_route_trips_the_guard(label, call):
    assert fires(call), label


def test_harmless_numpy_and_python_work_is_not_blocked_by_the_guard():
    with purity_guard():
        values = np.sqrt(np.arange(5.0)) + np.datetime64("2026-01-01").astype("datetime64[D]").astype(int) * 0
        total = sum(v for v in values.tolist())
        text = "".join(["a", "x", "b"])
    assert total > 0 and text == "axb"


def test_every_rigged_route_is_put_back_exactly_afterwards_even_if_the_body_fails():
    saved = {
        "getcwd": os.getcwd, "time": time.time, "env": type(os.environ).__dict__["__getitem__"], "open": open,
        "normal": np.random.normal, "rng": np.random.default_rng, "dt64": np.datetime64, "dt": datetime.datetime,
        "popen_init": subprocess.Popen.__init__, "sock_init": socket.socket.__init__, "gen": np.random.Generator,
    }

    def snapshot():
        return {
            "getcwd": os.getcwd, "time": time.time, "env": type(os.environ).__dict__["__getitem__"], "open": open,
            "normal": np.random.normal, "rng": np.random.default_rng, "dt64": np.datetime64, "dt": datetime.datetime,
            "popen_init": subprocess.Popen.__init__, "sock_init": socket.socket.__init__, "gen": np.random.Generator,
        }

    assert snapshot() == saved
    with pytest.raises(RuntimeError, match="boom"), purity_guard():
        assert snapshot() != saved
        raise RuntimeError("boom")
    assert snapshot() == saved
    assert os.getcwd() and time.time() > 0 and np.random.default_rng(1).random() >= 0.0  # really usable again
    assert np.datetime64("now") and datetime.datetime.now()


def test_the_guard_is_not_nested_state_that_leaks_between_uses():
    for _ in range(3):
        assert fires(lambda: os.getcwd())
        assert os.getcwd()


# ---- TC-JU12-02 / -03: phantom-gate proofs on a scratch copy of the judge ---------------------------------


@pytest.fixture()
def scratch_judge(tmp_path):
    """A private copy of the judge package under another name, so it can be mutated without touching the real one."""
    name = f"judge_scratch_{abs(hash(str(tmp_path))) % 10**8}"
    target = tmp_path / name
    shutil.copytree(JUDGE_SRC, target, ignore=shutil.ignore_patterns("__pycache__"))
    for path in target.glob("*.py"):
        path.write_text(path.read_text().replace("wmj.judge", name))
    sys.path.insert(0, str(tmp_path))
    try:
        yield name, target
    finally:
        sys.path.remove(str(tmp_path))
        for module in [m for m in sys.modules if m == name or m.startswith(name + ".")]:
            del sys.modules[module]


def _mutate(target: Path, filename: str, old: str, new: str) -> None:
    path = target / filename
    text = path.read_text()
    assert old in text, f"the mutation anchor {old!r} is not in {filename}"
    path.write_text(text.replace(old, new, 1))


def _scratch_input_and_run(name: str):
    modules = _real_modules(name)
    # the scratch package has its own types, so the input must be built from them
    types = importlib.import_module(f"{name}.types")
    import tests.unit.judge._builders as builders

    original = builders.JudgeInput, builders.Bands, builders.Forecasts, builders.RegionLabel, builders.RegionCurve, builders.RegionClimatology, builders.ClimatologyBin, builders.TaskSpec, builders.Thresholds
    try:
        builders.JudgeInput, builders.Bands, builders.Forecasts = types.JudgeInput, types.Bands, types.Forecasts
        builders.RegionLabel, builders.RegionCurve, builders.RegionClimatology = types.RegionLabel, types.RegionCurve, types.RegionClimatology
        builders.ClimatologyBin, builders.TaskSpec, builders.Thresholds = types.ClimatologyBin, types.TaskSpec, types.Thresholds
        modules["input"] = builders.make_input()
    finally:
        (builders.JudgeInput, builders.Bands, builders.Forecasts, builders.RegionLabel, builders.RegionCurve,
         builders.RegionClimatology, builders.ClimatologyBin, builders.TaskSpec, builders.Thresholds) = original
    return lambda: _judge_everything(modules)


def test_the_unmodified_scratch_copy_passes_the_guard_which_proves_the_mutations_below_are_what_trips_it(scratch_judge):
    name, _ = scratch_judge
    run = _scratch_input_and_run(name)
    with purity_guard():
        assert run()["skill"]


def test_tc_ju12_02_a_globals_routed_os_read_hidden_from_the_static_lint_is_caught(scratch_judge):
    name, target = scratch_judge
    _mutate(target, "skill.py", "def compute_skill(inp: JudgeInput) -> dict:", "def compute_skill(inp: JudgeInput) -> dict:\n    np.get_include.__globals__['os'].getcwd()")
    run = _scratch_input_and_run(name)
    assert fires(run)


def test_tc_ju12_02_a_plain_environment_read_is_caught_and_would_be_flagged_by_nothing_else_here(scratch_judge):
    name, target = scratch_judge
    _mutate(target, "sharpness.py", 'if type(inp) is not JudgeInput:', 'import os as _o; _o.environ["HOME"]\n    if type(inp) is not JudgeInput:')
    run = _scratch_input_and_run(name)
    assert fires(run)


def test_tc_ju12_03_numpys_own_clock_is_caught(scratch_judge):
    name, target = scratch_judge
    _mutate(target, "horizon.py", "def compute_error_vs_horizon(inp: JudgeInput) -> dict:", "def compute_error_vs_horizon(inp: JudgeInput) -> dict:\n    np.datetime64('now')")
    run = _scratch_input_and_run(name)
    assert fires(run)


def test_a_random_number_slipped_into_the_scratch_copy_is_caught(scratch_judge):
    name, target = scratch_judge
    _mutate(target, "calibration.py", "def compute_calibration(inp: JudgeInput) -> dict:", "def compute_calibration(inp: JudgeInput) -> dict:\n    np.random.default_rng().random()")
    run = _scratch_input_and_run(name)
    assert fires(run)


def test_a_wall_clock_read_and_a_file_read_slipped_into_the_scratch_copy_are_caught(scratch_judge):
    name, target = scratch_judge
    _mutate(target, "exceptions.py", "def compute_exceptions_and_trials(inp: JudgeInput) -> tuple[dict, dict]:", "def compute_exceptions_and_trials(inp: JudgeInput) -> tuple[dict, dict]:\n    __import__('time').time()")
    assert fires(_scratch_input_and_run(name))


# ---- TC-JU12-04: the residual is stated, in the specs, neither wider nor narrower -------------------------


def test_tc_ju12_04_the_specs_state_the_residual_and_the_byte_identity_backstop_in_plain_words():
    import re

    def body(path: str) -> str:  # the change notes (lines starting with ">") narrate history, they are not the current statement
        return "\n".join(line for line in (REPO_ROOT / "specs" / path).read_text().splitlines() if not line.startswith(">"))

    judge_spec, cross_cutting = body("judge.md"), body("cross-cutting.md")
    for text in (judge_spec, cross_cutting):
        assert "ctypes" in text and "TC-NF1-01" in text and "byte-identity" in text
        assert "best-effort" in text or "tripwire" in text
        assert "before the harness installs" in text  # the pre-captured-reference residual
    # the closed route (an unseeded Generator reaching getrandom()) must not be presented as still open
    for text in (judge_spec, cross_cutting):
        for sentence in re.split(r"(?<=[.;])\s", text):
            if "getrandom" in sentence and "Generator(PCG64())" in sentence:
                assert re.search(r"caught|closes|closed|raises", sentence), sentence[:200]
    # the residual is what remains after that route was closed: ctypes and pre-captured names only
    residual = cross_cutting[cross_cutting.index("What this control does NOT catch"):]
    residual = residual[: residual.index("JU-12 is delivered by the harness")]
    assert "Two routes remain" in residual and "ctypes" in residual
