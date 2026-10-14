# ruff: noqa: DTZ002, DTZ003, DTZ005, DTZ011, SIM115  (the tests deliberately call the clock and open files to prove the guard stops them)
"""TC-JU12-01..04: the judge's runtime purity tripwire, its phantom-gate proofs, and the stated residual.

In plain words: while the whole judge runs — building its input, refusing a bad one, and computing every
block — every known route to the outside world (clock, files, environment variables, network, other programs,
random numbers) is rigged to raise; the judge must finish without tripping one, even by an error it tries to
swallow, and give exactly the verdict it gives unrigged. Then the rigging itself is proved to work: a table-
driven check that every named route (and every place numpy keeps its own reference to a random-number class or
to `datetime64`) is rigged, a before-and-after snapshot that everything is put back exactly, and copies of the
judge deliberately made impure — one hidden from the static source scan, one swallowing the error, one inside
the input checks — that the rigging catches while the unmodified copy passes. Finally the specs are checked to
state, in the right paragraph, what this cannot catch. Nothing here executes a dangerous call for real: process
replacement and forking are proved by identity or by an impossible target, so a broken guard fails loudly
instead of replacing the test process.
"""

from __future__ import annotations

import ast
import datetime
import importlib
import io
import os
import random as stdlib_random
import re
import shutil
import socket
import subprocess
import sys
import time
import uuid
from pathlib import Path

import numpy as np
import pytest

from tests.gates import _purity_guard as guard
from tests.gates._purity_guard import AmbientAccessError, installed, purity_guard
from tests.gates.test_import_graph import run_judge_gate
from tests.unit.judge._builders import judge_input_kwargs
from wmj.judge.errors import JudgeInputError

REPO_ROOT = Path(__file__).resolve().parents[2]
JUDGE_SRC = REPO_ROOT / "src" / "wmj" / "judge"
BLOCKS = ("skill", "horizon", "calibration", "sharpness", "exceptions", "climatology", "verdict")


def _judge_everything(modules, inp):
    """The judge's whole public call graph on one input, returned as plain data."""
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


def test_importing_the_whole_judge_in_a_fresh_interpreter_under_the_guard_makes_no_ambient_call():
    """A module-level read (at import) would never be seen by a guard that starts after the import; so import under it, in a new process."""
    script = (
        "import sys; sys.path[:0] = ['src', '.']\n"
        "from tests.gates._purity_guard import purity_guard, AmbientAccessError\n"
        "import numpy\n"
        "with purity_guard() as hits:\n"
        "    import wmj.judge.types, wmj.judge.errors, wmj.judge.limitations, wmj.judge.regions, wmj.judge.distance, wmj.judge._normal\n"
        "    import wmj.judge.skill, wmj.judge.calibration, wmj.judge.sharpness, wmj.judge.exceptions, wmj.judge.climatology, wmj.judge.horizon, wmj.judge.verdict\n"
        "assert not hits\n"
        "print('imported-clean')\n"
    )
    result = subprocess.run([sys.executable, "-c", script], cwd=REPO_ROOT, capture_output=True, text=True, check=False)
    assert result.returncode == 0 and "imported-clean" in result.stdout, result.stderr[-600:]


def _modules(package: str = "wmj.judge"):
    return {name: importlib.import_module(f"{package}.{name}") for name in BLOCKS}


def _kwargs_variants():
    """Three inputs: the builder's; a perfect model (non-zero trust, agreement holds); a world that never drifts (no switch step)."""
    ordinary = judge_input_kwargs()
    perfect = judge_input_kwargs()
    n, h, d = perfect["predictions"].mean.shape
    bin_mean = perfect["climatology"][0].bins[0].mean
    target = np.broadcast_to(bin_mean, (n, h, d)).copy()
    perfect.update(predictions=type(perfect["predictions"])(target, np.ones((n, h, d))), outcomes=target.copy(),
                   invariant_bins=np.zeros((n, h), dtype=np.int64))
    quiet = judge_input_kwargs()
    quiet["divergence_curves"] = tuple(type(c)(c.region_name, np.zeros(c.curve.size)) for c in quiet["divergence_curves"])
    return [ordinary, perfect, quiet]


def fires(call) -> bool:
    """True iff `call()` makes a rigged call (even if the error is swallowed). The guard is left before anything is asserted."""
    with purity_guard(expect_hits=True) as hits:
        try:
            call()
        except AmbientAccessError:
            pass
        return bool(hits)


# ---- TC-JU12-01: the load-bearing tripwire ------------------------------------------------------------------


def test_the_whole_judge_builds_its_input_refuses_a_bad_one_and_computes_every_block_inside_the_guard():
    from wmj.judge.types import JudgeInput

    modules = _modules()
    variants = _kwargs_variants()  # the arrays are drawn outside (the builder uses a seeded generator); the input *checks* run inside
    unguarded = [_judge_everything(modules, JudgeInput(**dict(kw))) for kw in variants]
    with purity_guard() as hits:
        guarded = [_judge_everything(modules, JudgeInput(**dict(kw))) for kw in variants]  # the input checks run rigged too
        bad = dict(variants[0], world="")
        with pytest.raises(JudgeInputError):
            JudgeInput(**bad)  # a refusal path
    assert guarded == unguarded and not hits
    ordinary, perfect, quiet = unguarded
    assert all(e["steps"] == 0 for e in ordinary["trust_horizons"]["per_task"])
    assert any(e["steps"] > 0 for e in perfect["trust_horizons"]["per_task"]) and any(e["agrees"] is True for e in perfect["climatology"]["per_task"])
    assert all(e["switch_step"] is None for e in quiet["climatology"]["per_task"])


def test_a_harmless_call_does_not_trip_the_guard_so_that_a_trip_means_something():
    assert fires(lambda: np.sqrt(np.arange(5.0)).sum() + sum(range(3))) is False
    assert fires(lambda: os.getcwd()) is True


# ---- the guard proves itself: table-driven, by identity, and with a full before/after snapshot --------------


def test_every_name_in_every_table_is_rigged_while_the_guard_is_on_and_none_is_called_for_real():
    with purity_guard():
        for name in guard.OS_FUNCTIONS:
            if hasattr(os, name):
                assert installed(getattr(os, name)), f"os.{name}"
        for name in guard.ENVIRON_METHODS:
            assert installed(type(os.environ).__dict__[name]), f"os.environ.{name}"
        for name in guard.TIME_FUNCTIONS:
            if hasattr(time, name):
                assert installed(getattr(time, name)), f"time.{name}"
        for name in guard.RANDOM_MODULE_FUNCTIONS:
            if hasattr(stdlib_random, name):
                assert installed(getattr(stdlib_random, name)), f"random.{name}"
        for name in guard.SOCKET_FUNCTIONS:
            if hasattr(socket, name):
                assert installed(getattr(socket, name)), f"socket.{name}"
        for name in guard.SUBPROCESS_FUNCTIONS:
            assert installed(getattr(subprocess, name)), f"subprocess.{name}"
        for name in ("uuid1", "getnode"):
            assert installed(getattr(uuid, name))
        assert installed(socket.socket.__init__) and installed(subprocess.Popen.__init__)
        assert installed(open) and installed(io.open)
        assert issubclass(datetime.datetime, datetime.datetime) and datetime.datetime is not datetime.datetime.__mro__[1]


FULL_OS = {
    "system", "getcwd", "getcwdb", "getenv", "getenvb", "urandom", "getpid", "getppid", "listdir", "scandir", "stat", "lstat", "uname",
    "getlogin", "times", "fork", "forkpty", "kill", "popen", "getrandom", "open", "cpu_count", "getuid", "geteuid", "getgid", "getegid",
    "sched_getaffinity", "sysconf", "getloadavg", "putenv", "unsetenv", "mkdir", "makedirs", "remove", "unlink", "rmdir", "rename",
    "chdir", "readlink", "walk", "execl", "execle", "execlp", "execlpe", "execv", "execve", "execvp", "execvpe", "posix_spawn",
    "posix_spawnp", "spawnl", "spawnle", "spawnlp", "spawnlpe", "spawnv", "spawnve", "spawnvp", "spawnvpe",
}
FULL_ENVIRON = {"__getitem__", "__contains__", "__iter__", "__len__", "get", "keys", "values", "items", "copy"}
FULL_TIME = {
    "time", "time_ns", "monotonic", "monotonic_ns", "perf_counter", "perf_counter_ns", "process_time", "process_time_ns", "thread_time",
    "thread_time_ns", "sleep", "localtime", "gmtime", "ctime", "asctime", "strftime", "clock_gettime", "clock_gettime_ns", "clock_getres",
    "get_clock_info",
}
FULL_SOCKET = {
    "create_connection", "getaddrinfo", "gethostbyname", "gethostbyname_ex", "gethostbyaddr", "gethostname", "getnameinfo", "getservbyname",
    "getservbyport", "socketpair", "create_server", "fromfd",
}
FULL_SUBPROCESS = {"run", "call", "check_call", "check_output", "getoutput", "getstatusoutput"}


def test_the_tables_are_pinned_in_full_so_an_entry_cannot_quietly_disappear():
    assert set(guard.OS_FUNCTIONS) >= FULL_OS and set(guard.ENVIRON_METHODS) >= FULL_ENVIRON
    assert set(guard.TIME_FUNCTIONS) >= FULL_TIME and set(guard.SOCKET_FUNCTIONS) >= FULL_SOCKET
    assert set(guard.SUBPROCESS_FUNCTIONS) >= FULL_SUBPROCESS
    assert {"random", "randint", "randrange", "choice", "shuffle", "sample", "uniform", "gauss", "seed", "getrandbits", "_urandom"} <= set(guard.RANDOM_MODULE_FUNCTIONS)


def test_the_installed_check_has_a_negative_control_nothing_is_rigged_outside_the_guard():
    assert not installed(os.getcwd) and not installed(time.time) and not installed(open) and not installed(np.random.normal)
    assert not installed(sys.modules["numpy.random.mtrand"]._rand) and not installed(type(os.environ).__dict__["__getitem__"])


def test_the_tables_still_contain_every_route_the_spec_names_so_they_cannot_quietly_shrink():
    assert {"system", "getcwd", "getenv", "urandom", "getpid", "listdir", "stat", "fork", "execve", "execv", "posix_spawn", "spawnv"} <= set(guard.OS_FUNCTIONS)
    assert {"__getitem__", "__contains__", "__iter__", "__len__", "get", "keys", "values", "items"} <= set(guard.ENVIRON_METHODS)
    assert {"time", "time_ns", "monotonic", "perf_counter", "sleep", "localtime", "gmtime"} <= set(guard.TIME_FUNCTIONS)
    assert {"create_connection", "getaddrinfo", "gethostbyname"} <= set(guard.SOCKET_FUNCTIONS)
    assert {"run", "call", "check_call", "check_output"} <= set(guard.SUBPROCESS_FUNCTIONS)
    assert {"random", "randint", "shuffle", "seed", "_urandom"} <= set(guard.RANDOM_MODULE_FUNCTIONS)


def test_every_place_numpy_keeps_a_reference_to_a_random_class_or_to_datetime64_is_rigged():
    random_sites = {
        (module.__name__, attribute)
        for name in guard._random_names()
        for module, attribute in guard.bindings_of(getattr(np.random, name), "numpy.random")
    }
    datetime64_sites = {(module.__name__, attribute) for module, attribute in guard.bindings_of(np.datetime64, "numpy")}
    # the places the review found by hand, so the identity search cannot quietly stop finding them
    assert {("numpy.random._pickle", "Generator"), ("numpy.random._pickle", "PCG64"), ("numpy.random._pcg64", "PCG64"),
            ("numpy.random._generator", "Generator"), ("numpy.random.mtrand", "_MT19937"), ("numpy.random.bit_generator", "SeedSequence"),
            ("numpy.random._philox", "Philox"), ("numpy.random._sfc64", "SFC64"), ("numpy.random._mt19937", "MT19937")} <= random_sites
    assert {("numpy", "datetime64"), ("numpy.core.numerictypes", "datetime64"), ("numpy.core.numeric", "datetime64"),
            ("numpy.core", "datetime64"), ("numpy._typing._array_like", "datetime64")} <= datetime64_sites
    with purity_guard():
        for module_name, attribute in random_sites:
            assert installed(getattr(sys.modules[module_name], attribute)), (module_name, attribute)
        for module_name, attribute in datetime64_sites:
            assert getattr(sys.modules[module_name], attribute) is guard._GuardedDatetime64, (module_name, attribute)
        assert installed(sys.modules["numpy.random.mtrand"]._rand)


@pytest.mark.parametrize(
    "label, call",
    [
        ("os.getcwd", lambda: os.getcwd()),
        ("os.getenv", lambda: os.getenv("HOME")),
        ("os.urandom", lambda: os.urandom(4)),
        ("os.getrandom", lambda: os.getrandom(4)),
        ("os.listdir", lambda: os.listdir(".")),
        ("os.stat", lambda: os.stat(".")),
        ("os.open", lambda: os.open(__file__, os.O_RDONLY)),
        ("os.execv (impossible target: a broken guard raises OSError, never replaces the process)", lambda: os.execv("/nonexistent/x", ["x"])),
        ("os.posix_spawn (impossible target)", lambda: os.posix_spawn("/nonexistent/x", ["x"], {})),
        ("os.spawnv (impossible target)", lambda: os.spawnv(os.P_WAIT, "/nonexistent/x", ["x"])),
        ("os.environ[x]", lambda: os.environ["HOME"]),
        ("x in os.environ", lambda: "HOME" in os.environ),
        ("dict(os.environ)", lambda: dict(os.environ)),
        ("list(os.environ)", lambda: list(os.environ)),
        ("os.environ.get", lambda: os.environ.get("HOME")),
        ("len(os.environ)", lambda: len(os.environ)),
        ("open", lambda: open(__file__)),
        ("io.open", lambda: open(__file__)),
        ("time.time", lambda: time.time()),
        ("time.gmtime()", lambda: time.gmtime()),
        ("time.clock_gettime", lambda: time.clock_gettime(time.CLOCK_REALTIME)),
        ("time.perf_counter", lambda: time.perf_counter()),
        ("datetime.now", lambda: datetime.datetime.now()),
        ("datetime.utcnow", lambda: datetime.datetime.utcnow()),
        ("datetime.today", lambda: datetime.datetime.today()),
        ("date.today", lambda: datetime.date.today()),
        ("stdlib random", lambda: stdlib_random.random()),
        ("socket.socket", lambda: socket.socket()),
        ("socket.gethostbyaddr", lambda: socket.gethostbyaddr("127.0.0.1")),
        ("socket.getnameinfo", lambda: socket.getnameinfo(("127.0.0.1", 80), 0)),
        ("subprocess.run (impossible target)", lambda: subprocess.run(["/nonexistent/x"], check=False)),
        ("subprocess.Popen (impossible target, explicit env)", lambda: subprocess.Popen(["/nonexistent/x"], env={})),
        ("np.random.normal", lambda: np.random.normal()),
        ("np.random.default_rng", lambda: np.random.default_rng()),
        ("np.random.Generator(PCG64)", lambda: np.random.Generator(np.random.PCG64(1))),
        ("legacy singleton mtrand._rand", lambda: sys.modules["numpy.random.mtrand"]._rand.random_sample()),
        ("legacy MT19937 via mtrand", lambda: sys.modules["numpy.random.mtrand"]._MT19937()),
        ("the pickle helper's entropy Generator", lambda: sys.modules["numpy.random._pickle"].__generator_ctor()),
        ("PCG64 via the private module", lambda: sys.modules["numpy.random._pcg64"].PCG64()),
        ("SeedSequence via bit_generator", lambda: sys.modules["numpy.random.bit_generator"].SeedSequence()),
        ("np.datetime64('now')", lambda: np.datetime64("now")),
        ("np.datetime64(' NOW ')", lambda: np.datetime64(" NOW ")),
        ("np.datetime64('today')", lambda: np.datetime64("today")),
        ("np.datetime64(b'now')", lambda: np.datetime64(b"now")),
        ("np.datetime64 through numpy.core.numerictypes", lambda: np.core.numerictypes.datetime64("now")),
    ],
)
def test_a_known_outside_world_route_trips_the_guard(label, call):
    assert fires(call), label


def _snapshot():
    """Every attribute of every object the guard touches, by identity — and numpy's modules, where it rebinds."""
    objects = [os, type(os.environ), __import__("builtins"), io, time, socket, socket.socket, subprocess, subprocess.Popen,
               datetime, stdlib_random, uuid, np, np.random]
    objects += [m for name, m in list(sys.modules.items()) if m is not None and (name == "numpy" or name.startswith("numpy."))]
    seen, snap = set(), []
    for obj in objects:
        if id(obj) not in seen:
            seen.add(id(obj))
            snap.append((obj, dict(vars(obj))))
    return snap


def _same(snap_a, snap_b) -> list[str]:
    differences = []
    for (obj, before), (_, after) in zip(snap_a, snap_b, strict=True):
        for key in before.keys() | after.keys():
            if before.get(key, _same) is not after.get(key, _same):
                differences.append(f"{getattr(obj, '__name__', obj)}.{key}")
    return differences


def test_everything_is_put_back_exactly_after_a_normal_exit_an_error_an_interrupt_and_nested_use():
    before = _snapshot()
    with purity_guard():
        assert _same(before, _snapshot())  # the snapshot really sees the rigging
    assert _same(before, _snapshot()) == []
    for exception in (RuntimeError("boom"), KeyboardInterrupt(), AmbientAccessError("swallowed")):
        with pytest.raises(type(exception)), purity_guard(expect_hits=True):
            raise exception
        assert _same(before, _snapshot()) == []
    with purity_guard():
        with purity_guard():
            pass
        assert installed(os.getcwd)  # the inner exit did not unrig the outer guard
    assert _same(before, _snapshot()) == []
    assert os.getcwd() and time.time() > 0 and np.random.default_rng(1).random() >= 0.0 and np.datetime64("now")
    assert datetime.datetime.now() and stdlib_random.random() >= 0.0


def test_the_guard_does_not_leak_state_between_uses():
    for _ in range(3):
        assert fires(lambda: os.getcwd())
        assert os.getcwd()


# ---- TC-JU12-02 / -03: phantom-gate proofs on a scratch copy of the judge -----------------------------------


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


def _mutate(target: Path, filename: str, old: str, new: str) -> str:
    """Apply the mutation to the scratch copy; returns the mutated source as it would read in the real package (for the static lint)."""
    path = target / filename
    text = path.read_text()
    assert old in text, f"the mutation anchor {old!r} is not in {filename}"
    mutated = text.replace(old, new, 1)
    path.write_text(mutated)
    return mutated.replace(target.name, "wmj.judge")


def _scratch_input(name: str, real_kwargs: dict):
    """Rebuild a real input's kwargs out of the scratch package's own types (class identities differ between copies)."""
    t = importlib.import_module(f"{name}.types")

    def forecasts(f):
        return t.Forecasts(f.mean, f.spread)

    out = dict(real_kwargs)
    for key in ("predictions", "persistence", "linear"):
        out[key] = forecasts(real_kwargs[key])
    out["region_labels"] = tuple(t.RegionLabel(x.region_name, x.axis) for x in real_kwargs["region_labels"])
    out["divergence_curves"] = tuple(t.RegionCurve(c.region_name, c.curve) for c in real_kwargs["divergence_curves"])
    out["climatology"] = tuple(
        t.RegionClimatology(c.region_name, tuple(t.ClimatologyBin(b.invariant_lo, b.invariant_hi, b.mean, b.sd, b.n_samples) for b in c.bins))
        for c in real_kwargs["climatology"]
    )
    out["tasks"] = tuple(t.TaskSpec(x.name, x.kind, x.tolerance, x.horizon) for x in real_kwargs["tasks"])
    th = real_kwargs["thresholds"]
    out["thresholds"] = t.Thresholds(t.Bands(th.bands.n, th.bands.p, th.bands.green, th.bands.amber_outer), th.sharpness_hedge_threshold, th.agreement_threshold)
    return t.JudgeInput(**out)


def _build_and_run(name: str):
    """A function that builds the scratch package's input from pre-drawn arrays (so the input checks run in whatever context calls it) and runs every block."""
    real_kwargs = judge_input_kwargs()  # drawn here, outside any guard

    def run():
        modules = _modules(name)  # imported inside whatever context calls this, so a module-level read is seen too
        return _judge_everything(modules, _scratch_input(name, real_kwargs))

    return run


def _lint_clean(real_source: str) -> bool:
    return run_judge_gate(ast.parse(real_source)) == []


def test_the_unmodified_scratch_copy_passes_the_guard_which_proves_the_mutations_below_are_what_trips_it(scratch_judge):
    name, _ = scratch_judge
    run = _build_and_run(name)
    with purity_guard():
        assert run()["skill"]


def test_tc_ju12_02_a_read_reached_through_numpys_own_os_reference_is_invisible_to_the_static_lint_but_caught_here(scratch_judge):
    name, target = scratch_judge
    source = _mutate(target, "skill.py", "def compute_skill(inp: JudgeInput) -> dict:", "def compute_skill(inp: JudgeInput) -> dict:\n    np.lib.utils.os.getcwd()")
    assert _lint_clean(source)  # the static gate sees nothing wrong: this is exactly the case the runtime guard exists for
    assert fires(_build_and_run(name))


def test_tc_ju12_03_numpys_own_clock_is_invisible_to_the_static_lint_but_caught_here(scratch_judge):
    name, target = scratch_judge
    source = _mutate(target, "horizon.py", "def compute_error_vs_horizon(inp: JudgeInput) -> dict:", "def compute_error_vs_horizon(inp: JudgeInput) -> dict:\n    np.datetime64('now')")
    assert _lint_clean(source)
    assert fires(_build_and_run(name))


def test_a_module_level_ambient_read_at_import_time_is_caught_because_the_scratch_copy_is_imported_under_the_guard(scratch_judge):
    name, target = scratch_judge
    source = _mutate(target, "skill.py", "from __future__ import annotations", "from __future__ import annotations\n\nimport numpy as np\n\n_X = np.lib.utils.os.getcwd()")
    assert _lint_clean(source)
    assert fires(_build_and_run(name))


def test_a_module_level_clock_read_at_import_time_is_caught(scratch_judge):
    name, target = scratch_judge
    _mutate(target, "horizon.py", "from __future__ import annotations", "from __future__ import annotations\n\nimport numpy as np\n\n_X = np.datetime64('now')")
    assert fires(_build_and_run(name))


def test_the_older_routes_are_caught_by_both_layers_independently_the_lint_and_the_guard(tmp_path):
    cases = (
        ("sharpness.py", "def compute_sharpness(inp: JudgeInput) -> dict:", "import os as _o; _o.environ['HOME']"),
        ("calibration.py", "def compute_calibration(inp: JudgeInput) -> dict:", "np.random.default_rng().random()"),
        ("exceptions.py", "def compute_exceptions_and_trials(inp: JudgeInput) -> tuple[dict, dict]:", "__import__('time').time()"),
        ("skill.py", "def compute_skill(inp: JudgeInput) -> dict:", "np.get_include.__globals__['os'].getcwd()"),
    )
    for index, (filename, anchor, injected) in enumerate(cases):
        name = f"judge_scratch_older_{index}"
        target = tmp_path / name
        shutil.copytree(JUDGE_SRC, target, ignore=shutil.ignore_patterns("__pycache__"))
        for path in target.glob("*.py"):
            path.write_text(path.read_text().replace("wmj.judge", name))
        sys.path.insert(0, str(tmp_path))
        importlib.invalidate_caches()
        try:
            source = _mutate(target, filename, anchor, anchor + "\n    " + injected)
            assert not _lint_clean(source), filename  # the static gate flags these, as a second, independent layer
            assert fires(_build_and_run(name)), filename
        finally:
            sys.path.remove(str(tmp_path))
            for module in [m for m in sys.modules if m == name or m.startswith(name + ".")]:
                del sys.modules[module]


def test_an_ambient_call_whose_error_the_judge_swallows_still_fails_the_guard(scratch_judge):
    name, target = scratch_judge
    _mutate(target, "skill.py", "def compute_skill(inp: JudgeInput) -> dict:",
            "def compute_skill(inp: JudgeInput) -> dict:\n    try:\n        np.lib.utils.os.getcwd()\n    except BaseException:\n        pass")
    run = _build_and_run(name)
    with pytest.raises(AmbientAccessError, match="swallowed"), purity_guard():
        run()


def test_an_ambient_read_inside_an_input_check_is_caught_because_the_input_is_built_under_the_guard(scratch_judge):
    name, target = scratch_judge
    _mutate(target, "types.py", "def _exact_str(name: str, value) -> str:", "def _exact_str(name: str, value) -> str:\n    np.lib.utils.os.getcwd()")
    assert fires(_build_and_run(name))


# ---- TC-JU12-04: the residual is stated, in the specs, neither wider nor narrower ---------------------------


def _spec_body(name: str) -> str:
    """The spec without its change notes (lines starting with '>'), which narrate history rather than state the current rule."""
    return "\n".join(line for line in (REPO_ROOT / "specs" / name).read_text().splitlines() if not line.startswith(">"))


def test_tc_ju12_04_the_residual_paragraph_names_exactly_the_two_remaining_routes_and_the_backstop():
    cross_cutting = _spec_body("cross-cutting.md")
    start = cross_cutting.index("What this control does NOT catch")
    residual = cross_cutting[start: cross_cutting.index("JU-12 is delivered by the harness", start)]
    assert "Two routes remain" in residual
    assert "(a)" in residual and "ctypes" in residual
    assert "(b)" in residual and "before the harness installs" in residual
    assert "(c)" not in residual  # no third, unlisted route
    assert "TC-NF1-01" in residual and "byte-identity" in residual and "separate processes" in residual
    # the closed route is described as caught — never as still open
    closed = [s for s in re.split(r"(?<=[.;])\s", residual) if "getrandom" in s]
    assert closed and all(re.search(r"closes|now raises|actually \*\*caught\*\*|actually caught", s) for s in closed), closed
    assert not any(re.search(r"remains (open|uncaught)|still (open|uncaught)", s) for s in closed)


def test_tc_ju12_04_the_judge_spec_states_the_same_residual_and_the_backstop():
    judge_spec = _spec_body("judge.md")
    start = judge_spec.index("What TC-JU12-01 does not catch")
    residual = judge_spec[start: judge_spec.index("\n\n", start)]  # this paragraph only
    assert "ctypes" in residual and "bound to a local name before the harness installs" in residual
    assert "TC-NF1-01" in residual and "separate processes" in residual
    assert "primary tripwire" in judge_spec[judge_spec.index("Purity enforcement"): start]
