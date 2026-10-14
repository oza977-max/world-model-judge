"""The runtime purity tripwire for the judge (cross-cutting ADR-002 rule 3, judge spec §4, TC-JU12-01).

In plain words: the judge must be a pure calculation — numbers in, numbers out, no clock, no files,
no environment variables, no network, no random numbers. A static scan of the source can be fooled
by clever indirection, so this is the second line: while the judge runs, every *known* way of
touching the outside world is rigged to raise an error the moment it is used, whichever route the
code took to reach it. The rigging is done by changing the real shared objects (`os`, `time`,
`socket`, `numpy.random` …) in place, so a hidden route such as `some_function.__globals__['os']`
ends at the same rigged object. Everything is put back afterwards, even if the code under test
fails.

What it cannot do, stated honestly: it guards a finite list of known routes, not every possible
one. It cannot see `ctypes` (raw system calls with no Python object to rig) or an object captured
before the guard was switched on; and `numpy.datetime64` and `datetime` are C types that cannot be
changed in place, so those three (`numpy.datetime64`, `datetime.datetime`, `datetime.date`) are *replaced* (a disclosed difference). The ten-run, separate-process
byte-identity gate (TC-NF1-01) is the real backstop for an accidental slip; this guard's value is
saying *which* outside call was made.
"""

from __future__ import annotations

import builtins
import contextlib
import datetime
import io
import os
import random as stdlib_random
import socket
import subprocess
import sys
import time
import uuid
from collections.abc import Iterator
from typing import Any, NoReturn

import numpy as np
import numpy.random as np_random


class AmbientAccessError(BaseException):
    """Raised by a rigged capability: the code under test reached for the outside world."""


_hits: list[str] = []  # every rigged call made while a guard is active, even one whose error was swallowed


def _trip(label: str) -> NoReturn:
    _hits.append(label)
    raise AmbientAccessError(f"ambient access: {label}")


def _raiser(label: str):
    def raiser(*_args: Any, **_kwargs: Any) -> Any:
        _trip(label)

    raiser.__name__ = f"guarded_{label.replace('.', '_')}"
    return raiser


OS_FUNCTIONS = (
    "system", "getcwd", "getcwdb", "getenv", "getenvb", "urandom", "getpid", "getppid", "listdir", "scandir", "stat",
    "lstat", "uname", "getlogin", "times", "fork", "forkpty", "kill", "popen", "getrandom", "open", "cpu_count", "getuid",
    "geteuid", "getgid", "getegid", "sched_getaffinity", "sysconf", "getloadavg", "putenv", "unsetenv", "mkdir", "makedirs",
    "remove", "unlink", "rmdir", "rename", "chdir", "readlink", "walk",
    *(n for n in dir(os) if n.startswith(("exec", "spawn", "posix_spawn"))),
)
ENVIRON_METHODS = ("__getitem__", "__contains__", "__iter__", "__len__", "get", "keys", "values", "items", "copy")
TIME_FUNCTIONS = (
    "time", "time_ns", "monotonic", "monotonic_ns", "perf_counter", "perf_counter_ns", "process_time", "process_time_ns",
    "thread_time", "thread_time_ns", "sleep", "localtime", "gmtime", "ctime", "asctime", "strftime", "clock_gettime",
    "clock_gettime_ns", "clock_getres", "get_clock_info",
)
RANDOM_MODULE_FUNCTIONS = (
    "random", "randint", "randrange", "choice", "choices", "shuffle", "sample", "uniform", "gauss", "normalvariate", "seed",
    "getrandbits", "randbytes", "triangular", "betavariate", "expovariate", "gammavariate", "lognormvariate",
    "vonmisesvariate", "paretovariate", "weibullvariate", "_urandom",
)
SOCKET_FUNCTIONS = (
    "create_connection", "getaddrinfo", "gethostbyname", "gethostbyname_ex", "gethostbyaddr", "gethostname", "getnameinfo",
    "getservbyname", "getservbyport", "socketpair", "create_server", "fromfd",
)
SUBPROCESS_FUNCTIONS = ("run", "call", "check_call", "check_output", "getoutput", "getstatusoutput")
_NOT_RANDOM_CALLABLES = {"bit_generator", "mtrand", "test"}


def _random_names() -> list[str]:
    return [n for n in dir(np_random) if not n.startswith("_") and n not in _NOT_RANDOM_CALLABLES and callable(getattr(np_random, n))]


def bindings_of(target: object, module_prefix: str) -> list[tuple[Any, str]]:
    """Every `(module, attribute)` under `module_prefix` that holds exactly `target` — found by identity, not by a hand-kept list.

    numpy re-exports its random classes and its `datetime64` from several private modules, each keeping its own
    reference to the one object; rigging only the best-known name would leave the others open."""
    found = []
    for module_name, module in list(sys.modules.items()):
        if module is None or not (module_name == module_prefix or module_name.startswith(module_prefix + ".")):
            continue
        for attribute, value in list(vars(module).items()):
            if value is target:
                found.append((module, attribute))
    return found


class _RaisingProxy:
    """Stands in for numpy's hidden legacy generator object: any use of it is an ambient (random) access."""

    def __getattr__(self, name: str) -> Any:
        _trip(f"numpy.random.mtrand._rand.{name}")


class _Rig:
    """Records every change so `restore` can undo all of them."""

    def __init__(self) -> None:
        self._saved: list[tuple[Any, str, bool, Any]] = []

    def set(self, obj: Any, name: str, value: Any) -> None:
        own = getattr(obj, "__dict__", {})
        existed = name in own
        self._saved.append((obj, name, existed, own[name] if existed else None))  # the raw entry, so descriptors restore exactly
        setattr(obj, name, value)

    def restore(self) -> None:
        for obj, name, existed, original in reversed(self._saved):
            if existed:
                setattr(obj, name, original)
            else:
                with contextlib.suppress(AttributeError):
                    delattr(obj, name)
        self._saved.clear()


class _GuardedDatetime64(np.datetime64):
    """`numpy.datetime64` is an immutable C type: a subclass that refuses 'now' replaces it (a disclosed rebind)."""

    def __new__(cls, *args: Any, **kwargs: Any):  # type: ignore[no-untyped-def]
        if args and isinstance(args[0], str | bytes) and (args[0].decode() if isinstance(args[0], bytes) else args[0]).strip().lower() in {"now", "today"}:
            _trip("numpy.datetime64('now' / 'today')")
        return super().__new__(cls, *args, **kwargs)


def _guarded_datetime_classes() -> tuple[type, type]:
    class GuardedDatetime(datetime.datetime):
        @classmethod
        def now(cls, tz=None):  # type: ignore[no-untyped-def]
            _trip("datetime.datetime.now")

        @classmethod
        def utcnow(cls):  # type: ignore[no-untyped-def]
            _trip("datetime.datetime.utcnow")

        @classmethod
        def today(cls):  # type: ignore[no-untyped-def]
            _trip("datetime.datetime.today")

    class GuardedDate(datetime.date):
        @classmethod
        def today(cls):  # type: ignore[no-untyped-def]
            _trip("datetime.date.today")

    return GuardedDatetime, GuardedDate


def installed(obj: Any) -> bool:
    """True iff `obj` is one of this guard's raising stand-ins (so a test can check a route without calling it)."""
    return isinstance(obj, _RaisingProxy) or getattr(obj, "__name__", "").startswith("guarded_")


@contextlib.contextmanager
def purity_guard(expect_hits: bool = False) -> Iterator[list[str]]:
    """Rig every known outside-world route to raise `AmbientAccessError`; undo it all on exit.

    Yields the list of rigged calls made. Unless `expect_hits`, leaving the guard with a non-empty list raises
    `AmbientAccessError` even if the code under test swallowed the error itself."""
    global _hits
    outer, _hits = _hits, []
    hits = _hits
    rig = _Rig()
    try:
        for name in OS_FUNCTIONS:
            if hasattr(os, name):
                rig.set(os, name, _raiser(f"os.{name}"))
        for name in ENVIRON_METHODS:  # the real class: `obj[x]`, `x in obj`, `dict(obj)` resolve against the type
            rig.set(type(os.environ), name, _raiser(f"os.environ.{name}"))
        rig.set(builtins, "open", _raiser("builtins.open"))
        rig.set(io, "open", _raiser("io.open"))
        for name in TIME_FUNCTIONS:
            if hasattr(time, name):
                rig.set(time, name, _raiser(f"time.{name}"))
        for name in RANDOM_MODULE_FUNCTIONS:
            if hasattr(stdlib_random, name):
                rig.set(stdlib_random, name, _raiser(f"random.{name}"))
        for name in ("uuid1", "getnode"):
            rig.set(uuid, name, _raiser(f"uuid.{name}"))
        guarded_datetime, guarded_date = _guarded_datetime_classes()
        rig.set(datetime, "datetime", guarded_datetime)
        rig.set(datetime, "date", guarded_date)
        for name in SOCKET_FUNCTIONS:
            if hasattr(socket, name):
                rig.set(socket, name, _raiser(f"socket.{name}"))
        rig.set(socket.socket, "__init__", _raiser("socket.socket"))
        for name in SUBPROCESS_FUNCTIONS:
            rig.set(subprocess, name, _raiser(f"subprocess.{name}"))
        rig.set(subprocess.Popen, "__init__", _raiser("subprocess.Popen"))
        for name in _random_names():  # at every module that holds the same object, found by identity
            for module, attribute in bindings_of(getattr(np_random, name), "numpy.random"):
                rig.set(module, attribute, _raiser(f"{module.__name__}.{attribute}"))
        mtrand = sys.modules.get("numpy.random.mtrand")
        if mtrand is not None and hasattr(mtrand, "_rand"):
            rig.set(mtrand, "_rand", _RaisingProxy())
        for module, attribute in bindings_of(np.datetime64, "numpy"):  # a rebind: the C type cannot be changed in place
            rig.set(module, attribute, _GuardedDatetime64)
        yield hits
        if hits and not expect_hits:
            raise AmbientAccessError(f"ambient access was attempted and swallowed by the code under test: {sorted(set(hits))}")
    finally:
        rig.restore()
        _hits = outer
