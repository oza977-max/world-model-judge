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
changed in place, so those two are *replaced* (a disclosed difference). The ten-run, separate-process
byte-identity gate (TC-NF1-01) is the real backstop for an accidental slip; this guard's value is
saying *which* outside call was made.
"""

from __future__ import annotations

import builtins
import contextlib
import datetime
import io
import os
import socket
import subprocess
import time
from collections.abc import Iterator
from typing import Any

import numpy as np
import numpy.random as np_random


class AmbientAccessError(AssertionError):
    """Raised by a rigged capability: the code under test reached for the outside world."""


def _raiser(label: str):
    def raiser(*_args: Any, **_kwargs: Any) -> Any:
        raise AmbientAccessError(f"ambient access: {label}")

    raiser.__name__ = f"guarded_{label.replace('.', '_')}"
    return raiser


OS_FUNCTIONS = (
    "system", "getcwd", "getcwdb", "getenv", "getenvb", "urandom", "getpid", "getppid", "listdir", "scandir", "stat",
    "lstat", "uname", "getlogin", "times", "fork", "forkpty", "kill", "popen",
    *(n for n in dir(os) if n.startswith(("exec", "spawn", "posix_spawn"))),
)
ENVIRON_METHODS = ("__getitem__", "__contains__", "__iter__", "__len__", "get", "keys", "values", "items", "copy")
TIME_FUNCTIONS = (
    "time", "time_ns", "monotonic", "monotonic_ns", "perf_counter", "perf_counter_ns", "process_time", "process_time_ns",
    "thread_time", "thread_time_ns", "sleep", "localtime", "ctime", "asctime", "strftime",
)
SOCKET_FUNCTIONS = ("create_connection", "getaddrinfo", "gethostbyname", "gethostname", "socketpair", "create_server")
SUBPROCESS_FUNCTIONS = ("run", "call", "check_call", "check_output", "getoutput", "getstatusoutput")
RANDOM_SUBMODULES = (
    "numpy.random._pcg64", "numpy.random._generator", "numpy.random.bit_generator", "numpy.random._mt19937",
    "numpy.random._philox", "numpy.random._sfc64", "numpy.random.mtrand", "numpy.random._common",
)
_NOT_RANDOM_CALLABLES = {"bit_generator", "mtrand", "test"}


def _random_names() -> list[str]:
    return [n for n in dir(np_random) if not n.startswith("_") and n not in _NOT_RANDOM_CALLABLES and callable(getattr(np_random, n))]


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
        if args and isinstance(args[0], str) and args[0].strip().lower() == "now":
            raise AmbientAccessError("ambient access: numpy.datetime64('now')")
        return super().__new__(cls, *args, **kwargs)


def _guarded_datetime_classes() -> tuple[type, type]:
    class GuardedDatetime(datetime.datetime):
        @classmethod
        def now(cls, tz=None):  # type: ignore[no-untyped-def]
            raise AmbientAccessError("ambient access: datetime.datetime.now")

        @classmethod
        def utcnow(cls):  # type: ignore[no-untyped-def]
            raise AmbientAccessError("ambient access: datetime.datetime.utcnow")

        @classmethod
        def today(cls):  # type: ignore[no-untyped-def]
            raise AmbientAccessError("ambient access: datetime.datetime.today")

    class GuardedDate(datetime.date):
        @classmethod
        def today(cls):  # type: ignore[no-untyped-def]
            raise AmbientAccessError("ambient access: datetime.date.today")

    return GuardedDatetime, GuardedDate


@contextlib.contextmanager
def purity_guard() -> Iterator[None]:
    """Rig every known outside-world route to raise `AmbientAccessError`; undo it all on exit."""
    import sys

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
            rig.set(time, name, _raiser(f"time.{name}"))
        guarded_datetime, guarded_date = _guarded_datetime_classes()
        rig.set(datetime, "datetime", guarded_datetime)
        rig.set(datetime, "date", guarded_date)
        for name in SOCKET_FUNCTIONS:
            rig.set(socket, name, _raiser(f"socket.{name}"))
        rig.set(socket.socket, "__init__", _raiser("socket.socket"))
        for name in SUBPROCESS_FUNCTIONS:
            rig.set(subprocess, name, _raiser(f"subprocess.{name}"))
        rig.set(subprocess.Popen, "__init__", _raiser("subprocess.Popen"))
        for name in _random_names():  # at the top level AND in each private submodule that binds the same object
            rig.set(np_random, name, _raiser(f"numpy.random.{name}"))
            for module_name in RANDOM_SUBMODULES:
                module = sys.modules.get(module_name)
                if module is not None and hasattr(module, name):
                    rig.set(module, name, _raiser(f"{module_name}.{name}"))
        rig.set(np, "datetime64", _GuardedDatetime64)
        yield
    finally:
        rig.restore()
