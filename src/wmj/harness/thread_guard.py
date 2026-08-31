"""wmj.harness.thread_guard — forces single-threaded BLAS before NumPy loads.

In plain words: NumPy's math library (BLAS) can silently split a
computation across threads, and thread scheduling order isn't fixed —
so the same computation can come out with different rounding on
different runs (Goldberg: floating-point addition is not associative).
Setting these three environment variables to "1" *before* NumPy is
ever imported tells BLAS not to do that (cross-cutting ADR-002 rule 1).
"""

from __future__ import annotations

import os

from wmj.errors import WmjError

THREAD_ENV_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")


class ThreadGuardError(WmjError):
    """Raised when a thread-count env var is missing or not '1'."""


def ensure_single_threaded() -> None:
    """Set the three thread-count env vars to "1".

    Must be called before `import numpy` appears anywhere in the
    process — NumPy reads these at import time to configure BLAS
    threading; setting them afterwards has no effect.
    """
    for name in THREAD_ENV_VARS:
        os.environ[name] = "1"


def assert_single_threaded() -> None:
    """Raise ThreadGuardError if any thread-count env var isn't "1".

    This asserts the *command* was given, not that BLAS obeyed it —
    NumPy exposes no runtime thread-count introspection API, so the
    ten-run byte-identity gate (TC-NF1-01) is the empirical backstop
    for any residual threaded-reduction nondeterminism.
    """
    for name in THREAD_ENV_VARS:
        value = os.environ.get(name)
        if value != "1":
            raise ThreadGuardError(
                f"{name} is {value!r}, expected '1' — call "
                f"ensure_single_threaded() before importing numpy"
            )
