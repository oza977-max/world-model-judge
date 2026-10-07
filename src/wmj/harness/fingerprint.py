"""wmj.harness.fingerprint — what kind of machine and library build produced these numbers.

In plain words: trained models give identical bytes every time *on the same kind of
machine*, but not necessarily across different processors: NumPy and its maths library pick
different low-level routines depending on the CPU (measured: the same training run gives
different weights under different routines, and `exp` differs between machines with and
without a wide-vector unit). So the project does not promise identical bytes across
different CPUs (NF-1 is narrowed to "the same CPU family and library build",
backlog A29). Instead every run records a fingerprint — Python and NumPy versions, the
maths library NumPy was built against, the platform, and which CPU features NumPy
detected — so a sceptic re-running on another machine can see *why* a number differs in
its last digits, and can tell a real disagreement from a hardware one.

The fingerprint goes into the harness envelope's `meta` (judge spec §5), never into the
judge (the judge may not read the environment, JU-12). Wiring it into `wmj run` is
P6-C01's job.
"""

from __future__ import annotations

import os
import platform
import sys
from typing import Any

import numpy as np


def _cpu_features() -> list[str]:
    """The CPU features NumPy detected and enabled, sorted; empty if this NumPy does not say."""
    try:
        detected = np.core._multiarray_umath.__cpu_features__  # a private NumPy table; absent => unknown
    except AttributeError:
        return []
    return sorted(name for name, enabled in detected.items() if enabled)


def _blas_description() -> str:
    """The maths library NumPy was built against, as "name version" ("unknown" if not reported)."""
    try:
        info = np.show_config(mode="dicts")["Build Dependencies"]["blas"]
        return f"{info.get('name', 'unknown')} {info.get('version', '')}".strip()
    except Exception:  # noqa: BLE001 - reporting must never be the reason a run fails
        return "unknown"


def runtime_fingerprint() -> dict[str, Any]:
    """A plain, JSON-able record of the machine and library build (see the module docstring)."""
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "blas": _blas_description(),
        "platform": f"{sys.platform}-{platform.machine()}",
        "cpu_features": _cpu_features(),
        "openblas_coretype_override": os.environ.get("OPENBLAS_CORETYPE", ""),
    }
