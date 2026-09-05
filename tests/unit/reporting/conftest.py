"""Restore matplotlib's global rcParams around every reporting test.

`style.apply_style()` writes into the process-wide `matplotlib.rcParams`
by design (see that module's docstring for why that is safe with one
style). A test must not leak that mutation into unrelated tests
(code-review-001, Panel E); `rc_context()` snapshots and restores it.
"""

from __future__ import annotations

import matplotlib
import pytest


@pytest.fixture(autouse=True)
def _restore_rcparams():
    with matplotlib.rc_context():
        yield
