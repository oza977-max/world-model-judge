"""wmj's `python -m wmj` entry point.

The thread guard must run before anything that imports NumPy — which
includes every other wmj module — so it is the first executable
statement in this file, before the `wmj.harness.cli` import
(cross-cutting ADR-002 rule 1).
"""

from __future__ import annotations

from wmj.harness.thread_guard import ensure_single_threaded

ensure_single_threaded()

from wmj.harness.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
