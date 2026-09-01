"""wmj's `python -m wmj` entry point.

In plain words: this is the very first thing that runs when a user
types `python -m wmj`. Its only job before handing off to the real CLI
is to tell NumPy's math library to stay single-threaded — and that
has to happen before NumPy is imported at all, which is why the
thread guard call sits above every other import in this file
(cross-cutting ADR-002 rule 1).
"""

from __future__ import annotations

from wmj.harness.thread_guard import ensure_single_threaded

ensure_single_threaded()

from wmj.harness.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
