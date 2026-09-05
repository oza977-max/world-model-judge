"""Root conftest — the single-thread guard for the test process itself.

In plain words: cross-cutting ADR-002 rule 1 applies to *every* process
that runs the harness, and pytest is one of them. NumPy's math library
decides its thread count the moment NumPy is imported, so the three
environment variables must be set before that happens anywhere in the
process. pytest loads this file before it collects any test module,
which is early enough. `cli.main()` then *asserts* the guard on entry
(code-review-001 I3), so a test that reaches it by any route — not
only `python -m wmj` — is checked rather than silently running with
whatever thread count was ambient.
"""

from __future__ import annotations

from wmj.harness.thread_guard import ensure_single_threaded

ensure_single_threaded()
