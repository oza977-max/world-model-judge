"""wmj.harness.cli — the `wmj` command-line entry point.

In plain words: this is what a user actually types. Today it has
`run --skeleton`, which proves the pipeline runs end to end (P1-C02's
MVP slice), and `chart-preview`, an internal-only command (reporting
ADR-R5) that draws the first real chart from real trajectories
(P2-C05). Later chunks add the real `wmj run`, `wmj verify` and
`wmj list-models`.

Every command first *verifies* the single-thread guard (cross-cutting
ADR-002 rule 1): `python -m wmj` sets the three BLAS thread variables
before NumPy loads, but anything that reaches `main()` another way —
a test importing it directly, a library caller — would otherwise run
with whatever thread count is ambient and nothing saying so
(code-review-001 I3). The check is on the *command* being given;
byte-identity (TC-NF1-01) remains the empirical backstop.

Concurrent invocations against the same `out/` directory are not
supported: every artefact writer replaces its target atomically, but
two runs writing the same paths would still interleave at the level
of whole files (code-review-001, Panel E).
"""

from __future__ import annotations

import argparse
from typing import Sequence

from wmj.harness import preview
from wmj.harness.skeleton import write_skeleton_report
from wmj.harness.thread_guard import assert_single_threaded


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="wmj")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="run the judge pipeline")
    run_parser.add_argument(
        "--skeleton",
        action="store_true",
        help="run the smallest end-to-end slice (P1-C02 MVP)",
    )

    preview_parser = subparsers.add_parser(
        "chart-preview",
        help="internal-only: render the scoped Chart 2 for persistence on LV (P2-C05)",
    )
    preview_parser.add_argument("--n-starts", type=int, default=preview.DEFAULT_N_STARTS,
                                help="divergence benchmark starts per region")
    preview_parser.add_argument("--n-trials", type=int, default=preview.DEFAULT_N_TRIALS,
                                help="evaluation trials per region")
    preview_parser.add_argument("--horizon", type=int, default=preview.DEFAULT_HORIZON,
                                help="rollout horizon in steps")
    preview_parser.add_argument("--seed", type=int, default=preview.DEFAULT_SEED)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    assert_single_threaded()
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "run" and args.skeleton:
        write_skeleton_report()
        return 0

    if args.command == "chart-preview":
        result = preview.run_chart_preview(
            n_starts=args.n_starts, n_trials=args.n_trials, horizon=args.horizon, seed=args.seed
        )
        print(f"chart-preview: wrote {result.png}, {result.svg}, {result.caption}")
        one_step = result.one_step
        print(
            f"chart-preview: one-step CRPS persistence={one_step.crps_persistence:.5f} "
            f"linear={one_step.crps_linear:.5f}; skill of persistence vs linear = "
            f"{one_step.skill:+.3f}"
        )
        return 0

    parser.error("no runnable command selected")
    return 2  # pragma: no cover - argparse.error() exits before this
