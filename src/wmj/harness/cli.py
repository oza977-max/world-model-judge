"""wmj.harness.cli — the `wmj` command-line entry point.

In plain words: this is what a user actually types. Today it has one
subcommand, `run --skeleton`, which proves the pipeline runs end to
end (P1-C02's MVP slice). Later chunks add the real `wmj run` and
`wmj list-models`.
"""

from __future__ import annotations

import argparse
from typing import Sequence

from wmj.harness.skeleton import write_skeleton_report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="wmj")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="run the judge pipeline")
    run_parser.add_argument(
        "--skeleton",
        action="store_true",
        help="run the smallest end-to-end slice (P1-C02 MVP)",
    )

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "run" and args.skeleton:
        write_skeleton_report()
        return 0

    parser.error("no runnable command selected")
    return 2  # pragma: no cover - argparse.error() exits before this
