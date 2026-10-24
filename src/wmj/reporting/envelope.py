"""wmj.reporting.envelope — reading the harness's envelope and naming the files a chart writes.

In plain words: every chart starts from the same envelope the harness hands over — which model, whether
it is a deliberately broken test fixture, which world, and the verdict (judge §5) — and ends by writing
a PNG, an SVG and a caption under names built from those same words. Both steps live here once, so
every chart refuses the same bad input the same way and names its files by the same rule
(`{world}-{model}-{chart}…`, reporting ADR-R5). Names are 1 to 40 letters, digits, '.', '_' or '-':
safe in a file name and short enough to fit a chart's header strip.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from wmj.reporting import style

_FILE_SAFE = re.compile(r"[A-Za-z0-9._-]{1,40}")


@dataclass(frozen=True)
class ChartPaths:
    """Where one chart was written: its PNG, its SVG duplicate and its caption text."""

    png: Path
    svg: Path
    caption: Path


@dataclass(frozen=True)
class Envelope:
    """The parts of a harness envelope every chart reads."""

    world: str
    model_name: str
    is_fixture: bool
    verdict: dict


def file_part(label: str, value: object) -> str:
    """`value` if it can be part of a file name and a header strip, else ValueError."""
    if type(value) is not str or not _FILE_SAFE.fullmatch(value):
        raise ValueError(f"{label} {value!r} cannot be part of a file name (1 to 40 letters, digits, '.', '_', '-')")
    return value


def read_envelope(envelope: object) -> Envelope:
    """The world, model name, fixture flag and verdict of an envelope, checked; ValueError if any is unusable."""
    if not isinstance(envelope, dict) or not isinstance(envelope.get("verdict"), dict):
        raise ValueError(f"an envelope needs a 'verdict' (judge §5), got {envelope!r:.80}")  # noqa: TRY004 - one error type for every bad envelope
    model, is_fixture = envelope.get("model_name"), envelope.get("is_fixture")
    try:
        style.display_name(model, is_fixture)  # a real name and a real flag
    except style.FixtureLabelError as error:
        raise ValueError(str(error)) from error
    verdict = envelope["verdict"]
    return Envelope(file_part("world", verdict.get("world")), file_part("model name", model), is_fixture, verdict)


def chart_paths(charts_dir: Path, captions_dir: Path, stem: str) -> ChartPaths:
    return ChartPaths(Path(charts_dir) / f"{stem}.png", Path(charts_dir) / f"{stem}.svg", Path(captions_dir) / f"{stem}.txt")
