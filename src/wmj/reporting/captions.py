"""wmj.reporting.captions — the fixed caption templates (reporting ADR-R3).

In plain words: the sentence under each chart is written here, once,
and reviewed as part of the spec. The renderer fills in numbers where a
template has a slot; it never composes prose of its own at run time.
Every caption is at most three sentences (RP-5).

Chart 2 has two forms: the full ADR-R3 text, and the scoped form P2-C05's
preview uses. The full text's third sentence narrates the JU-6 switch
step, so the scoped form is the same text with that sentence removed. It
is *derived* from the full one, so the two can never drift apart.

Charts 1, 3 and 4 (P5-C01) are filled by functions that check every
number before printing it. A count above the number of runs, an unknown
band, or a confidence level the judge never reports is refused, never
printed under a chart. The Chart 1 text differs from ADR-R3's: that text
described a vertical range per run, but since the owner's ruling of
2026-10-22 (backlog A39 item 9) the chart plots each run's whole-answer
miss, in the model's own stated sizes, against one line. A caption must
describe the chart that is drawn (recorded for the next design review).
"""

from __future__ import annotations

import math
import re
from pathlib import Path

from wmj.errors import WmjError

MAX_SENTENCES = 3  # RP-5

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")

CHART2_FULL = (
    "How wrong the model gets as it predicts further ahead. The dashed black "
    "line is how fast the world drifts away from itself — only the gap above "
    "that line is the model's fault. Past each task's marked step, exact paths "
    "stop being gradable and the judge switches to checking the overall pattern."
)


# Chart 1, amended for the A39 item 9 chart (backlog): dots in the model's own stated sizes, one line.
# Every fill function ends in `_checked`, so a future template edit that grows a fourth sentence is refused.
CHART1_TEMPLATE = (
    "Each dot is one test run: how far the outcome landed from the model's guess, measured in the "
    "model's own stated error sizes and combined over everything it predicts. The line marks the edge "
    "of the range the model called 90% likely; dots above it are the misses, circled in red. A model this "
    "confident should miss about {expected} of {n} times — it missed {observed}, which lands it in the "
    "{band} band drawn on the right."
)
CHART3_TEMPLATE = (  # ADR-R3 verbatim
    "When the model said it was {level}% sure, was it right that often? Of every 100 ranges it drew at "
    "90% confidence, about 90 should contain the true outcome. The solid line is familiar territory; the "
    "dashed line is territory it never saw in training — the honest model's lines both hug the diagonal."
)
CHART4_HEAD = "Ordinary accuracy and the judge's trust horizon, side by side."  # ADR-R3 verbatim
CHART4_DISAGREE = "Ordinary error says {A} and {B} are equals; the judge does not."
CHART4_AGREE = (
    "On this run the two rankings agree — the interesting case did not occur, and saying so plainly "
    "beats manufacturing one."
)

BANDS = ("green", "amber", "red")  # ADR-J4's three bands, as the verdict spells them
LEVELS_PERCENT = (50, 80, 90, 95)  # ADR-J2's four confidence levels


class CaptionLengthError(WmjError):
    """Raised when a caption exceeds RP-5's three-sentence limit."""


class CaptionSlotError(WmjError):
    """Raised when a number or name handed to a caption could not be true (ADR-R3: never print it)."""


def sentences(text: str) -> list[str]:
    """Split a caption into sentences at `. `, `! `, `? ` boundaries."""
    return [s for s in _SENTENCE_BOUNDARY.split(text.strip()) if s]


def sentence_count(text: str) -> int:
    return len(sentences(text))


def _drop_last_sentence(text: str) -> str:
    return " ".join(sentences(text)[:-1])


def _whole(name: str, value: object, low: int, high: int | None = None) -> int:
    if type(value) is not int or value < low or (high is not None and value > high):
        top = "" if high is None else f" and <= {high}"
        raise CaptionSlotError(f"{name} must be a whole number >= {low}{top}, got {value!r}")
    return value


def _checked(text: str) -> str:
    if sentence_count(text) > MAX_SENTENCES:
        raise CaptionLengthError(f"RP-5: caption has {sentence_count(text)} sentences, limit is {MAX_SENTENCES}")
    return text


def chart1_caption(*, expected: float, n: int, observed: int, band: str) -> str:
    """Chart 1's caption. `expected` is the verdict's n x p, shown rounded ("about"): display arithmetic."""
    n = _whole("n (the number of runs)", n, 1)
    observed = _whole("observed (the misses counted)", observed, 0, n)
    if type(expected) not in (int, float) or not math.isfinite(expected) or not 0.0 <= expected <= n:
        raise CaptionSlotError(f"expected must be a finite number of misses between 0 and n = {n}, got {expected!r}")
    if band not in BANDS:
        raise CaptionSlotError(f"band must be one of {BANDS}, got {band!r}")
    shown = round(expected)
    # a whole number reads as one ("about 20"); otherwise one decimal, never a rounded-away "about 0" or a half-to-even 2
    text = str(shown) if abs(expected - shown) < 1e-9 * max(1.0, abs(expected)) else f"{expected:.1f}"
    return _checked(CHART1_TEMPLATE.format(expected=text, n=n, observed=observed, band=band))


def chart3_caption(*, level: int) -> str:
    """Chart 3's caption, at one of the four confidence levels the judge reports (in percent)."""
    if type(level) is not int or level not in LEVELS_PERCENT:
        raise CaptionSlotError(f"level must be one of {LEVELS_PERCENT} (percent), got {level!r}")
    return _checked(CHART3_TEMPLATE.format(level=level))


def chart4_caption(disagreeing: tuple[str, str] | None) -> str:
    """Chart 4's caption: the agreement form for None, else the disagreement form naming the two models."""
    if disagreeing is None:
        return _checked(f"{CHART4_HEAD} {CHART4_AGREE}")
    if type(disagreeing) is not tuple or len(disagreeing) != 2:
        raise CaptionSlotError(f"the disagreeing models must be a pair of two names, got {disagreeing!r}")
    a, b = disagreeing
    for name in (a, b):
        if type(name) is not str or not name.strip() or name != name.strip() or len(sentences(name)) != 1 or name[-1] in ".!?":
            raise CaptionSlotError(f"a model name in the caption must be one plain name, got {name!r}")
    if a == b:
        raise CaptionSlotError(f"the pair must name two different models, got {disagreeing!r}")
    return _checked(f"{CHART4_HEAD} {CHART4_DISAGREE.format(A=a, B=b)}")


# The scoped Chart-2 caption used by P2-C05's preview (the full one comes with P5-C03's switch lines).
CHART2_SCOPED = _drop_last_sentence(CHART2_FULL)


def write_caption(path: Path, text: str) -> Path:
    """Write one rendered caption as UTF-8 text; refuse an over-long one."""
    if sentence_count(text) > MAX_SENTENCES:
        raise CaptionLengthError(
            f"RP-5: caption has {sentence_count(text)} sentences, limit is "
            f"{MAX_SENTENCES}: {text[:60]!r}..."
        )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Write-then-replace so a reader never sees a half-written caption
    # (code-review-001, Panel E). `Path.replace` is the atomic rename;
    # reporting deliberately imports no `os` (TC-NF6-10 — the gate
    # caught the first draft of this very line).
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)
    return path
