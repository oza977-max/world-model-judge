"""wmj.reporting.captions — the fixed caption templates (reporting ADR-R3).

In plain words: the sentence under each chart is written here, once,
and reviewed as part of the spec. The renderer fills in numbers where a
template has a slot; it never composes prose of its own at run time.
Every caption is at most three sentences (RP-5).

P2-C05 scope: only Chart 2's template exists yet, in two forms. The
full ADR-R3 text (its third sentence narrates the JU-6 switch step,
which needs `climatology.per_task`, first computed at P4-C05) and the
scoped form — the same text with that sentence removed. The scoped form
is *derived* from the full one, so the two can never drift apart.
"""

from __future__ import annotations

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


class CaptionLengthError(WmjError):
    """Raised when a caption exceeds RP-5's three-sentence limit."""


def sentences(text: str) -> list[str]:
    """Split a caption into sentences at `. `, `! `, `? ` boundaries."""
    return [s for s in _SENTENCE_BOUNDARY.split(text.strip()) if s]


def sentence_count(text: str) -> int:
    return len(sentences(text))


def _drop_last_sentence(text: str) -> str:
    return " ".join(sentences(text)[:-1])


# The scoped Chart-2 caption used until P5-C03 delivers the switch lines.
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
