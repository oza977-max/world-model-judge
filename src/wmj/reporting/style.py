"""wmj.reporting.style — one shared chart style and fixed colour semantics.

In plain words: every chart in this project looks the same and uses
colour to mean the same thing — the model under discussion is dark
blue, the baselines are grey, the world's own drift-apart reference is
a black dashed line, misses are red, and anything from a deliberately
broken test fixture is orange *and* labelled in words. Charts are
8x5 inches at 150 dpi, white, with no box around them and only light
dotted gridlines (reporting ADR-R1, Tufte's data-ink rule).

Two pins make the SVG copy of every chart byte-identical run to run:
`svg.hashsalt` (matplotlib otherwise salts its element ids at random)
and `metadata={'Date': None}` at save time (it otherwise stamps the
wall clock into the file). Reporting spec §7 executed both: either one
alone leaves a difference. PNG bytes are not claimed identical
(ADR-R5) — font rasterisation is not something this project controls.

The fixture rule lives in one place (ADR-R4, P5-C01): `save_figure` takes a
required `contains_fixture` flag and reads the figure object itself. A
figure holding fixture output cannot be saved without the fixture sentence
on it, and a figure of real models cannot be saved wearing it. No renderer
has discretion, and none can forget. `display_name` and `line_colour` give
every chart the same `FIXTURE:` prefix and orange line.

Figures are created directly from `matplotlib.figure.Figure`, never
through `pyplot`: no global figure registry, no display backend, no
hidden state between renders.

One deliberate piece of process-wide state remains: `apply_style()`
writes into `matplotlib.rcParams`, which is global. This is safe only
because the project has exactly one style and every `new_figure()`
re-applies the same fixed dict, so no render can observe a different
style depending on what ran before it. If a second style is ever
needed, scope it with `matplotlib.rc_context` around both figure
creation *and* save — `svg.hashsalt` must be in force at save time
for the SVG bytes to be reproducible (code-review-001, Panel E).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
from matplotlib.axes import Axes
from matplotlib.collections import Collection
from matplotlib.colors import to_hex
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.text import Text

from wmj.errors import WmjError

# --- ADR-R1 fixed colour semantics -----------------------------------------
MODEL_COLOUR = "#1f4e9c"  # model under discussion: dark blue
BASELINE_COLOUR = "#8c8c8c"  # baselines: mid grey
DIVERGENCE_COLOUR = "#000000"  # world's own divergence reference: black (dashed)
EXCEPTION_COLOUR = "#c8102e"  # misses: red circles
FIXTURE_COLOUR = "#e07b00"  # fixtures: orange, and always labelled in words
FIXTURE_TEXT_COLOUR = "#a35400"  # the fixture orange darkened for text: 5.5:1 on white (review of P5-C04)
BAND_COLOURS = {"green": "#cfe8cf", "amber": "#f5e1a4", "red": "#f2c4c4"}
REGION_EDGE_COLOUR = "#4d4d4d"  # the edge of a model's stated 90% range (Chart 1's line): dark neutral grey
REGION_FILL_COLOUR = "#efefef"  # the inside of that range: the lightest grey, behind everything

# --- ADR-R1 geometry ---------------------------------------------------------
FIGURE_SIZE = (8.0, 5.0)  # inches
DPI = 150

# --- §7 SVG reproducibility pin ---------------------------------------------
SVG_HASHSALT = "world-model-judge"

# --- ADR-R4 fixture sentence, verbatim ---------------------------------------
FIXTURE_LABEL = (
    "TEST FIXTURE — deliberately broken model; a detected fault here is the "
    "instrument working, not a finding."
)

FIXTURE_PREFIX = "FIXTURE: "  # ADR-R4: on every on-chart label of a fixture


class FixtureLabelError(WmjError):
    """A figure's fixture label does not match whether it holds fixture output (ADR-R4, RP-8)."""


_RC = {
    "figure.figsize": FIGURE_SIZE,
    "figure.dpi": DPI,
    "savefig.dpi": DPI,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.linestyle": ":",
    "grid.color": "#c8c8c8",
    "grid.linewidth": 0.6,
    "axes.titleweight": "normal",
    "font.size": 9.0,
    "svg.hashsalt": SVG_HASHSALT,
    "svg.fonttype": "none",  # text stays text: diff-able, grep-able SVG
}


def apply_style() -> None:
    """Install the ADR-R1 style into matplotlib's rcParams (idempotent)."""
    matplotlib.rcParams.update(_RC)


def new_figure() -> Figure:
    """A styled, pyplot-free figure at ADR-R1's geometry."""
    apply_style()
    return Figure(figsize=FIGURE_SIZE, dpi=DPI)


FIXTURE_FONT_SIZE = 7.5


def _wrapped(text: str, max_chars: int) -> str:
    """Break `text` at spaces into lines of at most `max_chars` (a single longer word keeps its own line)."""
    lines: list[str] = []
    for word in text.split():
        if lines and len(lines[-1]) + 1 + len(word) <= max_chars:
            lines[-1] += " " + word
        else:
            lines.append(word)
    return "\n".join(lines)


def _same_words(a: str, b: str) -> bool:
    return a.split() == b.split()


def mark_fixture(ax: Axes) -> Text:
    """ADR-R4: draw the fixture sentence inside the axes, top-left, wrapped to the axes' own width.

    Call it once the figure's layout is final (after `subplots_adjust`): the sentence is broken into lines
    that fit this panel, so in a row of narrow panels it is never cut off by the next one. Returns the Text
    artist so a test can assert the call happened on the figure object before it was saved (TC-RP8-01).
    """
    width_points = ax.get_position().width * ax.figure.get_figwidth() * 72.0
    max_chars = max(20, int(width_points * 0.95 / (0.5 * FIXTURE_FONT_SIZE)))  # about half an em per character, measured
    return ax.text(
        0.02,
        0.97,
        _wrapped(FIXTURE_LABEL, max_chars),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=FIXTURE_FONT_SIZE,
        color=FIXTURE_COLOUR,
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "alpha": 0.85,
              "edgecolor": FIXTURE_COLOUR},
        zorder=10,
    )


def _require_flag(value: object) -> bool:
    if type(value) is not bool:
        raise FixtureLabelError(f"the fixture flag must be True or False, got {value!r}")
    return value


def display_name(name: str, is_fixture: bool) -> str:
    """The name a chart prints for a model: fixtures carry the `FIXTURE: ` prefix, exactly once."""
    _require_flag(is_fixture)
    if type(name) is not str or not name.strip() or name != name.strip():
        raise ValueError(f"a model name must be a non-empty string with no surrounding spaces, got {name!r}")
    if is_fixture and not name.startswith(FIXTURE_PREFIX):
        return FIXTURE_PREFIX + name
    return name


def line_colour(is_fixture: bool, *, baseline: bool = False) -> str:
    """ADR-R1's colour for a model's line or marker; a fixture is orange whatever its role."""
    if _require_flag(is_fixture):
        return FIXTURE_COLOUR
    return BASELINE_COLOUR if baseline else MODEL_COLOUR


def _shown(text: Text) -> bool:
    """A text that will actually appear: visible, on a visible axes (if any), not fully transparent."""
    alpha = text.get_alpha()
    return text.get_visible() and (text.axes is None or text.axes.get_visible()) and (alpha is None or alpha > 0)


def has_fixture_mark(fig: Figure) -> bool:
    """True if the exact ADR-R4 sentence is drawn, and shown, on the figure: in any axes or on the figure itself."""
    texts = list(fig.texts)
    for ax in fig.get_axes():
        texts.extend(ax.texts)
    return any(_same_words(text.get_text(), FIXTURE_LABEL) and _shown(text) for text in texts)


def _colours(artist) -> list:
    if isinstance(artist, Line2D):
        return [artist.get_color(), artist.get_markerfacecolor(), artist.get_markeredgecolor()]
    if isinstance(artist, Collection):
        return [*artist.get_facecolors(), *artist.get_edgecolors()]
    if isinstance(artist, Patch):
        return [artist.get_facecolor(), artist.get_edgecolor()]
    if isinstance(artist, Text):
        return [artist.get_color()]
    return []


def fixture_traces(fig: Figure) -> list[str]:
    """What on the figure looks like fixture output: a line, collection, patch or text in the fixture colour,
    or a text or line label starting `FIXTURE:` (annotation arrows, text boxes and image pixels are not read).

    A figure saved as holding no fixture output must show none of these (review of P5-C01: the flag is
    the caller's claim, so the figure is checked against it)."""
    found = []
    for artist in fig.findobj():
        if isinstance(artist, Text) and artist.get_text().lstrip().startswith(FIXTURE_PREFIX.strip()):
            found.append(f"text {artist.get_text()[:40]!r}")
        if any(to_hex(colour, keep_alpha=False) == FIXTURE_COLOUR for colour in _colours(artist)):
            label = artist.get_label() if hasattr(artist, "get_label") else ""
            found.append(f"{type(artist).__name__} {label!r} in the fixture colour")
    for ax in fig.get_axes():  # line labels a legend would show
        for line in ax.get_lines():
            if str(line.get_label()).startswith(FIXTURE_PREFIX.strip()):
                found.append(f"line label {line.get_label()!r}")
    return found


def direct_labels(ax: Axes, items: list[tuple[float, float, str, str]], *, fontsize: float = 8.0) -> list[Text]:
    """Label line ends directly (Tufte: no legend), nudging labels apart vertically so none overprints.

    `items` are `(x, y, text, colour)` in data coordinates. Each label sits just right of its point, centred on
    it. Two neighbours' centres must be at least half of each label's height apart, plus air: 0.65 x the font
    size per line of each (review of P5-C03: a two-line label over a one-line one needs more than one line's
    gap). Labels are pushed down from the top, then back up from the bottom of the axes, so none falls below
    the panel. Call after the axes' limits are final."""
    ax.autoscale_view()
    px_per_point = ax.figure.dpi / 72.0
    placed = sorted(((ax.transData.transform((x, y))[1], x, y, text, colour) for x, y, text, colour in items),
                    key=lambda item: item[0], reverse=True)

    def half(text: str) -> float:  # half a label's height plus air, in pixels
        return 0.65 * fontsize * (text.count("\n") + 1) * px_per_point

    targets = [item[0] for item in placed]
    for i in range(1, len(placed)):  # top-down: each label at most at its own height, and clear of the one above
        targets[i] = min(targets[i], targets[i - 1] - half(placed[i - 1][3]) - half(placed[i][3]))
    if placed:  # bottom-up: nothing below the panel's lower edge
        targets[-1] = max(targets[-1], ax.get_window_extent().y0 + half(placed[-1][3]))
        for i in range(len(placed) - 2, -1, -1):
            targets[i] = max(targets[i], targets[i + 1] + half(placed[i + 1][3]) + half(placed[i][3]))
    texts = []
    for (display_y, x, y, text, colour), target in zip(placed, targets):
        offset = (target - display_y) / px_per_point
        texts.append(ax.annotate(text, xy=(x, y), xytext=(4, offset), textcoords="offset points",
                                 ha="left", va="center", fontsize=fontsize, color=colour))
    return texts


def save_figure(fig: Figure, png_path: Path, svg_path: Path, *, contains_fixture: bool) -> None:
    """Write the PNG and its SVG duplicate (ADR-R1) with both §7 pins.

    `contains_fixture` must be stated: a figure holding fixture output is refused unless the ADR-R4
    sentence is on it, and a figure without fixture output is refused if it carries the sentence
    (TC-RP8-01: the label is checked on the figure object before anything is written)."""
    marked = has_fixture_mark(fig)
    if _require_flag(contains_fixture) and not marked:
        raise FixtureLabelError(
            "this figure holds fixture output but the fixture sentence is not on it (ADR-R4, RP-8): "
            "call mark_fixture on its axes before saving"
        )
    if not contains_fixture and marked:
        raise FixtureLabelError("this figure holds no fixture output but carries the fixture sentence (ADR-R4)")
    if not contains_fixture and (traces := fixture_traces(fig)):
        raise FixtureLabelError(
            f"this figure was declared to hold no fixture output, but it shows {len(traces)} sign(s) of it "
            f"(first: {traces[0]}): pass contains_fixture=True if any model on it is a fixture, and call "
            "mark_fixture(ax) on its axes"
        )
    png_path = Path(png_path)
    svg_path = Path(svg_path)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    svg_path.parent.mkdir(parents=True, exist_ok=True)
    # Save to a sibling temp path and replace atomically, so a reader
    # never sees a partly-written chart (code-review-001, Panel E). The
    # format is passed explicitly because the temp name's extension is
    # not the format. `Path.replace` is the atomic rename; reporting
    # deliberately imports no `os` (TC-NF6-10).
    png_tmp = png_path.with_name(png_path.name + ".tmp")
    svg_tmp = svg_path.with_name(svg_path.name + ".tmp")
    fig.savefig(png_tmp, format="png", dpi=DPI)
    png_tmp.replace(png_path)
    fig.savefig(svg_tmp, format="svg", metadata={"Date": None})
    svg_tmp.replace(svg_path)
