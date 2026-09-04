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

Figures are created directly from `matplotlib.figure.Figure`, never
through `pyplot`: no global figure registry, no display backend, no
hidden state between renders.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.text import Text

# --- ADR-R1 fixed colour semantics -----------------------------------------
MODEL_COLOUR = "#1f4e9c"  # model under discussion: dark blue
BASELINE_COLOUR = "#8c8c8c"  # baselines: mid grey
DIVERGENCE_COLOUR = "#000000"  # world's own divergence reference: black (dashed)
EXCEPTION_COLOUR = "#c8102e"  # misses: red circles
FIXTURE_COLOUR = "#e07b00"  # fixtures: orange, and always labelled in words
BAND_COLOURS = {"green": "#cfe8cf", "amber": "#f5e1a4", "red": "#f2c4c4"}

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


def mark_fixture(ax: Axes) -> Text:
    """ADR-R4: draw the fixture sentence inside the axes, top-left.

    Returns the Text artist so a test can assert the call happened on
    the figure object before it was saved (TC-RP8-01).
    """
    return ax.text(
        0.02,
        0.97,
        FIXTURE_LABEL,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.5,
        color=FIXTURE_COLOUR,
        wrap=True,
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "alpha": 0.85,
              "edgecolor": FIXTURE_COLOUR},
        zorder=10,
    )


def save_figure(fig: Figure, png_path: Path, svg_path: Path) -> None:
    """Write the PNG and its SVG duplicate (ADR-R1) with both §7 pins."""
    png_path = Path(png_path)
    svg_path = Path(svg_path)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    svg_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, format="png", dpi=DPI)
    fig.savefig(svg_path, format="svg", metadata={"Date": None})
