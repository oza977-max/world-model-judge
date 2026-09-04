"""Tests for wmj.reporting.style — one shared style, fixed colour semantics.

Reporting ADR-R1 (8x5 in at 150 dpi, white background, no top/right
spines, dotted grid, SVG duplicates with `svg.hashsalt` pinned),
ADR-R4 (`mark_fixture` draws the fixture sentence inside the axes —
TC-RP8-01 asserts the label call is present in the figure object
before save), §7 (SVG reproducibility needs both pins).
"""

from __future__ import annotations

import matplotlib

from wmj.reporting import style


def test_colour_semantics_are_fixed_and_distinct():
    colours = {
        style.MODEL_COLOUR,
        style.BASELINE_COLOUR,
        style.DIVERGENCE_COLOUR,
        style.FIXTURE_COLOUR,
        style.EXCEPTION_COLOUR,
    }
    assert len(colours) == 5
    assert style.DIVERGENCE_COLOUR == "#000000"  # black dashed reference


def test_apply_style_pins_adr_r1_figure_geometry_and_svg_hashsalt():
    style.apply_style()
    assert tuple(matplotlib.rcParams["figure.figsize"]) == (8.0, 5.0)
    assert matplotlib.rcParams["savefig.dpi"] == 150
    assert matplotlib.rcParams["figure.facecolor"] == "white"
    assert matplotlib.rcParams["axes.spines.top"] is False
    assert matplotlib.rcParams["axes.spines.right"] is False
    assert matplotlib.rcParams["grid.linestyle"] == ":"
    assert matplotlib.rcParams["svg.hashsalt"] == style.SVG_HASHSALT


def test_new_figure_uses_the_style_without_pyplot():
    fig = style.new_figure()
    assert tuple(fig.get_size_inches()) == (8.0, 5.0)
    assert fig.dpi == 150


def test_mark_fixture_places_the_adr_r4_sentence_inside_the_axes():
    fig = style.new_figure()
    ax = fig.subplots()
    text = style.mark_fixture(ax)
    assert text in ax.texts
    assert text.get_text() == style.FIXTURE_LABEL
    assert text.get_text().startswith("TEST FIXTURE")
    assert text.get_color() == style.FIXTURE_COLOUR


def test_save_figure_writes_png_and_dateless_svg_reproducibly(tmp_path):
    fig = style.new_figure()
    ax = fig.subplots()
    ax.plot([1, 2, 3], [1.0, 2.0, 4.0], color=style.MODEL_COLOUR)

    png, svg = tmp_path / "a" / "c.png", tmp_path / "a" / "c.svg"
    style.save_figure(fig, png, svg)
    assert png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    first = svg.read_bytes()
    assert b"<svg" in first and b"dc:date" not in first

    style.save_figure(fig, png, svg)
    assert svg.read_bytes() == first
