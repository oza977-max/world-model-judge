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
    style.save_figure(fig, png, svg, contains_fixture=False)
    assert png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    first = svg.read_bytes()
    assert b"<svg" in first and b"dc:date" not in first

    style.save_figure(fig, png, svg, contains_fixture=False)
    assert svg.read_bytes() == first


# --- P5-C01: the fixture rule is enforced at the one place every chart passes through ---------------
# In plain words: a chart holding a deliberately broken test model must say so in words on the image
# (RP-8, ADR-R4). The save step checks the figure object itself, so no renderer can forget, and it
# refuses the opposite mistake too: a chart of real models wearing the fixture label.

import pytest


def _figure_with(*, marked: bool, at_figure_level: bool = False):
    fig = style.new_figure()
    ax = fig.subplots()
    ax.plot([1, 2], [1.0, 2.0])
    if marked and at_figure_level:
        fig.text(0.1, 0.9, style.FIXTURE_LABEL)
    elif marked:
        style.mark_fixture(ax)
    return fig


def test_tc_rp8_01_a_fixture_figure_is_saved_only_when_the_label_is_on_it(tmp_path):
    png, svg = tmp_path / "c.png", tmp_path / "c.svg"
    with pytest.raises(style.FixtureLabelError, match="fixture"):
        style.save_figure(_figure_with(marked=False), png, svg, contains_fixture=True)
    assert not png.exists() and not svg.exists()  # refused before anything is written
    style.save_figure(_figure_with(marked=True), png, svg, contains_fixture=True)
    assert style.FIXTURE_LABEL.encode() in svg.read_bytes()  # the words are on the image itself


def test_a_figure_without_fixture_output_may_not_carry_the_fixture_label(tmp_path):
    png, svg = tmp_path / "c.png", tmp_path / "c.svg"
    with pytest.raises(style.FixtureLabelError, match="no fixture"):
        style.save_figure(_figure_with(marked=True), png, svg, contains_fixture=False)
    style.save_figure(_figure_with(marked=False), png, svg, contains_fixture=False)
    assert style.FIXTURE_LABEL.encode() not in svg.read_bytes()


def test_the_fixture_flag_must_be_stated_and_must_be_a_real_boolean(tmp_path):
    png, svg = tmp_path / "c.png", tmp_path / "c.svg"
    with pytest.raises(TypeError):
        style.save_figure(_figure_with(marked=False), png, svg)  # no silent default
    for flag in (1, 0, "yes", None):
        with pytest.raises(style.FixtureLabelError, match="True or False"):
            style.save_figure(_figure_with(marked=False), png, svg, contains_fixture=flag)


def test_has_fixture_mark_reads_axes_and_figure_level_text_and_ignores_lookalikes():
    assert style.has_fixture_mark(_figure_with(marked=True))
    assert style.has_fixture_mark(_figure_with(marked=True, at_figure_level=True))
    assert not style.has_fixture_mark(_figure_with(marked=False))
    fig = _figure_with(marked=False)
    fig.axes[0].set_title("TEST FIXTURE")  # a title that merely starts the same way is not the label
    fig.axes[0].text(0.5, 0.5, style.FIXTURE_LABEL[:-1])
    assert not style.has_fixture_mark(fig)


def test_display_name_prefixes_fixtures_once_and_leaves_other_models_alone():
    assert style.display_name("fx-brittle", True) == "FIXTURE: fx-brittle"
    assert style.display_name("direct", False) == "direct"
    assert style.display_name("FIXTURE: fx-brittle", True) == "FIXTURE: fx-brittle"  # never doubled
    for bad in ("", "   ", " direct", "direct\n"):
        with pytest.raises(ValueError, match="name"):
            style.display_name(bad, False)
    with pytest.raises(style.FixtureLabelError, match="True or False"):
        style.display_name("direct", 1)


def test_line_colour_follows_the_fixed_semantics_and_fixture_orange_wins():
    assert style.line_colour(False) == style.MODEL_COLOUR
    assert style.line_colour(False, baseline=True) == style.BASELINE_COLOUR
    assert style.line_colour(True) == style.FIXTURE_COLOUR
    assert style.line_colour(True, baseline=True) == style.FIXTURE_COLOUR


def test_a_figure_declared_fixture_free_is_refused_if_it_shows_fixture_output(tmp_path):
    # review of P5-C01: the flag is the caller's claim; a renderer that passes the first envelope's flag
    # instead of any(...) must not publish a fixture line unlabelled
    png, svg = tmp_path / "c.png", tmp_path / "c.svg"
    cases = []
    fig = style.new_figure(); ax = fig.subplots(); ax.plot([1, 2], [1.0, 2.0], color=style.FIXTURE_COLOUR); cases.append(fig)
    fig = style.new_figure(); ax = fig.subplots(); ax.plot([1, 2], [1.0, 2.0], label="FIXTURE: fx-brittle"); cases.append(fig)
    fig = style.new_figure(); ax = fig.subplots(); ax.scatter([1, 2], [1.0, 2.0], color=style.FIXTURE_COLOUR); cases.append(fig)
    fig = style.new_figure(); ax = fig.subplots(); ax.annotate("FIXTURE: fx", xy=(0.5, 0.5)); cases.append(fig)
    fig = style.new_figure(); ax = fig.subplots(); ax.bar([1], [1.0], color=style.FIXTURE_COLOUR); cases.append(fig)
    for fig in cases:
        with pytest.raises(style.FixtureLabelError, match="shows fixture output"):
            style.save_figure(fig, png, svg, contains_fixture=False)
    assert not svg.exists()
    clean = style.new_figure(); ax = clean.subplots()
    ax.plot([1, 2], [1.0, 2.0], color=style.MODEL_COLOUR, label="direct"); ax.plot([1, 2], [2.0, 3.0], color=style.BASELINE_COLOUR)
    ax.fill_between([1, 2], 0, 1, color=style.BAND_COLOURS["amber"])
    style.save_figure(clean, png, svg, contains_fixture=False)  # ordinary colours and names pass
    assert style.fixture_traces(clean) == []


def test_a_hidden_label_does_not_count_as_the_label(tmp_path):
    png, svg = tmp_path / "c.png", tmp_path / "c.svg"
    for hide in ("text", "axes", "alpha"):
        fig = style.new_figure(); ax = fig.subplots(); ax.plot([1, 2], [1.0, 2.0])
        label = style.mark_fixture(ax)
        if hide == "text":
            label.set_visible(False)
        elif hide == "axes":
            ax.set_visible(False)
        else:
            label.set_alpha(0.0)
        assert not style.has_fixture_mark(fig), hide
        with pytest.raises(style.FixtureLabelError, match="not on it"):
            style.save_figure(fig, png, svg, contains_fixture=True)
