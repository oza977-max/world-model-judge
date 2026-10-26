"""results.html: everything escaped, nothing fetched, the model card first, fixtures labelled.

In plain words: the page is plain text glued into HTML, so these tests check that a name can never turn
into markup, that the page links only its own images, and that a fixture model's section says in words
what it is.
"""

from __future__ import annotations

import pytest

from wmj.reporting import page, style

CARD = ("first limit", "second <limit> & more")
NEVER = ("never this",)


def _section(name="direct", is_fixture=False, tasks=("lv-control", "lv-planning")):
    shown = page.Shown("charts/x.png", "a caption")
    exceptions = tuple((t, r, s, page.Shown(f"charts/{t}-{r}-h{s}.png", f"{t} {r} {s}"))
                       for t in tasks for r in ("training", "out-of-range") for s in (1, 4))
    return page.ModelSection(name, is_fixture, shown, shown, exceptions)


def _world(*models):
    return page.WorldSection("lv", page.Shown("charts/lv-comparison.png", "table caption"), models or (_section(),))


def test_every_text_is_escaped_so_a_name_never_becomes_markup():
    html = page.render_page(CARD, NEVER, (_world(_section(name='<script>alert("x")</script>')),))
    assert "<script" not in html and "&lt;script&gt;alert(&quot;x&quot;)&lt;/script&gt;" in html
    assert "second &lt;limit&gt; &amp; more" in html


def test_the_page_is_static_and_links_only_its_own_images():
    html = page.render_page(CARD, NEVER, (_world(),))
    assert "<script" not in html and "http://" not in html and "https://" not in html
    sources = [part.split('"')[0] for part in html.split('src="')[1:]]
    assert sources and all(s.startswith("charts/") and s.endswith(".png") for s in sources)


def test_the_model_card_and_methods_note_come_before_any_chart():
    html = page.render_page(CARD, NEVER, (_world(),))
    first_chart = html.index("<img")
    assert html.index(page.CARD_HEADING) < html.index(page.NEVER_TESTED_HEADING) < html.index(page.METHODS_HEADING) < first_chart
    assert html.index(page.escape(page.METHODS_NOTE)) < first_chart


def test_a_tasks_exception_charts_sit_side_by_side_in_one_row():
    html = page.render_page(CARD, NEVER, (_world(),))
    rows = html.split('<div class="row">')[1:]
    assert len(rows) == 2  # one row per task
    assert all(row.split("</div>")[0].count("<figure>") == 4 for row in rows)  # its regions and steps together


def test_a_fixture_section_is_labelled_in_words():
    html = page.render_page(CARD, NEVER, (_world(_section(), _section(name="fx-brittle", is_fixture=True)),))
    assert "FIXTURE: fx-brittle" in html and page.escape(style.FIXTURE_LABEL) in html
    plain = page.render_page(CARD, NEVER, (_world(),))
    assert page.escape(style.FIXTURE_LABEL) not in plain


@pytest.mark.parametrize(
    "args",
    [((), NEVER, None), (CARD, (), None), (CARD, NEVER, ())],
)
def test_a_page_without_its_model_card_or_results_is_refused(args):
    limitations, never, worlds = args
    with pytest.raises(page.PageError):
        page.render_page(limitations, never, (_world(),) if worlds is None else worlds)


def test_only_text_goes_on_the_page():
    with pytest.raises(page.PageError):
        page.escape(3)


def test_attribute_values_are_escaped_too():
    shown = page.Shown("charts/x.png", 'say "hi" & <b>it\'s</b>')
    model = page.ModelSection("direct", False, shown, shown, ())
    html = page.render_page(CARD, NEVER, (page.WorldSection("lv", shown, (model,)),))
    assert 'alt="say &quot;hi&quot; &amp; &lt;b&gt;it&#x27;s&lt;/b&gt;"' in html and 'alt="say "' not in html



def test_chart_rows_are_an_equal_column_grid_at_most_two_wide_so_no_chart_is_stretched():
    # review of P5-C04, pass 2: a flex row stretched a lone chart on its line to twice the others' size
    assert "grid-template-columns: repeat(auto-fill, minmax(min(560px, 100%), 1fr))" in page.STYLE
    assert "flex" not in page.STYLE
