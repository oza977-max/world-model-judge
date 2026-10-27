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
    return page.ModelSection(name, is_fixture, shown, shown, exceptions, "an action sentence")


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
    model = page.ModelSection("direct", False, shown, shown, (), "an action sentence")
    html = page.render_page(CARD, NEVER, (page.WorldSection("lv", shown, (model,)),))
    assert 'alt="say &quot;hi&quot; &amp; &lt;b&gt;it&#x27;s&lt;/b&gt;"' in html and 'alt="say "' not in html



def test_chart_rows_are_an_equal_column_grid_at_most_two_wide_so_no_chart_is_stretched():
    # review of P5-C04, pass 2: a flex row stretched a lone chart on its line to twice the others' size
    assert "grid-template-columns: repeat(auto-fill, minmax(min(560px, 100%), 1fr))" in page.STYLE
    assert "flex" not in page.STYLE



def test_the_comparison_table_scrolls_in_its_own_box_rather_than_shrinking_on_a_narrow_screen():
    html = page.render_page(CARD, NEVER, (_world(),))
    assert '<div class="scroll"><img src="charts/lv-comparison.png"' in html
    assert html.count('<div class="scroll">') == 1  # only the table: the other charts shrink with the page
    assert ".scroll { overflow-x: auto; }" in page.STYLE


def test_the_pages_claims_are_pinned_word_for_word():
    # code-review-004: flipping "not a finding" or "must be byte-for-byte identical" passed every test
    assert ("A model whose name starts with “FIXTURE:” is a deliberately broken test model; a fault found in "
            "it is the instrument working, not a finding.") in page.INTRO
    assert ("the whole run is repeated ten times, in separate processes, and the verdict files must be "
            "byte-for-byte identical every time.") in page.METHODS_NOTE
    assert "That check is a best-effort tripwire, not a guarantee" in page.METHODS_NOTE
    assert page.ACTION_BASELINE.endswith("This is expected and is not a finding.")
    assert page.ACTION_FIXTURE.endswith("the instrument working, not a finding.")


@pytest.mark.parametrize(
    "blind, fixture, baseline, expected",
    [
        (False, False, False, "Responds to its action lever: its guess or its error bar moved in 5 of 48"),
        (False, True, True, "Responds to its action lever"),
        (True, False, True, "Ignores its action lever, by construction"),
        (True, True, False, "Ignores its action lever: it is a deliberately broken test model"),
        (True, False, False, "Flagged (MU-3): this model's guess and error bar did not change with the action lever in any of 48"),
    ],
)
def test_the_action_sentence_never_calls_a_baseline_or_a_fixture_a_finding(blind, fixture, baseline, expected):
    assert page.action_sentence(blind, 48, 0 if blind else 5, is_fixture=fixture, is_baseline=baseline).startswith(expected)
