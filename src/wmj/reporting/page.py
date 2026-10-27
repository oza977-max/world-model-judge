"""wmj.reporting.page — `results.html`, the one page a reader opens (reporting ADR-R5).

In plain words: a single static web page that opens from disk, with no scripts and nothing fetched from
the internet. From the top it shows:
1. what the page is;
2. the model card: the judge's own statements of what this verdict does not tell you, then the list of
   what was never tested. These are copied word for word from the verdicts (judge ADR-J7), never
   reworded here, and come before any chart, so nobody reads a result without its limits;
3. a short note on how the judge was kept honest, and what that check cannot catch (TC-JU12-04);
4. for each world, the comparison table, then each model's charts. Every chart carries its caption
   underneath as text.

Every piece of text is escaped before it goes into the page, so a name can never become markup. Images are
linked by relative path, so the `out/` folder can be copied anywhere and still works.
"""

from __future__ import annotations

from dataclasses import dataclass

from wmj.errors import WmjError
from wmj.reporting import style

PAGE_TITLE = "World Model Judge — results"
CARD_HEADING = "What this verdict does not tell you"
NEVER_TESTED_HEADING = "Never tested"
METHODS_HEADING = "How the judge was kept honest"
INTRO = (
    "This page is the judge's verdict on each learned simulator: how far ahead it can be believed, for which "
    "task, measured against what, and what was never tested. Read the section below first: it is the judge's "
    "own account of its limits. Then each world has a comparison table and every model's charts, each with a "
    "caption saying how to read it. A model whose name starts with “FIXTURE:” is a deliberately broken "
    "test model; a fault found in it is the instrument working, not a finding."
)
# TC-JU12-04: the purity check's residual exactly as the build measured it (backlog A37), neither wider nor
# narrower; the reproducibility claim as narrowed by owner decision (backlog A29).
METHODS_NOTE = (
    "The judge must reach its verdict from the numbers it is handed and nothing else: no clock, no file, no "
    "random draw, no setting of the machine it runs on. While the tests run the judge, a purity check makes "
    "every such access it knows about fail loudly. That check is a best-effort tripwire, not a guarantee: it "
    "lists the routes it knows, and Python always has others. It cannot see a call made through ctypes (a "
    "low-level door that numpy already has open), a capability saved under another name before the check "
    "starts, or a few of numpy's date routines that read the clock from inside compiled code. What actually "
    "guarantees that nothing like that changed a result is a second check: the whole run is repeated ten "
    "times, in separate processes, and the verdict files must be byte-for-byte identical every time. That "
    "guarantee holds on the same CPU family and the same library build; a different processor may round "
    "differently."
)
STYLE = """
body { font-family: Inter, "Helvetica Neue", Arial, sans-serif; color: #1a1a1a; background: #ffffff;
       max-width: 1200px; margin: 0 auto; padding: 24px 16px; line-height: 1.5; }
h1 { font-size: 1.6em; } h2 { margin-top: 2em; border-bottom: 1px solid #d0d0d0; }
section.card { border: 2px solid #1f4e9c; padding: 8px 20px 16px; margin: 1.5em 0; } section.card h2 { margin-top: 0.6em; }
figure { margin: 1em 0 2em; } figure img { max-width: 100%; height: auto; border: 1px solid #eeeeee; }
.scroll { overflow-x: auto; } .scroll img { box-sizing: border-box; max-width: none; width: max(100%, 900px); }
figcaption { max-width: 60em; font-size: 0.95em; }
.row { display: grid; grid-template-columns: repeat(auto-fill, minmax(min(560px, 100%), 1fr)); gap: 16px; }
.row figure { margin: 0; }  /* equal columns, at most two per row on a wide screen: every chart the same size */
.fixture { color: #a35400; font-weight: 600; }  /* style.FIXTURE_TEXT_COLOUR: 5.5:1 on white */
"""


class PageError(WmjError):
    """The page cannot be assembled as specified (reporting ADR-R5); nothing is written."""


@dataclass(frozen=True)
class Shown:
    """One chart on the page: its image's path relative to `out/`, and its caption text."""

    image: str
    caption: str


@dataclass(frozen=True)
class ModelSection:
    name: str
    is_fixture: bool
    horizon: Shown
    calibration: Shown
    exceptions: tuple[tuple[str, str, int, Shown], ...]  # (task, region, horizon step, chart), in verdict order
    action: str  # the MU-3 action-response sentence for this model (`action_sentence`)


@dataclass(frozen=True)
class WorldSection:
    world: str
    comparison: Shown
    models: tuple[ModelSection, ...]


# MU-3 (owner decision 2026-10-26, code-review-004): what the action-response check found, in words. A baseline
# ignores the lever by construction and a fixture may be built to; neither is ever presented as a finding.
ACTION_RESPONDS = ("Responds to its action lever: its guess or its error bar moved in {k} of {n} probe pairs "
                   "(MU-3 check).")
ACTION_BASELINE = ("Ignores its action lever, by construction: it is a simple baseline that only looks at the past. "
                   "This is expected and is not a finding.")
ACTION_FIXTURE = ("Ignores its action lever: it is a deliberately broken test model, and the check catching it is the "
                  "instrument working, not a finding.")
ACTION_FLAGGED = ("Flagged (MU-3): this model's guess and error bar did not change with the action lever in any of "
                  "{n} probe pairs, at the three tested action levels, so it shows no use of the action it is given.")


def action_sentence(action_blind: bool, n_probes: int, n_responding: int, *, is_fixture: bool, is_baseline: bool) -> str:
    """The one sentence the page prints about a model's action-response check."""
    if not action_blind:
        return ACTION_RESPONDS.format(k=n_responding, n=n_probes)
    if is_baseline:
        return ACTION_BASELINE
    if is_fixture:
        return ACTION_FIXTURE
    return ACTION_FLAGGED.format(n=n_probes)


def escape(text: str) -> str:
    """Make text safe inside HTML, element content and attribute values alike."""
    if type(text) is not str:
        raise PageError(f"only text goes on the page, got {type(text).__name__}")
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;").replace("'", "&#x27;")


def _figure(shown: Shown, *, wide: bool = False) -> str:
    """One chart and its caption. A wide one (the comparison table) scrolls sideways inside its own box on a
    narrow screen rather than shrinking until its numbers cannot be read (review of P5-C04, pass 4)."""
    image = f'<img src="{escape(shown.image)}" alt="{escape(shown.caption)}">'
    if wide:
        image = f'<div class="scroll">{image}</div>'
    return f"<figure>{image}<figcaption>{escape(shown.caption)}</figcaption></figure>"


def _model(section: ModelSection) -> list[str]:
    name = style.display_name(section.name, section.is_fixture)
    css = ' class="fixture"' if section.is_fixture else ""
    parts = [f"<h3{css}>{escape(name)}</h3>"]
    if section.is_fixture:
        parts.append(f'<p class="fixture">{escape(style.FIXTURE_LABEL)}</p>')
    parts.append(f"<p>{escape(section.action)}</p>")
    parts += [_figure(section.horizon), _figure(section.calibration)]
    tasks = list(dict.fromkeys(task for task, _r, _s, _c in section.exceptions))
    for task in tasks:  # ADR-R2: a task's exception charts side by side
        parts.append(f"<h4>Misses against the stated 90% range — task {escape(task)}</h4>")
        row = [_figure(shown) for t, _region, _step, shown in section.exceptions if t == task]
        parts.append('<div class="row">' + "".join(row) + "</div>")
    return parts


def render_page(limitations: tuple[str, ...], not_tested: tuple[str, ...], worlds: tuple[WorldSection, ...]) -> str:
    """The whole page as text. The model card's lists come from the verdicts, verbatim and in order."""
    if not limitations or not not_tested:
        raise PageError("the model card needs the verdict's limitations and not-tested list (ADR-R5, JU-10)")
    if not worlds:
        raise PageError("the page needs at least one world's results")
    parts = [
        "<!DOCTYPE html>",
        '<html lang="en"><head><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        f"<title>{escape(PAGE_TITLE)}</title><style>{STYLE}</style></head><body>",
        f"<h1>{escape(PAGE_TITLE)}</h1>",
        f"<p>{escape(INTRO)}</p>",
        '<section class="card">',
        f"<h2>{escape(CARD_HEADING)}</h2>",
        "<ul>" + "".join(f"<li>{escape(s)}</li>" for s in limitations) + "</ul>",
        f"<h3>{escape(NEVER_TESTED_HEADING)}</h3>",
        "<ul>" + "".join(f"<li>{escape(s)}</li>" for s in not_tested) + "</ul>",
        "</section>",
        f"<h2>{escape(METHODS_HEADING)}</h2>",
        f"<p>{escape(METHODS_NOTE)}</p>",
    ]
    for world in worlds:
        parts.append(f"<h2>World: {escape(world.world)}</h2>")
        parts.append(_figure(world.comparison, wide=True))
        for model in world.models:
            parts += _model(model)
    parts.append("</body></html>")
    return "\n".join(parts) + "\n"
