"""wmj.reporting.comparison_table — Chart 4, the comparison table (reporting ADR-R2, RP-4).

In plain words: every model judged on one world, one row each, set side by side.
- **Ordinary error.** The first columns give each model's rank on ordinary error: its CRPS on the control
  task, in the training region, one step ahead. Its skill against the two simple baselines sits beside the
  rank, so an error number never appears alone (ADR-J1).
- **Trust horizon.** The other columns give, for every task and region, how far ahead the judge says the
  model can be trusted, in steps, in world time and as a share of the world's natural cycle ("—" where the
  world has none).
- **Disagreement.** Where ordinary error puts one real model strictly ahead of another and some
  trust-horizon column puts it strictly behind, both rows are marked "⇄ disagree". When no pair
  disagrees, the caption says so plainly, and no row is marked, because the marker is not decoration
  (TC-RP4-02).
- **The caption names one disagreeing pair** (owner decision 2026-10-26). It may call the two models
  "equals" only when they are equal by the pre-registered test: their one-step skill scores against both
  baselines differ by no more than the matching margin (MU-5's own test, `harness.prereg`). Such a pair
  is preferred, and among those the closest on ordinary error. Otherwise the caption names the closest
  disagreeing pair and says only what the table shows: "Ordinary error ranks A ahead of B; the judge
  trusts B further ahead on {task}, {region}", naming the column where B's lead is largest.
- **Fixtures.** Fixture rows are shown, labelled and tinted, but never marked and never named in the
  caption: a deliberately broken model is not a finding (MU-4).

The control task is the one with the tightest tolerance: that is how the worlds define it, and the verdict
does not carry a task's kind. Ranks and markers compare verdict numbers for display; nothing here computes
a judgment (reporting §4). A block that does not add up is refused before anything is drawn or written.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from matplotlib.figure import Figure
from matplotlib.table import Table

from wmj.errors import WmjError
from wmj.reporting import captions, style
from wmj.reporting.envelope import ChartPaths, chart_paths, read_envelope

TRAINING = "training"
DISAGREE = "⇄ disagree"
DISAGREE_FILL = "#fbefc4"  # a light highlight for marked rows
FIXTURE_FILL = "#fbe6d2"  # a light tint of the fixture orange for fixture rows (the orange itself is for words)
HEADER_FILL = "#f2f2f2"
CHAR_INCHES = 0.062  # a generous average character width at the table's 7.5 pt, so columns never clip
SKILL_KEYS = frozenset({"task", "region", "vs_persistence", "vs_linear", "crps"})
HORIZON_KEYS = frozenset({"task", "region", "tolerance", "steps", "world_time", "natural_units"})


class ComparisonBlockError(WmjError):
    """The envelopes for one world cannot be set out as Chart 4 specifies; nothing is drawn."""


@dataclass(frozen=True)
class Horizon:
    steps: int
    world_time: float
    natural_units: str | None


@dataclass(frozen=True)
class Row:
    """One model's row: its name, flags, its ordinary-error entry and its trust horizon per column."""

    name: str
    is_fixture: bool
    crps: float
    vs_persistence: float
    vs_linear: float
    horizons: tuple[Horizon, ...]  # in `Comparison.columns` order


@dataclass(frozen=True)
class Comparison:
    """Chart 4's content for one world, in display order (ordinary-error rank, then name)."""

    world: str
    control_task: str
    columns: tuple[tuple[str, str], ...]  # (task, region)
    rows: tuple[Row, ...]
    ranks: tuple[int, ...]
    marked: tuple[bool, ...]
    caption_pair: tuple[str, str] | None
    pair_matched: bool  # the caption pair is equal by the pre-registered margin, so "equals" is true
    lead_column: tuple[str, str] | None  # where the judge puts the caption's second model furthest ahead


@dataclass(frozen=True)
class ComparisonFigure:
    figure: Figure
    table: Table
    comparison: Comparison
    caption: str


def _number(value: object) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def _entries(block: object, name: str, keys: frozenset, label: str) -> list[dict]:
    if not isinstance(block, dict) or set(block) != {name} or not isinstance(block[name], list) or not block[name]:
        raise ComparisonBlockError(f"ADR-R2 Chart 4: {label} must be {{{name!r}: [...]}} with entries (judge §5)")
    for entry in block[name]:
        if not isinstance(entry, dict) or set(entry) != keys:
            raise ComparisonBlockError(f"ADR-R2 Chart 4: each {label} entry needs exactly {sorted(keys)}")
        if type(entry["task"]) is not str or not entry["task"] or type(entry["region"]) is not str or not entry["region"]:
            raise ComparisonBlockError(f"ADR-R2 Chart 4: a {label} entry's task and region must be non-empty text")
    keyed = [(e["task"], e["region"]) for e in block[name]]
    if len(set(keyed)) != len(keyed):
        raise ComparisonBlockError(f"ADR-R2 Chart 4: {label} has two entries for one (task, region)")
    return block[name]


def _columns(horizons: list[dict]) -> tuple[tuple[str, str], ...]:
    """Tasks in verdict order; within each, `training` first, then the other regions in verdict order."""
    tasks = list(dict.fromkeys(e["task"] for e in horizons))
    columns = []
    for task in tasks:
        regions = [e["region"] for e in horizons if e["task"] == task]
        if TRAINING not in regions:
            raise ComparisonBlockError(f"ADR-R2 Chart 4: task {task!r} has no {TRAINING!r} trust horizon (ADR-W4)")
        columns += [(task, TRAINING)] + [(task, r) for r in regions if r != TRAINING]
    return tuple(columns)


def _control_task(horizons: list[dict]) -> str:
    tolerances: dict[str, float] = {}
    for entry in horizons:
        tol = entry["tolerance"]
        if not _number(tol) or tol <= 0:
            raise ComparisonBlockError(f"ADR-R2 Chart 4: tolerance must be a positive number, got {tol!r}")
        if tolerances.setdefault(entry["task"], tol) != tol:
            raise ComparisonBlockError(f"ADR-R2 Chart 4: task {entry['task']!r} states two tolerances")
    tightest = min(tolerances.values())
    control = [task for task, tol in tolerances.items() if tol == tightest]
    if len(control) != 1:
        raise ComparisonBlockError(
            f"ADR-R2 Chart 4: tasks {control} share the tightest tolerance; the control task is ambiguous"
        )
    return control[0]


def _tolerances(horizons: list[dict]) -> tuple[tuple[str, float], ...]:
    """Each task's one tolerance, in verdict order (checked by `_control_task`)."""
    return tuple({e["task"]: e["tolerance"] for e in horizons}.items())


def _row(env, columns, tolerances, control: str) -> Row:
    skill = {(e["task"], e["region"]): e for e in _entries(env.verdict.get("skill"), "per_task_region", SKILL_KEYS, "skill")}
    horizons = _entries(env.verdict.get("trust_horizons"), "per_task", HORIZON_KEYS, "trust_horizons")
    _control_task(horizons)  # each task states one positive tolerance
    if _columns(horizons) != columns or _tolerances(horizons) != tolerances:
        raise ComparisonBlockError(
            f"ADR-R2 Chart 4: {env.model_name!r} was judged on different tasks, regions or tolerances from the others"
        )
    entry = skill.get((control, TRAINING))
    if entry is None:
        raise ComparisonBlockError(f"ADR-R2 Chart 4: {env.model_name!r} has no skill entry for ({control!r}, {TRAINING!r})")
    for key in ("crps", "vs_persistence", "vs_linear"):
        if not _number(entry[key]) or (key == "crps" and entry[key] < 0):
            raise ComparisonBlockError(f"ADR-R2 Chart 4: {env.model_name!r} skill {key} must be a finite number, got {entry[key]!r}")
    by_key = {(e["task"], e["region"]): e for e in horizons}
    cells = []
    for column in columns:
        h = by_key[column]
        if type(h["steps"]) is not int or h["steps"] < 0 or not _number(h["world_time"]) or h["world_time"] < 0:
            raise ComparisonBlockError(f"ADR-R2 Chart 4: {env.model_name!r} trust horizon {column} is not a valid length")
        if h["natural_units"] is not None and (type(h["natural_units"]) is not str or not h["natural_units"].strip()):
            raise ComparisonBlockError(f"ADR-R2 Chart 4: {env.model_name!r} natural_units must be text or null")
        cells.append(Horizon(h["steps"], h["world_time"], h["natural_units"]))
    return Row(env.model_name, env.is_fixture, entry["crps"], entry["vs_persistence"], entry["vs_linear"], tuple(cells))


def _discordant(a: Row, b: Row) -> bool:
    """Ordinary error puts one strictly ahead; some trust-horizon column puts the other strictly ahead."""
    if a.crps == b.crps:
        return False
    a_first = a.crps < b.crps
    return any((ha.steps < hb.steps) if a_first else (ha.steps > hb.steps) for ha, hb in zip(a.horizons, b.horizons, strict=True))


def disagreements(rows: tuple[Row, ...]) -> list[tuple[Row, Row]]:
    """Every discordant pair of real (non-fixture) models (owner decision 2026-10-26: strict pairs, no fixtures)."""
    real = [r for r in rows if not r.is_fixture]
    return [(a, b) for i, a in enumerate(real) for b in real[i + 1:] if _discordant(a, b)]


def matched(a: Row, b: Row, margin: float) -> bool:
    """Equal by the pre-registered test: one-step skill against each baseline within the matching margin (MU-5)."""
    return abs(a.vs_persistence - b.vs_persistence) <= margin and abs(a.vs_linear - b.vs_linear) <= margin


def _closest(pairs: list[tuple[Row, Row]], margin: float) -> tuple[tuple[str, str] | None, bool]:
    """The pair the caption names, better-CRPS model first, and whether it may be called equal.

    A pair equal by the pre-registered margin is preferred; among the candidates, the closest on ordinary
    error (ratio of the larger CRPS to the smaller), ties broken by name."""
    if not pairs:
        return None, False
    equal = [pair for pair in pairs if matched(*pair, margin)]
    candidates = equal or pairs

    def ordered(pair):
        return tuple(sorted(pair, key=lambda r: (r.crps, r.name)))

    def closeness(pair):
        better, worse = ordered(pair)
        ratio = math.inf if better.crps == 0 else worse.crps / better.crps
        return (ratio, better.name, worse.name)

    better, worse = ordered(min(candidates, key=closeness))
    return (better.name, worse.name), bool(equal)


def _check_margin(margin: object) -> float:
    if type(margin) is not float or not 0.0 < margin < 1.0:
        raise ComparisonBlockError(f"ADR-R2 Chart 4: the matching margin must be a number strictly between 0 and 1 (MU-5), got {margin!r}")
    return margin


def comparison_rows(envelopes: list[dict], matching_margin: float) -> Comparison:
    """Validate one world's envelopes and set out Chart 4's rows, ranks, markers and caption pair."""
    margin = _check_margin(matching_margin)
    if not isinstance(envelopes, list) or not envelopes:
        raise ComparisonBlockError("ADR-R2 Chart 4: needs the envelopes of every model on one world (reporting §5)")
    try:
        envs = [read_envelope(e) for e in envelopes]
    except (ValueError, TypeError) as error:
        raise ComparisonBlockError(f"ADR-R2 Chart 4: {error}") from error
    worlds = {e.world for e in envs}
    if len(worlds) != 1:
        raise ComparisonBlockError(f"ADR-R2 Chart 4: one table per world, got envelopes for {sorted(worlds)}")
    names = [e.model_name for e in envs]
    if len(set(names)) != len(names):
        raise ComparisonBlockError(f"ADR-R2 Chart 4: a model appears twice: {sorted(n for n in set(names) if names.count(n) > 1)}")
    first = _entries(envs[0].verdict.get("trust_horizons"), "per_task", HORIZON_KEYS, "trust_horizons")
    columns = _columns(first)
    control = _control_task(first)
    tolerances = _tolerances(first)
    rows = sorted((_row(e, columns, tolerances, control) for e in envs), key=lambda r: (r.crps, r.name))
    if all(r.is_fixture for r in rows):
        raise ComparisonBlockError("ADR-R2 Chart 4: every model is a fixture; there is nothing to compare (MU-4)")
    ranks = tuple(1 + sum(other.crps < r.crps for other in rows) for r in rows)  # competition rank: ties share
    pairs = disagreements(tuple(rows))
    in_pair = {r.name for pair in pairs for r in pair}
    pair, pair_matched = _closest(pairs, margin)
    lead = None
    if pair is not None:
        by_name = {r.name: r for r in rows}
        ahead, behind = by_name[pair[0]], by_name[pair[1]]
        gaps = [(hb.steps - ha.steps, -i) for i, (ha, hb) in enumerate(zip(ahead.horizons, behind.horizons, strict=True))]
        gap, index = max(gaps)
        if gap <= 0:  # cannot happen for a discordant pair; refuse rather than print an untrue sentence
            raise ComparisonBlockError("ADR-R2 Chart 4: the caption pair has no column where the judge ranks the second ahead")
        lead = columns[-index]
    marked = tuple(r.name in in_pair for r in rows)
    return Comparison(worlds.pop(), control, columns, tuple(rows), ranks, marked, pair, pair_matched, lead)


def _two(value: float) -> str:
    """Two decimals, never "-0.00": a skill a hair below zero reads as zero, not as a sign without a value."""
    text = f"{value:.2f}"
    return "0.00" if text == "-0.00" else text


def _horizon_text(h: Horizon) -> str:
    steps = f"{h.steps} step" if h.steps == 1 else f"{h.steps} steps"
    time = f"{h.world_time:g} time unit" if h.world_time == 1 else f"{h.world_time:g} time units"
    return f"{steps}\n{time}\n{h.natural_units if h.natural_units is not None else '—'}"


def build_comparison_figure(comparison: Comparison) -> ComparisonFigure:
    """Draw the table (one row per model) with the fixture sentence above it when any row is a fixture."""
    has_fixture = any(r.is_fixture for r in comparison.rows)
    header = ["model", f"ordinary error rank\n({comparison.control_task}, {TRAINING},\none step)",
              *(f"trust horizon\n{task}\n{region}" for task, region in comparison.columns), DISAGREE]
    body = []
    for row, rank, marked in zip(comparison.rows, comparison.ranks, comparison.marked, strict=True):
        error = f"rank {rank} · CRPS {row.crps:.3g}\nskill vs persistence {_two(row.vs_persistence)}\nskill vs linear {_two(row.vs_linear)}"
        cells = [style.display_name(row.name, row.is_fixture), error, *(_horizon_text(h) for h in row.horizons)]
        body.append([*cells, DISAGREE if marked else ""])
    # each column as wide as its longest line (plus air), so no text leaves its cell; the figure grows to fit
    chars = [3 + max(len(line) for text in column for line in text.split("\n")) for column in zip(header, *body, strict=True)]
    # fixed geometry in inches: a title band, the fixture band when needed, then one band per table row
    title_in, fixture_in, row_in = 0.55, (0.45 if has_fixture else 0.0), 0.62
    table_in = row_in * (len(comparison.rows) + 1)
    height = title_in + fixture_in + table_in + 0.1
    fig = style.new_figure()
    fig.set_size_inches(max(8.0, sum(chars) * CHAR_INCHES / 0.96), height)
    ax = fig.add_axes([0.02, 0.1 / height, 0.96, (fixture_in + table_in) / height])
    ax.set_axis_off()
    top = table_in / (fixture_in + table_in)
    table = ax.table(cellText=body, colLabels=header, cellLoc="center", loc="upper center", bbox=[0.0, 0.0, 1.0, top],
                     colWidths=[c / sum(chars) for c in chars])
    table.auto_set_font_size(False)
    table.set_fontsize(7.5)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#d0d0d0")
        cell.set_linewidth(0.6)
        if r == 0:
            cell.set_facecolor(HEADER_FILL)
            continue
        row = comparison.rows[r - 1]
        if row.is_fixture:
            cell.set_facecolor(FIXTURE_FILL)
            if c == 0:
                cell.get_text().set_color(style.FIXTURE_TEXT_COLOUR)  # darker, readable on the tint
        elif comparison.marked[r - 1]:
            cell.set_facecolor(DISAGREE_FILL)
    fig.suptitle(f"Ordinary error and trust horizon — {comparison.world}", fontsize=10)
    if has_fixture:
        style.mark_fixture(ax)
    caption = captions.chart4_caption(comparison.caption_pair, equals=comparison.pair_matched, where=comparison.lead_column)
    return ComparisonFigure(fig, table, comparison, caption)


def render_comparison_chart(envelopes: list[dict], charts_dir: Path, captions_dir: Path, matching_margin: float) -> ChartPaths:
    """Chart 4 for one world: `{world}-comparison` PNG + SVG + caption; validated and drawn before writing."""
    comparison = comparison_rows(envelopes, matching_margin)
    chart = build_comparison_figure(comparison)
    paths = chart_paths(charts_dir, captions_dir, f"{comparison.world}-comparison")
    style.save_figure(chart.figure, paths.png, paths.svg, contains_fixture=any(r.is_fixture for r in comparison.rows))
    captions.write_caption(paths.caption, chart.caption)
    return paths
