"""wmj.reporting.exception_plot — Chart 1, the backtesting exception plot (reporting ADR-R2, RP-1).

In plain words: the project's primary chart. For one model, one region, one task and one of the two
judged steps, every test run is a dot: how far its outcome landed from the model's guess, measured
in the model's own stated error sizes and combined over everything the model predicts. A line marks
the edge of the range the model called 90% likely; the runs the verdict counts as misses are
circled in red. A strip above the chart states the expected number of misses beside the actual
number and names the colour band; a small scale on the right shows where the count sits among
green, amber and red (TC-RP1-01). A deliberately broken test model's chart says so in words on the
image (ADR-R4, TC-RP8-01).

This module draws what the verdict's `exceptions` and `trials` blocks say (judge §5) and decides
nothing (reporting §4): the misses are circled exactly where `is_exception` is true, never
re-derived from the dot and the line. The only arithmetic is for display: the y-axis switches to
a logarithmic scale above the line when a miss lies more than eight line-heights out (so a
million-sized miss does not flatten every other dot onto the axis), and the inset scale's extent. A
block that does not add up (counts that disagree, arrays of different lengths, a distance that is
negative or not a number) is refused before anything is drawn or written (reporting §7: a partial
chart set would misrepresent the run).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import PathCollection
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.text import Text

from wmj.errors import WmjError
from wmj.reporting import captions, style
from wmj.reporting.envelope import ChartPaths, chart_paths, file_part, read_envelope

EXCEPTION_KEYS = frozenset(
    {"task", "region", "horizon_step", "n_trials", "expected", "observed", "band", "low_side_sharpness_flag", "bands"}
)
TRIAL_KEYS = frozenset({"task", "region", "horizon_step", "distance_unit", "outcome_distance", "band_lo", "band_hi", "is_exception"})
LOG_ABOVE = 8.0  # a miss more than this many line-heights out switches the y-axis to log above the line
_FLOAT_MAX = float(np.finfo(np.float64).max)
_LOG10_MAX = math.log10(_FLOAT_MAX)
HEADROOM = 1.3  # a linear axis reaches 1.3 x the largest value ...
LOG_HEADROOM = 1.45  # ... a log axis 1.45 x the decades above the line (its linear part below the line takes room)


class ExceptionBlockError(WmjError):
    """An `exceptions`/`trials` block (or its envelope) cannot be drawn as specified; nothing is drawn."""


@dataclass(frozen=True)
class ExceptionFigure:
    """One Chart-1 figure, unsaved, with the handles a test can inspect."""

    figure: Figure
    panel: Axes
    inset: Axes
    dots: PathCollection
    misses: PathCollection
    line: Line2D
    marker: Line2D
    header: Text
    caption: str


def _is_whole(value: object, low: int) -> bool:
    return type(value) is int and value >= low


def _distances(entry: dict, field: str, n: int, where: str) -> list[float]:
    values = entry[field]
    if not isinstance(values, list) or len(values) != n:
        raise ExceptionBlockError(f"{where}: {field} has length {len(values) if isinstance(values, list) else '?'}, n_trials is {n}")
    for v in values:
        if type(v) not in (int, float) or (type(v) is int and abs(v) >= 2**63) or not math.isfinite(v) or v < 0:
            raise ExceptionBlockError(f"{where}: {field} holds {v!r}; a distance is a finite number >= 0")
    return values


def _check_bands(bands: object, where: str) -> None:
    """The judge's record: green [g_lo, g_hi]; amber [a_lo, g_lo - 1] and [g_hi + 1, a_hi], touching green; red the rest."""
    ok = isinstance(bands, dict) and set(bands) == {"green", "amber", "red"}
    ok = ok and type(bands["red"]) is str and bool(bands["red"].strip())
    if ok:
        green, amber = bands["green"], bands["amber"]
        pairs = [green, *amber] if isinstance(amber, list) and len(amber) == 2 else []
        ok = isinstance(green, list) and len(pairs) == 3 and all(
            isinstance(p, list) and len(p) == 2 and all(_is_whole(v, 0) for v in p) for p in pairs
        )
        ok = ok and amber[0][0] <= amber[0][1] + 1 == green[0] <= green[1] == amber[1][0] - 1 <= amber[1][1]
    if not ok:
        raise ExceptionBlockError(
            f"{where}: bands must be green [lo, hi] with amber ranges touching it on both sides, as the judge records "
            f"them (judge §5), got {bands!r}"
        )


def _band_of(observed: int, bands: dict) -> str:
    green, (amber_low, amber_high) = bands["green"], bands["amber"]
    if green[0] <= observed <= green[1]:
        return "green"
    return "amber" if amber_low[0] <= observed <= amber_high[1] else "red"


def _check_keys(entry: object, keys: frozenset, name: str) -> None:
    if not isinstance(entry, dict) or set(entry) != keys:
        got = sorted(map(str, entry)) if isinstance(entry, dict) else type(entry).__name__
        raise ExceptionBlockError(f"each {name} entry needs exactly the keys {sorted(keys)}, got {got}")
    if type(entry["task"]) is not str or type(entry["region"]) is not str or not _is_whole(entry["horizon_step"], 1):
        raise ExceptionBlockError(
            f"{name} entry: task and region must be text and horizon_step a whole number >= 1, "
            f"got {entry['task']!r}, {entry['region']!r}, {entry['horizon_step']!r}"
        )


def _check_pair(entry: dict, trial: dict) -> None:
    _check_keys(entry, EXCEPTION_KEYS, "exceptions")
    _check_keys(trial, TRIAL_KEYS, "trials")
    where = f"exceptions/trials {entry['task']!r}/{entry['region']!r}/step {entry['horizon_step']!r}"
    if (trial["task"], trial["region"], trial["horizon_step"]) != (entry["task"], entry["region"], entry["horizon_step"]):
        raise ExceptionBlockError(f"{where}: the trials entry belongs to another task, region or step")
    if not _is_whole(entry["horizon_step"], 1):
        raise ExceptionBlockError(f"{where}: horizon_step must be a whole number >= 1")
    n = entry["n_trials"]
    if not _is_whole(n, 1):
        raise ExceptionBlockError(f"{where}: n_trials must be a whole number >= 1, got {n!r}")
    if entry["band"] not in captions.BANDS:
        raise ExceptionBlockError(f"{where}: band must be one of {captions.BANDS}, got {entry['band']!r}")
    if type(entry["low_side_sharpness_flag"]) is not bool:
        raise ExceptionBlockError(f"{where}: low_side_sharpness_flag must be True or False")
    _check_bands(entry["bands"], where)
    _distances(trial, "outcome_distance", n, where)
    _distances(trial, "band_hi", n, where)
    if any(v != 0 for v in _distances(trial, "band_lo", n, where)):
        raise ExceptionBlockError(f"{where}: band_lo is always 0 (judge §5)")
    flags = trial["is_exception"]
    if not isinstance(flags, list) or len(flags) != n:
        raise ExceptionBlockError(f"{where}: is_exception has the wrong length for n_trials = {n}")
    if any(type(f) is not bool for f in flags):
        raise ExceptionBlockError(f"{where}: is_exception must hold only True or False")
    distances, lines = trial["outcome_distance"], trial["band_hi"]
    if any(flag != (d > line) for flag, d, line in zip(flags, distances, lines, strict=True)):
        # code-review-004: the caption says "dots above it are the misses, circled in red", so a flag must
        # agree with its dot; the judge decides both the same way (miss iff distance > line)
        raise ExceptionBlockError(f"{where}: an is_exception flag disagrees with its dot and the line; the chart "
                                  "would circle a dot inside the range, or miss one outside it")
    unit = trial["distance_unit"]
    if type(unit) is not str or unit.split(":")[0] != "stated-size":
        raise ExceptionBlockError(f"{where}: distance_unit must be in the model's stated sizes, as the axis says, "
                                  f"got {unit!r}")
    if not _is_whole(entry["observed"], 0) or entry["observed"] != sum(flags):
        raise ExceptionBlockError(f"{where}: observed = {entry['observed']!r} but is_exception flags {sum(flags)} misses")
    if _band_of(entry["observed"], entry["bands"]) != entry["band"]:
        raise ExceptionBlockError(
            f"{where}: band {entry['band']!r} disagrees with the bands for {entry['observed']} misses "
            f"({_band_of(entry['observed'], entry['bands'])!r}); the chart would contradict itself"
        )
    try:
        captions.chart1_caption(expected=entry["expected"], n=n, observed=entry["observed"], band=entry["band"])
    except captions.CaptionSlotError as error:
        raise ExceptionBlockError(f"{where}: {error}") from error


def chart1_panels(exceptions: dict, trials: dict) -> list[tuple[dict, dict]]:
    """Pair each `exceptions.per_task` entry with its one `trials.per_task` entry, validating both."""
    for name, block in (("exceptions", exceptions), ("trials", trials)):
        if not isinstance(block, dict) or set(block) != {"per_task"} or not isinstance(block["per_task"], list):
            raise ExceptionBlockError(f"{name} must be {{'per_task': [...]}} (judge §5), got {block!r:.80}")
    by_key: dict[tuple, list[dict]] = {}
    for trial in trials["per_task"]:
        _check_keys(trial, TRIAL_KEYS, "trials")
        by_key.setdefault((trial["task"], trial["region"], trial["horizon_step"]), []).append(trial)
    pairs = []
    for entry in exceptions["per_task"]:
        _check_keys(entry, EXCEPTION_KEYS, "exceptions")
        key = (entry["task"], entry["region"], entry["horizon_step"])
        found = by_key.pop(key, [])
        if len(found) != 1:
            raise ExceptionBlockError(
                f"exceptions entry {key} has no trials entry" if not found else f"exceptions entry {key} has {len(found)} trials entries"
            )
        _check_pair(entry, found[0])
        pairs.append((entry, found[0]))
    if by_key:
        raise ExceptionBlockError(f"trials entries with no exceptions entry: {sorted(by_key)}")
    return pairs


def _header(entry: dict) -> str:
    """The strip above the panel; the §7 hedging note, when flagged, takes a second line so it is never cut off."""
    text = (
        f"region: {entry['region']} · observed exceptions: {entry['observed']} · "
        f"expected: {captions.format_expected(entry['expected'])} of {entry['n_trials']} · band: {entry['band'].upper()}"
    )
    if entry["low_side_sharpness_flag"]:
        text += f"\n{entry['observed']} misses — but see sharpness: possible hedging"
    return text


def _inset(ax: Axes, entry: dict) -> tuple[Axes, Line2D]:
    green, (amber_low, amber_high) = entry["bands"]["green"], entry["bands"]["amber"]
    observed = entry["observed"]
    top = max(amber_high[1], observed, entry["expected"]) * 1.15 + 1.0
    inset = ax.inset_axes([1.04, 0.0, 0.06, 1.0])
    spans = [
        (0.0, amber_low[0] - 0.5, "red"),
        (amber_low[0] - 0.5, green[0] - 0.5, "amber"),
        (green[0] - 0.5, green[1] + 0.5, "green"),
        (green[1] + 0.5, amber_high[1] + 0.5, "amber"),
        (amber_high[1] + 0.5, top, "red"),
    ]
    for low, high, band in spans:
        if high > max(low, 0.0):
            inset.axhspan(max(low, 0.0), high, color=style.BAND_COLOURS[band], linewidth=0)
    inset.axhline(entry["expected"], color="black", linewidth=0.8, linestyle=":")
    (marker,) = inset.plot([0.5], [observed], marker="<", markersize=7, color="black", linestyle="none", clip_on=False)
    inset.annotate(str(observed), xy=(0.5, observed), xytext=(-12, 0), textcoords="offset points", ha="right", va="center", fontsize=7.5)
    inset.set_xlim(0.0, 1.0)
    inset.set_ylim(-0.02 * top, top)  # a small margin below 0, so a zero count's marker is drawn whole
    inset.set_xticks([])
    inset.yaxis.tick_right()
    inset.tick_params(labelsize=7)
    inset.grid(False)
    inset.set_title("misses\n(dotted: expected)", fontsize=6.5)
    return inset, marker


def build_exception_figure(entry: dict, trial: dict, model_name: str, is_fixture: bool) -> ExceptionFigure:
    """Draw one Chart-1 figure from a validated (exceptions entry, trials entry) pair; returns it unsaved."""
    _check_pair(entry, trial)
    name = style.display_name(model_name, is_fixture)
    n = entry["n_trials"]
    x = list(range(n))
    y, line_y = trial["outcome_distance"], trial["band_hi"]
    fig = style.new_figure()
    ax = fig.subplots()
    fig.subplots_adjust(left=0.09, right=0.85, top=0.80, bottom=0.12)
    ax.fill_between(x, trial["band_lo"], line_y, color=style.REGION_FILL_COLOUR, linewidth=0, zorder=1)
    (line,) = ax.plot(x, line_y, color=style.REGION_EDGE_COLOUR, linewidth=1.0, zorder=2)
    dots = ax.scatter(x, y, s=9, color=style.line_colour(is_fixture), zorder=3, linewidths=0, clip_on=False)  # a 0 is drawn whole
    flagged = [i for i, f in enumerate(trial["is_exception"]) if f]
    misses = ax.scatter(
        flagged, [y[i] for i in flagged], s=55, facecolors="none", edgecolors=style.EXCEPTION_COLOUR, linewidths=1.1, zorder=4,
        clip_on=False,  # a miss at the first or last run is circled whole
    )
    misses.set_visible(bool(flagged))  # an empty unclipped scatter would still stamp one circle at the figure's corner
    largest, edge = max(y), max(line_y)
    ylabel = "miss, in the model's own stated error sizes"
    if edge > 0 and largest > LOG_ABOVE * edge:
        decades = math.log10(largest) - math.log10(edge)
        # the linear part (0 to the line) keeps about a tenth of the axis whatever the decades above it,
        # so the "0" and the line's own tick never overprint into a false "100" (review pass 2)
        ax.set_yscale("symlog", linthresh=edge, linscale=max(1.0, 0.12 * decades))
        ylabel += " (log scale above the line)"
    # headroom: the top fifth of the panel stays empty, so the fixture label (top-left) never hides a dot
    if ax.get_yscale() == "linear":
        ax.set_ylim(0.0, min(max(largest, edge, 1e-12) * HEADROOM, _FLOAT_MAX))
    else:  # in log space, capped at the largest float: a miss of 1e300 stated sizes is drawn, not crashed on
        exponent = math.log10(edge) + LOG_HEADROOM * (math.log10(largest) - math.log10(edge))
        ax.set_ylim(0.0, 10.0**exponent if exponent < _LOG10_MAX else _FLOAT_MAX)
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_xlabel("test run (in the order the runs were drawn)")
    ax.set_ylabel(ylabel)
    header = ax.set_title(_header(entry), loc="left", fontsize=8)
    inset, marker = _inset(ax, entry)
    fig.suptitle(f"Backtesting exceptions — {name} · task {entry['task']} · step {entry['horizon_step']}", fontsize=10)
    if is_fixture:
        style.mark_fixture(ax)
    caption = captions.chart1_caption(expected=entry["expected"], n=n, observed=entry["observed"], band=entry["band"])
    return ExceptionFigure(fig, ax, inset, dots, misses, line, marker, header, caption)


def _file_part(label: str, value: object) -> str:
    try:
        return file_part(label, value)  # 1 to 40 safe characters: also keeps the header strip clear of the scale
    except ValueError as error:
        raise ExceptionBlockError(str(error)) from error


def render_exception_charts(envelope: dict, charts_dir: Path, captions_dir: Path) -> list[ChartPaths]:
    """Every Chart-1 figure for one envelope: PNG + SVG under `charts_dir`, caption under `captions_dir`.

    Everything is validated and drawn before the first file is written, so a bad block writes nothing."""
    try:
        env = read_envelope(envelope)
    except ValueError as error:
        raise ExceptionBlockError(f"envelope: {error}") from error
    verdict, world, model, is_fixture = env.verdict, env.world, env.model_name, env.is_fixture
    for entry in verdict.get("exceptions", {}).get("per_task", []) if isinstance(verdict.get("exceptions"), dict) else []:
        if isinstance(entry, dict):
            _file_part("region", entry.get("region"))  # long names are refused before anything is drawn
    pairs = chart1_panels(verdict.get("exceptions"), verdict.get("trials"))
    built = []
    for entry, trial in pairs:
        stem = "-".join(
            [world, model, "exceptions", _file_part("task", entry["task"]), _file_part("region", entry["region"]), f"h{entry['horizon_step']}"]
        )
        built.append((stem, build_exception_figure(entry, trial, model, is_fixture)))
    written = []
    for stem, chart in built:
        paths = chart_paths(charts_dir, captions_dir, stem)
        style.save_figure(chart.figure, paths.png, paths.svg, contains_fixture=is_fixture)
        captions.write_caption(paths.caption, chart.caption)
        written.append(paths)
    return written
