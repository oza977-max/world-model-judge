"""wmj.reporting.calibration_plot — Chart 3, calibration (reporting ADR-R2, RP-3).

In plain words: when a model said it was 50%, 80%, 90% or 95% sure, was it right that often? One panel
per task. Each region is one line, plotting the share of runs whose truth landed inside the model's
stated range (its whole answer, all quantities together: the judge's joint region) against the stated
level. The diagonal, labelled "perfectly honest", is where an honest model's line lies. Below it means
overconfident; above it means more cautious than it claimed. The training region is solid and every
other region dashed, each labelled at its end. Above each panel the strip states how wide the model's
90% ranges were in that region (sharpness, JU-5), because a model can sit on the diagonal by being
vague.

The error bars are display arithmetic, not a judgment (reporting §4). Each bar is plus or minus one
binomial standard error at the stated level, sqrt(L × (1 − L) / n): how far an honest model's coverage
moves by luck with this many runs. A point further from the diagonal than its bar is unlikely to be
luck. (The standard error of the observed share would shrink to nothing at 0% or 100%, which is exactly
where an overconfident model sits.) Every coverage figure comes from the verdict as given. A block that
does not add up is refused before anything is drawn or written.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.text import Text

from wmj.errors import WmjError
from wmj.reporting import captions, style
from wmj.reporting.envelope import ChartPaths, chart_paths, read_envelope

LEVELS = [0.5, 0.8, 0.9, 0.95]  # ADR-J2: exactly these four, in this order
CALIBRATION_KEYS = frozenset({"task", "region", "levels", "coverage", "n_trials", "per_dimension"})
SHARPNESS_KEYS = frozenset({"task", "region", "mean_width_90"})
TRAINING = "training"  # the one reserved region name every world uses for its training region
DIAGONAL_LABEL = "perfectly honest"


class CalibrationBlockError(WmjError):
    """A `calibration`/`sharpness` block (or its envelope) cannot be drawn as specified; nothing is drawn."""


@dataclass(frozen=True)
class CalibrationFigure:
    """One Chart-3 figure, unsaved, with the handles a test can inspect."""

    figure: Figure
    panels: tuple[Axes, ...]
    tasks: tuple[str, ...]
    region_lines: tuple  # (panel index, region, line)
    diagonals: tuple[Line2D, ...]
    headers: tuple[Text, ...]
    error_bars: tuple  # (panel index, region, [half-widths])
    caption: str


def _block(block: object, name: str) -> list:
    if not isinstance(block, dict) or set(block) != {"per_task"} or not isinstance(block["per_task"], list):
        raise CalibrationBlockError(f"ADR-R2 Chart 3: {name} must be {{'per_task': [...]}} (judge §5), got {block!r:.80}")
    return block["per_task"]


def _key(entry: object, keys: frozenset, name: str) -> tuple[str, str]:
    if not isinstance(entry, dict) or set(entry) != keys:
        raise CalibrationBlockError(f"ADR-R2 Chart 3: each {name} entry needs exactly {sorted(keys)}")
    task, region = entry["task"], entry["region"]
    if type(task) is not str or not task or type(region) is not str or not region:
        raise CalibrationBlockError(f"ADR-R2 Chart 3: a {name} entry's task and region must be non-empty text, got {task!r}, {region!r}")
    return task, region


def _plain_number(value: object) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def calibration_panels(calibration: object, sharpness: object) -> dict[str, list[tuple[dict, float]]]:
    """`{task: [(calibration entry, mean 90% width), ...]}` in the verdict's order, each entry paired with its sharpness."""
    widths: dict[tuple[str, str], float] = {}
    for entry in _block(sharpness, "sharpness"):
        key = _key(entry, SHARPNESS_KEYS, "sharpness")
        width = entry["mean_width_90"]
        if not _plain_number(width) or width < 0:
            raise CalibrationBlockError(f"ADR-R2 Chart 3: sharpness {key} mean_width_90 must be a finite number >= 0, got {width!r}")
        if key in widths:
            raise CalibrationBlockError(f"ADR-R2 Chart 3: two sharpness entries for {key}")
        widths[key] = width
    panels: dict[str, list[tuple[dict, float]]] = {}
    seen = set()
    for entry in _block(calibration, "calibration"):
        key = _key(entry, CALIBRATION_KEYS, "calibration")
        if key in seen:
            raise CalibrationBlockError(f"ADR-R2 Chart 3: two calibration entries for {key}")
        seen.add(key)
        if entry["levels"] != LEVELS or any(type(v) is not float for v in entry["levels"]):
            raise CalibrationBlockError(f"ADR-R2 Chart 3: {key} levels must be exactly {LEVELS} (ADR-J2), got {entry['levels']!r}")
        coverage = entry["coverage"]
        if not isinstance(coverage, list) or len(coverage) != len(LEVELS) or not all(_plain_number(c) and 0.0 <= c <= 1.0 for c in coverage):
            raise CalibrationBlockError(f"ADR-R2 Chart 3: {key} coverage must be four shares between 0 and 1, got {coverage!r}")
        if type(entry["n_trials"]) is not int or entry["n_trials"] < 1:
            raise CalibrationBlockError(f"ADR-R2 Chart 3: {key} n_trials must be a whole number >= 1, got {entry['n_trials']!r}")
        if key not in widths:
            raise CalibrationBlockError(f"ADR-R2 Chart 3: calibration {key} has no sharpness entry (JU-5: sharpness is shown beside it)")
        panels.setdefault(key[0], []).append((entry, widths.pop(key)))
    if widths:
        raise CalibrationBlockError(f"ADR-R2 Chart 3: sharpness entries with no calibration entry: {sorted(widths)}")
    if not panels:
        raise CalibrationBlockError("ADR-R2 Chart 3: no calibration entries to draw")
    return panels


def honest_luck(level: float, n_trials: int) -> float:
    """One binomial standard error at the stated level: how far an honest model's coverage moves by luck."""
    return math.sqrt(level * (1.0 - level) / n_trials)


def _sharpness_strip(pairs: list[tuple[dict, float]]) -> str:
    """One region per line, so a panel's strip never runs into the next panel."""
    return "sharpness (mean 90% width):\n" + "\n".join(f"  {entry['region']} {width:.3g}" for entry, width in pairs)


def build_calibration_figure(panels: dict[str, list[tuple[dict, float]]], model_name: str, is_fixture: bool) -> CalibrationFigure:
    """Draw Chart 3 (one panel per task) from validated pairs; returns it unsaved."""
    name = style.display_name(model_name, is_fixture)
    colour = style.line_colour(is_fixture)
    fig = style.new_figure()
    axes = fig.subplots(1, len(panels), sharey=True, squeeze=False)[0]
    fig.subplots_adjust(left=0.09, right=0.84, top=0.68, bottom=0.13, wspace=0.42)
    region_lines, diagonals, headers, error_bars = [], [], [], []
    for index, (ax, (task, pairs)) in enumerate(zip(axes, panels.items())):
        (diagonal,) = ax.plot([0.45, 1.0], [0.45, 1.0], color=style.DIVERGENCE_COLOUR, linestyle="-", linewidth=0.8)
        ax.annotate(DIAGONAL_LABEL, xy=(0.47, 0.47), xytext=(6, -2), textcoords="offset points", rotation=0,
                    ha="left", va="top", fontsize=7.5, color=style.DIVERGENCE_COLOUR)
        diagonals.append(diagonal)
        for entry, _width in pairs:
            n = entry["n_trials"]
            bars = [honest_luck(level, n) for level in LEVELS]
            container = ax.errorbar(
                LEVELS, entry["coverage"], yerr=bars, color=colour, linewidth=1.5, capsize=2.5, marker="o", markersize=3.5,
                linestyle="-" if entry["region"] == TRAINING else "--",
            )
            line = container.lines[0]
            region_lines.append((index, entry["region"], line))
            error_bars.append((index, entry["region"], bars))
        ax.set_xlim(0.45, 1.0)
        ax.set_ylim(0.0, 1.3)  # headroom above 100%, so the fixture label (top-left) never hides a line
        ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        style.direct_labels(ax, [(LEVELS[-1], e["coverage"][-1], e["region"], colour) for e, _w in pairs], fontsize=7.5)
        ax.set_xticks(LEVELS, [str(round(v * 100)) for v in LEVELS])
        ax.tick_params(axis="x", labelsize=8)
        ax.set_xlabel("stated confidence (%)")
        headers.append(ax.set_title(f"task {task}\n{_sharpness_strip(pairs)}", loc="left", fontsize=7.5))
        if is_fixture:
            style.mark_fixture(ax)
    axes[0].set_ylabel("share of runs whose truth landed inside\n(bars: ±1 standard error of honest luck)")
    fig.suptitle(f"Calibration — {name}", fontsize=10)
    caption = captions.chart3_caption(level=90)
    return CalibrationFigure(fig, tuple(axes), tuple(panels), tuple(region_lines), tuple(diagonals), tuple(headers),
                             tuple(error_bars), caption)


def render_calibration_chart(envelope: dict, charts_dir: Path, captions_dir: Path) -> ChartPaths:
    """Chart 3 for one model: `{world}-{model}-calibration` PNG + SVG + caption; validated and drawn before writing."""
    try:
        env = read_envelope(envelope)
    except ValueError as error:
        raise CalibrationBlockError(f"ADR-R2 Chart 3: {error}") from error
    panels = calibration_panels(env.verdict.get("calibration"), env.verdict.get("sharpness"))
    for task, pairs in panels.items():
        for entry, _width in pairs:
            for label, value in (("task", task), ("region", entry["region"])):
                if len(value) > 40:
                    raise CalibrationBlockError(f"ADR-R2 Chart 3: {label} {value!r} is longer than 40 characters")
    chart = build_calibration_figure(panels, env.model_name, env.is_fixture)
    paths = chart_paths(charts_dir, captions_dir, f"{env.world}-{env.model_name}-calibration")
    style.save_figure(chart.figure, paths.png, paths.svg, contains_fixture=env.is_fixture)
    captions.write_caption(paths.caption, chart.caption)
    return paths
