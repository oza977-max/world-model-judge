"""wmj.reporting.horizon_plot — Chart 2, error against horizon (ADR-R2).

In plain words: one panel per region showing how wrong the model gets
the further ahead it predicts, with the world's own drift-apart curve
drawn as a black dashed reference. Only the gap above that reference
is the model's fault. The vertical axis is logarithmic so the early
steps — where the interesting gap lives — stay readable even when the
late steps blow up (TC-RP2-01).

This module draws exactly what it is handed — the judge's
`error_vs_horizon` block (judge spec §5) — and computes nothing
(reporting spec §4). Step 0 is dropped from the *plot* only: the
schema guarantees `median_error[0] == 0.0` (no rollout has happened
yet) and `log(0)` is undefined; the arrays themselves are untouched.

Full scope (P5-C03): the two simple baselines are drawn as grey lines
from their own verdicts' blocks, and each task's switch step (from
`climatology.per_task`) is a labelled vertical line on its region's panel:
past it, exact paths stop being graded and the judge checks the overall
pattern. The envelope does not say which models are baselines, so the
caller (`wmj run`, which reads the registry) passes their envelopes in; a
baseline drawn on a different world, region set or step range is refused.
The full three-sentence caption is used when at least one switch line is
drawn; a world with no switch step anywhere (predator-prey) gets the
two-sentence form, so the caption never points at marks that are not
there. Lines are labelled once, at the right of the last panel: the
panels repeat the same line styles (small multiples).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable  # noqa: UP035 - the reporting import gate (TC-NF6-10) allows typing, not collections.abc

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.text import Text
from matplotlib.transforms import Bbox, offset_copy

from wmj.errors import WmjError
from wmj.reporting import captions, style
from wmj.reporting.envelope import ChartPaths, chart_paths, read_envelope

REQUIRED_ENTRY_KEYS = frozenset({"region", "steps", "median_error", "divergence_reference"})
REFERENCE_LABEL = "world divergence\n(reference)"
SWITCH_COLOUR = "#555555"  # the switch lines: dark neutral grey, dotted
CLIMATOLOGY_KEYS = frozenset({"task", "region", "switch_step", "agreement_mean_abs_z", "agrees"})


class HorizonBlockError(WmjError):
    """Raised when an `error_vs_horizon` block cannot be drawn as specified.

    A malformed block is refused outright rather than drawn partially
    (reporting §7: partial chart sets would misrepresent the run).
    """


@dataclass(frozen=True)
class HorizonFigure:
    """The rendered figure plus handles a test can inspect before saving."""

    figure: Figure
    panels: tuple[Axes, ...]
    secondary_axes: tuple[Axes, ...]
    world_time_functions: tuple[Callable[[float], float], Callable[[float], float]]
    model_lines: tuple = ()  # one per panel
    reference_lines: tuple = ()  # one per panel
    baseline_lines: tuple = ()  # (panel index, label, line)
    switch_lines: tuple = ()  # (panel index, task, line)


def _validate(block: dict) -> tuple[float, list[dict]]:
    if not isinstance(block, dict) or set(block) != {"dt", "per_region"}:
        raise HorizonBlockError(
            "ADR-R2 Chart 2: error_vs_horizon must have exactly the keys "
            f"{{'dt', 'per_region'}} (judge §5), got {sorted(block) if isinstance(block, dict) else type(block)}"
        )
    dt = block["dt"]
    if not isinstance(dt, (int, float)) or not math.isfinite(dt) or dt <= 0.0:
        raise HorizonBlockError(f"ADR-R2 Chart 2: dt must be a positive finite number, got {dt!r}")
    entries = block["per_region"]
    if not isinstance(entries, list) or not entries:
        raise HorizonBlockError("ADR-R2 Chart 2: per_region must be a non-empty list (one panel per region)")
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != REQUIRED_ENTRY_KEYS:
            raise HorizonBlockError(
                f"ADR-R2 Chart 2: each per_region entry needs exactly {sorted(REQUIRED_ENTRY_KEYS)}, "
                f"got {sorted(entry) if isinstance(entry, dict) else type(entry)}"
            )
        try:
            steps = list(entry["steps"])
            arrays = {key: np.asarray(entry[key], dtype=float) for key in ("median_error", "divergence_reference")}
        except (TypeError, ValueError) as error:
            raise HorizonBlockError(f"ADR-R2 Chart 2: region {entry['region']!r} has steps or values that are not numbers: {error}") from error
        n = len(steps)
        if n < 2 or steps != list(range(n)):
            raise HorizonBlockError(
                f"ADR-R2 Chart 2: region {entry['region']!r} steps must be 0..H "
                f"(judge §5 shares the divergence artefact's step-zero origin), got {steps[:5]}..."
            )
        for key in ("median_error", "divergence_reference"):
            values = arrays[key]
            if values.shape != (n,):
                raise HorizonBlockError(
                    f"ADR-R2 Chart 2: region {entry['region']!r} {key} has {values.shape} "
                    f"entries, steps has {n}"
                )
            plotted = values[1:]
            if not np.all(np.isfinite(plotted)) or np.any(plotted <= 0.0):
                raise HorizonBlockError(
                    f"ADR-R2 Chart 2: region {entry['region']!r} {key} has a non-positive or "
                    "non-finite value at step >= 1; a log axis cannot draw it and a silent gap "
                    "would misrepresent the run (reporting §7)"
                )
    return float(dt), entries


def horizon_switches(climatology: dict) -> dict[str, list[tuple[str, int]]]:
    """The switch steps to mark, by region: `{region: [(task, step), ...]}` from `climatology.per_task`."""
    if not isinstance(climatology, dict) or set(climatology) != {"per_task"} or not isinstance(climatology["per_task"], list):
        raise HorizonBlockError(f"ADR-R2 Chart 2: climatology must be {{'per_task': [...]}} (judge §5), got {climatology!r:.80}")
    found: dict[str, list[tuple[str, int]]] = {}
    for entry in climatology["per_task"]:
        if not isinstance(entry, dict) or set(entry) != CLIMATOLOGY_KEYS:
            raise HorizonBlockError(f"ADR-R2 Chart 2: each climatology entry needs exactly {sorted(CLIMATOLOGY_KEYS)}")
        task, region, step = entry["task"], entry["region"], entry["switch_step"]
        if type(task) is not str or type(region) is not str:
            raise HorizonBlockError(f"ADR-R2 Chart 2: a climatology entry's task and region must be text, got {task!r}, {region!r}")
        if step is None:
            continue
        if type(step) is not int or step < 1:
            raise HorizonBlockError(f"ADR-R2 Chart 2: switch_step must be a whole number >= 1 or null, got {step!r}")
        found.setdefault(region, []).append((task, step))
    return found


def _check_baseline(dt: float, entries: list[dict], label: object, block: dict) -> list[dict]:
    if type(label) is not str or not label.strip():
        raise HorizonBlockError(f"ADR-R2 Chart 2: a baseline needs a name, got {label!r}")
    b_dt, b_entries = _validate(block)
    same = b_dt == dt and [e["region"] for e in b_entries] == [e["region"] for e in entries]
    same = same and all(list(b["steps"]) == list(e["steps"]) for b, e in zip(b_entries, entries))
    if not same:
        raise HorizonBlockError(
            f"ADR-R2 Chart 2: baseline {label!r} was not run on the same world, regions and steps as the model "
            "(its dt, region list or steps differ); drawing it on these axes would mislead"
        )
    return b_entries


SWITCH_FONT_SIZE = 7.0


def _extent(text: Text) -> Bbox:
    """Where a text sits on the figure, in pixels, including its box if it has one."""
    box = text.get_window_extent()
    patch = text.get_bbox_patch()
    return Bbox.union([box, patch.get_window_extent()]) if patch is not None else box


def _label_switches(ax: Axes, marks: list[tuple[str, float]], taken: list[Text]) -> list[Text]:
    """Write each switch line's label beside it, upright along the line, where it overprints nothing.

    Two tasks often switch at the same step, or a few steps apart (review of P5-C03: the labels printed on top
    of each other). Each label names its step, so it still reads as its own line's when it cannot sit right
    beside it. Each label tries, in turn: just right of its line, just left of it, then half a line of text
    further out on either side, and so on; then the same places again with the label broken over two
    lines, which is shorter (a long task name on a fixture's chart would otherwise reach the fixture sentence).
    A spot must lie inside the panel (not over the tick labels, the next panel or the panel's top), and must not
    touch a label already placed, the fixture sentence, or another task's switch line (review of P5-C03, pass 2:
    on the pendulum two tasks switch a few hundred steps apart, and one line ran through the other's label).
    If no spot is free the chart is refused rather than drawn unreadable (reporting §7)."""
    occupied = [_extent(t) for t in taken]
    inside = ax.get_window_extent()
    line_xs = {float(ax.transData.transform((step, 1.0))[0]) for _task, step in marks}
    nudge = 0.5 * SWITCH_FONT_SIZE  # points: half a rotated line of text, so labels pack close to their lines
    placed = []
    for task, step in marks:
        where_said = f"switches to climate here (step {step:g})"
        spots = [(words, shift, side) for words in (f" {task} {where_said}", f" {task}\n {where_said}")
                 for shift in range(16) for side in ("left", "right")]  # ha="left" puts the upright text right of the line
        own_x = float(ax.transData.transform((step, 1.0))[0])
        lines = [Bbox([[x - 1.0, inside.y0], [x + 1.0, inside.y1]]) for x in line_xs if x != own_x]
        for words, shift, side in spots:
            dx = (1.5 + shift * nudge) * (1 if side == "left" else -1)
            where = offset_copy(ax.get_xaxis_transform(), fig=ax.figure, x=dx, y=0.0, units="points")
            text = ax.text(step, 0.03, words, transform=where, rotation=90,
                           ha=side, va="bottom", fontsize=SWITCH_FONT_SIZE, color=SWITCH_COLOUR)
            box = _extent(text)
            within = inside.x0 <= box.x0 and box.x1 <= inside.x1 and box.y1 <= inside.y1
            if within and not any(box.overlaps(o) for o in occupied + lines):
                break
            text.remove()
        else:
            raise HorizonBlockError(
                f"ADR-R2 Chart 2: the switch label for {task!r} at step {step:g} cannot be placed without "
                "overprinting another label; refusing to draw an unreadable chart (reporting §7)"
            )
        occupied.append(box)
        placed.append(text)
    return placed


def build_horizon_figure(
    error_vs_horizon: dict,
    model_label: str,
    is_fixture: bool = False,
    *,
    baselines: tuple = (),
    switches: dict | None = None,
) -> HorizonFigure:
    """Draw Chart 2 from a judge-shaped block; returns the figure unsaved.

    `baselines` is a sequence of `(name, error_vs_horizon block)` drawn in grey; `switches` maps a region
    to its `(task, switch step)` pairs (see `horizon_switches`), each drawn as a labelled vertical line."""
    dt, entries = _validate(error_vs_horizon)
    baseline_entries = [(label, _check_baseline(dt, entries, label, block)) for label, block in baselines]
    switches = switches or {}
    regions = [e["region"] for e in entries]
    for region, pairs in switches.items():
        if region not in regions:
            raise HorizonBlockError(f"ADR-R2 Chart 2: a switch step is given for region {region!r}, which has no panel")
        horizon = len(entries[regions.index(region)]["steps"]) - 1
        for task, step in pairs:
            if not 1 <= step <= horizon:
                raise HorizonBlockError(f"ADR-R2 Chart 2: {task!r} switches at step {step}, outside 1..{horizon}")
    fig = style.new_figure()
    axes = fig.subplots(1, len(entries), sharey=True, squeeze=False)[0]

    def to_world_time(step: float) -> float:
        return step * dt

    def to_steps(world_time: float) -> float:
        return world_time / dt

    model_colour = style.line_colour(is_fixture)
    label = style.display_name(model_label, is_fixture)
    secondaries, model_lines, reference_lines, baseline_lines, switch_lines = [], [], [], [], []
    for index, (ax, entry) in enumerate(zip(axes, entries)):
        steps = np.asarray(entry["steps"])[1:]
        error = np.asarray(entry["median_error"], dtype=float)[1:]
        reference = np.asarray(entry["divergence_reference"], dtype=float)[1:]
        last = index == len(entries) - 1  # small multiples: the line styles are labelled once, on the last panel

        ax.set_yscale("log")
        for name, b_entries in baseline_entries:
            b_error = np.asarray(b_entries[index]["median_error"], dtype=float)[1:]
            (line,) = ax.plot(steps, b_error, color=style.BASELINE_COLOUR, linestyle="-", linewidth=1.0)
            baseline_lines.append((index, name, line))
        (model_line,) = ax.plot(steps, error, color=model_colour, linestyle="-", linewidth=1.6)
        (reference_line,) = ax.plot(steps, reference, color=style.DIVERGENCE_COLOUR, linestyle="--", linewidth=1.1)
        model_lines.append(model_line)
        reference_lines.append(reference_line)
        if last:
            ends = [(steps[-1], error[-1], label, model_colour), (steps[-1], reference[-1], REFERENCE_LABEL, style.DIVERGENCE_COLOUR)]
            ends += [(steps[-1], float(b_entries[index]["median_error"][-1]), name, style.BASELINE_COLOUR) for name, b_entries in baseline_entries]
            pending_labels = (ax, ends)
        for task, step in switches.get(entry["region"], []):
            line = ax.axvline(step, color=SWITCH_COLOUR, linestyle=":", linewidth=1.0)
            switch_lines.append((index, task, line))
        ax.set_title(str(entry["region"]))
        ax.set_xlabel("rollout step")
        ax.set_xlim(1, int(steps[-1]))
        secondary = ax.secondary_xaxis("top", functions=(to_world_time, to_steps))
        secondary.set_xlabel(f"world time = step × dt\n(dt = {dt:g})")  # two lines: three panels' labels must not meet
        secondaries.append(secondary)

    axes[0].set_ylabel("median normalised error (log scale)")
    fig.suptitle(f"Error against horizon — {label}", fontsize=10)
    fig.subplots_adjust(left=0.10, right=0.80, top=0.76, bottom=0.12, wspace=0.12)
    # headroom: the top fifth of each panel stays empty, so the fixture label never hides a line
    axes[0].autoscale_view()
    low, high = axes[0].get_ylim()
    axes[0].set_ylim(low, high * (high / low) ** 0.3)
    style.direct_labels(*pending_labels, fontsize=7.5)  # after the shared y-limits are final
    fixture_marks = [style.mark_fixture(ax) for ax in axes] if is_fixture else []  # after the layout: it wraps to the panel
    if switch_lines:
        fig.draw_without_rendering()  # places the fixture sentence's box, so its true extent can be read
    for index, ax in enumerate(axes):
        taken = [fixture_marks[index]] if fixture_marks else []
        _label_switches(ax, [(task, line.get_xdata()[0]) for i, task, line in switch_lines if i == index], taken)
    return HorizonFigure(
        figure=fig,
        panels=tuple(axes),
        secondary_axes=tuple(secondaries),
        world_time_functions=(to_world_time, to_steps),
        model_lines=tuple(model_lines),
        reference_lines=tuple(reference_lines),
        baseline_lines=tuple(baseline_lines),
        switch_lines=tuple(switch_lines),
    )


def render_horizon_chart(
    error_vs_horizon: dict,
    model_label: str,
    out_png: Path,
    out_svg: Path,
    is_fixture: bool = False,
) -> HorizonFigure:
    """Build Chart 2 and write its PNG and SVG (the only side effect)."""
    chart = build_horizon_figure(error_vs_horizon, model_label, is_fixture=is_fixture)
    style.save_figure(chart.figure, out_png, out_svg, contains_fixture=is_fixture)
    return chart


def render_horizon_charts(
    envelope: dict, baseline_envelopes: list[dict], charts_dir: Path, captions_dir: Path
) -> ChartPaths:
    """Chart 2 for one model at full scope: `{world}-{model}-horizon` PNG + SVG + caption.

    `baseline_envelopes` are the two baselines' envelopes for the same world. Everything is validated and
    drawn before anything is written."""
    try:
        env = read_envelope(envelope)
        baselines = [read_envelope(b) for b in baseline_envelopes]
    except (ValueError, TypeError) as error:
        raise HorizonBlockError(f"ADR-R2 Chart 2: {error}") from error
    # the chart of a baseline model itself: it is drawn once, as the model, not again in grey (review of P5-C03)
    baselines = [b for b in baselines if b.model_name != env.model_name]
    for b in baselines:
        if b.world != env.world:
            raise HorizonBlockError(f"ADR-R2 Chart 2: baseline {b.model_name!r} is for world {b.world!r}, not {env.world!r}")
        if b.is_fixture:
            raise HorizonBlockError(f"ADR-R2 Chart 2: {b.model_name!r} is a fixture, not a baseline")
    switches = horizon_switches(env.verdict.get("climatology"))
    chart = build_horizon_figure(
        env.verdict.get("error_vs_horizon"), env.model_name, env.is_fixture,
        baselines=tuple((b.model_name, b.verdict.get("error_vs_horizon")) for b in baselines), switches=switches,
    )
    caption = captions.CHART2_FULL if chart.switch_lines else captions.CHART2_SCOPED
    paths = chart_paths(charts_dir, captions_dir, f"{env.world}-{env.model_name}-horizon")
    style.save_figure(chart.figure, paths.png, paths.svg, contains_fixture=env.is_fixture)
    captions.write_caption(paths.caption, caption)
    return paths

