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

P2-C05 scope: the per-task switch lines (from `climatology.per_task`)
arrive at P5-C03 together with the full caption.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from wmj.errors import WmjError
from wmj.reporting import style

REQUIRED_ENTRY_KEYS = frozenset({"region", "steps", "median_error", "divergence_reference"})
REFERENCE_LABEL = "world divergence (reference)"


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
        steps = list(entry["steps"])
        n = len(steps)
        if n < 2 or steps != list(range(n)):
            raise HorizonBlockError(
                f"ADR-R2 Chart 2: region {entry['region']!r} steps must be 0..H "
                f"(judge §5 shares the divergence artefact's step-zero origin), got {steps[:5]}..."
            )
        for key in ("median_error", "divergence_reference"):
            values = np.asarray(entry[key], dtype=float)
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


def build_horizon_figure(
    error_vs_horizon: dict, model_label: str, is_fixture: bool = False
) -> HorizonFigure:
    """Draw Chart 2 from a judge-shaped block; returns the figure unsaved."""
    dt, entries = _validate(error_vs_horizon)
    fig = style.new_figure()
    axes = fig.subplots(1, len(entries), sharey=True, squeeze=False)[0]

    def to_world_time(step: float) -> float:
        return step * dt

    def to_steps(world_time: float) -> float:
        return world_time / dt

    model_colour = style.FIXTURE_COLOUR if is_fixture else style.MODEL_COLOUR
    label = f"FIXTURE: {model_label}" if is_fixture else model_label
    secondaries = []
    for ax, entry in zip(axes, entries):
        steps = np.asarray(entry["steps"])[1:]
        error = np.asarray(entry["median_error"], dtype=float)[1:]
        reference = np.asarray(entry["divergence_reference"], dtype=float)[1:]

        ax.set_yscale("log")
        ax.plot(steps, error, color=model_colour, linestyle="-", linewidth=1.6)
        ax.plot(steps, reference, color=style.DIVERGENCE_COLOUR, linestyle="--", linewidth=1.1)
        ax.annotate(
            label, xy=(steps[-1], error[-1]), xytext=(4, 0), textcoords="offset points",
            va="center", ha="left", color=model_colour, fontsize=8,
        )
        ax.annotate(
            REFERENCE_LABEL, xy=(steps[-1], reference[-1]), xytext=(4, 0),
            textcoords="offset points", va="center", ha="left",
            color=style.DIVERGENCE_COLOUR, fontsize=8,
        )
        ax.set_title(str(entry["region"]))
        ax.set_xlabel("rollout step")
        ax.set_xlim(1, int(steps[-1]))
        secondary = ax.secondary_xaxis("top", functions=(to_world_time, to_steps))
        secondary.set_xlabel(f"world time (step × dt, dt = {dt:g})")
        secondaries.append(secondary)
        if is_fixture:
            style.mark_fixture(ax)

    axes[0].set_ylabel("median normalised error (log scale)")
    fig.suptitle(f"Error against horizon — {label}", fontsize=10)
    fig.subplots_adjust(left=0.08, right=0.86, top=0.80, bottom=0.12, wspace=0.12)
    return HorizonFigure(
        figure=fig,
        panels=tuple(axes),
        secondary_axes=tuple(secondaries),
        world_time_functions=(to_world_time, to_steps),
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
    style.save_figure(chart.figure, out_png, out_svg)
    return chart
