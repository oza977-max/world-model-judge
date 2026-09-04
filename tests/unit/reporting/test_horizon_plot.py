"""Tests for wmj.reporting.horizon_plot — Chart 2 at P2-C05 scope.

Reporting ADR-R2 Chart 2: one panel per region from
`error_vs_horizon.per_region`; X starts at step 1 (index 0 is zero by
construction and log(0) is undefined); world-time secondary axis =
step x `dt`; Y log scale (TC-RP2-01's legibility clause); model curve
solid dark blue, world divergence reference black dashed, direct
labels. §4: reporting draws what it is handed and computes nothing.
"""

from __future__ import annotations

import numpy as np
import pytest

from wmj.reporting import style
from wmj.reporting.horizon_plot import (
    HorizonBlockError,
    build_horizon_figure,
    render_horizon_chart,
)


def _block(n_regions: int = 2, horizon: int = 20) -> dict:
    steps = list(range(horizon + 1))
    per_region = []
    for i in range(n_regions):
        growth = np.linspace(0.0, 1.0, horizon + 1) * (i + 1)
        per_region.append(
            {
                "region": ["training", "out-high-amplitude"][i],
                "steps": steps,
                "median_error": (growth * 0.1).tolist(),
                "divergence_reference": (growth * 0.01).tolist(),
            }
        )
    return {"dt": 0.02, "per_region": per_region}


def test_one_panel_per_region_in_declared_order():
    chart = build_horizon_figure(_block(2), model_label="persistence")
    assert len(chart.panels) == 2
    assert [ax.get_title() for ax in chart.panels] == ["training", "out-high-amplitude"]


def test_y_axis_is_log_scale_for_early_gap_legibility():
    chart = build_horizon_figure(_block(1), model_label="persistence")
    assert chart.panels[0].get_yscale() == "log"


def test_x_starts_at_step_one_not_zero_and_arrays_are_untouched():
    block = _block(1)
    before = list(block["per_region"][0]["median_error"])
    chart = build_horizon_figure(block, model_label="persistence")
    for line in chart.panels[0].get_lines():
        assert line.get_xdata()[0] == 1
    assert block["per_region"][0]["median_error"] == before


def test_colour_semantics_model_blue_solid_reference_black_dashed():
    chart = build_horizon_figure(_block(1), model_label="persistence")
    lines = chart.panels[0].get_lines()
    styles = {(line.get_color(), line.get_linestyle()) for line in lines}
    assert (style.MODEL_COLOUR, "-") in styles
    assert (style.DIVERGENCE_COLOUR, "--") in styles


def test_direct_labels_name_the_model_and_the_reference():
    chart = build_horizon_figure(_block(1), model_label="persistence")
    texts = {t.get_text() for t in chart.panels[0].texts}
    assert "persistence" in texts
    assert any("divergence" in t for t in texts)


def test_world_time_secondary_axis_is_step_times_dt():
    chart = build_horizon_figure(_block(1, horizon=50), model_label="persistence")
    secondary = chart.secondary_axes[0]
    assert "world time" in secondary.get_xlabel()
    forward, _ = chart.world_time_functions
    assert forward(50) == pytest.approx(1.0)  # 50 steps x 0.02


def test_fixture_flag_calls_mark_fixture_once_per_panel():
    chart = build_horizon_figure(_block(2), model_label="fx-broken", is_fixture=True)
    for ax in chart.panels:
        assert sum(t.get_text() == style.FIXTURE_LABEL for t in ax.texts) == 1


def test_non_fixture_carries_no_fixture_label():
    chart = build_horizon_figure(_block(1), model_label="persistence")
    assert all(t.get_text() != style.FIXTURE_LABEL for t in chart.panels[0].texts)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda b: b.pop("dt"),
        lambda b: b["per_region"][0].pop("divergence_reference"),
        lambda b: b["per_region"][0]["steps"].__setitem__(0, 1),
        lambda b: b["per_region"][0]["median_error"].pop(),
        lambda b: b["per_region"][0]["median_error"].__setitem__(5, 0.0),
        lambda b: b.__setitem__("per_region", []),
    ],
    ids=["no-dt", "no-reference", "steps-not-from-zero", "length-mismatch", "zero-on-log", "no-regions"],
)
def test_malformed_block_fails_loudly(mutate):
    block = _block(1)
    mutate(block)
    with pytest.raises(HorizonBlockError):
        build_horizon_figure(block, model_label="persistence")


def test_render_horizon_chart_writes_both_files(tmp_path):
    png, svg = tmp_path / "c.png", tmp_path / "c.svg"
    render_horizon_chart(_block(2), "persistence", png, svg)
    assert png.stat().st_size > 0 and svg.stat().st_size > 0
    body = svg.read_bytes()
    assert b"dc:date" not in body
    assert b"training" in body  # svg.fonttype none keeps labels as text
