"""Outside-in acceptance test for `python -m wmj chart-preview`.

TDD-1: this is the FIRST deliverable written for P2-C05 (the second
early user-facing slice, implementation guide line 43) — every unit
test in this chunk falls out from this test's failures.

The chunk's acceptance criterion, in the guide's own words: "the
rendered chart and its scoped caption exist and are non-empty."
Scope disclosure carried over from the guide: the caption is reporting
ADR-R3's Chart-2 template WITHOUT the switch-step clause (that clause
narrates `climatology.per_task`, first computed at P4-C05).

`wmj chart-preview` is internal-only (reporting ADR-R5), so this test
exercises it at reduced scale via `--n-starts/--n-trials/--horizon`.
"""

from __future__ import annotations

from pathlib import Path

from wmj.harness.cli import main

FAST_ARGS = ["chart-preview", "--n-starts", "4", "--n-trials", "6", "--horizon", "40"]

# ADR-R3's Chart-2 template, minus its third (switch-step) sentence.
READING_RULE = "only the gap above that line is the model's fault"


def _outputs(root: Path) -> tuple[Path, Path, Path]:
    charts = root / "out" / "charts"
    return (
        charts / "lv-persistence-horizon.png",
        charts / "lv-persistence-horizon.svg",
        root / "out" / "captions" / "lv-persistence-horizon.txt",
    )


def test_chart_preview_writes_chart_and_scoped_caption(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)

    assert main(FAST_ARGS) == 0

    png, svg, caption_path = _outputs(tmp_path)
    for path in (png, svg, caption_path):
        assert path.exists(), f"missing acceptance output: {path}"
        assert path.stat().st_size > 0, f"empty acceptance output: {path}"

    assert png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert b"<svg" in svg.read_bytes()

    caption = caption_path.read_text(encoding="utf-8")
    assert READING_RULE in caption  # TC-RP2-01's reading-rule clause
    assert caption.count(". ") + 1 <= 3  # RP-5: at most three sentences
    assert "switch" not in caption  # the scoped-out clause is absent, not paraphrased

    # the CRPS/skill call path runs over real trajectories and reports on stdout
    out = capsys.readouterr().out
    assert "skill" in out and "persistence" in out and "linear" in out


def test_chart_preview_svg_is_byte_identical_across_two_runs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _, svg, _ = _outputs(tmp_path)

    assert main(FAST_ARGS) == 0
    first = svg.read_bytes()
    assert main(FAST_ARGS) == 0
    assert svg.read_bytes() == first  # ADR-R1/§7: hashsalt + Date=None pins


def test_chart_preview_svg_carries_no_date(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert main(FAST_ARGS) == 0
    _, svg, _ = _outputs(tmp_path)
    assert b"dc:date" not in svg.read_bytes()
