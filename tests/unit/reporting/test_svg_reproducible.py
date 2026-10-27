"""TC-RP7-02: the same chart rendered in two separate processes gives byte-identical SVG.

In plain words: a sceptic diffing two runs' SVG files should see no difference unless the results
differ. Matplotlib salts its element ids and stamps a date by default; reporting pins both (ADR-R1, §7).
Pinning works only if it holds in a fresh process, so this test renders every kind of chart twice, each
time in a new Python interpreter, and compares the bytes. (NF-1's formal guarantee is still the verdict
files and the manifest; this is the practical property the README states.)
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = """
import sys
from pathlib import Path
sys.path.insert(0, {root!r})
from tests.unit.reporting._verdict_blocks import FINGERPRINT, run_envelopes
from wmj.reporting.writer import write_run
write_run(run_envelopes(), frozenset({{"persistence", "linear"}}), Path({out!r}), 0.05, FINGERPRINT)
"""


def _render(out: Path, cwd: Path = ROOT, extra_env: dict | None = None, suffix: str = "*.svg") -> dict[str, bytes]:
    env = {"PYTHONDONTWRITEBYTECODE": "1", "PYTHONHASHSEED": "random", "PATH": "/usr/bin:/bin", **(extra_env or {})}
    subprocess.run([sys.executable, "-c", SCRIPT.format(root=str(ROOT), out=str(out))], check=True, cwd=cwd, env=env)
    return {p.name: p.read_bytes() for p in sorted((out / "charts").glob(suffix))}


def test_tc_rp7_02_every_chart_kind_is_byte_identical_across_two_processes(tmp_path):
    first, second = _render(tmp_path / "a"), _render(tmp_path / "b")
    kinds = {"comparison", "horizon", "calibration", "exceptions"}
    assert {kind for kind in kinds if any(f"-{kind}" in name for name in first)} == kinds
    assert first.keys() == second.keys()
    differing = [name for name in first if first[name] != second[name]]
    assert not differing, f"SVG differs between processes: {differing[:3]}"
    verdicts_a = {p.name: p.read_bytes() for p in (tmp_path / "a" / "verdicts").glob("*.json")}
    verdicts_b = {p.name: p.read_bytes() for p in (tmp_path / "b" / "verdicts").glob("*.json")}
    assert verdicts_a == verdicts_b and verdicts_a
    assert (tmp_path / "a" / "run-manifest.json").read_bytes() == (tmp_path / "b" / "run-manifest.json").read_bytes()
    assert (tmp_path / "a" / "results.html").read_bytes() == (tmp_path / "b" / "results.html").read_bytes()


HOSTILE_RC = "savefig.bbox: tight\nfont.family: serif\naxes.unicode_minus: False\nlines.linewidth: 5\nfigure.dpi: 72\n"


def test_a_settings_file_on_the_machine_changes_no_chart_byte(tmp_path):
    # code-review-004: matplotlib reads ./matplotlibrc, $MATPLOTLIBRC and the user's config folder at import;
    # the style must start from matplotlib's built-in defaults, so none of them can change a chart
    clean = _render(tmp_path / "clean", suffix="*.*")
    hostile_dir = tmp_path / "hostile"
    (hostile_dir / "config").mkdir(parents=True)
    (hostile_dir / "matplotlibrc").write_text(HOSTILE_RC)
    (hostile_dir / "config" / "matplotlibrc").write_text(HOSTILE_RC)
    hostile = _render(tmp_path / "with-settings", cwd=hostile_dir, suffix="*.*",
                      extra_env={"MPLCONFIGDIR": str(hostile_dir / "config"), "PYTHONPATH": str(ROOT / "src")})
    assert clean.keys() == hostile.keys() and clean
    assert [n for n in clean if clean[n] != hostile[n]] == []  # every SVG and PNG identical
