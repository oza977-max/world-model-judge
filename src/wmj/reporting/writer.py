"""wmj.reporting.writer — the one writer of the `out/` folder (reporting ADR-R5; cross-cutting ADR-002 rule 4).

In plain words: `wmj run` hands over every model's envelope for every world, and this module writes the whole
results folder:
- one verdict file per model and world;
- the run manifest;
- every chart and its caption;
- the results page.

Nothing else in the project writes there (design-review-008 C8). It works in three steps:
1. **Check everything first.** Each verdict goes back through the judge's own door, so a record the judge
   would refuse is refused here too (`VerdictSchemaError`, reporting §7). Each envelope must carry exactly
   its five parts. The run's four `meta` fields must be the same in every envelope.
2. **Write into a fresh staging folder** beside `out/`.
3. **Move it into place**, only once every file is written. If anything fails part-way, the staging folder,
   which this module itself created and filled, is removed. A failed run leaves no `out/` that could be
   mistaken for a result.

The verdict files and the manifest go through the project's one canonical serializer
(`wmj.harness.serialize`; owner decision 2026-10-26: that single module is on reporting's import
allowlist), so `wmj verify` can compare them byte for byte. The manifest adds the exact numpy and
matplotlib versions, read here at write time (reporting §6).
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path

from wmj.errors import WmjError
from wmj.harness.serialize import canonical_serialize
from wmj.judge.errors import VerdictIncompleteError
from wmj.judge.verdict import VERDICT_SCHEMA, Verdict
from wmj.reporting import page
from wmj.reporting.calibration_plot import render_calibration_chart
from wmj.reporting.comparison_table import render_comparison_chart
from wmj.reporting.envelope import ChartPaths, read_envelope
from wmj.reporting.exception_plot import render_exception_charts
from wmj.reporting.horizon_plot import render_horizon_charts

ENVELOPE_KEYS = frozenset({"model_ref", "model_name", "is_fixture", "verdict", "meta"})
META_KEYS = frozenset({"seed", "platform", "prereg_commit", "thresholds_file"})  # judge §5: exactly four
VERDICT_FIELDS = ("world", "skill", "error_vs_horizon", "calibration", "sharpness", "exceptions", "trials",
                  "climatology", "trust_horizons", "not_tested", "limitations")


class VerdictSchemaError(WmjError):
    """An envelope or its verdict does not match the judge's schema (reporting §7); nothing is written."""


class OutputFolderError(WmjError):
    """The output folder cannot be written as specified (it holds files, or a staging folder is in the way)."""


@dataclass(frozen=True)
class RunPaths:
    out_dir: Path
    verdicts: tuple[Path, ...]
    manifest: Path
    page: Path
    charts: tuple[ChartPaths, ...]


def _check_verdict(index: int, verdict: dict) -> None:
    fields = dict(verdict)
    schema = fields.pop("schema", None)
    if schema != VERDICT_SCHEMA:
        raise VerdictSchemaError(f"envelope {index}: verdict schema must be {VERDICT_SCHEMA!r}, got {schema!r}")
    if set(fields) != set(VERDICT_FIELDS):
        raise VerdictSchemaError(f"envelope {index}: verdict fields must be exactly {sorted(VERDICT_FIELDS)} (JU-9)")
    try:
        rebuilt = Verdict(**fields).to_dict()
    except VerdictIncompleteError as error:
        raise VerdictSchemaError(f"envelope {index}: the judge's own check refuses this verdict: {error}") from error
    if rebuilt != verdict:
        raise VerdictSchemaError(
            f"envelope {index}: the verdict is not in the judge's plain form (it would not be written back as given)"
        )


def _check_meta(index: int, meta: object) -> dict:
    if not isinstance(meta, dict) or set(meta) != META_KEYS:
        raise VerdictSchemaError(f"envelope {index}: meta must have exactly {sorted(META_KEYS)} (judge §5)")
    if type(meta["seed"]) is not int:
        raise VerdictSchemaError(f"envelope {index}: meta seed must be a whole number, got {meta['seed']!r}")
    for key in ("platform", "prereg_commit", "thresholds_file"):
        if type(meta[key]) is not str or not meta[key].strip():
            raise VerdictSchemaError(f"envelope {index}: meta {key} must be non-empty text, got {meta[key]!r}")
    return meta


def check_run(envelopes: object, baseline_names: object) -> dict[str, list[dict]]:
    """Every envelope checked, grouped by world in the order given; refuses anything that would mislead."""
    if not isinstance(envelopes, list) or not envelopes:
        raise VerdictSchemaError("write_run needs the envelope of every model on every world (reporting §5)")
    if not isinstance(baseline_names, (set, frozenset)) or not all(type(n) is str for n in baseline_names):
        raise VerdictSchemaError("baseline_names must be a set of model names")
    worlds: dict[str, list[dict]] = {}
    meta = None
    for index, env in enumerate(envelopes):
        if not isinstance(env, dict) or set(env) != ENVELOPE_KEYS:
            raise VerdictSchemaError(f"envelope {index} must have exactly {sorted(ENVELOPE_KEYS)} (judge §5)")
        if type(env["model_ref"]) is not str or not env["model_ref"].strip():
            raise VerdictSchemaError(f"envelope {index}: model_ref must be non-empty text")
        try:
            read = read_envelope(env)
        except ValueError as error:
            raise VerdictSchemaError(f"envelope {index}: {error}") from error
        _check_verdict(index, env["verdict"])
        this_meta = _check_meta(index, env["meta"])
        if meta is None:
            meta = this_meta
        elif this_meta != meta:
            raise VerdictSchemaError(f"envelope {index}: meta differs from the first envelope's; one run has one meta")
        worlds.setdefault(read.world, []).append(env)
    for world, group in worlds.items():
        names = [e["model_name"] for e in group]
        if len(set(names)) != len(names):
            raise VerdictSchemaError(f"world {world!r}: a model appears twice")
        missing = sorted(set(baseline_names) - set(names))
        if missing:
            raise VerdictSchemaError(f"world {world!r}: baseline(s) {missing} have no envelope (Chart 2 draws them)")
    return worlds


def manifest(envelopes: list[dict]) -> dict:
    """The run manifest: `meta`'s four fields plus the exact numpy and matplotlib versions (reporting §6)."""
    return {**envelopes[0]["meta"], "package_versions": {"numpy": version("numpy"), "matplotlib": version("matplotlib")}}


def _write_bytes(path: Path, data: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def _remove_staging(folder: Path) -> None:
    """Remove the staging folder this module created, deepest entries first."""
    for entry in sorted(folder.rglob("*"), key=lambda p: len(p.parts), reverse=True):
        entry.rmdir() if entry.is_dir() else entry.unlink()
    folder.rmdir()


def _shown(stage: Path, paths: ChartPaths) -> page.Shown:
    return page.Shown(paths.png.relative_to(stage).as_posix(), paths.caption.read_text(encoding="utf-8"))


def _write_all(stage: Path, worlds: dict[str, list[dict]], baseline_names, envelopes) -> tuple[list, list]:
    charts_dir, captions_dir = stage / "charts", stage / "captions"
    charts, verdict_files, sections = [], [], []
    for world, group in worlds.items():
        comparison = render_comparison_chart(group, charts_dir, captions_dir)
        charts.append(comparison)
        baselines = [e for e in group if e["model_name"] in baseline_names]
        models = []
        for env in group:
            name = env["model_name"]
            verdict_files.append(_write_bytes(stage / "verdicts" / f"{world}-{name}.json", canonical_serialize(env)))
            horizon = render_horizon_charts(env, baselines, charts_dir, captions_dir)
            calibration = render_calibration_chart(env, charts_dir, captions_dir)
            exception_paths = render_exception_charts(env, charts_dir, captions_dir)
            entries = env["verdict"]["exceptions"]["per_task"]
            exceptions = []
            for entry, paths in zip(entries, exception_paths, strict=True):
                stem = f"{world}-{name}-exceptions-{entry['task']}-{entry['region']}-h{entry['horizon_step']}"
                if paths.png.stem != stem:  # the page labels each chart by its verdict entry: they must agree
                    raise OutputFolderError(f"Chart 1 {paths.png.name} does not match its verdict entry {stem}")
                exceptions.append((entry["task"], entry["region"], entry["horizon_step"], _shown(stage, paths)))
            charts += [horizon, calibration, *exception_paths]
            models.append(page.ModelSection(name, env["is_fixture"], _shown(stage, horizon), _shown(stage, calibration),
                                            tuple(exceptions)))
        sections.append(page.WorldSection(world, _shown(stage, comparison), tuple(models)))
    verdict = envelopes[0]["verdict"]
    html = page.render_page(tuple(verdict["limitations"]), tuple(verdict["not_tested"]), tuple(sections))
    _write_bytes(stage / "results.html", html.encode("utf-8"))
    _write_bytes(stage / "run-manifest.json", canonical_serialize(manifest(envelopes)))
    return charts, verdict_files


def write_run(envelopes: list[dict], baseline_names: frozenset[str], out_dir: Path) -> RunPaths:
    """Check every envelope, write the whole `out/` folder into a staging folder, then move it into place."""
    worlds = check_run(envelopes, baseline_names)
    out_dir = Path(out_dir)
    if out_dir.exists() and (not out_dir.is_dir() or any(out_dir.iterdir())):
        raise OutputFolderError(f"{out_dir} already holds files; clear it first (a run never mixes with an older one)")
    stage = out_dir.with_name(out_dir.name + ".partial")
    if stage.exists():
        raise OutputFolderError(f"{stage} is in the way (a previous run stopped part-way); remove it first")
    stage.mkdir(parents=True)
    try:
        charts, verdict_files = _write_all(stage, worlds, baseline_names, envelopes)
    except BaseException:
        _remove_staging(stage)
        raise
    stage.replace(out_dir)  # one rename: `out/` appears complete or not at all
    moved = [Path(out_dir) / p.relative_to(stage) for p in verdict_files]
    charts = [ChartPaths(*(Path(out_dir) / p.relative_to(stage) for p in (c.png, c.svg, c.caption))) for c in charts]
    return RunPaths(out_dir, tuple(moved), out_dir / "run-manifest.json", out_dir / "results.html", tuple(charts))
