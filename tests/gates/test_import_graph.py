"""The judge import-allowlist gate and its family (cross-cutting ADR-003).

TC-NF6-01: judge package imports only {numpy, math, dataclasses, typing}.
TC-NF6-02: banned identifiers anywhere in the judge package (fast lint,
not a completeness proof — see module docstrings in cross-cutting.md).
TC-NF6-03: numpy.random banned wholesale in the judge.
TC-NF6-04: the evasion-fixture regression corpus — every fixture must
be flagged by the gate.
TC-NF6-06: the clean-pass guard — the real judge source must NOT be
flagged (checks 02/03 are deliberately over-broad).
TC-NF6-07/08/09: models package never imports wmj.worlds / wmj.harness
/ wmj.judge / wmj.reporting, in any direction, via the joined
module+name check design-review-005/008 pinned.
"""

from __future__ import annotations

import ast
from pathlib import Path

from tests.gates._ast_utils import (
    full_import_names,
    joined_importfrom_names,
    scan_banned_identifiers,
    scan_numpy_random_usage,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src" / "wmj"
FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"

JUDGE_ALLOWLIST = {"numpy", "math", "dataclasses", "typing"}

# `from __future__ import annotations` is a compile-time language
# directive (Development Conventions mandate it on every file), not an
# import reaching ambient capability — cross-cutting ADR-003's pinned
# {numpy, math, dataclasses, typing} allowlist doesn't name it, but a
# literal reading would refuse the judge's own required convention on
# every single file, which is plainly not the rule's intent.
ALWAYS_ALLOWED = {"__future__"}

# The judge package importing FROM ITSELF (wmj.judge.* importing another
# wmj.judge.* module, e.g. skill.py importing _normal.py) is not the
# "never imports the other wmj packages" ADR-003 forbids — that rule is
# about worlds/models/harness/reporting, not intra-package structure.
JUDGE_OWN_PACKAGE = "wmj.judge"


def _is_allowed_judge_import(name: str) -> bool:
    if name == JUDGE_OWN_PACKAGE or name.startswith(JUDGE_OWN_PACKAGE + "."):
        return True
    for allowed in JUDGE_ALLOWLIST | ALWAYS_ALLOWED:
        if name == allowed or name.startswith(allowed + "."):
            return True
    return False

# The pinned identifier list (cross-cutting ADR-003, TC-NF6-02) —
# deliberately over-broad; TC-NF6-06 proves it doesn't cry wolf on the
# real judge source.
BANNED_IDENTIFIERS = {
    "exec",
    "eval",
    "compile",
    "__import__",
    "__builtins__",
    "globals",
    "vars",
    "getattr",
    "__globals__",
    "__subclasses__",
    "__bases__",
    "__mro__",
    "__class__",
}

MODELS_FORBIDDEN_PACKAGES = ("wmj.worlds", "wmj.harness", "wmj.judge", "wmj.reporting")


def _parse_file(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _judge_source_files() -> list[Path]:
    judge_dir = SRC_ROOT / "judge"
    return sorted(judge_dir.rglob("*.py"))


def _models_source_files() -> list[Path]:
    models_dir = SRC_ROOT / "models"
    return sorted(models_dir.rglob("*.py"))


def _fixture_files() -> list[Path]:
    return sorted(FIXTURES_DIR.glob("*.py"))


def run_judge_gate(tree: ast.Module) -> list[str]:
    """The combined judge gate: allowlist + banned identifiers + numpy.random.

    Returns a list of human-readable violation descriptions — empty
    means the gate passes.
    """
    violations: list[str] = []

    all_names = full_import_names(tree) | joined_importfrom_names(tree)
    disallowed_imports = {
        name for name in all_names if not _is_allowed_judge_import(name)
    }
    if disallowed_imports:
        violations.append(
            f"TC-NF6-01: import(s) outside the judge allowlist {JUDGE_ALLOWLIST}: "
            f"{sorted(disallowed_imports)}"
        )

    banned = scan_banned_identifiers(tree, BANNED_IDENTIFIERS)
    if banned:
        violations.append(f"TC-NF6-02: banned identifier(s) found: {sorted(banned)}")

    if scan_numpy_random_usage(tree):
        violations.append("TC-NF6-03: numpy.random usage found")

    return violations


def _models_gate_violations(tree: ast.Module) -> list[str]:
    """TC-NF6-07/08/09: models must never import worlds/harness/judge/reporting."""
    names = full_import_names(tree) | joined_importfrom_names(tree)
    violations = []
    for name in sorted(names):
        for forbidden in MODELS_FORBIDDEN_PACKAGES:
            if name == forbidden or name.startswith(forbidden + "."):
                violations.append(f"models imports {name!r} (forbidden: {forbidden})")
    return violations


# --- TC-NF6-01/02/03: the judge allowlist/identifier/numpy.random checks ---


def test_tc_nf6_01_02_03_judge_package_only_uses_allowlisted_imports():
    for path in _judge_source_files():
        tree = _parse_file(path)
        violations = run_judge_gate(tree)
        assert violations == [], f"{path}: {violations}"


# --- TC-NF6-04: the evasion-fixture regression corpus ---


def test_tc_nf6_04_every_fixture_is_flagged():
    fixtures = _fixture_files()
    assert len(fixtures) >= 12, "the evasion corpus should cover every named category"
    for path in fixtures:
        tree = _parse_file(path)
        violations = run_judge_gate(tree)
        assert violations != [], f"fixture {path} was NOT flagged (regression!)"


# --- TC-NF6-06: the clean-pass / false-positive guard ---


def test_tc_nf6_06_real_judge_source_passes_clean():
    judge_files = _judge_source_files()
    assert len(judge_files) > 0, "expected real judge source files to exist by now"
    for path in judge_files:
        tree = _parse_file(path)
        violations = run_judge_gate(tree)
        assert violations == [], f"{path} was flagged (false positive): {violations}"


# --- TC-NF6-07/08/09: the models-side sideways-import gates ---


def test_tc_nf6_07_08_09_models_never_imports_worlds_harness_judge_reporting():
    models_files = _models_source_files()
    assert len(models_files) > 0, "expected real models source files to exist by now"
    for path in models_files:
        tree = _parse_file(path)
        violations = _models_gate_violations(tree)
        assert violations == [], f"{path}: {violations}"


def test_tc_nf6_07_08_09_negative_catches_a_forbidden_import_from_shape():
    """Phantom-gate proof: from wmj.worlds import lv must be caught by
    the joined comparison, not only `import wmj.worlds.lv`."""
    tree = ast.parse("from wmj.worlds import lv\n")
    violations = _models_gate_violations(tree)
    assert violations != []


def test_tc_nf6_07_08_09_allows_the_sanctioned_models_base_import():
    """models.base/registry importing each other, or being imported by
    the harness, is the allowed direction — this gate only checks what
    *models* imports outward, and wmj.models.base is not forbidden."""
    tree = ast.parse("from wmj.models.base import SeedSource, component_key\n")
    violations = _models_gate_violations(tree)
    assert violations == []
