"""The judge import-allowlist gate and its family (cross-cutting ADR-003).

TC-NF6-01: judge package imports only {numpy, math, dataclasses, typing}.
TC-NF6-02: banned identifiers anywhere in the judge package (fast lint,
not a completeness proof — see module docstrings in cross-cutting.md).
TC-NF6-03: numpy.random banned wholesale in the judge.
TC-NF6-04: the evasion-fixture regression corpus — every fixture must
be flagged by the gate.
TC-NF6-06: the clean-pass guard — the real judge source must NOT be
flagged (checks 02/03 are deliberately over-broad).
TC-NF6-07/08/09: models package's outward imports are a full allowlist
(not just the three named forbidden directions), matching cross-cutting
ADR-003's own claim that models' "only sanctioned outward imports are
now, completely: numpy, math, dataclasses, typing, wmj.models.base,
wmj.models.registry — nothing else, in any direction." **Corrected here
to also admit `wmj.errors` and `hashlib` (independent-review finding,
P1-C03 pass 1, extended while building the real allowlist):** that
sentence is falsified by the project's own already-reviewed code —
`wmj/models/base.py` legitimately imports `wmj.errors.WmjError`
(the pinned "every wmj exception subclasses WmjError" convention) and
`hashlib` (`component_key`'s blake2b digest, ADR-002 rule 2), both
built and reviewed at P1-C01, both omitted from cross-cutting.md's
list. Rather than silently building a gate that lets the real code
pass while leaving the spec's overclaim uncorrected, both are added to
the allowlist here and the omission is flagged as a future
design-review correction to cross-cutting.md (not amended in this
build chunk).
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
#
# **Extended past the literal pinned 13 (independent-review finding,
# P1-C03 pass 2): `__dict__` and `__getattribute__` added.** Pass 2
# proved by execution that `module.__dict__[computed_name]` reaches and
# calls a dynamically-named attribute (e.g. `math.__dict__["sqrt"]`)
# using zero identifiers from the originally-pinned list and only
# allowlisted imports — a real, working bypass, not a theoretical one.
# `__getattribute__` is the same reflection primitive in method form
# (`obj.__getattribute__(name)` is `getattr(obj, name)` restated) and
# is added alongside it for the same reason. Both are exactly the class
# of *known, enumerable* reflection primitive this list already bans
# (`vars`/`getattr`/`__globals__`/`__subclasses__` are the same kind of
# thing) — omitting them was a gap in the pinned list, not evidence
# that this class of route is inherently undetectable (that honest
# residual is `ctypes` and pre-capture routes, per cross-cutting ADR-002
# rule 3, not this). Flagged as a future design-review correction to
# cross-cutting.md's pinned list, not silently edited there.
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
    "__dict__",
    "__getattribute__",
    "__subclasses__",
    "__bases__",
    "__mro__",
    "__class__",
}

# cross-cutting ADR-003's pinned "only sanctioned outward imports"
# list, PLUS wmj.errors and hashlib (see module docstring's Post-review
# addendum above — the spec text omitted both; wmj/models/base.py's
# component_key needs hashlib.blake2b per ADR-002 rule 2, built and
# reviewed at P1-C01).
MODELS_ALLOWLIST = {"numpy", "math", "dataclasses", "typing", "wmj.errors", "hashlib"}
MODELS_OWN_PACKAGE = "wmj.models"


def _is_allowed_models_import(name: str) -> bool:
    if name == MODELS_OWN_PACKAGE or name.startswith(MODELS_OWN_PACKAGE + "."):
        return True
    for allowed in MODELS_ALLOWLIST | ALWAYS_ALLOWED:
        if name == allowed or name.startswith(allowed + "."):
            return True
    return False


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
    """TC-NF6-07/08/09: models' outward imports are a full allowlist,
    not just a denylist of the three named forbidden directions —
    matching cross-cutting ADR-003's own "nothing else, in any
    direction" claim (with the wmj.errors correction documented above)."""
    names = full_import_names(tree) | joined_importfrom_names(tree)
    return [
        f"models imports {name!r}, outside the allowlist {MODELS_ALLOWLIST}"
        for name in sorted(names)
        if not _is_allowed_models_import(name)
    ]


# --- TC-NF6-01/02/03: the judge allowlist/identifier/numpy.random checks ---


def test_tc_nf6_01_02_03_judge_package_only_uses_allowlisted_imports():
    for path in _judge_source_files():
        tree = _parse_file(path)
        violations = run_judge_gate(tree)
        assert violations == [], f"{path}: {violations}"


# --- TC-NF6-04: the evasion-fixture regression corpus ---


def test_tc_nf6_04_every_fixture_is_flagged():
    fixtures = _fixture_files()
    assert len(fixtures) == 15, (
        "the evasion corpus should cover exactly the 15 named categories "
        "(cross-cutting ADR-003 / TC-NF6-04, plus the pass-2-added "
        "__dict__ namespace-lookup route) — a looser bound would "
        "tolerate silently losing a fixture"
    )
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


def test_tc_nf6_07_08_09_allows_wmj_errors():
    """wmj.errors is the documented allowlist correction (see this
    file's module docstring) — real code needs it and it isn't one of
    the four forbidden sideways packages."""
    tree = ast.parse("from wmj.errors import WmjError\n")
    assert _models_gate_violations(tree) == []


def test_tc_nf6_07_08_09_is_a_real_allowlist_not_just_a_four_item_denylist():
    """Independent-review finding, P1-C03 pass 1: the gate must reject
    ANY import outside the sanctioned set, not only the four named
    forbidden packages — otherwise `import os` or `import requests`
    inside a models file would sail through uncaught."""
    for source in ("import os\n", "import requests\n", "import sys\n"):
        tree = ast.parse(source)
        assert _models_gate_violations(tree) != [], f"{source!r} was not caught"
