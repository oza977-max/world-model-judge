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

**Metaclass ban, added pass 3 (independent-review finding):** a custom
metaclass's `__new__` receives the class body's namespace as a plain
dict, handed over as an ORDINARY parameter — the same capability
`__dict__`/`getattr` expose, but reaching it this way triggers no
banned Name/Attribute node at all, since the dict just arrives as a
function argument. This is a structural check (`scan_metaclass_usage`),
not one more identifier to enumerate — the reviewer's own point was
that identifier-by-identifier patching of this class of route is
whack-a-mole; banning `metaclass=` outright closes the whole family in
one rule, since no wmj code has a legitimate reason to define one.
Applied to both the judge gate and the models gate.
"""

from __future__ import annotations

import ast
from pathlib import Path

from tests.gates._ast_utils import (
    full_import_names,
    joined_importfrom_names,
    scan_banned_identifiers,
    scan_metaclass_usage,
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
#
# **`locals` added, pass 4 (independent-review finding):** the third
# member of Python's own namespace-introspection trio — `locals`/
# `globals`/`vars` are conventionally documented together, and `globals`
# and `vars` were already banned; omitting `locals` left the identical
# computed-name lookup open via `locals()["s"+"qrt"]`, confirmed by
# execution to reach `math.sqrt` with zero previously-banned identifiers
# touched. Same "obvious sibling omission" class as `__dict__`, not a
# new kind of gap.
#
# **Traceback/frame identifiers added, pass 5 (independent-review
# finding).** `except X as e: e.__traceback__.tb_frame` hands over a
# live frame object with no import and no identifier from the list
# above — frame objects expose `f_globals`/`f_locals`/`f_builtins`
# (the SAME capability `globals`/`locals`/`__builtins__` already ban,
# reached via a different route: exception handling, a core language
# feature, not a named builtin call) and `f_back`/`tb_next` (walking
# the frame/traceback chain to outer scopes). Confirmed by execution:
# `frame.f_builtins["ev"+"al"]` reaches the entire builtins table in
# one hop — more consequential than any prior finding, since it is not
# scoped to one module's namespace. Not import-blocked (raising/
# catching needs no import) and not the disclosed ctypes/pre-capture
# residual (this is a pure syntax-level route, exactly what this lint
# exists to catch).
BANNED_IDENTIFIERS = {
    "exec",
    "eval",
    "compile",
    "__import__",
    "__builtins__",
    "globals",
    "locals",
    "vars",
    "getattr",
    "__globals__",
    "__dict__",
    "__getattribute__",
    "__subclasses__",
    "__bases__",
    "__mro__",
    "__class__",
    "__traceback__",
    "tb_frame",
    "tb_next",
    "f_globals",
    "f_locals",
    "f_builtins",
    "f_back",
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

    if scan_metaclass_usage(tree):
        violations.append(
            "metaclass usage found (structural ban, P1-C03 pass 3 — see "
            "module docstring)"
        )

    return violations


def _models_gate_violations(tree: ast.Module) -> list[str]:
    """TC-NF6-07/08/09: models' outward imports are a full allowlist,
    not just a denylist of the three named forbidden directions —
    matching cross-cutting ADR-003's own "nothing else, in any
    direction" claim (with the wmj.errors correction documented above).
    Also applies the metaclass structural ban (see module docstring)."""
    names = full_import_names(tree) | joined_importfrom_names(tree)
    violations = [
        f"models imports {name!r}, outside the allowlist {MODELS_ALLOWLIST}"
        for name in sorted(names)
        if not _is_allowed_models_import(name)
    ]
    if scan_metaclass_usage(tree):
        violations.append(
            "metaclass usage found (structural ban, P1-C03 pass 3 — see "
            "module docstring)"
        )
    return violations


# --- Metaclass structural ban (P1-C03 pass 3) ---


def test_metaclass_fixture_is_caught_specifically_by_the_metaclass_check():
    """Isolation proof, matching how the __dict__ fixture was verified:
    this fixture must be flagged BECAUSE of the metaclass mechanism,
    not confounded by some already-banned identifier also present."""
    tree = _parse_file(
        FIXTURES_DIR / "metaclass_namespace_capture_route.py"
    )
    assert scan_metaclass_usage(tree) is True
    assert scan_banned_identifiers(tree, BANNED_IDENTIFIERS) == set()
    assert run_judge_gate(tree) != []


def test_locals_fixture_is_caught_specifically_by_the_locals_identifier():
    """Isolation proof: the pass-4 fixture must be flagged because
    `locals` itself is now banned, not confounded by anything else."""
    tree = _parse_file(FIXTURES_DIR / "locals_namespace_lookup_route.py")
    banned_found = scan_banned_identifiers(tree, BANNED_IDENTIFIERS)
    assert banned_found == {"locals"}
    assert run_judge_gate(tree) != []


def test_traceback_fixture_is_caught_specifically_by_the_frame_identifiers():
    """Isolation proof: the pass-5 fixture must be flagged because of
    the traceback/frame identifiers, not any pre-existing banned name —
    it legitimately uses three of the newly-added ones together
    (__traceback__, tb_frame, f_locals), none of which existed on the
    list before this round."""
    tree = _parse_file(FIXTURES_DIR / "traceback_frame_capture_route.py")
    banned_found = scan_banned_identifiers(tree, BANNED_IDENTIFIERS)
    assert banned_found == {"__traceback__", "tb_frame", "f_locals"}
    assert run_judge_gate(tree) != []


# --- TC-NF6-01/02/03: the judge allowlist/identifier/numpy.random checks ---


def test_tc_nf6_01_02_03_judge_package_only_uses_allowlisted_imports():
    for path in _judge_source_files():
        tree = _parse_file(path)
        violations = run_judge_gate(tree)
        assert violations == [], f"{path}: {violations}"


# --- TC-NF6-04: the evasion-fixture regression corpus ---


def test_tc_nf6_04_every_fixture_is_flagged():
    fixtures = _fixture_files()
    assert len(fixtures) == 18, (
        "the evasion corpus should cover exactly the 18 named categories "
        "(cross-cutting ADR-003 / TC-NF6-04, plus the pass-2-added "
        "__dict__ namespace-lookup route, the pass-3-added metaclass "
        "namespace-capture route, the pass-4-added locals() route, and "
        "the pass-5-added traceback/frame-capture route) — a looser "
        "bound would tolerate silently losing a fixture"
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
