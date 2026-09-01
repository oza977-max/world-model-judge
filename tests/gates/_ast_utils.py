"""Shared AST-walking helpers for the TC-NF6 import/identifier gates.

In plain words: every gate in this family asks one of three questions
about a piece of source code — "what does it import?", "does it use
one of a fixed list of dangerous names?", "does it touch
numpy.random?" — and each question needs to be answered the same
mechanical way everywhere it's asked (cross-cutting ADR-003). These
four functions are that one shared answer, so TC-NF6-01/07/08/09
(imports), TC-NF6-02 (identifiers), and TC-NF6-03 (numpy.random) never
each grow their own slightly-different AST walk.

This module is test-support infrastructure, not a product deliverable
— it is never imported by anything under src/.
"""

from __future__ import annotations

import ast


def top_level_import_names(tree: ast.Module) -> set[str]:
    """The top-level module name of every Import/ImportFrom in tree.

    `import numpy.random` and `import numpy.random as npr` both yield
    `numpy` (Python only ever binds the top-level package name unless
    aliased with `as`, in which case the alias is irrelevant here —
    what matters is which package was reached). `from wmj.worlds import
    lv` yields `wmj`. A relative import with no module (`from . import
    x`) contributes nothing — it has no top-level absolute name.
    """
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                names.add(node.module.split(".")[0])
    return names


def full_import_names(tree: ast.Module) -> set[str]:
    """The complete dotted name of every plain `Import` statement.

    Unlike `top_level_import_names`, this keeps the whole path:
    `import wmj.worlds.lv` yields `wmj.worlds.lv`, not just `wmj` —
    the models→worlds/harness/judge/reporting gates (TC-NF6-07/08/09)
    need the full path to tell a same-package import from a forbidden
    sideways one.
    """
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name)
    return names


def joined_importfrom_names(tree: ast.Module) -> set[str]:
    """Every ImportFrom's module, AND module+'.'+name, joined.

    This is the check TC-NF6-07/08/09 need: `from wmj.models import
    direct` must be caught as reaching `wmj.models.direct`, not waved
    through because `node.module` alone is just `wmj.models` (design-
    review-005's fix — `node.module`-only comparisons let `from
    wmj.models import direct` and `from wmj.models import registry`
    look identical). Including the bare `node.module` too lets a
    caller allow `from wmj.models.base import SeedSource` by checking
    for the literal string `wmj.models.base` without a spurious join
    to `wmj.models.base.SeedSource` (design-review-008 I4's fix).
    """
    joined: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            joined.add(node.module)
            for alias in node.names:
                joined.add(f"{node.module}.{alias.name}")
    return joined


def scan_banned_identifiers(tree: ast.Module, banned: set[str]) -> set[str]:
    """Every banned name that appears as a Name id, Attribute attr, or
    import alias anywhere in tree (TC-NF6-02's identifier scan).

    Deliberately over-broad by design (cross-cutting ADR-003): this
    cannot tell a legitimate use from an evasion, so a real judge
    source file must independently pass TC-NF6-06's clean-pass check.
    """
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in banned:
            found.add(node.id)
        elif isinstance(node, ast.Attribute) and node.attr in banned:
            found.add(node.attr)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                bound_name = alias.asname or alias.name
                if bound_name in banned:
                    found.add(bound_name)
    return found


def scan_numpy_random_usage(tree: ast.Module) -> bool:
    """True if tree imports numpy.random (any form) or accesses a
    `.random` attribute anywhere (TC-NF6-03's wholesale ban).

    The bare `attr == "random"` half is deliberately over-broad, like
    the identifier scan above — a real judge source file has no
    `.random` attribute of any kind, so TC-NF6-06 is what proves this
    doesn't cry wolf on legitimate code.
    """
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "numpy.random" or alias.name.startswith(
                    "numpy.random."
                ):
                    return True
        elif isinstance(node, ast.ImportFrom):
            if node.module and (
                node.module == "numpy.random"
                or node.module.startswith("numpy.random.")
            ):
                return True
        elif isinstance(node, ast.Attribute) and node.attr == "random":
            return True
    return False
