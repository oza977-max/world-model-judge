"""Tests for tests.gates._ast_utils — the shared AST-walking helpers
every TC-NF6 import/identifier gate is built from.

Each helper is checked against tiny synthetic source snippets, not
real files, so a bug in the helper itself is caught independently of
whatever it is later pointed at (fixtures or real source).
"""

from __future__ import annotations

import ast

from tests.gates._ast_utils import (
    full_import_names,
    joined_importfrom_names,
    scan_banned_identifiers,
    scan_metaclass_usage,
    scan_numpy_random_usage,
    top_level_import_names,
)


def _parse(source: str) -> ast.Module:
    return ast.parse(source)


def test_top_level_import_names_finds_bare_import():
    tree = _parse("import os\n")
    assert top_level_import_names(tree) == {"os"}


def test_top_level_import_names_finds_aliased_import():
    tree = _parse("import os as o\n")
    assert top_level_import_names(tree) == {"os"}


def test_top_level_import_names_finds_dotted_import_top_level_only():
    tree = _parse("import numpy.random\n")
    assert top_level_import_names(tree) == {"numpy"}


def test_top_level_import_names_finds_from_import_module():
    tree = _parse("from wmj.worlds import lv\n")
    assert top_level_import_names(tree) == {"wmj"}


def test_top_level_import_names_ignores_relative_import_with_no_module():
    tree = _parse("from . import lv\n")
    assert top_level_import_names(tree) == set()


def test_full_import_names_keeps_the_complete_dotted_path():
    tree = _parse("import wmj.worlds.lv\n")
    assert full_import_names(tree) == {"wmj.worlds.lv"}


def test_full_import_names_ignores_the_alias():
    tree = _parse("import wmj.worlds.lv as world\n")
    assert full_import_names(tree) == {"wmj.worlds.lv"}


def test_full_import_names_handles_a_bare_top_level_import():
    tree = _parse("import numpy\n")
    assert full_import_names(tree) == {"numpy"}


def test_joined_importfrom_names_joins_module_and_alias():
    tree = _parse("from wmj.models import direct\n")
    joined = joined_importfrom_names(tree)
    assert "wmj.models.direct" in joined
    assert "wmj.models" in joined


def test_joined_importfrom_names_also_includes_bare_module():
    tree = _parse("from wmj.models.base import SeedSource\n")
    # node.module itself already names the allowed submodule directly
    joined = joined_importfrom_names(tree)
    assert "wmj.models.base" in joined
    assert "wmj.models.base.SeedSource" in joined


def test_joined_importfrom_names_handles_multiple_aliases():
    tree = _parse("from wmj.models import direct, ensemble\n")
    joined = joined_importfrom_names(tree)
    assert "wmj.models.direct" in joined
    assert "wmj.models.ensemble" in joined


def test_scan_banned_identifiers_finds_name_node():
    tree = _parse("x = eval('1')\n")
    found = scan_banned_identifiers(tree, {"eval"})
    assert "eval" in found


def test_scan_banned_identifiers_finds_attribute_node():
    tree = _parse("x = some_fn.__globals__\n")
    found = scan_banned_identifiers(tree, {"__globals__"})
    assert "__globals__" in found


def test_scan_banned_identifiers_finds_import_alias():
    tree = _parse("import builtins as __builtins__\n")
    found = scan_banned_identifiers(tree, {"__builtins__"})
    assert "__builtins__" in found


def test_scan_banned_identifiers_clean_source_finds_nothing():
    tree = _parse("import numpy as np\nx = np.array([1, 2])\n")
    found = scan_banned_identifiers(tree, {"eval", "exec", "__globals__"})
    assert found == set()


def test_scan_metaclass_usage_finds_a_metaclass_keyword():
    tree = _parse("class Probe(metaclass=SomeMeta):\n    pass\n")
    assert scan_metaclass_usage(tree) is True


def test_scan_metaclass_usage_clean_source_finds_nothing():
    tree = _parse("class Probe:\n    pass\n")
    assert scan_metaclass_usage(tree) is False


def test_scan_metaclass_usage_ignores_ordinary_base_classes():
    tree = _parse("class Probe(SomeBase, OtherBase):\n    pass\n")
    assert scan_metaclass_usage(tree) is False


def test_scan_numpy_random_usage_finds_direct_import():
    tree = _parse("import numpy.random\n")
    assert scan_numpy_random_usage(tree) is True


def test_scan_numpy_random_usage_finds_from_import():
    tree = _parse("from numpy.random import rand\n")
    assert scan_numpy_random_usage(tree) is True


def test_scan_numpy_random_usage_finds_attribute_access():
    tree = _parse("import numpy as np\nx = np.random.normal(0, 1)\n")
    assert scan_numpy_random_usage(tree) is True


def test_scan_numpy_random_usage_clean_source_finds_nothing():
    tree = _parse("import numpy as np\nx = np.array([1.0])\n")
    assert scan_numpy_random_usage(tree) is False
