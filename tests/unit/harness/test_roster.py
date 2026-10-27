"""MU-9: adding a model is one file, and nothing outside the models package names a model module.

In plain words: a stub model file is placed in a temporary folder that is grafted onto the models package.
The harness's own roster function must find it with no other file touched (TC-MU9-01 (a) and (b)). Then the
judge, worlds, reporting and harness code is read for any import of a model module other than the two
shared ones, the registry and the base types (TC-MU9-01 (c)). The check is shown to fail when such an import
is planted (TC-MU9-02). Discovery lives in the harness, by owner decision D7 (backlog A42 item 1).
"""

from __future__ import annotations

import ast
import importlib
import sys
from pathlib import Path

import wmj.models
from wmj.harness import trials
from wmj.models import registry

SRC = Path(__file__).resolve().parents[3] / "src" / "wmj"
ALLOWED = ("wmj.models.registry", "wmj.models.base")
STUB = "zz_stub_model"
REAL_MODELS = {"direct", "ensemble", "fx-action-blind", "fx-brittle", "fx-honest-rough", "fx-overconfident",
               "linear", "persistence"}


def test_tc_mu9_01_ab_a_grafted_model_file_is_found_by_the_production_roster(tmp_path, monkeypatch):
    before = trials.roster()  # forces every real model to register before the snapshot (TC-MU9-01 teardown rule)
    assert set(before) == REAL_MODELS
    snapshot = dict(registry._REGISTRY)
    (tmp_path / f"{STUB}.py").write_text(
        "from wmj.models.registry import register\n"
        "register('zz-stub', lambda ctx, seeds, training: None)\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(wmj.models, "__path__", [*wmj.models.__path__, str(tmp_path)])
    try:
        importlib.invalidate_caches()
        found = trials.roster()  # (b) the production path
        assert set(found) == REAL_MODELS | {"zz-stub"}
        assert "zz-stub" in registry.all_models()  # (a) the registry now holds it
        assert list(found) == sorted(found)
    finally:
        monkeypatch.undo()
        registry._REGISTRY.clear()
        registry._REGISTRY.update(snapshot)
        sys.modules.pop(f"wmj.models.{STUB}", None)
    assert set(trials.roster()) == REAL_MODELS


def forbidden_imports(source: str) -> list[str]:
    """Every import of a `wmj.models` submodule other than the registry and the base types."""
    bad = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            if node.module in ALLOWED:  # design-review-008 I4: `from wmj.models.base import X` passes on the module
                continue
            names = [f"{node.module}.{alias.name}" for alias in node.names]  # design-review-005: join module + name
        else:
            continue
        # whole module names only: "wmj.models.baselines" merely starts with the text "wmj.models.base"
        bad += [n for n in names if n.startswith("wmj.models.") and not any(n == a or n.startswith(a + ".") for a in ALLOWED)]
    return bad


def test_tc_mu9_01_c_no_other_package_imports_a_model_module():
    found = {}
    for package in ("judge", "worlds", "reporting", "harness"):
        for path in sorted((SRC / package).rglob("*.py")):
            bad = forbidden_imports(path.read_text(encoding="utf-8"))
            if bad:
                found[str(path.relative_to(SRC))] = bad
    assert found == {}


def test_tc_mu9_01_c_the_sanctioned_shapes_pass():
    clean = (
        "from wmj.models.base import SeedSource, component_key\n"
        "from wmj.models.registry import all_models\n"
        "from wmj.models import registry\n"
        "import wmj.models\n"
        "import wmj.models.base\n"
    )
    assert forbidden_imports(clean) == []


def test_tc_mu9_02_a_planted_model_import_in_a_scratch_copy_of_a_judge_file_fails_the_check():
    original = (SRC / "judge" / "skill.py").read_text(encoding="utf-8")
    for planted in ("from wmj.models import direct\n", "import wmj.models.ensemble\n",
                    "from wmj.models.baselines import linear_factory\n"):
        assert forbidden_imports(planted + original), planted
