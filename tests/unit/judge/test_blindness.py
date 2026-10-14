"""TC-JU1-02 (blindness under relabelling and reordering) and TC-JU13-01 (nothing in the judge is learned).

In plain words: the judge must not be able to tell *which* model it is judging, so the only things
that can change a verdict are the numbers. These tests relabel the regions consistently, shuffle the
trials consistently, and judge the same content twice, and require the verdict to be identical up to
the relabelling or the order. A separate test reads the judge's own source and its imports and checks
that nothing in it is trained: no weights files, no model-building or inference vocabulary, nothing
imported beyond numpy and the standard library.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import numpy as np

from tests.unit.judge._builders import judge_input_kwargs
from wmj.judge.calibration import compute_calibration
from wmj.judge.climatology import compute_climatology
from wmj.judge.exceptions import compute_exceptions_and_trials
from wmj.judge.horizon import compute_error_vs_horizon, compute_trust_horizons
from wmj.judge.sharpness import compute_sharpness
from wmj.judge.skill import compute_skill
from wmj.judge.types import (
    Forecasts,
    JudgeInput,
    RegionClimatology,
    RegionCurve,
    RegionLabel,
)
from wmj.judge.verdict import assemble_verdict

JUDGE_DIR = Path(__file__).resolve().parents[3] / "src" / "wmj" / "judge"


def _verdict(inp: JudgeInput) -> dict:
    exceptions, trials = compute_exceptions_and_trials(inp)
    return assemble_verdict(
        world=inp.world, skill=compute_skill(inp), error_vs_horizon=compute_error_vs_horizon(inp),
        calibration=compute_calibration(inp), sharpness=compute_sharpness(inp), exceptions=exceptions, trials=trials,
        climatology=compute_climatology(inp), trust_horizons=compute_trust_horizons(inp),
    ).to_dict()


def _random_kwargs(seed: int) -> dict:
    kwargs = judge_input_kwargs()
    rng = np.random.default_rng(seed)
    n, h, d = kwargs["predictions"].mean.shape
    kwargs["predictions"] = Forecasts(rng.normal(size=(n, h, d)), rng.uniform(0.3, 2.0, size=(n, h, d)))
    kwargs["outcomes"] = rng.normal(size=(n, h, d))
    return kwargs


def _renamed(kwargs: dict, mapping: dict[str, str]) -> dict:
    out = dict(kwargs)
    out["region_labels"] = tuple(RegionLabel(mapping[x.region_name], x.axis) for x in kwargs["region_labels"])
    out["divergence_curves"] = tuple(RegionCurve(mapping[c.region_name], c.curve) for c in kwargs["divergence_curves"])
    out["climatology"] = tuple(RegionClimatology(mapping[t.region_name], t.bins) for t in kwargs["climatology"])
    return out


def _rename_in_verdict(record: dict, mapping: dict[str, str]) -> dict:
    """The verdict with every region name replaced by `mapping` and every per-region list put in sorted-name order."""
    def fix(node):
        if isinstance(node, dict):
            return {k: (mapping[v] if k == "region" and isinstance(v, str) else fix(v)) for k, v in node.items()}
        if isinstance(node, list):
            return [fix(x) for x in node]
        return node

    renamed = fix(record)
    for block, key in (("skill", "per_task_region"), ("calibration", "per_task"), ("sharpness", "per_task"), ("climatology", "per_task"),
                       ("trust_horizons", "per_task"), ("exceptions", "per_task"), ("trials", "per_task")):
        renamed[block][key].sort(key=lambda e: (e["task"], e["region"], e.get("horizon_step", 0)))
    renamed["error_vs_horizon"]["per_region"].sort(key=lambda e: e["region"])
    return renamed


def _same_up_to_rounding(a, b, path="") -> None:
    """Equal structure and text; counts and flags exactly; floats to 1e-12 (summing trials in another order rounds differently)."""
    assert type(a) is type(b), path
    if isinstance(a, dict):
        assert a.keys() == b.keys(), path
        for k in a:
            _same_up_to_rounding(a[k], b[k], f"{path}/{k}")
    elif isinstance(a, list | tuple):
        assert len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b, strict=True)):
            _same_up_to_rounding(x, y, f"{path}[{i}]")
    elif isinstance(a, float):
        assert a == b or abs(a - b) <= 1e-12 * max(1.0, abs(a), abs(b)), (path, a, b)
    else:
        assert a == b, (path, a, b)


def test_identical_content_judged_twice_gives_identical_verdicts():
    kwargs = _random_kwargs(1)
    assert _verdict(JudgeInput(**kwargs)) == _verdict(JudgeInput(**kwargs))


def test_tc_ju1_02_relabelling_the_regions_changes_only_the_labels_over_many_random_inputs():
    for seed in range(12):
        kwargs = _random_kwargs(seed)
        mapping = {"training": "zeta", "out-of-range": "alpha"}  # also flips the sorted order of the regions
        original = _verdict(JudgeInput(**kwargs))
        relabelled = _verdict(JudgeInput(**_renamed(kwargs, mapping)))
        assert relabelled == _rename_in_verdict(original, mapping), seed


def test_tc_ju1_02_reordering_the_trials_consistently_changes_nothing_but_the_order_of_the_per_trial_lists():
    for seed in range(8):
        kwargs = _random_kwargs(seed)
        n = len(kwargs["region_labels"])
        order = np.random.default_rng(100 + seed).permutation(n)
        shuffled = dict(kwargs)
        for name in ("predictions", "persistence", "linear"):
            f = kwargs[name]
            shuffled[name] = Forecasts(f.mean[order], f.spread[order])
        shuffled["outcomes"] = kwargs["outcomes"][order]
        shuffled["region_labels"] = tuple(kwargs["region_labels"][i] for i in order)
        shuffled["invariant_bins"] = kwargs["invariant_bins"][order]
        a, b = _verdict(JudgeInput(**kwargs)), _verdict(JudgeInput(**shuffled))
        for record in (a, b):  # the per-trial chart lists legitimately follow the trial order: compare them as multisets
            for entry in record["trials"]["per_task"]:
                rows = sorted(zip(entry["outcome_distance"], entry["band_hi"], entry["is_exception"], strict=True))
                entry["outcome_distance"], entry["band_hi"], entry["is_exception"] = (list(c) for c in zip(*rows, strict=True))
        _same_up_to_rounding(a, b)


def test_the_input_has_no_field_that_could_carry_a_model_identity_so_there_is_nothing_to_swap():
    fields = set(JudgeInput.__dataclass_fields__)
    assert fields == {
        "world", "dt", "natural_cycle_length", "predictions", "outcomes", "persistence", "linear", "region_labels",
        "divergence_curves", "climatology", "invariant_bins", "tasks", "thresholds",
    }
    assert not any(re.search(r"model|name_of|fixture|label_of|provenance|architecture|weights", f) for f in fields)


# ---- TC-JU13-01 -----------------------------------------------------------------------------------------

LEARNED_VOCABULARY = re.compile(r"\b(torch|tensorflow|sklearn|scikit|keras|jax|onnx|pickle|joblib|state_dict|checkpoint|backprop|optimizer|optimiser|gradient_descent|train(ed|ing)?|epochs?|fit_transform)\b", re.IGNORECASE)
WEIGHT_FILE_SUFFIXES = {".pt", ".pth", ".npz", ".npy", ".pkl", ".pickle", ".h5", ".onnx", ".bin", ".safetensors", ".joblib"}


def test_tc_ju13_01_the_judge_directory_holds_only_source_no_weights_files():
    files = [p for p in JUDGE_DIR.rglob("*") if p.is_file() and "__pycache__" not in p.parts]
    assert files and all(p.suffix == ".py" for p in files), [p.name for p in files if p.suffix != ".py"]
    assert not [p for p in files if p.suffix in WEIGHT_FILE_SUFFIXES]


def test_tc_ju13_01_the_judge_imports_nothing_but_numpy_and_the_standard_library_and_has_no_learning_vocabulary():
    allowed_top = {"numpy", "math", "dataclasses", "typing", "wmj", "__future__"}
    for path in JUDGE_DIR.glob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                assert {a.name.split(".")[0] for a in node.names} <= allowed_top, (path.name, [a.name for a in node.names])
            elif isinstance(node, ast.ImportFrom):
                assert (node.module or "").split(".")[0] in allowed_top, (path.name, node.module)
        code_only = "\n".join(
            line for line in path.read_text().splitlines() if not line.lstrip().startswith("#")
        )
        identifiers = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} | {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
        assert not [i for i in identifiers if LEARNED_VOCABULARY.fullmatch(i)], (path.name, identifiers)
        assert "def predict" not in code_only and "def fit" not in code_only and "def train" not in code_only


def test_tc_ju13_01_the_inspection_can_fail_a_planted_learned_component_is_flagged():
    planted = "import torch\nclass Net:\n    def predict(self, x):\n        return x\n"
    tree = ast.parse(planted)
    imports = {a.name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    assert imports - {"numpy", "math", "dataclasses", "typing", "wmj", "__future__"} == {"torch"}
    assert "def predict" in planted
    assert LEARNED_VOCABULARY.search("optimizer.step()")
