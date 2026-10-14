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
    ClimatologyBin,
    Forecasts,
    JudgeInput,
    RegionClimatology,
    RegionCurve,
    RegionLabel,
    TaskSpec,
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


def _random_kwargs(seed: int, *, near_perfect: bool = False) -> dict:
    """Random forecasts with a different curve and climatology table per region (so a mix-up of regions changes the numbers).

    `near_perfect` puts the forecasts close to the outcomes and the outcomes on the climatology, so trust horizons are
    non-zero and the agreement test holds — the branches a random input never reaches."""
    kwargs = judge_input_kwargs()
    rng = np.random.default_rng(seed)
    n, h, d = kwargs["predictions"].mean.shape
    kwargs["predictions"] = Forecasts(rng.normal(size=(n, h, d)), rng.uniform(0.3, 2.0, size=(n, h, d)))
    kwargs["outcomes"] = rng.normal(size=(n, h, d))
    kwargs["divergence_curves"] = (
        RegionCurve("training", np.array([0.0, 0.01, 0.02, 0.2, 0.5, 0.6])),
        RegionCurve("out-of-range", np.array([0.0, 0.05, 0.3, 0.4, 0.45, 0.5])),
    )
    kwargs["climatology"] = tuple(
        RegionClimatology(
            name,
            tuple(
                ClimatologyBin(-np.inf if i == 0 else float(i), np.inf if i == 3 else float(i + 1),
                               np.array([1.0, 2.0]) * (i + 1) * scale, np.array([0.1, 0.2]) * (i + 1) * scale, 100)
                for i in range(4)
            ),
        )
        for name, scale in (("training", 1.0), ("out-of-range", 7.0))
    )
    kwargs["tasks"] = (TaskSpec("alpha-task", "control", 0.1, 3), TaskSpec("beta-task", "planning", 0.3, 5))
    if near_perfect:
        bins = np.zeros((n, h), dtype=np.int64)
        means = np.stack([kwargs["climatology"][0 if x.region_name == "training" else 1].bins[0].mean for x in kwargs["region_labels"]])
        target = np.broadcast_to(means[:, None, :], (n, h, d)).copy()
        target += rng.normal(scale=1e-4, size=target.shape)
        kwargs.update(outcomes=target, predictions=Forecasts(target + rng.normal(scale=1e-3, size=target.shape), np.full((n, h, d), 0.5)),
                      invariant_bins=bins)
    return kwargs


def _renamed(kwargs: dict, regions: dict[str, str], tasks: dict[str, str] | None = None, world: str | None = None) -> dict:
    out = dict(kwargs)
    out["region_labels"] = tuple(RegionLabel(regions[x.region_name], x.axis) for x in kwargs["region_labels"])
    out["divergence_curves"] = tuple(RegionCurve(regions[c.region_name], c.curve) for c in kwargs["divergence_curves"])
    out["climatology"] = tuple(RegionClimatology(regions[t.region_name], t.bins) for t in kwargs["climatology"])
    if tasks:
        out["tasks"] = tuple(TaskSpec(tasks[t.name], t.kind, t.tolerance, t.horizon) for t in kwargs["tasks"])
    if world:
        out["world"] = world
    return out


def _permuted(kwargs: dict, seed: int) -> dict:
    """The same content with the order of the regions' curves/tables and of the tasks changed."""
    rng = np.random.default_rng(seed)
    out = dict(kwargs)
    for key in ("divergence_curves", "climatology", "tasks"):
        items = list(kwargs[key])
        out[key] = tuple(items[i] for i in rng.permutation(len(items)))
    return out


def _rename_in_verdict(record: dict, regions: dict[str, str], tasks: dict[str, str], world: str) -> dict:
    """The verdict with every region/task/world name replaced and every list put back in sorted-name order."""
    def fix(node):
        if isinstance(node, dict):
            out = {}
            for k, v in node.items():
                if k == "region" and isinstance(v, str):
                    out[k] = regions[v]
                elif k == "task" and isinstance(v, str):
                    out[k] = tasks[v]
                else:
                    out[k] = fix(v)
            return out
        if isinstance(node, list):
            return [fix(x) for x in node]
        return node

    renamed = fix(record)
    renamed["world"] = world
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


def _variants():
    """Random inputs and near-perfect ones (non-zero trust horizons, agreement holding)."""
    return [_random_kwargs(seed) for seed in range(8)] + [_random_kwargs(100 + seed, near_perfect=True) for seed in range(4)]


def test_the_inputs_really_reach_the_branches_a_random_input_never_does():
    records = [_verdict(JudgeInput(**kw)) for kw in _variants()]
    steps = [e["steps"] for r in records for e in r["trust_horizons"]["per_task"]]
    agrees = [e["agrees"] for r in records for e in r["climatology"]["per_task"]]
    assert 0 in steps and any(x > 0 for x in steps)
    assert True in agrees and False in agrees


def test_identical_content_judged_twice_from_separately_built_inputs_gives_identical_verdicts():
    first = _verdict(JudgeInput(**_random_kwargs(1)))
    second = _verdict(JudgeInput(**_random_kwargs(1)))  # a fresh build of the same content
    assert first == second


def test_tc_ju1_02_relabelling_regions_tasks_and_world_changes_only_the_labels_whichever_way_the_names_sort():
    for index, kwargs in enumerate(_variants()):
        for regions in ({"training": "zeta", "out-of-range": "alpha"}, {"training": "aaa", "out-of-range": "zzz"}, {"training": "out-of-range", "out-of-range": "training"}):
            tasks = {"alpha-task": "zz-first", "beta-task": "aa-second"}
            original = _verdict(JudgeInput(**kwargs))
            relabelled = _verdict(JudgeInput(**_renamed(kwargs, regions, tasks, world="a-different-world")))
            identity_regions, identity_tasks = {v: v for v in regions.values()}, {v: v for v in tasks.values()}
            _same_up_to_rounding(
                _rename_in_verdict(relabelled, identity_regions, identity_tasks, "a-different-world"),  # (only puts the lists in sorted order)
                _rename_in_verdict(original, regions, tasks, "a-different-world"),
            )


def test_tc_ju1_02_the_order_of_the_regions_curves_tables_and_tasks_makes_no_difference():
    for index, kwargs in enumerate(_variants()):
        original = _verdict(JudgeInput(**kwargs))
        for seed in range(4):
            permuted = _verdict(JudgeInput(**_permuted(kwargs, index * 10 + seed)))
            # tasks are reported in the order given, so compare after sorting by name
            a, b = (_rename_in_verdict(r, {"training": "training", "out-of-range": "out-of-range"}, {"alpha-task": "alpha-task", "beta-task": "beta-task"}, "lv") for r in (original, permuted))
            _same_up_to_rounding(a, b)


def test_tc_ju1_02_reordering_the_trials_consistently_changes_nothing_but_the_order_of_the_per_trial_lists():
    for index, kwargs in enumerate(_variants()):
        n = len(kwargs["region_labels"])
        order = np.random.default_rng(100 + index).permutation(n)
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

LEARNED_VOCABULARY = re.compile(
    r"^(torch|tensorflow|sklearn|scikit.*|keras|jax|onnx|pickle|joblib|state_dict|checkpoint|backprop|optimi[sz]er|gradient_descent|"
    r"train(ed|ing|er)?|epochs?|fit|fit_transform|predict|forward|weights?|biases|bias)$", re.IGNORECASE)
WEIGHT_FILE_SUFFIXES = {".pt", ".pth", ".npz", ".npy", ".pkl", ".pickle", ".h5", ".onnx", ".bin", ".safetensors", ".joblib"}
LOADERS = {"load", "loads", "fromfile", "memmap", "loadtxt", "genfromtxt"}


def learned_component_findings(source: str) -> list[str]:
    """Why this source looks like it holds or uses a learned component (empty = nothing found)."""
    from tests.gates.test_import_graph import run_judge_gate

    tree = ast.parse(source)
    findings = [f"static gate: {v}" for v in run_judge_gate(tree)]  # the same import allowlist as the purity lint (wmj.models etc. are refused)
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
            if node.attr in LOADERS:
                findings.append(f"loads data from outside: .{node.attr}")
        elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            names.add(node.name)
        elif isinstance(node, ast.arg) or isinstance(node, ast.keyword) and node.arg:
            names.add(node.arg)
        elif isinstance(node, ast.alias):
            names.add(node.name.split(".")[-1])
        elif isinstance(node, ast.Constant) and isinstance(node.value, bytes) and len(node.value) > 100:
            findings.append(f"a literal of {len(node.value)} bytes (embedded weights?)")
        elif isinstance(node, ast.List | ast.Tuple) and len(node.elts) > 40 and all(isinstance(e, ast.Constant) and isinstance(e.value, int | float) for e in node.elts):
            findings.append(f"a literal array of {len(node.elts)} numbers (embedded weights?)")
    findings += [f"learning vocabulary: {n}" for n in sorted(names) if LEARNED_VOCABULARY.match(n)]
    return findings


def test_tc_ju13_01_the_judge_directory_holds_only_source_no_weights_files():
    files = [p for p in JUDGE_DIR.rglob("*") if p.is_file() and "__pycache__" not in p.parts]
    assert files and all(p.suffix == ".py" for p in files), [p.name for p in files if p.suffix != ".py"]
    assert not [p for p in files if p.suffix in WEIGHT_FILE_SUFFIXES]


def test_tc_ju13_01_nothing_in_the_judges_source_looks_learned():
    for path in sorted(JUDGE_DIR.glob("*.py")):
        assert learned_component_findings(path.read_text()) == [], path.name


def test_tc_ju13_01_the_same_inspection_flags_planted_learned_components():
    weights = ", ".join(str(i / 7) for i in range(60))
    planted = {
        "a torch import": "import torch\n",
        "a model import": "from wmj.models.base import SeedSource\n",
        "a harness import": "import wmj.harness\n",
        "a predict method": "class Net:\n    def predict(self, x):\n        return x\n",
        "a forward method": "class Net:\n    def forward(self, x):\n        return x\n",
        "embedded weights": f"W = [{weights}]\n",
        "a weights loader": "import numpy as np\nw = np.load('w.npy')\n",
        "a trainer keyword": "def f(x):\n    return g(x, epochs=3)\n",
        "a weights variable": "import numpy as np\nweights = np.zeros(3)\n",
        "embedded bytes": f"B = {b'x' * 200!r}\n",
    }
    for label, source in planted.items():
        assert learned_component_findings(source), label
    assert learned_component_findings("import numpy as np\n\n\ndef f(x):\n    return np.sqrt(x)\n") == []
