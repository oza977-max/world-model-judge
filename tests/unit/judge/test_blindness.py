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
    kwargs["tasks"] = (TaskSpec("lv-control", "control", 0.1, 3), TaskSpec("lv-planning", "planning", 0.3, 5))
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


def _same_up_to_rounding(a, b, path="", tolerance=1e-12) -> None:
    """Equal structure and text; counts and flags exactly; floats to `tolerance` (summing trials in another order rounds differently)."""
    assert type(a) is type(b), path
    if isinstance(a, dict):
        assert a.keys() == b.keys(), path
        for k in a:
            _same_up_to_rounding(a[k], b[k], f"{path}/{k}", tolerance)
    elif isinstance(a, list | tuple):
        assert len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b, strict=True)):
            _same_up_to_rounding(x, y, f"{path}[{i}]", tolerance)
    elif isinstance(a, float):
        assert a == b or abs(a - b) <= tolerance * max(1.0, abs(a), abs(b)), (path, a, b)
    else:
        assert a == b, (path, a, b)


def _variants():
    """Random inputs and near-perfect ones (non-zero trust horizons, agreement holding), plus structural variants:
    no natural cycle (the pendulum's case), out-of-region axes, and a third task."""
    variants = [_random_kwargs(seed) for seed in range(8)] + [_random_kwargs(100 + seed, near_perfect=True) for seed in range(4)]
    no_cycle = _random_kwargs(200, near_perfect=True)
    no_cycle["natural_cycle_length"] = None
    axes = _random_kwargs(201)
    axes["region_labels"] = tuple(RegionLabel(x.region_name, ("action", "both", "state")[i % 3] if x.region_name != "training" else None) for i, x in enumerate(axes["region_labels"]))
    three_tasks = _random_kwargs(202, near_perfect=True)
    three_tasks["tasks"] = (*three_tasks["tasks"], TaskSpec("lv-extra", "planning", 0.05, 4))
    return [*variants, no_cycle, axes, three_tasks]


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
            tasks = {t.name: f"{'zz' if i == 0 else 'aa'}-renamed-{i}" for i, t in enumerate(kwargs["tasks"])}  # also flips the sort order
            original = _verdict(JudgeInput(**kwargs))
            relabelled = _verdict(JudgeInput(**_renamed(kwargs, regions, tasks, world="renamed-world")))
            identity_regions, identity_tasks = {v: v for v in regions.values()}, {v: v for v in tasks.values()}
            _same_up_to_rounding(
                _rename_in_verdict(relabelled, identity_regions, identity_tasks, "renamed-world"),  # (only puts the lists in sorted order)
                _rename_in_verdict(original, regions, tasks, "renamed-world"),
                tolerance=0.0,  # relabelling must not move a single bit
            )


def test_tasks_are_reported_in_the_order_given_and_regions_in_sorted_name_order():
    kwargs = _random_kwargs(3)
    first = _verdict(JudgeInput(**kwargs))
    swapped = _verdict(JudgeInput(**dict(kwargs, tasks=tuple(reversed(kwargs["tasks"])))))
    for record, expected in ((first, ["lv-control", "lv-planning"]), (swapped, ["lv-planning", "lv-control"])):
        entries = record["skill"]["per_task_region"]
        assert [e["task"] for e in entries] == [t for t in expected for _ in range(2)]
        assert [e["region"] for e in entries[:2]] == ["out-of-range", "training"]


def test_tc_ju1_02_the_order_of_the_regions_curves_tables_and_tasks_makes_no_difference():
    for index, kwargs in enumerate(_variants()):
        original = _verdict(JudgeInput(**kwargs))
        for seed in range(4):
            permuted = _verdict(JudgeInput(**_permuted(kwargs, index * 10 + seed)))
            # tasks are reported in the order given, so compare after sorting by name
            same_regions, same_tasks = {"training": "training", "out-of-range": "out-of-range"}, {t.name: t.name for t in kwargs["tasks"]}
            a, b = (_rename_in_verdict(r, same_regions, same_tasks, "lv") for r in (original, permuted))
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

LEARNED_TOKENS = {
    "torch", "tensorflow", "sklearn", "scikit", "keras", "jax", "onnx", "pickle", "joblib", "checkpoint", "backprop", "optimizer",
    "optimiser", "gradient", "epoch", "epochs", "trained", "trainer", "fit", "predict", "forward", "weights", "biases", "layer",
    "layers", "hidden", "neuron", "neurons", "loss", "lr", "learning",
}
WEIGHT_FILE_SUFFIXES = {".pt", ".pth", ".npz", ".npy", ".pkl", ".pickle", ".h5", ".onnx", ".bin", ".safetensors", ".joblib"}
LOADERS = {"load", "loads", "fromfile", "fromstring", "memmap", "loadtxt", "genfromtxt"}
FITTERS = {"lstsq", "polyfit", "curve_fit", "pinv"}


def _tokens(identifier: str) -> set[str]:
    spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", identifier)
    return {t.lower() for t in spaced.split("_") if t}


def _number(node: ast.AST) -> bool:
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub | ast.UAdd):
        return _number(node.operand)
    return isinstance(node, ast.Constant) and isinstance(node.value, int | float) and not isinstance(node.value, bool)


def _count_numbers(node: ast.AST) -> int:
    """Numbers written out literally inside a list/tuple/set/dict (any nesting, signs allowed)."""
    if _number(node):
        return 1
    if isinstance(node, ast.List | ast.Tuple | ast.Set):
        return sum(_count_numbers(e) for e in node.elts)
    if isinstance(node, ast.Dict):
        return sum(_count_numbers(v) for v in node.values if v is not None)
    return 0


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
        elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            names.add(node.name)
        elif isinstance(node, ast.arg) or isinstance(node, ast.keyword) and node.arg:
            names.add(node.arg)
        elif isinstance(node, ast.alias):
            names.add(node.name.split(".")[-1])
        if isinstance(node, ast.Attribute | ast.Name) and getattr(node, "attr", getattr(node, "id", "")) in LOADERS | FITTERS:
            findings.append(f"loads data or fits a model: {getattr(node, 'attr', getattr(node, 'id', ''))}")
        if isinstance(node, ast.ImportFrom) and any(a.name in LOADERS | FITTERS for a in node.names):
            findings.append(f"imports a loader or fitter: {[a.name for a in node.names]}")
        if isinstance(node, ast.Constant) and isinstance(node.value, bytes | str) and len(node.value) > 200 and len(set(node.value)) > 12 and not (isinstance(node.value, str) and any(c.isspace() for c in node.value)):
            findings.append(f"a literal of {len(node.value)} characters (encoded weights?)")
        if isinstance(node, ast.List | ast.Tuple | ast.Set | ast.Dict) and _count_numbers(node) > 40:
            findings.append(f"a literal holding {_count_numbers(node)} numbers (embedded weights?)")
    findings += [f"learning vocabulary: {n}" for n in sorted(names) if _tokens(n) & LEARNED_TOKENS]
    return sorted(set(findings))


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
        "signed embedded weights": "W = [" + ", ".join(str(((-1) ** i) * i / 7) for i in range(60)) + "]\n",
        "nested embedded weights": "W = [[" + ", ".join(str(i / 3) for i in range(30)) + "], [" + ", ".join(str(-i / 3) for i in range(30)) + "]]\n",
        "weights in a dict": "W = {" + ", ".join(f"'k{i}': {i / 5}" for i in range(60)) + "}\n",
        "encoded weights": "B = '" + "AbCdEfGhIjKlMnOpQrStUvWxYz0123456789+/" * 8 + "'\n",
        "a loader imported by name": "from numpy import load\nw = load('w.npy')\n",
        "a fromstring loader": "import numpy as np\nw = np.fromstring('1 2 3', sep=' ')\n",
        "a fit at judge time": "import numpy as np\nc = np.polyfit([1, 2, 3], [1, 2, 3], 1)\n",
        "a learning-rate variable": "learning_rate = 0.1\n",
        "a model-weights variable": "model_weights = 1\n",
        "a hidden layer": "hidden_layer = 1\n",
        "a camel-case trainer": "class NetTrainer:\n    pass\n",
        "a loss variable": "loss = 1\n",
        "a weights loader": "import numpy as np\nw = np.load('w.npy')\n",
        "a trainer keyword": "def f(x):\n    return g(x, epochs=3)\n",
        "a weights variable": "import numpy as np\nweights = np.zeros(3)\n",
        "embedded bytes": f"B = {bytes(range(33, 253))!r}\n",
    }
    for label, source in planted.items():
        assert learned_component_findings(source), label
    assert learned_component_findings("import numpy as np\n\n\ndef f(x):\n    return np.sqrt(x)\n") == []
    assert learned_component_findings("import numpy as np\n\nz = np.array([1.0, -2.0, 3.0])\nm = np.average(z)\n") == []  # a short signed literal is fine
