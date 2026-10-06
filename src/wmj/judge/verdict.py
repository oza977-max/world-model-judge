"""wmj.judge.verdict — the output door: a verdict that cannot exist half-finished (JU-9).

In plain words: the verdict is the product. It has nine required groups — skill scores,
error against the world's own drift, calibration and sharpness, exception counts against the
pre-registered bands, the per-trial points behind those counts, the climatology check, the
trust horizons, what was never tested, and the seven limitations. This module makes a
`Verdict` that cannot be built unless all nine are present: it refuses (raises) if any
group is missing, empty or null, if any entry lacks one of its required fields or carries one
the spec does not define (so no identity field can ride along), if a number is not finite,
if the groups do not cover the same (task, region) pairs, or if the exception counts disagree
with the per-trial points they are made of. Once built, a verdict is read-only: its groups
are frozen (mappings that refuse writes, tuples instead of lists), so an inconsistent verdict
cannot be produced by editing a good one. Facts that identify a model or a fixture belong to
the harness envelope that wraps a verdict (judge spec §5). The fixed text (limitations,
not-tested) is inserted here from `limitations.py` and checked word for word.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from wmj.judge.errors import VerdictIncompleteError
from wmj.judge.limitations import JU10_DISCLOSURES, NOT_TESTED

VERDICT_SCHEMA = "wmj-verdict/1"

# block name -> (name of its entry list, key fields every entry must carry) — judge spec §5's keying table
BLOCK_KEYS: dict[str, tuple[str, tuple[str, ...]]] = {
    "skill": ("per_task_region", ("task", "region")),
    "error_vs_horizon": ("per_region", ("region",)),
    "calibration": ("per_task", ("task", "region")),
    "sharpness": ("per_task", ("task", "region")),
    "exceptions": ("per_task", ("task", "region", "horizon_step")),
    "trials": ("per_task", ("task", "region", "horizon_step")),
    "climatology": ("per_task", ("task", "region")),
    "trust_horizons": ("per_task", ("task", "region")),
}
BANNED_KEY_SUFFIXES = ("_in", "_out", "_in_region", "_out_region")  # suffix-encoded axes are banned (§5)
TRIAL_ARRAYS = ("outcome_distance", "band_lo", "band_hi", "is_exception")
MAX_DEPTH = 40
BANDS = ("green", "amber", "red")

# Value fields of each block's entries (judge spec §5's schema): name -> kind. The key fields
# (task/region/horizon_step) are checked separately. A field not listed here is refused, so
# nothing the spec does not define — in particular nothing identifying a model — can ride along.
# Kinds: num, posint, nonnegint, bool, str, band, numlist, intlist, boollist, list, dict; a
# trailing "?" lets the value be null (only where the spec says so: no switch step, no natural cycle).
ENTRY_FIELDS: dict[str, dict[str, str]] = {
    "skill": {"vs_persistence": "num", "vs_linear": "num", "crps": "num"},
    "error_vs_horizon": {"steps": "intlist", "median_error": "numlist", "divergence_reference": "numlist"},
    "calibration": {"levels": "numlist", "coverage": "numlist", "n_trials": "posint", "per_dimension": "list"},
    "sharpness": {"mean_width_90": "num"},
    "exceptions": {
        "n_trials": "posint", "expected": "num", "observed": "nonnegint", "band": "band",
        "low_side_sharpness_flag": "bool",
    },
    "trials": {
        "distance_unit": "str", "outcome_distance": "numlist", "band_lo": "numlist", "band_hi": "numlist",
        "is_exception": "boollist",
    },
    "climatology": {"switch_step": "nonnegint?", "agreement_mean_abs_z": "num?", "agrees": "bool?"},
    "trust_horizons": {"tolerance": "num", "steps": "nonnegint", "world_time": "num", "natural_units": "str?"},
}
OPTIONAL_ENTRY_FIELDS: dict[str, dict[str, str]] = {"exceptions": {"bands": "dict"}}
BLOCK_LEVEL_FIELDS: dict[str, tuple[str, ...]] = {"error_vs_horizon": ("dt",)}
TASK_REGION_BLOCKS = ("skill", "calibration", "sharpness", "climatology", "trust_horizons")


def _fail(message: str) -> VerdictIncompleteError:
    return VerdictIncompleteError(message)


def _is_int(x) -> bool:
    return type(x) is int


def _is_num(x) -> bool:
    return type(x) in (int, float)


def _plain(value, path: str, depth: int = 0):
    """A deep, JSON-able copy: numpy converted, every float finite, every dict key a clean string."""
    if depth > MAX_DEPTH:
        raise _fail(f"{path} is nested more than {MAX_DEPTH} levels deep (or refers to itself)")
    if isinstance(value, np.ndarray):
        return _plain(value.tolist(), path, depth + 1)
    if isinstance(value, np.generic):
        kind = value.dtype.kind
        if kind == "b":
            return bool(value)
        if kind in "iu":
            return int(value)
        if kind == "f":
            return _plain(float(value), path, depth + 1)
        raise _fail(f"{path} holds a numpy {value.dtype} value, which a verdict cannot carry")
    if value is None or type(value) in (bool, str, int):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise _fail(f"{path} holds a number that is not finite ({value!r}); a verdict never carries NaN or infinity")
        return value
    if isinstance(value, (list, tuple)):
        return [_plain(v, f"{path}[{i}]", depth + 1) for i, v in enumerate(value)]
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            if type(key) is not str or not key.strip():
                raise _fail(f"{path} has a key that is not a non-blank string: {key!r}")
            if key.endswith(BANNED_KEY_SUFFIXES):
                raise _fail(f"{path}.{key}: suffix-encoded axes are banned — use explicit key fields (judge spec §5)")
            out[key] = _plain(item, f"{path}.{key}", depth + 1)
        return out
    raise _fail(f"{path} holds a {type(value).__name__}, which a verdict cannot carry")


class _FrozenMap(dict):
    """A mapping that refuses writes: a verdict's groups cannot be edited once built."""

    def _refuse(self, *args, **kwargs):
        raise TypeError("a verdict is read-only: its groups cannot be changed after it is built")

    __setitem__ = __delitem__ = clear = pop = popitem = setdefault = update = __ior__ = _refuse

    def __reduce__(self):
        return (_FrozenMap, (dict(self),))


def _freeze(value):
    if isinstance(value, dict):
        return _FrozenMap({k: _freeze(v) for k, v in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(v) for v in value)
    return value


def _thaw(value):
    if isinstance(value, dict):
        return {k: _thaw(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_thaw(v) for v in value]
    return value


def _kind_ok(kind: str, value) -> bool:
    if kind.endswith("?"):
        return value is None or _kind_ok(kind[:-1], value)
    if kind == "num":
        return _is_num(value)
    if kind == "posint":
        return _is_int(value) and value >= 1
    if kind == "nonnegint":
        return _is_int(value) and value >= 0
    if kind == "bool":
        return type(value) is bool
    if kind == "str":
        return type(value) is str and bool(value.strip())
    if kind == "band":
        return value in BANDS and type(value) is str
    if kind == "numlist":
        return isinstance(value, list) and bool(value) and all(_is_num(v) for v in value)
    if kind == "intlist":
        return isinstance(value, list) and bool(value) and all(_is_int(v) for v in value)
    if kind == "boollist":
        return isinstance(value, list) and bool(value) and all(type(v) is bool for v in value)
    if kind == "list":
        return isinstance(value, list) and bool(value)
    if kind == "dict":
        return isinstance(value, dict) and bool(value)
    raise AssertionError(kind)


def _check_key_field(name: str, i: int, field: str, value) -> None:
    if field == "horizon_step":
        if not _is_int(value) or value < 0:
            raise _fail(f"{name}[{i}].horizon_step must be a whole number >= 0, got {value!r}")
    elif type(value) is not str or not value.strip():
        raise _fail(f"{name}[{i}] has no '{field}' (every entry carries its task/region/step keys)")


def _check_block(name: str, block) -> dict:
    if block is None:
        raise _fail(f"the {name} group is missing (null)")
    if not isinstance(block, dict) or not block:
        raise _fail(f"the {name} group must be a non-empty mapping, got {type(block).__name__}")
    block = _plain(block, name)
    list_name, key_fields = BLOCK_KEYS[name]
    allowed_block = {list_name, *BLOCK_LEVEL_FIELDS.get(name, ())}
    if set(block) - allowed_block:
        raise _fail(f"{name} has fields the spec does not define: {sorted(set(block) - allowed_block)}")
    entries = block.get(list_name)
    if not isinstance(entries, list) or not entries:
        raise _fail(f"{name}.{list_name} is missing or empty — a verdict never carries an empty group")
    required = ENTRY_FIELDS[name]
    optional = OPTIONAL_ENTRY_FIELDS.get(name, {})
    seen = set()
    for i, entry in enumerate(entries):
        where = f"{name}.{list_name}[{i}]"
        if not isinstance(entry, dict):
            raise _fail(f"{where} must be a mapping")
        for field in key_fields:
            if field not in entry:
                raise _fail(f"{where} has no '{field}' (every entry carries its task/region/step keys)")
            _check_key_field(f"{name}.{list_name}", i, field, entry[field])
        unknown = set(entry) - set(key_fields) - set(required) - set(optional)
        if unknown:
            raise _fail(f"{where} has fields the spec does not define: {sorted(unknown)}")
        for field, kind in {**required, **optional}.items():
            if field not in entry:
                if field in required:
                    raise _fail(f"{where} is missing its required field '{field}'")
                continue
            if not _kind_ok(kind, entry[field]):
                raise _fail(f"{where}.{field} must be {kind} (got {entry[field]!r}) — a field that could not be computed aborts the run")
        key = tuple(entry[f] for f in key_fields)
        if key in seen:
            raise _fail(f"{name}.{list_name} has two entries for the same key {key}")
        seen.add(key)
    return block


def _check_calibration(block: dict) -> None:
    for i, entry in enumerate(block["per_task"]):
        levels, coverage = entry["levels"], entry["coverage"]
        if len(levels) != len(coverage):
            raise _fail(f"calibration.per_task[{i}]: levels and coverage differ in length")
        if not all(0.0 < v < 1.0 for v in levels) or not all(0.0 <= v <= 1.0 for v in coverage):
            raise _fail(f"calibration.per_task[{i}]: levels must lie in (0, 1) and coverage in [0, 1]")


def _check_cross_block(blocks: dict) -> None:
    """Every group covers the same (task, region) pairs; error_vs_horizon covers the same regions."""
    pairs = {name: {(e["task"], e["region"]) for e in blocks[name][BLOCK_KEYS[name][0]]} for name in TASK_REGION_BLOCKS}
    reference = pairs["skill"]
    for name in (*TASK_REGION_BLOCKS[1:], "exceptions", "trials"):
        if name in ("exceptions", "trials"):
            found = {(e["task"], e["region"]) for e in blocks[name]["per_task"]}
        else:
            found = pairs[name]
        if found != reference:
            raise _fail(
                f"{name} covers different (task, region) pairs than skill: missing {sorted(reference - found)}, "
                f"extra {sorted(found - reference)} — a verdict is complete or it is not issued (JU-9)"
            )
    regions = {r for _, r in reference}
    covered = {e["region"] for e in blocks["error_vs_horizon"]["per_region"]}
    if covered != regions:
        raise _fail(f"error_vs_horizon covers regions {sorted(covered)} but the other groups cover {sorted(regions)}")


def _check_error_vs_horizon(block: dict) -> None:
    dt = block.get("dt")
    if not _is_num(dt) or not dt > 0:
        raise _fail("error_vs_horizon.dt must be a positive number (reporting draws world time from it)")
    for i, entry in enumerate(block["per_region"]):
        lengths = set()
        for field in ("steps", "median_error", "divergence_reference"):
            series = entry.get(field)
            if not isinstance(series, list) or not series:
                raise _fail(f"error_vs_horizon.per_region[{i}].{field} is missing or empty")
            lengths.add(len(series))
        if len(lengths) != 1:
            raise _fail(f"error_vs_horizon.per_region[{i}]: steps, median_error and divergence_reference differ in length")
        if entry["steps"][0] != 0:
            raise _fail(f"error_vs_horizon.per_region[{i}].steps must start at 0 (the shared step-zero origin)")
        if any(b <= a for a, b in zip(entry["steps"], entry["steps"][1:])):
            raise _fail(f"error_vs_horizon.per_region[{i}].steps must be strictly increasing")


def _check_trials_and_exceptions(trials: dict, exceptions: dict) -> None:
    by_key = {}
    for i, entry in enumerate(trials["per_task"]):
        lengths = {len(entry[field]) for field in TRIAL_ARRAYS}
        if len(lengths) != 1:
            raise _fail(f"trials.per_task[{i}]: the per-trial arrays differ in length")
        if any(lo > hi for lo, hi in zip(entry["band_lo"], entry["band_hi"])):
            raise _fail(f"trials.per_task[{i}]: band_lo must not exceed band_hi")
        if any(v < 0 for v in entry["outcome_distance"]):
            raise _fail(f"trials.per_task[{i}]: a distance is never negative")
        by_key[(entry["task"], entry["region"], entry["horizon_step"])] = (sum(entry["is_exception"]), lengths.pop())
    for i, entry in enumerate(exceptions["per_task"]):
        key = (entry["task"], entry["region"], entry["horizon_step"])
        if key not in by_key:
            raise _fail(f"exceptions.per_task[{i}] {key} has no matching trials entry")
        observed_sum, n_points = by_key[key]
        if entry["observed"] != observed_sum:
            raise _fail(
                f"exceptions.per_task[{i}] {key}: observed={entry['observed']} but trials.is_exception sums to "
                f"{observed_sum} — they are one fact and must agree (TC-JU9-03)"
            )
        if entry["n_trials"] != n_points:
            raise _fail(
                f"exceptions.per_task[{i}] {key}: n_trials={entry['n_trials']} but the trials entry holds "
                f"{n_points} points — the header count and the points are one fact"
            )
    if {(e["task"], e["region"], e["horizon_step"]) for e in exceptions["per_task"]} != set(by_key):
        raise _fail("every trials entry needs exactly one matching exceptions entry")


@dataclass(frozen=True, slots=True)
class Verdict:
    """The pure verdict (judge spec §5). Construction is the check: an incomplete one cannot exist."""

    world: str
    skill: dict
    error_vs_horizon: dict
    calibration: dict
    sharpness: dict
    exceptions: dict
    trials: dict
    climatology: dict
    trust_horizons: dict
    not_tested: tuple
    limitations: tuple

    def __post_init__(self) -> None:
        if type(self.world) is not str or not self.world.strip():
            raise _fail("the verdict needs the world's name")
        checked = {name: _check_block(name, block) for name, block in self._blocks().items()}
        _check_error_vs_horizon(checked["error_vs_horizon"])
        _check_calibration(checked["calibration"])
        _check_trials_and_exceptions(checked["trials"], checked["exceptions"])
        _check_cross_block(checked)
        for name, block in checked.items():
            object.__setattr__(self, name, _freeze(block))
        if self.limitations is None or tuple(self.limitations) != JU10_DISCLOSURES:
            raise _fail("limitations must be the seven JU-10 disclosures, verbatim and in order (ADR-J7)")
        if self.not_tested is None or tuple(self.not_tested) != NOT_TESTED:
            raise _fail("not_tested must be the fixed not-tested list (judge spec §5)")
        object.__setattr__(self, "limitations", tuple(self.limitations))
        object.__setattr__(self, "not_tested", tuple(self.not_tested))

    def _blocks(self) -> dict:
        """The eight computed groups by name (explicit, so no name lookup by string is needed)."""
        return {
            "skill": self.skill,
            "error_vs_horizon": self.error_vs_horizon,
            "calibration": self.calibration,
            "sharpness": self.sharpness,
            "exceptions": self.exceptions,
            "trials": self.trials,
            "climatology": self.climatology,
            "trust_horizons": self.trust_horizons,
        }

    def to_dict(self) -> dict:
        """The record in schema order, as plain JSON-able data (a deep copy — editing it changes nothing)."""
        out = {"schema": VERDICT_SCHEMA, "world": self.world}
        for name, block in self._blocks().items():
            out[name] = _thaw(block)  # a fresh deep copy of plain data: editing the result changes nothing
        out["not_tested"] = list(self.not_tested)
        out["limitations"] = list(self.limitations)
        return out


def assemble_verdict(
    *, world, skill, error_vs_horizon, calibration, sharpness, exceptions, trials, climatology, trust_horizons
) -> Verdict:
    """Put the eight computed groups together with the fixed text, or refuse (TC-JU9-01/02).

    Every group is required by keyword, so a group cannot be left out by accident, and
    `Verdict` itself refuses any that is missing, null, empty, non-finite or mis-keyed.
    """
    return Verdict(
        world=world,
        skill=skill,
        error_vs_horizon=error_vs_horizon,
        calibration=calibration,
        sharpness=sharpness,
        exceptions=exceptions,
        trials=trials,
        climatology=climatology,
        trust_horizons=trust_horizons,
        not_tested=NOT_TESTED,
        limitations=JU10_DISCLOSURES,
    )
