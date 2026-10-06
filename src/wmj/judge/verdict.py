"""wmj.judge.verdict — the output door: a verdict that cannot exist half-finished (JU-9).

In plain words: the verdict is the product. It has nine required groups — skill scores,
error against the world's own drift, calibration and sharpness, exception counts against the
pre-registered bands, the per-trial points behind those counts, the climatology check, the
trust horizons, what was never tested, and the seven limitations. This module makes a
`Verdict` that is *impossible to hold unless all nine are present*: it refuses (raises) if
any group is missing, empty, null, contains a number that is not finite, is keyed
differently from the canonical rule (every entry names its task / region / step), or if the
exception counts disagree with the per-trial points they are supposed to be made of.
Nothing identifying a model or a fixture can be in it: those facts belong to the harness
envelope that wraps a verdict (judge spec §5). The fixed text (limitations, not-tested) is
inserted here from `limitations.py` and checked word for word.
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


def _fail(message: str) -> VerdictIncompleteError:
    return VerdictIncompleteError(message)


def _plain(value, path: str):
    """A deep, JSON-able copy: numpy converted, every float finite, every dict key a clean string."""
    if isinstance(value, np.ndarray):
        return _plain(value.tolist(), path)
    if isinstance(value, np.generic):
        return _plain(value.item(), path)
    if value is None or isinstance(value, (bool, str, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise _fail(f"{path} holds a number that is not finite ({value!r}); a verdict never carries NaN or infinity")
        return value
    if isinstance(value, (list, tuple)):
        return [_plain(v, f"{path}[{i}]") for i, v in enumerate(value)]
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise _fail(f"{path} has a key that is not a non-empty string: {key!r}")
            if key.endswith(BANNED_KEY_SUFFIXES):
                raise _fail(f"{path}.{key}: suffix-encoded axes are banned — use explicit key fields (judge spec §5)")
            out[key] = _plain(item, f"{path}.{key}")
        return out
    raise _fail(f"{path} holds a {type(value).__name__}, which a verdict cannot carry")


def _check_block(name: str, block) -> dict:
    if block is None:
        raise _fail(f"the {name} group is missing (null)")
    if not isinstance(block, dict) or not block:
        raise _fail(f"the {name} group must be a non-empty mapping, got {type(block).__name__}")
    block = _plain(block, name)
    list_name, key_fields = BLOCK_KEYS[name]
    entries = block.get(list_name)
    if not isinstance(entries, list) or not entries:
        raise _fail(f"{name}.{list_name} is missing or empty — a verdict never carries an empty group")
    seen = set()
    for i, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise _fail(f"{name}.{list_name}[{i}] must be a mapping")
        for field in key_fields:
            value = entry.get(field)
            if value is None or value == "":
                raise _fail(f"{name}.{list_name}[{i}] has no '{field}' (every entry carries its task/region/step keys)")
        key = tuple(entry[f] for f in key_fields)
        if key in seen:
            raise _fail(f"{name}.{list_name} has two entries for the same key {key}")
        seen.add(key)
    return block


def _check_error_vs_horizon(block: dict) -> None:
    dt = block.get("dt")
    if isinstance(dt, bool) or not isinstance(dt, (int, float)) or not dt > 0:
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


def _check_trials_and_exceptions(trials: dict, exceptions: dict) -> None:
    by_key = {}
    for i, entry in enumerate(trials["per_task"]):
        lengths = set()
        for field in TRIAL_ARRAYS:
            series = entry.get(field)
            if not isinstance(series, list) or not series:
                raise _fail(f"trials.per_task[{i}].{field} is missing or empty")
            lengths.add(len(series))
        if len(lengths) != 1:
            raise _fail(f"trials.per_task[{i}]: the per-trial arrays differ in length")
        if not all(isinstance(flag, bool) for flag in entry["is_exception"]):
            raise _fail(f"trials.per_task[{i}].is_exception must hold true/false for every trial")
        by_key[(entry["task"], entry["region"], entry["horizon_step"])] = sum(entry["is_exception"])
    for i, entry in enumerate(exceptions["per_task"]):
        key = (entry["task"], entry["region"], entry["horizon_step"])
        if key not in by_key:
            raise _fail(f"exceptions.per_task[{i}] {key} has no matching trials entry")
        observed = entry.get("observed")
        if isinstance(observed, bool) or not isinstance(observed, int):
            raise _fail(f"exceptions.per_task[{i}].observed must be a whole number")
        if observed != by_key[key]:
            raise _fail(
                f"exceptions.per_task[{i}] {key}: observed={observed} but trials.is_exception sums to "
                f"{by_key[key]} — they are one fact and must agree (TC-JU9-03)"
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
        if not isinstance(self.world, str) or not self.world:
            raise _fail("the verdict needs the world's name")
        checked = {name: _check_block(name, block) for name, block in self._blocks().items()}
        _check_error_vs_horizon(checked["error_vs_horizon"])
        _check_trials_and_exceptions(checked["trials"], checked["exceptions"])
        for name, block in checked.items():
            object.__setattr__(self, name, block)
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
            out[name] = _plain(block, name)  # a fresh deep copy: editing the result changes nothing
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
