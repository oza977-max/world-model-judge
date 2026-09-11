"""wmj.models.registry — the model registry (models ADR-M1, MU-9).

In plain words: the single table that maps a model's name to the function
that builds it. A model "exists" to the harness only if it is in here, and
adding a model is one file that calls `register(...)` at import time —
touching nothing else (MU-9's one-file extensibility). `all_models()` hands
the roster back in a fixed **sorted-name order**, so the `model_ref` indices
the judge envelope assigns from it are reproducible run to run (ADR-M1,
judge §5).

**Staging note (P3-C02 builds this; P6-C01 finishes it).** This chunk
builds the *registration substrate*: `register()`, `all_models()` in the
pinned sorted-name order, and `DuplicateModelError`. The **auto-discovery**
half of `all_models()` — the `importlib.invalidate_caches()` +
`pkgutil.iter_modules(wmj.models.__path__)` scan that imports every model
file so its module-level `register()` fires (cross-cutting ADR-003) — is
wired at **P6-C01**, where `harness.trials` first consumes an enumerated
roster and **TC-MU9-01** tests the isolation property. Nothing enumerates
the roster before P6-C01, so deferring discovery changes no behaviour
reachable today; `all_models()` below returns exactly the factories that
have registered so far (a model registers when its module is imported).

**Why discovery is deferred, not built here (backlog A11).** Cross-cutting
ADR-003 specifies discovery via `importlib`/`pkgutil`, but the *same*
ADR-003's models import-allowlist says `wmj/models/*` may import "numpy,
math, dataclasses, typing, `__future__`, `wmj.errors`, `hashlib`,
`wmj.models.base`, `wmj.models.registry` — nothing else, in any direction"
(enforced by the TC-NF6 gate). Those two clauses contradict:
`importlib`/`pkgutil` are required by the discovery clause and forbidden by
the allowlist clause. Building discovery here would force a carve-out in a
load-bearing security gate for a mechanism nothing consumes until P6-C01;
deferring it keeps this module inside the allowlist (`typing` + `wmj.errors`
only) and leaves the carve-out decision to P6-C01, where it is consumed and
testable. Recorded as A11 in `build/spec-corrections-backlog.md`.
"""

from __future__ import annotations

from typing import Callable

from wmj.errors import WmjError

# A model factory: factory(ctx, seeds, training) -> Model (models ADR-M1).
# Kept a loose Callable alias on purpose — the concrete WorldContext /
# SeedSource / TrainingData / Model types live in wmj.models.base, and a
# name->factory table needs none of them to do its one job.
Factory = Callable[..., object]


class DuplicateModelError(WmjError):
    """Raised when two models try to register the same name.

    Fails loudly (cross-cutting Error-Handling rule 1) rather than silently
    overwriting the first factory — a name collision is a build mistake, and
    a silent overwrite would make the roster depend on import order.
    """


_REGISTRY: dict[str, Factory] = {}


def register(name: str, factory: Factory) -> None:
    """Add a model's factory under `name` (a module-level call at import
    time, ADR-M1). Refuses a duplicate name with `DuplicateModelError`."""
    if name in _REGISTRY:
        raise DuplicateModelError(
            f"model name {name!r} is already registered — two models cannot "
            f"share a name (models ADR-M1); registration is name-keyed and the "
            f"first registration is never silently overwritten"
        )
    _REGISTRY[name] = factory


def all_models() -> dict[str, Factory]:
    """The registered factories, name-keyed, in **sorted-name order**.

    The ordering is part of the contract (ADR-M1): `model_ref` indices are
    assigned from this order, so they are reproducible within one fixed
    roster. A fresh dict is returned on every call, so a caller that mutates
    the result cannot corrupt the registry.

    Auto-discovery (the `pkgutil` scan that imports every model file so its
    `register()` fires) is added at P6-C01 — see the module docstring. Until
    then this returns exactly the models whose modules have been imported.
    """
    return dict(sorted(_REGISTRY.items()))
