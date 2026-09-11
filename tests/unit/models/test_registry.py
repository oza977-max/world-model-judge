"""Tests for wmj.models.registry — the name->factory table (models ADR-M1, MU-9).

The registry is the whole of MU-9's extensibility: a model exists to the
harness only if it registered, and `all_models()` hands the roster back in
a pinned sorted-name order (so `model_ref` indices are reproducible). This
chunk (P3-C02) builds the registration substrate only; the `pkgutil`
auto-discovery half of `all_models()` and its isolation test (TC-MU9-01)
are built at P6-C01, where the roster is first consumed.

Because the two baselines register themselves at import time into the same
module-global registry, tests that register throwaway names snapshot and
restore `_REGISTRY` so they never corrupt the real roster for a later test.
"""

from __future__ import annotations

import pytest

from wmj.models import registry
from wmj.models.registry import DuplicateModelError, all_models, register


@pytest.fixture
def clean_registry():
    """Snapshot the registry, yield, then restore it exactly — so a test
    that registers throwaway names leaves the real roster untouched."""
    snapshot = dict(registry._REGISTRY)
    try:
        yield
    finally:
        registry._REGISTRY.clear()
        registry._REGISTRY.update(snapshot)


def _factory(tag):
    def factory(ctx, seeds, training):
        return tag

    return factory


def test_register_then_all_models_returns_the_factory(clean_registry):
    f = _factory("m")
    register("zzz-test-model", f)
    assert all_models()["zzz-test-model"] is f


def test_all_models_is_in_sorted_name_order(clean_registry):
    # Register deliberately out of alphabetical order.
    register("zzz-b", _factory("b"))
    register("zzz-a", _factory("a"))
    register("zzz-c", _factory("c"))
    names = [n for n in all_models() if n.startswith("zzz-")]
    assert names == sorted(names)


def test_register_refuses_a_duplicate_name(clean_registry):
    """Phantom-gate: the fail-loud guard must actually fire, or a second
    model could silently overwrite the first (models ADR-M1 / McConnell)."""
    register("zzz-dup", _factory("first"))
    with pytest.raises(DuplicateModelError):
        register("zzz-dup", _factory("second"))


def test_all_models_returns_a_fresh_dict(clean_registry):
    register("zzz-frozen", _factory("x"))
    roster = all_models()
    roster["zzz-injected"] = _factory("not real")
    # Mutating the returned dict must not reach the real registry.
    assert "zzz-injected" not in all_models()


def test_both_baselines_are_registered_under_their_names():
    # Importing wmj.models.baselines runs its module-level register() calls.
    from wmj.models import baselines

    roster = all_models()
    assert roster["persistence"] is baselines.persistence_factory
    assert roster["linear"] is baselines.linear_factory


def test_baselines_sort_before_later_models_is_not_assumed():
    """persistence/linear sort LAST is a P6-C01 concern (the baseline
    pre-pass exists precisely because they sort late); here we only assert
    the contract this chunk owns — all_models() is sorted by name."""
    from wmj.models import baselines  # noqa: F401  (ensures baselines registered)

    names = list(all_models())
    assert names == sorted(names)
