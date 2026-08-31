"""Tests for wmj.models.base: component_key and SeedSource.

Covers cross-cutting ADR-002 rule 2, pinned construction:
component_key(*parts) -> blake2b(":".join(parts), digest_size=8) as one
big-endian int in a 1-tuple; SeedSource wraps a run_seed + optional
my_name and derives numpy Generators from it, content-addressed so
adding a component never shifts another's stream (TC-NF1-03), two
independent implementations of the pinned text converge (TC-NF1-04),
a ':' in any part is rejected (TC-NF1-07), a non-str part is rejected
(TC-NF1-08), and distinct seed purposes give distinct streams (TC-MU7-02).
"""

from __future__ import annotations

import hashlib

import pytest

from wmj.models.base import SeedKeyError, SeedSource, component_key


def test_component_key_no_shift_on_roster_change():
    """TC-NF1-03: adding a new component never shifts an existing one's key."""
    roster = ["direct", "ensemble", "fx-brittle", "linear", "persistence"]
    before = {name: component_key(name, "lv", "weights") for name in roster}

    roster_with_new_model = ["aardvark", *roster]
    after = {
        name: component_key(name, "lv", "weights") for name in roster_with_new_model
    }

    for name in roster:
        assert before[name] == after[name]


def test_component_key_two_implementations_converge():
    """TC-NF1-04: an independent implementation of the pinned text matches."""

    def reference_component_key(*parts: str) -> tuple[int]:
        joined = ":".join(parts).encode("utf-8")
        digest = hashlib.blake2b(joined, digest_size=8).digest()
        return (int.from_bytes(digest, "big"),)

    cases = [
        ("direct", "lv", "training", "weights"),
        ("ensemble", "pendulum", "eval", "shuffle"),
        ("persistence",),
    ]
    for parts in cases:
        assert component_key(*parts) == reference_component_key(*parts)


def test_component_key_rejects_colon_in_a_part():
    """TC-NF1-07: a ':' inside any part is rejected — the join is not
    injective across a part boundary otherwise."""
    with pytest.raises(SeedKeyError):
        component_key("a:b", "c")

    # the legitimate, colon-free call still succeeds
    component_key("lv", "training", "eval-starts")


def test_component_key_colon_collision_is_real_without_the_guard():
    """The guarded collision, demonstrated directly: 'a:b' + 'c' and
    'a' + 'b:c' join to the identical text if the ':' were allowed."""
    assert ":".join(["a:b", "c"]) == ":".join(["a", "b:c"])


def test_component_key_rejects_non_str_part():
    """TC-NF1-08: a non-str part is rejected — the old str(p) coercion
    collapsed component_key(1, "a") and component_key("1", "a") to the
    same joined text and therefore the same seed stream."""
    with pytest.raises(SeedKeyError):
        component_key(1, "a")

    with pytest.raises(SeedKeyError):
        component_key("ensemble", 3)

    # the caller's own str() form is the sanctioned way to pass an index
    component_key("ensemble", "3")


def test_seed_source_rng_for_is_deterministic():
    seeds_a = SeedSource(run_seed=20260825, my_name=None)
    seeds_b = SeedSource(run_seed=20260825, my_name=None)
    draw_a = seeds_a.rng_for("direct", "weights").random()
    draw_b = seeds_b.rng_for("direct", "weights").random()
    assert draw_a == draw_b


def test_seed_source_rng_uses_own_name():
    seeds = SeedSource(run_seed=20260825, my_name="direct")
    assert seeds.rng("weights").random() == seeds.rng_for(
        "direct", "weights"
    ).random()


def test_seed_source_rng_for_purpose_streams_are_independent():
    """TC-MU7-02: train/eval/benchmark starts for the same (world, region)
    are distinct streams, not accidental collisions."""
    seeds = SeedSource(run_seed=20260825, my_name=None)
    train = seeds.rng_for("lv", "training", "train-starts").random()
    eval_ = seeds.rng_for("lv", "training", "eval-starts").random()
    benchmark = seeds.rng_for("lv", "training", "benchmark-starts").random()
    assert len({train, eval_, benchmark}) == 3
