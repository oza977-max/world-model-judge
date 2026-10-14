"""Tests for wmj.harness.climatology — the reference table and the measured bins (worlds §5, judge ADR-J5).

In plain words: the table says, for each of sixteen ranges of the world's conserved quantity, what
states the world typically visits there. These tests use a tiny made-up world where every number can
be worked out by hand (the conserved quantity is just the first coordinate, and the dynamics do
nothing), and check: sixteen equal-sized ranges with open ends, the right averages and spreads,
the same answer for the same seed, only null actions used, a refusal when the reference is
degenerate (the very defect the original one-trajectory recipe had), the bin lookup at the edges,
and that a kick which changes the conserved quantity changes the bin (it is re-measured per step,
never frozen).
"""

from __future__ import annotations

import numpy as np
import pytest

from wmj.harness.climatology import (
    ClimatologyError,
    build_climatology_table,
    judge_tables,
    measure_invariant_bins,
    truth_mean_abs_z,
)
from wmj.models.base import SeedSource
from wmj.worlds.base import RegionSpec, Task


class ToyWorld:
    """State (x, y). Null action: nothing changes. The conserved quantity is x."""

    a = 1
    scale = np.array([1.0, 1.0])

    def __init__(self, box=((0.0, 1.0), (10.0, 20.0)), dynamics=None):
        self._box = np.array(box)
        self._dynamics = dynamics
        self.seen_actions = []

    def transition_batch(self, states, actions):
        self.seen_actions.append(np.array(actions))
        out = np.array(states)
        return out if self._dynamics is None else self._dynamics(out, actions)

    def conserved(self, state):
        return float(state[0])

    def regions(self):
        return RegionSpec(self._box, np.array([[-1.0, 1.0]]), ())

    def tasks(self):
        return (Task("t", "control", 0.1, 5),)


def _seeds(seed=7):
    return SeedSource(run_seed=seed, my_name=None)


def _table(world=None, **kw):
    kw = {"n_starts": 160, "steps_per_start": 5, "min_samples": 5, **kw}
    return build_climatology_table("toy", world or ToyWorld(), _seeds(), **kw)


def test_sixteen_equal_population_bins_with_open_ends_and_contiguous_edges():
    t = _table()
    bins = t["bins"]
    assert t["n_bins"] == 16 and t["bin_method"] == "equal-population" and len(bins) == 16
    assert bins[0]["invariant_range"][0] == -np.inf and bins[-1]["invariant_range"][1] == np.inf
    for a, b in zip(bins, bins[1:], strict=False):  # noqa: RUF007
        assert a["invariant_range"][1] == b["invariant_range"][0] and a["invariant_range"][0] < a["invariant_range"][1]
    assert sum(b["n_samples"] for b in bins) == 160 * 5
    assert max(b["n_samples"] for b in bins) - min(b["n_samples"] for b in bins) <= 5  # 800 samples / 16 bins = 50 each (starts repeat across steps)


def test_each_bin_holds_exactly_the_samples_in_its_range_with_their_mean_and_population_sd():
    world = ToyWorld()
    t = _table(world)
    rng = _seeds().rng_for("toy", "training", "climatology-starts")
    starts = rng.uniform([0.0, 10.0], [1.0, 20.0], size=(160, 2))
    samples = np.repeat(starts, 5, axis=0)  # nothing moves, so every start is sampled 5 times
    for entry in t["bins"]:
        lo, hi = entry["invariant_range"]
        inside = samples[(samples[:, 0] >= lo) & (samples[:, 0] < hi)]
        assert entry["n_samples"] == len(inside)
        assert entry["mean"] == pytest.approx(inside.mean(axis=0).tolist(), rel=1e-12)
        assert entry["sd"] == pytest.approx(inside.std(axis=0).tolist(), rel=1e-12)


def test_the_table_is_the_same_for_the_same_seed_and_different_for_another():
    assert _table() == _table()
    assert _table() != build_climatology_table("toy", ToyWorld(), _seeds(8), n_starts=160, steps_per_start=5, min_samples=5)


def test_only_null_actions_are_ever_applied_and_every_start_is_a_sample_at_every_step():
    world = ToyWorld()
    _table(world)
    assert len(world.seen_actions) == 5 and all(np.all(a == 0.0) and a.shape == (160, 1) for a in world.seen_actions)


def test_the_reference_run_length_is_reported():
    t = _table()
    assert (t["world"], t["reference_starts"], t["steps_per_start"], t["reference_run_steps"]) == ("toy", 160, 5, 800)
    assert t["reference_region"] == "training"


def test_a_table_can_be_built_from_another_declared_regions_starts_and_an_undeclared_one_is_refused():
    class TwoRegions(ToyWorld):
        def regions(self):
            from wmj.worlds.base import OutRegion

            far = OutRegion("far", "state", np.array([[5.0, 6.0], [30.0, 40.0]]), np.array([[-1.0, 1.0]]))
            return RegionSpec(self._box, np.array([[-1.0, 1.0]]), (far,))

    t = _table(TwoRegions(), region_name="far")
    assert t["reference_region"] == "far"
    assert t["bins"][0]["mean"][0] >= 5.0 and t["bins"][-1]["mean"][1] <= 40.0 and t["bins"][3]["mean"][0] >= 5.0
    assert t != _table(TwoRegions())
    with pytest.raises(ClimatologyError, match="declared"):
        _table(TwoRegions(), region_name="nowhere")


def test_a_degenerate_reference_where_the_conserved_quantity_never_varies_is_refused():
    class Frozen(ToyWorld):
        def conserved(self, state):
            return 3.0  # the original recipe's failure: one level, so no sixteen ranges

    with pytest.raises(ClimatologyError, match="distinct"):
        _table(Frozen())


def test_too_few_samples_in_a_bin_is_refused():
    with pytest.raises(ClimatologyError, match="samples"):
        _table(min_samples=51)  # 800 / 16 = 50 per bin


def test_a_bin_with_no_spread_in_some_quantity_is_refused():
    class Flat(ToyWorld):
        def regions(self):
            return RegionSpec(np.array([[0.0, 1.0], [5.0, 5.0]]), np.array([[-1.0, 1.0]]), ())

    with pytest.raises(ClimatologyError, match="spread"):
        _table(Flat())


def test_judge_tables_are_accepted_by_the_judge_for_every_region_name_and_carry_the_same_bins():
    t = _table()
    tables = judge_tables(t, ("training", "out-of-range"))
    assert [x.region_name for x in tables] == ["training", "out-of-range"]
    assert all(len(x.bins) == 16 and x.bins[0].invariant_lo == -np.inf and x.bins[-1].invariant_hi == np.inf for x in tables)
    assert tables[0].bins[3].mean.tolist() == t["bins"][3]["mean"] and tables[0].bins[3].n_samples == t["bins"][3]["n_samples"]


def test_the_bin_of_a_state_follows_its_invariant_with_a_value_on_an_edge_in_the_upper_bin():
    t = _table()
    edges = [b["invariant_range"][1] for b in t["bins"][:-1]]
    states = np.zeros((1, len(edges) * 3 + 2, 2))
    values = []
    for e in edges:
        values += [float(np.nextafter(e, -np.inf)), e, float(np.nextafter(e, np.inf))]
    values += [-1e9, 1e9]
    states[0, :, 0] = values
    got = measure_invariant_bins(ToyWorld(), states, t)[0].tolist()
    expected = []
    for k in range(len(edges)):
        expected += [k, k + 1, k + 1]
    expected += [0, 15]
    assert got == expected
    assert measure_invariant_bins(ToyWorld(), states, t).dtype == np.int64


def test_a_kick_that_changes_the_invariant_changes_the_measured_bin_it_is_not_frozen_at_the_start():
    t = _table()
    first_edge, last_edge = t["bins"][0]["invariant_range"][1], t["bins"][-1]["invariant_range"][0]
    steady = np.zeros((1, 6, 2))
    steady[0, :, 0] = first_edge - 0.01  # stays in bin 0
    kicked = steady.copy()
    kicked[0, 3:, 0] = last_edge + 0.01  # a kick at step 4 moves it into the top bin
    assert measure_invariant_bins(ToyWorld(), steady, t)[0].tolist() == [0] * 6
    assert measure_invariant_bins(ToyWorld(), kicked, t)[0].tolist() == [0, 0, 0, 15, 15, 15]


def test_truth_agreement_is_the_mean_absolute_standardised_gap_of_true_states_from_their_bins():
    t = _table()
    states = np.zeros((2, 3, 2))
    bins = np.zeros((2, 3), dtype=np.int64)
    bins[:, :] = 4
    b = t["bins"][4]
    mean, sd = np.array(b["mean"]), np.array(b["sd"])
    states[:, :, :] = mean + sd * np.array([1.0, 3.0])  # one sd off in x, three in y: mean |z| = 2
    assert truth_mean_abs_z(t, states, bins) == pytest.approx(2.0, rel=1e-12)


def test_a_value_exactly_on_a_range_edge_belongs_to_the_upper_range_in_the_built_table_too():
    class Coarse(ToyWorld):
        def conserved(self, state):
            return float(np.floor(state[0] * 400.0) / 400.0)  # many exact ties, so edges land on sample values

    world = Coarse()
    t = _table(world)
    rng = _seeds().rng_for("toy", "training", "climatology-starts")
    starts = rng.uniform([0.0, 10.0], [1.0, 20.0], size=(160, 2))
    values = np.repeat(np.floor(starts[:, 0] * 400.0) / 400.0, 5)
    on_an_edge = 0
    for entry in t["bins"]:
        lo, hi = entry["invariant_range"]
        assert entry["n_samples"] == int(np.sum((values >= lo) & (values < hi)))  # [lo, hi): the edge value is in the upper bin
        on_an_edge += int(np.sum(values == lo))
    assert on_an_edge > 0  # the test really exercises ties on an edge


def test_edges_that_are_not_strictly_increasing_are_refused():
    class Lumpy(ToyWorld):
        def conserved(self, state):
            x = state[0]  # 16+ distinct values, but the value 0 holds about nine samples in ten
            return 0.0 if x < 0.9 else float(1 + np.floor((x - 0.9) * 300.0))

    with pytest.raises(ClimatologyError, match="distinct"):
        build_climatology_table("toy", Lumpy(), _seeds(), n_starts=2000, steps_per_start=1, min_samples=1)


def test_a_bin_with_exactly_the_required_number_of_samples_is_accepted_and_one_fewer_refused():
    smallest = min(b["n_samples"] for b in _table()["bins"])
    assert _table(min_samples=smallest)["n_bins"] == 16
    with pytest.raises(ClimatologyError, match="samples"):
        _table(min_samples=smallest + 1)
