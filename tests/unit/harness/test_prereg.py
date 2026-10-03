"""Tests for wmj.harness.prereg — the MU-6 / JU-11 certification (harness).

In plain words: before the judge may grade the two practice models, the
recipe, the written prediction and the pass/fail thresholds must be
*frozen* — locked by one commit that adds a small file, `prereg/FREEZE`.
These tests build throwaway git repositories and try every honest and
dishonest route we could think of: a recipe revised openly *before* the
freeze (fine), an honestly-dated edit *after* it (refused), freezing twice
(refused), freezing, deleting and re-adding (refused), hiding the freeze
on a side branch that was merged away (refused), moving the git tag
(ignored), and a freeze that comes after the run (refused). Each rule has
a test that would fail if the rule were removed (models ADR-M5,
design-review-010; TC-MU6-01/-04/-06/-07/-08/-09).

Every test uses real git as the external boundary; nothing owned is mocked.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pytest

from wmj.harness.prereg import (
    PreregContentError,
    PreregEntryMissingError,
    PreregError,
    PreregNotCommittedError,
    PreregNotFrozenError,
    PreregOrderingError,
    PreregRefrozenError,
    PreregWorldConstantError,
    check_prereg,
    check_recipe_world_constants,
    freeze_commit,
    read_matching_margin,
    within_matching_margin,
)


@dataclass
class _Model:
    name: str
    is_baseline: bool = False
    is_fixture: bool = False


def _git(repo, *args, date=None):
    env = {**os.environ}
    if date is not None:
        env.update({"GIT_AUTHOR_DATE": str(date), "GIT_COMMITTER_DATE": str(date)})
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True, env=env
    ).stdout


def _init_repo(repo):
    repo.mkdir(parents=True, exist_ok=True)
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@example.com")
    _git(repo, "config", "user.name", "t")


def _commit(repo, message, *, at):
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", message, date=f"@{at} +0000")


def _write_prereg(repo, *, recipe, prediction, thresholds="{}"):
    pdir = repo / "prereg"
    pdir.mkdir(exist_ok=True)
    (pdir / "recipe.md").write_text(recipe)
    (pdir / "prediction.md").write_text(prediction)
    (pdir / "thresholds.json").write_text(thresholds)


def _commit_prereg(repo, *, recipe, prediction, thresholds="{}", at):
    _write_prereg(repo, recipe=recipe, prediction=prediction, thresholds=thresholds)
    _commit(repo, "prereg", at=at)


def _freeze(repo, *, at, text="Frozen. Nothing in prereg/ changes after this commit.\n"):
    (repo / "prereg" / "FREEZE").write_text(text)
    _commit(repo, "freeze", at=at)
    return _git(repo, "rev-parse", "HEAD").strip()


PREREG_FILES = ["recipe.md", "prediction.md", "thresholds.json"]
KICK_KEYS = (
    "lv_kick_rate_per_s: 0.5\n"
    "lv_action_max: 0.1\n"
    "pendulum_kick_rate_per_s: 1.0\n"
    "pendulum_action_max: 1.0\n"
)
# A recipe naming both unrigged models and pinning the worlds' kick settings.
RECIPE = "matching_margin: 0.05\nmodels: direct, ensemble\n" + KICK_KEYS
PREDICTION = "We predict ensemble ranks at or above direct.\ndirect vs ensemble.\n"
MODELS = [_Model("direct"), _Model("ensemble")]


def _frozen_repo(tmp_path, *, prereg_at=1_000_000, freeze_at=1_100_000):
    repo = tmp_path / "repo"
    _init_repo(repo)
    _commit_prereg(repo, recipe=RECIPE, prediction=PREDICTION, at=prereg_at)
    sha = _freeze(repo, at=freeze_at)
    return repo, sha


# --- the happy path and what is returned (TC-MU6-07, check half) ---


def test_happy_path_passes_and_returns_the_freeze_commit(tmp_path):
    repo, freeze_sha = _frozen_repo(tmp_path)
    models = [*MODELS, _Model("persistence", is_baseline=True)]
    assert check_prereg(repo, PREREG_FILES, models, run_timestamp=2_000_000) == freeze_sha


def test_tc_mu6_07_the_returned_commit_is_the_one_that_added_freeze_not_the_recipes_first_add(
    tmp_path,
):
    # A recipe revised openly BEFORE the freeze is legitimate (the spec's
    # own remedy). The commit-of-record is the freeze, not the recipe's
    # original first commit.
    repo = tmp_path / "repo"
    _init_repo(repo)
    _commit_prereg(repo, recipe="matching_margin: 0.06\n", prediction=PREDICTION, at=1_000_000)
    first_add = _git(repo, "rev-parse", "HEAD").strip()
    _write_prereg(repo, recipe=RECIPE, prediction=PREDICTION)  # open revision, before the freeze
    _commit(repo, "revise recipe openly", at=1_050_000)
    freeze_sha = _freeze(repo, at=1_100_000)
    result = check_prereg(repo, PREREG_FILES, MODELS, run_timestamp=2_000_000)
    assert result == freeze_sha != first_add
    assert result == freeze_commit(repo)


def test_freeze_may_be_committed_together_with_the_prereg_files(tmp_path):
    repo = tmp_path / "repo"
    _init_repo(repo)
    _write_prereg(repo, recipe=RECIPE, prediction=PREDICTION)
    (repo / "prereg" / "FREEZE").write_text("Frozen.\n")
    _commit(repo, "everything at once", at=1_000_000)
    sha = _git(repo, "rev-parse", "HEAD").strip()
    assert check_prereg(repo, PREREG_FILES, MODELS, run_timestamp=2_000_000) == sha


# --- TC-MU6-01: the freeze must come before the run ---


def test_tc_mu6_01_refuses_when_the_freeze_comes_after_the_run(tmp_path):
    repo, _ = _frozen_repo(tmp_path, prereg_at=1_000_000, freeze_at=5_000_000)
    with pytest.raises(PreregOrderingError, match="TC-MU6-01"):
        check_prereg(repo, PREREG_FILES, MODELS, run_timestamp=4_000_000)


def test_tc_mu6_01_a_freeze_at_the_same_instant_as_the_run_is_refused(tmp_path):
    # strictly precede: equal is not before
    repo, _ = _frozen_repo(tmp_path, freeze_at=3_000_000)
    with pytest.raises(PreregOrderingError):
        check_prereg(repo, PREREG_FILES, MODELS, run_timestamp=3_000_000)


def test_the_prereg_files_may_be_older_than_the_freeze_but_the_freeze_is_what_is_timed(tmp_path):
    # Files committed long before the freeze are fine; the clock that must
    # precede the run is the freeze's.
    repo, _ = _frozen_repo(tmp_path, prereg_at=10, freeze_at=1_999_999)
    check_prereg(repo, PREREG_FILES, MODELS, run_timestamp=2_000_000)


# --- TC-MU6-04: nothing certified may change after the freeze ---


@pytest.mark.parametrize("victim", ["recipe.md", "prediction.md", "thresholds.json", "FREEZE"])
def test_tc_mu6_04_refuses_an_honestly_dated_edit_after_the_freeze(tmp_path, victim):
    """The Round 7 scenario, now against the freeze: an ordinary second
    commit (no --amend, no history rewrite, genuine date) that changes any
    certified file, including FREEZE itself, after the freeze."""
    repo, _ = _frozen_repo(tmp_path)
    path = repo / "prereg" / victim
    path.write_text(path.read_text() + "\n# tuned after seeing results\n")
    _commit(repo, "tune", at=1_500_000)
    with pytest.raises(PreregContentError, match=victim):
        check_prereg(repo, PREREG_FILES, MODELS, run_timestamp=2_000_000)


def test_an_edit_that_is_later_reverted_to_the_frozen_content_passes(tmp_path):
    # What is judged is the content in force; if it equals the frozen
    # content byte for byte, nothing moved.
    repo, freeze_sha = _frozen_repo(tmp_path)
    path = repo / "prereg" / "recipe.md"
    original = path.read_text()
    path.write_text(original + "oops\n")
    _commit(repo, "edit", at=1_200_000)
    path.write_text(original)
    _commit(repo, "revert", at=1_300_000)
    assert check_prereg(repo, PREREG_FILES, MODELS, run_timestamp=2_000_000) == freeze_sha


def test_refuses_a_dirty_uncommitted_prereg_file(tmp_path):
    repo, _ = _frozen_repo(tmp_path)
    (repo / "prereg" / "recipe.md").write_text(RECIPE + "uncommitted edit\n")
    # Not-committed is checked before content, so this can only be
    # PreregNotCommittedError; a reorder that called it content drift
    # would be caught here.
    with pytest.raises(PreregNotCommittedError):
        check_prereg(repo, PREREG_FILES, MODELS, run_timestamp=2_000_000)


def test_refuses_a_certified_file_deleted_from_the_working_tree(tmp_path):
    repo, _ = _frozen_repo(tmp_path)
    (repo / "prereg" / "thresholds.json").unlink()
    with pytest.raises(PreregNotCommittedError):
        check_prereg(repo, PREREG_FILES, MODELS, run_timestamp=2_000_000)


def test_a_file_added_after_the_freeze_is_not_part_of_the_frozen_set(tmp_path):
    # thresholds.json committed only after the freeze was never frozen.
    repo = tmp_path / "repo"
    _init_repo(repo)
    _write_prereg(repo, recipe=RECIPE, prediction=PREDICTION)
    (repo / "prereg" / "thresholds.json").unlink()
    _commit(repo, "prereg without thresholds", at=1_000_000)
    _freeze(repo, at=1_100_000)
    (repo / "prereg" / "thresholds.json").write_text("{}")
    _commit(repo, "thresholds, late", at=1_200_000)
    with pytest.raises(PreregNotFrozenError, match="thresholds.json"):
        check_prereg(repo, PREREG_FILES, MODELS, run_timestamp=2_000_000)


# --- TC-MU6-08: not frozen -> refused ---


def test_tc_mu6_08_refuses_when_there_is_no_freeze_file_at_all(tmp_path):
    repo = tmp_path / "repo"
    _init_repo(repo)
    _commit_prereg(repo, recipe=RECIPE, prediction=PREDICTION, at=1_000_000)
    with pytest.raises(PreregNotFrozenError, match="not frozen yet"):
        check_prereg(repo, PREREG_FILES, MODELS, run_timestamp=2_000_000)


def test_tc_mu6_08_a_freeze_file_that_was_never_committed_does_not_count(tmp_path):
    repo = tmp_path / "repo"
    _init_repo(repo)
    _commit_prereg(repo, recipe=RECIPE, prediction=PREDICTION, at=1_000_000)
    (repo / "prereg" / "FREEZE").write_text("Frozen.\n")  # written, not committed
    with pytest.raises(PreregNotFrozenError):
        check_prereg(repo, PREREG_FILES, MODELS, run_timestamp=2_000_000)


# --- TC-MU6-06: one freeze, ever ---


def test_tc_mu6_06_refuses_a_freeze_that_was_deleted_and_re_added(tmp_path):
    repo, _ = _frozen_repo(tmp_path)
    _git(repo, "rm", "-q", "prereg/FREEZE")
    _commit(repo, "unfreeze", at=1_200_000)
    (repo / "prereg" / "FREEZE").write_text("Frozen again, with a new recipe.\n")
    _commit(repo, "refreeze", at=1_300_000)
    with pytest.raises(PreregRefrozenError, match="TC-MU6-06"):
        check_prereg(repo, PREREG_FILES, MODELS, run_timestamp=2_000_000)


def test_tc_mu6_06_refuses_a_freeze_that_was_deleted_and_never_restored(tmp_path):
    repo, _ = _frozen_repo(tmp_path)
    _git(repo, "rm", "-q", "prereg/FREEZE")
    _commit(repo, "unfreeze", at=1_200_000)
    with pytest.raises(PreregRefrozenError):
        check_prereg(repo, PREREG_FILES, MODELS, run_timestamp=2_000_000)


def test_tc_mu6_06_refuses_a_freeze_renamed_away_and_back(tmp_path):
    # Executed before pinning: rename away and back gives 3 adds, 2 deletes
    # at the path.
    repo, _ = _frozen_repo(tmp_path)
    _git(repo, "mv", "prereg/FREEZE", "prereg/FREEZE.old")
    _commit(repo, "move away", at=1_200_000)
    _git(repo, "mv", "prereg/FREEZE.old", "prereg/FREEZE")
    _commit(repo, "move back", at=1_300_000)
    with pytest.raises(PreregRefrozenError):
        check_prereg(repo, PREREG_FILES, MODELS, run_timestamp=2_000_000)


def test_tc_mu6_06_refuses_a_freeze_hidden_on_a_side_branch_that_was_merged_away(tmp_path):
    """Executed before pinning: plain `git log` shows NOTHING for a freeze
    that was added and removed on a side branch whose merge resolved to
    "no FREEZE" — history simplification hides it. `--full-history` shows
    it, and this test is what keeps the flag in place."""
    repo = tmp_path / "repo"
    _init_repo(repo)
    _commit_prereg(repo, recipe=RECIPE, prediction=PREDICTION, at=1_000_000)
    main = _git(repo, "rev-parse", "--abbrev-ref", "HEAD").strip()
    _git(repo, "checkout", "-q", "-b", "side")
    (repo / "prereg" / "FREEZE").write_text("Frozen on a side branch.\n")
    _commit(repo, "freeze on side", at=1_100_000)
    _git(repo, "rm", "-q", "prereg/FREEZE")
    _commit(repo, "unfreeze on side", at=1_150_000)
    _git(repo, "checkout", "-q", main)
    (repo / "unrelated.txt").write_text("x")
    _commit(repo, "unrelated", at=1_200_000)
    _git(repo, "merge", "-q", "--no-edit", "-X", "ours", "side", date="@1250000 +0000")
    assert not (repo / "prereg" / "FREEZE").exists()
    with pytest.raises(PreregRefrozenError):
        freeze_commit(repo)


def test_tc_mu6_06_refuses_two_branches_that_each_added_a_freeze_and_were_merged(tmp_path):
    """Two adds with no delete at all: two side branches each froze the
    recipe (different declarations) and were merged. Only the add count
    catches it — the delete count is zero."""
    repo = tmp_path / "repo"
    _init_repo(repo)
    _commit_prereg(repo, recipe=RECIPE, prediction=PREDICTION, at=1_000_000)
    main = _git(repo, "rev-parse", "--abbrev-ref", "HEAD").strip()
    for n, at in (("a", 1_100_000), ("b", 1_150_000)):
        _git(repo, "checkout", "-q", "-b", f"freeze-{n}", main)
        (repo / "prereg" / "FREEZE").write_text(f"Frozen on branch {n}.\n")
        _commit(repo, f"freeze {n}", at=at)
    _git(repo, "checkout", "-q", "freeze-a")
    _git(repo, "merge", "-q", "--no-edit", "-X", "ours", "freeze-b", date="@1200000 +0000")
    assert (repo / "prereg" / "FREEZE").exists()
    with pytest.raises(PreregRefrozenError, match="2 time"):
        freeze_commit(repo)


def test_freeze_commit_returns_the_single_adding_commit(tmp_path):
    repo, sha = _frozen_repo(tmp_path)
    assert freeze_commit(repo) == sha


# --- the git tag is a label only ---


def test_a_prereg_freeze_tag_is_never_read(tmp_path):
    repo, freeze_sha = _frozen_repo(tmp_path)
    (repo / "later.txt").write_text("x")
    _commit(repo, "later", at=1_900_000)
    later = _git(repo, "rev-parse", "HEAD").strip()
    _git(repo, "tag", "prereg-freeze", later)  # points somewhere else
    assert check_prereg(repo, PREREG_FILES, MODELS, run_timestamp=2_000_000) == freeze_sha
    _git(repo, "tag", "-f", "prereg-freeze", _git(repo, "rev-list", "--max-parents=0", "HEAD").strip())
    assert check_prereg(repo, PREREG_FILES, MODELS, run_timestamp=2_000_000) == freeze_sha


# --- TC-MU6-03 / TC-MU6-05(a): per-model entries, baselines and fixtures exempt ---


def test_refuses_an_unrigged_model_absent_from_the_recipe(tmp_path):
    repo, _ = _frozen_repo(tmp_path)
    with pytest.raises(PreregEntryMissingError):
        check_prereg(repo, PREREG_FILES, [_Model("newcomer")], run_timestamp=2_000_000)


def test_baseline_and_fixture_are_exempt_from_the_entry_check(tmp_path):
    repo, _ = _frozen_repo(tmp_path)
    models = [
        _Model("persistence", is_baseline=True),
        _Model("fx-brittle", is_fixture=True),
        *MODELS,
    ]
    check_prereg(repo, PREREG_FILES, models, run_timestamp=2_000_000)


def test_refuses_a_name_that_is_only_a_substring_of_the_recipe(tmp_path):
    """'ir' must not be blessed because 'direct' contains i-r."""
    repo, _ = _frozen_repo(tmp_path)
    with pytest.raises(PreregEntryMissingError):
        check_prereg(repo, PREREG_FILES, [_Model("ir")], run_timestamp=2_000_000)


def test_hyphenated_model_name_is_recognised_as_a_whole_token(tmp_path):
    repo = tmp_path / "repo"
    _init_repo(repo)
    recipe = RECIPE + "We also register deep-direct as an unrigged contestant.\n"
    prediction = PREDICTION + "deep-direct is predicted mid-pack.\n"
    _commit_prereg(repo, recipe=recipe, prediction=prediction, at=1_000_000)
    _freeze(repo, at=1_100_000)
    check_prereg(repo, PREREG_FILES, [_Model("deep-direct")], run_timestamp=2_000_000)


def test_requires_recipe_and_prediction_among_certified_files(tmp_path):
    repo, _ = _frozen_repo(tmp_path)
    with pytest.raises(PreregError):
        check_prereg(repo, ["thresholds.json"], [_Model("direct")], run_timestamp=2_000_000)


# --- TC-MU6-09: the frozen recipe pins the worlds' kick settings ---


def test_tc_mu6_09_the_real_recipe_matches_the_world_constants():
    root = Path(__file__).resolve().parents[3]
    check_recipe_world_constants((root / "prereg" / "recipe.md").read_text(encoding="utf-8"))


@pytest.mark.parametrize(
    "line",
    [
        "lv_kick_rate_per_s: 0.5",
        "lv_action_max: 0.1",
        "pendulum_kick_rate_per_s: 1.0",
        "pendulum_action_max: 1.0",
    ],
)
def test_tc_mu6_09_a_recipe_value_that_differs_from_the_world_is_refused(line):
    key, value = line.split(": ")
    wrong = RECIPE.replace(line, f"{key}: {float(value) * 1.5}")
    assert wrong != RECIPE
    with pytest.raises(PreregWorldConstantError, match=key):
        check_recipe_world_constants(wrong)


def test_tc_mu6_09_a_missing_key_is_refused_not_ignored():
    without = "\n".join(
        line for line in RECIPE.splitlines() if not line.startswith("lv_action_max")
    )
    with pytest.raises(PreregWorldConstantError, match="lv_action_max"):
        check_recipe_world_constants(without)


def test_tc_mu6_09_comments_after_a_value_are_ignored():
    check_recipe_world_constants(RECIPE.replace("0.5\n", "0.5  # per second\n", 1))


def test_tc_mu6_09_is_enforced_through_check_prereg(tmp_path):
    repo = tmp_path / "repo"
    _init_repo(repo)
    _commit_prereg(
        repo, recipe=RECIPE.replace("lv_action_max: 0.1", "lv_action_max: 0.5"),
        prediction=PREDICTION, at=1_000_000,
    )
    _freeze(repo, at=1_100_000)
    with pytest.raises(PreregWorldConstantError, match="lv_action_max"):
        check_prereg(repo, PREREG_FILES, MODELS, run_timestamp=2_000_000)


# --- MU-5 matching margin (TC-MU5-01) ---


def test_read_matching_margin_parses_the_recipe(tmp_path):
    recipe = tmp_path / "recipe.md"
    recipe.write_text("epochs: 100\nmatching_margin: 0.05\ntraining_trajectories: 2000\n")
    assert read_matching_margin(recipe) == 0.05


def test_within_matching_margin_boundary():
    # at the margin: satisfied; over it: not yet satisfied (judging does not proceed)
    assert within_matching_margin(0.30, 0.34, margin=0.05) is True  # diff 0.04 <= 0.05
    assert within_matching_margin(0.30, 0.35, margin=0.05) is True  # diff 0.05 == 0.05
    assert within_matching_margin(0.30, 0.36, margin=0.05) is False  # diff 0.06 > 0.05
