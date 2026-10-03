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
import struct
import subprocess
import zlib
from dataclasses import dataclass
from pathlib import Path

import pytest

from wmj.harness.prereg import (
    PreregContentError,
    PreregEntryMissingError,
    PreregError,
    PreregHistoryError,
    PreregLineEndingError,
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
    _git(repo, "config", "core.autocrlf", "false")  # raw bytes, whatever the machine default


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
    "no FREEZE" — history simplification hides it. The commit-by-commit walk
    shows it."""
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


def test_the_certified_set_is_fixed_and_does_not_depend_on_the_callers_list(tmp_path):
    """A caller that forgets thresholds.json (the most important judged-
    against content) must not get a certificate for tuned thresholds
    (independent review, P3-C10 pass 1)."""
    repo, _ = _frozen_repo(tmp_path)
    (repo / "prereg" / "thresholds.json").write_text('{"tuned": true}')
    _commit(repo, "tune thresholds", at=1_500_000)
    for files in (PREREG_FILES, ["recipe.md", "prediction.md"], [], ["thresholds.json"]):
        with pytest.raises(PreregContentError, match="thresholds.json"):
            check_prereg(repo, files, MODELS, run_timestamp=2_000_000)


@pytest.mark.parametrize("bad", ["../x", "/etc/passwd", "sub/recipe.md", ""])
def test_extra_file_names_must_be_plain_names_inside_prereg(tmp_path, bad):
    repo, _ = _frozen_repo(tmp_path)
    with pytest.raises(PreregError, match="plain file name"):
        check_prereg(repo, [*PREREG_FILES, bad], MODELS, run_timestamp=2_000_000)


# --- independent review, P3-C10 pass 1: git history must be trustworthy ---


def test_a_shallow_clone_is_refused_not_certified(tmp_path):
    """A shallow clone shows its boundary commit as having ADDED every file,
    so a tuned file is compared with itself and passes. Repro: freeze, then
    an honestly-dated tune, then `git clone --depth=1`."""
    repo, _ = _frozen_repo(tmp_path)
    (repo / "prereg" / "thresholds.json").write_text('{"tuned": true}')
    _commit(repo, "tune", at=1_500_000)
    clone = tmp_path / "clone"
    subprocess.run(
        ["git", "clone", "-q", "--depth=1", f"file://{repo}", str(clone)], check=True,
        capture_output=True,
    )
    with pytest.raises(PreregHistoryError, match="shallow"):
        check_prereg(clone, PREREG_FILES, MODELS, run_timestamp=2_000_000)


def test_a_replace_ref_cannot_make_history_say_the_tuned_tree_was_frozen(tmp_path):
    """`git replace <freeze> <tuned-commit>` is local, never pushed, and
    leaves every sha valid in the public repo — yet makes `git show` serve
    tuned bytes for the freeze. Every git call must ignore replace refs."""
    repo, freeze_sha = _frozen_repo(tmp_path)
    (repo / "prereg" / "thresholds.json").write_text('{"tuned": true}')
    _commit(repo, "tune", at=1_500_000)
    tuned = _git(repo, "rev-parse", "HEAD").strip()
    _git(repo, "replace", freeze_sha, tuned)
    with pytest.raises(PreregContentError, match="thresholds.json"):
        check_prereg(repo, PREREG_FILES, MODELS, run_timestamp=2_000_000)


def test_a_grafts_file_is_refused(tmp_path):
    """`.git/info/grafts` rewrites parentage without touching any commit.
    Executed: git 2.43 still honours it and `--no-replace-objects` does not
    stop it, so the file's mere presence is refused."""
    repo, _ = _frozen_repo(tmp_path)
    head = _git(repo, "rev-parse", "HEAD").strip()
    (repo / ".git" / "info").mkdir(exist_ok=True)
    (repo / ".git" / "info" / "grafts").write_text(f"{head}\n")
    with pytest.raises(PreregHistoryError, match="grafts"):
        check_prereg(repo, PREREG_FILES, MODELS, run_timestamp=2_000_000)


def test_ambient_git_dir_cannot_redirect_the_check_to_another_repository(tmp_path, monkeypatch):
    """A caller inside a git hook has GIT_DIR set; the check must still read
    the repository it was given."""
    repo, freeze_sha = _frozen_repo(tmp_path)
    other = tmp_path / "other"
    _init_repo(other)
    monkeypatch.setenv("GIT_DIR", str(other / ".git"))
    monkeypatch.setenv("GIT_WORK_TREE", str(other))
    assert check_prereg(repo, PREREG_FILES, MODELS, run_timestamp=2_000_000) == freeze_sha


def test_the_repo_argument_must_be_the_top_level_of_the_repository(tmp_path):
    repo, _ = _frozen_repo(tmp_path)
    with pytest.raises(PreregError, match="top level"):
        check_prereg(repo / "prereg", PREREG_FILES, MODELS, run_timestamp=2_000_000)


# --- independent review, P3-C10 pass 2: history that `git log` cannot show ---


def _evil_merge(repo, branch, *, at, edit):
    """Merge `branch` with --no-commit, apply `edit()`, commit: a merge whose
    own change to prereg/ is invisible to `git log --diff-filter` (git shows
    no diff for a merge commit)."""
    _git(repo, "merge", "--no-ff", "--no-commit", "-q", branch)
    edit()
    _commit(repo, f"merge {branch}", at=at)


def _side_branch_commit(repo, name, main, *, at):
    _git(repo, "checkout", "-qb", name, main)
    (repo / f"{name}.txt").write_text(name)
    _commit(repo, name, at=at)
    _git(repo, "checkout", "-q", main)


def test_tc_mu6_06_refuses_a_freeze_lifted_and_redone_inside_merge_commits(tmp_path):
    """The pass-2 attack: freeze, run, dislike the result, lift the freeze and
    tune the recipe — both inside merge commits (git log shows no diff for a
    merge, so add/delete counts from `log --diff-filter` stay at 1 and 0) —
    then freeze again with an ordinary commit. Honestly dated, no rewrite."""
    repo = tmp_path / "repo"
    _init_repo(repo)
    _commit_prereg(repo, recipe=RECIPE, prediction=PREDICTION, at=1_000)
    main = _git(repo, "rev-parse", "--abbrev-ref", "HEAD").strip()
    _side_branch_commit(repo, "x", main, at=1_010)
    (repo / "m.txt").write_text("m")
    _commit(repo, "m1", at=1_020)
    freeze = repo / "prereg" / "FREEZE"
    _evil_merge(repo, "x", at=1_100, edit=lambda: freeze.write_text("frozen v1\n"))
    _side_branch_commit(repo, "y", main, at=1_300)
    (repo / "m2.txt").write_text("m2")
    _commit(repo, "m2", at=1_310)

    def lift_and_tune():
        freeze.unlink()
        (repo / "prereg" / "recipe.md").write_text(RECIPE.replace("0.05", "0.09"))

    _evil_merge(repo, "y", at=1_400, edit=lift_and_tune)
    freeze.write_text("frozen v2\n")
    _commit(repo, "freeze again", at=1_500)
    with pytest.raises(PreregRefrozenError, match="2 time"):
        check_prereg(repo, PREREG_FILES, MODELS, run_timestamp=2_000)


def test_a_freeze_lifted_inside_a_merge_commit_and_never_restored_is_refused_as_refrozen(tmp_path):
    """The delete half on its own: freeze, then a merge that quietly removes
    FREEZE. The refusal must name the lifted freeze, not just a missing file."""
    repo = tmp_path / "repo"
    _init_repo(repo)
    _commit_prereg(repo, recipe=RECIPE, prediction=PREDICTION, at=1_000)
    main = _git(repo, "rev-parse", "--abbrev-ref", "HEAD").strip()
    freeze = repo / "prereg" / "FREEZE"
    freeze.write_text("Frozen.\n")
    _commit(repo, "freeze", at=1_100)
    _side_branch_commit(repo, "y", main, at=1_200)
    (repo / "m.txt").write_text("m")
    _commit(repo, "m", at=1_210)
    _evil_merge(repo, "y", at=1_300, edit=freeze.unlink)
    with pytest.raises(PreregRefrozenError, match="deleted 1"):
        check_prereg(repo, PREREG_FILES, MODELS, run_timestamp=2_000)


def test_a_directory_named_freeze_is_not_a_freeze(tmp_path):
    repo = tmp_path / "repo"
    _init_repo(repo)
    _commit_prereg(repo, recipe=RECIPE, prediction=PREDICTION, at=1_000)
    (repo / "prereg" / "FREEZE").mkdir()
    (repo / "prereg" / "FREEZE" / "note.txt").write_text("not a file\n")
    _commit(repo, "a directory called FREEZE", at=1_100)
    with pytest.raises(PreregNotFrozenError, match="not frozen yet"):
        check_prereg(repo, PREREG_FILES, MODELS, run_timestamp=2_000)


def test_a_freeze_made_inside_a_merge_commit_is_recognised_as_the_freeze(tmp_path):
    repo = tmp_path / "repo"
    _init_repo(repo)
    _commit_prereg(repo, recipe=RECIPE, prediction=PREDICTION, at=1_000)
    main = _git(repo, "rev-parse", "--abbrev-ref", "HEAD").strip()
    _side_branch_commit(repo, "x", main, at=1_010)
    (repo / "m.txt").write_text("m")
    _commit(repo, "m1", at=1_020)
    _evil_merge(
        repo, "x", at=1_100, edit=lambda: (repo / "prereg" / "FREEZE").write_text("Frozen.\n")
    )
    merge_sha = _git(repo, "rev-parse", "HEAD").strip()
    assert check_prereg(repo, PREREG_FILES, MODELS, run_timestamp=2_000) == merge_sha


def test_a_root_commit_freeze_is_found_even_if_log_hides_root_diffs(tmp_path):
    repo = tmp_path / "repo"
    _init_repo(repo)
    _git(repo, "config", "log.showRoot", "false")
    _write_prereg(repo, recipe=RECIPE, prediction=PREDICTION)
    (repo / "prereg" / "FREEZE").write_text("Frozen at the root.\n")
    _commit(repo, "all at once", at=1_000_000)
    sha = _git(repo, "rev-parse", "HEAD").strip()
    assert check_prereg(repo, PREREG_FILES, MODELS, run_timestamp=2_000_000) == sha


def _corrupt_commit_graph_parent(repo, child, new_parent):
    """Rewrite `child`'s first parent inside .git/objects/info/commit-graph —
    changing what git reports as history without touching any commit or ref."""
    _git(repo, "commit-graph", "write", "--reachable")
    graph = repo / ".git" / "objects" / "info" / "commit-graph"
    data = bytearray(graph.read_bytes())
    n_chunks = data[6]
    table = {}
    offset = 8
    for _ in range(n_chunks + 1):
        table[bytes(data[offset : offset + 4])] = struct.unpack(">Q", data[offset + 4 : offset + 12])[0]
        offset += 12
    count = struct.unpack(">256I", data[table[b"OIDF"] : table[b"OIDF"] + 1024])[255]
    oids = [data[table[b"OIDL"] + 20 * i : table[b"OIDL"] + 20 * i + 20].hex() for i in range(count)]
    position = table[b"CDAT"] + 36 * oids.index(child) + 20
    data[position : position + 4] = struct.pack(">I", oids.index(new_parent))
    graph.write_bytes(bytes(data))


def test_a_tampered_commit_graph_cannot_rewrite_parentage(tmp_path):
    """Same class as grafts and replace refs: a binary cache git trusts over
    the commits themselves. history shows freeze1, unfreeze+tune, freeze2; the
    tampered graph says freeze2's parent is the base commit."""
    repo = tmp_path / "repo"
    _init_repo(repo)
    _commit_prereg(repo, recipe=RECIPE, prediction=PREDICTION, at=1_000)
    base = _git(repo, "rev-parse", "HEAD").strip()
    (repo / "prereg" / "FREEZE").write_text("v1\n")
    _commit(repo, "freeze1", at=1_100)
    _git(repo, "rm", "-q", "prereg/FREEZE")
    (repo / "prereg" / "recipe.md").write_text(RECIPE.replace("0.05", "0.09"))
    _commit(repo, "unfreeze and tune", at=1_500)
    (repo / "prereg" / "FREEZE").write_text("v2\n")
    _commit(repo, "freeze2", at=1_600)
    freeze2 = _git(repo, "rev-parse", "HEAD").strip()
    _corrupt_commit_graph_parent(repo, freeze2, base)
    # the lie is in place: plain `git log` now skips the unfreeze-and-tune
    assert _git(repo, "log", "--format=%s").splitlines() == ["freeze2", "prereg"]
    with pytest.raises(PreregRefrozenError):
        check_prereg(repo, PREREG_FILES, MODELS, run_timestamp=2_000)


def test_one_freeze_ever_is_counted_across_every_branch_not_only_the_checked_out_one(tmp_path):
    """A branch forked from before the freeze can tune the recipe and freeze
    again; checking out that branch hides the first freeze from HEAD's own
    ancestry. Nothing is rewritten and every commit is honestly dated."""
    repo = tmp_path / "repo"
    _init_repo(repo)
    _commit_prereg(repo, recipe=RECIPE, prediction=PREDICTION, at=1_000)
    main = _git(repo, "rev-parse", "--abbrev-ref", "HEAD").strip()
    _freeze(repo, at=1_100)
    _git(repo, "checkout", "-q", "-b", "tuned", f"{main}~1")
    (repo / "prereg" / "recipe.md").write_text(RECIPE.replace("0.05", "0.09"))
    _commit(repo, "tune after seeing the result", at=1_300)
    (repo / "prereg" / "FREEZE").write_text("Frozen again, on a branch.\n")
    _commit(repo, "freeze the tuned recipe", at=1_400)
    with pytest.raises(PreregRefrozenError, match="2 time"):
        check_prereg(repo, PREREG_FILES, MODELS, run_timestamp=2_000)
    _git(repo, "checkout", "-q", main)
    with pytest.raises(PreregRefrozenError, match="2 time"):
        check_prereg(repo, PREREG_FILES, MODELS, run_timestamp=2_000)


def test_a_freeze_that_exists_only_on_another_branch_is_not_this_checkouts_freeze(tmp_path):
    repo = tmp_path / "repo"
    _init_repo(repo)
    _commit_prereg(repo, recipe=RECIPE, prediction=PREDICTION, at=1_000)
    main = _git(repo, "rev-parse", "--abbrev-ref", "HEAD").strip()
    _git(repo, "checkout", "-q", "-b", "elsewhere")
    _freeze(repo, at=1_100)
    _git(repo, "checkout", "-q", main)
    with pytest.raises(PreregNotFrozenError, match="checked-out"):
        check_prereg(repo, PREREG_FILES, MODELS, run_timestamp=2_000)


def test_a_stash_is_not_a_second_freeze_or_a_lifted_one(tmp_path):
    repo, freeze_sha = _frozen_repo(tmp_path)
    (repo / "prereg" / "FREEZE").unlink()
    _git(repo, "stash", "-q")  # the stash commit lacks FREEZE; the checkout gets it back
    assert (repo / "prereg" / "FREEZE").exists()
    assert check_prereg(repo, PREREG_FILES, MODELS, run_timestamp=2_000_000) == freeze_sha


def test_the_freeze_time_not_head_time_is_what_is_compared_with_the_run(tmp_path):
    # A perfectly honest commit made after the run (or a later re-check of an
    # old run) must not be refused because HEAD is newer than the run.
    repo, freeze_sha = _frozen_repo(tmp_path, freeze_at=1_100_000)
    (repo / "notes.txt").write_text("written after the run")
    _commit(repo, "later work", at=3_000_000)
    assert check_prereg(repo, PREREG_FILES, MODELS, run_timestamp=2_000_000) == freeze_sha


def test_a_freeze_lifted_in_a_merge_that_only_one_parent_had_it_is_still_a_delete(tmp_path):
    """A merge of a branch forked BEFORE the freeze: one parent has FREEZE,
    the other never did. The merge drops it. Then the recipe is restored and
    FREEZE re-added by merging a branch that still carries it — so the add is
    hidden too. Only 'some parent had it' counts the delete."""
    repo = tmp_path / "repo"
    _init_repo(repo)
    _commit_prereg(repo, recipe=RECIPE, prediction=PREDICTION, at=1_000)
    main = _git(repo, "rev-parse", "--abbrev-ref", "HEAD").strip()
    _side_branch_commit(repo, "x", main, at=1_010)
    freeze_text = "Frozen.\n"
    (repo / "prereg" / "FREEZE").write_text(freeze_text)
    _commit(repo, "freeze", at=1_100)
    _side_branch_commit(repo, "z", main, at=1_200)
    recipe = repo / "prereg" / "recipe.md"

    def lift_and_tune():
        (repo / "prereg" / "FREEZE").unlink()
        recipe.write_text(RECIPE.replace("0.05", "0.09"))

    _evil_merge(repo, "x", at=1_300, edit=lift_and_tune)

    def restore():
        recipe.write_text(RECIPE)
        (repo / "prereg" / "FREEZE").write_text(freeze_text)

    _evil_merge(repo, "z", at=1_400, edit=restore)
    with pytest.raises(PreregRefrozenError, match="deleted 1"):
        check_prereg(repo, PREREG_FILES, MODELS, run_timestamp=2_000)


def _partial_clone_of(repo, dest):
    _git(repo, "config", "uploadpack.allowFilter", "true")
    _git(repo, "config", "uploadpack.allowAnySHA1InWant", "true")
    subprocess.run(
        ["git", "clone", "-q", "--filter=blob:none", f"file://{repo}", str(dest)],
        check=True, capture_output=True,
    )


def test_a_partial_clone_is_refused_and_its_configured_remote_program_never_runs(tmp_path):
    """In a partial clone git fetches missing blobs on demand, and the
    repository's own `remote.origin.uploadpack` names the program that
    serves them — author-controlled code, run in the middle of the check
    (and able to rewrite the working tree after the byte comparison)."""
    repo, _ = _frozen_repo(tmp_path)
    (repo / "prereg" / "thresholds.json").write_text('{"tuned": true}')
    _commit(repo, "tune", at=1_500_000)  # the frozen blob is now absent from the clone
    clone = tmp_path / "clone"
    _partial_clone_of(repo, clone)
    marker = tmp_path / "ran-uploadpack"
    hook = tmp_path / "uploadpack.sh"
    hook.write_text(f'#!/bin/sh\ntouch {marker}\nexec git-upload-pack "$@"\n')
    hook.chmod(0o755)
    _git(clone, "config", "remote.origin.uploadpack", str(hook))
    with pytest.raises(PreregHistoryError, match="partial"):
        check_prereg(clone, PREREG_FILES, MODELS, run_timestamp=2_000_000)
    assert not marker.exists(), "the repo-configured upload-pack program was run"


def test_the_text_that_is_checked_is_the_text_that_was_byte_compared(tmp_path, monkeypatch):
    """The recipe and prediction are decoded from the very bytes that passed
    the comparison, not re-read from disk afterwards."""
    from pathlib import Path as _Path

    repo, freeze_sha = _frozen_repo(tmp_path)
    original = _Path.read_text
    reads = []

    def spy(self, *args, **kwargs):
        reads.append(self.name)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(_Path, "read_text", spy)
    assert check_prereg(repo, PREREG_FILES, MODELS, run_timestamp=2_000_000) == freeze_sha
    assert "recipe.md" not in reads and "prediction.md" not in reads


def test_a_forged_object_whose_bytes_do_not_match_its_name_is_refused(tmp_path):
    """`git show` serves whatever bytes a loose object file holds; overwrite
    the frozen recipe's object with the tuned recipe's bytes and every
    comparison looks clean. fsck notices the hash-path mismatch."""
    repo, freeze_sha = _frozen_repo(tmp_path)
    tuned = RECIPE.replace("0.05", "0.09")
    (repo / "prereg" / "recipe.md").write_text(tuned)
    _commit(repo, "tune", at=1_500_000)
    blob = _git(repo, "rev-parse", f"{freeze_sha}:prereg/recipe.md").strip()
    obj = repo / ".git" / "objects" / blob[:2] / blob[2:]
    obj.chmod(0o644)
    body = tuned.encode()
    obj.write_bytes(zlib.compress(b"blob %d\x00" % len(body) + body))
    assert _git(repo, "show", f"{freeze_sha}:prereg/recipe.md") == tuned  # the forgery works
    with pytest.raises(PreregHistoryError, match="fsck"):
        check_prereg(repo, PREREG_FILES, MODELS, run_timestamp=2_000_000)


def test_repo_config_cannot_make_the_check_execute_commands(tmp_path):
    """The repository's own config is under the author's control. An fsmonitor
    hook or a clean filter would run on `git status`; the check must not
    execute either."""
    repo, _ = _frozen_repo(tmp_path)
    marker = tmp_path / "ran-fsmonitor"
    hook = tmp_path / "hook.sh"
    hook.write_text(f"#!/bin/sh\ntouch {marker}\nexit 1\n")
    hook.chmod(0o755)
    _git(repo, "config", "core.fsmonitor", str(hook))
    filter_marker = tmp_path / "ran-filter"
    _git(repo, "config", "filter.evil.clean", f"sh -c 'touch {filter_marker}; cat'")
    (repo / ".git" / "info").mkdir(exist_ok=True)
    (repo / ".git" / "info" / "attributes").write_text("prereg/* filter=evil\n")
    # touch a file's mtime so git would have reason to re-hash it
    path = repo / "prereg" / "recipe.md"
    path.write_bytes(path.read_bytes())
    check_prereg(repo, PREREG_FILES, MODELS, run_timestamp=2_000_000)
    assert not marker.exists(), "core.fsmonitor was executed"
    assert not filter_marker.exists(), "a clean filter was executed"


def test_signature_display_config_does_not_corrupt_the_timestamp_or_history(tmp_path):
    repo, freeze_sha = _frozen_repo(tmp_path)
    _git(repo, "config", "log.showSignature", "true")
    assert check_prereg(repo, PREREG_FILES, MODELS, run_timestamp=2_000_000) == freeze_sha


def test_a_working_tree_symlink_standing_in_for_a_frozen_file_is_refused(tmp_path):
    # The symlink points at a file holding EXACTLY the frozen bytes, so only
    # the symlink check — not the byte comparison — can refuse it. Its target
    # could be changed after certification and the judge would read the new
    # bytes.
    repo, _ = _frozen_repo(tmp_path)
    path = repo / "prereg" / "thresholds.json"
    twin = tmp_path / "outside-copy.json"
    twin.write_bytes(path.read_bytes())
    path.unlink()
    path.symlink_to(twin)
    with pytest.raises(PreregNotCommittedError, match="symlink"):
        check_prereg(repo, PREREG_FILES, MODELS, run_timestamp=2_000_000)


@pytest.mark.parametrize("edit", [lambda b: b + b" ", lambda b: b" " + b, lambda b: b + b"\n"])
def test_a_whitespace_only_uncommitted_edit_is_reported_as_uncommitted(tmp_path, edit):
    repo, _ = _frozen_repo(tmp_path)
    path = repo / "prereg" / "recipe.md"
    path.write_bytes(edit(path.read_bytes()))
    with pytest.raises(PreregNotCommittedError, match="uncommitted"):
        check_prereg(repo, PREREG_FILES, MODELS, run_timestamp=2_000_000)


def test_an_uncommitted_crlf_checkout_is_reported_as_a_line_ending_difference(tmp_path):
    # What a Windows checkout with core.autocrlf=true looks like to us: the
    # working copy is the committed text with CRLF line endings.
    repo, _ = _frozen_repo(tmp_path)
    path = repo / "prereg" / "recipe.md"
    path.write_bytes(path.read_bytes().replace(b"\n", b"\r\n"))
    with pytest.raises(PreregLineEndingError, match="autocrlf"):
        check_prereg(repo, PREREG_FILES, MODELS, run_timestamp=2_000_000)


def test_a_commit_time_git_cannot_parse_is_a_prereg_error(tmp_path, monkeypatch):
    # `_git` is this module's wrapper around the external git process — the
    # boundary — so faking its answer here is mocking the edge, not owned logic.
    from wmj.harness import prereg

    real = prereg._git

    def fake(repo, *args, **kwargs):
        if args[:2] == ("show", "-s"):
            return "No signature\n1791042387\n"
        return real(repo, *args, **kwargs)

    repo, _ = _frozen_repo(tmp_path)
    monkeypatch.setattr(prereg, "_git", fake)
    with pytest.raises(PreregError, match="committer time"):
        check_prereg(repo, PREREG_FILES, MODELS, run_timestamp=2_000_000)


def test_a_file_that_is_in_the_working_tree_but_not_at_head_is_refused(tmp_path):
    # Deleted in a later commit, then quietly re-created, untracked, with the
    # frozen bytes: not committed, so not certified.
    repo, _ = _frozen_repo(tmp_path)
    path = repo / "prereg" / "thresholds.json"
    frozen = path.read_bytes()
    _git(repo, "rm", "-q", "prereg/thresholds.json")
    _commit(repo, "delete", at=1_500_000)
    path.write_bytes(frozen)
    with pytest.raises(PreregNotCommittedError, match="HEAD"):
        check_prereg(repo, PREREG_FILES, MODELS, run_timestamp=2_000_000)


# --- independent review: the comparison is byte-exact (TC-MU6-04) ---


@pytest.mark.parametrize(
    ("label", "tamper"),
    [
        ("trailing space", lambda b: b.rstrip(b"\n") + b" \n"),
        ("extra trailing newline", lambda b: b + b"\n"),
        ("byte-order mark", lambda b: b"\xef\xbb\xbf" + b),
        ("leading blank line", lambda b: b"\n" + b),
    ],
)
def test_tc_mu6_04_any_committed_byte_change_is_refused(tmp_path, label, tamper):
    repo, _ = _frozen_repo(tmp_path)
    path = repo / "prereg" / "recipe.md"
    path.write_bytes(tamper(path.read_bytes()))
    _commit(repo, label, at=1_500_000)
    with pytest.raises(PreregContentError):
        check_prereg(repo, PREREG_FILES, MODELS, run_timestamp=2_000_000)


def test_a_line_ending_only_change_is_refused_with_a_message_that_says_so(tmp_path):
    # What a Windows checkout with core.autocrlf=true looks like: the bytes
    # differ only by CRLF. It must fail closed, but say WHY rather than
    # look like tampering.
    repo, _ = _frozen_repo(tmp_path)
    path = repo / "prereg" / "recipe.md"
    path.write_bytes(path.read_bytes().replace(b"\n", b"\r\n"))
    _commit(repo, "crlf", at=1_500_000)
    with pytest.raises(PreregLineEndingError, match="autocrlf"):
        check_prereg(repo, PREREG_FILES, MODELS, run_timestamp=2_000_000)


def test_the_repo_pins_prereg_files_to_raw_bytes_for_every_checkout(tmp_path):
    """Actually check out under core.autocrlf=true (the Windows default): with
    the repo's real .gitattributes the bytes survive; without it they become
    CRLF — so the control proves the test can tell the difference."""
    root = Path(__file__).resolve().parents[3]
    attributes = (root / ".gitattributes").read_text(encoding="utf-8")

    def clone_with_autocrlf(with_attributes):
        src = tmp_path / ("src-with" if with_attributes else "src-without")
        _init_repo(src)
        (src / "prereg").mkdir()
        (src / "prereg" / "recipe.md").write_bytes(b"line one\nline two\n")
        if with_attributes:
            (src / ".gitattributes").write_text(attributes)
        _commit(src, "files", at=1_000)
        dest = tmp_path / ("dst-with" if with_attributes else "dst-without")
        subprocess.run(
            ["git", "clone", "-q", "-c", "core.autocrlf=true", f"file://{src}", str(dest)],
            check=True, capture_output=True,
        )
        return (dest / "prereg" / "recipe.md").read_bytes()

    assert clone_with_autocrlf(with_attributes=True) == b"line one\nline two\n"
    assert b"\r\n" in clone_with_autocrlf(with_attributes=False)


def test_a_certified_file_deleted_in_a_later_commit_is_refused(tmp_path):
    repo, _ = _frozen_repo(tmp_path)
    _git(repo, "rm", "-q", "prereg/thresholds.json")
    _commit(repo, "delete thresholds", at=1_500_000)
    with pytest.raises(PreregNotCommittedError, match="no working-tree copy"):
        check_prereg(repo, PREREG_FILES, MODELS, run_timestamp=2_000_000)


# --- independent review: timestamps and entry-check inputs ---


@pytest.mark.parametrize("bad", [float("nan"), 2_000_000.5, True, None, "2000000"])
def test_the_run_timestamp_must_be_a_plain_integer(tmp_path, bad):
    repo, _ = _frozen_repo(tmp_path)
    with pytest.raises(PreregError, match="run_timestamp"):
        check_prereg(repo, PREREG_FILES, MODELS, run_timestamp=bad)


def test_the_committer_date_not_the_author_date_is_what_is_timed(tmp_path):
    # ADR-M5 times the freeze by the commit's committer date. A freeze
    # authored long ago but committed after the run must not pass.
    repo = tmp_path / "repo"
    _init_repo(repo)
    _commit_prereg(repo, recipe=RECIPE, prediction=PREDICTION, at=1_000_000)
    (repo / "prereg" / "FREEZE").write_text("Frozen.\n")
    _git(repo, "add", "-A")
    env = {**os.environ, "GIT_AUTHOR_DATE": "@100 +0000", "GIT_COMMITTER_DATE": "@3000000 +0000"}
    subprocess.run(["git", "-C", str(repo), "commit", "-q", "-m", "freeze"], check=True,
                   capture_output=True, env=env)
    with pytest.raises(PreregOrderingError):
        check_prereg(repo, PREREG_FILES, MODELS, run_timestamp=2_000_000)


def test_a_model_must_be_named_in_both_the_recipe_and_the_prediction(tmp_path):
    repo = tmp_path / "repo"
    _init_repo(repo)
    _commit_prereg(
        repo, recipe=RECIPE + "recipe-only-model is declared here.\n",
        prediction=PREDICTION + "prediction-only-model is predicted here.\n", at=1_000_000,
    )
    _freeze(repo, at=1_100_000)
    for name in ("recipe-only-model", "prediction-only-model"):
        with pytest.raises(PreregEntryMissingError):
            check_prereg(repo, PREREG_FILES, [_Model(name)], run_timestamp=2_000_000)


def test_a_blank_model_name_is_refused_not_vacuously_accepted(tmp_path):
    repo, _ = _frozen_repo(tmp_path)
    for name in ("", "  "):
        with pytest.raises(PreregEntryMissingError, match="blank"):
            check_prereg(repo, PREREG_FILES, [_Model(name)], run_timestamp=2_000_000)


def test_a_recipe_that_is_not_utf8_is_a_prereg_error_not_a_decode_crash(tmp_path):
    repo = tmp_path / "repo"
    _init_repo(repo)
    _write_prereg(repo, recipe="", prediction=PREDICTION)
    (repo / "prereg" / "recipe.md").write_bytes(b"\xff\xfe not utf-8 \xff")
    _commit(repo, "prereg", at=1_000_000)
    _freeze(repo, at=1_100_000)
    with pytest.raises(PreregError, match="UTF-8"):
        check_prereg(repo, PREREG_FILES, MODELS, run_timestamp=2_000_000)


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


def test_tc_mu6_09_a_later_conflicting_duplicate_is_refused():
    with pytest.raises(PreregWorldConstantError, match="more than once"):
        check_recipe_world_constants(RECIPE + "lv_kick_rate_per_s: 0.9\n")


def test_tc_mu6_09_a_near_miss_value_is_refused_exactly():
    with pytest.raises(PreregWorldConstantError, match="lv_action_max"):
        check_recipe_world_constants(RECIPE.replace("lv_action_max: 0.1", "lv_action_max: 0.1000001"))


@pytest.mark.parametrize("value", ["nan", "inf", "1_0e-1", "０.５", "0x1", "."])
def test_tc_mu6_09_only_plain_decimal_numbers_are_accepted(value):
    with pytest.raises(PreregWorldConstantError):
        check_recipe_world_constants(RECIPE.replace("lv_kick_rate_per_s: 0.5", f"lv_kick_rate_per_s: {value}"))


@pytest.mark.parametrize("value", ["0.5", ".5", "5e-1", "0.50"])
def test_tc_mu6_09_equivalent_plain_decimal_spellings_are_accepted(value):
    check_recipe_world_constants(RECIPE.replace("lv_kick_rate_per_s: 0.5", f"lv_kick_rate_per_s: {value}"))


@pytest.mark.parametrize(
    "decoy",
    ["  lv_kick_rate_per_s: 0.5", "> lv_kick_rate_per_s: 0.5", "see lv_kick_rate_per_s: 0.5"],
)
def test_tc_mu6_09_only_a_line_that_starts_with_the_key_counts(decoy):
    without = "\n".join(
        line for line in RECIPE.splitlines() if not line.startswith("lv_kick_rate_per_s")
    )
    with pytest.raises(PreregWorldConstantError, match="lv_kick_rate_per_s"):
        check_recipe_world_constants(without + "\n" + decoy + "\n")


def test_tc_mu6_09_a_value_on_the_next_line_does_not_count():
    with pytest.raises(PreregWorldConstantError, match="lv_kick_rate_per_s"):
        check_recipe_world_constants(RECIPE.replace("lv_kick_rate_per_s: 0.5", "lv_kick_rate_per_s:\n0.5"))


@pytest.mark.parametrize("world", ["lv", "pendulum"])
def test_tc_mu6_09_each_key_is_compared_with_its_own_constant(monkeypatch, world):
    # The two real values coincide for the pendulum (1.0 and 1.0), so give
    # each constant a distinct value and check the right one is used.
    from wmj.worlds import lv, pendulum

    module = {"lv": lv, "pendulum": pendulum}[world]
    monkeypatch.setattr(module, "KICK_RATE_PER_S", 7.0)
    monkeypatch.setattr(module, "TRAINED_ACTION_MAX", 3.0)
    recipe = RECIPE.replace(f"{world}_kick_rate_per_s: {1.0 if world == 'pendulum' else 0.5}",
                            f"{world}_kick_rate_per_s: 7.0")
    recipe = recipe.replace(f"{world}_action_max: {1.0 if world == 'pendulum' else 0.1}",
                            f"{world}_action_max: 3.0")
    check_recipe_world_constants(recipe)


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


def test_a_file_named_head_in_the_repo_root_does_not_confuse_the_history_walk(tmp_path):
    """Review pass 3 (Minor): an untracked file called `HEAD` made `rev-list HEAD`
    abort as ambiguous — a false refusal. The check must still pass."""
    repo, freeze_sha = _frozen_repo(tmp_path)
    (repo / "HEAD").write_text("not a revision\n")
    assert check_prereg(repo, PREREG_FILES, MODELS, run_timestamp=2_000_000) == freeze_sha


@pytest.mark.parametrize(
    "signal",
    ["extensions.partialclone", "remote.origin.promisor", "promisor-pack-file"],
)
def test_each_sign_of_a_partial_clone_is_refused_on_its_own(tmp_path, signal):
    """Mutation check, review pass 3: the three signs of a partial clone are
    independent defences, so each needs its own test — a full clone with just
    one sign planted must be refused."""
    repo, _ = _frozen_repo(tmp_path)
    if signal == "promisor-pack-file":
        packdir = repo / ".git" / "objects" / "pack"
        packdir.mkdir(parents=True, exist_ok=True)
        (packdir / "pack-0000.promisor").write_text("")
    elif signal == "extensions.partialclone":
        _git(repo, "config", signal, "origin")
    else:
        _git(repo, "config", signal, "true")
    with pytest.raises(PreregHistoryError, match="partial"):
        check_prereg(repo, PREREG_FILES, MODELS, run_timestamp=2_000_000)


def test_a_replace_ref_does_not_add_a_second_freeze_to_the_count(tmp_path):
    """`git replace` refs live under refs/replace/ and are ignored everywhere
    else here; counting them as history would turn a harmless local replace
    into a false 'frozen twice' refusal."""
    repo, freeze_sha = _frozen_repo(tmp_path)
    main = _git(repo, "rev-parse", "--abbrev-ref", "HEAD").strip()
    first = _git(repo, "rev-parse", "HEAD~1").strip()
    _git(repo, "checkout", "-qb", "other", first)
    _freeze(repo, at=1_200_000, text="a different freeze\n")
    other = _git(repo, "rev-parse", "HEAD").strip()
    _git(repo, "checkout", "-q", main)
    _git(repo, "branch", "-q", "-D", "other")
    _git(repo, "update-ref", f"refs/replace/{first}", other)
    assert check_prereg(repo, PREREG_FILES, MODELS, run_timestamp=2_000_000) == freeze_sha


def test_the_recipe_and_prediction_files_are_read_from_disk_exactly_once(tmp_path, monkeypatch):
    """Mutation check, review pass 3: the byte comparison reads each certified
    file once; the later text checks must reuse those bytes, not read again."""
    from pathlib import Path as _Path

    repo, freeze_sha = _frozen_repo(tmp_path)
    original = _Path.read_bytes
    reads = []

    def spy(self):
        reads.append(self.name)
        return original(self)

    monkeypatch.setattr(_Path, "read_bytes", spy)
    assert check_prereg(repo, PREREG_FILES, MODELS, run_timestamp=2_000_000) == freeze_sha
    assert reads.count("recipe.md") == 1 and reads.count("prediction.md") == 1


# --- independent review, P3-C10 pass 4: pin what the code already got right ---


def _signed_looking(repo):
    """Rewrite HEAD as a commit that carries a `gpgsig` header (not a valid
    signature) so git has something to try to verify when told to show it."""
    raw = _git(repo, "cat-file", "commit", "HEAD")
    head, _, message = raw.partition("\n\n")
    signed = head + "\ngpgsig -----BEGIN PGP SIGNATURE-----\n \n abcd\n -----END PGP SIGNATURE-----\n\n" + message
    new = subprocess.run(
        ["git", "-C", str(repo), "hash-object", "-t", "commit", "-w", "--stdin"],
        input=signed, check=True, capture_output=True, text=True,
    ).stdout.strip()
    _git(repo, "update-ref", "HEAD", new)
    return new


def test_a_signed_looking_freeze_commit_is_not_disturbed_by_signature_display(tmp_path):
    repo, _ = _frozen_repo(tmp_path)
    _git(repo, "config", "log.showSignature", "true")
    freeze_sha = _signed_looking(repo)
    assert check_prereg(repo, PREREG_FILES, MODELS, run_timestamp=2_000_000) == freeze_sha


@pytest.mark.parametrize("order", ["newcomer-first", "newcomer-last", "baseline-first", "two-undeclared"])
def test_every_unrigged_model_is_checked_not_just_the_first_or_last(tmp_path, order):
    """The contest has two unrigged models; one never declared must fail the
    certificate wherever it sits in the list and whatever precedes it."""
    repo, _ = _frozen_repo(tmp_path)
    models = {
        "newcomer-first": [_Model("newcomer"), _Model("direct")],
        "newcomer-last": [_Model("direct"), _Model("newcomer")],
        "baseline-first": [_Model("persistence", is_baseline=True), _Model("newcomer")],
        "two-undeclared": [_Model("alpha"), _Model("beta")],
    }[order]
    with pytest.raises(PreregEntryMissingError):
        check_prereg(repo, PREREG_FILES, models, run_timestamp=2_000_000)


def test_a_freeze_on_a_feature_branch_that_was_merged_and_deleted_is_the_freeze(tmp_path):
    """An honest workflow: freeze on a branch, merge it with a merge commit,
    delete the branch. The add sits on the merge's SECOND parent and must be
    found, not refused and not missed."""
    repo = tmp_path / "repo"
    _init_repo(repo)
    _commit_prereg(repo, recipe=RECIPE, prediction=PREDICTION, at=1_000_000)
    main = _git(repo, "rev-parse", "--abbrev-ref", "HEAD").strip()
    _git(repo, "checkout", "-qb", "s")
    freeze_sha = _freeze(repo, at=1_100_000)
    _git(repo, "checkout", "-q", main)
    (repo / "other.txt").write_text("x")
    _commit(repo, "main moves", at=1_200_000)
    _git(repo, "merge", "--no-ff", "-q", "-m", "merge s", "s", date=None)
    _git(repo, "branch", "-q", "-D", "s")
    assert check_prereg(repo, PREREG_FILES, MODELS, run_timestamp=2_000_000) == freeze_sha


def test_a_second_freeze_hidden_behind_a_merge_second_parent_is_still_counted(tmp_path):
    """Freeze 1 on branch `s`; tuned thresholds and freeze 2 on main; `s` merged
    with `-X ours` and deleted. Only a walk of the merge's second parent sees
    freeze 1, so only it shows there were two."""
    repo = tmp_path / "repo"
    _init_repo(repo)
    _commit_prereg(repo, recipe=RECIPE, prediction=PREDICTION, at=1_000_000)
    main = _git(repo, "rev-parse", "--abbrev-ref", "HEAD").strip()
    _git(repo, "checkout", "-qb", "s")
    _freeze(repo, at=1_100_000)
    _git(repo, "checkout", "-q", main)
    (repo / "prereg" / "thresholds.json").write_text('{"tuned": true}')
    _freeze(repo, at=1_200_000, text="second freeze\n")
    _git(repo, "merge", "--no-ff", "-q", "-X", "ours", "-m", "merge s", "s")
    _git(repo, "branch", "-q", "-D", "s")
    with pytest.raises(PreregRefrozenError, match="2 time"):
        check_prereg(repo, PREREG_FILES, MODELS, run_timestamp=2_000_000)


@pytest.mark.parametrize(
    "ref", ["refs/tags/old-freeze", "refs/remotes/origin/feature", "refs/notes/hidden"]
)
def test_a_second_freeze_held_only_by_a_tag_remote_ref_or_notes_ref_is_counted(tmp_path, ref):
    repo, _ = _frozen_repo(tmp_path)
    main = _git(repo, "rev-parse", "--abbrev-ref", "HEAD").strip()
    first = _git(repo, "rev-parse", "HEAD~1").strip()
    _git(repo, "checkout", "-qb", "other", first)
    _freeze(repo, at=1_200_000, text="a different freeze\n")
    other = _git(repo, "rev-parse", "HEAD").strip()
    _git(repo, "checkout", "-q", main)
    _git(repo, "branch", "-q", "-D", "other")
    _git(repo, "update-ref", ref, other)
    with pytest.raises(PreregRefrozenError, match="2 time"):
        check_prereg(repo, PREREG_FILES, MODELS, run_timestamp=2_000_000)


def test_the_refusal_names_the_commits_so_a_stale_ref_can_be_found(tmp_path):
    repo, freeze_sha = _frozen_repo(tmp_path)
    first = _git(repo, "rev-parse", "HEAD~1").strip()
    _git(repo, "checkout", "-qb", "stale", first)
    _freeze(repo, at=1_200_000, text="a different freeze\n")
    other = _git(repo, "rev-parse", "HEAD").strip()
    _git(repo, "checkout", "-q", "-")
    with pytest.raises(PreregRefrozenError) as info:
        check_prereg(repo, PREREG_FILES, MODELS, run_timestamp=2_000_000)
    assert freeze_sha[:10] in str(info.value) and other[:10] in str(info.value)


def test_a_symlinked_prereg_directory_is_refused(tmp_path):
    """The directory could be repointed after certification just like a file."""
    repo, _ = _frozen_repo(tmp_path)
    copy = tmp_path / "prereg-copy"
    (repo / "prereg").rename(copy)
    (repo / "prereg").symlink_to(copy, target_is_directory=True)
    with pytest.raises(PreregNotCommittedError, match="symlink"):
        check_prereg(repo, PREREG_FILES, MODELS, run_timestamp=2_000_000)


def test_an_extra_certified_file_is_byte_compared_like_the_fixed_ones(tmp_path):
    repo = tmp_path / "repo"
    _init_repo(repo)
    _commit_prereg(repo, recipe=RECIPE, prediction=PREDICTION, at=1_000_000)
    (repo / "prereg" / "notes.md").write_text("before\n")
    _commit(repo, "notes", at=1_050_000)
    _freeze(repo, at=1_100_000)
    (repo / "prereg" / "notes.md").write_text("after\n")
    _commit(repo, "edit notes", at=1_500_000)
    with pytest.raises(PreregContentError, match="notes.md"):
        check_prereg(repo, [*PREREG_FILES, "notes.md"], MODELS, run_timestamp=2_000_000)


def test_a_numpy_integer_run_timestamp_is_accepted(tmp_path):
    import numpy as np

    repo, freeze_sha = _frozen_repo(tmp_path)
    assert check_prereg(repo, PREREG_FILES, MODELS, run_timestamp=np.int64(2_000_000)) == freeze_sha


# --- independent review, P3-C10 pass 5: boundaries, layouts, exit codes ---


@pytest.mark.parametrize(
    ("text", "name", "expected"),
    [
        ("we use direct here", "direct", True),
        ("direct", "direct", True),
        ("(direct)", "direct", True),
        ("direct-v2", "direct", False),  # right-hand hyphen continues the name
        ("deep-direct", "direct", False),  # left-hand hyphen
        ("directional", "direct", False),  # right-hand letter
        ("redirect", "direct", False),  # left-hand letter
        ("direct_2", "direct", False),  # underscore continues the name
        ("direct2", "direct", False),  # digit continues the name
        ("2direct", "direct", False),
        ("DIRECT", "direct", False),  # names are case-sensitive
        ("a.b", "a.b", True),
        ("axb", "a.b", False),  # a regex metacharacter in a name is literal
        ("fx-action-blind", "fx-action-blind", True),
    ],
)
def test_declares_matches_whole_tokens_only(text, name, expected):
    from wmj.harness.prereg import _declares

    assert _declares(text, name) is expected


def _worktree(repo, dest):
    _git(repo, "worktree", "add", "-q", "--detach", str(dest), "HEAD")


def test_a_grafts_file_is_refused_when_the_checkout_is_a_linked_worktree(tmp_path):
    """`.git` is a file there and grafts live in the common git dir; a check that
    looked only under `<repo>/.git/info` would see nothing and the freeze
    commit could be hidden (the tuned commit becomes a root)."""
    repo, _ = _frozen_repo(tmp_path)
    (repo / "prereg" / "recipe.md").write_text(RECIPE.replace("0.05", "0.09"))
    _commit(repo, "tune", at=1_500_000)
    tuned = _git(repo, "rev-parse", "HEAD").strip()
    wt = tmp_path / "wt"
    _worktree(repo, wt)
    (repo / ".git" / "info").mkdir(exist_ok=True)
    (repo / ".git" / "info" / "grafts").write_text(f"{tuned}\n")
    with pytest.raises(PreregHistoryError, match="grafts"):
        check_prereg(wt, PREREG_FILES, MODELS, run_timestamp=2_000_000)


def test_a_linked_worktree_of_an_honest_frozen_repo_passes(tmp_path):
    repo, freeze_sha = _frozen_repo(tmp_path)
    wt = tmp_path / "wt"
    _worktree(repo, wt)
    assert check_prereg(wt, PREREG_FILES, MODELS, run_timestamp=2_000_000) == freeze_sha


def test_any_nonzero_fsck_exit_is_refused_not_only_exit_one(tmp_path):
    """fsck reports different faults with different exit bits (1 object error,
    2 connectivity, 4 ref error, ...). A ref pointing at nothing exits 2."""
    repo, _ = _frozen_repo(tmp_path)
    (repo / ".git" / "refs" / "heads" / "broken").write_text("0123456789012345678901234567890123456789\n")
    with pytest.raises(PreregHistoryError, match="fsck"):
        check_prereg(repo, PREREG_FILES, MODELS, run_timestamp=2_000_000)


def test_a_relative_repo_path_is_accepted(tmp_path, monkeypatch):
    repo, freeze_sha = _frozen_repo(tmp_path)
    monkeypatch.chdir(repo)
    assert check_prereg(Path("."), PREREG_FILES, MODELS, run_timestamp=2_000_000) == freeze_sha


def test_a_symlinked_repo_path_is_accepted(tmp_path):
    repo, freeze_sha = _frozen_repo(tmp_path)
    link = tmp_path / "link-to-repo"
    link.symlink_to(repo, target_is_directory=True)
    assert check_prereg(link, PREREG_FILES, MODELS, run_timestamp=2_000_000) == freeze_sha


def test_a_certified_path_that_was_a_directory_at_the_freeze_is_refused(tmp_path):
    """`byte-equal to its blob` needs a blob. If `prereg/recipe.md` was a
    directory when frozen, `git show` prints its listing; a later FILE holding
    exactly that listing (and satisfying every text rule) must still be refused,
    because nothing was ever frozen under that name as a file."""
    repo = tmp_path / "repo"
    _init_repo(repo)
    pdir = repo / "prereg"
    (pdir / "recipe.md").mkdir(parents=True)
    for entry in [*KICK_KEYS.strip().splitlines(), "direct", "ensemble"]:
        (pdir / "recipe.md" / entry).write_text("x")
    (pdir / "prediction.md").write_text(PREDICTION)
    (pdir / "thresholds.json").write_text("{}")
    _commit(repo, "dirs", at=1_000_000)
    freeze_sha = _freeze(repo, at=1_100_000)
    listing = _git(repo, "show", f"{freeze_sha}:prereg/recipe.md")
    import shutil

    shutil.rmtree(pdir / "recipe.md")
    (pdir / "recipe.md").write_text(listing)
    _commit(repo, "now a file", at=1_500_000)
    with pytest.raises(PreregNotFrozenError, match="recipe.md"):
        check_prereg(repo, PREREG_FILES, MODELS, run_timestamp=2_000_000)


def test_a_non_utf8_prediction_is_refused_with_a_clear_message(tmp_path):
    repo = tmp_path / "repo"
    _init_repo(repo)
    _write_prereg(repo, recipe=RECIPE, prediction="x")
    (repo / "prereg" / "prediction.md").write_bytes(b"direct ensemble \xff\xfe\n")
    _commit(repo, "prereg", at=1_000_000)
    _freeze(repo, at=1_100_000)
    with pytest.raises(PreregError, match="UTF-8"):
        check_prereg(repo, PREREG_FILES, MODELS, run_timestamp=2_000_000)


def test_a_nul_in_a_certified_file_name_is_a_clear_refusal(tmp_path):
    repo, _ = _frozen_repo(tmp_path)
    with pytest.raises(PreregError, match="plain file name"):
        check_prereg(repo, ["recipe\0.md"], MODELS, run_timestamp=2_000_000)


def test_the_world_constants_are_compared_exactly(tmp_path):
    from wmj.harness.prereg import check_recipe_world_constants

    with pytest.raises(PreregWorldConstantError):
        check_recipe_world_constants(RECIPE.replace("lv_kick_rate_per_s: 0.5", "lv_kick_rate_per_s: 0.50000000001"))
    with pytest.raises(PreregWorldConstantError):  # underscores are not a plain decimal; 5_0e-2 == 0.5
        check_recipe_world_constants(RECIPE.replace("lv_kick_rate_per_s: 0.5", "lv_kick_rate_per_s: 5_0e-2"))
    # An honest note glued to the value is not part of the number.
    check_recipe_world_constants(RECIPE.replace("lv_kick_rate_per_s: 0.5", "lv_kick_rate_per_s: 0.5#note"))
