"""Tests for wmj.harness.prereg — the MU-6 / JU-11 certification (harness).

check_prereg is how "you cannot move a threshold after seeing results"
becomes mechanically true: every prereg/ file must be committed, its
first-added commit must predate the judged run, and its content must not
have changed since that first-added commit. The two phantom-gates
(TC-MU6-04 content drift, TC-JU11-02 post-run edit) prove the checks can
actually fail. Each test builds a throwaway git repo in tmp_path with
committed prereg files at controlled dates.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass

import pytest

from wmj.harness.prereg import (
    PreregContentError,
    PreregEntryMissingError,
    PreregError,
    PreregNotCommittedError,
    PreregOrderingError,
    check_prereg,
    first_added_commit,
    read_matching_margin,
    within_matching_margin,
)


@dataclass
class _Model:
    name: str
    is_baseline: bool = False
    is_fixture: bool = False


def _git(repo, *args, date=None):
    env = None
    if date is not None:
        env = {"GIT_AUTHOR_DATE": str(date), "GIT_COMMITTER_DATE": str(date)}
    import os

    full_env = {**os.environ, **(env or {})}
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
        env=full_env,
    ).stdout


def _init_repo(repo):
    repo.mkdir(parents=True, exist_ok=True)
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@example.com")
    _git(repo, "config", "user.name", "t")


def _commit_prereg(repo, *, recipe="", prediction="", thresholds="{}", at):
    """Create prereg/ files and commit them at unix time `at`."""
    pdir = repo / "prereg"
    pdir.mkdir(exist_ok=True)
    (pdir / "recipe.md").write_text(recipe)
    (pdir / "prediction.md").write_text(prediction)
    (pdir / "thresholds.json").write_text(thresholds)
    _git(repo, "add", "prereg")
    _git(repo, "commit", "-q", "-m", "prereg", date=f"@{at} +0000")


PREREG_FILES = ["recipe.md", "prediction.md", "thresholds.json"]
# A recipe naming both unrigged models, as the real one does.
RECIPE = "matching_margin: 0.05\nmodels: direct, ensemble\n"
PREDICTION = "We predict ensemble ranks at or above direct.\ndirect vs ensemble.\n"


def test_happy_path_passes_and_returns_a_commit_of_record(tmp_path):
    repo = tmp_path / "repo"
    _init_repo(repo)
    _commit_prereg(repo, recipe=RECIPE, prediction=PREDICTION, at=1_000_000)
    models = [_Model("direct"), _Model("ensemble"), _Model("persistence", is_baseline=True)]
    sha = check_prereg(repo, PREREG_FILES, models, run_timestamp=2_000_000)
    assert isinstance(sha, str) and len(sha) >= 7


def test_refuses_when_prereg_commit_is_after_the_run(tmp_path):
    """TC-JU11-02 / TC-MU6-01: prereg committed AFTER the run must fail."""
    repo = tmp_path / "repo"
    _init_repo(repo)
    _commit_prereg(repo, recipe=RECIPE, prediction=PREDICTION, at=5_000_000)
    with pytest.raises(PreregOrderingError):
        check_prereg(repo, PREREG_FILES, [_Model("direct"), _Model("ensemble")], run_timestamp=4_000_000)


def test_refuses_when_content_changed_since_first_commit(tmp_path):
    """TC-MU6-04 (content-invariance): an honestly-dated second commit that
    edits the recipe after it was first added must fail the content-hash."""
    repo = tmp_path / "repo"
    _init_repo(repo)
    _commit_prereg(repo, recipe=RECIPE, prediction=PREDICTION, at=1_000_000)
    # A later, honestly-dated edit (no --amend, no history rewrite).
    (repo / "prereg" / "recipe.md").write_text(
        "matching_margin: 0.049  # tuned after seeing results\nmodels: direct, ensemble\n"
    )
    _git(repo, "add", "prereg/recipe.md")
    _git(repo, "commit", "-q", "-m", "tune", date="@1500000 +0000")
    with pytest.raises(PreregContentError):
        check_prereg(repo, PREREG_FILES, [_Model("direct"), _Model("ensemble")], run_timestamp=2_000_000)


def test_first_added_commit_resolves_the_oldest_add_not_the_readd(tmp_path):
    """The --reverse pin: a delete-then-readd produces two `A` events;
    first_added_commit must return the OLDEST (design-review-008 I6)."""
    repo = tmp_path / "repo"
    _init_repo(repo)
    _commit_prereg(repo, recipe=RECIPE, prediction=PREDICTION, at=1_000_000)
    oldest = first_added_commit(repo, "prereg/recipe.md")
    # delete then re-add the same path with different content
    _git(repo, "rm", "-q", "prereg/recipe.md")
    _git(repo, "commit", "-q", "-m", "del", date="@1200000 +0000")
    (repo / "prereg" / "recipe.md").write_text("different\n")
    _git(repo, "add", "prereg/recipe.md")
    _git(repo, "commit", "-q", "-m", "readd", date="@1300000 +0000")
    assert first_added_commit(repo, "prereg/recipe.md") == oldest


def test_refuses_an_unrigged_model_absent_from_the_recipe(tmp_path):
    """TC-MU6-03: a new unrigged model with no prereg entry is refused."""
    repo = tmp_path / "repo"
    _init_repo(repo)
    _commit_prereg(repo, recipe=RECIPE, prediction=PREDICTION, at=1_000_000)
    with pytest.raises(PreregEntryMissingError):
        check_prereg(repo, PREREG_FILES, [_Model("newcomer")], run_timestamp=2_000_000)


def test_baseline_and_fixture_are_exempt_from_the_entry_check(tmp_path):
    """TC-MU6-05(a): is_baseline / is_fixture models are skipped by
    check_prereg (adding a baseline stays one file)."""
    repo = tmp_path / "repo"
    _init_repo(repo)
    _commit_prereg(repo, recipe=RECIPE, prediction=PREDICTION, at=1_000_000)
    models = [
        _Model("persistence", is_baseline=True),
        _Model("fx-brittle", is_fixture=True),
        _Model("direct"),
        _Model("ensemble"),
    ]
    # No raise: the baseline/fixture need no prereg entry; direct/ensemble are present.
    check_prereg(repo, PREREG_FILES, models, run_timestamp=2_000_000)


def test_refuses_a_dirty_uncommitted_prereg_file(tmp_path):
    repo = tmp_path / "repo"
    _init_repo(repo)
    _commit_prereg(repo, recipe=RECIPE, prediction=PREDICTION, at=1_000_000)
    (repo / "prereg" / "recipe.md").write_text(RECIPE + "uncommitted edit\n")  # dirty, not committed
    # The not-committed check runs before the content-hash check, so this
    # fixture can only raise PreregNotCommittedError — assert exactly that, so
    # a future reorder that misclassifies it as content-drift would be caught.
    with pytest.raises(PreregNotCommittedError):
        check_prereg(repo, PREREG_FILES, [_Model("direct"), _Model("ensemble")], run_timestamp=2_000_000)


def test_refuses_a_name_that_is_only_a_substring_of_the_recipe(tmp_path):
    """The model-name entry check must match whole tokens, not substrings:
    an undeclared model named 'ir' must NOT be blessed just because 'direct'
    contains i-r (the silent MU-6 bypass P3-C07's review caught)."""
    repo = tmp_path / "repo"
    _init_repo(repo)
    _commit_prereg(repo, recipe=RECIPE, prediction=PREDICTION, at=1_000_000)
    with pytest.raises(PreregEntryMissingError):
        check_prereg(repo, PREREG_FILES, [_Model("ir")], run_timestamp=2_000_000)


def test_hyphenated_model_name_is_recognised_as_a_whole_token(tmp_path):
    """A declared hyphenated name (e.g. an unrigged 'deep-direct') is found
    as a token — the whole-token match must not reject legitimate hyphens."""
    repo = tmp_path / "repo"
    _init_repo(repo)
    recipe = RECIPE + "We also register deep-direct as an unrigged contestant.\n"
    prediction = PREDICTION + "deep-direct is predicted mid-pack.\n"
    _commit_prereg(repo, recipe=recipe, prediction=prediction, at=1_000_000)
    # No raise: deep-direct is present as a token in both files.
    check_prereg(repo, PREREG_FILES, [_Model("deep-direct")], run_timestamp=2_000_000)


def test_requires_recipe_and_prediction_among_certified_files(tmp_path):
    """If a caller omits recipe.md/prediction.md from the certified files, the
    per-model entry check would read uncommitted text — refuse up front."""
    repo = tmp_path / "repo"
    _init_repo(repo)
    _commit_prereg(repo, recipe=RECIPE, prediction=PREDICTION, at=1_000_000)
    with pytest.raises(PreregError):
        check_prereg(repo, ["thresholds.json"], [_Model("direct")], run_timestamp=2_000_000)


def test_content_drift_is_caught_end_to_end_through_check_prereg_on_readd(tmp_path):
    """The --reverse anti-gaming pin, exercised THROUGH check_prereg (not just
    first_added_commit): delete then re-add recipe.md with different content,
    and check_prereg must still reject it via the content-hash vs the OLDEST
    add — catching a regression that decoupled the hash from the oldest SHA."""
    repo = tmp_path / "repo"
    _init_repo(repo)
    _commit_prereg(repo, recipe=RECIPE, prediction=PREDICTION, at=1_000_000)
    _git(repo, "rm", "-q", "prereg/recipe.md")
    _git(repo, "commit", "-q", "-m", "del", date="@1_200_000 +0000".replace("_", ""))
    (repo / "prereg" / "recipe.md").write_text(
        "matching_margin: 0.049  # re-added, tuned\nmodels: direct, ensemble\n"
    )
    _git(repo, "add", "prereg/recipe.md")
    _git(repo, "commit", "-q", "-m", "readd", date="@1300000 +0000")
    with pytest.raises(PreregContentError):
        check_prereg(repo, PREREG_FILES, [_Model("direct"), _Model("ensemble")], run_timestamp=2_000_000)


# --- MU-5 matching margin (TC-MU5-01) ---


def test_read_matching_margin_parses_the_recipe(tmp_path):
    recipe = tmp_path / "recipe.md"
    recipe.write_text("epochs: 100\nmatching_margin: 0.05\ntraining_trajectories: 2000\n")
    assert read_matching_margin(recipe) == 0.05


def test_within_matching_margin_boundary():
    # at the margin: satisfied; over it: not yet satisfied (judging does not proceed)
    assert within_matching_margin(0.30, 0.34, margin=0.05) is True   # diff 0.04 <= 0.05
    assert within_matching_margin(0.30, 0.35, margin=0.05) is True   # diff 0.05 == 0.05
    assert within_matching_margin(0.30, 0.40, margin=0.05) is False  # diff 0.10 > 0.05
