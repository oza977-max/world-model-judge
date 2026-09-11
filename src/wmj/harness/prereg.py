"""wmj.harness.prereg — MU-6 / JU-11 certification: you can't move a threshold.

In plain words: before the judge is allowed to grade an unrigged model, this
checks that the recipe, the ranking prediction, and the thresholds were
written down, committed to git, and have not changed since — *before* any
result was seen. That is what makes the pre-registration real rather than a
promise. The judge itself cannot read git (it is a pure function, JU-12), so
the harness owns this check and hands the judge only the thresholds, as data.

Three things are asserted for every `prereg/` file (models ADR-M5):
  1. it is committed (not a dirty working-tree edit);
  2. its *first-added* commit timestamp strictly precedes the judged run;
  3. its content has not changed since that first-added commit — the blob at
     the first add hashes equal to the working-tree file.
And for every *unrigged* model (`not is_baseline and not is_fixture`): its
name appears in the committed recipe and prediction (a per-model entry, not
just "the files exist" — TC-MU6-03). Baselines and fixtures are exempt
(adding one stays a one-file change — TC-MU6-05).

Why (2) uses `--reverse` (design-review-008 I6): a file deleted and later
re-added has *two* "added" events; git's default newest-first order would
resolve to the re-add, whose content matches the working tree by
construction — silently passing the exact gaming this check exists to catch.
`--reverse` pins the oldest add.

This is a harness module: it shells out to git and reads files, which the
pure judge may not. Nothing here reads the network.

**Disclosed residuals — what `check_prereg` does NOT close (named, not
solved, in the same spirit as models ADR-M5's own residual risks).** These
are inherent to a single-author, offline, no-server-side-witness project;
no mechanical control here fully closes them, so they are disclosed rather
than pretended away (surfaced by the P3-C07 independent review):
  1. **Commit-timestamp forgeability** (ADR-M5's already-disclosed residual
     #3): the ordering check reads the committer date, which the committer
     supplies (`git commit --date=...`, `GIT_COMMITTER_DATE`). A run
     executed, disliked, then "pre-registered" with a back-dated commit
     passes the ordering check. The content-hash (TC-MU6-04) still catches
     the *un*-back-dated drift, which is the likelier accident.
  2. **Entry substance** (new, same family — backlog A13): the per-model
     check confirms the model's name appears as a token in the committed
     recipe and prediction; it cannot verify the surrounding text is a
     genuine recipe/prediction for that model. A hollow mention passes.
     Verifying substance would need semantic understanding of prose, which
     is not mechanizable; the honesty of the entry's *content* rests on the
     author, exactly as ADR-M5 already discloses for the recipe as a whole.
"""

from __future__ import annotations

import hashlib
import os
import re
import subprocess
from pathlib import Path
from typing import Iterable, Protocol

from wmj.errors import WmjError

PREREG_DIR = "prereg"

# The files whose committed content the per-model entry check reads — they
# MUST be among the certified files so that content is commit-verified, not
# read unguarded from the working tree.
_ENTRY_FILES = ("recipe.md", "prediction.md")


class PreregError(WmjError):
    """Base for every pre-registration certification failure."""


class PreregNotCommittedError(PreregError):
    """A prereg/ file is missing from git or has uncommitted working-tree edits."""


class PreregOrderingError(PreregError):
    """A prereg/ file's first-added commit does not precede the judged run."""


class PreregContentError(PreregError):
    """A prereg/ file's content changed since its first-added commit."""


class PreregEntryMissingError(PreregError):
    """An unrigged model has no named entry in the committed recipe/prediction."""


class _Classified(Protocol):
    name: str
    is_baseline: bool
    is_fixture: bool


def _git(repo: Path, *args: str) -> str:
    """Run `git -C repo <args>` and return stdout (text).

    A git failure (not a repo, git absent) is wrapped in `PreregError` with a
    clear message naming the gate — never a bare `CalledProcessError` /
    `FileNotFoundError` (errors.py convention: every wmj failure says what
    failed and which gate caught it)."""
    try:
        result = subprocess.run(
            ["git", "-C", str(repo), *args],
            check=True,
            capture_output=True,
            text=True,
            env={**os.environ},
        )
    except FileNotFoundError as exc:
        raise PreregError(
            "git executable not found — cannot certify pre-registration "
            "(the prereg gate needs git to read commit history)"
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise PreregError(
            f"`git {' '.join(args)}` failed (exit {exc.returncode}) in {repo} — "
            f"cannot certify pre-registration (is this a git repository?): "
            f"{(exc.stderr or '').strip()}"
        ) from exc
    return result.stdout


def first_added_commit(repo: Path, relpath: str) -> str:
    """The oldest commit that ADDED `relpath`, or "" if git knows no such add.

    `git log --reverse --diff-filter=A` lists add-events oldest-first; the
    first line is therefore the original add (never a later delete-then-readd).
    """
    out = _git(
        repo,
        "log",
        "--reverse",
        "--diff-filter=A",
        "--format=%H",
        "--",
        relpath,
    )
    lines = [line for line in out.splitlines() if line.strip()]
    return lines[0] if lines else ""


def _commit_timestamp(repo: Path, sha: str) -> int:
    """The committer unix timestamp of `sha`."""
    return int(_git(repo, "show", "-s", "--format=%ct", sha).strip())


def _blob_at(repo: Path, sha: str, relpath: str) -> bytes:
    """The bytes of `relpath` as it stood at commit `sha` (wrapped errors)."""
    try:
        result = subprocess.run(
            ["git", "-C", str(repo), "show", f"{sha}:{relpath}"],
            check=True,
            capture_output=True,
            env={**os.environ},
        )
    except FileNotFoundError as exc:
        raise PreregError(
            "git executable not found — cannot certify pre-registration"
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise PreregError(
            f"cannot read {relpath!r} at commit {sha[:10]} — cannot certify "
            f"pre-registration: {(exc.stderr or b'').decode('utf-8', 'replace').strip()}"
        ) from exc
    return result.stdout


def _has_uncommitted_changes(repo: Path, relpath: str) -> bool:
    return bool(_git(repo, "status", "--porcelain", "--", relpath).strip())


def _declares(text: str, name: str) -> bool:
    """True iff `name` appears in `text` as a whole token, not a substring.

    A plain `name in text` would count a model named "ir" as declared merely
    because "direct" contains the letters i-r — a silent bypass of MU-6's
    per-model entry guarantee (the exact defect P3-C07's review caught). A
    token is delimited by anything outside the model-name alphabet
    `[A-Za-z0-9_-]` (model names may contain hyphens, e.g. `fx-action-blind`),
    or by the start/end of the text.
    """
    pattern = r"(?<![A-Za-z0-9_-])" + re.escape(name) + r"(?![A-Za-z0-9_-])"
    return re.search(pattern, text) is not None


def check_prereg(
    repo: Path,
    files: Iterable[str],
    models: Iterable[_Classified],
    *,
    run_timestamp: int,
) -> str:
    """Certify the prereg/ files before a judged run; return the commit-of-record.

    Raises the specific `PreregError` subclass on the first failure, naming the
    file or model at fault. On success returns the recipe's first-added commit
    SHA — recorded in the verdict metadata so a later after-the-fact edit is
    detectable (TC-JU11-02).
    """
    repo = Path(repo)
    files = list(files)

    # The per-model entry check (below) reads recipe.md/prediction.md from the
    # working tree; that is only safe if those files went through the
    # commit/timestamp/content-hash certification in this same loop. Require
    # them in `files` so a caller that certifies only (say) thresholds cannot
    # have the entry check silently trust uncommitted recipe/prediction text.
    missing_entry_files = [f for f in _ENTRY_FILES if f not in files]
    if missing_entry_files:
        raise PreregError(
            f"check_prereg needs {list(_ENTRY_FILES)} among its certified `files` "
            f"to verify per-model entries against committed content; missing "
            f"{missing_entry_files}"
        )

    for file in files:
        relpath = f"{PREREG_DIR}/{file}"
        worktree = repo / relpath

        sha = first_added_commit(repo, relpath)
        if not sha:
            raise PreregNotCommittedError(
                f"prereg file {relpath!r} is not committed to git — it must be "
                f"committed before any judged run (models ADR-M5, MU-6)"
            )
        if _has_uncommitted_changes(repo, relpath):
            raise PreregNotCommittedError(
                f"prereg file {relpath!r} has uncommitted working-tree changes — "
                f"the judged-against content must be the committed content (MU-6)"
            )
        if not worktree.is_file():
            raise PreregNotCommittedError(
                f"prereg file {relpath!r} has a commit history but no working-tree "
                f"copy (deleted?) — there is no present content to certify (MU-6)"
            )

        ts = _commit_timestamp(repo, sha)
        if ts >= run_timestamp:
            raise PreregOrderingError(
                f"prereg file {relpath!r} was first committed at {ts}, which does "
                f"not strictly precede the judged run at {run_timestamp} — "
                f"pre-registration must come first (MU-6 / JU-11)"
            )

        first_add_hash = hashlib.sha256(_blob_at(repo, sha, relpath)).hexdigest()
        worktree_hash = hashlib.sha256(worktree.read_bytes()).hexdigest()
        if first_add_hash != worktree_hash:
            raise PreregContentError(
                f"prereg file {relpath!r} has changed since its first-added commit "
                f"{sha[:10]} — a threshold/recipe moved after pre-registration is "
                f"not pre-registered (models ADR-M5, TC-MU6-04)"
            )

    recipe_text = (repo / PREREG_DIR / "recipe.md").read_text(encoding="utf-8")
    prediction_text = (repo / PREREG_DIR / "prediction.md").read_text(encoding="utf-8")
    for model in models:
        if model.is_baseline or model.is_fixture:
            continue  # exempt: adding a baseline/fixture stays one file (TC-MU6-05)
        if not (_declares(recipe_text, model.name) and _declares(prediction_text, model.name)):
            raise PreregEntryMissingError(
                f"unrigged model {model.name!r} has no pre-registration entry in the "
                f"committed recipe and prediction — a contestant never declared in "
                f"advance is the exact gaming MU-6 prevents (TC-MU6-03)"
            )

    return first_added_commit(repo, f"{PREREG_DIR}/recipe.md")


def read_matching_margin(recipe_path: str | Path) -> float:
    """Read the pinned `matching_margin: <float>` line from recipe.md (MU-5)."""
    for line in Path(recipe_path).read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("matching_margin:"):
            return float(stripped.split(":", 1)[1].split("#", 1)[0].strip())
    raise PreregError(
        f"no 'matching_margin:' line in {recipe_path} — MU-5's margin must be "
        f"pinned in the recipe (TC-MU5-01)"
    )


def within_matching_margin(skill_a: float, skill_b: float, *, margin: float) -> bool:
    """True iff the two unrigged models' one-step skill scores are accuracy-
    matched to within the pre-registered margin (MU-5 / TC-MU5-01). A False
    means the pre-registration is not yet satisfied and judging must not
    proceed — the remedy is revising the recipe openly and re-running, never
    nudging a trained model."""
    return abs(skill_a - skill_b) <= margin
