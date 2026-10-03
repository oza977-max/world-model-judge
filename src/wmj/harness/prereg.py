"""wmj.harness.prereg — MU-6 / JU-11 certification: you can't move a threshold.

In plain words: before the judge is allowed to grade an unrigged model, this
checks that the recipe, the ranking prediction, and the thresholds were
**frozen** — locked by one commit that added a small file, `prereg/FREEZE` —
before the run, and have not changed by a single byte since. That is what makes
the pre-registration real rather than a promise. Until the freeze, the recipe
may be revised openly (each revision dated in its own log); after it, nothing
locked may change, and the freeze itself can happen once, ever. The judge
itself cannot read git (it is a pure function, JU-12), so the harness owns this
check and hands the judge only the thresholds, as data.

The rules (models ADR-M5, design-review-010):
  1. **The freeze commit is the commit that added `prereg/FREEZE`** — read from
     git history, never typed or chosen. `FREEZE` holds only a human-readable
     declaration; a commit cannot name its own id, and the git tag
     `prereg-freeze`, if anyone makes one, is a label this module never reads
     (a tag can be moved with one command and no history rewrite).
  2. **One freeze, ever** (TC-MU6-06): exactly one add and no delete of
     `prereg/FREEZE` anywhere in history. A second freeze, a delete-then-re-add,
     a rename away and back, and a freeze added and removed on a side branch
     that was merged away (which plain `git log` hides — hence `--full-history`)
     are all refused. No `FREEZE` at all means "not frozen yet" (TC-MU6-08).
  3. **Every certified file — `recipe.md`, `prediction.md`, `thresholds.json`
     and `FREEZE` itself — must exist at the freeze commit, be clean in the
     working tree, and hash byte-equal to its blob at the freeze commit**
     (TC-MU6-04).
  4. **The freeze commit's timestamp must strictly precede the run**
     (TC-MU6-01).
  5. For every *unrigged* model (`not is_baseline and not is_fixture`): its
     name appears as a whole token in the committed recipe and prediction
     (TC-MU6-03). Baselines and fixtures are exempt (TC-MU6-05).
  6. **The recipe's kick settings equal the worlds' own** (TC-MU6-09), so the
     frozen recipe pins them and the code cannot drift away silently.
`check_prereg` returns the freeze commit's SHA, which the harness records as
`meta.prereg_commit` in every verdict (TC-MU6-07).

This is a harness module: it shells out to git and reads files, which the
pure judge may not. Nothing here reads the network.

**Disclosed residuals — what `check_prereg` does NOT close (named, not
solved; models ADR-M5's five residuals).** These are inherent to a
single-author, offline, no-server-side-witness project; no mechanical control
here fully closes them, so they are disclosed rather than pretended away:
  1. **Rewritable git history.** A determined author can rewrite the freeze
     commit itself. The content check closes ordinary, honestly-dated drift,
     which is the likelier accident, not deliberate history surgery.
  2. **Non-publication.** Nothing here can show that a judged run happened if
     its output was never committed.
  3. **Commit-timestamp forgeability.** The ordering check reads the committer
     date, which the committer supplies (`git commit --date=...`,
     `GIT_COMMITTER_DATE`). A run executed, disliked, then "frozen" with a
     back-dated commit passes the ordering check.
  4. **Entry substance** (backlog A13): the per-model check confirms a model's
     name appears as a token in the committed recipe and prediction; it cannot
     verify the surrounding text is a genuine recipe/prediction for that model.
  5. **No evaluation before the freeze is a rule, not a mechanism.** No
     evaluation-trial metric of either unrigged model may be computed before
     the freeze. An unrecorded informal run before the freeze leaves no trace.
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
FREEZE_FILE = "FREEZE"

# The files whose committed content the per-model entry check reads — they
# MUST be among the certified files so that content is commit-verified, not
# read unguarded from the working tree.
_ENTRY_FILES = ("recipe.md", "prediction.md")


class PreregError(WmjError):
    """Base for every pre-registration certification failure."""


class PreregNotCommittedError(PreregError):
    """A prereg/ file is missing from git or has uncommitted working-tree edits."""


class PreregNotFrozenError(PreregError):
    """There is no freeze yet, or a certified file was never part of it."""


class PreregRefrozenError(PreregError):
    """`prereg/FREEZE` was added more than once, or ever deleted (TC-MU6-06)."""


class PreregOrderingError(PreregError):
    """The freeze commit does not strictly precede the judged run (TC-MU6-01)."""


class PreregContentError(PreregError):
    """A frozen prereg/ file's content differs from its blob at the freeze commit."""


class PreregWorldConstantError(PreregError):
    """The recipe's kick settings differ from, or omit, the worlds' own (TC-MU6-09)."""


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


def _commit_lines(repo: Path, diff_filter: str, relpath: str) -> list[str]:
    """Commits that added (`A`) or deleted (`D`) exactly `relpath`, any branch.

    `--full-history` is load-bearing: without it, git's history
    simplification can hide an add or delete made on a side branch whose
    merge resolved the file away (executed, design-review-010). No
    `--follow`: a rename shows as a delete plus an add at the path.
    """
    out = _git(
        repo, "log", "--full-history", f"--diff-filter={diff_filter}", "--format=%H", "--", relpath
    )
    return [line for line in out.splitlines() if line.strip()]


def freeze_commit(repo: Path) -> str:
    """The commit that added `prereg/FREEZE` — the one and only freeze.

    Raises `PreregNotFrozenError` if `FREEZE` was never committed, and
    `PreregRefrozenError` unless history shows exactly one add and no delete
    (models ADR-M5, TC-MU6-06/-08).
    """
    repo = Path(repo)
    relpath = f"{PREREG_DIR}/{FREEZE_FILE}"
    adds = _commit_lines(repo, "A", relpath)
    deletes = _commit_lines(repo, "D", relpath)
    if not adds:
        raise PreregNotFrozenError(
            f"{relpath!r} has never been committed — the pre-registration is not frozen "
            f"yet; the recipe, prediction and thresholds must be locked by committing "
            f"it before any judged run (models ADR-M5, TC-MU6-08)"
        )
    if len(adds) > 1 or deletes:
        raise PreregRefrozenError(
            f"{relpath!r} was added {len(adds)} time(s) and deleted {len(deletes)} "
            f"time(s) — the freeze happens once, ever; a second freeze, or a "
            f"delete-and-re-add, lets the locked content move (models ADR-M5, TC-MU6-06)"
        )
    return adds[0]


def _exists_at(repo: Path, sha: str, relpath: str) -> bool:
    """True iff `relpath` is part of the tree at commit `sha`."""
    try:
        result = subprocess.run(
            ["git", "-C", str(repo), "cat-file", "-e", f"{sha}:{relpath}"],
            check=False,
            capture_output=True,
            env={**os.environ},
        )
    except FileNotFoundError as exc:
        raise PreregError(
            "git executable not found — cannot certify pre-registration"
        ) from exc
    return result.returncode == 0


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


_WORLD_KEYS = (
    ("lv_kick_rate_per_s", "lv", "KICK_RATE_PER_S"),
    ("lv_action_max", "lv", "TRAINED_ACTION_MAX"),
    ("pendulum_kick_rate_per_s", "pendulum", "KICK_RATE_PER_S"),
    ("pendulum_action_max", "pendulum", "TRAINED_ACTION_MAX"),
)


def check_recipe_world_constants(recipe_text: str) -> None:
    """Raise unless the recipe's four kick keys equal the worlds' constants.

    In plain words: the recipe says how often and how hard each world is
    kicked; the worlds say the same thing in code. Once the recipe is frozen
    the two must agree exactly, so nobody can change a kick setting in code
    after the lock and have the frozen recipe quietly describe something else
    (TC-MU6-09). A missing key is refused, not skipped.
    """
    from wmj.worlds import lv, pendulum  # harness may import worlds; kept local

    modules = {"lv": lv, "pendulum": pendulum}
    for key, world, attribute in _WORLD_KEYS:
        match = re.search(rf"^{re.escape(key)}:\s*([^#\s]+)", recipe_text, re.MULTILINE)
        if match is None:
            raise PreregWorldConstantError(
                f"the recipe has no '{key}:' line — the worlds' kick settings must be "
                f"pinned in the frozen recipe (TC-MU6-09)"
            )
        try:
            recipe_value = float(match.group(1))
        except ValueError as exc:
            raise PreregWorldConstantError(
                f"the recipe's '{key}:' value {match.group(1)!r} is not a number (TC-MU6-09)"
            ) from exc
        world_value = getattr(modules[world], attribute)
        if recipe_value != world_value:
            raise PreregWorldConstantError(
                f"the recipe pins {key} = {recipe_value!r} but wmj.worlds.{world}.{attribute} "
                f"is {world_value!r} — the frozen recipe and the code must agree "
                f"(TC-MU6-09)"
            )


def check_prereg(
    repo: Path,
    files: Iterable[str],
    models: Iterable[_Classified],
    *,
    run_timestamp: int,
) -> str:
    """Certify the frozen prereg/ files before a judged run; return the freeze commit.

    Raises the specific `PreregError` subclass on the first failure, naming the
    file or model at fault. On success returns the freeze commit's SHA — the
    commit that added `prereg/FREEZE` — recorded in the verdict metadata so a
    later after-the-fact edit is detectable (TC-MU6-07, TC-JU11-02).
    """
    repo = Path(repo)
    files = list(files)

    # The per-model entry check (below) reads recipe.md/prediction.md; that is
    # only safe if those files went through the certification in this same
    # loop. Require them in `files` so a caller that certifies only (say)
    # thresholds cannot have the entry check trust unverified text.
    missing_entry_files = [f for f in _ENTRY_FILES if f not in files]
    if missing_entry_files:
        raise PreregError(
            f"check_prereg needs {list(_ENTRY_FILES)} among its certified `files` "
            f"to verify per-model entries against committed content; missing "
            f"{missing_entry_files}"
        )

    freeze_sha = freeze_commit(repo)  # raises if not frozen, or frozen twice

    # FREEZE is certified like the rest: it cannot be edited after the fact.
    certified = [*files, *([FREEZE_FILE] if FREEZE_FILE not in files else [])]
    for file in certified:
        relpath = f"{PREREG_DIR}/{file}"
        worktree = repo / relpath

        if not _exists_at(repo, freeze_sha, relpath):
            raise PreregNotFrozenError(
                f"prereg file {relpath!r} was not part of the frozen commit "
                f"{freeze_sha[:10]} — a file added after the freeze was never locked "
                f"(models ADR-M5)"
            )
        if _has_uncommitted_changes(repo, relpath):
            raise PreregNotCommittedError(
                f"prereg file {relpath!r} has uncommitted working-tree changes — "
                f"the judged-against content must be the committed, frozen content (MU-6)"
            )
        if not worktree.is_file():
            raise PreregNotCommittedError(
                f"prereg file {relpath!r} is frozen but has no working-tree copy "
                f"(deleted?) — there is no present content to certify (MU-6)"
            )
        frozen_hash = hashlib.sha256(_blob_at(repo, freeze_sha, relpath)).hexdigest()
        worktree_hash = hashlib.sha256(worktree.read_bytes()).hexdigest()
        if frozen_hash != worktree_hash:
            raise PreregContentError(
                f"prereg file {relpath!r} has changed since the freeze commit "
                f"{freeze_sha[:10]} — a threshold/recipe moved after the lock is not "
                f"pre-registered (models ADR-M5, TC-MU6-04)"
            )

    freeze_ts = _commit_timestamp(repo, freeze_sha)
    if freeze_ts >= run_timestamp:
        raise PreregOrderingError(
            f"the freeze commit {freeze_sha[:10]} was made at {freeze_ts}, which does not "
            f"strictly precede the judged run at {run_timestamp} — the lock must come "
            f"first (MU-6 / JU-11, TC-MU6-01)"
        )

    recipe_text = (repo / PREREG_DIR / "recipe.md").read_text(encoding="utf-8")
    prediction_text = (repo / PREREG_DIR / "prediction.md").read_text(encoding="utf-8")
    check_recipe_world_constants(recipe_text)
    for model in models:
        if model.is_baseline or model.is_fixture:
            continue  # exempt: adding a baseline/fixture stays one file (TC-MU6-05)
        if not (_declares(recipe_text, model.name) and _declares(prediction_text, model.name)):
            raise PreregEntryMissingError(
                f"unrigged model {model.name!r} has no pre-registration entry in the "
                f"frozen recipe and prediction — a contestant never declared in "
                f"advance is the exact gaming MU-6 prevents (TC-MU6-03)"
            )

    return freeze_sha


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
