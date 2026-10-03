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
     that was merged away (which plain `git log` hides — so history is walked
     commit by commit, every parent of every merge)
     are all refused. No `FREEZE` at all means "not frozen yet" (TC-MU6-08).
  3. **Every certified file must exist at the freeze commit, be clean in the
     working tree, and be byte-equal to its blob at the freeze commit**
     (TC-MU6-04). The certified set is fixed — `recipe.md`, `prediction.md`,
     `thresholds.json` and `FREEZE` itself — whatever list the caller passes;
     `files` can only add to it. (A caller that forgot `thresholds.json`, the
     most important judged-against content, would otherwise get a
     certificate for tuned thresholds.) The comparison is raw bytes: a
     trailing space, a byte-order mark or CRLF line endings all count. A
     CRLF-only difference raises `PreregLineEndingError`, which says so
     (the repo's `.gitattributes` pins `prereg/* -text` so a Windows
     checkout does not trigger it).
  3a. **The git history itself must be trustworthy.** A shallow clone shows
     its boundary commit as having *added* every file (so a tuned file is
     compared with itself); a `.git/info/grafts` file or a `git replace` ref
     rewrites what history says without touching any commit and without
     being pushed. Every git call here ignores replace refs and ambient
     `GIT_*` settings, and a shallow repository or a grafts file is refused
     (`PreregHistoryError`), as is a partial clone (git would run the
     repository's own fetch program to fill a gap). `repo` must be the
     repository's top level.
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
  6. **The certificate is a commit id, not the bytes the judge will read.** The
     files are checked here and read again by the caller afterwards; a change
     between the two is not detected. (`check_prereg` itself decides entries
     from the very bytes it compared.) The caller should read the files
     straight away, or re-check, and the judged run's own record names the
     freeze commit so a later comparison is possible.
  7. **Only the repository on this disk is examined.** "One freeze, ever" is
     counted over every local ref (branches, tags, remote-tracking refs, notes),
     so a fork that re-freezes is refused — but a re-freeze that exists only in
     a repository nobody has fetched from is invisible (the same shape as 2),
     and so is one parked under `refs/stash` or `refs/replace/` (both are
     deliberately left out: a stash is not history, and a replace ref is
     ignored everywhere here). Deleting a branch is likewise rewriting history
     (residual 1).
"""

from __future__ import annotations

import numbers
import os
import re
import subprocess
from pathlib import Path
from typing import Iterable, Protocol

from wmj.errors import WmjError

PREREG_DIR = "prereg"
FREEZE_FILE = "FREEZE"
# The fixed set that is locked, whatever list a caller passes (see rule 3).
CERTIFIED_FILES = ("recipe.md", "prediction.md", "thresholds.json", FREEZE_FILE)


class PreregError(WmjError):
    """Base for every pre-registration certification failure."""


class PreregNotCommittedError(PreregError):
    """A prereg/ file is missing from git or has uncommitted working-tree edits."""


class PreregNotFrozenError(PreregError):
    """There is no freeze yet, or a certified file was never part of it."""


class PreregHistoryError(PreregError):
    """Git history cannot be trusted: shallow or partial clone, grafts file, failed fsck, or
    a parent commit that history does not contain."""


class PreregRefrozenError(PreregError):
    """`prereg/FREEZE` was added more than once, or ever deleted (TC-MU6-06)."""


class PreregOrderingError(PreregError):
    """The freeze commit does not strictly precede the judged run (TC-MU6-01)."""


class PreregContentError(PreregError):
    """A frozen prereg/ file's content differs from its blob at the freeze commit."""


class PreregLineEndingError(PreregContentError):
    """A frozen file differs from its blob only in line endings (CRLF vs LF)."""


class PreregWorldConstantError(PreregError):
    """The recipe's kick settings differ from, or omit, the worlds' own (TC-MU6-09)."""


class PreregEntryMissingError(PreregError):
    """An unrigged model has no named entry in the committed recipe/prediction."""


class _Classified(Protocol):
    name: str
    is_baseline: bool
    is_fixture: bool


def _git_env() -> dict[str, str]:
    """The environment every git call runs in.

    Every ambient `GIT_*` variable is dropped (a caller inside a git hook has
    `GIT_DIR` set and would otherwise read a different repository; others can
    inject config), user and system config are switched off, and replace refs
    are ignored — a local `git replace` could otherwise make `git show` serve
    different bytes for the freeze commit than the public history holds — and
    lazy fetching is off (see below).
    """
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    env.update(
        {
            "GIT_NO_REPLACE_OBJECTS": "1",
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_OPTIONAL_LOCKS": "0",
            # No call here may fetch anything: in a partial clone git would
            # otherwise run the repo's own `remote.*.uploadpack` program to
            # fetch a missing object, mid-check (partial clones are also
            # refused outright in `_require_trustworthy_history`).
            "GIT_NO_LAZY_FETCH": "1",
        }
    )
    return env


# Repo-local config is under the author's control, so each of these is pinned
# on the command line (which outranks the repo's .git/config):
#   core.commitGraph=false    a tampered commit-graph file can rewrite parentage
#                             without touching a commit, a ref or a grafts file
#   core.useReplaceRefs=false replace refs ignored. Redundant with the
#                             GIT_NO_REPLACE_OBJECTS env var above: removing either one
#                             alone changes nothing, removing both fails the test.
#   log.showSignature=false   gpg chatter must not appear in parsed output
#   core.fsmonitor=false, core.hooksPath=<null>   no repo-supplied program runs
_PINNED_CONFIG = (
    "core.commitGraph=false",
    "core.useReplaceRefs=false",
    "log.showSignature=false",
    "core.fsmonitor=false",
    f"core.hooksPath={os.devnull}",
)


def _git_argv(repo: Path, *args: str) -> list[str]:
    pinned = [part for setting in _PINNED_CONFIG for part in ("-c", setting)]
    return ["git", "-C", str(repo), *pinned, *args]


def _run_git(
    repo: Path, *args: str, stdin: str | None = None, binary: bool = False
) -> subprocess.CompletedProcess:
    """Run git under the hardened environment; never raises on a non-zero exit."""
    try:
        return subprocess.run(
            _git_argv(repo, *args),
            check=False,
            capture_output=True,
            text=not binary,
            input=stdin,
            env=_git_env(),
        )
    except FileNotFoundError as exc:
        raise PreregError(
            "git executable not found — cannot certify pre-registration "
            "(the prereg gate needs git to read commit history)"
        ) from exc


def _stderr_text(result: subprocess.CompletedProcess) -> str:
    err = result.stderr
    if isinstance(err, bytes):
        err = err.decode("utf-8", "replace")
    return (err or "").strip()


def _git(repo: Path, *args: str, stdin: str | None = None) -> str:
    """Run `git -C repo <args>` and return stdout (text).

    A git failure (not a repo, git absent) is wrapped in `PreregError` with a
    clear message naming the gate — never a bare `CalledProcessError` /
    `FileNotFoundError` (errors.py convention: every wmj failure says what
    failed and which gate caught it)."""
    result = _run_git(repo, *args, stdin=stdin)
    if result.returncode != 0:
        raise PreregError(
            f"`git {' '.join(args)}` failed (exit {result.returncode}) in {repo} — "
            f"cannot certify pre-registration (is this a git repository?): "
            f"{_stderr_text(result)}"
        )
    return result.stdout


def _refuse_partial_clone(repo: Path) -> None:
    """Refuse a partial (`--filter=`) clone: it has objects it has not fetched.

    Git would fetch a missing object on demand from the `origin` remote by running
    the program that repo's own config names (`remote.origin.uploadpack`),
    so a "read-only" check could run author-chosen code and change files on
    disk mid-check (executed, design-review-010 review pass 3). Refusing is
    simpler and safer than trying to make that fetch harmless.
    """
    why = None
    if _run_git(repo, "config", "--get", "extensions.partialclone").returncode == 0:
        why = "extensions.partialclone is set"
    elif _run_git(repo, "config", "--get-regexp", r"^remote\..*\.promisor$").returncode == 0:
        why = "a remote is marked as a promisor"
    else:
        packdir = Path(_git(repo, "rev-parse", "--git-path", "objects/pack").strip())
        if not packdir.is_absolute():
            packdir = repo / packdir
        if packdir.is_dir() and any(packdir.glob("*.promisor")):
            why = "the object store holds promisor packs"
    if why:
        raise PreregHistoryError(
            f"this is a partial clone ({why}) — objects it has not fetched would be "
            f"fetched on demand by a program the repository's own config names, in the "
            f"middle of the check; clone the full history before certifying the "
            f"pre-registration (models ADR-M5)"
        )


def _require_trustworthy_history(repo: Path) -> None:
    """Refuse a repository whose history git itself cannot vouch for."""
    top = _git(repo, "rev-parse", "--show-toplevel").strip()
    if os.path.realpath(top) != os.path.realpath(repo):
        raise PreregError(
            f"{repo} is not the top level of its git repository ({top}) — `check_prereg` "
            f"must be given the repository's top level, because history paths and file "
            f"paths are resolved from different roots otherwise"
        )
    _refuse_partial_clone(repo)
    if _git(repo, "rev-parse", "--is-shallow-repository").strip() == "true":
        raise PreregHistoryError(
            "this is a shallow clone — git shows its boundary commit as having added "
            "every file, so a changed file would be compared with itself; fetch the full "
            "history before certifying the pre-registration (models ADR-M5)"
        )
    grafts = Path(_git(repo, "rev-parse", "--git-path", "info/grafts").strip())
    if not grafts.is_absolute():
        grafts = repo / grafts
    if grafts.exists():
        raise PreregHistoryError(
            f"{grafts} exists — a grafts file rewrites what history says without "
            f"changing any commit, so the pre-registration cannot be certified "
            f"against it (models ADR-M5)"
        )
    # Object integrity: `git show` serves whatever bytes a loose object file
    # holds, even if they do not hash to its name, so a forged object would make
    # every later comparison look clean. fsck re-hashes every object and checks
    # pack checksums (cheap at this project's size).
    fsck = _run_git(repo, "fsck", "--no-dangling", "--no-progress")
    if fsck.returncode != 0:
        raise PreregHistoryError(
            f"`git fsck` reports a damaged or forged object store in {repo} — the "
            f"pre-registration cannot be certified against objects git cannot verify: "
            f"{(_stderr_text(fsck) or fsck.stdout or '').strip()[:300]}"
        )


_ALL_REFS = ("--exclude=refs/replace/*", "--exclude=refs/stash", "--all")


def _freeze_history(repo: Path, relpath: str) -> tuple[list[str], list[str]]:
    """`(adds, deletes)`: commits where `relpath` came into or went out of existence.

    Decided by *presence in each commit's tree*, not by diff output. A merge
    commit has no diff for `git log --diff-filter` to show (even with
    `--full-history`), so a freeze added or lifted inside a merge — with an
    honest date and no history rewrite — would be invisible to a diff-based
    count (executed, design-review-010 review pass 2). Instead: every commit
    reachable from HEAD *or from any other ref* (a branch forked from before the
    freeze can re-freeze without touching HEAD's own ancestry) is asked whether
    the path exists as a file; an *add* is
    a commit that has it while none of its parents do (or a root), a *delete*
    is a commit without it while some parent has it. A rename away and back is
    a delete plus an add at the path.
    """
    parents: dict[str, list[str]] = {}
    for row in _git(repo, "rev-list", "--parents", *_ALL_REFS, "HEAD", "--").splitlines():
        shas = row.split()
        if shas:
            parents[shas[0]] = shas[1:]
    queries = "".join(f"{sha}:{relpath}\n" for sha in parents)
    answers = _git(repo, "cat-file", "--batch-check", stdin=queries).splitlines()
    if len(answers) != len(parents):
        raise PreregError(
            f"git answered {len(answers)} object queries for {len(parents)} commits — "
            f"cannot establish the history of {relpath!r}"
        )
    present: dict[str, bool] = {}
    for sha, answer in zip(parents, answers):
        tokens = answer.split()
        present[sha] = len(tokens) == 3 and tokens[1] == "blob"
    for sha, its_parents in parents.items():
        for parent in its_parents:
            # Defence in depth: unreachable while fsck, shallow and grafts refusals
            # hold (rev-list lists every ancestor), so no test can reach it; it
            # stays so a future change to the walk fails loudly, not silently.
            if parent not in present:
                raise PreregHistoryError(
                    f"commit {sha[:10]} names a parent {parent[:10]} that history does not "
                    f"contain — the history is incomplete (models ADR-M5)"
                )
    adds = [c for c in parents if present[c] and not any(present[p] for p in parents[c])]
    deletes = [c for c in parents if not present[c] and any(present[p] for p in parents[c])]
    return adds, deletes


def _head_ancestry(repo: Path) -> set[str]:
    return set(_git(repo, "rev-list", "HEAD", "--").split())


def freeze_commit(repo: Path) -> str:
    """The commit that added `prereg/FREEZE` — the one and only freeze.

    Raises `PreregNotFrozenError` if `FREEZE` was never committed, and
    `PreregRefrozenError` unless history shows exactly one add and no delete
    (models ADR-M5, TC-MU6-06/-08).
    """
    repo = Path(repo)
    _require_trustworthy_history(repo)
    relpath = f"{PREREG_DIR}/{FREEZE_FILE}"
    adds, deletes = _freeze_history(repo, relpath)
    if not adds:
        raise PreregNotFrozenError(
            f"{relpath!r} has never been committed — the pre-registration is not frozen "
            f"yet; the recipe, prediction and thresholds must be locked by committing "
            f"it before any judged run (models ADR-M5, TC-MU6-08)"
        )
    if len(adds) > 1 or deletes:  # counted over every ref, not only the checked-out one
        raise PreregRefrozenError(
            f"{relpath!r} was added {len(adds)} time(s) (at {', '.join(c[:10] for c in adds)}) "
            f"and deleted {len(deletes)} time(s) (at {', '.join(c[:10] for c in deletes) or 'nowhere'}) "
            f"in the commits reachable from any local ref — if a stale branch or tag from a "
            f"squash-merged branch is the cause, delete that ref; the freeze happens once, "
            f"ever; a second freeze, or a "
            f"delete-and-re-add, lets the locked content move (models ADR-M5, TC-MU6-06)"
        )
    if adds[0] not in _head_ancestry(repo):
        raise PreregNotFrozenError(
            f"{relpath!r} was added at {adds[0][:10]}, which is not part of the checked-out "
            f"history — the pre-registration is not frozen on the branch being judged "
            f"(models ADR-M5, TC-MU6-08)"
        )
    return adds[0]


def _exists_at(repo: Path, rev: str, relpath: str) -> bool:
    """True iff `relpath` is part of the tree at `rev` (a commit sha or HEAD)."""
    return _run_git(repo, "cat-file", "-e", f"{rev}:{relpath}").returncode == 0


def _commit_timestamp(repo: Path, sha: str) -> int:
    """The committer unix timestamp of `sha`."""
    text = _git(repo, "show", "-s", "--format=%ct", sha, "--").strip()
    try:
        return int(text)
    except ValueError as exc:
        raise PreregError(
            f"git gave {text!r} as the committer time of {sha[:10]}, not a unix timestamp "
            f"— cannot order the freeze against the run"
        ) from exc


def _blob_at(repo: Path, rev: str, relpath: str) -> bytes:
    """The bytes of `relpath` as it stood at `rev` (wrapped errors)."""
    result = _run_git(repo, "show", f"{rev}:{relpath}", binary=True)
    if result.returncode != 0:
        raise PreregError(
            f"cannot read {relpath!r} at {rev[:10]} — cannot certify pre-registration: "
            f"{_stderr_text(result)}"
        )
    return result.stdout


def _line_endings_only(a: bytes, b: bytes) -> bool:
    return a.replace(b"\r\n", b"\n") == b.replace(b"\r\n", b"\n")


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


_PLAIN_DECIMAL = re.compile(r"[-+]?(?:[0-9]+\.?[0-9]*|\.[0-9]+)(?:[eE][-+]?[0-9]+)?")


def check_recipe_world_constants(recipe_text: str) -> None:
    """Raise unless the recipe's four kick keys equal the worlds' constants.

    In plain words: the recipe says how often and how hard each world is
    kicked; the worlds say the same thing in code. Once the recipe is frozen
    the two must agree exactly, so nobody can change a kick setting in code
    after the lock and have the frozen recipe quietly describe something else
    (TC-MU6-09). Each key must appear exactly once, at the start of a line,
    with a plain decimal number; a missing, repeated, indented or
    oddly-spelled key is refused, not skipped.
    """
    from wmj.worlds import lv, pendulum  # harness may import worlds; kept local

    modules = {"lv": lv, "pendulum": pendulum}
    for key, world, attribute in _WORLD_KEYS:
        found = re.findall(rf"^{re.escape(key)}:[ \t]*([^#\s]*)", recipe_text, re.MULTILINE)
        if not found:
            raise PreregWorldConstantError(
                f"the recipe has no '{key}:' line — the worlds' kick settings must be "
                f"pinned in the frozen recipe, one per line, starting at the left margin "
                f"(TC-MU6-09)"
            )
        if len(found) > 1:
            raise PreregWorldConstantError(
                f"the recipe pins '{key}' more than once ({found}) — a later line could "
                f"disagree with the one that was checked (TC-MU6-09)"
            )
        if not _PLAIN_DECIMAL.fullmatch(found[0]):
            raise PreregWorldConstantError(
                f"the recipe's '{key}:' value {found[0]!r} is not a plain decimal number "
                f"(TC-MU6-09)"
            )
        recipe_value = float(found[0])
        world_value = getattr(modules[world], attribute)
        if recipe_value != world_value:
            raise PreregWorldConstantError(
                f"the recipe pins {key} = {recipe_value!r} but wmj.worlds.{world}.{attribute} "
                f"is {world_value!r} — the frozen recipe and the code must agree "
                f"(TC-MU6-09)"
            )


def _decode(data: bytes, label: str) -> str:
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise PreregError(
            f"{label} is not valid UTF-8 text — the per-model entry check cannot read it"
        ) from exc


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
    if isinstance(run_timestamp, bool) or not isinstance(run_timestamp, numbers.Integral):
        raise PreregError(
            f"run_timestamp must be a plain integer of unix seconds, got {run_timestamp!r} "
            f"— a NaN or a float would make the ordering comparison silently false"
        )
    repo = Path(repo)
    extras = list(files)
    for name in extras:
        if not name.strip() or Path(name).name != name or name in (".", ".."):
            raise PreregError(
                f"certified file {name!r} must be a plain file name inside {PREREG_DIR}/ "
                f"(no directories, no '..')"
            )

    freeze_sha = freeze_commit(repo)  # raises if not frozen, frozen twice, or history untrusted

    # The certified set is fixed; the caller's list can only add to it.
    certified = list(dict.fromkeys([*CERTIFIED_FILES, *extras]))
    verified: dict[str, bytes] = {}  # the exact bytes that passed the comparison
    for file in certified:
        relpath = f"{PREREG_DIR}/{file}"
        worktree = repo / relpath

        if not _exists_at(repo, freeze_sha, relpath):
            raise PreregNotFrozenError(
                f"prereg file {relpath!r} was not part of the frozen commit "
                f"{freeze_sha[:10]} — a file added after the freeze was never locked "
                f"(models ADR-M5)"
            )
        if (repo / PREREG_DIR).is_symlink() or worktree.is_symlink() or not worktree.is_file():
            what = (
                "is a symlink, or sits in a symlinked directory"
                if (repo / PREREG_DIR).is_symlink() or worktree.is_symlink()
                else "has no working-tree copy (deleted?)"
            )
            raise PreregNotCommittedError(
                f"prereg file {relpath!r} {what} — there is no present file content to "
                f"certify (MU-6)"
            )
        if not _exists_at(repo, "HEAD", relpath):
            raise PreregNotCommittedError(
                f"prereg file {relpath!r} is not at HEAD (deleted in a later commit, or "
                f"re-created uncommitted) — the judged-against content must be committed "
                f"(MU-6)"
            )
        # Compare bytes with the committed blobs directly — never `git status`,
        # which would run any clean filter or fsmonitor program the repo's own
        # config names, and can be told a modified file is unchanged.
        actual_bytes = worktree.read_bytes()
        head_bytes = _blob_at(repo, "HEAD", relpath)
        frozen_bytes = _blob_at(repo, freeze_sha, relpath)
        if actual_bytes != head_bytes and not _line_endings_only(actual_bytes, head_bytes):
            raise PreregNotCommittedError(
                f"prereg file {relpath!r} has uncommitted working-tree changes — "
                f"the judged-against content must be the committed, frozen content (MU-6)"
            )
        if actual_bytes != frozen_bytes:
            if _line_endings_only(actual_bytes, frozen_bytes):
                raise PreregLineEndingError(
                    f"prereg file {relpath!r} differs from the freeze commit "
                    f"{freeze_sha[:10]} only in line endings (CRLF vs LF) — if this is a "
                    f"Windows checkout, set core.autocrlf=false or keep the repo's "
                    f"`prereg/* -text` line in .gitattributes; if the file was "
                    f"re-saved, restore the frozen bytes (models ADR-M5, TC-MU6-04)"
                )
            raise PreregContentError(
                f"prereg file {relpath!r} has changed since the freeze commit "
                f"{freeze_sha[:10]} — a threshold/recipe moved after the lock is not "
                f"pre-registered (models ADR-M5, TC-MU6-04)"
            )
        verified[file] = actual_bytes

    freeze_ts = _commit_timestamp(repo, freeze_sha)
    if freeze_ts >= run_timestamp:
        raise PreregOrderingError(
            f"the freeze commit {freeze_sha[:10]} was made at {freeze_ts}, which does not "
            f"strictly precede the judged run at {run_timestamp} — the lock must come "
            f"first (MU-6 / JU-11, TC-MU6-01)"
        )

    # Decoded from the bytes that were just compared — never re-read from disk.
    recipe_text = _decode(verified["recipe.md"], f"{PREREG_DIR}/recipe.md")
    prediction_text = _decode(verified["prediction.md"], f"{PREREG_DIR}/prediction.md")
    check_recipe_world_constants(recipe_text)
    for model in models:
        if model.is_baseline or model.is_fixture:
            continue  # exempt: adding a baseline/fixture stays one file (TC-MU6-05)
        if not model.name.strip():
            raise PreregEntryMissingError(
                "an unrigged model has a blank name — it cannot have a pre-registration "
                "entry, and an empty name would match everywhere (TC-MU6-03)"
            )
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
