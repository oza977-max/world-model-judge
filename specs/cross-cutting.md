# World Model Judge — Cross-Cutting Specification

Version 1.3 · 30 August 2026 · Derived from Requirements v1.2 and Test Cases v1.0 (design-review fixes applied 25 and 30 Aug 2026)

> **Change note (v1.3).** Revised after `/gvm-design-review` design-review-003 (Round 3, dual/blind — 14 independent reviewers). Fixed: the NF-4 scan now also walks historical file-content blobs, not just the current tree and commit messages; the pre-commit hook now has a named, committed installation path (`.githooks/pre-commit` + a documented one-time `git config` step), with the run-time scan disclosed as a real, hook-independent safety net; the AST import-graph gate grew from three checks to five with individually-numbered test IDs (TC-NF6-01..05) — banning `importlib`/`__import__` outright rather than enumerating call shapes (closing alias/`getattr` evasions), banning `exec`/`eval` outright, and catching `numpy.random.*` legacy-global calls; the ambient-module blocklist widened to include `random`/`threading`/`multiprocessing`/`platform`/`uuid`; `wmj.models.registry`'s auto-discovery mechanism is now named explicitly (`pkgutil.iter_modules`); TC-MU9-01's enforcement mechanism was redesigned from a self-referential git-diff test (found structurally incapable of failing) to a structural test of the actual architectural property; the spec-parity check is now traced to NF-5 with a test case and a build chunk. See design-review-003.html for the full findings.

> **Change note (v1.2).** Revised after `/gvm-design-review` design-review-002 (Round 2, independent re-check under strict criteria, including a new Security panel). Fixed: the v1.1 NF-4 confidentiality-scan mechanism instructed a maintainer to write the actual forbidden term into a tracked source file (`wmj/harness/confidentiality.py`) — following it as written would cause the exact violation it exists to prevent; redesigned around a gitignored local terms file, checked pre-commit, with the residual gap named honestly rather than hidden (ADR-002.5, below). Also fixed: the NF-4 scan now covers commit messages, not file contents only; the AST import-graph gate (ADR-003) now also blocks ambient-module imports and dynamic-import calls inside the judge, not only `wmj.*` imports; ADR-002 rule 4 now names the harness-owned envelope explicitly as going through the canonical serializer; ADR-003 now names a concrete mechanism for TC-MU9-01's repository-diff check; added a lightweight `.md`/`.html` spec-parity check. See design-review-002.html for the full findings.

> **Change note (v1.1).** Revised after `/gvm-design-review` design-review-001. Removed `JudgeInput`'s `trial boundaries` field from the Data-Model Overview — judge spec v1.1 explains why the pre-shaped `[n_trials, H, d]` array design makes it unnecessary and, as previously worded, undefined. Added the `JudgedResult` envelope type (judge spec v1.1 §5) to the Data-Model Overview, since it — not the bare `Verdict` — is what reporting and `out/verdicts/` actually consume. Added the NF-4 forbidden-terms list, previously referenced by three separate specs and declared in none of them. See design-review-001.html for the full findings.

**What this document is.** The decisions every other spec depends on: the tech stack, the project structure, the determinism rules, the error-handling conventions, and the dependency budget. Domain specs (worlds, models, judge, reporting) reference this document rather than repeating it.

**In plain words:** this is the rulebook the whole build follows. If a domain spec and this document ever disagree, this document wins, and the disagreement is a bug to fix, not a choice to make.

---

## Expert Panel

| Expert | Work | Role in This Document |
|--------|------|----------------------|
| Luciano Ramalho | *Fluent Python* (2nd ed.) | Dataclasses for value types, type hints as documentation, data-model leverage |
| Harry Percival & Bob Gregory | *Architecture Patterns with Python* | Framework-independent domain core, TDD workflow, layer separation |
| Steve McConnell | *Code Complete* (2nd ed.) | Defensive programming: fail loudly at startup, never mask with defaults |
| Andrew Hunt & David Thomas | *The Pragmatic Programmer* | DRY as knowledge (not code), tracer-bullet thin slices |
| Michael Keeling | *Design It!* | ADR format used throughout |
| Ernst Hairer, Syvert Nørsett & Gerhard Wanner | *Solving Ordinary Differential Equations I* (2nd ed.), Springer (1993) | Integrator selection and error behaviour (discovered expert — see below) |
| David Goldberg | *What Every Computer Scientist Should Know About Floating-Point Arithmetic* (ACM Computing Surveys, 1991) | Floating-point determinism rules (discovered expert — see below) |

### Expert Discovery: Numerical Computing

The existing GVM roster has no specialist for numerical integration or floating-point reproducibility — both load-bearing here. Per the discovery protocol:

**Ernst Hairer, Syvert Nørsett & Gerhard Wanner** — *Solving Ordinary Differential Equations I: Nonstiff Problems* (2nd ed.), Springer (1993)
- **Method choice follows problem structure**: fixed-step explicit Runge–Kutta methods are the reference choice for smooth non-stiff systems like ours; their local truncation error is well-characterised, which is what WD-3's drift measurement quantifies.
- **Conserved quantities drift under generic integrators**: a non-symplectic method does not preserve energy or other invariants exactly — the drift is systematic, measurable, and must be bounded rather than assumed away (exactly WD-3's second clause).

**David Goldberg** — "What Every Computer Scientist Should Know About Floating-Point Arithmetic", *ACM Computing Surveys* 23(1) (1991)
- **Floating-point arithmetic is deterministic but not associative**: the same operations in the same order give the same bits; a different summation order gives different bits. Reproducibility (NF-1) is therefore an *ordering* discipline, not a hardware property.
- **Same platform, same libraries, same order → same bits**: NF-1's "machines of the same platform" scope is exactly the scope within which IEEE-754 double arithmetic is bit-reproducible.

---

## ADR-001 — Tech stack: pure NumPy scientific Python

**Decision:** Python 3.12, with NumPy as the only numerical dependency, Matplotlib as the only charting dependency, pytest as the only test dependency. The two learned models (MU-5) are hand-rolled NumPy MLPs — no deep-learning framework.

**Status:** Accepted (user decision, 25 Aug 2026).

**Context:** NF-1 requires byte-identical verdicts across runs and machines of the same platform. NF-2 requires laptop-scale runtime. NF-3 requires a small, named, long-established dependency set. The state spaces are 2–4 dimensions; the models are deliberately trivial (MU-10). [Requirement: NF-1, NF-2, NF-3, MU-5, MU-10]

**Options considered:**
1. **Pure NumPy** — hand-rolled MLPs (~150 lines including backprop). Strongest determinism (single-threaded, seeded, no framework kernels), smallest dependency set, code readable end-to-end by Dev.
2. **NumPy + PyTorch (CPU)** — standard tooling, less hand-rolled code; but a ~2 GB dependency for two tiny MLPs, framework-level nondeterminism to fight (threaded kernels, versioned numerics), and a weaker NF-3 story.
3. **Stdlib only** — maximum purity; but slow, unreadable matrix code, and Matplotlib is required for charts anyway, so the purity is spoiled regardless.

**Decision rationale:** Option 1. At this problem size a framework buys nothing and costs determinism. Writing backprop by hand is a feature for this project: the judge's subject matter is *trust in learned models*, and a reader can verify every line of what was learned.

**Consequences:** We own the correctness of the MLP implementation (mitigated: gradient-check test against finite differences, part of the models spec). Exact package versions must be pinned (see Dependency Budget). No GPU path exists, which NF-2 makes irrelevant.

---

## ADR-002 — Determinism strategy

**Decision:** Determinism is achieved by four enforced rules: (1) single-threaded execution, (2) one seeded generator per component with explicit seed plumbing, (3) no wall-clock, filesystem, or environment reads inside any computation path, (4) canonical serialization for every artefact that NF-1 compares.

**Status:** Accepted.

**Context:** NF-1 (byte-identical verdicts), WD-7 (byte-identical trajectories), MU-8 (reproducible training), JU-12 (pure judge). Assumption 5 of the requirements flags this as the assumption most likely to bite. [Requirement: NF-1, WD-7, MU-8, JU-12] [Test: TC-NF1-01, TC-NF1-02, TC-WD7-01, TC-WD7-02, TC-MU8-01, TC-JU12-01]

**The four rules, precisely:**

1. **Single-threaded.** The entry point sets `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1` before importing NumPy, and the test suite asserts NumPy reports single-threaded configuration. Unordered parallel reduction is the canonical source of run-to-run float drift (Goldberg: non-associativity); we remove it entirely. NF-2's minutes-scale budget survives this — the workloads are small.
2. **Seeding.** All randomness flows from `numpy.random.Generator(numpy.random.PCG64(seed))` instances created at the orchestration layer and passed down explicitly. No module creates its own generator from entropy; no code calls the legacy global `numpy.random.*` functions. Component seeds are derived from one run-level seed via `PCG64` jumped/spawned streams so adding a component never shifts another component's stream.
3. **No ambient inputs.** Nothing in worlds, models, or judge reads the clock, the filesystem, environment variables, or network. The judge specifically is a pure function (JU-12): its entire call graph takes arguments and returns values. File I/O happens only in the orchestration and reporting layers, at the edges. This includes run-identifying metadata — platform string, prereg commit SHA — which the harness reads and attaches to the `JudgedResult` envelope (Data-Model Overview) *after* calling the judge, never inside it (design-review fix: an earlier draft of the judge spec's Verdict schema carried a `meta` block the judge could not have produced without violating this rule). **This rule's enforcement mechanism is the import-graph gate below (ADR-003) — design-review-002 broadened that single gate to also catch ambient-module imports and dynamic-import calls, since Round 2 found the original AST-import-only scope missed both.**
4. **Canonical serialization.** One shared serializer produces every machine-readable artefact — including `wmj/harness/results.py`'s harness-owned envelope write (design-review-002 fix: this is the artefact NF-1's byte-comparison actually diffs, per judge spec v1.3 §5, and Round 2 found no prior spec made the link to this serializer explicit, unlike every other artefact in this rule) — as JSON with sorted keys, UTF-8, `\n` newlines, no trailing whitespace, floats rendered with Python's `repr` (shortest round-trip representation — bit-exact by construction), arrays as nested lists, no timestamps or hostnames anywhere in the payload. NF-1's ten-run byte comparison (TC-NF1-01) compares these bytes.

**Options considered:** (a) tolerance-based comparison instead of byte identity — rejected: NF-1 explicitly demands byte identity, and tolerances rot; (b) hash-based comparison of floats at reduced precision — rejected for the same reason; (c) the four rules above — accepted.

**Consequences:** Every function that needs randomness takes a `Generator` parameter (visible in every signature — this is a feature, not noise). Cross-*platform* identity (e.g. x86 vs ARM) is explicitly out of scope, matching NF-1's "machines of the same platform" wording; the verdict record states the platform.

---

## ADR-003 — Project structure: four packages plus a harness, judge imports nothing

**Decision:** A `src`-layout Python package `wmj` with five sub-packages: `wmj.worlds`, `wmj.models`, `wmj.judge`, `wmj.reporting`, `wmj.harness`. The judge imports only the standard library and NumPy — never the other `wmj` packages. The harness is the only package that imports everything.

**Status:** Accepted.

**Context:** NF-6 (separable layers, judge imports nothing from the others), JU-12 (pure judge), MU-9 (new model touches nothing but its own file), JU-1 (judge input structurally cannot carry identity). [Requirement: NF-6, JU-12, MU-9, JU-1] [Test: TC-NF6-01..05, TC-JU1-01, TC-MU9-01]

**Layout:**

```
wmj/
  worlds/        # Domain 1: LV + pendulum, integrator, divergence, regions, tasks
  models/        # Domain 2: baselines, fixtures, two unrigged models, training
  judge/         # Domain 3: pure functions state → verdict; imports stdlib + numpy ONLY
  reporting/     # Domain 4: charts, captions, verdict file writer
  harness/       # orchestration: seeds, data flow, pre-registration checks, CLI
tests/           # mirrors the package layout; plus tests/gates/ for the enforced gates
prereg/          # committed-before-judging artefacts: recipe, margins, thresholds (MU-6, JU-11)
out/             # generated: verdicts + charts (gitignored except published results)
```

**How the structure enforces the requirements rather than promising them:**
- `tests/gates/test_import_graph.py` walks `wmj.judge`'s AST and fails on any of **five** things (TC-NF6-01 through TC-NF6-05, one fixture per check — design-review-003 fix: Round 3 found the gate's own test-ID traceability wasn't widened when its scope was, so nothing confirmed each check actually had an exercised fixture; splitting the IDs makes that checkable):
  1. **TC-NF6-01 — `import wmj.*`** (the original check).
  2. **TC-NF6-02 — an `Import`/`ImportFrom` of an ambient-access module** (design-review-002 addition; design-review-003 widened the list after two independent Round 3 panels each found different omissions: `os`, `sys`, `time`, `datetime`, `socket`, `pathlib`, `subprocess`, `random`, `threading`, `multiprocessing`, `platform`, `uuid`).
  3. **TC-NF6-03 — any `Import`/`ImportFrom` of `importlib` itself, and any use of the name `__import__` anywhere in the AST, not only as a direct `Call` target** (design-review-003 fix, replacing the narrower "`Call` node whose target is `importlib.import_module`/`__import__`/`importlib.__import__`" check — Round 3 found the narrower form evadable by alias indirection, e.g. `import importlib as il; il.import_module(...)`, or `getattr` indirection, e.g. `getattr(importlib, "import_module")(...)`. Banning the import of `importlib` outright, and banning any reference to the `__import__` name — whether called directly, aliased to another name, or passed to `getattr` — closes the enabling statement rather than enumerating call shapes built from it; the judge has no legitimate use for either).
  4. **TC-NF6-04 — a `Call` node whose target is `exec` or `eval`** (design-review-003 addition — Round 3 found a string payload passed to either is a second, unparsed sub-language the other four checks cannot see inside; the judge has no legitimate use for either, so the ban is unconditional, not scoped to suspicious arguments).
  5. **TC-NF6-05 — an `Attribute` access of the form `<numpy-alias>.random.<name>` where `<name>` is not `Generator` or `PCG64`** (design-review-003 addition — Round 3 found `numpy.random.rand()`-style legacy-global calls invisible to every import-shaped check, since `numpy` is the one module the judge is *required* to import per judge spec §4 and can never be blocklisted; ADR-002 rule 2 explicitly forbids these calls and rule 3 names this gate as rule 2's enforcement, so the gate needed a check that actually does that).

  All five checks live in the same AST walk and the same gate; a violation of any one fails the run. This is a build gate, not a convention. **Disclosed limit, honestly stated rather than implied closed:** a denylist of AST shapes — however broadened — cannot in principle enumerate every way a sufficiently determined author could reach ambient state (LANGSEC's own point); TC-NF6-01 through -05 close every concrete evasion found across three rounds of adversarial review, not every theoretically possible one.
- The judge's input types (defined *in the judge package*, since it can import nothing) are plain dataclasses of arrays and floats with no name, id, or provenance field — model identity is structurally unrepresentable (TC-JU1-01).
- **`wmj.models.registry`'s auto-discovery mechanism, named explicitly (design-review-003 fix — Round 3 found no document ever specified how a new model file's `register()` call actually executes, given this document's own ban on `__init__.py` re-export mazes below):** `registry.all_models()` performs a `pkgutil.iter_modules` walk over `wmj/models/`'s own package directory at call time, importing every submodule it finds (excluding `base.py` and `registry.py` themselves) before returning whatever has been registered. A new model file is discovered the moment `all_models()` is next called — no edit to any other file is needed, and no hardcoded import list exists anywhere to maintain.
- **The zero-diff-outside-own-file property (TC-MU9-01), redesigned as a structural test rather than a git-diff test (design-review-003 fix — Round 3 found the v1.2 `test_registry_isolation.py` mechanism staged, committed, and diffed a fixture it fully controlled end-to-end, so the diff could only ever show the one file the test itself chose to write; it was structurally incapable of failing, regardless of what a real change did):** `tests/gates/test_registry_isolation.py` (a) writes a throwaway stub model module directly into `wmj/models/` at test time and asserts `wmj.models.registry.all_models()` discovers it via the auto-discovery mechanism above — proving a new file needs no other edit to be found; (b) statically AST-walks every other package (`wmj/judge`, `wmj/worlds`, `wmj/reporting`, `wmj/harness`) for any reference to the stub's class or function name, asserting zero matches — proving nothing outside `wmj/models/` needs to know a new model exists; (c) deletes the stub file. This test can genuinely fail: if auto-discovery is broken, part (a) fails; if some other package's code does need to reference model internals directly, part (b) fails. No git operations are involved and no scratch worktree is created, closing the undisclosed object-store-sharing concern the prior mechanism also raised.
- `prereg/` is where MU-6/JU-11 artefacts live; the harness refuses to judge unrigged models unless the committed `prereg/` files predate the run (mechanics in the models and judge specs).
- **Spec/build parity [Requirement: NF-5] (design-review-002 addition; design-review-003 fix traces it to a requirement, a test case, and a build chunk — Round 3 found it was an orphan design element with none of the three, despite the Wiring Matrix claiming completeness):** every generated `specs/*.html` and `requirements/*.html` file carries an HTML comment `<!-- generated-from: {path}.md sha256:{hash} -->` written at generation time. A pre-commit/CI check recomputes the hash of the current `.md` file and fails if it doesn't match the hash embedded in the committed `.html` twin — a byte-level parity check, not a text diff, consistent with this project's NF-1 byte-identity discipline. This does not replace human approval of the `.md`; it only guarantees the `.html` a human might have actually read was generated from that exact `.md`, not a stale or hand-edited copy. Traced to NF-5 ("no claim exceeds what the evidence supports") since a stale `.html` would let an approval and a build silently diverge; covered by TC-NF5-02 (new) and built in P6-C02 alongside the other NF gates.

**Consequences:** Data flows one way: harness pulls from worlds and models, hands plain arrays to the judge, hands the verdict to reporting. The judge cannot even *name* a model. The duplication cost (judge defines its own input dataclasses rather than importing shared types) is the price of NF-6, paid deliberately.

---

## Error-Handling Conventions

**The rule (McConnell, defensive programming): fail loudly and completely, never partially.** This project's credibility requirements make silent degradation worse than crashing:

1. **Refuse, don't improvise.** Missing baselines → the judge raises `MissingBaselineError`; no verdict is produced (TC-MU2-01). A required verdict field that cannot be computed → the run aborts; no partial record is written (TC-JU9-02). Pre-registration files missing or dated after the run → the harness refuses to judge unrigged models (TC-JU11-02).
2. **Typed exceptions, single module per package.** Each package defines its exceptions in one `errors.py`. Every exception message names what failed, what was expected, and which requirement's gate fired (e.g. `"WD-3 gate: model integrator step 0.02 != world step 0.01"`).
3. **Gates fail the run, not the assertion count.** The enforced gates (integrator match, drift bound, determinism, import graph, prereg timestamps) run both as pytest tests and as startup checks in the harness — a violated gate stops the pipeline before any output is produced.
4. **No `except: pass`, no default fallbacks for configuration.** Absent configuration is an error, per McConnell — a default that masks a missing value would quietly break pre-registration.

---

## Data-Model Overview

Full schemas live in the domain specs; the shared shapes every package agrees on are fixed here. All arrays are `numpy.float64`; all types are frozen dataclasses (Ramalho: value types as dataclasses).

| Type | Shape | Owner spec | Consumed by |
|---|---|---|---|
| `State` | `float64[d]` — d=2 (LV: prey, predator), d=4 (pendulum: θ₁, θ₂, ω₁, ω₂) | worlds | everyone |
| `Action` | `float64[a]` — a=1 both worlds (cull/restock rate; pivot impulse) | worlds | everyone |
| `Prediction` | `mean: float64[d]`, `spread: float64[d]` — per-dimension mean and standard deviation (the MU-1 fixed uncertainty format; the ensemble's mapping to this format is pre-registered, see models spec) | models | judge |
| `JudgeInput` | arrays only: predictions, spreads, outcomes, `region_labels: [n_trials]` of `{region_name, axis}` (one canonical shape, cited by worlds/models/judge specs alike — design-review-003 fix, see judge spec ADR-J4/§4), divergence curve `[H+1]`, climatology table, task tolerances — defined inside `wmj.judge`, no identity fields, no separate boundary-marker field (a trial's row in the array's trial axis *is* its boundary — design-review fix, see judge spec ADR-J4) | judge | judge |
| `Verdict` | the JU-9 record — full schema in the judge spec, serialized canonically per ADR-002. Contains only what the judge itself computes: no model identity, no fixture flag, no run metadata (design-review fix — these were removed from `Verdict` because the judge cannot honestly produce them; JU-1/JU-12) | judge | harness |
| `JudgedResult` | the harness-owned envelope `{model_ref, model_name, is_fixture, verdict, meta}` — one per (model, world); wraps a `Verdict` with the identity/metadata facts only the harness holds (design-review addition, judge spec v1.3 §5) | harness | reporting, `out/verdicts/*.json` |

**The `Prediction.spread` convention** — one standard deviation per dimension — is the single uncertainty vocabulary of the whole system (MU-1). How a model *derives* its spread is its own business; what it *means* is fixed here and never varies (TC-MU1-01, TC-MU1-02). This settles Open Question 4's format half; the ensemble's spread-mapping rule is settled in the models spec.

---

## Dependency Budget (settles the NF-3 half of Open Question 5)

**Runtime dependencies — the complete list (TC-NF3-01's `<spec-value>`):**

| Package | Pinned | Why |
|---|---|---|
| `numpy` | `>=1.26,<2.0` — exact version pinned in lockfile | all numerics |
| `matplotlib` | `>=3.8,<4.0` — exact version pinned in lockfile | the four required charts |

**Development dependencies:** `pytest` (test runner). Nothing else — no coverage plugins, no linters as *dependencies* (contributors may run whatever tools they like; the build depends on none of them).

The dependency gate (TC-NF3-01) reads `pyproject.toml` and fails if any name outside `{numpy, matplotlib}` appears in runtime dependencies. The lockfile (`requirements.lock`, pip-compiled) pins exact versions and is part of the reproducibility statement: RP-7's single documented command installs from the lockfile.

**In plain words:** two packages may be imported by shipped code, ever. The test suite enforces the list; this table is the list.

---

## Confidentiality Scan (NF-4) — the forbidden-terms list

**Design-review addition (v1.1); redesigned (v1.2).** Architecture-overview and the implementation guide both referenced "the forbidden-terms scan" and TC-NF4-01 explicitly promised the list would be "declared and kept current in the technical spec" — but no prior spec draft actually declared it. v1.1 declared it, but design-review-002 (Round 2, Panel A) found the v1.1 mechanism self-defeating: its own remediation instruction told a maintainer to write the actual forbidden term into `wmj/harness/confidentiality.py`, a tracked file on a public repository — following the ADR as written would cause the exact NF-4 violation it exists to prevent. Separately, the empty seed list meant the scan always passed vacuously regardless of repository contents. Both are fixed below.

**The mechanism (v1.2): a gitignored local terms file, checked pre-commit, with the gap named honestly.**

- The real, specific forbidden terms — if any are ever identified — live in `wmj/harness/confidentiality_terms.local.txt` (one term per line), which is listed in `.gitignore` and **is never committed**. Following this project's confidentiality rule requires writing a real term only into a file that structurally cannot reach the public repository.
- A committed template, `wmj/harness/confidentiality_terms.example.txt`, ships instead — generic placeholder guidance only ("employer name", "internal team/committee name" as category labels, no actual terms), so a fresh clone has something to copy from without ever seeing a real term.
- `scan_repository()` reads `confidentiality_terms.local.txt` if present, plus the small set of structural pattern classes that are safe to keep generic and committed (e.g., a regex for common internal-only URL/hostname shapes). It walks every tracked file in the working tree (via `git ls-files`, so it naturally excludes `out/` and other gitignored paths), **scans commit-message history** (`git log --format=%B --all`), and **(design-review-003 fix — Round 3 found the same permanence argument that motivated the commit-message scan applies equally to file content, and the v1.2 scan never covered it) walks every historical blob across every commit** (`git rev-list --objects --all` enumerated once, each blob's content fetched via `git cat-file -p` and searched the same way as the working tree) — a forbidden term committed in a file and removed in a later commit is exactly as permanently visible via `git show <sha>:<path>` as one typed into a commit message, and is now scanned the same way. Case-insensitive search for each term; fails loudly (per the Error-Handling Conventions above) listing every match's file/commit/blob and location if any are found.
- **The precondition that replaces the vacuous-pass gap:** if `confidentiality_terms.local.txt` does not exist or is empty, the **pre-commit hook** fails loudly with an instructional message ("no local confidentiality terms configured — see `confidentiality_terms.example.txt`") rather than silently reporting a clean scan. An unconfigured scan is now a stop, not a false green.
- **The hook's installation path, named explicitly (design-review-003 fix — Round 3 found three independent panels flagging that no document ever specified how a pre-commit hook reaches `.git/hooks/`, which `git clone` never populates):** the hook script is committed and tracked at `.githooks/pre-commit` (a normal file, unlike `.git/hooks/`, which is versioned like any other repository content). The repository's one-time setup step — documented in the README and named explicitly in this ADR rather than left to be inferred — is `git config core.hooksPath .githooks`, run once per clone. This makes the hook's existence auditable (it's a file in the repository anyone can read) even though its *activation* still depends on that one command having been run.
- **The residual gap, narrower now and named rather than hidden (matching this project's own established pattern — models spec ADR-M5's git-rewritability disclosure):** a fresh clone that never runs the one-time `git config` step has no active hook, and a commit made with `git commit --no-verify` bypasses it regardless. CI cannot read the gitignored local file either, so CI's own re-run of `scan_repository()` only ever has the committed structural patterns and historical-blob scan to check against — real terms are enforceable at commit time only locally, by the one author who holds them. What is **not** hook-dependent: `python -m wmj run`/`verify` (P6-C02) re-runs the full scan — content, commit messages, and historical blobs — every time the product actually executes, which is a real, non-hypothetical check independent of whether the hook was ever installed, even though it fires at run time rather than commit time. No technical mechanism in a single-author, no-server-side-review project fully closes the commit-time gap without infrastructure this toy-scale project doesn't otherwise need; it is disclosed here rather than presented as solved.

**Maintenance:** `confidentiality_terms.local.txt` is append-only by the same git-hygiene discipline as `prereg/` in spirit (though, being gitignored, its own history isn't tracked by git — the discipline is a personal-process one, not a mechanical one, and is named as such).

---

## Development Conventions

- **TDD, strictly (Beck, via test-cases.md):** each build chunk starts from its named TC-IDs; the failing test precedes the code. Tests are co-located per package under `tests/`, named for behaviour (`test_wd3_gate_fails_on_step_size_mismatch`, per Metz).
- **Type hints on every public signature** (Ramalho); `from __future__ import annotations` throughout. No runtime type-checking dependency — hints are documentation and IDE fuel, per NF-3.
- **Plain-English docstrings carry the "In plain words" discipline (NF-5 territory):** every public module and function docstring says what it does in ordinary language. A reader of the source meets the same voice as a reader of the requirements.
- **Naming:** requirement IDs appear in gate-test names and error messages, so a failing check names the requirement it protects.
- **No `__init__.py` re-export mazes:** import paths mirror the layout; `wmj.judge.calibration` is where calibration lives.
- **Git hygiene:** `out/` artefacts are regenerated, never hand-edited; `prereg/` files are append-only once committed (their history *is* the pre-registration evidence).

---

## Traceability

| This section | Requirements | Test cases |
|---|---|---|
| ADR-001 stack | NF-2, NF-3, MU-5, MU-10 | TC-NF2-01, TC-NF3-01 |
| ADR-002 determinism | NF-1, WD-7, MU-8, JU-12 | TC-NF1-01/02, TC-WD7-01/02, TC-MU8-01, TC-JU12-01 |
| ADR-003 structure | NF-6, JU-1, JU-12, MU-9 | TC-NF6-01..05, TC-JU1-01, TC-MU9-01, TC-JU12-01 |
| Spec/build parity | NF-5 | TC-NF5-02 |
| Error handling | MU-2, JU-9, JU-11, NF-5 | TC-MU2-01, TC-JU9-02, TC-JU11-02 |
| Data model | MU-1, WD-2 | TC-MU1-01, TC-MU1-02 |
| Dependency budget | NF-3 | TC-NF3-01 |
| Confidentiality scan (design-review addition) | NF-4 | TC-NF4-01 |

Open Questions settled here: **OQ-5 (dependency half)** — the named list is `{numpy, matplotlib}` + dev `pytest`. OQ-5's runtime-budget half is settled in the judge spec alongside sample size (OQ-1/OQ-2), as the requirements direct. **OQ-4 (format half)** — per-dimension mean + one standard deviation; the ensemble mapping is settled in the models spec.

---

## Changelog

| Version | Date | Change |
|---|---|---|
| 1.0 | 2026-08-25 | Initial version. Stack decision (ADR-001) made by user; determinism and structure ADRs derived from requirements v1.2. |
| 1.1 | 2026-08-25 | Design-review fixes (design-review-001): removed the undefined `trial_boundaries` field, added the `JudgedResult` envelope type, clarified ADR-002 rule 3 to explicitly exclude run metadata from the judge's purity boundary, and declared the NF-4 forbidden-terms list and scan mechanism (previously referenced by two other specs, declared nowhere). |
| 1.2 | 2026-08-30 | Design-review fixes (design-review-002, Round 2): redesigned the NF-4 scan mechanism around a gitignored local terms file with a fail-loud-if-unconfigured pre-commit precondition (the v1.1 mechanism instructed committing the exact secrets it forbade, and its empty list passed vacuously); extended the same scan to commit-message history; broadened the import-graph gate to also block ambient-module imports and dynamic-import calls (two independent Round 2 panels found the same gap from different angles); named a concrete git-diff-based mechanism for TC-MU9-01; made the canonical serializer's coverage of the harness-owned envelope explicit; added a `.md`/`.html` spec-parity hash check. |
| 1.3 | 2026-08-30 | Design-review fixes (design-review-003, Round 3, dual/blind): NF-4 scan extended to historical blobs; pre-commit hook given a named installation path (`.githooks/pre-commit` + documented `git config` step), with the run-time scan disclosed as a real, hook-independent safety net; AST gate widened to 5 individually-tested checks (bans `importlib`/`__import__`/`exec`/`eval` outright rather than enumerating call shapes, closing alias/`getattr` evasions; catches `numpy.random.*`; wider ambient-module list); named `wmj.models.registry`'s `pkgutil`-based auto-discovery mechanism; replaced TC-MU9-01's tautological git-diff mechanism with a structural test of the actual architectural property; traced the spec-parity check to NF-5 with a test case and build chunk; `JudgeInput.region_labels` given one canonical shape cited consistently across all three domain specs. |

---

*Developed using the Grounded Vibe Methodology*
