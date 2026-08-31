# World Model Judge — Cross-Cutting Specification

Version 1.4 · 31 August 2026 · Derived from Requirements v1.2 and Test Cases v1.1 (design-review fixes applied 25, 30 and 31 Aug 2026)

> **Change note (v1.4 — the design-review-004 structural repair).** This round repairs rules, not instances. The NF-6/JU-12 gate is restructured from an evadable denylist to an **import allowlist** (`{numpy, math, dataclasses, typing}` — judge.md's own "nothing else" claim, finally enforced as stated) plus a banned-identifier check and a wholesale `numpy.random` ban (the judge needs no randomness at all), with the gate's completeness claim relocated from prose to an **executable evasion-fixture corpus** (TC-NF6-04) — every evasion any round documented becomes a fixture the gate must flag, ending the four-round enumerate-evade-patch cycle. TC-MU9-01's mechanism v3 writes its stub to a `tmp_path` (never the production directory), asserts the *production* roster path (`harness.trials`) genuinely delegates to the registry, and replaces the identifier-grep with a mechanical import-allowlist check. Registry mechanics fully pinned (`invalidate_caches`, explicit `__path__` argument, sorted-name contract). NF-4: the phantom "CI" layer is deleted (no CI exists in this architecture), the history scan upgraded to `git cat-file --batch-all-objects` (reaches reflog-only and unreachable objects), and `wmj run` now verifies `core.hooksPath` wiring loudly. ADR-002 rule 3's enforcement claim scoped honestly to the judge. `WorldContext` added to the Data-Model Overview (models spec ADR-M1). See design-review-004.html and `reviews/calibration.md`'s structural recommendation.

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
3. **No ambient inputs.** Nothing in worlds, models, or judge reads the clock, the filesystem, environment variables, or network. The judge specifically is a pure function (JU-12): its entire call graph takes arguments and returns values. File I/O happens only in the orchestration and reporting layers, at the edges. This includes run-identifying metadata — platform string, prereg commit SHA — which the harness reads and attaches to the `JudgedResult` envelope (Data-Model Overview) *after* calling the judge, never inside it (design-review fix: an earlier draft of the judge spec's Verdict schema carried a `meta` block the judge could not have produced without violating this rule). **Enforcement, scoped honestly (design-review-004 repair — the previous sentence credited ADR-003's gate with enforcing this rule for worlds, models, and judge alike, but that gate inspects `wmj.judge` only):** the judge gets *structural* enforcement (ADR-003's allowlist gate plus TC-JU12-01's dynamic purity test); worlds and models get *empirical* enforcement only — the byte-identity tests (TC-WD7-01/02, TC-MU8-01, TC-NF1-01/02) plus code review — which catches an ambient read that varies between runs but not one that happens to be stable on the test machine. That asymmetry is deliberate (the judge is the load-bearing purity boundary; JU-12 names it, not worlds/models) and is stated here so no reader assumes a protection that doesn't exist.
4. **Canonical serialization.** One shared serializer produces every machine-readable artefact — including `wmj/harness/results.py`'s harness-owned envelope write (design-review-002 fix: this is the artefact NF-1's byte-comparison actually diffs, per judge spec v1.3 §5, and Round 2 found no prior spec made the link to this serializer explicit, unlike every other artefact in this rule) — as JSON with sorted keys, UTF-8, `\n` newlines, no trailing whitespace, floats rendered with Python's `repr` (shortest round-trip representation — bit-exact by construction), arrays as nested lists, no timestamps or hostnames anywhere in the payload. NF-1's ten-run byte comparison (TC-NF1-01) compares these bytes.

**Options considered:** (a) tolerance-based comparison instead of byte identity — rejected: NF-1 explicitly demands byte identity, and tolerances rot; (b) hash-based comparison of floats at reduced precision — rejected for the same reason; (c) the four rules above — accepted.

**Consequences:** Every function that needs randomness takes a `Generator` parameter (visible in every signature — this is a feature, not noise). Cross-*platform* identity (e.g. x86 vs ARM) is explicitly out of scope, matching NF-1's "machines of the same platform" wording; the verdict record states the platform.

---

## ADR-003 — Project structure: four packages plus a harness, judge imports nothing

**Decision:** A `src`-layout Python package `wmj` with five sub-packages: `wmj.worlds`, `wmj.models`, `wmj.judge`, `wmj.reporting`, `wmj.harness`. The judge imports only the standard library and NumPy — never the other `wmj` packages. The harness is the only package that imports everything.

**Status:** Accepted.

**Context:** NF-6 (separable layers, judge imports nothing from the others), JU-12 (pure judge), MU-9 (new model touches nothing but its own file), JU-1 (judge input structurally cannot carry identity). [Requirement: NF-6, JU-12, MU-9, JU-1] [Test: TC-NF6-01..04, TC-NF6-06, TC-JU1-01, TC-MU9-01, TC-MU9-02]

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
- `tests/gates/test_import_graph.py` walks `wmj.judge`'s AST. **Design-review-004 restructure — from denylist to allowlist, with an executable fixture corpus as the gate's final authority.** Four review rounds established the pattern: every enumerated denylist of "bad shapes" was evaded within one round by a shape nobody enumerated (Round 2's dynamic imports, Round 3's aliases and `getattr`, Round 4's `from numpy.random import rand`, `__builtins__`, and `compile`). The repair fixes the *rule*, not the latest instance:
  1. **TC-NF6-01 — the import allowlist.** Any `Import`/`ImportFrom` whose top-level module is not in **`{numpy, math, dataclasses, typing}`** fails. This is judge.md §4's own "nothing else" claim, finally enforced as stated — it subsumes, with nothing left to enumerate, every prior import-shaped check (`wmj.*`, the twelve ambient modules, `importlib`, `random`, `builtins`, and every module nobody thought to list). An allowlist cannot have the omission problem a blocklist has.
  2. **TC-NF6-02 — banned execution-primitive identifiers.** Any occurrence of the identifiers `exec`, `eval`, `compile`, `__import__`, `__builtins__`, `globals`, or `vars` anywhere in the AST — as a `Name` id, an `Attribute` attr, or an import alias, not only as a direct `Call` target (the call-shape scoping was Round 4's confirmed evasion route: `e = eval; e(...)` and `__builtins__.eval(...)` both slipped the v1.3 wording). The judge is pure arithmetic; none of these names has a legitimate use in it. Combined with check 1, the realistic indirection routes close: `getattr(builtins, "eval")` needs `import builtins` (fails check 1); `getattr(__builtins__, "eval")` names `__builtins__` (fails check 2).
  3. **TC-NF6-03 — `numpy.random` banned wholesale in the judge.** Any `Import`/`ImportFrom` whose full dotted module path begins `numpy.random`, and any `Attribute` node whose attr is `random`, fails. Design-review-004 simplification: the judge computes deterministic metrics and needs **no randomness at all** (JU-12) — the v1.3 check's Generator/PCG64 exception implied otherwise and forced a name-by-name pattern (`<alias>.random.<name>`) that `from numpy.random import rand` and `import numpy.random as npr` both walked straight past. A total ban has no such pattern to evade. (The bare-`attr == "random"` rule is deliberately over-broad; TC-NF6-06 guards against false positives on the real judge source, which has no `.random` attribute of any kind.)
  4. **TC-NF6-04 — the evasion-fixture corpus, the gate's real completeness contract.** Every concrete evasion any review round has documented (bare/aliased/from-imports of forbidden modules; `importlib` aliasing; `getattr` indirection; `__builtins__` attribute routes; `exec`/`eval`/`compile` payloads; both `numpy.random` import idioms; identifier rebinding) exists as a one-file fixture under `tests/gates/fixtures/`, and the gate must flag **every** fixture. **The gate's completeness claim is exactly this corpus, no more** — a new evasion found later is added as a fixture, turning "is the gate complete?" from a prose argument (which failed four times) into an executable, growing regression suite. Static analysis of Python cannot be proven complete (LANGSEC's own point, disclosed rather than implied away); the fixture corpus is the honest, checkable form of the claim.
  5. **TC-NF6-05 — superseded (design-review-004).** Its former content (the `<numpy-alias>.random.<name>` attribute pattern) is absorbed and strengthened by TC-NF6-03. The ID is retained as a tombstone per this project's append-only discipline so no ID silently vanishes; it is not an executable case.
  6. **TC-NF6-06 — clean-pass / false-positive guard.** The gate run against the real, legitimate judge source passes — necessary because checks 2 and 3 are deliberately over-broad, and a gate that cries wolf gets disabled.

  All checks live in one AST walk, one gate; any violation fails the run. This is a build gate, not a convention. **The static gate is defence-in-depth, not proof** — final runtime authority is TC-JU12-01's dynamic purity test (the judge's full call graph executed under a meta-path import hook that raises on any import attempted mid-call, with ambient builtins poisoned; exact mechanics are settled at build time by that test's own phantom-gate discipline, not legislated further in prose — per this project's "enforced by test, never by convention" rule, applied at last to the gates themselves).
- The judge's input types (defined *in the judge package*, since it can import nothing) are plain dataclasses of arrays and floats with no name, id, or provenance field — model identity is structurally unrepresentable (TC-JU1-01).
- **`wmj.models.registry`'s auto-discovery mechanism, fully pinned (design-review-003 named it; design-review-004 pinned the arguments Round 4 found missing):** `registry.all_models()` calls `importlib.invalidate_caches()` and then `pkgutil.iter_modules(wmj.models.__path__)` — the explicit `path` argument matters: called with no argument, `iter_modules` walks all of `sys.path`, the exact wrong scope, and Round 4 found the argument unstated — importing each discovered submodule via `importlib.import_module` (excluding `base` and `registry` themselves), then returning the registered factories per the contract in models spec ADR-M1: name-keyed, **sorted-name order as a stated part of the contract** (so downstream determinism, including `model_ref` assignment, never silently depends on any discovery mechanism's internal ordering), duplicate names refused at `register()` time with `DuplicateModelError`. Repeat calls are safe by construction: already-imported modules are not re-executed (Python import caching) and registration is name-keyed, so nothing accumulates twice. A new model file is discovered the moment `all_models()` is next called — no edit to any other file, no hardcoded import list anywhere.
- **The zero-edit-outside-own-file property (TC-MU9-01), mechanism v3 (design-review-004 repair — Round 4 found the v1.3 version wrote its stub into the real production `wmj/models/` directory with no exception-safe cleanup, so its own designed-to-fail case could leave a phantom model for a later real `wmj run` to discover and execute; found nothing proving the production path actually uses the registry; and found its "reference to the stub's name" identifier-grep carried the same string-evasion imprecision the NF-6 gate's own history proves fatal):** `tests/gates/test_registry_isolation.py`:
  (a) creates the stub model module in a pytest `tmp_path` directory and appends that directory to `wmj.models.__path__` via monkeypatch — undone automatically on teardown, **pass or fail**, and the real package directory is never written to; calls `importlib.invalidate_caches()`; asserts the stub appears in `all_models()`.
  (b) asserts the stub also appears in the roster built by the **actual production code path** — `wmj.harness.trials`'s roster-construction function invoked directly — proving the harness genuinely delegates to the registry rather than holding a second, separately-maintained list that happens to agree today.
  (c) statically verifies, by the same AST **import-allowlist** technique as the NF-6 gate (mechanical, no string-matching to evade), that no module in `wmj/judge`, `wmj/worlds`, `wmj/reporting`, or `wmj/harness` imports any `wmj.models` submodule other than `wmj.models.registry` and `wmj.models.base`. Referencing the seven pinned model *name strings* (models spec ADR-M1) as data is explicitly permitted and untouched by this check — names are the sanctioned join keys, imports are the forbidden coupling.
  Registry state is snapshot-and-restored in fixture teardown so no registration leaks between tests. Each of (a), (b), (c) can genuinely fail, and TC-MU9-02's phantom-gate case proves (c) fails when a forbidden import is injected.
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
| `JudgedResult` | the harness-owned envelope `{model_ref, model_name, is_fixture, verdict, meta}` — one per (model, world); wraps a `Verdict` with the identity/metadata facts only the harness holds (design-review addition, judge spec §5); `model_ref` indices follow `all_models()`'s sorted-name contract (models spec ADR-M1), so they are reproducible across machines | harness | reporting, `out/verdicts/*.json` |
| `WorldContext` | `{world_name, state_dim, action_dim, training_state_box, training_action_interval}` — frozen dataclass defined in `wmj.models.base` (deliberate duplication, same pattern as the judge's own input types); constructed **only** by the harness from the world's declared constants, passed identically to every model factory (models spec ADR-M1, design-review-004 repair — the one sanctioned channel for world facts to reach models) | harness (constructs), models (defines type) | every model factory |

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
- `scan_repository()` reads `confidentiality_terms.local.txt` if present, plus the small set of structural pattern classes that are safe to keep generic and committed (e.g., a regex for common internal-only URL/hostname shapes). It walks every tracked file in the working tree (via `git ls-files`, so it naturally excludes `out/` and other gitignored paths), **scans commit-message history** (`git log --format=%B --all --reflog`), and **walks every object in the local object database** (design-review-003 added historical blobs via `git rev-list --objects --all`; design-review-004 repair: `--all` reaches only ref-reachable objects, missing reflog-only content — amended-away commits, deleted branches, dropped stash entries — which are exactly as recoverable via `git cat-file -p <sha>`; the scan now enumerates via `git cat-file --batch-all-objects --batch-check`, which lists **every** object in the database, reachable or not, with each blob's content then searched the same way as the working tree). The remaining bound is stated plainly: objects already pruned by git's garbage collection are genuinely gone from the local database and cannot be scanned — but they are equally gone from what a `git push` could ever publish, so the scan's coverage matches the actual exposure surface. Case-insensitive search for each term; fails loudly (per the Error-Handling Conventions above) listing every match's file/commit/blob and location if any are found.
- **The precondition that replaces the vacuous-pass gap:** if `confidentiality_terms.local.txt` does not exist or is empty, the **pre-commit hook** fails loudly with an instructional message ("no local confidentiality terms configured — see `confidentiality_terms.example.txt`") rather than silently reporting a clean scan. An unconfigured scan is now a stop, not a false green.
- **The hook's installation path, named explicitly (design-review-003 fix — Round 3 found three independent panels flagging that no document ever specified how a pre-commit hook reaches `.git/hooks/`, which `git clone` never populates):** the hook script is committed and tracked at `.githooks/pre-commit` (a normal file, unlike `.git/hooks/`, which is versioned like any other repository content). The repository's one-time setup step — documented in the README and named explicitly in this ADR rather than left to be inferred — is `git config core.hooksPath .githooks`, run once per clone. This makes the hook's existence auditable (it's a file in the repository anyone can read) even though its *activation* still depends on that one command having been run.
- **There are exactly two enforcement layers, both local — no others exist (design-review-004 repair: v1.3's disclosure described a "CI re-run" of this scan as an existing backstop, but no CI system is built, configured, or listed anywhere in this design, and architecture-overview §1's own System Context states there are no external systems beyond git itself; the phantom layer is deleted rather than built, since building it would contradict the architecture):** (1) the pre-commit hook, and (2) `python -m wmj run`/`verify`'s startup gates (P6-C02), which re-run the full scan — working tree, commit messages, all objects — on every execution, independent of whether the hook was ever installed. **The startup gates additionally verify `git config core.hooksPath` resolves to `.githooks` and fail with an instructional message if it doesn't** (design-review-004 addition — Round 4 found a pre-existing `core.hooksPath` from unrelated tooling would silently defeat the hook with no visible signal; this converts that silent failure into a loud one at the first `wmj run`).
- **The residual gap, named rather than hidden (matching models spec ADR-M5's disclosure pattern):** a fresh clone that never runs the one-time `git config` step *and* never runs `wmj run` has no active check at all, and `git commit --no-verify` bypasses the hook regardless. No technical mechanism in a single-author, no-server-side-review project fully closes the commit-time gap without infrastructure this project's own architecture deliberately excludes; disclosed, not presented as solved.

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
| ADR-003 structure | NF-6, JU-1, JU-12, MU-9 | TC-NF6-01..04, TC-NF6-06, TC-JU1-01, TC-MU9-01, TC-MU9-02, TC-JU12-01 |
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
| 1.4 | 2026-08-30 | Design-review-004 structural repair: NF-6 gate restructured to import-allowlist + banned identifiers + wholesale `numpy.random` ban, with completeness relocated to an executable evasion-fixture corpus (TC-NF6-04; TC-NF6-05 superseded); TC-MU9-01 mechanism v3 (tmp_path stub, production-path assertion, import-allowlist part c); registry mechanics pinned (`invalidate_caches`, `__path__` arg, sorted-name contract); NF-4 phantom-CI layer deleted, history scan upgraded to `--batch-all-objects`, `core.hooksPath` verified loudly at `wmj run`; ADR-002 rule 3 enforcement claim scoped to the judge; `WorldContext` row added. |

---

*Developed using the Grounded Vibe Methodology*
