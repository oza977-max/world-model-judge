# World Model Judge — Cross-Cutting Specification

Version 1.1 · 25 August 2026 · Derived from Requirements v1.2 and Test Cases v1.0 (design-review fixes applied 25 Aug 2026)

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
3. **No ambient inputs.** Nothing in worlds, models, or judge reads the clock, the filesystem, environment variables, or network. The judge specifically is a pure function (JU-12): its entire call graph takes arguments and returns values. File I/O happens only in the orchestration and reporting layers, at the edges. This includes run-identifying metadata — platform string, prereg commit SHA — which the harness reads and attaches to the `JudgedResult` envelope (Data-Model Overview) *after* calling the judge, never inside it (design-review fix: an earlier draft of the judge spec's Verdict schema carried a `meta` block the judge could not have produced without violating this rule).
4. **Canonical serialization.** One shared serializer produces every machine-readable artefact: JSON with sorted keys, UTF-8, `\n` newlines, no trailing whitespace, floats rendered with Python's `repr` (shortest round-trip representation — bit-exact by construction), arrays as nested lists, no timestamps or hostnames anywhere in the payload. NF-1's ten-run byte comparison (TC-NF1-01) compares these bytes.

**Options considered:** (a) tolerance-based comparison instead of byte identity — rejected: NF-1 explicitly demands byte identity, and tolerances rot; (b) hash-based comparison of floats at reduced precision — rejected for the same reason; (c) the four rules above — accepted.

**Consequences:** Every function that needs randomness takes a `Generator` parameter (visible in every signature — this is a feature, not noise). Cross-*platform* identity (e.g. x86 vs ARM) is explicitly out of scope, matching NF-1's "machines of the same platform" wording; the verdict record states the platform.

---

## ADR-003 — Project structure: four packages plus a harness, judge imports nothing

**Decision:** A `src`-layout Python package `wmj` with five sub-packages: `wmj.worlds`, `wmj.models`, `wmj.judge`, `wmj.reporting`, `wmj.harness`. The judge imports only the standard library and NumPy — never the other `wmj` packages. The harness is the only package that imports everything.

**Status:** Accepted.

**Context:** NF-6 (separable layers, judge imports nothing from the others), JU-12 (pure judge), MU-9 (new model touches nothing but its own file), JU-1 (judge input structurally cannot carry identity). [Requirement: NF-6, JU-12, MU-9, JU-1] [Test: TC-NF6-01, TC-JU1-01, TC-MU9-01]

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
- `tests/gates/test_import_graph.py` walks `wmj.judge`'s AST and fails on any `wmj.*` import (TC-NF6-01). This is a build gate, not a convention.
- The judge's input types (defined *in the judge package*, since it can import nothing) are plain dataclasses of arrays and floats with no name, id, or provenance field — model identity is structurally unrepresentable (TC-JU1-01).
- `wmj.models` exposes one registry; adding a model is one new file registering itself (TC-MU9-01's zero-diff-outside-own-file check).
- `prereg/` is where MU-6/JU-11 artefacts live; the harness refuses to judge unrigged models unless the committed `prereg/` files predate the run (mechanics in the models and judge specs).

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
| `JudgeInput` | arrays only: predictions, spreads, outcomes, region name + labels, divergence curve `[H+1]`, climatology table, task tolerances — defined inside `wmj.judge`, no identity fields, no separate boundary-marker field (a trial's row in the array's trial axis *is* its boundary — design-review fix, see judge spec v1.1 ADR-J4) | judge | judge |
| `Verdict` | the JU-9 record — full schema in the judge spec, serialized canonically per ADR-002. Contains only what the judge itself computes: no model identity, no fixture flag, no run metadata (design-review fix — these were removed from `Verdict` because the judge cannot honestly produce them; JU-1/JU-12) | judge | harness |
| `JudgedResult` | the harness-owned envelope `{model_ref, model_name, is_fixture, verdict, meta}` — one per (model, world); wraps a `Verdict` with the identity/metadata facts only the harness holds (design-review addition, judge spec v1.1 §5) | harness | reporting, `out/verdicts/*.json` |

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

**Design-review addition.** Architecture-overview and the implementation guide both referenced "the forbidden-terms scan" and TC-NF4-01 explicitly promised the list would be "declared and kept current in the technical spec" — but no prior spec draft actually declared it. This is the declaration.

**The list, maintained here (`wmj/harness/confidentiality.py` imports it as a constant, never redefines it):**

```
FORBIDDEN_TERMS = [
    # Employer/professional-context terms — NF-4's absolute rule. This seed list
    # covers the pattern classes named in NF-4 and CLAUDE.md; anyone extending
    # this project adds specific terms here, never relaxes the rule itself.
    # (Left deliberately generic in this public spec, per NF-4 — the actual
    # private seed list, if any specific terms are ever identified, is added
    # to this file directly, not enumerated in a public document.)
]
```

**The scan mechanism:** `wmj/harness/confidentiality.py`'s `scan_repository()` walks every tracked file in the working tree (via `git ls-files`, so it naturally excludes `out/` and other gitignored paths), case-insensitively searches file contents for each term in `FORBIDDEN_TERMS`, and fails loudly (per the Error-Handling Conventions above) listing every match's file and line if any are found. Run in CI on every push and as a pre-commit hook; TC-NF4-01 runs it once against a clean checkout.

**Maintenance:** `FORBIDDEN_TERMS` is append-only, per the same git-hygiene discipline as `prereg/` — a term is never removed once added, only added to, and any addition is its own commit with a message stating why.

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
| ADR-003 structure | NF-6, JU-1, JU-12, MU-9 | TC-NF6-01, TC-JU1-01, TC-MU9-01 |
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

---

*Developed using the Grounded Vibe Methodology*
