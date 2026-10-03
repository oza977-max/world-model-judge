# Spec corrections backlog — input to the next design-review round

**In plain words:** while building, we found places where the written
specs turned out to be wrong, stale, or missing something the code
genuinely needed — and one idea from outside the project worth
considering. None of these were fixed in the spec documents
themselves during the build, on purpose: a build chunk that edits an
already-reviewed spec to match what it just built is moving the
goalposts after the fact, which is the exact habit this project exists
to police (MU-6, JU-11). Instead each was fixed in code where it had
to be, documented at the point of discovery, and collected here as the
single input list for Round 9 of design review (already owed under
BC-2 to independently check the v1.8 fixes).

Every item below is also recorded, with its executed evidence, in the
handover or prompt file named next to it.

---

## A. Spec text falsified by the build (corrections)

| # | Document | What the spec says | What the build found | Where recorded |
|---|---|---|---|---|
| A1 | `specs/worlds.md` §8 | Pins the LV TC-WD1-01 hand-verified reference state as `(4.0, 2.5)`. | That point is the LV world's own equilibrium (dx/dt = dy/dt = 0 there exactly). RK4 of an exactly-zero derivative returns the input unchanged whatever the integrator does, so a test at that state cannot catch an integrator bug — only a wrong-constant bug. The build hand-verified at `(4.5, 2.0)` instead. | `build/prompts/P1-C02.md`, `tests/unit/worlds/test_lv.py` docstring |
| A2 | `specs/worlds.md` §8 | Says the pendulum reference value "must be (re-)computed from the design-review-002-corrected EOM" — but records no recomputed value. | The build computed one via an independent plain-Python RK4 from `(0.1, 0.1, 0, 0)`, `dt=0.002`, `u=0`; a second independent re-derivation during review matched to all 16 printed significant digits. The spec should pin this value. | `build/prompts/P2-C02.md`, `tests/unit/worlds/test_pendulum.py` |
| A3 | `specs/cross-cutting.md` ADR-003, TC-NF6-02 | Pins the judge's banned-identifier list at 13 names. | Six independent review rounds on the gate found real, executed bypasses using names not on the list. The build's list is 23 plus a structural metaclass check: `__dict__`, `__getattribute__`, `locals`, `__traceback__`, `tb_frame`, `tb_next`, `f_globals`, `f_locals`, `f_builtins`, `f_back`, and `class X(metaclass=...)`. Each is a well-known reflection primitive of the same class as the already-banned `vars`/`getattr` — omissions from the pinned list, not new kinds of route. (The static lint remains, by design, lint and not a completeness proof; ADR-002 rule 3's `ctypes`/pre-capture residual stands.) | `build/prompts/P1-C03.md` (five addenda), `tests/gates/test_import_graph.py` module docstring |
| A4 | `specs/cross-cutting.md` ADR-003, TC-NF6-09 | Claims `wmj/models/*`'s "only sanctioned outward imports are now, completely: numpy, math, dataclasses, typing, wmj.models.base, wmj.models.registry — nothing else, in any direction." | Already-reviewed P1-C01 code falsifies it: `wmj/models/base.py` needs `wmj.errors` (the pinned "every wmj exception subclasses WmjError" convention) and `hashlib` (`component_key`'s blake2b digest, ADR-002 rule 2). The build's models allowlist includes both. | `build/prompts/P1-C03.md`, `tests/gates/test_import_graph.py` |
| A5 | `specs/cross-cutting.md` ADR-003, TC-NF6-01 | Judge allowlist `{numpy, math, dataclasses, typing}` with no mention of `__future__` or same-package imports. | Every file carries `from __future__ import annotations` (mandated by the Development Conventions), and `judge/skill.py` legitimately imports `judge/_normal.py`. A literal reading refuses both. The build allows `__future__` and `wmj.judge.*`. The spec should say so. | `tests/gates/test_import_graph.py` |
| A6 | `specs/worlds.md` ADR-W3; `test-cases.md` TC-WD4-01 | LV's divergence curve "grows sub-exponentially (log-separation vs step is concave/linear)" / "grows roughly linearly rather than exponentially". | Executed at full scale (64 starts, δ₀=1e-6, H=700): LV's median separation is **flat** over its declared horizon (1.0e-6 → 1.0e-6, oscillating) — orbits are neutrally stable, so a nearby orbit neither converges nor diverges. The linear "phase drift" the spec names is real but only visible over many cycles (3.7× at 7,000 steps ≈ 20 cycles). The honest, executable assertion is *bounded / sub-exponential over the declared horizon*, which the build asserts; "grows roughly linearly" is not observable at H=700 and should be reworded. | `build/prompts/P2-C03.md`, `tests/unit/worlds/test_divergence.py` |
| A7 | `specs/worlds.md` ADR-W1; `test-cases.md` TC-WD3-03; §5 artefact field `conserved_rel_drift_max` | Drift bound "1e-6 (relative)" with the normaliser unstated (implicitly relative to the initial value). | LV's invariant V(x,y) crosses zero inside the training box (min |V₀| over 64 starts = 8.2e-4), so drift/|V₀| reports **1.6e-6 — over the bound — for an absolute drift of 2.2e-9**. A normaliser that passes through zero is not a measurement. The bound's stated purpose (protect the climatology's *binning* of the invariant) implies the invariant's dynamic range over the region as the unit: 6.7e-9 for LV. The build defines `conserved_rel_drift_max = max|ΔV| / span(V₀ over all benchmark starts)` and reports the literal figure alongside as `conserved_rel_drift_max_vs_initial`. The spec should pin the normaliser. | `build/prompts/P2-C03.md`, `src/wmj/harness/benchmarks.py` docstring |
| A8 | `specs/models.md` ADR-M2; `test-cases.md` TC-MU2-01 | Persistence spread is "the per-dimension standard deviation of one-step state changes over the training dataset" — `ddof` (population vs sample) is left unpinned. | `ddof` **changes the emitted bytes** of every persistence forecast, hence every downstream CRPS, skill score and Verdict, so it cannot stay a renderer's-discretion detail under NF-1 byte-reproducibility. The build makes the fit an explicit public `fit_persistence_spread` with **`ddof=1`** (sample std — the training set is a sample of the world, and ADR-M3's ensemble spread also carries Bessel's correction), and refuses a zero/non-finite spread loudly (`DegenerateSpreadError`) rather than emitting a zero-width forecast the CRPS would reject. The spec should pin `ddof=1` (and the same for linear's residual spread, P3-C02). | `build/prompts/P2-C05.md`, `src/wmj/models/baselines.py` |
| A9 | `specs/cross-cutting.md` ADR-003; `specs/reporting.md` §4; `test-cases.md` TC-NF6-xx family | The import-graph gate (`tests/gates/test_import_graph.py`) mechanically pins the **judge**'s allowlist (TC-NF6-01..09) and the **models** outward-import rule, but there is no automated gate for **reporting**'s stated layering (matplotlib / numpy / stdlib / judge-types / own-package only, ADR-003). | P2-C05's independent review verified reporting's imports by grep this pass, but a future chunk could add `from wmj.worlds import lv` to `reporting/` with no test catching it — the same silent-drift risk the judge gate exists to close. Cheap to add a `REPORTING_ALLOWLIST` check mirroring the models one. Design review should decide whether reporting's layering warrants a mechanical gate. | independent review of P2-C05; `tests/gates/test_import_graph.py` |

## B. New-requirement candidate (from outside the project)

| # | Source | The idea | Why it fits | Why it is NOT added now |
|---|---|---|---|---|
| B1 | *What-If World: A Causal Benchmark for General World Models in Embodied Scenarios* (arXiv 2605.27589), surfaced while looking at World Labs' Atlas release (Sept 2026) | An **action-blind** check for models under test: same starting state, two different actions, the model's two predictions must differ. That paper's headline finding: across nine models, in 13.1% of cases every single-trajectory check passed yet the paired check failed — the model produced two convincing futures that were causally identical, i.e. it silently ignored the action. | This project already checks that the *world's* action lever is real (TC-WD2-01: a non-null action must change the outcome). It does not check that a *model* responds to the action. A model that ignores its action input scores fine on null-action trials and fails with no signal — the "disguised `(state) → next_state` forecaster" trap TC-WD2-01 guards against, one layer up. Fixture-shaped: it would sit beside the three MU-3 fixtures as a fourth engineered failure mode (labelled as a fixture, never a finding — MU-4, RP-8). Scope-neutral: one more check of the kind the judge already makes. | Adding a new failure-mode check mid-build, after reading a paper, is the goalpost-moving pattern MU-6/JU-11 police. It should enter the way everything else did: requirements → test case → design review → build. |

Context, for the record: World Labs' own taxonomy splits "world
models" into renderers (visual fidelity), simulators (physically
faithful state), and planners. Atlas is evaluated as a renderer /
reconstructor (camera-conditioned generation quality, 3D
reconstruction accuracy). Its "control" is camera pose, not an
intervention on world state. It is therefore not the kind of model
this judge grades — and its release reports no calibration, no trust
horizon, and no independent verdict, which restates the essay's thesis
rather than challenging it. Public sources only:
worldlabs.ai/blog/atlas, worldlabs.ai/blog/taxonomy-of-world-models,
arXiv 2605.27589, arXiv 2601.21282 (WorldBench: SOTA video world
models degrade sharply after 5–9 autoregressive frames — a trust
horizon in frames, reported as accuracy only).

---

## How to consume this list

Round 9 (`/gvm-design-review`) takes this file as its stated focus.
Section A items are corrections: the spec should be brought to match
the executed evidence, or the evidence challenged. Section B is a
decision: accept into requirements (then `/gvm-test-cases` and a build
chunk), or decline with a reason. Either way the outcome is recorded
in `reviews/calibration.md` like every other round.

---

## Disposition (Round 9, 2026-09-04)

Every Section A item resolved this round, per the user's explicit direction
after Round 9's verdict ("Build with caveats") and triage: fix everything
byte-affecting now, before pre-registration makes it a one-way door.

| # | Disposition |
|---|---|
| A1 | Spec corrected (`specs/worlds.md` §4.1/§8): LV reference moved to (4.5, 2.0), independently re-derived a second time this round (two independent plain-Python RK4 implementations, matching to every printed digit) per the blind panel's condition. |
| A2 | Spec corrected (`specs/worlds.md` §8): pendulum reference value pinned, matching the already-built test constant to all 16 digits. |
| A3 | Spec corrected (`specs/cross-cutting.md`): identifier list updated to the 23 actually enforced + metaclass ban; one sentence added naming the round-over-round growth pattern honestly (blind panel's condition). Traced to a test case (TC-NF6-11). |
| A4 | Spec corrected (`specs/cross-cutting.md`, `test-cases/test-cases.md` TC-NF6-09): models allowlist sentence now includes `__future__`, `wmj.errors`, `hashlib` — Contracts panel's finding that the originally-proposed fix text was itself still incomplete was applied. |
| A5 | Spec corrected alongside A4 (`__future__` + same-package `wmj.judge.*` already covered by A4's edit's neighbouring text). |
| A6 | Spec corrected (`specs/worlds.md` ADR-W3, `test-cases/test-cases.md` TC-WD4-01): "grows roughly linearly" replaced with "bounded / sub-exponential over the declared horizon," asserted two-sided in both the spec prose and the code (`tests/unit/worlds/test_divergence.py`). |
| A7 | **Code fixed, not just documented.** The interim per-run pooled normaliser (itself already a correction, but pooling every declared region's starts into one span) was replaced with a per-region deterministic grid (`wmj.worlds.divergence.conserved_quantity_range`) — no RNG, no `n_starts`, never pooled across regions. Executed at full scale for both worlds; the `1e-6` bound re-validated against the new figures (worst case 2.7× margin, not "orders of magnitude" — the spec now says so honestly). `specs/worlds.md` ADR-W1 and §5's worked example both corrected. |
| A8 | **Code fixed.** `_fit_linear_spread` (private, `ddof=0`, unguarded) replaced by public `fit_linear_spread`, sharing one implementation with `fit_persistence_spread` (`ddof=1`, `DegenerateSpreadError` guard) — the exact asymmetry four panels (C1) flagged. `specs/models.md` ADR-M2 now pins `ddof=1` for both baselines explicitly. |
| A9 | **Code added.** `REPORTING_ALLOWLIST` gate (`tests/gates/test_import_graph.py`, `TC-NF6-10`) mirroring the models gate, per three of four panels' recommendation. `specs/cross-cutting.md` records the rationale (reporting is the sole `out/` writer; byte-identity cannot catch a deterministic leak there). |

Two orphan fail-loud mechanisms Requirements Coverage flagged (the metaclass
structural check, `DegenerateSpreadError`) now have phantom-gate test cases
(`TC-NF6-11`, `TC-MU2-03`).

**Section B (B1, the action-blind model check):** not decided in this pass —
routed separately, per the panels' own recommendation, through
`requirements/requirements.md` (which the user must explicitly approve,
per this project's standing gate) before any test case or build chunk.

All fixes re-verified: full fast suite green (`pytest -m "not slow"`), the
full-scale slow gate green for both worlds, `tests/gates/` green including
the two new gate additions.

---

## Taxonomy notes — senses of "world model" this judge does not grade

Recorded for the essay's scope statement, not as scope changes. Each is a
public artefact that uses the phrase "world model" for something the judge
deliberately does not grade; together they are evidence for the thesis
that the term has no shared definition, let alone shared measurement.

| Source | What it calls a "world model" | Why the judge does not grade it |
|---|---|---|
| World Labs, *Atlas* (Sept 2026, worldlabs.ai/blog/atlas) | A camera-conditioned renderer / 3-D reconstructor. | Its "control" is camera pose, not an intervention on world state; it reports no calibration, no trust horizon, no independent verdict. See B1's context above. |
| DeepLethe, *Utopia* (github.com/deeplethe/utopia, tagline "World's first open-source enterprise world model") | A Rust enterprise knowledge-management system: a bitemporal knowledge graph with document ingestion, hybrid text/vector search, an editable ontology, and a RAG chat interface. | It simulates nothing, predicts no next state, and responds to no action — there is no dynamics, trajectory, or horizon to grade. Its "governs itself" claim is conflict detection over graph facts, with no calibration, trust horizon, pre-registration, or independent verdict: the "governance in name only" the essay argues against, on a different object. Its bitemporal "what was true / when we believed it" idea is a weaker cousin of this project's own pre-registration proof (a git blob hash at first commit, `check_prereg`, TC-MU6-04 — a content hash, not a database timestamp). Nothing to borrow. |

---

## Disposition (code-review-001, 2026-09-05)

The first code review's findings, closed on the user's "close them"
authorisation in one pass. Every fix was verified by the fast suite (223
tests, 18 new), the gates (including the ten-run separate-process
byte-identity gate), and the full-scale slow gate for both worlds.

| Finding | Disposition |
|---|---|
| I1 — this session's own reconciliation gaps | **Fixed.** `__future__` added to cross-cutting item 10's allowlist sentence; TC-NF6-10 wired into the four reporting-gate test names, TC-NF6-11 into the metaclass isolation test, TC-MU2-03 into all five degenerate-spread tests; the full-scale TC-WD4-01 gate now asserts the two-sided band its fast counterpart has. |
| I2 — stale `ddof` docstring in `baselines.py` | **Fixed.** Docstring now states `ddof=1` as ADR-M2's pinned decision and cites it; the "backlog candidate" framing is gone. |
| I3 — `assert_single_threaded()` dead code | **Fixed.** `cli.main()` asserts the guard on every entry; `tests/conftest.py` now sets it for the test process itself before NumPy loads (the test process was previously running at ambient thread count with nothing saying so); a phantom-gate test proves `main()` refuses when a variable is wrong. |
| I4 — tautological `within_bound` | **Fixed.** The field is now derived from the same comparison the gate makes, with the gate raising immediately after; a comment states plainly that in any *returned* artefact it is necessarily True and why the field is kept. |
| I5 — phantom-gate discipline gap | **Fixed.** Seven new `_validate` cases in `test_horizon_plot.py` (dt negative/NaN/wrong-type, per_region not a list, entry not a dict, block not a dict, single step), `SeedKeyError` on `my_name=None`, `CaptionLengthError` on a fourth sentence, and the CLI's `wmj run` fallthrough (exit 2) — every fail-loud guard the panel named now has a test proving it fires. |
| I6 — unguarded `ZeroDivisionError` from `--n-trials 0` | **Fixed.** `PreviewArgumentError` (a `WmjError`) names the argument and why; `n_starts`, `n_trials`, `horizon` are all validated `>= 1` at the producer, with tests. |
| I7 — frozen dataclasses handing back shared mutable arrays | **Fixed.** `Prediction`, `WorldContext`, `RegionSpec`, `OutRegion` mark their arrays read-only in `__post_init__`; `_fit_spread` returns a read-only vector. A stray in-place write now raises immediately. Tests prove it for all four types. |
| I8 — silent `span<=0` unit-change fallback | **Fixed.** `DegenerateInvariantRangeError` is raised for a region whose invariant has no finite positive range; a stub constant-invariant world proves it fires. |
| Panel F — five stub-detection findings | **Dismissed** as heuristic false positives: `PersistenceModel.reset()` (documented no-op), `within_tolerance`, `declared_region_names`, `lv.regions`, `pendulum.regions` are fully-implemented single-statement functions, not placeholders. No code change. |
| Minor — `skill_score` precondition | **Adopted.** `NonPositiveBaselineError` (judge-local `ValueError`, per the judge's no-`wmj.errors` rule) guards the division; tested. |
| Minor — `conserved_rel_drift_max_vs_initial` nullability | **Adopted.** worlds.md §5 now states the field is `float \| null` and why. |
| Minor — `DEFAULT_N_STARTS` duplicated | **Adopted.** `preview.py` imports it from `benchmarks.py`. The dev-seed literal in tests is left as-is on purpose — each test pins its own seed for readability. |
| Minor — `conserved_quantity_range` untested directly | **Adopted.** A fast direct test covers the 2-D and 4-D boxes, determinism, the convex minimum at the equilibrium, and the per-axis floor. |
| Minor — `null_action` hardcoded shape | **Adopted.** `build_divergence_artefact` passes `np.zeros(world.a)` explicitly. |
| Minor — stale `regions.py` docstring claim | **Adopted.** Names the test that covers the path. |
| Minor — comment/assertion band mismatch | **Adopted.** The comment now says the asserted band is deliberately wider than the observed one, and why. |
| Minor — "hand" → "handed" typo | **Adopted.** |
| Minor — global `rcParams` mutation | **Adopted (test-side).** An autouse `rc_context` fixture restores rcParams around every reporting test; `style.py` documents why the single global style is safe and what to do if a second style is ever needed (scope creation *and* save, because `svg.hashsalt` must be in force at save time). Production code unchanged on purpose. |
| Minor — non-atomic artefact writes | **Adopted.** Every writer (`skeleton`, `captions`, `style.save_figure`) writes to a sibling `.tmp` and `Path.replace()`s it. **The reporting gate (TC-NF6-10) caught the first draft of this fix** — it used `import os` for `os.replace`, precisely the ambient-capability import the gate exists to keep out of the package that writes public output; `Path.replace` is atomic and already sanctioned. Concurrent invocation against one `out/` is documented as unsupported in the CLI docstring. |

Two things the review changed beyond its own findings: the pytest process
is now single-threaded by construction (ADR-002 rule 1 applied to the
test runner, not only to `python -m wmj`), and a gate added one commit
earlier caught a defect in the fix pass that adopted it.

---

## Discovered during build — Phase 3 (input to the next design-review round)

**In plain words:** these were found while writing code, after Round 9
closed. They are *not* fixed in the spec documents (same reason as
Section A: a build chunk editing an already-reviewed spec to match its
own code is the goalpost-moving this project polices). Each is fixed in
code where it had to be, documented at the point of discovery, and
collected here for the next design-review round to ratify or challenge.

| # | Document | What the spec says | What the build found | Where recorded |
|---|---|---|---|---|
| A10 | `specs/models.md` §8 ("Model training") | "**Gradient check:** one full-batch pass, checked against finite differences to **1e-6 relative**." The metric (how relative error is computed) and how near-zero gradients are handled are unstated. | Two distinct problems, both surfaced only by executing the check against the **real** ADR-M3 net (2 hidden × 64), which no toy net exposes. **(1) The naive relative-error metric misreports correct backprop.** `\|a − num\| / (\|a\| + \|num\|)` (the CS231n formula this check was first written with) blows up for a parameter whose true gradient is near zero: on `[5,64,64,4]` seed 15 one weight's gradient is `2.6e-8`, analytic and numeric agree to four significant figures (absolute difference `~1e-11`), yet the formula reports `1.76e-4` relative — a defect in the *measurement*, not the backprop (proven: the epsilon U-curve, and every other parameter agreeing). **(2) `1e-6` is too tight even with the metric fixed.** A *correct* hand-rolled central-difference check on a 3-weight-layer net floors at a few `1e-7` on float64 rounding + truncation alone — a 240-net sweep across both worlds' real I/O shapes (`[3,64,64,4]`, `[3,64,64,2]`, `[5,64,64,8]`, `[5,64,64,4]`, batch 32, normalised O(1) inputs, 60 seeds each) reads worst `3.6e-7` under the fixed metric. `1e-6` leaves only ~3× margin — inadequate for a gate that must never flake, especially on trained nets downstream (P3-C03/C04) whose gradients drive closer to zero. | `build/prompts/P3-C01.md`, `build/handovers/P3-C01.md`, `src/wmj/models/mlp.py` (`gradient_check` docstring, `GRADIENT_SCALE_FLOOR`), `tests/unit/models/test_mlp.py::test_gradient_check_holds_on_the_real_adr_m3_architecture` |

**What the build did (both changes are in code, flagged here for ratification):**

1. **Metric floor.** The relative-error denominator is floored at
   `GRADIENT_SCALE_FLOOR = 1e-3` times the network's largest gradient
   magnitude: a parameter below 0.1% of the peak gradient is judged by
   absolute agreement at the peak scale, not by its own near-zero
   magnitude. The floor is **scale-invariant** (multiply the loss by
   1000 and every reading is unchanged to float64 noise — executed), so
   it carries to P3-C03/C04's NLL and mean losses unchanged. Under the
   floored metric the 240-net worst falls from `1.76e-4` to `3.6e-7`.
2. **Tolerance `1e-6 → 1e-5`.** `1e-5` gives ~30× margin over the
   correct-backprop noise floor while still sitting four orders of
   magnitude below any real bug — the phantom-gate ×1.5 first-layer
   corruption reads `0.2` under the same metric. The check therefore
   catches every backprop error it exists to catch and no longer trips
   on float64 finite-difference noise.

**The decision this asks of design review** (stated openly, because the
floor value is a free parameter and its sensitivity must not be hidden):
`GRADIENT_SCALE_FLOOR = 1e-3` passes `1e-6` on some nets but only `1e-5`
robustly across all 240; a laxer floor (`1e-4`) reads worst `1.6e-6` and
would itself need the `1e-5` tolerance. The build chose the pair
(`1e-3` floor, `1e-5` tolerance) for a never-flake gate with ~30×
margin. Round 10 should ratify **both** the metric floor and the
tolerance, or prefer an alternative (e.g. a numpy-`allclose`-style
combined absolute+relative criterion). Until then the spec text stands
at `1e-6` and the code carries the correction, exactly as A7/A8 did
between their discovery and Round 9.

The floor's **disclosed tradeoff** (flagged by the P3-C01 independent
review, and stated in the `gradient_check` docstring): holding a
sub-0.1%-of-peak parameter to absolute agreement gives up sensitivity to
a backprop bug *localised to such a near-dead unit* — a 5× error on a
`1e-9` gradient reads `~4e-6` and passes `1e-5`. This is judged
acceptable (such a parameter is negligible to the optimisation step, and
a structural formula bug corrupts the peak parameters too and is caught
there), but it is a genuine blind spot the `allclose`-style alternative
would not have, and Round 10 owns that choice. Separately, the same
review hardened the check against a **silent NaN**: a non-finite analytic
gradient, or a non-finite finite-difference loss, now raises
`GradientCheckError` explicitly rather than being dropped by a bare
`max()` (Python's `max()` discards a NaN that is not its first argument) —
regression-tested (`test_gradient_check_raises_on_a_nan_analytic_gradient`,
`..._when_the_finite_difference_loss_is_non_finite`).

---

### A11 — the registry's auto-discovery contradicts the models import-allowlist (both in cross-cutting ADR-003)

**What the spec says.** ADR-003 pins the registry's auto-discovery
mechanism (cross-cutting.md, discovery bullet): "`registry.all_models()`
calls `importlib.invalidate_caches()` and then
`pkgutil.iter_modules(wmj.models.__path__)` … importing each discovered
submodule via `importlib.import_module`." The **same** ADR-003 pins the
models import-allowlist (item 9, TC-NF6-09): "`wmj/models/*`'s only
sanctioned outward imports are, completely: numpy, math, dataclasses,
typing, `__future__`, `wmj.errors`, `hashlib`, `wmj.models.base`,
`wmj.models.registry` — nothing else, in any direction," enforced by the
gate's `MODELS_ALLOWLIST`.

**What the build found.** The two clauses contradict. `registry.py` lives
under `wmj/models/`, so the import gate scans it; yet the discovery clause
requires it to import `importlib` and `pkgutil`, neither of which is in
the allowlist. Building discovery as specified would trip the TC-NF6
models gate (executed: the gate flags any import outside the allowlist).
Nine design-review rounds did not catch it because the registry had not
been built — P3-C02 is the first chunk that needs it. This is the same
class as A4 (the allowlist text falsified by a legitimately-required
import — there, `wmj.errors`/`hashlib`; here, `importlib`/`pkgutil`).

**What the build did.** Split the registry across chunks along the seam
the wiring matrix already draws: the discovery half is consumed by
`harness.trials` and tested by TC-MU9-01, both at **P6-C01**. P3-C02
builds only the registration substrate (`register`, `all_models` in
sorted-name order, `DuplicateModelError`), which imports only `typing` +
`wmj.errors` and stays inside the allowlist — gate untouched. Nothing
enumerates the roster before P6-C01, so no behaviour reachable today
depends on discovery. `all_models()` returns the models registered so far
(a model registers when its module is imported); its docstring says so
plainly and does not claim to discover.

**The decision this asks of design review / P6-C01.** When discovery is
built at P6-C01, `registry.py` must import `importlib`/`pkgutil`, which
needs a gate change. The recommended form is a **narrow, registry-scoped
carve-out** — only `wmj/models/registry.py` may import `importlib` and
`pkgutil`; every other model file (the contestants and fixtures, which
are the files that could reach sideways to game the judge) stays on the
strict allowlist — with a phantom-gate test proving a non-registry model
file importing `importlib` still fails. The alternative (widening
`MODELS_ALLOWLIST` package-wide) is weaker and not recommended. Either
way it touches a load-bearing security gate, so P6-C01 should make the
change explicitly and a design-review round should ratify it; the spec
text (ADR-003's "nothing else" list) should then be corrected to name the
registry's two discovery imports as the sole exception. Recorded now so
the decision is not made silently inside a build chunk.

*Where recorded:* `build/prompts/P3-C02.md`, `build/handovers/P3-C02.md`,
`src/wmj/models/registry.py` (module docstring).

---

### A12 — two pre-registration values the spec left open, pinned at P3-C07

**What the spec says.** (1) models §4 pins `training_trajectories: 2000`
but records the epoch count only as `epochs: <pinned>` — no value. (2)
judge ADR-J4 pins the sharpness-hedge threshold as "a fixed value … set
per world from the world's own scale vector," but not the *formula*.

**What the build did (both are genuine pre-registration modelling choices,
fixed in advance — which is what pre-registration is, not a defect).**
1. **`epochs: 100`** — a round figure, no early stopping. Pinned in
   `prereg/recipe.md`.
2. **`sharpness_hedge_threshold = scale`** (per dimension) — a 90% interval
   as wide as the state's own characteristic magnitude has said nothing, so
   anything wider is hedging. Computed per world by `derive_thresholds.py`,
   committed in `prereg/thresholds.json`.

**The real tension this asks design review to weigh (the one worth naming).**
The implementation guide orders P3-C07 **before** P3-C03/C04/C06 because
those chunks read the epoch count and trajectory count from the recipe. But
the *right* epoch count (and the training-trajectory horizon, itself
unpinned) depends on what those later chunks measure — convergence, and
runtime against the NF-2 600 s budget. So these counts are pinned before
their feasibility is observable: a genuine ordering chicken-and-egg. Two
honest resolutions: (a) a quick feasibility spike on real training before
the recipe is frozen; or (b) accept an open recipe revision before any
judged run if `100` proves wrong — which is the spec's own sanctioned
remedy ("revise the recipe openly and re-run, never nudge a trained
model"), **but** note that revising `recipe.md` after its first commit will
make `check_prereg`'s content-hash (TC-MU6-04) fail at P6-C03 against the
original first-add blob, so an open revision is not free: the
pre-registration's "commit-of-record" has to be re-established and the
revision's own honesty rests on it predating the judged run (which, since
judging is P6-C03, it still does). P3-C03/C06 should report measured
convergence + runtime at `epochs: 100` and, if a change is needed, make it
openly there — before P6-C03 — not silently. Design review should also
ratify the `sharpness_hedge_threshold = scale` formula.

*Where recorded:* `build/prompts/P3-C07.md`, `build/handovers/P3-C07.md`,
`prereg/recipe.md` (the `epochs` note), `src/wmj/harness/derive_thresholds.py`
(`sharpness_hedge_threshold` docstring).

---

### A13 — a new disclosed residual for `check_prereg`: entry substance is not verifiable

**What the build found (P3-C07 independent review).** `check_prereg`'s
per-model entry check (TC-MU6-03) confirms the model's *name* appears as a
token in the committed, pre-dated recipe and prediction. It cannot verify
that the surrounding text is a *genuine* recipe/prediction for that model:
a hollow incidental mention ("possible future model: newmodel") in both
files passes. Verifying substance would require semantic understanding of
prose, which is not mechanizable.

**Disposition — disclose, do not "fix" (the project's own residual-risk
discipline).** This is the same family as models ADR-M5's three named
residual risks (git-history rewritability, non-publication, and — the
closest sibling — timestamp forgeability), all of which are "named, not
solved" because no technical control in a single-author, offline,
no-server-side-witness project closes them. A fake "substance" check would
give false assurance, which is worse than an honest disclosure. Recorded
in `src/wmj/harness/prereg.py`'s module docstring as disclosed residual #2.
**Design review should formally add it to ADR-M5's disclosed-residuals
list** so the spec and the code agree on what pre-registration does and
does not guarantee.

**Also confirmed (not a new finding): ADR-M5 residual #3 is now real in
code.** The same review noted the ordering check reads a locally-forgeable
committer timestamp — exactly ADR-M5's already-disclosed residual #3. The
code correctly implements the specified timestamp check; the forgeability
is the accepted, spec-disclosed limitation (the spec even *removed* the
GitHub-API push-time mitigation for introducing a network dependency). No
action beyond noting it in prereg.py's disclosed residual #1.

*Where recorded:* `build/handovers/P3-C07.md`, `src/wmj/harness/prereg.py`
(module docstring, "Disclosed residuals").

---

## Horizon scan (2026-09-19) — external research, for positioning not build

**In plain words:** a scan of public research on world-model evaluation, to
see if anything should change the build. **Conclusion: nothing changes now.**
We are mid-Phase-3, and adding a check or metric after reading a paper is the
exact goalpost-moving MU-6/JU-11 police — the same reason B1 (action-blind)
went through the front door (requirements → test-cases → design review →
build) rather than a mid-build patch. Everything below is either evidence for
the essay's thesis (taxonomy/positioning) or a **future-version candidate**
for a later design review. Recorded in the same spirit as the Atlas/Utopia
taxonomy notes.

**Verification honesty:** this environment's egress proxy blocks `arxiv.org`
and `huggingface.co`, so the arXiv items below are **surfaced by web search
only, not verified against the primary source** — confirm each (title,
authors, claims, that the ID resolves) before citing it in any published
artefact. The two GitHub/vendor items were fetched and are verified.

### Verified (primary source fetched)

| Source | What it is | Bearing on this project |
|---|---|---|
| **`github.com/megazron/trusthorizon-worldmodel`** (MIT, NumPy-only) — verified via README | A near-twin: a harness that decides "whether a learned world model's rollout can be trusted, and for how many steps" — trust horizon vs a task tolerance, linear-vs-exponential drift classification, calibration of uncertainty bands, divergence onset, physics-plausibility, toy 1-D worlds, three example models (well-calibrated / biased-drift / unstable). | **The strongest thesis exhibit found — "measurement without institution."** Its own README confirms it does **not** report sharpness alongside calibration, does **not** pre-register thresholds, and has **no independent-verdict mechanism**. It is the measurement fragment built without the governance discipline — precisely the essay's claim. Sharpens differentiation; belongs in the essay's scope statement. Nothing to borrow into the build. |
| **Jev / TypeSafe AI** (typesafe.ai; covered by Arize, LangChain, DataCamp, litellm docs) — verified via multiple vendor/independent pages | A "System One" model returning **typed structured decisions + probabilities** instead of generated text ("can't hallucinate"; ~40–400× cheaper than an LLM on classification); invoked with a state + questions. Not a world model. | Tangential to the build. Quiet validation of the design instinct: serious evaluation is moving toward structured, calibrated, **typed** outputs — which is what this project's pure-function `Verdict` already is (vs. a chatty LLM judge). A landscape note, not an input. |

### Search-surfaced, NOT primary-verified (arXiv blocked this session)

| Source (arXiv ID, unverified) | What search reports it says | Bearing |
|---|---|---|
| *A Definition and Roadmap for World Models* (2607.06401) | A 2026 paper attempting to **define** "world model" and chart evaluation. | Thesis evidence: the term is contested enough that defining it is itself a 2026 research contribution. Positioning only. |
| *How Should World Models Be Evaluated? A Decision-Making-Centric Position* (2606.15032) | Argues world-model eval is fragmented; proposes dimensions incl. counterfactual branches, policy-ranking agreement, optimization lift, exploitability, **uncertainty calibration**. | Thesis evidence (fragmentation) **and** a possible source of future eval dimensions (esp. policy-ranking agreement) — a **future-version** design-review candidate, not now. |
| *Conformal Orbit-Valid Trust Horizons for Equivariant World Models* (2606.24946); *Certified World Models* (2606.13092) | The field now uses the phrase **"trust horizon"** and applies **conformal prediction** to certify it with distribution-free finite-sample coverage (reported median certified-to-measured ratio ~0.67). | Both validation (shared framing/vocabulary — we are not alone) **and** a genuine methodological alternative to JU-11's fixed binomial bands: conformal calibration gives coverage guarantees our binomial-band approach does not. **Strongest future-version candidate.** Adopting it now would be goalpost-moving + scope creep; note for a future design review. |
| *RoboTrustBench* (2606.01600); *WorldPrediction* (2506.04363); DreamX-World, 4DWorldBench, WBench, WorldMark, WorldScore, VBench-2.0 | Video / 3-D-4-D / interactive world-model benchmarks (visual quality, controllability, physical plausibility). | Mostly the renderer/generator sense of "world model" this judge deliberately does not grade (cf. the Atlas note) — taxonomy evidence, not build input. |

**Net disposition:** no change to Phase 3 or any spec. One future-version
design-review candidate worth remembering — **conformal trust horizons** as a
possible complement/alternative to JU-11's binomial bands. Verify the arXiv
IDs before any of this reaches a published artefact.

### Addendum (2026-09-27) — abstracts read, twin's code read

**Papers — abstract level.** arXiv and every mirror (alphaXiv, Semantic
Scholar, OpenAlex, HuggingFace, Bytez) are still refused by this
environment's network policy. Abstracts below were read through the search
engine's arXiv index, so titles, authors and abstract claims are now
confirmed; **full text is not**. Nothing here changes the build.

| Paper | What its abstract says | Where it bears |
|---|---|---|
| MiraBench (2605.29360) | Action-conditioned reliability as a benchmark target: physics adherence, **action-following fidelity**, **optimism bias**. Findings: visual fidelity is a poor proxy for action fidelity; scale does not reliably fix action following; optimism bias is pervasive. | Second independent source (after What-If World) for **P3-C08 / TC-MU3-04**, the action-blind check. |
| *Overcoming Statistical Bias in Action-Controllable World Models* (2608.04653, Shi et al.) | Models can fit data "without making their dynamics meaningfully depend on the action… different actions may produce similar futures, while motion may persist even under zero action"; proposes Action Response Consistency. | Names P3-C08's exact failure mode. Its "motion persists under zero action" is a possible **second** action check — a future front-door candidate, not now. |
| *How Should World Models Be Evaluated for Embodied Decision-Making?* (2606.15032, Yu et al.) | "Metric diversity and a recurring problem of **claim/evidence mismatch**" — papers claim more than their evaluation can show; proposes a decision-centric protocol including uncertainty calibration. | The essay's thesis, stated independently. Thesis evidence. |
| *A Definition and Roadmap for World Models* (2607.06401) | "There is no consensus on what a world model fundamentally is, what it should predict, or how it should be built." | Quotable thesis evidence for the essay's scope statement. |
| *Conformal Orbit-Valid Trust Horizons* (2606.24946) and *Certified World Models* (2606.13092), both H. Wang | A computable trust-horizon certificate, calibrated by split-conformal, audited with an **exact-binomial** 95% bound (the same test JU-11 uses). Theorem B: under the finite-time Lyapunov spectrum, expanding directions give a **logarithmic** horizon and neutral directions accumulate error **linearly**. | Future-version candidate for JU-11 (conformal calibration). Theorem B's log-vs-linear picture is consistent with ADR-W3 and backlog A6 (pendulum chaotic, LV orbits neutral with linear phase drift) — supporting context for design review ratifying A6, not proof. Equivariance-specific, single author. |
| Seitzer et al., *On the Pitfalls of Heteroscedastic Uncertainty Estimation* (2203.09168, ICLR 2022) | Training a variance head with Gaussian NLL down-weights the squared error where predicted variance is high, so fit quality can suffer. | **Known risk for P3-C03 (Model A).** The recipe's Model A is exactly this setup. If it under-fits, MU-5's accuracy match can fail for reasons unrelated to calibration. The recipe is committed and must not be tuned; P3-C03 should measure and report fit honestly, and any change goes through an open recipe revision (A12). |

Snippet-level only (not a full abstract; lower confidence): *Do Robotic
World Models Really Follow Actions?* (2608.24885), *AD-WM* (2609.30264 —
factual accuracy vs counterfactual action comparison mismatch), JEPA
action-consistency (2608.12939), ActSWM (2607.26712). Renderer/video sense,
taxonomy only: RoboTrustBench (2606.01600), WorldExam, H2R-Bench,
DreamX-World, WorldScore, VBench-2.0, 4DWorldBench, WBench, WorldMark.

**The twin — code read in full** (`megazron/trusthorizon-worldmodel`,
HEAD `ed92eed`, 6 commits, all 16–17 Sep 2026, ~1,260 lines, MIT). Its
second commit reads "remove project-specific origin references" — it was
extracted from another project. It shares **no** identifiers, files or
distinctive terms with this repo (grep for wmj, CRPS, sharpness,
pre-registration, persistence, pendulum, SR 11-7: none). The "trust
horizon" wording also appears in independent June 2026 papers. No evidence
of derivation either way.

Where it lands on this project's own requirements:

| This project's rule | The twin |
|---|---|
| Strictly proper scoring rule (JU-4) + sharpness beside it (JU-5) | Coverage gap at ±1/2/3σ vs Gaussian targets — a calibration check, not a proper score; no sharpness. |
| Thresholds fixed before judging (JU-11, MU-6) | Tolerance is a CLI flag / default argument (`tol=0.05`) chosen at run time. |
| Separate "model is wrong" from "world is unpredictable" | One non-chaotic 1-D world; its "divergence onset" is model-vs-truth, not the world's own divergence floor. |
| Truth and model share one integrator/step, enforced by test (WD-3) | `check_aligned` checks horizon, start state and actions — not the timestep. |
| Trials and statistics (N=200, binomial bands) | Trust horizon is a first-crossing on one rollout per scenario, then a median; RMSE mixes state units with no scale normalisation. |
| Baselines and skill (MU-2) | None. |
| Fixtures labelled as fixtures (MU-4, RP-8) | `models.py` says plainly the models are injected-error mocks, but the README chart caption calls them "three learned models". |
| Action lever graded (state + action → next state) | Yes, and it refuses to compare rollouts driven by different actions. No action-response check. |

One idea worth remembering: its **physics-plausibility checks on the
model's rollout** (no NaN, passive energy must not rise). This project
checks the *world's* conserved quantity (the drift bound) but not whether
a *model's* predictions break it. Both worlds have a conserved quantity, so
"model violates the invariant" is a natural future check — front door
only, not now.

---

### A14 — the pre-registered training recipe is ~140× over the runtime budget (measured, 2026-09-27)

**What the spec says.** models §4: `training_trajectories: 2000` per world,
"of full horizon", "well inside the NF-2 budget". judge ADR-J6: the whole
run completes in under 600 s, with training "two small MLPs … budgeted at
≤ 6 minutes combined". `prereg/recipe.md`: `epochs: 100`, `batch_size: 32`.

**What the build measured (before any model was trained or judged).** One
world transition ≈ 26–30 µs; one Adam step on the real ADR-M3 net at batch
32 ≈ 230–250 µs (this machine, single-threaded). Full-horizon training sets
are 2,000 × 700 = 1.4 M examples (LV) and 2,000 × 5,000 = 10 M (pendulum).
At 100 epochs, batch 32, and **six** networks per world (direct + five
ensemble members — ADR-J6's "two small MLPs" undercounts), training costs
≈ 1.8 h (LV) + 11.9 h (pendulum) ≈ **13.7 h**, against a 6-minute training
allowance. Generating the pendulum data one trajectory at a time adds
≈ 263 s. "Well inside the NF-2 budget" is falsified by execution.

**Why it matters now.** P3-C06 reads these counts from the recipe; P3-C03/04
train with them. No faithful build of P3-C06 is possible until the numbers
are chosen. Any fix changes `prereg/recipe.md`, which runs into the
freeze-point contradiction (REMEMBER.md D2). No results exist yet, so a
revision is legitimate — through design review and an open, dated recipe
revision.

**Options for design review:** (a) keep the numbers, train once offline and
cache weights, and restate NF-2 to exclude training; (b) keep 2,000
training starts but use short training segments (the models are one-step
predictors — a full 5,000-step pendulum trajectory adds correlated examples,
not new information) and set epochs from a measured budget; (c) a larger
batch and vectorised data generation to cut cost per example. (b) + (c)
together is the recommendation. ADR-J6's "× 7 models" and "two small MLPs"
also need correcting (roster is 8; networks trained are 12).

### A15 — the form of the action sequences is not specified

**What the spec says.** models §4: training uses "seeded trained-range
action sequences"; judge ADR-J4: each evaluation trial has "its own seeded
action sequence". Nowhere says how an action sequence is drawn.

**Why it is a design decision, not a detail.** A fresh random push every
step (i.i.d.) at the pendulum's 0.002 s step averages out to noise, so a
model could ignore the action and lose almost nothing — which would make the
action lever, and P3-C08's action-blind check, close to meaningless. A push
held for a stretch of world time (piecewise-constant, resampled every k
steps from the training action interval) has a visible effect. The same
generator should serve training and evaluation so train and test actions
are alike. **Recommendation:** piecewise-constant, resampled from the
training interval every fixed span of world time, pinned per world in the
recipe; ratify at design review.

**A15 correction and measurement (same day).** The first version of A15
guessed that a fresh random push every step would "average out to noise".
Measured on 20 random training-region starts per world, that guess was
**wrong** — the truth is the opposite. Each action is an impulse applied
at every step (worlds ADR-W2), and at the declared training magnitude
(±0.5 prey units; ±0.5 rad/s on the first joint) a kick every step
overwhelms the worlds:

| Style | LV (10 s) | Pendulum (5 s) |
|---|---|---|
| Kick every step (i.i.d.) | 16/20 runs crash through the prey floor | state driven ~13× the world's scale |
| Kick every step, held 0.2 s | 20/20 crash | ~118× scale |
| Occasional kick (~1 per second of world time) | 0 crash; ~0.19× scale effect by 10 s | 0 crash; ~0.16× scale by 5 s |

Only occasional discrete kicks are safe and meaningful — and that is what
worlds ADR-W2 already intends ("a discrete intervention: 'remove rabbits
now', 'kick the pivot now'"); the spec just never said how often. The
recommendation is therefore **sparse discrete kicks** at a pinned rate,
drawn from the training action interval, one generator for training and
evaluation. Consequence to design for: kicks are rare, so the training
sample must deliberately include enough kick transitions for the models to
learn the lever. Also found: `lv.transition` given a batch of states
silently returns a wrong-shaped array instead of refusing (a fail-loud gap;
it is only ever called one state at a time today).

---

## Disposition (Round 10, 2026-09-28) — the recipe revision, pinned

**In plain words:** Round 10 (`design-review/design-review-010.html`) found
twenty holes in the plan for training the two practice models. The owner
decided every one on 2026-09-28: fix nineteen, defer one. This section is
the single authoritative statement of *what was decided*, with every
number, name and formula pinned, so that the spec edits below, the one
open revision of `prereg/recipe.md`, and the independent fix-check all
read from one place. Where a spec says "per design-review-010", this is
what it means. Every measured figure here was run on 2026-09-28 in this
environment (single-threaded; not the 4-core reference laptop).

**Owner decisions:** NF-2 keeps "whole run in minutes" and states its
purpose (re-runnability), 600 s a revisable measured target, evaluation
batched; kicks made smaller/rarer so nothing crashes (no redraw
filtering); the prediction kept, its reasoning rewritten; all other
recommendations accepted. 19 fix, 1 defer (R10-F19), 0 dismiss.

### R10-P1 — NF-2 (F01 context) — requirements v1.4
NF-2 now names its purpose: anyone can re-run the whole judge on a laptop
and check the verdict. The number lives in judge ADR-J6 as a revisable
target backed by measurement.

### R10-P2 — Batched evaluation (F01) — models ADR-M1, judge ADR-J6, cross-cutting ADR-004
- The `Model` protocol gains `stateless: bool` (True iff `predict` depends
  only on `(state, action)` — no rollout-local memory) and, for stateless
  models, `predict_batch(states: float64[n,d], actions: float64[n,a]) ->
  (means float64[n,d], spreads float64[n,d])`. **Row `i` must be
  bit-identical to `predict(states[i], actions[i])`** (new TC-MU1-04, every
  registered stateless model, batch sizes {1, 2, 7, 64, 200}).
- Stateless: `persistence`, `direct`, `ensemble`, all four fixtures.
  Stateful: `linear` (keeps the previous state) — rolled out per trial with
  `reset()` as before. The harness rolls all N trials of a region forward
  together, one `predict_batch` per step, for stateless models.
- **Why the prediction path needs its own forward (measured):** NumPy's
  BLAS matrix multiply is *not* batch-size invariant on this platform —
  odd batch sizes differ from one-at-a-time in the last bit (~2e-16;
  even sizes matched). `np.einsum('bi,io->bo', A, W, optimize=False)` was
  row-invariant at every size tested and ~1.7× slower than BLAS. So the
  MLP gains a prediction forward (`forward_invariant`: per-layer einsum +
  bias, tanh hidden, linear output) used by every `predict` and
  `predict_batch`; training keeps the BLAS `forward`. TC-MU1-04 is the
  guard — the invariance is tested, not assumed.

### R10-P3 — Runtime envelope (F01, F02, F20) — judge ADR-J6
Measured components (this environment): training 12 networks at M = 50,000,
batch 256, 100 epochs ≈ 196 s (LV ≈ 13.8 s, pendulum ≈ 18.9 s per network);
batched evaluation ≈ 1–1.5 min (3 regions × 200 trials × (700 + 5000) steps,
~10 network forwards per step via `forward_invariant`, plus `linear`'s
per-trial loop); batched true-world rollouts ≈ 4 s; gradient checks ≈ 11 s (12 networks).
Whole run ≈ 5–6 min here. Unbatched it would be ≈ 12–14 min (≈ 34M one-at-a-time network calls ≈ 6–8 min, plus unbatched truth and benchmark rollouts — corrected by the fix-check; the first figure, 10–11 min, reused the proposal's two-region count). **These
evaluation and whole-run figures are estimates built from measured
per-call costs, not an end-to-end measurement** (the gradient checks are ≈ 11 s for 12 networks — corrected by fix-check pass 2) — the pipeline does not
exist yet; TC-NF2-01 is the first end-to-end measurement (P6-C02). The target
stays 600 s **on the reference 4-core laptop** (TC-NF2-01 measures it there;
this environment is not that machine). Revisable; any revision is recorded
with a measurement.

### R10-P4 — Gradient check (F02, A10 ratified) — models ADR-M3 / §8
Runs once per training run on a fixed **64-pair** batch (`gradcheck_pairs:
64`), the first 64 of a permutation of the M training pairs drawn with
`seeds.rng_for(world, "training", "gradcheck-batch")`. It validates
**backprop**, not the loss: `direct`'s net is checked under the plain
Gaussian NLL, each ensemble member under the MSE loss. **A10 ratified:**
relative-error denominator floored at `GRADIENT_SCALE_FLOOR = 1e-3` × the
largest gradient magnitude; tolerance **1e-5** (spec's 1e-6 superseded); the
near-dead-unit blind spot is disclosed. Measured: ≈ 0.8–0.9 s per network at 64 pairs (≈ 11 s for all 12) vs
≈ 400–700 s at 50,000 — the "full-batch" wording is retired.

### R10-P5 — Training data (F03, F04, F08, F10, F14) — models ADR-M1 `TrainingData`, cross-cutting ADR-002
Built **once per world by the harness** (P3-C06) and handed identically to
every factory (ensemble members, fixtures rebuilding `direct` — TC-MU4-02).
- 2,000 full-horizon trajectories from training-region starts
  (`"train-starts"`), actions from the kick generator (R10-P6), trajectory
  `i` drawing kicks from `seeds.rng_for(world, "training", "train-kicks",
  str(i))`, generated with the batched world step (R10-P7).
- A **pair** is `(state_t, action_t, state_{t+1})`; a **kick pair** has
  `action_t ≠ 0`.
- **Kick pairs:** a fixed count **`kick_pairs: 12500`**, the first 12,500
  of a permutation of all kick-pair indices drawn with
  `seeds.rng_for(world, "training", "subsample-kick")`. If fewer than 12,500
  kick pairs exist, refuse loudly (`TrainingDataError`) — never shrink the
  count silently. (Expected available: LV ≈ 14,000, pendulum ≈ 20,000.)
- **Non-kick pairs:** `subsample_pairs − kick_pairs` = 37,500, the first of
  a permutation drawn with `"subsample-nonkick"`. Training set = kick pairs
  then non-kick pairs, each in permutation order (epochs shuffle it anyway).
- **Held-out pairs:** 10,000 (`heldout_pairs: 10000`) from the pairs *not*
  selected, first of a permutation drawn with `"heldout"` — a natural mix.
  Used only by the sufficiency check and the kick/non-kick report; never
  for early stopping or tuning.
- `TrainingData` gains: `train_pairs` (state[M,d], action[M,a],
  next[M,d], is_kick[M]), `heldout_pairs` (same shape, 10,000),
  `gradcheck_index` int[64]; the raw `states`/`actions` trajectories stay
  (the baselines fit from them).
- **Disclosed (F10):** training pairs are 25% kicks; evaluation steps are
  ~1% (LV) / ~0.2% (pendulum) kicks. Identical for both unrigged models.
  P3-C03/C04 report each model's held-out one-step error split by
  kick/non-kick pairs.

### R10-P6 — Kicks (F06, F07, F09) — worlds ADR-W2/W3/W4, §4, §7
- **One generator,** `wmj.worlds.actions.kick_sequence(rng, horizon,
  p_step, band, umax)` (pure; the harness passes a seeded `Generator`).
  Draw: `r = rng.random((horizon, 3))`; a kick at step t iff `r[t,0] <
  p_step`; `band="in"`: `u = umax·(2·r[t,1] − 1)`; `band="out"`: `u =
  s·umax·(2 − r[t,1])`, `s = +1` if `r[t,2] < 0.5` else `−1` (so
  `|u| ∈ (umax, 2·umax]`); non-kick steps `u = 0.0`. Returns
  `float64[horizon, 1]`. Draws are unconditional (three per step) so the
  stream is layout-stable.
- `p_step = kick_rate_per_s × dt`: **LV 0.5 /s × 0.02 = 0.01; pendulum
  1.0 /s × 0.002 = 0.002.**
- **LV action intervals shrink (owner: smaller/rarer kicks, measured):**
  trained `[−0.1, 0.1]` (was `[−0.5, 0.5]`), out-of-range `|u| ∈ (0.1, 0.2]`
  (was `(0.5, 1.0]`); declared full range `ACTION_RANGE` stays `(−1, 1)`.
  Pendulum unchanged: trained `[−1, 1]`, out-of-range `(1, 2]`.
  Measured over the full 700-step horizon, 2,000 runs each: LV
  training/in-range, LV out-high-amplitude/in-range, LV training/out-of-range
  — **0 floor crashes in all three** at 0.1 max, 0.5 /s (the proposal's 0.5
  max at 1 /s crashed 7, 477 and 337 respectively). The cost: the LV lever is
  smaller — mean horizon effect 0.12 state units (training region) vs 0.93.
- **New action-axis out-region per world, `"out-large-action"`:** state box
  = the training box, action band = the out-of-range band, `axis="action"`.
  This is what makes WD-5/TC-WD5-02's action axis producible (no world
  declared one before). Each world now has 3 regions.
- **Evaluation trials:** 200 per region; trial `i` of region `r` draws kicks
  from `seeds.rng_for(world, r, "eval-kicks", str(i))` with that region's
  band (`"in"` for training and state-axis regions, `"out"` for
  `out-large-action`).
- **Benchmarks (F09):** 64 starts per region; start `i` draws kicks from
  `seeds.rng_for(world, r, "benchmark-kicks", str(i))` with the region's
  band, applied **identically to the trajectory and its perturbed twin** —
  so the reference line is the world's own drift under the same kind of
  kicks. The TC-WD3-03 conserved-quantity drift check stays **null-action**
  (the quantity is only conserved without kicks). Measured with kicks: LV
  final/initial ratio 0.92 (gate 0.1–10), pendulum inverted/training at
  half-horizon 3,466× (gate ≥ 5×) — both sanity gates still hold.
- **§7 unchanged:** any floor breach aborts the run loudly. Because every
  kick sequence is seeded, the real job either always passes or always fails
  at build time; new **TC-WD2-02** runs the pinned generator at full scale
  in every region and asserts no clamp.

### R10-P7 — Batched world step (F11) — worlds ADR-W1, §4.3
One batch-aware `_deriv` per world (indexes `S[..., i]`, uses `np.sin`/
`np.cos`), used by both `transition` (1-D) and new
`transition_batch(states[n,d], actions[n,a])`; both call the **same
`rk4_step` function object** (TC-WD3-01's identity check is unchanged).
`transition` refuses non-1-D input (`WorldInputShapeError`) — closing the
silent wrong-shape bug. New **TC-WD3-04**: `np.array_equal` of every row of
`transition_batch` against `transition`, n ∈ {1, 2, 7, 64, 200}, both
worlds, including kicked actions. Measured: bit-identical here; `np.sin`/
`np.cos` equalled `math.sin`/`math.cos` on 2,000,000 values (so TC-WD1-01's
pinned reference values should be unaffected — the test re-confirms).

### R10-P8 — Model A's loss: β-NLL (F12) — models ADR-M3
β = 0.5 (`beta_nll: 0.5`). Per dimension, per example, with `r = y − μ`,
`s = log σ` and weight `w = σ^(2β)` **computed from the current forward
pass and treated as a constant** (stop-gradient):
`L = w · [0.5·log(2π) + s + r²/(2σ²)]`, summed over dimensions, averaged
over the batch. Gradients handed to backprop (`d_output`):
`∂L/∂μ = −w·r/σ²`, `∂L/∂s = w·(1 − r²/σ²)`, each divided by the batch
size. New **TC-MU5-04**: the `d_output` closure matches central finite
differences taken over the network *outputs* with `w` held fixed. The
backprop gradient check (R10-P4) uses plain NLL, because finite differences
of β-NLL through the weights would also differentiate `w`.

### R10-P9 — The prediction's reasoning (F13) — `prereg/prediction.md`
Prediction kept (ensemble better calibrated, longer trust horizon); its
"why" rewritten to not rely on plain NLL's overconfidence: a single
network's self-predicted error bar learns the noise it saw in training and
has no signal for its own ignorance away from the data, while disagreement
among independently initialised members grows where data is thin — most
visibly out of region and at longer horizons, where errors compound.
Changed before any training, so it is still a prediction.

### R10-P10 — Is M enough? (F14) — models ADR-M3, recipe
Outcome-independent, pre-registered, checked once at build time (P3-C03
for `direct`, P3-C04 for `ensemble`), before the freeze, and never against
the MU-5 margin. Train at M = 50,000 and at 2M = 100,000 (the 2M set = the
same 12,500 kick pairs plus the first 87,500 non-kick pairs of the same
permutation; everything else identical). Held-out error = mean over
held-out pairs of mean over dimensions of `((mean − y)/scale)²`. **M is
sufficient iff `err(M) ≤ 1.10 × err(2M)`** (`sufficiency_tolerance: 0.10`)
for **both** models. If either fails, `subsample_pairs` becomes **100,000**
(`subsample_pairs_fallback: 100000`, the only pre-registered fallback), for
both models, recorded in the recipe's revision log; no further iteration.
Runtime at 100,000: ≈ 8.5 min here — still inside the target.

### R10-P11 — The freeze (F15–F18) — models ADR-M5, TC-MU6-*
- `prereg/FREEZE` is a short human-readable declaration (date, statement);
  it contains no commit id.
- **The freeze commit is the commit that added `prereg/FREEZE`** — resolved
  from git history, never chosen. `check_prereg` refuses if `prereg/FREEZE`
  has ever been added more than once or ever deleted
  (`git log --diff-filter=A --format=%H -- prereg/FREEZE` must print exactly
  one line and `git log --diff-filter=D --format=%H -- prereg/FREEZE` none) — **one freeze, ever** — or if it is absent ("not frozen
  yet").
- Every certified file (`recipe.md`, `prediction.md`, `thresholds.json`,
  `FREEZE`) must equal its blob **at the freeze commit** and be clean in the
  working tree; the freeze commit's timestamp must precede the run.
- `check_prereg` returns the **freeze commit SHA**, recorded as
  `meta.prereg_commit` (P6-C01). The freeze mechanism (P3-C10) is built
  **before** P6-C01.
- The git tag `prereg-freeze` is an optional human label; `check_prereg`
  never reads it.
- Before the freeze, `recipe.md`/`prediction.md` may be revised openly (each
  revision dated in the recipe's revision log); their earlier versions stay
  in history.
- **Procedural rule + disclosed residual (F16):** no evaluation-trial metric
  of either unrigged model may be computed before the freeze. That cannot be
  enforced mechanically (an unrecorded informal run leaves no trace — the
  same family as ADR-M5's non-publication residual); it is disclosed, not
  claimed. The build-time sufficiency check (R10-P10) uses held-out
  *training-region* pairs only, never evaluation trials or the margin.
- Tests: TC-MU6-01 and TC-MU6-04 rewritten to the freeze commit; new
  TC-MU6-06 (second add or a delete refused), TC-MU6-07 (the returned SHA is
  the freeze commit and reaches `meta.prereg_commit`), TC-MU6-08 (no FREEZE
  → refused), TC-MU6-09 (the recipe's kick/interval keys equal the world
  modules' constants — the frozen recipe pins them and the code cannot drift
  silently).

### R10-P12 — Recipe keys (F05) — `prereg/recipe.md`
```
training_trajectories: 2000
epochs: 100
batch_size: 256
subsample_pairs: 50000
subsample_pairs_fallback: 100000
kick_pairs: 12500
heldout_pairs: 10000
gradcheck_pairs: 64
sufficiency_tolerance: 0.10
beta_nll: 0.5
matching_margin: 0.05
ensemble_members: 5
lv_kick_rate_per_s: 0.5
lv_action_max: 0.1
pendulum_kick_rate_per_s: 1.0
pendulum_action_max: 1.0
```
`p_step = kick_rate_per_s × dt` (stated beside the keys). `batch_size`
moves 32 → 256 (measured 2.5× cheaper per example); models ADR-M3's "batch
size 32" is superseded.

### R10-P13 — Seed purpose keys (F04) — cross-cutting ADR-002 rule 2
New purposes, all distinct from `"train-starts"`/`"eval-starts"`/
`"benchmark-starts"`: world-scoped `("training", "train-kicks", i)`,
`("training", "subsample-kick")`, `("training", "subsample-nonkick")`,
`("training", "heldout")`, `("training", "gradcheck-batch")`;
region-scoped `(region, "eval-kicks", i)`, `(region, "benchmark-kicks", i)`.
New **TC-NF1-10**: all are pairwise distinct and none collides with an
existing purpose.

### R10-P14 — Chunks (wiring) — implementation guide v2.0
- **New P3-C09 · Batched world step + kick generator + action-axis region +
  kicked benchmark.** Needs P2-C01..C04. Enables P3-C06. Updates `lv.py`,
  `pendulum.py`, `regions.py` as needed, `benchmarks.py`; adds
  `wmj/worlds/actions.py`. Tests: TC-WD2-02, TC-WD3-04, TC-WD5-02 (action
  axis now producible), TC-WD4-01/02 (re-run with kicks), TC-WD1-01
  (re-confirmed after the `_deriv` rewrite).
- **New P3-C10 · The freeze mechanism.** `check_prereg` rewrite per R10-P11.
  Needs P3-C07. Enables P6-C01 and P6-C03. Tests: TC-MU6-01, -04, -06, -07
  (return value half), -08, -09.
- **P3-C01 amendment (inside P3-C03):** `forward_invariant`.
- P3-C06: R10-P5. P3-C03: β-NLL, gradient check, sufficiency, kick report,
  `predict_batch`. P3-C04: batch 256, sufficiency, kick report,
  `predict_batch`. P3-C05/C08: `predict_batch` on fixtures. P3-C02:
  `persistence.predict_batch`, `stateless` flags on both baselines.
- **Wiring gaps closed:** TC-MU6-05(b) → **P4-C02**; recording the freeze SHA
  in `meta.prereg_commit` → **P6-C01** (TC-MU6-07 metadata half); the
  Round-9 climatology/drift note → P4-C05's text; batched rollouts → P6-C01.
- Roster, chunk total: 8 contestants; 31 chunks.

### R10-P15 — Ratifications (R10-6)
- **A10** ratified (R10-P4).
- **A12** ratified: `epochs: 100`; `sharpness_hedge_threshold = scale` per
  world (not per region).
- **A13** ratified: ADR-M5 gains the fourth disclosed residual — entry
  substance is not verifiable (a hollow mention of a model's name passes).
- **B1** text (MU-3's fourth fixture, TC-MU3-04, P3-C08) reviewed in this
  round with no finding against it.

### Deferred
- **R10-F19** (hashing cached weights): only needed if the judge ever ships
  pre-trained weights instead of training from scratch. The owner chose
  from-scratch runs, so it is moot; recorded in REMEMBER.md §6.

### A16 — found while applying Round 10: the climatology reference is still null-action (open, for the next review)

**In plain words:** after the model's forecast stops being trustworthy, the
judge compares it with "what the world usually looks like" — the
climatology. That reference comes from one long *unkicked* run. The test
runs now carry kicks, which nudge a trajectory between the reference's
bins. With kicks this small and rare the effect should be modest, but
nobody has measured it.

**What the spec says.** Worlds §5: the conditioned climatology is built from
"one continuous 200,000-step null-action reference trajectory per world",
binned by the conserved quantity into 16 equal-population bins; judge
ADR-J5/JU-6 grade post-switch predictions against it.

**What changed.** Round 10 put seeded kicks into every evaluation trial
(worlds ADR-W2) and into the divergence benchmark (ADR-W3), but did not
touch the climatology — no panel raised it, and it was noticed only while
writing the worlds edits. Recorded here rather than silently decided
(owner decision D15 in `REMEMBER.md`).

**Proposal.** Measure at P4-C05 how far kicked trials move across bins after
the switch step, and disclose the figure; if it matters, route a design
change (e.g. a kicked reference trajectory) through the next review.

### A17 — found building P3-C09: three small additions the specs do not yet name (for the next review)

**In plain words:** building the batched worlds and the kick generator
needed three things the specs don't mention by name. They are small and
follow the pinned decisions, but a later builder reading only the specs
would not know about them.

1. **The World interface gained `kick_rate_per_s`** (worlds §4.3 lists
   `transition_batch` but not this). Each world declares its own kick
   rate (LV 0.5, pendulum 1.0 per second), and the harness reads it to
   compute `p_step = kick_rate_per_s × dt`. Every future world must
   declare it.
2. **Two helper modules:** `wmj/worlds/regionspec.py`
   (`validate_out_regions`, the one region check both worlds call — it
   now also ties each region's declared action box to the kicks its
   trials really get: the trained interval for state-axis regions,
   exactly twice it for action-axis regions, and the training box as the
   state box of an action-only region) and `wmj/harness/kicks.py`
   (`seeded_kick_sequences`, the one place a seeded kick sequence is
   drawn for a world, region and purpose). Neither is in the
   implementation guide's wiring matrix; the next guide revision should
   list them (consumers: both worlds; `harness.benchmarks`,
   `harness.preview`, P3-C06's training data, P6-C01's trials).
3. **An `out-large-action` trial could, in principle, carry no kick.**
   Its kicks come only from the out band, so a trial with zero kicks
   would be labelled fully in-region (`axis` null) despite its region
   name. At the pinned rates this is rare (LV ≈ 0.09% of trials, the
   pendulum far less), and with the pinned seed it does not happen —
   `tests/gates/test_kick_safety_full_scale.py` now asserts every such
   trial is kicked at least once, so a seed that broke it would fail
   loudly. The next review should decide whether to guarantee it by
   construction (e.g. force at least one kick per action-axis trial).

*Where recorded:* `build/handovers/P3-C09.md`; independent review of
P3-C09, pass 1 (findings 2, 7, 8).

### A18 — found building P3-C10: the lock is stricter than ADR-M5's wording (for the next review)

**In plain words:** ADR-M5 says how the one-time freeze is checked using one
particular git command. Building it showed that command misses real ways to
cheat, so the code checks more. The spec text was *not* edited (it is
reviewed); this note is what the next design review should fold in.

1. **Counting adds and deletes.** ADR-M5 says to count with
   `git log --diff-filter=A|D`. That cannot see a freeze added or lifted
   *inside a merge commit* (git shows no diff for a merge). The code instead
   asks every commit whether `prereg/FREEZE` exists in its tree: an *add* is
   a commit that has it while none of its parents do; a *delete* is a commit
   without it while some parent has it.
2. **Every local ref, not only HEAD.** A branch forked from before the freeze
   can freeze again without touching the checked-out history. Adds and deletes
   are counted over all refs (branches, tags, remote-tracking refs; not
   `refs/replace/*` or `refs/stash`). The single freeze must also be in the
   checked-out history, else "not frozen on the branch being judged".
3. **History must be trustworthy first.** The code refuses: a repo that is not
   at its top level; a shallow clone; a `.git/info/grafts` file; a partial
   clone (git would run the repo's own fetch program to fill a gap); and
   anything `git fsck` rejects (forged objects). Git is run with ambient
   `GIT_*` dropped, replace refs and the commit-graph off, no lazy fetch, and
   no repo-supplied program (fsmonitor, hooks, clean filters — `git status` is
   never used; bytes are compared directly).
4. **The certified set is fixed** (`recipe.md`, `prediction.md`,
   `thresholds.json`, `FREEZE`); the caller's list can only add to it.
5. **Line endings.** `.gitattributes` pins `prereg/* -text`; a CRLF-only
   difference is reported as such (`PreregLineEndingError`).
6. **Residual #6 (new, disclosed in the module docstring).** The certificate
   is a commit id, not the bytes the judge later reads. P6-C01 should read
   and hold the verified bytes (or re-check immediately before use).
   Residual #7: only the repository on this disk is examined.

*Where recorded:* `build/handovers/P3-C10.md` (to be written at convergence);
independent review of P3-C10, passes 1–3.

### A19 — found building P3-C06: two things the training-data spec leaves open (for the next review)

**In plain words:** the spec says how to pick the training examples; building
it showed one choice the spec made silently and one consequence nobody
measured.

1. **`TrainingData`'s three new fields are optional in the code.** The spec
   shows `train_pairs`, `heldout_pairs` and `gradcheck_index` as plain
   required fields. The two earlier builders (the skeleton and the
   chart-preview) fit the baselines on a handful of trajectories and have no
   such pick, so the fields default to `None` there. The MLP factories
   (P3-C03/C04) must refuse `None` when they build. Next review: either make
   the fields required and give those two callers a small pick, or write the
   optional-with-refusal rule into ADR-M1.
2. **The held-out set contains almost no kicked examples** (measured at full
   scale, seed 20260825). The spec draws the held-out set from the pairs *not*
   chosen for training. The training set takes 12,500 of the kick pairs the
   world supplies — LV has 13,874, the pendulum 19,900 — so what is left is
   about 1,374 kick pairs in 1.35 million (LV) and 7,400 in 9.9 million
   (pendulum). A 10,000-pair held-out draw therefore has **6 kick pairs (LV)
   and 10 (pendulum)**. The build-time report "held-out error split by kick
   and non-kick pairs" (ADR-M3) will rest on those few examples for the kick
   half, so it cannot support any conclusion about the models' behaviour on
   kicks. Not a build error; a design question for the owner (**D17** in
   `REMEMBER.md`): e.g. give the held-out set its own kick quota, or report
   the kick split on a separately generated kicked held-out set. Any change
   is a pre-freeze recipe revision, logged openly. Also noted: LV's margin
   over the 12,500 quota is small (13,874 available, 11% spare), so a lower
   kick rate or a shorter horizon refuses the build (as designed).

*Where recorded:* `build/handovers/P3-C06.md`; `REMEMBER.md` D17.
