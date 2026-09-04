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
