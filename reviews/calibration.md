# Review Calibration — World Model Judge

Project-level calibration layer for GVM review skills (`/gvm-design-review`, `/gvm-code-review`, `/gvm-doc-review`). Per `review-reference.md`'s progressive-calibration protocol: this file is read at the start of every review round and updated at the end of it.

---

## Score History

| Round | Date | Type | Overall | Panel A (Coverage) | Panel B (Contracts) | Panel C (Structural) | Panel D (Implementability) | Panel E (Security) | Panel F1 (Reproducibility) | Panel F2 (Separability) |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 2026-08-25 | design | — (see per-panel) | 8 | 4 | 5 | 4 | n/a | n/a | n/a |
| 2 | 2026-08-30 | design | — (see per-panel) | 7 | 5 | 4 | 5 | 6 | 5 | 7 |
| 3 | 2026-08-30 | design (dual/blind) | — (see per-panel) | 6 | 4 | 4 | 6 | 5 | 4 | 3 |
| 4 | 2026-08-30 | design (dual/blind) | — (see per-panel) | 4 | 3 | 3 | 5 | 4 | 5 | 2 |

No overall single number is reported for design review, per `review-reference.md`. Dual/blind (shared rule 16) has now run twice — Round 4 again dispatched 7 panels × 2 (calibrated + blind), 14 independent reviewers, prioritising adversarial pressure-testing of Round 3's two flagship fixes (`fx-brittle`'s redesign; the registry auto-discovery/TC-MU9-01 mechanism) per this file's own recommendation after Round 3. **Both failed pressure-testing, in ways more serious than what they replaced** — see Resolved Findings below. Panel F1's score rose despite the general trend because the dual-review mechanism itself produced its cleanest demonstration yet of working correctly: F1-calibrated read the actual CPython `pkgutil` source and used it to disprove a plausible-sounding finding from F1-blind, rather than letting it stand. Panel F2 fell to its lowest score of any round: the replacement TC-MU9-01 mechanism introduced a new failure mode (production-directory pollution on test failure) worse in kind than the tautology it replaced.

---

## Dimension Benchmarks

| Dimension | R1 | R2 | R3 | R4 | Trend |
|---|---|---|---|---|---|
| Requirements Coverage (Panel A) | 8/10 | 7/10 | 6/10 | 4/10 | ↓↓ — BC-1 recurred a 4th consecutive round in the exact blocks Round 3 just restructured; 10 test cases found orphaned from any build chunk, one (TC-JU3-01) unwired since Round 1 |
| Interface Contracts (Panel B) | 4/10 | 5/10 | 4/10 | 3/10 | ↓ — `fx-brittle` contradiction; `region_labels` claimed-unified but never applied to worlds.md; registry return contract unspecified |
| Structural Soundness (Panel C) | 5/10 | 4/10 | 4/10 | 3/10 | ↓ — the two flagship Round 3 fixes found mutually exclusive on the one fixture exercising both |
| Implementability (Panel D) | 4/10 | 5/10 | 6/10 | 5/10 | ↓ — the "corrected" critical path is still wrong (a 3rd tied branch omitted); chunk count (28) held |
| Security (Panel E) | — | 6/10 | 5/10 | 4/10 | ↓ — two brand-new checks (`exec`/`eval`, `numpy.random`) both evadable; undisclosed CI and network dependencies introduced this round |
| Reproducibility (Panel F1) | — | 5/10 | 4/10 | 5/10 | ↑ — dual review's cleanest self-correction yet: a blind finding was checked against real CPython source and disproven |
| Separability (Panel F2) | — | 7/10 | 3/10 | 2/10 | ↓↓ — the replacement TC-MU9-01 mechanism can pollute the production model directory on its own designed-to-fail case |

**Round 4 pressure-tested Round 3's two flagship fixes, per this file's own recommendation, and both failed.** `fx-brittle`'s redesign requires an import models.md's own rules forbid, with no wiring mechanism that doesn't also break the new TC-MU9-01 test — the two Round 3 fixes are mutually exclusive on the one fixture that exercises both, undisclosed by either. Separately, dual review produced its clearest demonstration yet of working as designed: F1-blind's claim about nondeterministic registry-discovery order was checked by F1-calibrated against actual CPython `pkgutil` source and found false — recorded honestly as a disproven finding, not silently dropped. Four rounds have now each found a comparable-or-larger defect population despite continuously increasing scrutiny; **this round's entry carries an explicit structural recommendation (see Build Checks) that continued one-finding-at-a-time patching is no longer the right lever.** See `design-review/design-review-004.html` for the full findings, convergence table, and reconciliation record. **Verdict: Do not build**, unchanged since Round 1.

---

## Anchor Examples

### Panel A — Requirements Coverage
- **Best (9-10 band, illustrative — not yet achieved):** N/A this round.
- **Best observed this round:** 43/45 requirements fully Covered with concrete design elements, zero Missing — countable, verifiable, band-demonstrating (Marzano/Gilb).
- **Worst observed this round:** JU-10's seven disclosures counted but never authored — a placeholder string (`"<the seven JU-10 disclosures, fixed text>"`) stood in for real content in the Verdict schema. Concrete, verifiable, self-contained (Fagan).
- **Worst (1-2 band, illustrative):** N/A this round.

### Panel B — Interface Contracts
- **Best observed this round:** N/A — Panel B reported only mismatches; a clean boundary generates no finding by design (the skill's own instruction: "if a boundary is fully consistent, do not report it").
- **Worst observed this round:** `is_fixture` appeared as a field the judge-produced Verdict carried, while the same document's own input-type description said the judge structurally cannot know it — an internal self-contradiction within one file, independently rediscovered by two panels. Concrete, verifiable via direct quote comparison, self-contained.

### Panel C — Structural Soundness
- **Best observed this round:** the WD-3 gate finding — Panel C traced a named architectural mechanism ("the model-rollout driver's integrator") against the actual models spec and found no such component exists in any of the eight built contestants. A strong example of cross-reference-scan catching a claim that doesn't survive contact with its own dependencies.
- **Worst observed this round:** the architecture-overview's own Brooks-style self-check ("no second metric exists anywhere") was checked against the specs it was auditing and found false. Notable because it's a defect in a document whose entire purpose was to catch exactly this class of defect.

### Panel D — Implementability
- **Best (R1) observed:** the MVP-1/Tracer-Bullet Assessment — a structured three-part check (is the first slice real? is the sequence incremental? evidence for both) that produced a specific, actionable finding (chunk 19 of 23 before further user-visible value) rather than a vague "could be more incremental."
- **Worst (R2) observed:** re-deriving the double-pendulum equal-mass equations of motion from Euler–Lagrange found `denom1`/`denom2` and θ̈₁'s leading gravity term use coefficient `2·m` where the correct general-mass formula specializes to `3·m` — and the spec states its own hand-verified 15-sig-fig reference constants are computed from this exact (wrong) formula, so no test in the suite as designed can catch it. The single strongest example this round of "precision without a recount is not correctness": R1's finding (formula missing) was fully addressed on its own terms, and the replacement is worse than the gap it filled.

### Panel E — Security (new)
- **Best observed:** the positive finding — `JudgeInput` as a plain array/float dataclass has structurally nowhere to place a model identity, so the "two disagreeing recognisers" LANGSEC test has no live channel to exploit, independent of the import gate. Genuine defence in depth, not merely asserted.
- **Worst observed:** the NF-4 confidentiality-scan mechanism's own remediation instruction — write the actual forbidden term into a tracked source file — would cause the exact violation the mechanism exists to prevent, on the one requirement this project treats as absolute and irreversible.

### Panel F1 — Reproducibility (new)
- **Worst observed (R2):** the only described judge-purity gate (an AST walk scoped to `wmj.*` imports) has no coverage at all for direct ambient reads (`os`, `time`, `sys`) — a class of violation that can pass a same-machine repeated-run test while still breaking NF-1's explicit cross-machine scope.
- **Worst observed (R3):** the broadened gate still doesn't catch `numpy.random.*` legacy-global calls, even though ADR-002 explicitly names this gate as that exact rule's enforcement mechanism — `numpy` is the one module the judge is required to import, so it can never be blocklisted, and an attribute-call on an already-legitimate import matches none of the gate's three described shapes.

### Panel F2 — Separability (new R2)
- **Best (R2) observed:** the harness-owned envelope's construction was found to require zero model-specific branching — genuine, verified defence in depth for the modifiability scenario.
- **Worst (R3) observed:** the named TC-MU9-01 mechanism (`test_registry_isolation.py`) stages, commits, and diffs a fixture file it fully controls end-to-end — the diff can only ever show the one file the test itself chose to write. It is structurally incapable of ever failing, regardless of what a real developer's actual changeset does. Two independent reviewers (one blind, one calibrated) reached this conclusion with zero shared context — the strongest form of convergence this methodology produces.
- **Worst (R4) observed:** the replacement mechanism's cleanup step has no exception-safety, and its stub file is written into `wmj/models/` — the actual production discovery directory. A failed test run (which the design's own text says can happen) can leave a phantom model a subsequent real `wmj run` auto-discovers and executes. Independently confirmed by both the calibrated and blind reviewers. Worse in kind than the tautology it replaced: the old mechanism was inert; this one can corrupt a real run.

### Dual review self-correction, Round 4 (an anchor example for the mechanism itself, not a project dimension)
Panel F1-blind claimed `wmj.models.registry`'s directory-walk discovery order was filesystem-dependent and therefore nondeterministic. Panel F1-calibrated did not take this on faith — it ran `python3 -c "import pkgutil, inspect; print(inspect.getsource(pkgutil._iter_file_finder_modules))"` against the live CPython stdlib and found `filenames.sort()` runs before yielding, meaning discovery order is deterministic regardless of filesystem/inode layout. The finding was recorded as **disproven by direct evidence**, not silently dropped and not left as an unresolved disagreement. This is the calibration process's best demonstration yet of the difference between "plausible-sounding" and "verified" — worth keeping as the reference example for what adversarial pressure-testing should look like (per BC-2).

---

## Recurring Findings — both Build Checks recurred a 4th consecutive round

1. **Schema/consumer granularity mismatch (BC-1).** R1: Chart 1. R2: Chart 2, Chart 3. R3: `calibration`, `sharpness`, `climatology`, `region_labels`, Chart 4, Chart 1's horizon_step. **R4: `calibration`/`sharpness` STILL don't carry a region key** (Round 3's own restructuring only added the task key, kept `_in_region`/`_out_region` suffixes — a claim of "matching every sibling block" that was checkable and false against the JSON two lines below it) **and `error_vs_horizon` was left with no task dimension at all.** Four consecutive rounds, promoted after the third, and recurred through the very round meant to close it.
2. **A fix closes the literal finding and leaves an adjacent, structurally similar gap one level down (BC-2).** R1→R2: pendulum EOM missing, then wrong. R2→R3: NF-4 instruction fixed, hook/history gaps remain; AST gate's 2 evasions closed, 3 new ones remain. **R3→R4: `fx-brittle`'s redesign and the new TC-MU9-01 mechanism — the two fixes this file explicitly told Round 4 to pressure-test first — both failed, and failed into each other** (mutually exclusive, not merely each individually incomplete). The two newest AST checks (`exec`/`eval`, `numpy.random`) reintroduced the exact "enumerate call shapes" weakness a third check in the same document, same round, explicitly names and fixes.

**Both Build Checks have now recurred in the exact round assigned to close them, for a fourth consecutive time. Per shared rule 21 this does not call for retiring either check — they are still triggering, not stale — but it is itself evidence worth escalating: promotion to a Build Check has not measurably slowed the recurrence rate. See the structural recommendation below.

---

## Resolved Findings

**Round 1's 26 findings, independently re-checked in Round 2** (this is the re-verification `reviews/calibration.md` named as the priority after Round 1 — see the note under Dimension Benchmarks):

| # | Finding | Round 2 status |
|---|---|---|
| C1 | `is_fixture` conflicting producers | **CONFIRMED-RESOLVED** (Panels B, C) |
| C2 | Verdict `meta` requires ambient access JU-12 forbids | **CONFIRMED-RESOLVED** (Panel C) |
| C3 | Chart 1 needs undeclared per-trial data + 3 undefined drawing rules | **REGRESSION (partial)** — field-name mismatch + missing X-ordering data (Panel B); `trials` missing from JU-9's mandatory-field list, unreconciled exception definitions (Panel C) |
| C4 | Climatology underspecified on 3 axes | **CONFIRMED-RESOLVED** (Panels B, C) |
| C5 | WD-3 gate names a nonexistent component | **CONFIRMED-RESOLVED** (Panels B, C, D) |
| C6 | Double-pendulum EOM not written out | **REGRESSION** — formula now explicit but contains a coefficient error (`2·m` where `3·m` is correct), self-consistently baked into the reference test constants (Panel D) |
| C7 | z_p not derivable from erf | **CONFIRMED-RESOLVED**, independently re-derived (Panel D) |
| C8 | `fx-honest-rough` spread formula missing | **CONFIRMED-RESOLVED**, formula independently verified correct (Panel D) |
| C9 | Implementation guide's own build sequence violates MVP-1 | **PARTIAL** — the P2-C05 fix itself is sound and demonstrable, but the guide's chunk-count (28 actual vs. stated 24) and a file-ownership/parallel-safety claim (P2-C05 vs P3-C02 both target `baselines.py`) are wrong (Panel D) |
| C10 | 4 unpinned training constants threaten MU-8's determinism guarantee | **CONFIRMED-RESOLVED**, all four independently verified consistent (Panel D) |
| Important #1–16 | (see design-review-001.html) | 13 of 16 **CONFIRMED-RESOLVED**; 3 folded into the C3/C9 partial-resolution items above |

**16 of 26 hold cleanly. 3 are only partially resolved or regressed. 7 were folded into related items.** Full detail in `design-review/design-review-002.html`.

**Round 2's own 26 findings, independently dual/blind re-checked in Round 3:**

| # | Finding | Round 3 status |
|---|---|---|
| Double-pendulum EOM correction | **CONFIRMED-RESOLVED — independently re-derived from scratch by 2 reviewers** |
| Chunk count (28) | **CONFIRMED-RESOLVED — independently recounted from scratch by 2 reviewers** |
| TC-WD1-01 reference-constant flag | **CONFIRMED-RESOLVED** |
| Envelope/model-identity non-branching | **CONFIRMED-RESOLVED** |
| `trials.is_exception`/`observed` definition | **CONFIRMED-RESOLVED** (definition correct; no build-enforced test — see NI-5, design-review-003.html) |
| Chart 1 field name + X-ordering | **CONFIRMED-RESOLVED** |
| `trust_horizons.region` field | **CONFIRMED-RESOLVED** (data exists; Chart 4's column-mapping rule to consume it does not — see NI-7) |
| `calibration.n_trials` | **CONFIRMED-RESOLVED** |
| JU-9 eight-field mandatory list | **CONFIRMED-RESOLVED** |
| Reporting full-envelope-set input contract | **CONFIRMED-RESOLVED** |
| NF-4 self-defeating instruction | **CONFIRMED-RESOLVED** |
| NF-4 vacuous-pass precondition | **CONFIRMED-RESOLVED as literally scoped** (new gap found one level down — see NC-1/NC-2) |
| AST gate: 2 named evasions | **CONFIRMED-RESOLVED** (new evasions found — see NC-3/NC-4) |
| TC-MU9-01 named mechanism | **REGRESSION** — the mechanism exists but is tautological, cannot fail by construction (NC-5) |
| `fx-brittle` half-training-box split | **REGRESSION** — the named split contradicts ADR-M4's own wrapper framing and the world-level region label it's graded against (NC-6) |

**11 of 15 distinct Round 2 items hold cleanly (including both of the two flagged as highest-risk). 2 are confirmed-resolved-as-scoped with a new adjacent gap. 2 are genuine regressions.** Full detail in `design-review/design-review-003.html`.

**Round 3's own new findings** (~12 Critical, ~12 Important) are listed in `design-review/design-review-003.html`.

**Round 3's 15 distinct items, independently dual/blind re-checked in Round 4:**

| # | Finding | Round 4 status |
|---|---|---|
| `fx-brittle` redesign (post-hoc wrapper keyed to world's region box) | **REGRESSION** — requires an import models.md's own rules forbid ("models never import world internals"); 6 independent panel-runs converged |
| No world-selection mechanism for `fx-brittle` across 2 worlds | **NEW, follows directly from the above** |
| TC-MU9-01 structural test (auto-discovery + AST reference walk) | **REGRESSION, worse in kind** — no exception-safe cleanup; stub written into the production `wmj/models/` directory; a failed run (which the design says can happen) can leave a phantom model a real `wmj run` then executes |
| The above two fixes' interaction | **NEW CRITICAL, undisclosed by either Round 3 fix** — mutually exclusive: fixing `fx-brittle`'s layering violation the only clean way requires `harness.trials` to name `fx-brittle`, which the new TC-MU9-01 AST-walk is built to fail on |
| AST gate widened to 5 checks | **PARTIAL** — the 2 literally-named Round 2/3 evasions close; the 2 brand-new checks (`exec`/`eval`, `numpy.random`) both use the exact "enumerate call shapes" pattern a 3rd check in the same round explicitly names as insufficient; concrete one-line evasions found by 4 panels |
| `wmj.models.registry` auto-discovery named explicitly | **CONFIRMED-RESOLVED for the discovery-order concern** (F1-blind's nondeterminism claim disproven against real CPython source by F1-calibrated) **but PARTIAL overall** — return contract, idempotency, and mid-process cache-invalidation all unspecified |
| NF-4 historical-blob scan | **PARTIAL** — closes committed-then-removed file content; misses reflog-only-reachable commits and dropped stash entries |
| `.githooks/pre-commit` install path named | **PARTIAL** — the disclosed "never configured" gap is honest; an undisclosed silent-override case (`core.hooksPath` already set by unrelated tooling) is not named |
| Spec-parity check traced to NF-5 with test + chunk | **CONFIRMED-RESOLVED** |
| `calibration`/`sharpness` restructured to per-task arrays | **REGRESSION** — BC-1's exact shape: no region key added, only task; change note's "matches every sibling block" claim is false against its own JSON |
| `climatology.per_task` region field | **CONFIRMED-RESOLVED** |
| `region_labels` unified to one canonical shape | **REGRESSION** — the claim was never actually applied to worlds.md (still v1.2, no changelog entry); `axis`'s value set also disagrees between the two documents that define it |
| `is_exception`/`observed` build-enforced test | **CONFIRMED-RESOLVED** |
| Critical-path arithmetic corrected (9 chunks) | **PARTIAL** — chunk count (28, and the 9-chunk total) both independently reconfirmed; the two-named-branches claim is wrong, a 3rd tied branch was omitted, found independently by 2 panels |
| 3rd prereg residual risk (timestamp forgery) disclosed, GitHub mitigation added | **REGRESSION** — the mitigation itself introduces an undisclosed network dependency with no fail-open/closed spec and a ~90-day silent-decay window |

**3 of 15 hold cleanly. 1 resolves the literal concern but leaves the block partial. 8 are partial (closed the named case, left an adjacent one). 3 are outright regressions, one of them newly discovered as a direct interaction between two "separately successful" fixes.** Full detail in `design-review/design-review-004.html`.

**Round 4's own new findings** (~12 Critical, ~13 Important) are listed in `design-review/design-review-004.html`.

---

## Build Checks

**BC-1 — Schema/consumer granularity mismatch.** Recurred a 4th consecutive round, in the round assigned to close it. See Recurring Findings above.

**BC-2 — Self-verified enforcement-mechanism fixes must be adversarially pressure-tested.** Recurred a 4th consecutive round; this round's two flagship pressure-tests (per BC-2's own instruction) both failed, and failed into each other. See Recurring Findings above.

**BC-3 — New (promoted this round, shared rule 21: a specific instance recurring 3+ rounds within a single build-plan document).** `implementation-guide.md` has been edited in every round so far **only** for chunk-count/critical-path arithmetic, never reconciled against the rest of the growing spec/test-case set. Consequence, found only when a panel finally checked directly: its Wiring Matrix cites a mechanism (TC-MU9-01's git-diff test) that was explicitly disproven and replaced two rounds ago; 10 of 79 test cases (including the sole test for a Must requirement, unwired since Round 1) are never wired to any chunk; and its own critical-path fix is still wrong a third time. **Trigger for future rounds:** any round that edits a spec's schema, test cases, or mechanisms must also re-check `implementation-guide.md`'s Wiring Matrix and dependency table against those changes — not just the arithmetic a prior finding happened to name.

**Structural recommendation (new, not a Build Check — a recommendation to the user for how to proceed, per Hard Gate 6's triage-is-user-owned principle).** Four rounds, four "Do not build" verdicts, a defect population that has not shrunk despite constantly increasing review depth, and two Build Checks that recurred through the exact rounds assigned to close them. This is not evidence the review process is failing — every citation was independently reached by 2+ reviewers with zero shared context, and this round's dual-review mechanism correctly disproved a plausible-sounding wrong finding using real evidence (see the F1 anchor example above). **What isn't working is fixing findings one at a time and re-reviewing the whole document set from scratch each round.** Recommended before a Round 5: (1) one canonical `(task, region, horizon_step)` keying convention applied to every Verdict schema block in one pass, checked against every defining ADR at once, not chart-by-chart; (2) an explicit ADR resolving the `wmj.models` ↔ `wmj.worlds` layering question, since `fx-brittle`'s design has silently assumed different answers twice; (3) replace every remaining "enumerate call shapes" AST check with the "ban the enabling name outright" pattern the project's own TC-NF6-03 already proves works; (4) one full reconciliation pass of `implementation-guide.md` against the current requirements/test-case/spec set; (5) remove or properly scope the two new dependencies (GitHub API, disclosed-but-unbuilt CI) that contradict the project's own "no external systems but git" architecture claim.

**Round 3 fixes applied (30 Aug 2026):** see the Round 3 status table above for what held and what didn't. Summary: the user chose "fix everything item-by-item again" a third time; all ~12 Critical and ~12 Important Round 3 findings were patched into v1.3 in the same session, self-verified, not independently re-checked until this Round 4.

**Round 4 fixes applied (31 Aug 2026) — as a structural repair session, not item-by-item patching.** The user approved executing the structural recommendation above and then re-running the design review ("repair and then re-run"). All five recommended moves were executed at the size of the rule:

1. **One canonical keying convention (recommendation 1, closes BC-1's pattern if it holds):** judge spec v1.4 states the rule — every metric block carries explicit string-key fields for exactly the axes its defining ADR says the fact varies over; suffix-encoding (`_in_region`, `_out`) is banned — and verifies it once via a block×axis table in judge.md §5 covering all 8 schema blocks. `error_vs_horizon` gained `per_region`; `calibration`/`sharpness` gained region keys; reporting v1.4's charts read the new shapes (Chart 2 per-region panels, Chart 3 dynamic region lines + sharpness header strip).
2. **The layering ADR (recommendation 2):** models spec v1.4 defines `WorldContext` (frozen dataclass in `wmj.models.base`, constructed only by the harness) and one uniform factory signature `factory(ctx, rng)` for all seven models. `fx-brittle` reads its region box from `ctx` — no world import, no special-casing, per-world instances. This dissolves the mutual-exclusion finding (the Round 4 NEW CRITICAL) rather than trading it for a new coupling.
3. **Allowlist over enumerate-shapes (recommendation 3):** cross-cutting v1.4 restructures the gate — import allowlist `{numpy, math, dataclasses, typing}` (TC-NF6-01), banned execution-primitive identifiers anywhere in the AST (TC-NF6-02), `numpy.random` banned wholesale (TC-NF6-03), and the evasion-fixture corpus as the executable completeness contract (TC-NF6-04) with a clean-pass guard (TC-NF6-06); TC-NF6-05 is a superseded tombstone. TC-MU9-01 is mechanism v3: tmp-dir stub via monkeypatched `__path__` (exception-safe, production directory never written), production-path assertion through `wmj.harness.trials`, and an import-allowlist check replacing the identifier grep.
4. **Full implementation-guide reconciliation (recommendation 4, BC-3's trigger):** all 10 orphaned test cases wired; stale Wiring Matrix row replaced; critical path corrected to three tied branches; "eight registered contestants" → seven (the roster was never eight in any spec).
5. **Phantom dependencies removed (recommendation 5):** the undeclared CI layer and the GitHub-API mitigation are deleted from the design, not built; NF-4 is two local enforcement layers with the history scan via `git cat-file --batch-all-objects` and a loud `core.hooksPath` verification at `wmj run`.

Self-verified within the session (greps, from-scratch recounts, and a repo-wide stale-pattern sweep), **not yet independently re-checked — that is exactly Round 5's job**, with the calibrated panels instructed to pressure-test these five repair groups and the blind panels scanning fresh, per BC-2.

---

## Dual Review Metadata

**Round 4, second dual/blind round.** 7 defect-class panels × 2 (calibrated + blind) = 14 independent reviewers, zero shared context between blind panels and each other or the calibrated set.

| Category | Count | Examples |
|---|---|---|
| Confirmed by both calibrated and blind independently (strongest signal) | 9 findings | `fx-brittle` contradiction (6-way across 4 panels), TC-MU9-01 production-directory pollution, stale Wiring Matrix row (5-way), critical-path 3rd branch, `fx-brittle` action-axis gap, AST gate `exec`/`eval`/`numpy.random` evasions |
| New, calibrated-only | 5 findings | `all_models()` return-contract gap, `error_vs_horizon` missing task dimension, JU-8 N=200 multiplication ambiguity, JU-7 pendulum `natural_units` undefined, GitHub API 90-day decay window |
| New, blind-only | 4 findings | baseline/unrigged-model name-pinning gap, duplicate-registration check missing, Chart 2 caption sentence-count violation, ADR-002 rule 3's cross-package overclaim |
| Regressions (confirmed by dual review specifically) | 3 findings | `fx-brittle` (2nd regression in 2 rounds), TC-MU9-01 mechanism, `region_labels` unification claim |
| Disproven (blind finding checked and found factually wrong) | 1 finding | F1-blind's registry-discovery-order nondeterminism claim, refuted by F1-calibrated reading actual CPython `pkgutil` source |
| Noise (discarded on reconciliation) | 0 |

**Honesty check:** this run again did not confirm-and-drop-nothing. It surfaced a larger Critical-finding count than Round 3 despite a narrower, more targeted scope (explicit instructions to prioritise 2 named mechanisms), and it produced one clean disproof of a plausible-but-wrong finding via direct evidence rather than argument — the strongest available demonstration that panels are evaluating claims, not just generating them.

---

*Developed using the Grounded Vibe Methodology*
