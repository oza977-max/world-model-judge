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
| 5 | 2026-08-31 | design (dual/blind, post-repair) | — (see per-panel) | 6 | 4 | 4 | 4 | 2 | 5 | 3 |
| 6 | 2026-08-31 | design (dual/blind, post-v1.5-fix-all) | — (see per-panel) | 8 | 6 | 5 | 6 | 3 | 5 | 5 |

**Round 6 was the first round whose failure shape inverted, and the closest to buildable.** 14 reviewers; calibrated panels pressure-tested the v1.5 enforcement fixes with running code. Verdict is **Do not build** a sixth time (Security 3/10, multiple Criticals), but the design/contract/coverage surface **converged and verified clean** (A 9/8, B 6/8, C 5/7, D 6/6 — panels that scored 2–4 for five rounds scored 6–9) and **three v1.5 design decisions were confirmed working by execution**: content-addressed seeding's no-shift property (3 panels), the TC-MU9-01 registry teardown (2 panels reproduced both Round-5 leaks as regressions), and the import-join (2 panels, 16 ast.parse cases). The critical path (10 chunks) and chunk count (28) are correct on three independent re-derivations. The remaining Criticals are concentrated in two mechanism-precision clusters with known small fixes — the runtime purity harness and the determinism derivations — and **v1.5's own "fix all" pass planted 5 of the 10 Criticals** (self-verified, at speed: the BC-2 pattern). See `design-review/design-review-006.html`.

No overall single number is reported for design review, per `review-reference.md`. Dual/blind (shared rule 16) has now run four times

**Round 5 was the first round to review the structural repair, and the first whose findings changed *kind* rather than just recurring.** 14 reviewers (7 panels × calibrated+blind); calibrated panels adversarially pressure-tested the five repair groups, blind panels scanned fresh. The verdict is **Do not build** for the fifth time, but the composition is different: the repair's own targets verified clean (critical path and chunk count independently re-derived correct for the first time in 5 rounds; BC-3's 79/78-ID and ten-orphan reconciliation held; the Round-4 fx-brittle↔registry mutual exclusion is genuinely dissolved by WorldContext; pkgutil order, PNG scoping, JU-1 blindness, and 8th-model modifiability all verified clean, one of them refuting its own calibration steer). What Round 5's *executable* review then reached is a deeper layer four prose rounds never touched: three independent panels (E-blind, E-calibrated, F2-blind) executed live evasions of the judge-purity AST gate via Python reflection (`__globals__`, `__builtins__`, `__subclasses__`), and both F1 panels executed a demonstration that the determinism claim has no pinned seed-derivation. Security fell to its lowest score of any round (2) because the gate the project calls load-bearing is defeated three ways with the named runtime backstop structurally blind to two of them; Requirements Coverage rose (4→6) because BC-3's reconciliation genuinely held. See `design-review/design-review-005.html`.

No overall single number is reported for design review, per `review-reference.md`. Dual/blind (shared rule 16) has now run three times — Round 4 again dispatched 7 panels × 2 (calibrated + blind), 14 independent reviewers, prioritising adversarial pressure-testing of Round 3's two flagship fixes (`fx-brittle`'s redesign; the registry auto-discovery/TC-MU9-01 mechanism) per this file's own recommendation after Round 3. **Both failed pressure-testing, in ways more serious than what they replaced** — see Resolved Findings below. Panel F1's score rose despite the general trend because the dual-review mechanism itself produced its cleanest demonstration yet of working correctly: F1-calibrated read the actual CPython `pkgutil` source and used it to disprove a plausible-sounding finding from F1-blind, rather than letting it stand. Panel F2 fell to its lowest score of any round: the replacement TC-MU9-01 mechanism introduced a new failure mode (production-directory pollution on test failure) worse in kind than the tautology it replaced.

---

## Dimension Benchmarks

| Dimension | R1 | R2 | R3 | R4 | R5 | R6 | Trend |
|---|---|---|---|---|---|---|---|
| Requirements Coverage (Panel A) | 8/10 | 7/10 | 6/10 | 4/10 | 6/10 | 8/10 | ↑↑ — presence-wiring, counts, and block×axis schema all independently verified clean; only same-session self-citation nits remain (D-blind's deeper *satisfiability* check found 3 misplaced older cases) |
| Interface Contracts (Panel B) | 4/10 | 5/10 | 4/10 | 3/10 | 4/10 | 6/10 | ↑ — ~20 boundaries field-exact; but the h_task pin reworded skill's step (contradicts one-step h=1, MU-5 margin) and limitations/eight-groups miscount |
| Structural Soundness (Panel C) | 5/10 | 4/10 | 4/10 | 3/10 | 4/10 | 5/10 | ↑ — scale + training channel fixed for the 4 non-fixture models, but the fixture-seed channel is unbuildable, the Verdict example self-contradicts (dp-control in an lv verdict), and the orchestration loop is still unspecified |
| Implementability (Panel D) | 4/10 | 5/10 | 6/10 | 5/10 | 4/10 | 6/10 | ↑ — critical path (10 chunks) and count (28) correct on THREE independent re-derivations, all 9 new/re-placed wirings satisfiable; 3 older cases misplaced (D-blind), a fresh "nine chunks" miscount |
| Security (Panel E) | — | 6/10 | 5/10 | 4/10 | 2/10 | 3/10 | → — the v1.5 runtime purity harness (the new load-bearing control) is executed-broken by 4 panels (`sys.modules` swap misses numpy's pre-bound `os`); NF-4 scan command inert (`--batch-check` emits no content). Both fixable: mutate-in-place; `--batch` |
| Reproducibility (Panel F1) | — | 5/10 | 4/10 | 5/10 | 5/10 | 5/10 | → — content-addressed no-shift VERIFIED WORKING (3 panels executed), single-thread honest; but component_key's hash is "e.g." not pinned, the seed key has no purpose discriminator (train/eval/benchmark collide), model_ref shift undisclosed, platform string unnamed |
| Separability (Panel F2) | — | 7/10 | 3/10 | 2/10 | 3/10 | 5/10 | ↑↑ — the TC-MU9-01 teardown, the import-join, and judge blindness all VERIFIED WORKING by execution (both R5 leaks reproduced as regressions); but TC-MU9-03 (new) is unbuildable/lookalike and the baseline classification contradicts its own one-file claim |

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

**Round 5 result (31 Aug 2026) — the repair partly held, and the review reached deeper.** Of the five repair groups: (4) implementation-guide reconciliation held at the count/trace level but 3 of the 10 re-wirings are defectively placed (BC-3 recurrence); the arithmetic (critical path, chunk count) is correct for the first time in five rounds. (2) The WorldContext decision genuinely dissolves the Round-4 mutual exclusion — but two brand-new Criticals sit on the same interface: the sanctioned channel omits the scale vector `direct`/`ensemble` need, and there is no training/fit entry point in the pinned interface at all (6 of 7 models unbuildable as written). (3) The allowlist gate closed the "unlisted module" evasion class, but three panels executed live reflection-based escapes (`__globals__`/`__builtins__`/`__subclasses__`) it cannot see, and TC-JU12-01 (the named backstop) is structurally blind to two of them — Security's lowest score of any round. (1) The canonical keying table matches its JSON in every row but calibration/sharpness, where BC-1 recurs. (5) The phantom CI/GitHub-API deletions held, but new NF-4 gaps surfaced (commit-message traversal, web-UI bypass, no pre-push hook). Plus two determinism Criticals only executable review could find: no pinned seed-derivation (F1 both, executed), and TC-MU9-01 v3's in-process registry corruption (F2 both, executed). **Full record: `design-review/design-review-005.html`.**

**Round 5 fixes applied (31 Aug 2026 — the user chose "fix all"; specs → v1.5, worlds v1.4, test-cases v1.2).** The three design-level items were *decided*, not patched around:
- **Judge purity (C1/C2):** the static AST gate is demoted to fast-feedback *lint* (completeness no longer claimed); the **runtime purity harness TC-JU12-01 is now the load-bearing control**, specified to guard *effects* — `os`/`open`/clock/network/subprocess/`numpy.random` shared objects raise on access, so a reflection escape trips at the effect however it was reached — with a phantom-gate (TC-JU12-02) proving it catches the exact Round-5 `__globals__`-routed `os` read. Threat model stated: the judge author is not the adversary; the property is accidental impurity, not deliberate smuggling. (cross-cutting ADR-002/003, judge §4.)
- **Model interface (C5/C6):** `WorldContext` gains `scale`; the uniform factory gains a `TrainingData` argument — every model is now buildable through the one sanctioned channel without breaking the uniform-signature repair. (models ADR-M1/M3.)
- **Determinism (C3):** seed derivation is content-addressed (keyed on stable name strings, not iteration position); TC-NF1-03 asserts adding a model shifts no other's stream. (cross-cutting ADR-002 rule 2.)

The rest were fixed at rule-size: models→worlds import gate (TC-NF6-07); TC-MU9-01 teardown clears `sys.modules` + force-registers before snapshot, part (c) joins module+name; the model card (`limitations`+`not_tested`) now renders in `results.html` (reporting ADR-R5); calibration/sharpness pinned to `h_task` (closing the BC-1 recurrence); ADR-J6's per-region multiplier corrected; N=200 pinned shared-across-models (paired); Chart 4 shows the skill counterpart, Chart 2 gets `dt`; single-thread assertion corrected; NF-4 commit-message scan and web-UI/no-pre-push disclosure fixed; three `<spec-value>` placeholders filled; TC-WD3-01/JU4-01/MU3-03 corrected; the implementation-guide re-wirings re-placed and the **critical path honestly re-derived to 10 chunks** (the new file-ownership/prereg-count edges lengthen the reporting branch — a real graph change, disclosed as such, not a recount error). The seven stale `.html` twins were stamped with a deterministic staleness banner pointing to the authoritative `.md` (regenerating them by hand-render was judged higher-risk than the drift it would fix; the build's doc-generation step, P6-C02, is where they regenerate with the parity hash).

**Self-verified only (greps, version sweep, div-balance and confidentiality checks, from-scratch critical-path re-derivation) — not yet independently re-checked.** BC-2 says these enforcement-mechanism changes (the runtime purity harness especially) need adversarial pressure-testing; a Round 6 would be the place, if the user wants one.

**Build Check update (Round 5):** BC-1 recurred (C9, calibration/sharpness horizon-step). BC-2 recurred (C7, TC-MU9-01 v3 fixed the disk leak and introduced an in-process one). BC-3 recurred partially (C13, 3 of 10 wirings mis-placed) — though its count/trace half held. **New standing recommendation, BC-4 (candidate):** the judge-purity property cannot be secured by a static AST gate in Python — no finite identifier list bounds reflection (`__globals__`, `__builtins__`, `__subclasses__`, `__code__`, `__closure__`, `__dict__` walks). Any future round that touches the gate must treat this as an architecture decision (restrict the judge to a non-reflective style enforced another way; OR specify TC-JU12-01 to intercept attribute/dict access, not just imports/builtins; OR downgrade the purity claim from "structural" to "empirical + disclosed" to match what the gate delivers), not a patch. The same holds for two other design-level (not finding-level) gaps: no model training/provisioning channel (C6), and no pinned seed-derivation (C3).

**Round 6 result (31 Aug 2026 — dual/blind, ran the pressure-test BC-2 called for).** The verdict of the Round-5 fixes, checked with running code this time:
- **VERIFIED WORKING:** content-addressed seeding's no-shift property (3 panels executed), the TC-MU9-01 teardown (2 panels reproduced both R5 leaks as regressions), the module+name import-join (2 panels, 16 ast.parse cases), judge blindness under the new WorldContext/TrainingData channels, the model interface for the 4 non-fixture models, the critical path (10 chunks) and count (28) on 3 independent re-derivations, ~20 interface boundaries field-exact. **The design surface converged** (A/B/C/D at 5–9, up from 2–4).
- **BROKEN AS WRITTEN (5 of R6's 10 Criticals are v1.5's own fixes — the BC-2 pattern, self-verification failing again):** the runtime purity harness (BC-4's item 2 was written but the *mechanism* is wrong — `sys.modules` swap doesn't reach numpy's pre-bound `os`; needs mutate-in-place; **4 panels, 3 executed**); the NF-4 scan command (`--batch-check` emits no content, needs `--batch`, executed); the `h_task` pin reworded skill's step against the one-step (h=1) MU-5 margin; TC-MU9-03 (the new CLI test) unbuildable; the baseline-classification contradiction. Plus new precision gaps: `component_key` hash unpinned (executed), seed purpose-key collision, `model_ref` shift undisclosed (same class as C3, one layer up), ensemble K=5 split, platform-string source.

**BC-2 stands, sharper than ever:** every round where fixes were applied fast and self-verified, the next round found the fix itself broken. Round 6's own message: the fixes are now *mechanism-precision* corrections (mutate-in-place, `--batch`, pin the hash, add a purpose key) on a design that has otherwise converged — **but they MUST be verified by execution, not prose**, or Round 7 finds them broken too. This is the first round whose gap to "Build with caveats" is short and mostly mechanical.

**Round 6 fixes applied (31 Aug 2026 — the user chose "fix all with execution-verification this time"; specs → v1.6, test-cases v1.3).** The BC-2 discipline was finally applied: **every enforcement-mechanism fix was written as a python3 script and RUN before the spec was written**, and the executed result is cited in the spec text.
- **Purity harness (C1):** corrected to *mutate the shared `os`/`builtins`/clock objects in place* (`setattr(os,'system',raise)`, raising proxy on `os.environ`), never a `sys.modules` swap. Executed: the swap left the reflection escape running a real `os.getcwd()`; mutate-in-place tripped the guard. Also guards `numpy.datetime64` (executed: it read the wall clock while `sys.modules['time']` was guarded).
- **NF-4 scan (C2):** `--batch` not `--batch-check`. Executed: `--batch-check` found 0 matches for a planted term, `--batch` found it. TC-NF4-01 now carries a plant-and-detect can-fail proof.
- **Seeding (C3/C4/C7/ensemble):** `component_key` pinned exactly (no "e.g."; blake2b/8-byte/one-int) and moved with `SeedSource` to `wmj.models.base`. Executed: two-impl convergence True; fixture rebuilds `direct` bit-identically True; ensemble K=5 distinct+reproducible True; train/eval/benchmark `purpose` keys distinct True; no-shift True.
- **Prose/wiring:** skill pinned to one-step `h=1` (C6); Verdict example dp-control→lv-planning (C5); `limitations` the ninth mandatory group; `interval_levels` removed from JudgeInput (one producer); `meta.platform` pinned to the coarse composite; `is_baseline` flag (C9); `check_prereg` per-model (TC-MU6-03); TC-WD3-01/TC-NF3-01/TC-MU2-01 re-placed (C10); `wmj list-models` makes TC-MU9-03 buildable (C8); training `N_train=2000` pinned; `model_ref` shift disclosed; "nine chunks"→twelve.

Four new test cases (TC-JU12-03, TC-NF1-04, TC-MU6-03, TC-MU7-02); 89 IDs, 88 live. **Round 7 is dispatched to independently re-verify these (dual/blind, executable) — the BC-2 rule says self-verification is not enough, so this pass's own execution proofs get an independent adversarial re-run.**

**Round 7 result (31 Aug 2026 — dual/blind, the independent execution re-run BC-2 demanded).** Scores (cal/blind): Requirements 4/7, **Interface 9/8** (highest any dimension has scored — the surface has converged), Structural 4/6, Implementability 5/6, **Security 3/3** (the floor, both panels, convergent), Reproducibility 6/7, Separability 5/7. The round splits cleanly into three piles:

- **Converged and independently verified (execution, not prose):** the seeding mechanism (`component_key` two-impl convergence, `purpose`-key stream separation, ensemble K=5, fixture rebuild of `direct` bit-for-bit), the NF-4 `--batch` scan, the *specific* Round-6 purity escape now closed by mutate-in-place (guard fires), judge blindness under every v1.6 data shape, chunk count (28)/critical path (10)/MVP-1 sequencing, ADR-J6 arithmetic, and ~20 interface boundaries field-exact. The execution-verification discipline held for the mechanisms that were actually run.
- **v1.6 change notes overstated the edits (BC-2/BC-3 recurred):** judge §4's purity *body* still describes the disproven `sys.modules` swap (only the changelog claimed the mutate-in-place fix); TC-JU12-03 and TC-MU7-02 are change-note-claimed-wired but in no chunk's `[Test:]` tag; architecture-overview.md and reporting.md are stamped v1.6 with no v1.6 changelog row; the model_ref-shift disclosure promised in Round 6's own remediation plan was never written. All mechanical, ~1 hour to finish.
- **The design fork — BC-4 CONFIRMED, no longer a candidate:** both Security panels executed *new* accidental-impurity routes that escape the v1.6 mutate-in-place harness — `os.environ` pre-capture rebind (the identical rebind-vs-mutate bug class v1.6 fixed, one layer down); `np.random.normal()`/`uniform()` (the threat model's own textbook example, unenumerated legacy globals routing through `mtrand._rand`); reflection to the unenumerated `os` attribute surface; `ctypes` already resident via numpy reaching raw syscalls with no import to catch; and unseeded `Generator(PCG64())` reaching `getrandom()` with no Python object in the path (`Generator.__init__` is a C immutable type — architecturally uninterceptable by any variant of the mechanism). The runtime harness reproduces the static gate's finite-enumeration disease one layer down. This is now an architecture decision (narrow JU-12's claim to best-effort+empirical+disclosed, OR move purity out-of-process to an OS sandbox), not a patch.

Two standing spec gaps also remain open: the harness orchestration loop + JudgeInput per-region trial partitioning is still unspecified as an algorithm (open since Round 5, C-cal + C-blind), and no gate enforces `models→harness` in the one direction that matters (C-cal + C-blind + F2-cal, the invariant the SeedSource-in-`models.base` redesign depends on). Plus a real MU-6 gap: `check_prereg` verifies commit *order*, not content-invariance — an ordinary honestly-dated second commit tuning `matching_margin` after seeing results defeats pre-registration (E-blind, executed with real git), uncovered by ADR-M5's three named residual risks. **Full record: `design-review/design-review-007.html`.**

**Build Check update (Round 7):** BC-1 did NOT recur (canonical keying clean for the first time). BC-2 recurred (prose-only edits drifted from delivered changes again). BC-3 recurred (two of four new cases unwired). **BC-4 CONFIRMED** — the judge-purity completeness claim cannot be delivered by in-process finite enumeration; the project must decide what the control can truthfully promise before any build touches JU-12. Seventh consecutive "Do not build," but the first where the gap is one mechanical hour plus one honest design decision, not a field of scattered mechanism bugs.

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
