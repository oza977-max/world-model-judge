# Review Calibration — World Model Judge

Project-level calibration layer for GVM review skills (`/gvm-design-review`, `/gvm-code-review`, `/gvm-doc-review`). Per `review-reference.md`'s progressive-calibration protocol: this file is read at the start of every review round and updated at the end of it.

---

## Score History

| Round | Date | Type | Overall | Panel A (Coverage) | Panel B (Contracts) | Panel C (Structural) | Panel D (Implementability) | Panel E (Security) | Panel F1 (Reproducibility) | Panel F2 (Separability) |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 2026-08-25 | design | — (see per-panel) | 8 | 4 | 5 | 4 | n/a | n/a | n/a |
| 2 | 2026-08-30 | design | — (see per-panel) | 7 | 5 | 4 | 5 | 6 | 5 | 7 |
| 3 | 2026-08-30 | design (dual/blind) | — (see per-panel) | 6 | 4 | 4 | 6 | 5 | 4 | 3 |

No overall single number is reported for design review, per `review-reference.md`. **Round 3 is the first dual/blind round (shared rule 16, triggers automatically at 2+ prior rounds)** — each of the 7 panels ran twice: once calibrated (sees this file, checks named Round 2 fixes for regression) and once fully blind (no calibration data, no knowledge of any prior round). 14 independent reviewers total. Panel D's score rose despite the round's overall severity because its two highest-stakes checks — an independent from-scratch re-derivation of the double-pendulum EOM, and an independent recount of the implementation guide's chunk total — were BOTH confirmed correct by two separate reviewers each. Panel F2's score fell sharply because the TC-MU9-01 mechanism, credited in Round 2 as fixing a "no enforcement" finding, was found this round to be structurally incapable of ever failing.

---

## Dimension Benchmarks

| Dimension | R1 | R2 | R3 | Trend |
|---|---|---|---|---|
| Requirements Coverage (Panel A) | 8/10 | 7/10 | 6/10 | ↓ — 2 independent panels found the spec-parity check is an orphan design element; the NF-4 hook-install gap deepens |
| Interface Contracts (Panel B) | 4/10 | 5/10 | 4/10 | ↓ — `calibration`/`sharpness` both found missing their own required task dimension; `region_labels` found inconsistent across 3 documents |
| Structural Soundness (Panel C) | 5/10 | 4/10 | 4/10 | flat — `climatology` region gap and the `fx-brittle` regression offset the confirmed EOM/count fixes |
| Implementability (Panel D) | 4/10 | 5/10 | 6/10 | ↑ — the two highest-stakes checks (EOM re-derivation, chunk recount) both independently confirmed correct twice over |
| Security (Panel E, new R2) | — | 6/10 | 5/10 | ↓ — NF-4 self-defeat genuinely fixed, but the replacement hook has no install path and the scan still misses historical blobs |
| Reproducibility (Panel F1, new R2) | — | 5/10 | 4/10 | ↓ — the two literally-named Round 2 evasions are closed, but `numpy.random.*` legacy calls and an incomplete ambient-module list remain |
| Separability (Panel F2, new R2) | — | 7/10 | 3/10 | ↓↓ — the named TC-MU9-01 mechanism turned out to be tautological (cannot fail by construction); no registry discovery mechanism specified |

**Round 3 closes the honesty gap named after Round 2, and finds it was worth closing.** Genuine dual/blind independent re-verification (not self-review) confirmed the two highest-risk Round 2 fixes hold: the corrected double-pendulum EOM was independently re-derived from Euler–Lagrange from scratch by two separate reviewers and matches exactly; the corrected chunk count (28) was independently recounted by two separate reviewers and matches exactly. Everything else fixed in v1.2 was re-examined and found either solid or genuinely regressed/incomplete — see Resolved Findings below. Fresh strict scanning by all 14 reviewers surfaced roughly 12 new Critical and 12 new Important findings, dominated by one recurring structural theme (schema blocks missing a task/region/step key their own defining ADR requires — now the third round this exact shape has appeared) and one recurring process theme (every new enforcement mechanism Round 2 introduced has a real gap one level below what it explicitly checked for). See `design-review/design-review-003.html` for the full findings, findings-by-convergence table, and the structural recommendation. **Verdict: Do not build**, unchanged from Rounds 1–2, pending the user's triage.

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

---

## Recurring Findings — two shapes promoted to Build Checks this round (see below)

Round 2 flagged two defect *shapes* worth tracking, since GVM's methodology calls out recurring shapes, not just recurring exact findings. Round 3 confirmed both a third time, which is shared rule 21's exact promotion trigger:

1. **Schema/consumer granularity mismatch.** R1: Chart 1 (`trials` block missing). R2: Chart 2 (baseline curves), Chart 3 (`n_trials`). R3: `calibration` and `sharpness` missing their required task key, `climatology` missing its required region key, `region_labels` inconsistent across three documents, Chart 4's region-to-column mapping undeclared, Chart 1's horizon_step dimension unaccounted for — six more instances in one round. **Promoted to Build Check BC-1.**
2. **A fix closes the literal finding and leaves an adjacent, structurally similar gap one level down.** R1→R2: the pendulum EOM formula was missing, then wrong. R2→R3: the NF-4 self-defeating instruction is fixed, but the replacement hook has no install path and the scan still misses file-content history; the AST gate's two named evasions are closed, but alias/`exec`/`numpy.random` evasions remain; TC-MU9-01 gained a named mechanism that turns out to be tautological. **Promoted to Build Check BC-2.**

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

**Round 3's own new findings** (~12 Critical, ~12 Important) are listed in `design-review/design-review-003.html` — not duplicated here.

---

## Build Checks

**BC-1 — Schema/consumer granularity mismatch (promoted this round, shared rule 21: 3rd consecutive round).** Whenever a Verdict schema block represents a fact that varies by task, region, or horizon step, it must carry all three keys explicitly, checked against every ADR that defines how the fact is computed — not just against the chart that most recently needed it. **Trigger for future rounds:** any new schema block, or any edit to an existing one, must be checked against this rule before being marked resolved.

**BC-2 — Self-verified enforcement-mechanism fixes must be adversarially pressure-tested, not just read for internal consistency (promoted this round, shared rule 21: regression after resolution, twice — NF-4 and TC-MU9-01 both).** A fix to a security/purity/isolation mechanism is not "resolved" until someone has tried to independently derive it from first principles, break it, or simulate it against a real (not test-authored) input. Self-consistency review — does the new text agree with itself? — is necessary but has now proven insufficient twice in one round. **The two fixes that DID survive this round (the EOM, the chunk count) are exactly the two the review explicitly assigned an independent from-scratch check rather than a re-read — this is the model to repeat.**

**Round 3 fixes applied (30 Aug 2026):** the user chose "fix everything item-by-item again" a third time. All ~12 Critical and ~12 Important Round 3 findings were patched directly into the specs (now v1.3 — worlds.md stays at v1.2, assigned no fix) in the same session, by the same agent that synthesised them — **not independently re-verified**, the identical honesty gap this file has now named after every round. Fixed: `calibration`/`sharpness` restructured to per-task arrays; `climatology.per_task` gained a region field; `region_labels` unified to one canonical shape; the `is_exception`/`observed` reconciliation gained a build-enforced property test; `fx-brittle` redesigned as a genuine post-hoc wrapper keyed to the world's own region box; the NF-4 scan extended to historical blobs with a named pre-commit-hook install path; the AST gate widened to 5 individually-tested checks (banning `importlib`/`__import__`/`exec`/`eval` outright, catching `numpy.random.*`, wider ambient-module list); `wmj.models.registry`'s auto-discovery mechanism named explicitly; TC-MU9-01 redesigned as a structural test instead of a tautological git-diff; the spec-parity check traced to NF-5 with a test and build chunk; Chart 1/2/3/4 fixes (horizon_step panelling, step-1 start, per-task calibration, explicit region-column mapping); the critical-path arithmetic corrected; a third prereg residual risk disclosed. **9 new test cases added (70→79)**, independently recounted via `grep -c`. **Recommended for whoever runs Round 4, per the two Build Checks above:** pressure-test the `fx-brittle` redesign and the registry auto-discovery/TC-MU9-01 mechanisms first — these are this round's equivalent of "a physics formula and a security mechanism," the categories where a confident self-fix has twice now proven least trustworthy.

---

## Dual Review Metadata

**Round 3 is the first dual/blind round (shared rule 16).** 7 defect-class panels × 2 (calibrated + blind) = 14 independent reviewers, zero shared context between blind panels and each other or the calibrated set.

| Category | Count | Examples |
|---|---|---|
| Confirmed by both calibrated and blind independently (strongest signal) | 8 findings | EOM correctness, chunk count, `fx-brittle` contradiction (3-way), NF-4 hook-install gap (3-way), AST dynamic-import evasion (3-way), TC-MU9-01 tautology, `climatology` region gap, Chart 2 log(0) |
| New, calibrated-only | 6 findings | `calibration`/`sharpness` task-dimension gaps, Chart 4 region-mapping rule, `is_exception` enforcement gap, AST test-ID traceability, `chart-preview` CLI |
| New, blind-only | 4 findings | Chart 3 "no metric" self-contradiction, Chart 1 horizon_step dimension, prereg timestamp forgery, worktree isolation |
| Regressions (confirmed by dual review specifically) | 2 findings | TC-MU9-01, `fx-brittle` |
| Noise (blind finding discarded on reconciliation) | 0 |
| Rediscovered (already known, not new) | 0 — all Round 2 items tracked above were explicitly assigned to a calibrated panel for re-verification, not rediscovered incidentally |

**Honesty check:** this run did not confirm-and-drop-nothing (a suspect all-green pattern) — it surfaced substantial new findings across nearly every panel, and the two panels given an explicit re-derivation task (D-calibrated, D-blind) independently confirmed a prior fix rather than manufacturing a new finding to justify their dispatch. Both directions of result occurred, which is itself evidence the panels were not just pattern-matching toward "find something."

---

*Developed using the Grounded Vibe Methodology*
