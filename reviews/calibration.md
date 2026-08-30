# Review Calibration — World Model Judge

Project-level calibration layer for GVM review skills (`/gvm-design-review`, `/gvm-code-review`, `/gvm-doc-review`). Per `review-reference.md`'s progressive-calibration protocol: this file is read at the start of every review round and updated at the end of it.

---

## Score History

| Round | Date | Type | Overall | Panel A (Coverage) | Panel B (Contracts) | Panel C (Structural) | Panel D (Implementability) | Panel E (Security) | Panel F1 (Reproducibility) | Panel F2 (Separability) |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 2026-08-25 | design | — (see per-panel) | 8 | 4 | 5 | 4 | n/a | n/a | n/a |
| 2 | 2026-08-30 | design | — (see per-panel) | 7 | 5 | 4 | 5 | 6 | 5 | 7 |

No overall single number is reported for design review, per `review-reference.md` — the defect-class scores are independent dimensions (Marzano: score the performance, not the performer; don't collapse independent dimensions into one number). Panels E, F1, F2 are new in Round 2 (Security panel, and two ATAM-utility-tree-derived Quality-Attribute sub-panels — reproducibility and separability were the tree's only (H,H) leaves this project produced).

---

## Dimension Benchmarks

| Dimension | Baseline (R1) | Current (R2) | Trend |
|---|---|---|---|
| Requirements Coverage (Panel A) | 8/10 | 7/10 | ↓ — new Critical (NF-4 self-defeating scan mechanism) outweighs two confirmed-resolved items |
| Interface Contracts (Panel B) | 4/10 | 5/10 | ↑ — 7 of 9 R1 fixes hold; but 2 new chart-data gaps found on the remaining required charts |
| Structural Soundness (Panel C) | 5/10 | 4/10 | ↓ — all 6 R1 fixes hold, but 2 new Criticals surfaced on core mechanisms (trust_horizon region, JU-9 mandatory-field list) |
| Implementability (Panel D) | 4/10 | 5/10 | ↑ — most fixes hold; 1 new Critical (double-pendulum EOM coefficient bug), 2 new Important (bookkeeping) |
| Security (Panel E, new) | — | 6/10 | new — zero Critical (genuinely low-stakes threat model), 4 real Important gaps |
| Reproducibility (Panel F1, new) | — | 5/10 | new — 1 Critical (AST gate misses ambient reads), 1 Important |
| Separability (Panel F2, new) | — | 7/10 | new — registry/envelope split holds cleanly; 1 real test-enforcement gap |

**Round 2 closes the honesty gap named after Round 1.** All 26 Round-1 findings were independently re-checked against v1.1 (not by the same agent that fixed them, and under this round's strict criteria rather than R1's liberal criteria). Result: 16 of 26 hold cleanly (CONFIRMED-RESOLVED). 3 were only partially resolved or introduced a new defect in the same location (Chart 1's data contract, the double-pendulum EOM, the implementation guide's chunk bookkeeping). Independently of the R1 re-check, the two new panel types plus a strict fresh pass by the four original panels surfaced 8 new Critical and 8 new Important findings — a defect population comparable in size to Round 1's, despite the same seven specs having already been through one full fix pass. See `design-review/design-review-002.html` for the full findings. **Verdict: Do not build**, unchanged from Round 1, pending the user's triage of these findings.

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
- **Worst observed:** the only described judge-purity gate (an AST walk scoped to `wmj.*` imports) has no coverage at all for direct ambient reads (`os`, `time`, `sys`) — a class of violation that can pass a same-machine repeated-run test while still breaking NF-1's explicit cross-machine scope.

---

## Recurring Findings

None yet in the strict sense (a finding recurring unresolved across 2+ rounds against the *same* text). But Round 2 surfaced three cases worth tracking as a *pattern*, since GVM's own methodology reference calls out watching for recurring defect *shapes*, not just recurring exact findings: R1 found a schema/chart-data gap (Chart 1); R2 found the same shape twice more (Chart 2, Chart 3) plus a partial regression on the original. R1 found a missing-formula gap (pendulum EOM); R2 found the replacement formula is wrong. R1 found a sequencing gap (chunk order); R2 found the fix's own bookkeeping (chunk count, file-ownership claim) is wrong. If a Round 3 finds this shape a fourth time, promote it to a Build Check per shared rule 21.

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

**Round 2's own new findings** (8 Critical, 9 Important, several Minor/Observation) are listed in `design-review/design-review-002.html` — not duplicated here; this table tracks Round 1's items specifically, per this file's role as a cross-round comparison record.

---

## Build Checks

None promoted yet — promotion requires a finding to recur in 3+ consecutive rounds or regress after resolution (shared rule 21). Two Round 1 items (C6 double-pendulum EOM, C9 implementation-guide bookkeeping) **did** regress after their first "resolution" — per shared rule 21 this is exactly the condition that starts the clock toward promotion. **Flagged for Round 3's attention:** if either C6 or C9's underlying defect shape (a fix introducing a fresh, differently-wrong precision) recurs a third time, promote to a Build Check requiring mandatory independent numeric re-derivation before any fix in that category is accepted as closed.

---

## Dual Review Metadata

Not applicable — dual review activates at round 3+ per shared rule 16. This is round 2.

---

*Developed using the Grounded Vibe Methodology*
