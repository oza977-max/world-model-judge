# Review Calibration — World Model Judge

Project-level calibration layer for GVM review skills (`/gvm-design-review`, `/gvm-code-review`, `/gvm-doc-review`). Per `review-reference.md`'s progressive-calibration protocol: this file is read at the start of every review round and updated at the end of it.

---

## Score History

| Round | Date | Type | Overall | Panel A (Coverage) | Panel B (Contracts) | Panel C (Structural) | Panel D (Implementability) |
|---|---|---|---|---|---|---|---|
| 1 | 2026-08-25 | design | — (see per-panel) | 8 | 4 | 5 | 4 |

No overall single number is reported for design review, per `review-reference.md` — the four defect-class scores are independent dimensions (Marzano: score the performance, not the performer; don't collapse independent dimensions into one number).

---

## Dimension Benchmarks

| Dimension | Baseline (R1) | Current | Trend |
|---|---|---|---|
| Requirements Coverage (Panel A) | 8/10 | 8/10 | new |
| Interface Contracts (Panel B) | 4/10 | 4/10 (fixes applied same round, not yet re-verified) | new |
| Structural Soundness (Panel C) | 5/10 | 5/10 (fixes applied same round, not yet re-verified) | new |
| Implementability (Panel D) | 4/10 | 4/10 (fixes applied same round, not yet re-verified) | new |

**Note on "fixes applied same round, not yet re-verified":** the user chose "fix everything before build" immediately after the panels reported. All 10 Critical and 16 Important findings were patched directly into the seven specs (now v1.1) in the same session, by the same agent that synthesised the findings — not by an independent panel. The scores above are the *pre-fix* R1 scores; they have not been re-measured against the v1.1 specs. This is a real gap in the calibration record, named rather than hidden: a genuine Round 2 (even a lightweight one) against v1.1 would be the correct way to confirm the fixes actually closed what they claim to close, rather than trusting the same author's self-assessment. Recorded here so the next review round knows to prioritise re-verification over fresh discovery.

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
- **Best observed this round:** the MVP-1/Tracer-Bullet Assessment — a structured three-part check (is the first slice real? is the sequence incremental? evidence for both) that produced a specific, actionable finding (chunk 19 of 23 before further user-visible value) rather than a vague "could be more incremental."
- **Worst observed this round:** the double-pendulum equations of motion were specified only as "the textbook equations of motion" — for a system with multiple non-equivalent sign conventions, in a document whose own TC-WD1-01 needs a single unambiguous 15-significant-digit reference value. A concrete, verifiable "cannot build without guessing" gap.

---

## Recurring Findings

None yet — this is Round 1. A finding recurs when it appears in 2+ consecutive rounds against comparable scope; nothing to compare against until Round 2 runs (see the re-verification note above).

---

## Resolved Findings

All 10 Critical and 16 Important findings from Round 1 were addressed by direct spec edits in the same session (specs bumped to v1.1). Listed here for the record; **status is "fixed by the same author who found them, not independently re-verified"** — see the caveat under Dimension Benchmarks:

| # | Finding | Spec(s) touched | Fix |
|---|---|---|---|
| C1 | `is_fixture` conflicting producers | judge.md, models.md, reporting.md, cross-cutting.md | Removed from judge-produced Verdict; harness-owned `JudgedResult` envelope introduced as the single producer |
| C2 | Verdict `meta` requires ambient access JU-12 forbids | judge.md, cross-cutting.md | `meta` moved to the envelope; ADR-002 rule 3 clarified |
| C3 | Chart 1 needs undeclared per-trial data + 3 undefined drawing rules | judge.md, reporting.md | New `trials` Verdict block; aggregation/region-scope/ordering rules pinned |
| C4 | Climatology underspecified on 3 axes | judge.md, worlds.md | Producer contract added; out-of-range and no-switch behaviour defined; bin method + reference-run length pinned |
| C5 | WD-3 gate names a nonexistent component | worlds.md | Restated as two real, checkable properties |
| C6 | Double-pendulum EOM not written out | worlds.md | Explicit θ̈₁, θ̈₂, energy formulas with named sign convention |
| C7 | z_p not derivable from erf | judge.md | Four constants hardcoded and tabled |
| C8 | `fx-honest-rough` spread formula missing | models.md | Exact closed-form formula given |
| C9 | Implementation guide's own build sequence violates MVP-1 | implementation-guide.md | New chunk P2-C05 (second thin vertical slice) added at position 8 |
| C10 | 4 unpinned training constants threaten MU-8's determinism guarantee | models.md | Init scheme, NLL formula, batching, Adam ε all pinned |
| Important #1–16 | (see design-review-001.html for full text) | judge.md, worlds.md, models.md, reporting.md, cross-cutting.md, architecture-overview.md | All patched; TC-MU1-03 added to test-cases.md for the one finding that needed a new test, not just a spec clarification |

---

## Build Checks

None promoted yet — promotion requires a finding to recur in 3+ consecutive rounds or regress after resolution (shared rule 21). With only one round on record, nothing qualifies. **Recommended for the next round's attention regardless:** since Round 1's fixes are unverified self-assessment, the highest-value thing a Round 2 (even a quick scan) could check is whether the `is_fixture`/envelope split, the `trials` block, and the climatology producer contract are actually consistent across all seven v1.1 specs now that a second, independent pair of eyes looks at them.

---

## Dual Review Metadata

Not applicable — dual review activates at round 3+ per shared rule 16. This is round 1.

---

*Developed using the Grounded Vibe Methodology*
