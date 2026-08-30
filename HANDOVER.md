# Handover

## CURRENT STATE

**Phase:** Technical specification complete and **twice design-reviewed**,
all seven documents at **v1.2**. `/gvm-tech-spec` ran in full; Round 1 of
`/gvm-design-review` (quick-scan, 4 panels) found 10 Critical + 16
Important findings, all patched into v1.1. **Round 2** then ran as a
genuinely independent re-check (full-depth, strict criteria, the newly
imported `gvm-graph`-orchestrated skill with two brand-new panel types —
Security, and two ATAM-utility-tree Quality-Attribute sub-panels for
reproducibility and separability) specifically to close the honesty gap
Round 1 left open: were the v1.1 fixes real, or self-graded homework?
**Result: mostly real (16 of 26 Round 1 findings held cleanly), but not
entirely** — 3 Round-1 "fixes" were only partial or had regressed (most
seriously, the double-pendulum equations of motion, rewritten to fix a
Round 1 finding, contained a wrong coefficient that was self-consistently
baked into its own reference test values). Round 2 also surfaced 8 new
Critical and 9 new Important findings the two new panel types and the
stricter re-scan caught fresh — a defect population comparable in size to
Round 1's, despite a full prior fix pass. The user again chose "fix
everything before build"; all of it is patched into v1.2 (see
`design-review/design-review-002.html` and `reviews/calibration.md`).
**The same honesty gap is open a second time, and is explicitly named as
such in `reviews/calibration.md`:** the Round 2 fixes were applied by the
same agent that found them, in the same session, not independently
re-verified. Two of them — the double-pendulum EOM correction and the
NF-4 confidentiality-scan redesign — are exactly the kind of fix (a
physics formula; a security mechanism) where a confident self-fix is
least trustworthy, and are flagged in calibration.md as the highest-value
thing a Round 3 could check first.
**Code written:** none. Deliberately. Still true — specs only.
**Blocked on:** nothing. Next step is a genuine choice between: (a) a
Round 3 design review — likely with real dual/blind review this time,
since shared rule 16 activates it at round 3+ — prioritising re-verification
of the two fixes named above; or (b) accepting the residual self-verification
risk and proceeding to `/gvm-build` starting at Phase 1 / P1-C01, followed by
the P1-C02 → P2-C05 sequence (28 chunks total as of v1.2's implementation-guide
correction — the "24"/"23" counts in earlier versions were themselves wrong,
caught only by Round 2's independent recount).

Last updated 2026-08-30 (design-review-002 fixes).

---

## What exists

| File | What it is |
|---|---|
| `CLAUDE.md` | Project context and standing rules. Read this first. |
| `requirements/requirements.md` | v1.2. 45 requirements (42 Must, 0 Should, 3 Won't) across four domains plus non-functional. Revised after a six-expert GVM review board and two independent audit passes (technical-correctness, LeCun-grounded, novelty/prior-art). Every one carries an "In plain words" restatement. |
| `requirements/requirements.html` | Same content, Tufte-styled. |
| `requirements/wordsareamenu.html` | The source essay, now draft v2.7. Also revised post-review: narrowed novelty claim, corrected two factual errors (flight-simulator framing, SR 11-7 currency), added the employer disclaimer and an AI-assistance transparency note. |
| `risks/risk-assessment.md` | Four product risks, written before requirements. Still references essay draft v2.1 deliberately — the historical record of what it was written against. |
| `test-cases/test-cases.md` | v1.0. 69 cases, one `TC-{REQ-ID}-{NN}` per requirement minimum, more for credibility-fatal ones. Includes 5 negative/phantom-gate cases (proving WD-3, WD-7, NF-1, JU-9, JU-11's automated checks can actually fail, not just always pass) and 5 property-based cases. Full traceability matrix at the end — zero orphan requirements, zero orphan cases, every count independently grep-verified against the document body. |
| `specs/cross-cutting.md` (+.html) | v1.2. Stack (pure NumPy, hand-rolled MLPs — user's explicit choice over PyTorch/stdlib-only), the four-rule determinism strategy, five-package structure with the judge importing nothing, error-handling conventions, the two-package dependency budget (`numpy`, `matplotlib`). The import-graph gate now also blocks ambient reads and dynamic imports; the NF-4 scan is redesigned around a gitignored local terms file (Round 2 fix). |
| `specs/worlds.md` (+.html) | v1.2. LV and double-pendulum constants fully pinned (dt, horizons, scale vectors, regions, actions). Shared fixed-step RK4 with a 1e-6 relative drift bound (ADR-W1). Divergence benchmark: 64 seeded starts, median curve per region (ADR-W3). **Round 2 fixed a coefficient error in the double-pendulum EOM** (v1.1's explicit formula was wrong; the reference constant needs recomputing before TC-WD1-01 can be implemented). |
| `specs/models.md` (+.html) | v1.2. Baselines with honest training-residual spreads. Three one-corruption fixtures. The direct-vs-ensemble unrigged pair (Nix & Weigend / Lakshminarayanan et al., both newly discovered experts) with a pre-registered `sqrt(1+1/K)`-corrected spread mapping and a 0.05 matching margin. Pre-registration enforced by git commit ordering; a second disclosed residual risk (undetectable non-publication of an unfavourable run) added in Round 2. |
| `specs/judge.md` (+.html) | v1.2. CRPS as the scoring rule. Settles Open Questions 1/2: **N=200 independent trials, bands green [12,29] / amber [8,11]∪[30,35] / red beyond**, derived from the exact binomial CDF and verified computationally in-session (not eyeballed). Settles OQ-5's runtime half: 600s budget. Full verdict JSON schema, now 8 mandatory field groups with trust horizons carrying a region field (Round 2 fixes). |
| `specs/reporting.md` (+.html) | v1.2. All four required charts designed down to authored caption templates. Fixture labelling centralised so it survives a screenshot. `wmj run` / `wmj verify` command pair. Chart 1/2/3 data-contract gaps and Chart 4's region breakdown fixed in Round 2. |
| `specs/architecture-overview.md` (+.html, with an inline C4 container SVG) | v1.2. Synthesis of all five specs + a Brooks conceptual-integrity review — found and resolved one real tension (NF-1's byte-identity scope vs PNG rendering; resolved by disclosure, no spec change needed). Round 2 found no new conceptual-integrity defect here. |
| `specs/implementation-guide.md` (+.html) | v1.2. 6 phases, **28 chunks** (Round 2 corrected an arithmetic error present since v1.0 — the "24"/"23" counts were wrong). P2-C05, a second early real-chart slice, still lands at position 8 — the plan had otherwise reverted to a fully horizontal build with no further user-visible milestone until chunk 25. Full dependency network, critical path, parallel-work sets, and a complete wiring matrix — no empty `Demanded by` cells, no exemptions needed. |
| `design-review/design-review-001.html` | Round 1 design review. 4 parallel panels, 10 Critical + 16 Important findings, all fixed same-session (v1.1). Superseded as the current-trust record by Round 2 below — kept as historical record. |
| `design-review/design-review-002.html` | Round 2 design review — independent re-check of the v1.1 fixes under strict criteria, plus two new panel types (Security; ATAM Quality-Attribute sub-panels for reproducibility and separability). 16 of 26 Round 1 findings held cleanly; 3 were only partially resolved or regressed; 8 new Critical + 9 new Important findings surfaced. All patched into v1.2, same-session, again self-verified. Verdict was "Do not build" pre-fix. |
| `reviews/calibration.md` | GVM review-calibration record. Score history across both rounds, anchor examples, and the explicit, twice-repeated note that each round's fixes were applied by the same agent that found them and should be independently re-verified before full trust — Round 3 (or build) inherits this open item. |

## What does not exist

No source code. No worlds, no models, no judge, no charts. Nothing has
been built. This is intentional — build starts only after design review
(recommended next step) or explicit user sign-off to skip it.

---

## Review history (why the documents look the way they do)

Requirements went through more scrutiny than the standard pipeline calls
for, because the user explicitly did not trust a single self-review pass.
In order:

1. **Six-expert GVM review board** (model risk, world-model research,
   forecast verification, requirements engineering, design/communication,
   psychology) — 2 critical + ~20 important findings, all applied. Fixed:
   MU-5's unmeasurable accuracy criterion, JU-1's overclaimed
   "independence," WD-4's non-existent single divergence "rate," JU-8's
   statistically invalid exception pooling, three MoSCoW inversions
   (MU-9/JU-12/NF-6 promoted Should→Must).
2. **Independent technical audit** (world-model/physics/stats correctness,
   fresh agent, no prior context) — found the flight-simulator sentence was
   factually backwards in both essay and requirements, and that JU-6's
   conditioned-climatology reference assumed integrator-preserved invariants
   that a generic integrator doesn't guarantee. Both fixed (WD-3 now
   requires measuring integrator drift).
3. **LeCun-grounded review** — found "planning"/"control" (WD-6) names a
   passive open-loop prediction test, not the closed-loop action-selection
   his own framework treats as a world model's actual purpose. Disclosed in
   JU-10.
4. **Novelty check (live web search)** — found a July 2026 paper (Oefinger
   et al., RSS Workshop on Robot World Models) making an analogous
   institutional-transplant argument via aerospace/automotive safety
   validation instead of banking. The broad "nobody anywhere" claim didn't
   survive; the narrower claim (nobody's brought *banking's specific
   mechanism* here) does. Both essay and requirements narrowed and now cite
   the adjacent paper. Also caught SR 11-7 being superseded by SR 26-2.

Nothing in this history changed a settled scope decision (see below). All of
it changed wording, precision, or citations.

---

## Decisions already settled — do not reopen

Full table with reasoning is in `CLAUDE.md`. In short:

- Two worlds, both from the start. The user chose this over a recommendation to
  defer the pendulum.
- Every world has an action lever. Without it this is a forecasting tool, not a
  world-model judge.
- Truth and model share one integrator and step size, enforced by test.
- Trust horizons are task-relative: a tight control task and a loose planning task
  per world.
- At least two *unrigged* models, matched on accuracy, differing only in how they
  derive uncertainty.
- Deliberately-broken models are test fixtures, never findings, and are labelled
  as such on charts as well as in docs.
- Recipes, expected rankings, and all thresholds are recorded **before** judging.

---

## What comes next

Round 2 already happened — this is what Round 1's handover recommended, and it found a real, comparable-sized new defect population even after a full v1.1 fix pass. Two reasonable options remain, in order of recommendation:
1. **A Round 3 design review**, prioritising independent re-verification of the two Round 2 fixes least trustworthy as self-graded work (the double-pendulum EOM correction; the NF-4 scan redesign) ahead of fresh discovery — shared rule 16 means Round 3 is where genuine dual/blind review activates, which would finally close the honesty gap both `reviews/calibration.md` rounds have named and neither has closed.
2. **`/gvm-build`**, accepting the residual self-verification risk, starting at Phase 1 / chunk P1-C01 (foundations: scaffold, serializer, seed plumbing), followed by P1-C02 (the walking-skeleton MVP slice) and P2-C05 (the second early real-chart slice) — 28 chunks total per the v1.2-corrected implementation guide.

The GVM skill files are now committed in this repo's own `.claude/skills/`
(no longer an external-availability gap — resolved a few sessions back,
after the earlier note below about a missing `/gvm-tech-spec` install).

---

## Things offered to the user that touch the essay, not the code

1. ~~Meteorology rivals banking as "the closest working analogue."~~
   **Resolved.** The requirements now credit meteorology explicitly for the
   *measurement* techniques (Murphy's skill scores, Gneiting's calibration +
   sharpness, the weather-vs-climate switch — all in the Expert Panel table)
   while crediting banking only for the *institutional/enforcement*
   mechanism. Two disciplines, two different contributions, both named.
2. **"No threshold forces a stop" is still unaddressed and still open.**
   Frontier AI labs publish risk policies where crossing a capability
   threshold triggers mandatory mitigations. Self-written and self-assessed,
   and they cover dangerous capabilities rather than predictive fidelity —
   which would strengthen the essay's claim once stated precisely, not
   weaken it. Nobody has touched this line yet. The user's call.
3. ~~Deborah Raji's 2020 framework should be cited.~~ **Already resolved** —
   she's in the Expert Panel table and has been since before this handover
   was first written.

---

## Notes for whoever picks this up

- **Plain English is a functional requirement, not tone.** The user must be able
  to narrate what was built to other people. Explain every step in ordinary words,
  inside the artefacts, not only in conversation. Never assume domain knowledge
  carries across fields — including from the user's own field.
- **The repo is public from its first commit.** Nothing from the user's
  professional context, ever. Banking is described from published sources only.
- **Auto-commit and push at milestones.** The user does not run git commands.
- **Session memory does not travel.** Local memory files are tied to a different
  project path, so a session started here reads `CLAUDE.md` and this file and
  nothing else. Anything that must survive belongs in a file.
- The dominant project risk is not technical. It is that on a clean toy world
  every model passes and the judge never says anything surprising.
