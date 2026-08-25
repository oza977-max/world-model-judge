# Handover

## CURRENT STATE

**Phase:** Technical specification **complete and approved by the user**,
all seven documents. `/gvm-tech-spec` ran in full: cross-cutting, four
domain specs (worlds, models, judge, reporting), architecture overview,
implementation guide — each approved individually via AskUserQuestion
before the next was written, per gate. Coverage audit ran clean on the
first pass: all 45 requirements and all 69 test cases referenced in at
least one spec, zero orphans either direction.
**Code written:** none. Deliberately. Still true — specs only.
**Blocked on:** nothing. Next gate is `/gvm-design-review` (recommended,
not yet run), then `/gvm-build` starting at Phase 1 / P1-C01.
**Do not** write source code until design review has run, or the user
explicitly says to skip it and go straight to build.

Last updated 2026-08-25.

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
| `specs/cross-cutting.md` (+.html) | v1.0. Stack (pure NumPy, hand-rolled MLPs — user's explicit choice over PyTorch/stdlib-only), the four-rule determinism strategy, five-package structure with the judge importing nothing, error-handling conventions, the two-package dependency budget (`numpy`, `matplotlib`). |
| `specs/worlds.md` (+.html) | v1.0. LV and double-pendulum constants fully pinned (dt, horizons, scale vectors, regions, actions). Shared fixed-step RK4 with a 1e-6 relative drift bound (ADR-W1). Divergence benchmark: 64 seeded starts, median curve per region (ADR-W3). |
| `specs/models.md` (+.html) | v1.0. Baselines with honest training-residual spreads. Three one-corruption fixtures. The direct-vs-ensemble unrigged pair (Nix & Weigend / Lakshminarayanan et al., both newly discovered experts) with a pre-registered `sqrt(1+1/K)`-corrected spread mapping and a 0.05 matching margin. Pre-registration enforced by git commit ordering. |
| `specs/judge.md` (+.html) | v1.0. CRPS as the scoring rule. Settles Open Questions 1/2: **N=200 independent trials, bands green [12,29] / amber [8,11]∪[30,35] / red beyond**, derived from the exact binomial CDF and verified computationally in-session (not eyeballed). Settles OQ-5's runtime half: 600s budget. Full verdict JSON schema. |
| `specs/reporting.md` (+.html) | v1.0. All four required charts designed down to authored caption templates. Fixture labelling centralised so it survives a screenshot. `wmj run` / `wmj verify` command pair. |
| `specs/architecture-overview.md` (+.html, with an inline C4 container SVG) | v1.0. Synthesis of all five specs + a Brooks conceptual-integrity review — found and resolved one real tension (NF-1's byte-identity scope vs PNG rendering; resolved by disclosure, no spec change needed). |
| `specs/implementation-guide.md` (+.html) | v1.0. 6 phases, 23 chunks, P1-C02 satisfies MVP-1 (first user-facing chunk is a runnable skeleton). Full dependency network, critical path, parallel-work sets, and a complete wiring matrix — no empty `Demanded by` cells, no exemptions needed. |

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

`/gvm-design-review` — recommended, not yet run. Then `/gvm-build`, starting
at Phase 1 / chunk P1-C01 (foundations: scaffold, serializer, seed plumbing)
followed immediately by P1-C02, the walking-skeleton MVP slice (one world,
two baselines, the skill score, one deterministic serialized output —
already scoped in the implementation guide as satisfying MVP-1).

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
