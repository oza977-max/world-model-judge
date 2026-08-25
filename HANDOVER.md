# Handover

## CURRENT STATE

**Phase:** Requirements **approved by the user**. Test cases written (v1.0,
68 cases across all 45 requirements). Independent verification pass on the
test cases is in progress or complete — check the note below dated after
this line before trusting the test cases as final.
**Code written:** none. Deliberately.
**Blocked on:** nothing at the requirements/test-case stage. Next gate is
`/gvm-tech-spec`, which reads the requirements and the test cases together.
**Do not** write source code or scaffolding until the technical spec exists
and has cleared design review.

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
| `test-cases/test-cases.md` | v1.0. 68 cases, one `TC-{REQ-ID}-{NN}` per requirement minimum, more for credibility-fatal ones. Includes 5 negative/phantom-gate cases (proving WD-3, WD-7, NF-1's automated checks can actually fail, not just always pass) and 4 property-based cases. Full traceability matrix at the end — zero orphan requirements, zero orphan cases. |

## What does not exist

No source code. No worlds, no models, no judge, no charts. Nothing has
been built. This is intentional — build starts only after a technical spec
exists and clears design review.

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

`/gvm-tech-spec` — turning the requirements and the test cases together into
a buildable contract: per-component sections, chunks that each name the test
cases they make pass, and resolution of the Open Questions the requirements
deliberately left for this stage (exception-band derivation, sample size,
runtime budget, the divergence-curve implementation for each world). Still
writing, not building.

After that: `/gvm-design-review` (blocking, before any code), then build,
starting with a walking skeleton — one world, one baseline, the judge, one
chart, wired end to end — before either second world or second model exists.

Note: no `/gvm-tech-spec` skill is actually installed in this session,
same gap as `/gvm-test-cases` was. Whoever picks this up next should either
have the GVM skill files available, or do the equivalent work directly
following the method the requirements and test cases already model.

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
