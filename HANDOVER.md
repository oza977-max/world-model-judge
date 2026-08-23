# Handover

## CURRENT STATE

**Phase:** Discovery complete. Requirements written, **awaiting user approval**.
**Code written:** none. Deliberately.
**Blocked on:** the user reviewing `requirements/requirements.md`.
**Do not** write source code, tests, or scaffolding until the user explicitly approves.

Last updated 2026-08-20.

---

## What exists

| File | What it is |
|---|---|
| `CLAUDE.md` | Project context and standing rules. Read this first. |
| `requirements/requirements.md` | 45 requirements across four domains plus non-functional. Every one carries an "In plain words" restatement. |
| `requirements/requirements.html` | Same content, Tufte-styled, generated from the Markdown. |
| `risks/risk-assessment.md` | Four product risks, written before requirements because the methodology gates on it. |

A cleaner reading view of the requirements, with plain English leading and the
formal wording secondary, is published at
`https://claude.ai/code/artifact/4143013d-fd7a-4ccc-b383-e70f4f08c92d`.

## What does not exist

No source code. No tests. No worlds, no models, no judge, no charts. Nothing has
been built. This is intentional — the user asked for a review gate before any
build begins.

---

## The four questions the user was asked to review

Carried forward because they were not answered before the session ended:

1. Does the section **"What we are building, in one page"** make sense on its own?
   If that section fails, everything downstream fails.
2. Both worlds — predator-prey **and** double pendulum — is the biggest single
   commitment in the build. Still wanted?
3. The four **"Still open"** questions at the end of the requirements, especially
   whether predator-prey is too well-behaved to produce an interesting result.
4. Any line that still reads as jargon. A plain-English restatement that needs
   decoding is a defect.

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

`/gvm-test-cases` — turning each requirement into a concrete acceptance test.
This is still writing, not building, and it is the right next step **only after
the requirements are approved**, because test cases derive from requirements and
would be thrown away if a requirement changes.

After that: `/gvm-tech-spec`, then `/gvm-design-review`, then build.

---

## Things offered to the user that touch the essay, not the code

Raised during discovery, acknowledged, not yet acted on. These are the user's
calls, not the build's:

1. **Meteorology rivals banking as "the closest working analogue."** Weather
   forecasting has scored forecasts against reference baselines, separated
   calibration from sharpness, and graded statistics past the predictability
   horizon for decades. The essay's own phrase "the weather-versus-climate
   distinction" walks past this. The sharper claim: nobody has put meteorology's
   verification discipline behind banking's enforcement teeth, and world models
   have neither half.
2. **"No threshold forces a stop" is too strong.** Frontier AI labs publish risk
   policies where crossing a capability threshold triggers mandatory mitigations.
   They are self-written and self-assessed, and they cover dangerous capabilities
   rather than predictive fidelity — which actually strengthens the argument once
   stated precisely.
3. **Deborah Raji's 2020 internal-algorithmic-audit framework** is the closest
   existing relative to the thesis. The project's angle is still distinct, but it
   should cite her rather than implying empty ground.

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
