# World Model Judge — a checker for learned simulators

A deterministic harness that issues a **verdict** on a learned simulator: how
far ahead it can be believed, for which task, measured against what, and with an
explicit list of what was never tested.

The source thesis is the essay *Words Are a Menu. The World Is Not.* Its claim:
evaluation metrics for learned simulators exist in fragments across separate
fields, but the **institution** does not — nobody independent owns the number,
has to report it, or can force a stop when it crosses a line. Banking built that
institution for financial models. Meteorology built the measurement discipline.
Neither has been pointed at world models.

**The product is the judging.** The models being judged are deliberately
trivial, because the judging is the scarce thing. This is not a world model,
not a chatbot, and not an attempt to advance simulation.

---

## Read `REMEMBER.md` next — before doing anything

**`REMEMBER.md` is this project's memory.** It holds every standing rule, every
decision waiting on the owner, everything deferred to a later stage (by
chunk), and every promise made to the owner. Read it at the start of every
session. When an item closes, move it to its *Closed* section; when anything
new is deferred, add it the same day. A deferral that is not in that file will
be forgotten.

## Standing gates — read before doing anything

- **The build is approved and under way.** The requirements were approved on
  2026-08-25 (the old "no build until approved" gate is satisfied). Build
  with `/gvm-build`, one chunk at a time, independent review loop to zero
  Critical/Important on every chunk.
- **This judge can have no margin of error** (the owner's words). Verify by
  running things, not by reasoning about them.
- **Every step must be explained in plain English, inside the artefacts.** The
  user must be able to narrate what was built to other people; a working thing
  they cannot explain has failed its purpose. This is a functional requirement
  (NF-5 territory), not a tone preference. Every requirement in this repo
  carries an "In plain words" line — keep that pattern in every new document.
- **Auto-commit and push at milestones.** The user does not run git commands.
- **Confidentiality is absolute (NF-4).** The repo is public from its first
  commit. Nothing from the author's professional context — no internal figures,
  no employer name, no internal team or committee names. Banking practice is
  described from published public sources only (SR 11-7 — since superseded by
  SR 26-2 — and SS1/23). There is no window in which a mistake could be quietly
  fixed.
- **Never edit an already-reviewed spec to match code mid-build.** Record the
  correction in `build/spec-corrections-backlog.md` for the next design review.

---

## Scope decisions already made — do not re-litigate

| Decision | Why it was made |
|---|---|
| **Two worlds**: predator-prey *and* double pendulum | The user chose both over a recommendation to defer the pendulum. Predator-prey drifts apart slowly; the pendulum is chaotic. Having both forces the judge to separate *the model is wrong* from *the world is unpredictable*, and stops the judge being secretly specialised to one world. |
| **Every world has an action lever** | Without it this is a time-series backtester and the project's name is a lie. The judge must always grade `state + action → next state`. |
| **Truth and model share one integrator and step size** | Named in the essay as a trap. If they differ, every chart measures the integrator rather than the model — and still looks plausible. Enforced by test (WD-3), never by convention. |
| **Trust horizons are task-relative** | Each world declares a tight-tolerance control task and a loose-tolerance planning task. "Trusted for 14 steps" is unsayable without naming the task. |
| **At least two *unrigged* models** | Matched on accuracy, differing only in how they derive uncertainty (self-predicted error bar vs ensemble disagreement). This is what makes the headline claim possible: ordinary error ranks them equal, the judge does not. |
| **Deliberately-broken models are fixtures, never findings** | Detecting a failure you engineered is a passing unit test. Presenting it as a discovery would destroy the project's credibility faster than anything else. Labelled as fixtures in code, docs, and on charts (MU-4, RP-8). |
| **Recipes and expected rankings recorded before judging** | MU-6 and JU-11. A model tuned until its verdict looks good is a fixture in disguise; thresholds moved after seeing results are not thresholds. |

---

## Two design decisions that came from the expert panels

Both would have been missed otherwise, and both change behaviour:

- **Sharpness is reported alongside calibration (JU-5).** A model predicting
  "somewhere between zero and a million" is perfectly calibrated and useless.
  Without this rule the judge rewards hedging and the vaguest model wins.
- **Calibration uses a strictly proper scoring rule (JU-4).** With the wrong
  scoring rule an overconfident model scores well *by being overconfident* —
  the exact failure this project exists to catch.

And one honesty requirement pointed at ourselves: **every verdict states the
judge's own limits (JU-10)** — this method validates the middle of the
distribution, not the tails, which is where the damage happens (Rebonato); and
a toy world validates the harness, not the field (Derman).

---

## Layout

```
REMEMBER.md                     the project's memory — read every session
requirements/requirements.md    requirements (v1.4, approved; NF-2 purpose added 2026-09-28)
requirements/wordsareamenu.html the essay, draft v2.7 (the build must keep its promises)
risks/risk-assessment.md        four product risks, written before requirements
test-cases/test-cases.md        test cases (v1.7)
specs/                          the technical spec suite + implementation guide
design-review/, code-review/    review reports; reviews/calibration.md tracks them
build/prompts/, build/handovers/  one prompt + one handover per built chunk
build/spec-corrections-backlog.md  spec fixes found while building (for Round 10)
prereg/                         committed pre-registration (recipe, prediction, thresholds)
src/wmj/                        the code: worlds, models, judge, harness, reporting
tests/                          unit tests and gates
```

Build state (2026-09-28): Phases 1–2 done; Phase 3 in progress — P3-C01 (MLP
core), P3-C02 (registry), P3-C07 (pre-registration tooling) done. Round 10
design review (the training-recipe revision) closed 2026-09-28: 20 findings,
owner-triaged, fixes written into requirements v1.4 and the specs, and an
independent fix-check of those edits converged clean after five passes.
Building resumes at P3-C09 → P3-C06 (see `REMEMBER.md` §3). `HANDOVER.md` is a historical record
up to 2026-08-31.

The dominant risk is not technical: it is that on a clean toy world every model
passes and the judge never says anything surprising. Requirements MU-5 and MU-6
exist specifically to give that risk a chance to resolve honestly.

---

## Method

Built with the Grounded Vibe Methodology (`/gvm-*` skills, committed at
`.claude/skills/` — project-scoped, so any session on this repo has them).
Pipeline so far: risk assessment → requirements (v1.3, approved) → test
cases → tech spec → nine design-review rounds (Round 9, 2026-09-04: "Build
with caveats", the first verdict that cleared the build) → build (Phases 1–2
done, Phase 3 in progress) with code-review round 1 closed, and Round 10 (2026-09-28, the training-recipe
revision: "Do not build from the proposal as written" until its accepted fixes
pass an independent fix-check — see `REMEMBER.md` §3).

The five project-specific expert-scoring files (`model-risk-world-model-judge.md`,
`forecast-verification.md`, `ai-evaluation.md`, `world-models.md`,
`predictive-neuroscience.md`) referenced by requirements.md's Expert Panel
table exist and are scored, at `.claude/skills/gvm-design-system/references/industry/`
— an earlier version of this note flagged them as missing; they were
regenerated and verified later in the same session that wrote this
correction, including catching and fixing a collision where a project-
specific file was initially written over GVM's generic `model-risk.md`
(restored; the project-specific one lives at the World-Model-Judge-suffixed
filename instead).

**Update (30 Aug 2026):** five of the `/gvm-*` skills were upgraded from a
newer GVM version found in a sibling repo (`oza977-max/ai-raf-precheck`,
public, confidentiality-scanned before import): `gvm-design-review`,
`gvm-code-review`, `gvm-doc-review`, `gvm-site-survey` (each gained new
defect-class panels — Security and ATAM-driven Quality-Attribute panels
for design review), and `gvm-graph` (new — the fan-out/verify/merge
orchestration engine that wide review phases now dispatch through, with
independent blind verification built into the pattern rather than
deferred to a later review round). All five still depend on the same
`gvm-design-system/references/` library already in this repo. A sixth
skill from that source, `verify`, was deliberately **not** imported — it
is a project-specific verification checklist for that other repo's own
product, not portable methodology.
