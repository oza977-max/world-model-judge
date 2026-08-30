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

## Standing gates — read before doing anything

- **No build until the user explicitly approves the requirements.** Set
  2026-08-19 and still in force. `requirements/requirements.md` is written but
  NOT approved. Do not write source code, tests, or scaffolding until told.
- **Every step must be explained in plain English, inside the artefacts.** The
  user must be able to narrate what was built to other people; a working thing
  they cannot explain has failed its purpose. This is a functional requirement
  (NF-5 territory), not a tone preference. Every requirement in this repo
  carries an "In plain words" line — keep that pattern in every new document.
- **Auto-commit and push at milestones.** The user does not run git commands.
- **Confidentiality is absolute (NF-4).** The repo is public from its first
  commit. Nothing from the author's professional context — no internal figures,
  no employer name, no internal team or committee names. Banking practice is
  described from published public sources only (SR 11-7, SS1/23). There is no
  window in which a mistake could be quietly fixed.

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
requirements/requirements.md    45 requirements, four domains + non-functional
requirements/requirements.html  same content, Tufte-styled
risks/risk-assessment.md        four product risks, written before requirements
```

Nothing else exists yet. No source code has been written.

The dominant risk is not technical: it is that on a clean toy world every model
passes and the judge never says anything surprising. Requirements MU-5 and MU-6
exist specifically to give that risk a chance to resolve honestly.

---

## Method

Built with the Grounded Vibe Methodology (`/gvm-*` skills, committed at
`.claude/skills/` — project-scoped, so any session on this repo has them).
Pipeline so far: risk assessment → requirements (v1.2, approved) → test
cases (v1.0, 70 cases, independently verified) → tech spec (all 7 documents
at v1.1) → design review (round 1: 10 Critical + 16 Important findings,
all fixed same-session — see `design-review/design-review-001.html` and
`reviews/calibration.md`, which also names the honest gap that those
fixes were self-verified, not independently re-checked). Next stage is
either a Round 2 design review or `/gvm-build` starting at P1-C01.

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
