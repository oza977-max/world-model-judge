# REMEMBER — everything owed, promised, or waiting on a decision

**In plain words:** this file is the project's memory. Sessions don't carry
memory from one to the next; only files in this repo do. So everything the
owner asked to be kept, every item pushed to "a later stage", every decision
still waiting on the owner, and every promise the assistant made lives here,
with where it came from. **Every session reads this after `CLAUDE.md`.** When
an item is done, move it to the *Closed* section with the date and commit;
when something new is deferred, add it here the same day. Nothing is removed
silently.

*Compiled 2026-09-27 from a full sweep of: the conversation transcript,
`HANDOVER.md`, `CLAUDE.md`, `README.md`, `risks/`, the essay
(`requirements/wordsareamenu.html`), `requirements/`, `test-cases/`, all
`specs/`, `reviews/calibration.md`, design reviews 001–009, code-review-001,
every build handover and prompt, and `build/spec-corrections-backlog.md`.
Build state at compile time: P1-C01…P3-C02 and P3-C07 done; P3-C06 next.*

---

## 1. Standing rules — hold in every session, never expire

| Rule | Source |
|---|---|
| **This judge can have no margin of error.** When in doubt, verify by running it, not by reasoning about it. | Owner, 2026-09-04 |
| **Confidentiality is absolute (NF-4).** Public repo from the first commit. Nothing from the owner's professional context: no employer, internal figures, team or committee names. Banking described from published sources only (SR 11-7, since superseded by SR 26-2; PRA SS1/23). | `CLAUDE.md`, NF-4, `risks/` |
| **Plain English is a functional requirement.** Every artefact explains itself in ordinary words, inside the artefact; every requirement keeps its "In plain words" line. Never assume knowledge carries across fields, including from the owner's own. | `CLAUDE.md`, NF-5, `HANDOVER.md:310` |
| **Auto-commit and push at milestones.** The owner does not run git. | `CLAUDE.md` |
| **Anything that must survive goes in a file** — this one. | `HANDOVER.md:317` |
| **Never edit an already-reviewed spec to match code mid-build.** Record it in `build/spec-corrections-backlog.md` for the next design review. Changelog rows are append-only. | build practice since P1 |
| **No changes after seeing results.** Recipe, thresholds and predictions are fixed before judging (MU-6, JU-11). Changes *before* any result are allowed — through the proper route (requirements → test cases → design review → build, or an open, dated recipe revision), never a quiet edit. | MU-6, JU-11; clarified 2026-09-27 |
| **Fixtures are engineered failures, never findings** — labelled as fixtures in code, docs and charts. | MU-4, RP-8 |
| **The owner does not trust a single self-review.** Every build chunk runs the independent review loop until zero Critical/Important. Findings are presented to the owner; the assistant recommends, the owner decides. | `HANDOVER.md:223`, GVM Gate 3, shared rule 28 |
| **Design decisions (not patches) belong to the owner** unless explicitly delegated. | `HANDOVER.md:148` |
| **No model identifiers** in commit messages, PR bodies, code comments or any pushed artefact, beyond the required commit attribution footer. | session rule |
| **Requirement IDs appear** in gate-test names and error messages. | build practice |
| **Keep the essay's employer disclaimer verbatim** (essay line 73; "standing firm-required disclaimer"). | commit `71efc74` |
| **Keep `risks/risk-assessment.md` pointing at essay draft v2.1** — it is the historical record of what it was written against. | `HANDOVER.md:199` |
| **Editorial stance:** banking is the illustrative anchor; the project takes no side on which discipline should own AI risk. | commit `c37961f` |
| **The dominant risk is not technical:** on clean toy worlds every model passes and the judge says nothing surprising. If that happens, say so plainly — never pad (OQ-3). | `risks/`, requirements OQ-3 |
| **The build must deliver what essay draft v2.7 promised**, or state the difference openly (Constraint 4). Promises: two worlds with a lever; state+action→next state; one shared integrator/step; drift vs the world's own divergence, grading the pattern not the path (weather vs climate); one-step accuracy vs "nothing changes" and "straight line"; 90%-sure check including unseen starts; hedging (sharpness) check; task-named trust horizon; backtesting exception plots; "drift curves and all". | requirements:443, essay:182–191 |
| **Settled scope decisions — do not re-litigate:** see the table in `CLAUDE.md`. | `CLAUDE.md` |

## 2. Decisions waiting on the owner

Ordered by when they bite. **D1–D6 should be settled before P3-C03**
(Model A), ideally in one short Round 10 design review (D12).

| # | Decision | Recommendation | Source |
|---|---|---|---|
| D1 | **Model A's loss.** The recipe trains Model A with plain Gaussian NLL, a setup the literature (Seitzer et al., ICLR 2022) documents as fitting poorly. If kept and the ensemble wins, the headline looks rigged. Switch to β-NLL, or keep NLL and report the fit? | Decide before P3-C03; β-NLL is the standard fix and keeps Model A a self-predicted-error-bar model. | backlog horizon addendum; 2026-09-27 |
| D2 | **Freeze-point contradiction.** The spec's remedy is "revise the recipe openly", but `check_prereg` compares `recipe.md` to its *first* commit, so any revision — even an honest one before results — fails certification at P6-C03. Freeze at a declared pre-judging commit instead? | Yes; spec change via design review. | backlog A12; 2026-09-27 |
| D3 | **Pinned values:** `epochs: 100`, the unstated training-trajectory horizon, and `sharpness_hedge_threshold = scale` (per world, not per region). | Ratify after P3-C06 measures runtime. | backlog A12; DR-008 minor |
| D4 | **Gradient-check tolerance:** metric floor 1e-3 and tolerance 1e-5 (spec says 1e-6), including the disclosed near-dead-unit blind spot. | Ratify, or prefer an `allclose`-style check. | backlog A10 |
| D5 | **A13 disposition:** `check_prereg` cannot verify an entry's substance — disclosed rather than "fixed". Confirm "disclose, don't fake a check", and add it to ADR-M5's residuals. Never confirmed by the owner. | Confirm. | backlog A13; P3-C07 handover |
| D6 | **Does the new B1 text (MU-3's 4th fixture, TC-MU3-04, P3-C08) need its own design review?** Its route was "requirements → test cases → design review → build"; no design review of the new text has run. | Include in Round 10. | calibration:287 |
| D7 | **Registry discovery carve-out:** allow only `wmj/models/registry.py` to import `importlib`/`pkgutil`. | Decide at P6-C01; ratify in review. | backlog A11 |
| D8 | **BC-4 purity decision** (drop the completeness claim; byte-identity as backstop; no out-of-process sandbox) was delegated to the assistant and is "reversible in the spec text". | Owner ratifies or reverses. | calibration:228 |
| D9 | **Essay: "No threshold forces a stop" is still unaddressed** — frontier AI labs publish risk policies where crossing a capability threshold triggers mandatory action. Mention it? | Owner's call. | `HANDOVER.md:296` |
| D10 | **Essay: the AI-assistance transparency note.** Commits `71efc74` and `HANDOVER.md:198` say it was added "alongside the disclaimer"; **no committed version of the essay contains it** (checked 2026-09-27). Add it, or correct the record. | Add it — an undisclosed gap in a project about honest disclosure. | verified 2026-09-27 |
| D11 | **Essay Sources:** Deborah Raji is in the requirements' expert panel but not in the essay's Sources table. | Owner's call. | essay:197–215 |
| D12 | **When to run Round 10** (D1–D6, D8, plus the owed re-check in §3 "Now"). | Before P3-C03. | — |
| D13 | **P1-C01 check-in** "has not yet been presented to the user" per its handover; later chunks proceeded. | Treat as superseded unless the owner wants it. | P1-C01 handover:136 |

## 3. Owed work, by stage

**Now / Round 10**
- **Independent re-check of the v1.8 purity guards and the rewritten ADR-004 orchestration loop** — owed by BC-2 since `calibration.md:249`; Round 9 was scoped to the backlog and never looked at them. Neither is built yet (P4-C06, P6-C01).
- Fold the spec text corrections listed in §5 into the specs once ratified.

**P3-C06 — training data (next)**
- Full-horizon training trajectories (models §4: "trajectories of full horizon"), 2,000 per world.
- **Real actions from the training action range** — not zero actions — or no model can learn to respond to its action and P3-C08 tests nothing.
- **Measure runtime.** ADR-J6 budgets "training two small MLPs … ≤ 6 minutes combined", but the roster trains 6 MLPs per world (direct + 5 ensemble members) × 2 worlds = 12. Report measured numbers against the 600 s budget (feeds D3).
- Start-disjointness (TC-MU7-01), purpose-keyed seeds (TC-MU7-02), train-twice-identical (TC-MU8-01).

**P3-C03 / P3-C04 — Models A and B**
- Report Model A's fit honestly against the MU-5 margin (Seitzer risk).
- Report measured convergence + runtime at `epochs: 100`; any change is an open recipe revision, never silent (A12).
- Per iteration: `forward` → `backward` → `Adam.step`, in that order (no runtime guard exists).
- Gradient-check on real normalised training inputs before training; factories pass `seeds.rng("weights")` / per-member streams.
- P3-C04: `sqrt(1+1/K)·std(ddof=1)` spread mapping (TC-MU5-03).

**P3-C05, P3-C08 — fixtures, action-blind check** (TC-MU3-01..04, TC-MU4-01/02).

**Phase-3 end — Hard Gate 7 wiring audit.** Also explicitly re-run Phase 1's (no recorded run; Phase 2's inverse audit did cover all modules). Watch: the matrix says `harness.check_prereg` but the module is `harness/prereg.py`; include `harness.action_response`.

**Phase 4**
- **TC-MU6-05(b)** — the Verdict's skill block names exactly `vs_persistence` / `vs_linear` — has **no owning chunk** (wiring gap). Build it in P4-C02 (skill block).
- Judge-side `wmj/judge/distance.py` (ADR-J5) — chunk not named; place it in P4.
- **P4-C05 forward note:** reconcile the climatology reference trajectory's population with the drift benchmark's, or narrow the drift bound's rationale. Not recorded in the guide's P4-C05 text.
- P4-C05 grows the climatology producer in `harness.benchmarks`; the P2-C05 stand-in block's producer swaps to the real `error_vs_horizon`.
- P4-C06: runtime purity harness owns the lint's disclosed residual (`ctypes`, pre-capture).

**Phase 5** — P5-C03: full Chart-2 caption and switch lines. P5-C04: TC-RP7-02 SVG identity, TC-JU12-04, model card, `writer.py` sole writer to `out/`.

**P6-C01** — registry auto-discovery + D7 carve-out; `wmj run`, `wmj verify`, `wmj list-models` (output format unspecified — pin it); orchestration loop with baseline pre-pass; **record `check_prereg`'s commit-of-record SHA in verdict metadata**; TC-MU9-01/02/03, TC-MU2-02 full, TC-NF1-05/09, full TC-WD3-01, full TC-NF1-01/02; wire benchmarks, regions, action_response, fixtures into `wmj run`.

**P6-C02** — regenerate the stale `specs/*.html` twins with the parity hash; runtime/dependency/confidentiality gates; README must state the PNG byte-identity exclusion and the SVG reproducibility property.

**P6-C03 — the judged run** — `check_prereg` → full run → publish `out/` → **publish either way (TC-MU6-02) — the owner's job, not code**; human passes: TC-JU10-02 (a newcomer sees judge and models share an author), **TC-NF4-02 (a reader with banking knowledge checks every description is generic and publicly sourced)**, TC-NF5-01 (no overclaim), TC-RP5-01; answer OQ-3 honestly.

**Before anything is published** — verify every arXiv citation in full text (only abstracts were read; arXiv is blocked in this environment).

## 4. Promises the assistant made

| Promise | Made |
|---|---|
| Flag the moment the pieces exist to run the **MU-5 separability test** — the thesis's moment of truth — and offer a clearly labelled throwaway spike if the owner wants certainty sooner. | 2026-09-11 |
| Say when the build reaches **cosmetic work** (Phase 5/6 copy, README, changelog) so the owner can switch to Sonnet; keep Opus for the numerically delicate chunks. | 2026-09-11 |
| Surface the **measured epoch/runtime numbers** at P3-C06/C03. | 2026-09-11 |
| Report **Model A's fit openly** against the MU-5 margin. | 2026-09-27 |
| Keep the **verdict-sheet mock-up** (the picture of the finished product the owner asked for): https://claude.ai/code/artifact/75cc45a4-1164-4312-8299-9ff9297a78fa — illustrative placeholder numbers and fixture names only; regenerate from real output after the judged run. | 2026-09-05 |

## 5. Housekeeping debt (stale text)

*Reviewed specs — change only via Round 10:* `judge.md` ADR-J6 still says
"× 7 models" (now 8) and budgets training as "two small MLPs"; `worlds.md:257`
example emits `-Infinity` (not valid JSON — encode or disclose); implementation
guide says 28 chunks in one place and 29 in another, its P1-C03 row lists the
old models allowlist, its P3-C01 tag says 1e-6; `requirements.md:192` MU-1
cites JU-11 where MU-6 is meant, `:238` MU-9 lacks the unrigged-model
carve-out, `:96` still says the expert panel was single-pass scored (later
verified per `CLAUDE.md`); TC-MU2-03, TC-NF6-10, TC-NF6-11 appear in no guide
`[Test:]` tag (tests exist); ADR-J6 lacks the MU-9-vs-runtime tradeoff sentence.

*Other records:* `calibration.md` — BC-1/3/5 statuses never updated, Round 7–8
score rows missing, line 27 truncated, line 94 unclosed bold; code-review
recurrence tracking starts at code-review Round 2; the backlog's B1 row still
says "not decided" (it was accepted, `7fc4c59`); `HANDOVER.md` "What exists"
table stale since 2026-08-31 (now bannered); optional `.stub-allowlist` for
the five dismissed Panel-F symbols.

## 6. Future-version candidates — front door only, not now

Conformal trust horizons (complement to JU-11's binomial bands); policy-ranking
agreement; a "motion under zero action" check; a check that a model's rollout
respects the world's conserved quantity; `importlib.reload` as an evasion
shape in the TC-NF6-04 corpus; a WorldContext anti-drift check against WD-5's
constants; an NF-4 check whenever logging is designed; per-region sharpness
threshold; revisiting the Won'ts (WD-8, MU-10, JU-13) in a later round.

## 7. Disclosed limits — keep them disclosed, never quietly drop

ADR-M5's three residuals (rewritable git history, undetectable
non-publication, forgeable commit timestamps) plus A13 (entry substance);
JU-10's seven limitation texts, verbatim, in every verdict; the not-tested
list; TC-JU12-04 (`ctypes`, pre-capture); NF-4 gaps (no pre-push layer,
web-UI merges, gc-pruned objects); cross-platform identity out of scope; PNGs
outside byte-identity; BLAS thread settings not verifiable at runtime; worlds
hardcoded, not auto-discovered.

## Closed

*(Move items here with date and commit when done.)*
