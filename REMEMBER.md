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

Ordered by when they bite. *(D1–D6, D12 and D14 were settled at Round 10 on
2026-09-28 — see Closed.)*

| # | Decision | Recommendation | Source |
|---|---|---|---|
| D15 | **New, found while applying Round 10 (not yet reviewed): the climatology reference is still null-action.** JU-6's conditioned climatology comes from one 200,000-step *unkicked* trajectory, binned by the conserved quantity; evaluation trials now carry kicks, which move a trajectory between bins after the switch step. With small, sparse kicks the effect should be modest, but it is unmeasured. Options: disclose it, measure it at P4-C05, or build the climatology from a kicked trajectory. | Measure at P4-C05 and disclose the figure; route any design change through review. | backlog A16; 2026-09-28 |
| D16 | **New at P3-C09 (backlog A17):** should every `out-large-action` test run be *guaranteed* at least one kick (by construction), rather than checked by a test as now? Without a kick a run in that region would be labelled "fully familiar". Rare (≈0.09% of LV runs) and absent with the pinned seed. Also: two small helper modules and a `kick_rate_per_s` world setting are not yet named in the specs. | Ratify at the next review; guaranteeing a kick is a one-line rule. | backlog A17; 2026-09-29 |
| D17 | **CLOSED 2026-10-04 — implemented on the owner's "go ahead with your recommendations"** (backlog A23; `prereg/recipe.md` revision log). Held-out set now has its own kick quota (`heldout_kick_pairs: 1000`). | — | backlog A23 |
| D18 | **CLOSED 2026-10-04 — implemented on the owner's "go ahead with your recommendations"** (backlog A23; `prereg/recipe.md` revision log). Cosine learning-rate decay (`lr_initial` 1e-3 → `lr_final` 1e-5, epochs stay 100) and a median-of-5-seeds sufficiency test (`sufficiency_seeds: 5`); seed-to-seed spread of the held-out error fell from up to ~20× to ~1.3×. **The pre-registered fallback to 100,000 pairs was enacted 2026-10-04** (Model A failed the median test on both worlds: ratios 1.29 / 2.75; one failing model decides it). | — | backlog A23 |
| D19 | **New at P3-C04 (disclosure for the owner):** at build time I computed *calibration* statistics (z-scores, coverage of one error bar) of both practice models on the **held-out pairs** — not evaluation trials, not used to change any model, the recipe or `prereg/prediction.md` (all unchanged). Result: Model A well calibrated, the ensemble over-confident. Prereg residual 5 forbids *evaluation-trial* metrics before the freeze; held-out pairs are the build-time sufficiency/kick-report set, so this is not a breach, but it is a look at the thing the experiment will later test. | Decide whether this needs a line in the verdict's limits (JU-10) or the prediction's revision log; recommended: disclose in `prereg/prediction.md`'s revision note that the build-time calibration was seen and the prediction was left unchanged. | `build/measurements/p3-c04-model-b-real-run.md` |
| D7 | **Registry discovery carve-out:** allow only `wmj/models/registry.py` to import `importlib`/`pkgutil`. | Decide at P6-C01; ratify in review. | backlog A11 |
| D8 | **BC-4 purity decision** (drop the completeness claim; byte-identity as backstop; no out-of-process sandbox) was delegated to the assistant and is "reversible in the spec text". | Owner ratifies or reverses. | calibration:228 |
| D9 | **Essay: "No threshold forces a stop" is still unaddressed** — frontier AI labs publish risk policies where crossing a capability threshold triggers mandatory action. Mention it? | Owner's call. | `HANDOVER.md:296` |
| D10 | **Essay: the AI-assistance transparency note.** Commits `71efc74` and `HANDOVER.md:198` say it was added "alongside the disclaimer"; **no committed version of the essay contains it** (checked 2026-09-27). Add it, or correct the record. | Add it — an undisclosed gap in a project about honest disclosure. | verified 2026-09-27 |
| D11 | **Essay Sources:** Deborah Raji is in the requirements' expert panel but not in the essay's Sources table. | Owner's call. | essay:197–215 |
| D13 | **P1-C01 check-in** "has not yet been presented to the user" per its handover; later chunks proceeded. | Treat as superseded unless the owner wants it. | P1-C01 handover:136 |

## 3. Owed work, by stage

**Now — next: Phase-3 wiring audit (Hard Gate 7), then Phase 4. P3-C08 (action-response check + `fx-action-blind`, DONE 2026-10-04, two review passes, handover `build/handovers/P3-C08.md`), P3-C05 (the three fixtures, DONE 2026-10-04, five review passes, handover `build/handovers/P3-C05.md`), P3-C04 (the ensemble, with the P3-C02 re-open), P3-C03 (Model A) and P3-C06 are DONE (handovers in `build/handovers/`); owner decisions D17 and D18 are open and affect the recipe before the freeze. P3-C10 (the one-time lock) is DONE (2026-10-03, six review passes, handover `build/handovers/P3-C10.md`).**
- **Independent re-check of the v1.8 purity guards and the rewritten ADR-004 orchestration loop** — owed by BC-2 since `calibration.md:249`; Rounds 9 and 10 were scoped elsewhere. Neither is built yet (P4-C06, P6-C01).
- Fold the remaining spec text corrections listed in §5 into the specs at the next review.


**P3-C10 — DONE.** Carry-forward: P6-C01 must hold the bytes `check_prereg` verified (or re-check immediately before use) — the certificate is a commit id, not the bytes later read (residual 6). Making the freeze itself (committing `prereg/FREEZE`) is a human act after the recipe is final; D17 and the sufficiency-test result come first.

**P3-C06 — training data: DONE 2026-10-03 (five review passes, handover `build/handovers/P3-C06.md`; `src/wmj/harness/training.py`). Kick-safety gate calls the real generator. Full-scale: LV 13,874 kick pairs available (need 12,500), pendulum 19,900; builds in ~1 s / ~11 s. Open: D17 (held-out kick share ~0.1% — owner decision before P3-C03's kick report). Note for P6-C01: `read_training_recipe` has no from-bytes entry; add one if holding verified bytes.**
- Draw kicks with `wmj.harness.kicks.seeded_kick_sequences(..., "train-kicks", ...)` and step with `transition_batch` (both built at P3-C09); then point the training half of `tests/gates/test_kick_safety_full_scale.py` at the real generator.
- 2,000 kicked full-horizon trajectories per world, then the harness's subsample (12,500 kick pairs + non-kick pairs to M), held-out and grad-check sets — models §4 as revised at Round 10.
- **Measure runtime** for real and report it against judge ADR-J6's new measured envelope.
- Start-disjointness (TC-MU7-01), purpose-keyed seeds (TC-MU7-02, TC-NF1-10), subsample (TC-MU7-03), train-twice-identical (TC-MU8-01).

**P3-C03 / P3-C04 — Models A and B** (P3-C03 DONE 2026-10-04, ten review passes, handover `build/handovers/P3-C03.md`; built: `src/wmj/models/direct.py`, `harness/sufficiency.py`; see `build/measurements/p3-c03-model-a-real-run.md` and D18)
- β-NLL with the written-out gradient (TC-MU5-04); `forward_invariant` + `predict_batch` (TC-MU1-04).
- **Run the M-sufficiency test** (TC-MU5-05) and report the result, and the kick/non-kick held-out split, to the owner. The test never looks at the MU-5 margin.
- Report measured convergence + runtime at `epochs: 100`; any change is an open recipe revision, never silent — and only before the freeze.
- Per iteration: `forward` → `backward` → `Adam.step`, in that order (no runtime guard exists).
- Gradient-check on real normalised training inputs before training; factories pass `seeds.rng("weights")` / per-member streams.
- P3-C04: `sqrt(1+1/K)·std(ddof=1)` spread mapping (TC-MU5-03).
- **Independent re-check owed for the 2026-10-04 amendments (D17 quota in `harness/training.py`, D18 cosine decay in `models/direct.py`, median-of-seeds in `harness/sufficiency.py`):** built with tests and the author's own mutation checks only; fold them into P3-C04's review passes (they share the files).
- ~~P3-C02 must be re-opened~~ **DONE with P3-C04 (2026-10-04):** persistence/linear `stateless`, persistence `predict_batch`, TC-MU1-04 over the roster.
- **D17 clarification:** the 6 (LV) / 10 (pendulum) kicked held-out rows are for the 50,000-pair build; the sufficiency report scores the 100,000-pair build's held-out set, which has 10 (LV) / 7 (pendulum). Same conclusion — far too few for a kick-split claim.
- Serialising `KickSplit` counts (numpy ints) to JSON at P6-C01: cast to `int`.
- **Contract from P3-C06's review (read before building the models):**
  - *Factories must refuse `train_pairs is None`* (the pair fields are optional only for the skeleton/preview callers; backlog A19).
  - *Shuffle every epoch with your own seeded stream:* `train_pairs` lists the 12,500 kick pairs first, then the non-kick pairs.
  - *Sufficiency test:* build the 2M set with `dataclasses.replace(recipe, subsample_pairs=...)` (the M set is an exact prefix of it, verified at full scale). **The held-out set of an M build and a 2M build are different draws and overlap with the other's training set (361/10,000 LV rows, 57/10,000 pendulum rows of the M build's held-out set sit inside the 2M training set) — score *both* models on the 2M build's held-out set, never each on a different one or the 2M model on the M build's.** Also disclose next to D17 that held-out pairs are pair-level draws, so about 3.5% (LV) have their successor pair in the M training set (7% in the 2M set), which mildly favours the 2M model.
  - *The fallback count lives in the recipe* (`subsample_pairs_fallback: 100000`) and the sufficiency tolerance in `sufficiency_tolerance:`; `read_training_recipe` does not read them — add one shared reader rather than hard-coding either.

- **NF-2 runtime (new, from P3-C04):** training the two practice models on both worlds takes ≈ 6 min (ensemble ≈ 150 s/world, Model A ≈ 33 s/world) against a 600 s run target that must also include evaluation (≈ 5–6 min est.). P6-C01 re-measures end to end; revise NF-2's number openly if it does not fit.
- **Action-response check (from P3-C08, backlog A26):** `persistence` and `linear` are flagged action-blind *by construction* — P6-C01 must decide how a baseline's flag is presented (never as a finding or a defect). The tolerance `1e-9` of scale, the probe design and the new seed purpose `(action-response, <world>, states)` are chosen in code, not in `prereg/recipe.md`: the owner may want them pre-registered (low risk — the real margin is seven orders of magnitude). P6-C01 must call `check_action_response` for every model and carry the result into the verdict record.
- **Phase-3 wiring audit (2026-10-04): PASSED by hand-run greps; the matrix itself needs rows (backlog A27)** — `harness.training`, `harness.sufficiency` (build-time only), `harness.kicks`; `harness.check_prereg` is `harness/prereg.py`. At P6-C01 re-audit the rows whose consumer is `wmj run`: `harness.action_response`, `models.fixtures`, `Model.predict_batch`, `harness.training` (incl. `assert_eval_starts_disjoint`).
- **Shared-core cache is one slot (from P3-C05, backlog A25):** `shared_direct_core` holds the most recent (data, seed, world) network. P6-C01 must build models world-by-world (or widen the slot) or training doubles/quadruples; `fx-action-blind` (P3-C08) must also use `shared_direct_core`.
- **Fixtures must join the roster test:** add each new model module's import to `tests/unit/models/test_roster_batch.py` (auto-discovery lands at P6-C01), declare `stateless`, implement `predict_batch`.

**P3-C05, P3-C08 — fixtures, action-blind check** (TC-MU3-01..04, TC-MU4-01/02, TC-MU1-04). **P3-C08 must report whether the smaller LV lever (kicks ≤ 0.1, Round 10) is still enough for the action-blind fixture to be caught.**

**Phase-3 end — Hard Gate 7 wiring audit.** Also explicitly re-run Phase 1's (no recorded run; Phase 2's inverse audit did cover all modules). Watch: the matrix says `harness.check_prereg` but the module is `harness/prereg.py`; include `harness.action_response`.

**Phase 4**
- **TC-MU6-05(b)** — now owned by P4-C02 (Round 10 closed the wiring gap).
- Judge-side `wmj/judge/distance.py` (ADR-J5) — chunk not named; place it in P4.
- **P4-C05 forward note:** reconcile the climatology reference trajectory's population with the drift benchmark's, or narrow the drift bound's rationale (now in the guide's P4-C05 text). **Also measure D15** (kicked trials vs the unkicked climatology).
- P4-C05 grows the climatology producer in `harness.benchmarks`; the P2-C05 stand-in block's producer swaps to the real `error_vs_horizon`.
- P4-C06: runtime purity harness owns the lint's disclosed residual (`ctypes`, pre-capture).

**Phase 5** — P5-C03: full Chart-2 caption and switch lines. P5-C04: TC-RP7-02 SVG identity, TC-JU12-04, model card, `writer.py` sole writer to `out/`.

**P6-C01** — registry auto-discovery + D7 carve-out; `wmj run`, `wmj verify`, `wmj list-models` (output format unspecified — pin it); orchestration loop with baseline pre-pass and **batched rollouts** (Round 10); **record the freeze SHA `check_prereg` returns as `meta.prereg_commit`** (TC-MU6-07); TC-MU9-01/02/03, TC-MU2-02 full, TC-NF1-05/09, full TC-WD3-01, full TC-NF1-01/02; wire benchmarks, regions, action_response, fixtures into `wmj run`. **Also wire `harness.training.assert_eval_starts_disjoint` into the run** (models spec lines 289/313 say the harness asserts start-disjointness at run time; it is built and tested at P3-C06 but no caller in `src/` uses it yet). Read and hold the bytes `check_prereg` verified (P3-C10 residual 6).

**P6-C02** — regenerate the stale `specs/*.html` twins with the parity hash; runtime/dependency/confidentiality gates; README must state the PNG byte-identity exclusion and the SVG reproducibility property.

**Before P6-C03 — the freeze.** Commit `prereg/FREEZE` once (one freeze, ever — models ADR-M5). **No evaluation-trial metric of either unrigged model may be computed before it** (disclosed residual #5).

**P6-C03 — the judged run** — `check_prereg` → full run → publish `out/` → **publish either way (TC-MU6-02) — the owner's job, not code**; human passes: TC-JU10-02 (a newcomer sees judge and models share an author), **TC-NF4-02 (a reader with banking knowledge checks every description is generic and publicly sourced)**, TC-NF5-01 (no overclaim), TC-RP5-01; answer OQ-3 honestly.

**Before anything is published** — verify every arXiv citation in full text (only abstracts were read; arXiv is blocked in this environment).

## 4. Promises the assistant made

| Promise | Made |
|---|---|
| Flag the moment the pieces exist to run the **MU-5 separability test** — the thesis's moment of truth — and offer a clearly labelled throwaway spike if the owner wants certainty sooner. | 2026-09-11 |
| Say when the build reaches **cosmetic work** (Phase 5/6 copy, README, changelog) so the owner can switch to Sonnet; keep Opus for the numerically delicate chunks. | 2026-09-11 |
| Surface the **measured epoch/runtime numbers** at P3-C06/C03. | 2026-09-11 |
| Report **Model A's fit openly** — now via the pre-registered sufficiency test and the kick/non-kick held-out split (Round 10 replaced "fit vs the MU-5 margin", which would have tuned against the comparison). | 2026-09-27, revised 2026-09-28 |
| Report whether the **smaller LV lever** still lets the action-blind fixture be caught (P3-C08). | 2026-09-28 |
| Report the **first end-to-end runtime** (TC-NF2-01) against ADR-J6's measured envelope, and say plainly if it is over. | 2026-09-28 |
| Keep the **progress page** current (the owner asked to see what's complete and where we are, end to end): https://claude.ai/artifact/YaTnTEep1qZbjpnn4xueHR — republish from `judge-progress.html` (kept in the session scratchpad; regenerate from `build/handovers/` if lost) after each build step. | 2026-10-03 |
| Keep the **verdict-sheet mock-up** (the picture of the finished product the owner asked for): https://claude.ai/code/artifact/75cc45a4-1164-4312-8299-9ff9297a78fa — illustrative placeholder numbers and fixture names only; regenerate from real output after the judged run. | 2026-09-05 |

## 5. Housekeeping debt (stale text)

*Reviewed specs — change only at the next review:* `worlds.md:257`
example emits `-Infinity` (not valid JSON — encode or disclose); implementation
guide's P1-C03 row lists the old models allowlist, and its ASCII dependency
diagram predates P3-C09/P3-C10 (the text re-derivation is current); `requirements.md:192` MU-1
cites JU-11 where MU-6 is meant, `:238` MU-9 lacks the unrigged-model
carve-out, `:96` still says the expert panel was single-pass scored (later
verified per `CLAUDE.md`); TC-MU2-03, TC-NF6-10, TC-NF6-11 appear in no guide
`[Test:]` tag (tests exist); ADR-J6 lacks the MU-9-vs-runtime tradeoff sentence.

*Other records:* `requirements/requirements.html` banner still reads v1.2
(only NF-2 was synced to v1.4 on 2026-09-28 — regenerate at P6-C02 with the
other twins); `calibration.md` — BC-1/3/5 statuses never updated, Round 7–8
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
threshold; revisiting the Won'ts (WD-8, MU-10, JU-13) in a later round; **hash-published
cached weights** (Round 10 R10-F19, deferred: only needed if the judge ever
ships pre-trained weights instead of training from scratch).

## 7. Disclosed limits — keep them disclosed, never quietly drop

ADR-M5's five residuals (rewritable git history, undetectable
non-publication, forgeable commit timestamps, entry substance (A13), no
evaluation before the freeze — procedural, not enforceable); the gradient
check's near-dead-unit blind spot (A10); the train/eval kick-share shift (25%
vs ~1% / ~0.2%); the small LV lever (kicks ≤ 0.1);
JU-10's seven limitation texts, verbatim, in every verdict; the not-tested
list; TC-JU12-04 (`ctypes`, pre-capture); NF-4 gaps (no pre-push layer,
web-UI merges, gc-pruned objects); cross-platform identity out of scope; PNGs
outside byte-identity; BLAS thread settings not verifiable at runtime; worlds
hardcoded, not auto-discovered.

## Closed

*(Move items here with date and commit when done.)*

| Item | Closed | How |
|---|---|---|
| D1 Model A's loss | 2026-09-28 | β-NLL, β = 0.5, gradient written out (models ADR-M3, recipe). Round 10. |
| D2 Freeze-point contradiction | 2026-09-28 | One freeze, ever: the commit that adds `prereg/FREEZE` (models ADR-M5); build at P3-C10. |
| D3 Training recipe over budget | 2026-09-28 | 50,000-pair harness subsample, batch 256, batched evaluation; NF-2 purpose stated (requirements v1.4); ADR-J6 measured envelope. `sharpness_hedge_threshold = scale` ratified. |
| D4 Gradient-check tolerance | 2026-09-28 | A10 ratified: floor 1e-3, tolerance 1e-5, blind spot disclosed; check on 64 pairs. |
| D5 A13 disposition | 2026-09-28 | Confirmed "disclose, don't fake a check"; ADR-M5 residual #4. |
| D6 B1 design review | 2026-09-28 | Reviewed in Round 10; no finding against it. |
| D12 When to run Round 10 | 2026-09-28 | Run 2026-09-27/28; `design-review/design-review-010.html`. |
| Independent fix-check of the Round 10 edits | 2026-09-28 | Five fresh checkers until clean: passes [(1,5),(2,2),(3,6),(4,2),(final,0)]; commits 1ccf8ce, 3426161, 8d34120, 913231f. `reviews/calibration.md`. |
| P3-C09 built (batched worlds, kick generator, `out-large-action`, kicked benchmark) | 2026-10-03 | Review passes [(1,1),(2,1),(3,2),(4,0)]; 429 tests; `build/handovers/P3-C09.md`. |
| D14 How action sequences are drawn | 2026-09-28 | Sparse seeded kicks from one generator; LV kicks shrunk after crash measurements (worlds ADR-W2, §4.1). |
