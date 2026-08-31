# World Model Judge — Test Cases

Version 1.1 · Derived from Requirements v1.2 (25 August 2026, post-review-board and post-audit fixes)

> **Change note (v1.1, 31 August 2026 — design-review-004 repair session).** TC-MU9-01 rewritten to mechanism v3: the Round 3 version wrote its stub into the real production `wmj/models/` directory (its own designed-to-fail case could leave a phantom model behind for a later real run to execute), proved nothing about the production code path, and used an identifier-grep with the same string-evasion weakness the NF-6 gate's history proves fatal. The TC-NF6 family restructured from denylist to allowlist + fixture corpus (cross-cutting spec v1.4): four rounds of enumerated "bad shapes" were each evaded within one round by a shape nobody enumerated; the allowlist has no omission problem, and the evasion-fixture corpus makes the gate's completeness claim executable instead of prose. TC-NF6-05 is superseded (tombstone — absorbed by TC-NF6-03's wholesale `numpy.random` ban); the stale "69 cases total" in the header paragraph (a v1.0 number three additions old) is corrected to the greped count.

---

## How to read this document

Each requirement gets at least one test case. Every case has a stable ID
(`TC-{REQ-ID}-{NN}`), names the technique it uses, and is written as
Given/When/Then. Cases are marked **executable** (a script can pass/fail it
mechanically) or **judged** (a human or reviewer reads the output and decides
— true for anything about plain-language quality). A case marked
**negative/phantom-gate** exists specifically to prove the gate it's paired
with can actually fail, not just always pass — the review board's own finding
that a gate with no fail-case is a gate that's never been tested.

Where a requirement depends on a number Open Questions 1, 2, or 5 haven't
fixed yet (exception-band thresholds, sample size, runtime budget), the case
is written against the *property* the number must satisfy, with the number
itself left as `<spec-value>` — filled in once the technical spec sets it,
never guessed here.

**Coverage discipline (no silent caps):** every Must requirement gets at
least one case below. 79 case IDs total, of which one (TC-NF6-05) is a
superseded tombstone, leaving 78 live cases — count verified by
`grep -c '^\*\*TC-'`, not carried forward by arithmetic (the previous "69"
here was exactly such a carried-forward number, three additions stale).
Depth varies deliberately: single
cases for straightforward requirements, two to three for the requirements
this project's own risk assessment or the review board flagged as
credibility-fatal (fixtures never-findings, pre-registration, the blind
judge, calibration honesty, exception counting, and every place a prior
finding showed a "phantom gate" was possible). Read the traceability matrix
at the end to see the full map; nothing is silently dropped.

Wont's (WD-8, MU-10) get no case — they describe what's deliberately absent,
not a behaviour to verify. JU-13 (no learned component) is the one
Won't that *is* mechanically checkable, so it gets one.

---

## Domain 1 — Worlds

**TC-WD1-01** · scenario test · executable
Given the Lotka–Volterra world and the double pendulum world, each with a declared initial state,
When the world's transition function is called with no action,
Then both return a next state computed from the known closed-form/numerically-integrated equations, matching a hand-verified reference value to floating-point tolerance.

**TC-WD2-01** · scenario test · executable
Given any world,
When its transition function is called with a state and an action,
Then it returns a next state that differs from the no-action case whenever the action is non-null — proving the interface is genuinely `(state, action) → next_state`, not a disguised `(state) → next_state` forecaster.

**TC-WD3-01** · scenario test · executable
Given ground truth and a model under test configured to run,
When both are advanced one step from the same state,
Then an automated check confirms both used the identical integrator class and step size (not merely "close" values).

**TC-WD3-02 (negative/phantom-gate)** · boundary/mutation test · executable
Given the WD-3 integrator-match check,
When it is deliberately run against a model configured with a different step size,
Then the check must fail. (If it passes, WD-3's gate is a phantom — this case is the proof it actually inspects something.)

**TC-WD3-03** · scenario test · executable
Given the shared integrator advancing each world's conserved quantity (pendulum energy, LV orbit) over the rollout horizons JU-6 uses,
When drift in that quantity is measured against a declared bound,
Then the drift stays within bound, or the run fails loudly rather than silently producing an untrustworthy climatology reference.

**TC-WD4-01** · scenario test · executable
Given two trajectories from the same world started a declared small distance apart,
When their separation is measured at each step out to the rollout horizon,
Then the result is a curve (separation vs. step), not a single scalar rate — and for Lotka–Volterra specifically, the curve grows roughly linearly rather than exponentially (proving the fix that replaced the old single-"rate" wording actually holds).

**TC-WD4-02** · equivalence partitioning · executable
Given the double pendulum at a low-energy (near-regular) start and a high-energy (chaotic) start,
When WD-4's divergence curve is computed for each,
Then the two curves differ meaningfully — proving the benchmark is regime-dependent, not a single constant misapplied across energy levels.

**TC-WD5-01** · boundary value analysis · executable
Given a world's declared training region (states and actions) and at least one declared out-of-region start,
When an evaluation rollout is launched from a state just inside the boundary and one just outside it,
Then the harness correctly labels each as in-region / out-of-region.

**TC-WD5-02** · equivalence partitioning · executable
Given a model trained only on a declared action range,
When it is evaluated on a state inside the training region but with an action outside the trained action range,
Then the harness flags this as an out-of-region evaluation on the *action* axis, not silently treated as in-region because the state was familiar.

**TC-WD6-01** · scenario test · executable
Given a world's declared task list,
When inspected,
Then at least two tasks exist — one control, one planning — and their error tolerances are quantitatively different (tight ≠ loose), not two tasks with the same number relabeled.

**TC-WD6-02** · boundary value analysis · executable
Given the tight-tolerance control task's target band,
When a trajectory value sits exactly on the band's edge,
Then the pass/fail classification at that exact boundary is deterministic and documented (not an off-by-one ambiguity).

**TC-WD7-01** · scenario test · executable
Given a fixed seed,
When a world's trajectory is generated twice, in two separate processes,
Then the two outputs are byte-identical.

**TC-WD7-02 (negative/phantom-gate)** · mutation test · executable
Given the WD-7 determinism check,
When a deliberately unseeded random call is injected into the world's step function,
Then the byte-identical check must fail. (Same phantom-gate proof as WD-3.)

---

## Domain 2 — Models Under Test

**TC-MU1-01** · scenario test · executable
Given any model under test,
When it predicts a next state,
Then the returned uncertainty is in the single declared fixed format (per-dimension mean + spread, per MU-1's pre-registered choice) — not a format that varies by model.

**TC-MU1-02** · equivalence partitioning · executable
Given two different models under test,
When each returns its uncertainty,
Then both use identical units and dimensional structure, so JU-4/JU-8 can compare them without a conversion step.

**TC-MU1-03** (added post-design-review) · scenario test · executable
Given a model instance run through one rollout with a given action sequence, then `reset()`, then run through a second rollout with a different action sequence,
When the second rollout's predictions are compared against a fresh instance of the same model run on the same second action sequence,
Then the two match exactly — proving `reset()` actually clears rollout-local state rather than silently leaking it across rollouts, which would corrupt JU-8's independent-trials assumption undetected by any other case in this document.

**TC-MU2-01** · scenario test · executable
Given a model's verdict computation,
When the persistence and linear-extrapolation baselines are missing from the input,
Then the judge refuses to compute a verdict rather than silently proceeding without the comparison (the GVM "refuse when required evidence is missing" pattern, applied directly to MU-2's own "no verdict shall be issued without comparing" wording).

**TC-MU2-02 (supporting)** · scenario test · executable
Given the two baselines run on both worlds,
When their one-step skill scores are computed,
Then both baselines score below any competent model under test — a sanity floor confirming "beats nothing-changes" is a real, non-trivial bar. (Validates the baselines are non-trivial; primary coverage of MU-2's own "no verdict without comparing" clause is TC-MU2-01.)

**TC-MU3-01** · scenario test · executable
Given the overconfident-but-accurate fixture,
When it is judged,
Then its calibration score fails (its exception rate exceeds what its stated confidence implies) despite a good raw-accuracy score — the fixture doing its one job.

**TC-MU3-02** · scenario test · executable
Given the honest-but-less-accurate fixture,
When it is judged,
Then its calibration passes even though its raw accuracy is worse than MU3-01's fixture — proving ranking by accuracy alone would invert the correct order.

**TC-MU3-03** · scenario test · executable
Given the in-region-good/out-of-region-catastrophic fixture,
When judged separately in-region and out-of-region (per WD-5),
Then its in-region score is good and its out-of-region score is bad — proving the region split (JU-4) is what surfaces this failure, not the aggregate score.

**TC-MU4-01** · scenario test · executable
Given any fixture model's output — in code comments, in the verdict record, and on a rendered chart,
When each surface is inspected,
Then the fixture label appears on all three, not just one (RP-8's "travels with the image" requirement, checked at every surface named in MU-4).

**TC-MU5-01** · boundary value analysis · executable
Given the two unrigged models' one-step skill scores, per task and region,
When their difference is measured against the pre-registered accuracy-matching margin (MU-6),
Then the difference is at or under that margin, or the pre-registration is treated as not yet satisfied and judging does not proceed.

**TC-MU5-02** · scenario test · executable
Given the ensemble model's declared point-prediction rule (mean vs. designated member) and its spread-to-confidence-range mapping,
When the ensemble predicts,
Then the point prediction and the confidence range both follow exactly the pre-registered rule — checked against the written recipe, not against what "seems reasonable" at judging time.

**TC-MU5-03** · scenario test · executable
Given the ensemble's raw member spread,
When it is converted to a stated confidence range,
Then the conversion includes the pre-registered small-ensemble underdispersion correction — proving JU-8's exception count for the ensemble isn't inflated by an uncorrected, artificially narrow range.

**TC-MU6-01** · scenario test · executable
Given the repository's commit history,
When the commit timestamp of the MU-6 recipe-and-prediction document is compared to the commit timestamp of the first judged run against an unrigged model,
Then the recipe/prediction commit strictly precedes the first judging run — mechanically enforced, not asserted in prose (the same fix pattern as WD-3/JU-11).

**TC-MU6-02** · scenario test · judged
Given the pre-registered prediction of which unrigged model ranks better,
When the actual judged result comes in either matching or contradicting that prediction,
Then both outcomes are published unchanged and neither triggers a re-run, re-tuning, or quiet edit to the prediction document.

**TC-MU7-01** · scenario test · executable
Given a model's training data and its evaluation rollouts,
When the initial conditions of every evaluation rollout are checked against every training initial condition,
Then no evaluation rollout starts from a state used in training, for either world.

**TC-MU8-01** · scenario test · executable
Given a fixed training seed,
When a model is trained twice from that seed,
Then the two resulting models are identical (and therefore produce identical verdicts).

**TC-MU9-01** (rewritten design-review-004: mechanism v3) · scenario test · executable
Given a stub model module created in a pytest `tmp_path` directory that is grafted onto the models package by appending to `wmj.models.__path__` via monkeypatch — undone automatically on teardown pass or fail, with registry state snapshot-and-restored, so the real production directory is never written to and no registration leaks between tests (Round 4 found the v1.3 version wrote its stub into the real `wmj/models/`, where its own designed-to-fail case could leave a phantom model behind for a later real `wmj run` to discover and execute),
When (a) `importlib.invalidate_caches()` is called and `wmj.models.registry.all_models()` is invoked, (b) the roster-construction function of the **actual production code path**, `wmj.harness.trials`, is invoked directly, and (c) `wmj/judge`, `wmj/worlds`, `wmj/reporting`, and `wmj/harness` are AST-checked by the same import-allowlist technique as the NF-6 gate for any import of a `wmj.models` submodule other than `registry` and `base` (mechanical, no string-matching to evade — Round 4 found the v1.3 "reference to the stub's name" grep carried exactly the string-evasion weakness the NF-6 gate's own four-round history proves fatal),
Then the stub appears in both (a)'s and (b)'s rosters with no other file edited — proving the harness genuinely delegates to the registry rather than holding a second list that happens to agree today — and (c) finds zero forbidden imports (referencing the seven pinned model *name strings* as data is explicitly permitted: names are the sanctioned join keys, imports are the forbidden coupling). Cross-cutting spec v1.4 ADR-003; each of (a), (b), (c) can genuinely fail.

**TC-MU9-02** (negative/phantom-gate; rewritten design-review-004 to match mechanism v3) · mutation test · executable
Given the TC-MU9-01 gate,
When a deliberately injected import of a concrete `wmj.models` submodule (not `registry` or `base`) is added to a scratch copy of a file under `wmj/judge/` (the exact coupling MU-9 forbids),
Then the gate's part (c) import-allowlist check must fail. (Same phantom-gate proof as WD-3/WD-7/NF-1/JU-9/JU-11 — Round 3 found TC-MU9-01 had no such pairing and, under the git-diff mechanism, could not have one.)

---

## Domain 3 — The Judge

**TC-JU1-01** · scenario test · executable
Given the judge's input type,
When it is inspected,
Then it structurally cannot carry model identity, architecture, or training history — not merely "isn't passed one" by convention (the same enforcement-by-type pattern the review board recommended over enforcement-by-promise).

**TC-JU1-02** · property-based test (permutation invariance) · executable
Given two models with identical predictions/uncertainties but swapped labels,
When each is judged,
Then the resulting verdicts are identical apart from the label — proving blinding actually holds under the one attack that would break it.

**TC-JU2-01** · scenario test · executable
Given a model's one-step predictions,
When the judge reports accuracy,
Then the report is a skill score relative to both baselines (e.g. "beats persistence by X%"), and no code path emits a raw absolute-error number without an accompanying skill score.

**TC-JU3-01** · scenario test · executable
Given a model's error at each rollout step and the world's WD-4 divergence curve,
When JU-3's error-vs-horizon report is generated,
Then the model's error curve and the WD-4 reference curve are plotted on the same axes with the same units, so "error above the reference" is directly readable.

**TC-JU4-01** · scenario test · executable
Given a model's stated confidence ranges,
When JU-4's calibration diagnostic is computed,
Then it reports observed coverage (fraction of outcomes actually inside the stated range) separately in-region and out-of-region — and this is a distinct number from the JU-4(b) skill-summary score, never conflated into one.

**TC-JU4-02** · property-based test (strictly proper scoring) · executable
Given a model that reports its true predictive distribution vs. a model that reports a deliberately misstated (over- or under-confident) version of the same distribution,
When both are scored with JU-4(b)'s scoring rule,
Then the true-distribution report scores no worse than the misstated one, for randomly sampled distributions (the anti-gaming property the whole calibration approach depends on).

**TC-JU5-01** · scenario test · executable
Given two calibrated models, one with tight confidence ranges and one with deliberately wide ("always safe") ranges,
When JU-5's sharpness score is computed,
Then the wide-range model scores worse on sharpness despite equal or better calibration — proving hedging doesn't win.

**TC-JU5-02** · property-based test · executable
Given any single model's stated range,
When the range is artificially widened while holding the true outcome distribution fixed,
Then sharpness score strictly decreases (monotonic in range width) — the property JU-5 exists to guarantee.

**TC-JU6-01** · state transition test · executable
Given a task's declared tolerance and the world's WD-4 divergence curve,
When rollout length crosses the step at which divergence exceeds tolerance,
Then the judge's grading mode switches from trajectory-level to statistical-agreement, and the report states the exact step at which the switch happened.

**TC-JU6-02** · scenario test · executable
Given a rollout that crosses from one energy shell / conserved-orbit value to another mid-rollout (via an action or via measured integrator drift),
When JU-6's conditioned climatology is computed past the switch point,
Then the invariant used for conditioning is re-measured from the true trajectory at each comparison point, not held at its value from the start of the rollout.

**TC-JU7-01** · scenario test · executable
Given a computed trust horizon,
When it is reported anywhere in the verdict record or on a chart,
Then it always appears with its task name, its region, and tolerance attached (design-review-002 update: region added — judge spec v1.2 §5), and never as a bare number.

**TC-JU7-02** · scenario test · executable
Given a trust horizon computed in steps,
When it is reported,
Then it is also reported in the world's physical time units (or as a fraction of the world's natural cycle) alongside the step count.

**TC-JU8-01** · boundary value analysis · executable
Given a set of rollouts from a single continuous trajectory,
When JU-8's exception count is computed,
Then correlated in-rollout steps are never counted as independent trials — only the pre-declared independent-trial set (separate starts) contributes to the count. Feeding it a single long rollout instead of independent trials must be rejected, not silently accepted.

**TC-JU8-02** · scenario test · executable
Given a model with a known true confidence level (e.g. a fixture engineered to be exactly 95%-calibrated) run over `<spec-value>` independent trials,
When JU-8's band assignment runs,
Then the model lands in the "expected/green" band at the rate its own false-alarm probability at that sample size implies (checked against JU-11's declared statistical test, once Open Question 1/2 fix the sample size and confidence level).

**TC-JU8-03 (supporting)** · scenario test · executable
Given a model with padded, overly wide confidence ranges (near-zero exception rate),
When JU-8 runs,
Then it is flagged as a fault via the sharpness cross-check (JU-5), not silently rewarded for having "too few" exceptions. (Confirms JU-8 and JU-5 are wired together correctly; primary coverage of JU-5's own sharpness property is TC-JU5-01/02.)

**TC-JU9-01** · scenario test · executable
Given a completed judging run,
When the verdict record is produced,
Then it contains all eight required field groups (design-review-002 update: `trials` and `climatology` were added to the Verdict schema in v1.1 but omitted from this list until Round 2 found the gap — judge spec v1.2 §5) — skill scores, error-vs-horizon with divergence benchmark, calibration+sharpness in/out of region, exception counts vs. thresholds, per-trial outcome/band/exception data, conditioned-climatology agreement, per-task-and-region trust horizons, not-tested list — in one structured record.

**TC-JU9-02 (negative/phantom-gate)** · mutation test · executable
Given a judging run where one required field (e.g. the calibration data) cannot be computed,
When the verdict record is assembled,
Then the run aborts rather than emitting a partial record with that field silently missing or null. (Proof that "the run fails rather than emitting a partial record" actually fails when it should — the same phantom-gate discipline applied to WD-3/WD-7/NF-1.)

**TC-JU9-03** (added post-design-review-003) · property-based test · executable
Given any computed Verdict,
When, for every (task, region, horizon_step) key present, `exceptions.per_task[key].observed` is compared against `sum(t.is_exception for t in trials.per_task if same key)`,
Then the two are always equal — proving `exceptions.per_task.observed` is genuinely derived from `trials.per_task.is_exception` rather than two independently-computable definitions that happen to usually agree (design-review-002 named the correct relationship; design-review-003 found no test enforced it).

**TC-JU10-01** · scenario test · executable
Given any verdict,
When its limitations statement is inspected,
Then all seven required disclosures are present: middle-of-distribution-only validity, toy-not-field, same-author blind-not-independent, borrowed-mechanism-not-full-SR-11-7, thresholds-are-modelling-choices, no-genuine-off-model-surprises-by-construction (WD-8), and passive-prediction-not-action-selection (the LeCun-review finding).

**TC-JU10-02** · scenario test · judged
Given the limitations statement,
When read by someone unfamiliar with the project,
Then they correctly identify that the judge and the models share an author, without needing to read JU-1 separately.

**TC-JU11-01** · scenario test · executable
Given the committed thresholds/bands file, committed before the first judging run against an unrigged model,
When that run executes,
Then it reads the thresholds from the committed file and proceeds normally (mirrors TC-MU6-01's mechanism, applied to JU-11 specifically).

**TC-JU11-02 (negative/phantom-gate)** · mutation test · executable
Given a thresholds/bands file edited or committed *after* a judging run has already occurred,
When that run's validity is checked,
Then the run is flagged invalid / the check refuses to certify it — proof the "fixed before judging" requirement can actually catch a violation, not just describe one.

**TC-JU12-01** · property-based test (purity) · executable
Given the judge's full call graph,
When it is run under a monkeypatched/blocked environment (no filesystem, no clock, no RNG access),
Then it still produces the correct verdict from its passed-in arguments alone — no I/O access anywhere in the call graph.

**TC-JU13-01** · scenario test · executable
Given the judge's source code and its dependency list,
When both are inspected,
Then no learned/trained component (no model weights, no inference call) exists anywhere in the judge's own code path — confirmed by static inspection, not by trusting the description.

---

## Domain 4 — Showing the Result

**TC-RP1-01** · scenario test · executable
Given a model's predictions, actual outcomes, and JU-8's exception count,
When the backtesting exception plot is rendered,
Then it shows the expected exception count (from stated confidence) next to the actual count, with the resulting JU-8 band named on the chart itself — not just circled misses with no reference rate.

**TC-RP2-01** · scenario test · executable
Given error-vs-horizon data and WD-4's divergence curve,
When the chart is rendered,
Then the axis scale keeps the early-horizon gap between the two curves visually legible (not compressed flat by a late-horizon blow-up), and the caption states the reading rule ("only the gap above the reference line is the model's fault").

**TC-RP3-01** · scenario test · executable
Given calibration data in-region and out-of-region,
When the calibration chart is rendered,
Then the perfect-calibration diagonal is drawn and labelled, and the caption is phrased as a natural frequency ("about 90 of every 100... ") rather than a bare percentage.

**TC-RP4-01** · scenario test · executable
Given the ordinary-error ranking and the trust-horizon ranking for two models that disagree,
When the comparison table is rendered,
Then the disagreeing row is visually marked and the caption states the disagreement in one plain sentence — the project's actual headline result made impossible to miss.

**TC-RP4-02** · scenario test · executable
Given two models whose ordinary-error and trust-horizon rankings agree,
When the table is rendered,
Then no row is marked as disagreeing — proving the marking logic isn't just always-on decoration.

**TC-RP5-01** · scenario test · judged
Given any of the four required charts,
When its caption is read in isolation, with no other context,
Then a reader with no risk background can state what the chart shows, how to read it, and what to conclude — in three sentences or fewer of caption text.

**TC-RP6-01** · scenario test · executable
Given a completed verdict,
When the output directory is inspected,
Then a structured machine-readable file (matching JU-9's schema) exists alongside the rendered charts.

**TC-RP7-01** · scenario test · executable
Given a clean checkout of the repository,
When the single documented command is run,
Then every chart and every verdict number is regenerated from scratch with no manual steps, on an ordinary laptop.

**TC-RP8-01** · scenario test · executable
Given a chart that includes any fixture model's output,
When the chart image is viewed on its own, with no surrounding caption or documentation,
Then the fixture label is visible on the image itself.

---

## Non-Functional Requirements

**TC-NF1-01** · scenario test · executable
Given identical inputs and seed,
When the judge runs ten consecutive times,
Then the serialized verdict is byte-identical across all ten runs — the whole record compared, not selected fields.

**TC-NF1-02 (negative/phantom-gate)** · mutation test · executable
Given the NF-1 byte-identity check,
When a deliberately unseeded floating-point operation (e.g. unordered parallel summation) is injected into the judge,
Then the byte-identity check must fail. (Same phantom-gate proof as WD-3/WD-7 — this is the requirement the project's own Assumption 5 flags as most likely to bite, so it gets the same negative-case discipline.)

**TC-NF2-01** · scenario test · executable
Given the full result set,
When run on an ordinary laptop and timed with a wall-clock counter,
Then elapsed time is under `<spec-value>` (Open Question 5) — a mechanical bound check, unbuildable only until the technical spec fixes the number.

**TC-NF3-01** · scenario test · executable
Given the project's dependency manifest,
When compared against `<spec-value: the technical spec's named minimal package list>`,
Then no dependency outside that named list is present. (Unbuildable until the technical spec declares the list — flagged here rather than assumed.)

**TC-NF4-01** · scenario test · executable
Given the full repository text, its commit-message history, **and every historical file-content blob across every commit** (design-review-002 update: cross-cutting spec's scan mechanism covers `git log` as well as tracked file content; design-review-003 update: Round 3 found the identical permanence argument that motivated the commit-message scan applies to file content too — a term committed and later removed from a file is exactly as permanently visible via `git show` as one in a commit message — so historical blobs are now scanned the same way),
When scanned for a maintained list of forbidden terms (employer name, internal team/committee names — the list itself declared and kept current in the technical spec, in a gitignored local file per the mechanism, never in a tracked one),
Then no match is found — the one place this project's honesty requirements have real, checkable teeth on a public, first-commit repo.

**TC-NF4-03** (added post-design-review-003) · scenario test · executable
Given a fresh clone of the repository with the one-time `git config core.hooksPath .githooks` setup step never run,
When a commit is made containing a term listed in the (test-only, non-real) local terms file,
Then `python -m wmj run`/`verify`'s own re-run of the scan (P6-C02) still catches it at run time, even though the commit-time pre-commit hook was never active — proving the run-time check is a real, non-hook-dependent safety net, not merely a disclosed gap with no mitigation.

**TC-NF4-02** · scenario test · judged
Given every passage describing banking practice,
When read by someone with banking-industry knowledge,
Then each description is generic and traceable to a publicly documented source (SR 26-2, SS1/23), not phrased in a way that reads as insider or firm-specific detail — NF-4's second clause, which TC-NF4-01's keyword scan cannot catch on its own.

**TC-NF5-01** · scenario test · judged
Given any published output (chart, caption, verdict record, or prose),
When read against what the underlying data actually shows,
Then no claim exceeds what the evidence supports, and every place the harness cannot demonstrate something says so rather than omitting it.

**TC-NF5-02** (added post-design-review-003) · scenario test · executable
Given a committed `specs/*.html` or `requirements/*.html` file and its `.md` twin,
When the `.md` file's current SHA-256 hash is compared against the hash embedded in the `.html` file's `<!-- generated-from: ... -->` comment,
Then the two match — proving the `.html` a human might read is genuinely generated from the exact `.md` a human approves, not a stale or hand-edited copy (design-review-002 introduced this mechanism; design-review-003 found it had no test case, no requirement trace, and no build chunk, and traced it here to NF-5's no-overclaiming principle).

**TC-NF6-01** (rewritten design-review-004: denylist → allowlist) · property-based test (import graph) · executable
Given the judge package's AST,
When every `Import`/`ImportFrom` node's top-level module is checked against the allowlist **`{numpy, math, dataclasses, typing}`**,
Then any import outside the allowlist fails the gate — judge.md §4's own "nothing else" claim, enforced as stated. (This subsumes, with nothing left to enumerate, every prior import-shaped denylist check — the `wmj.*` packages, the twelve ambient modules, `importlib`, `random`, `builtins`, and every module nobody thought to list. Four review rounds each evaded the then-current denylist within one round; an allowlist cannot have the omission problem. NF-6's original "no imports from worlds/models/reporting" is a strict subset of this check.)

**TC-NF6-02** (rewritten design-review-004) · property-based test (AST identifiers) · executable
Given the judge package's AST,
When scanned for the identifiers `exec`, `eval`, `compile`, `__import__`, `__builtins__`, `globals`, or `vars` appearing **anywhere** — as a `Name` id, an `Attribute` attr, or an import alias, not only as a direct `Call` target,
Then none is found. (The call-shape scoping was Round 4's confirmed evasion route: `e = eval; e(...)` and `__builtins__.eval(...)` both slipped the v1.3 wording. The judge is pure arithmetic; none of these names has a legitimate use in it. Combined with TC-NF6-01, the realistic indirection routes close: `getattr(builtins, "eval")` needs `import builtins`, which fails the allowlist; `getattr(__builtins__, "eval")` names `__builtins__`, which fails here.)

**TC-NF6-03** (rewritten design-review-004) · property-based test (import graph) · executable
Given the judge package's AST,
When scanned for any `Import`/`ImportFrom` whose dotted module path begins `numpy.random`, and any `Attribute` node whose attr is `random`,
Then none is found — `numpy.random` is banned **wholesale** in the judge, which computes deterministic metrics and needs no randomness at all (JU-12). (The previous Generator/PCG64 exception implied otherwise and forced a name-by-name pattern that `from numpy.random import rand` and `import numpy.random as npr` both walked straight past; a total ban has no pattern to evade. The bare-`attr == "random"` rule is deliberately over-broad; TC-NF6-06 guards the false-positive side.)

**TC-NF6-04 (negative/phantom-gate — the evasion-fixture corpus)** (rewritten design-review-004) · mutation test (fixture corpus) · executable
Given one fixture file under `tests/gates/fixtures/` for every concrete evasion any review round has documented — bare/aliased/from-imports of forbidden modules; `importlib` aliasing; `getattr` indirection; `__builtins__` attribute routes; `exec`/`eval`/`compile` payloads; both `numpy.random` import idioms; identifier rebinding,
When the full gate is run against each fixture,
Then it flags **every** fixture. The corpus is the gate's completeness claim — exactly this, no more: a newly discovered evasion is added as a fixture, turning "is the gate complete?" from a prose argument (which failed four consecutive reviews) into an executable, growing regression suite. Static analysis of Python cannot be proven complete (LANGSEC's own point, disclosed rather than implied away); this is the honest, checkable form of the claim, and it is also the gate family's can-actually-fail proof.

**TC-NF6-05 — superseded (design-review-004)** · tombstone · not executable
Former content (the `<numpy-alias>.random.<name>` attribute pattern) is absorbed and strengthened by TC-NF6-03's wholesale ban. The ID is retained per this document's append-only discipline so no ID silently vanishes; it appears in no tally except the total.

**TC-NF6-06 (clean-pass / false-positive guard)** (rewritten design-review-004) · scenario test · executable
Given the real, legitimate judge source,
When the full gate (TC-NF6-01 through -03) runs against it,
Then it passes — necessary because TC-NF6-02 and TC-NF6-03 are deliberately over-broad, and a gate that cries wolf gets disabled by the people it inconveniences.

---

## Traceability Matrix

| Requirement | Test Cases | Requirement | Test Cases |
|---|---|---|---|
| WD-1 | TC-WD1-01 | JU-2 | TC-JU2-01 |
| WD-2 | TC-WD2-01 | JU-3 | TC-JU3-01 |
| WD-3 | TC-WD3-01, -02, -03 | JU-4 | TC-JU4-01, -02 |
| WD-4 | TC-WD4-01, -02 | JU-5 | TC-JU5-01, -02 |
| WD-5 | TC-WD5-01, -02 | JU-6 | TC-JU6-01, -02 |
| WD-6 | TC-WD6-01, -02 | JU-7 | TC-JU7-01, -02 |
| WD-7 | TC-WD7-01, -02 | JU-8 | TC-JU8-01, -02, -03 |
| WD-8 | — (Won't; nothing to verify) | JU-9 | TC-JU9-01, -02, -03 |
| MU-1 | TC-MU1-01, -02, -03 | JU-10 | TC-JU10-01, -02 |
| MU-2 | TC-MU2-01, -02 | JU-11 | TC-JU11-01, -02 |
| MU-3 | TC-MU3-01, -02, -03 | JU-12 | TC-JU12-01 |
| MU-4 | TC-MU4-01 | JU-13 | TC-JU13-01 |
| MU-5 | TC-MU5-01, -02, -03 | RP-1 | TC-RP1-01 |
| MU-6 | TC-MU6-01, -02 | RP-2 | TC-RP2-01 |
| MU-7 | TC-MU7-01 | RP-3 | TC-RP3-01 |
| MU-8 | TC-MU8-01 | RP-4 | TC-RP4-01, -02 |
| MU-9 | TC-MU9-01, -02 | RP-5 | TC-RP5-01 |
| MU-10 | — (Won't; nothing to verify) | RP-6 | TC-RP6-01 |
| JU-1 | TC-JU1-01, -02 | RP-7 | TC-RP7-01 |
| | | RP-8 | TC-RP8-01 |
| NF-1 | TC-NF1-01, -02 | NF-4 | TC-NF4-01, -02, -03 |
| NF-2 | TC-NF2-01 | NF-5 | TC-NF5-01, -02 |
| NF-3 | TC-NF3-01 | NF-6 | TC-NF6-01, -02, -03, -04, -06 (-05 superseded tombstone) |

**Every Must and the one mechanically-testable Won't (JU-13) has at least one case. Zero orphan requirements, zero orphan cases.** WD-8 and MU-10 are the two Won'ts with nothing to verify by design (they describe absence, not behaviour).

**Totals:** 79 test cases across 45 requirements (70 after design-review-001/002's additions; +9 in this pass, added after `/gvm-design-review` design-review-003 — Round 3, dual/blind — found: TC-MU9-02, a phantom-gate pairing TC-MU9-01 could not previously have, since its prior git-diff mechanism was structurally incapable of failing; TC-JU9-03, a property test enforcing the `exceptions.observed`/`trials.is_exception` invariant judge.md's prose only asserted; TC-NF6-02 through -06, splitting the single `TC-NF6-01` into one ID per AST-gate check plus a phantom-gate pairing, after the gate grew from one check to five across two rounds with no corresponding test-ID growth; TC-NF4-03, proving the run-time confidentiality scan is a real safety net independent of whether the pre-commit hook was ever installed; TC-NF5-02, covering the previously-orphaned `.md`/`.html` spec-parity hash check. Count independently verified via `grep -c '^\*\*TC-'` = 79, not carried forward by arithmetic; **v1.1 (design-review-004 repair): one of the 79, TC-NF6-05, is now a superseded tombstone, so 78 cases are live**). 7 negative/phantom-gate cases (TC-WD3-02, TC-WD7-02, TC-NF1-02, TC-JU9-02, TC-JU11-02, TC-MU9-02, TC-NF6-04 — the last replacing TC-NF6-06 in this list under v1.1's restructure: the evasion-fixture corpus is now the gate family's can-actually-fail proof, while TC-NF6-06 is its complementary clean-pass/false-positive guard). 8 property-based cases (TC-JU1-02, TC-JU4-02, TC-JU5-02, TC-JU12-01, TC-NF6-01, TC-NF6-02, TC-NF6-03, TC-JU9-03). 5 cases marked *judged* rather than *executable* (TC-MU6-02, TC-JU10-02, TC-RP5-01, TC-NF4-02, TC-NF5-01 — plain-language and human-comprehension checks that genuinely need a reader, not a script). 3 cases explicitly pending an Open Question being fixed by the technical spec, marked with `<spec-value>` rather than guessed at here (TC-JU8-02, TC-NF2-01, TC-NF3-01). 2 cases marked *(supporting)* — they validate wiring between two requirements rather than being the primary coverage of either (TC-MU2-02, TC-JU8-03).

---

*Developed using the Grounded Vibe Methodology*
