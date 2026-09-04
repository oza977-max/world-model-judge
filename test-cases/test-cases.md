# World Model Judge — Test Cases

Version 1.5 · Derived from Requirements v1.2 (25 August 2026, post-review-board and post-audit fixes)

> **Change note (v1.5, 31 August 2026 — design-review-008 repair, execution-verified).** Three new cases, each backed by a run proof: TC-NF6-09 (models→judge/reporting import gate — the same undisclosed asymmetry TC-NF6-07/08 closed for worlds/harness); TC-NF1-08 (seed-key non-str rejection — the sibling type-representation collision TC-NF1-07's colon fix left open, executed: `component_key(1,"a") == component_key("1","a")` before the fix); TC-NF1-09 (each `JudgeInput` carries every region a world declares, `n_trials = N × region count` — replaces TC-NF1-06, whose "one region per call" claim Round 8 found contradicted the multi-region `region_labels` design stable since Round 3; TC-NF1-06 tombstoned). Rewrites: TC-NF1-01 now pins ten *separate-process* runs, not ten in-process calls (executed: an in-process cached unseeded-RNG value passed ten "consecutive runs" while varying across processes — a phantom-gate risk for exactly the residual TC-JU12-04 depends on it catching); TC-JU12-04's disclosed residual narrowed (Round 8 executed the harness patched at both module and private-submodule level and found the Round 7 residual's own named example, `Generator(PCG64())` → `getrandom()`, is now actually caught — the residual is `ctypes` and pre-capture only, not both); TC-MU9-01(c)'s clean-pass fixture extended to cover `from wmj.models.base import SeedSource` (a literal join-and-compare implementation false-positived on the harness's own sanctioned import). 100 IDs, 98 live (TC-NF6-05, TC-NF1-06 tombstones). See design-review-008.html.

> **Change note (v1.4, 31 August 2026 — design-review-007 repair, execution-verified).** Eight new cases, each backed by a mechanism run before it was written: TC-NF6-08 (models→harness import gate — the layering the `SeedSource`-in-`models.base` design depends on, previously unenforced); TC-NF1-07 (seed-key colon rejection — the join is not injective across a `:`); TC-NF1-05/TC-NF1-06 (orchestration loop, cross-cutting ADR-004: paired eval starts across models, and one region per `JudgeInput` with n_trials=N — the harness-partitioning gap open since Round 5); TC-MU6-04 (prereg content-invariance — an honestly-dated second commit tuning the recipe now fails, executed with git); TC-MU6-05 (is_baseline is prereg-exempt but not a comparator); TC-JU12-04 (purity-harness disclosed residual: `ctypes`/`getrandom` uncatchable in-process, byte-identity is the backstop — BC-4); TC-RP7-02 (SVG byte-identity across processes needs `svg.hashsalt`+`metadata Date:None`, executed). Rewrite: TC-MU9-03 pinned to `cli.main(["list-models"])` so the argv→dispatch routing is exercised (Round 7 showed the "entry function" wording passes even with a broken subcommand table). 97 IDs, 96 live (TC-NF6-05 tombstone). See design-review-007.html.

> **Change note (v1.3, 31 August 2026 — design-review-006 repair, execution-verified).** The enforcement-mechanism cases now match mechanisms that were *run* before being written: TC-JU12-01 corrected to mutate-in-place (the `sys.modules` swap was executed failing), with TC-JU12-02 confirmed to fire and TC-JU12-03 added for numpy's C clock; TC-NF4-01 corrected to `--batch` (the `--batch-check` form emits no content — executed) with a plant-and-detect can-fail proof; TC-NF1-03 verified and TC-NF1-04 (two-implementation `component_key` convergence) added; TC-MU4-02 now rebuilds `direct` via `seeds.rng_for` (executed bit-identical); TC-MU9-03 made buildable via in-process `list-models`; TC-MU6-03 (per-model prereg) and TC-MU7-02 (seed-purpose independence) added; TC-JU9-01 now lists nine mandatory groups (adds `limitations`). Four new cases (TC-JU12-03, TC-NF1-04, TC-MU6-03, TC-MU7-02): 89 IDs, 88 live. See design-review-006.html.

> **Change note (v1.2, 31 August 2026 — design-review-005 repair session).** Six new cases wired to their chunks: TC-NF6-07 (models→worlds import gate — the claim the WorldContext ADR was built on, previously unenforced), TC-MU9-03 (the CLI path routes through the registry), TC-JU12-02 (runtime-purity phantom-gate — a `__globals__`-routed `os` read the guard must catch), TC-NF1-03 (seed derivation is content-addressed — no shift on roster change), TC-MU4-02 (fixture inner net weight-identical to `direct`), TC-RP-CARD-01 (the model card is rendered). Rewrites: TC-JU12-01 specified as the load-bearing effect-guarding purity control (the static AST lint no longer claims completeness — Round 5 executed reflection escapes it cannot see); TC-WD3-01 corrected to assert shared-ground-truth-integrator + dt-alignment (not the impossible "model used the identical integrator class"); TC-JU4-01 and TC-MU3-03 updated to the region-key shape and the both-axes `fx-brittle` scenario; TC-MU9-01 teardown clears `sys.modules` and force-registers before snapshot, part (c) joins module+name; TC-NF6-02/04 reframed as lint (completeness disclaimed). Three `<spec-value>` placeholders filled from the settled specs (N=200, 600 s, `{numpy, matplotlib}`). 85 IDs, 84 live (TC-NF6-05 tombstone). See design-review-005.html.

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
least one case below. 103 case IDs total, of which two (TC-NF6-05, TC-NF1-06) are
superseded tombstones, leaving 101 live cases — count verified by
`grep -c '^\*\*TC-'`, not carried forward by arithmetic (the previous "100"/"97"/"89"/"85"/"79"/"69"
here were exactly such carried-forward numbers, stale by later additions; design-review-009
added three — TC-MU2-03, TC-NF6-10, TC-NF6-11 — closing two orphan fail-loud
mechanisms with no phantom-gate case and reporting's missing import gate).
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

**TC-WD3-01** (rewritten design-review-005) · scenario test · executable
Given the world's shared integrator `wmj.worlds.integrator.rk4_step` and every ground-truth generator that must use it — `wmj.harness.benchmarks` (WD-4), training-data generation (models §4), and evaluation-truth generation,
When each generator's integrator call site is inspected and each is advanced one step from the same state,
Then all use the *same* `rk4_step` function object and the same step size `dt` — an identity check on the integrator and dt-alignment (worlds ADR-W1). **This does NOT assert a "model used the identical integrator class" (design-review-005 fix — Round 5 found the v1.1 wording asserted a property worlds.md's own ADR-W1 declares impossible: no model under test calls an integrator at all, since models are direct `(state, action) → Prediction` approximators; the checkable WD-3 property is that every *ground-truth* path shares one integrator, plus the model's `dt` matches the world's, TC-WD3-02 being the mismatched-dt negative).**

**TC-WD3-02 (negative/phantom-gate)** · boundary/mutation test · executable
Given the WD-3 integrator-match check,
When it is deliberately run against a model configured with a different step size,
Then the check must fail. (If it passes, WD-3's gate is a phantom — this case is the proof it actually inspects something.)

**TC-WD3-03** · scenario test · executable
Given the shared integrator advancing each world's conserved quantity (pendulum energy, LV orbit) over the rollout horizons JU-6 uses,
When drift in that quantity is measured against a declared bound,
Then the drift stays within bound, or the run fails loudly rather than silently producing an untrustworthy climatology reference.

**TC-WD4-01** (reworded design-review-009 A6/D3 — executed evidence corrected the growth claim) · scenario test · executable
Given two trajectories from the same world started a declared small distance apart,
When their separation is measured at each step out to the rollout horizon,
Then the result is a curve (separation vs. step), not a single scalar rate — and for Lotka–Volterra specifically, the curve is bounded and grows sub-exponentially over the declared horizon (proving the fix that replaced the old single-"rate" wording actually holds), asserted **two-sided**: the final-to-initial separation ratio stays inside a declared band (neither an exponential runaway above it nor a broken integrator collapsing the twin trajectories together below it). **The previous wording, "grows roughly linearly," was falsified by execution (design-review-009): LV's orbits are neutrally stable, so the curve is genuinely flat over the declared H=700 horizon — the linear phase-drift the original wording named is real but only visible over ~20 cycles (≈7,000 steps), not the declared horizon. A test asserting monotone/linear growth at H=700 would fail on correctly-behaving code — the exact implementability trap this correction closes.**

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

**TC-MU2-03 (negative/phantom-gate)** (new, design-review-009 A8/I6) · mutation test · executable
Given a baseline's spread fit (persistence or linear) presented with a training set whose one-step changes/residuals are constant in some dimension (zero variance) or too few to estimate a sample standard deviation from,
When the fit is attempted,
Then it raises `DegenerateSpreadError` rather than emitting a zero-width or non-finite spread — proving the fail-loud guard actually fires, not merely that it is present in the code (the same phantom-gate discipline applied elsewhere, e.g. TC-WD3-02, to every fail-loud mechanism in this project). Both baselines are covered by the same guard (`ddof=1`, sample std, per ADR-M2/ADR-M3's shared Bessel correction — design-review-009 C1 found an earlier build pinned this for persistence only, leaving linear on the population default with no guard at all; this case locks both).

**TC-MU3-01** · scenario test · executable
Given the overconfident-but-accurate fixture,
When it is judged,
Then its calibration score fails (its exception rate exceeds what its stated confidence implies) despite a good raw-accuracy score — the fixture doing its one job.

**TC-MU3-02** · scenario test · executable
Given the honest-but-less-accurate fixture,
When it is judged,
Then its calibration passes even though its raw accuracy is worse than MU3-01's fixture — proving ranking by accuracy alone would invert the correct order.

**TC-MU3-03** (strengthened design-review-005) · scenario test · executable
Given the `fx-brittle` fixture (in-region-good/out-of-region-catastrophic),
When it is exercised on **all four WD-5 region cases** — state-in/action-in, state-out/action-in, state-in/action-out, state-out/action-out (design-review-005: Round 5 found the v1.1 case only checked the pure in-region and pure out-region ends, never the mixed cases the Round-4 both-axes trigger fix exists for, so an implementer could ship the pre-fix state-box-only trigger and still pass) — and judged per region,
Then only the fully-in-region case (state-in **and** action-in) scores good, and every case where *either* axis is out-of-region scores bad — proving the trigger covers both WD-5 axes (models ADR-M4), and that the region split (JU-4), not the aggregate score, is what surfaces the failure.

**TC-MU4-01** · scenario test · executable
Given any fixture model's output — in code comments, in the verdict record, and on a rendered chart,
When each surface is inspected,
Then the fixture label appears on all three, not just one (RP-8's "travels with the image" requirement, checked at every surface named in MU-4).

**TC-MU4-02 (fixture inner net is weight-identical to `direct`)** (design-review-005; buildable design-review-006) · scenario test · executable
Given a fixture (e.g. `fx-overconfident`) and the registered `direct` model, both built by their factories from the same `(ctx, seeds, training)` in one run,
When the fixture — which builds its inner core via **`seeds.rng_for("direct", …)`** (the `SeedSource` handed to every factory, defined in `wmj.models.base`, so no `models→harness` import — design-review-006 gave the fixture this channel, which Round 6 found the v1.5 wording lacked) — has its inner-network weights compared to `direct`'s trained weights,
Then they are **bit-identical** — Round 6 executed exactly this rebuild (`C7 fixture rebuild == direct weights: True`), proving "a copy of Model A" (models ADR-M4) is genuinely the same trained network, not a second independent training run.

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

**TC-MU6-03 (per-model prereg entry, not just file presence)** (new, design-review-006) · scenario test · executable
Given a newly-registered unrigged model (`not is_baseline and not is_fixture`) whose `name` does **not** appear in the committed `prereg/recipe.md`/`prediction.md`,
When `check_prereg` runs before a judged run,
Then it refuses (the specific model has no pre-registration entry) — closing the gap Round 6 found where the v1.5 check was roster-agnostic (it asserted the `prereg/` files as a whole were committed and pre-dated the run, so a new unrigged model passed as "pre-registered" without any text about it ever being written into `prereg/`). This makes MU-6's "mechanically checkable" true per-model (models ADR-M5).

**TC-MU6-04 (prereg content-invariance since first commit)** (new, design-review-007) · mutation test · executable
Given a committed `prereg/recipe.md` whose *first-added* commit predates any judged run, and a working tree in which that file's content was later changed by an ordinary, honestly-dated second commit (no history rewrite, no `--amend`),
When `check_prereg` runs,
Then it refuses — because `check_prereg` hashes the blob at the first-added commit (`git show <first-add-sha>:prereg/recipe.md`) and asserts it equals the working-tree file's hash (models ADR-M5). Round 7 executed the gap: the order-only check passes a day-1 placeholder → day-243 `matching_margin: 0.049` edit (clean tree, genuine dates), exactly the MU-6 gaming the mechanism exists to prevent; the content-hash turns it into a hard failure. (This does not close the deliberate first-commit-rewrite risk disclosed in ADR-M5 — it closes the un-rewritten drift, which is the likelier accident.)

**TC-MU6-05 (is_baseline is prereg-exempt but not a comparator)** (new, design-review-007) · scenario test · executable
Given a newly-registered model declaring `is_baseline=True` (models ADR-M1),
When `check_prereg` runs and, separately, a judged run assembles the Verdict's `skill` block,
Then (a) `check_prereg` skips it (the `not is_baseline and not is_fixture` classification exempts it, so "adding a baseline is one file" holds for provisioning) **and** (b) it does not appear as a third comparator — the `skill` block still names exactly `vs_persistence` and `vs_linear` (judge §5), the blind judge cannot add a `vs_<newbaseline>` field, and MU-2's two reference baselines are unchanged. This pins the scope of the `is_baseline` flag design-review-007 found overstated: the flag's only consumer is prereg-exemption, not the comparison set.

**TC-MU7-01** · scenario test · executable
Given a model's training data and its evaluation rollouts,
When the initial conditions of every evaluation rollout are checked against every training initial condition,
Then no evaluation rollout starts from a state used in training, for either world.

**TC-MU7-02 (train/eval/benchmark start streams are independent by seed purpose)** (new, design-review-006) · property-based test · executable
Given the content-addressed seeding with distinct `purpose` key parts (`seeds.rng_for(world, region, "train-starts" | "eval-starts" | "benchmark-starts")`, cross-cutting ADR-002 rule 2),
When the first draws of the three streams for the same `(world, "training")` are compared,
Then they differ — Round 6 executed this (`C4 purpose keys differ: True`), proving the `purpose` discriminator prevents the collision Round 6 found in the v1.5 key (which had only {model, world, region, index}, so training starts and evaluation starts keyed to the *same* seed and would coincide, silently defeating MU-7 disjointness and JU-8 independence by construction rather than only by the after-the-fact check TC-MU7-01 performs).

**TC-MU8-01** · scenario test · executable
Given a fixed training seed,
When a model is trained twice from that seed,
Then the two resulting models are identical (and therefore produce identical verdicts).

**TC-MU9-01** (mechanism v3, teardown hardened design-review-005) · scenario test · executable
Given, in this order (design-review-005 — Round 5 executed two teardown leaks: restoring `__path__` leaves the stub in `sys.modules` so a later same-named stub is served the cached module and its `register()` never runs; and a naive registry snapshot taken before the real models register lazily is empty, so teardown wipes the real roster): the test **first calls `all_models()` once to force the seven real models to register, then snapshots the registry dict**, then creates a stub module in a pytest `tmp_path` directory grafted via `wmj.models.__path__` monkeypatch, then `importlib.invalidate_caches()`,
When (a) `all_models()` is invoked and the stub appears; (b) the roster from the **actual production path**, `wmj.harness.trials`'s roster function, is invoked directly and the stub appears; (c) `wmj/judge`, `wmj/worlds`, `wmj/reporting`, `wmj/harness` are AST-checked (import-allowlist, **joining `module + "." + name` for `ImportFrom`** — design-review-005: `from wmj.models import direct` and `from wmj.models import registry` share `node.module`, so a `node.module`-only check waves the forbidden one through) for any import of a `wmj.models` submodule other than `registry`/`base`,
Then the stub appears in both rosters with no other file edited, and (c) finds zero forbidden imports (naming the seven pinned model *strings* as data is permitted). **The (c) clean-pass fixture includes `from wmj.models.base import SeedSource, component_key` alongside `from wmj.models.registry import all_models` (design-review-008 I4 repair — Round 8 executed a false positive: a literal join-and-compare check blocks this legitimate harness import, because `node.module` is already `wmj.models.base` and joining it to `alias.name` produces `wmj.models.base.SeedSource`, not the literal allowed string; the rule is now: when `node.module` itself already equals `wmj.models.registry` or `wmj.models.base`, the check passes on `node.module` alone, no join)** — so this shape has a permanent regression check. **Teardown (a `finally`, pass or fail) restores `wmj.models.__path__`, restores the snapshotted registry exactly, and pops `sys.modules['wmj.models.<stub>']`** so nothing leaks into a later test. Cross-cutting spec v1.8 ADR-003; each of (a), (b), (c) can genuinely fail.

**TC-MU9-02** (negative/phantom-gate) · mutation test · executable
Given the TC-MU9-01 gate,
When a deliberately injected **`from wmj.models import <submodule>`** (the shape a `node.module`-only check misses — design-review-005) into a scratch copy of a file under `wmj/judge/` (the exact coupling MU-9 forbids),
Then the gate's part (c) import-allowlist check must fail. (Phantom-gate proof; the injected shape is specifically the one a naive implementer would let through, so the case exercises the real failure mode.)

**TC-MU9-03 (CLI routes through the registry)** (design-review-005; made buildable design-review-006; entry point pinned design-review-007) · scenario test · executable
Given the grafted-stub setup of TC-MU9-01 (an in-process `wmj.models.__path__` monkeypatch),
When **`wmj.harness.cli.main(["list-models"])` is called in-process** — the top-level argv entry point, so the string `"list-models"` is routed through the CLI's own subcommand-dispatch table to the `list-models` handler, which prints the roster `harness.trials` assembles and exits (the cheap internal command added in P6-C01), and the in-process graft is visible,
Then the stub appears in the printed roster — proving the real CLI dispatch path routes `argv → handler → harness.trials`/the registry, not a separately-maintained list. **Design-review-007 fix:** Round 7 executed the v1.6 wording ("invoke the entry function for `list-models`") and found it passes even when the argv→command mapping is broken (a `list_models`-vs-`list-models` typo in the subcommand table) — because calling the handler function directly never exercises the routing. Pinning the entry point to `cli.main(["list-models"])` (the argv the real `python -m wmj list-models` produces) closes the dispatch-table half of the gap this case exists to close, while staying in-process and cheap. **Design-review-006 context:** the in-process call (not a `python -m wmj` subprocess) is deliberate — a subprocess starts a fresh interpreter that cannot see the graft, and would trip the full-pipeline cost and `ModelContractError`.

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

**TC-JU4-01** (updated design-review-005) · scenario test · executable
Given a model's stated confidence ranges,
When JU-4's calibration diagnostic is computed,
Then it reports observed coverage (fraction of outcomes inside the stated range) as **one entry per (task, region) with explicit `region` key fields** — one for the `"training"` region and one per out-region present — evaluated at the task's single horizon `h_task` (judge ADR-J2); it does **not** encode the region in `_in_region`/`_out_region` field-name suffixes (design-review-005 fix — Round 5 found the v1.1 wording "separately in-region and out-of-region" reintroduced the exact suffix shape the v1.4 canonical keying rule banned; the region is a key, and a world may declare more than one out-region). This coverage is a distinct number from the JU-4(b) skill-summary score, never conflated into one.

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

**TC-JU8-02** (placeholder filled, design-review-005) · scenario test · executable
Given a model with a known true confidence level (e.g. a fixture engineered to be exactly 95%-calibrated) run over **200 independent trials** (judge ADR-J4 settled OQ-1/OQ-2: N=200),
When JU-8's band assignment runs,
Then the model lands in the "expected/green" band at the rate its own false-alarm probability at that sample size implies (checked against JU-11's exact two-sided binomial test at p=0.10 — green band [12, 29], declared false-alarm ≤ 5%, judge ADR-J4). *(Design-review-005: the `<spec-value>` is filled from judge ADR-J4, which settled the sample size the v1.0 case left pending.)*

**TC-JU8-03 (supporting)** · scenario test · executable
Given a model with padded, overly wide confidence ranges (near-zero exception rate),
When JU-8 runs,
Then it is flagged as a fault via the sharpness cross-check (JU-5), not silently rewarded for having "too few" exceptions. (Confirms JU-8 and JU-5 are wired together correctly; primary coverage of JU-5's own sharpness property is TC-JU5-01/02.)

**TC-JU9-01** · scenario test · executable
Given a completed judging run,
When the verdict record is produced,
Then it contains all **nine** required field groups (design-review-002 made it eight after `trials`/`climatology` were found omitted; design-review-006 adds `limitations` as the ninth — Round 6 found judge.md's own enumeration excluded it while reporting.md's model card called it mandatory) — skill scores, error-vs-horizon with divergence benchmark, calibration+sharpness per (task, region), exception counts vs. thresholds, per-trial outcome/band/exception data, conditioned-climatology agreement, per-task-and-region trust horizons, not-tested list, **and the seven `limitations` disclosures** — in one structured record, and `verdict.py` refuses (raises) if any of the nine is missing or null (TC-JU9-02).

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

**TC-JU12-01** (load-bearing purity control; mechanism corrected design-review-006 to mutate-in-place) · property-based test (purity) · executable
Given the judge's full call graph run on a representative real `JudgeInput`,
When it executes inside a context manager that, for every shared ambient capability object, **sets raising attributes on the one real object every holder already references** — `setattr(os, 'system'|'getcwd'|'getenv', raise)`, `os.environ` → a raising proxy, and likewise `builtins.open`/`io.open`, `time`/`datetime` entry points, `socket`, `subprocess`, `numpy.random`'s `Generator`/`default_rng`/legacy globals, **and numpy's own C-level clock `numpy.datetime64`** — restoring every saved attribute in a `finally` (cross-cutting ADR-002 rule 3, judge §4),
Then the judge produces the correct verdict from its passed-in arguments alone and **no guard fires**. Because the guard is on the shared object's own attributes — **not** a `sys.modules` swap (design-review-006: Round 6 *executed* the v1.5 `sys.modules['os'] = guard` wording and proved it never reaches numpy's already-bound `os`, so `np.get_include.__globals__['os'].getcwd()` ran for real; the mutate-in-place fix was then executed and tripped the guard) — an impure operation trips at the point of its *effect* however it was reached (reflection via `__globals__['os']` or `().__class__.__base__.__subclasses__()` ends at the same mutated object). This is the load-bearing control; the static AST gate (TC-NF6-01/02) is fast lint.

**TC-JU12-02 (negative/phantom-gate for the runtime purity harness)** (design-review-005; confirmed executable design-review-006) · mutation test · executable
Given the TC-JU12-01 harness (mutate-in-place),
When a judge function in a scratch copy is mutated to read `os.environ` reached via `some_fn.__globals__['os']` (the exact reflection escape, importing nothing and naming no banned identifier),
Then the harness's guard must fire (`AmbientAccessError`) — Round 6 confirmed the mutate-in-place harness catches this exact route (the v1.5 `sys.modules`-swap harness did NOT, so this phantom-gate would have failed its own assertion; it now passes as designed).

**TC-JU12-03 (numpy C-level clock guard)** (new, design-review-006) · mutation test · executable
Given the TC-JU12-01 harness,
When a judge function in a scratch copy is mutated to call `numpy.datetime64('now')` (numpy's own C clock, which reads wall time without touching Python-level `os`/`time`),
Then the guard must fire — Round 6 executed `np.datetime64('now')` returning real wall-clock time while `sys.modules['time']` was guarded, so `numpy.datetime64` needs its own explicit guard; this case proves that guard is present and can fail.

**TC-JU12-04 (purity-harness disclosed residual — documented limit, not a gate)** (new, design-review-007; residual narrowed design-review-008) · scenario test · judged
Given cross-cutting ADR-002 rule 3's stated residual — narrowed this round: Round 8 executed the harness patched at both module and private-submodule level (`numpy.random._pcg64.PCG64`, `numpy.random._generator.Generator`, etc.) and found the Round 7 disclosure's own named example, an unseeded `Generator(PCG64())` reaching `getrandom()`, is now actually **caught** (construction raises at either binding before any entropy syscall runs). What remains genuinely uninterceptable by object mutation is narrower: **`ctypes`** (already resident via numpy, reaches raw syscalls with no Python object to mutate) and **any capability bound to a local name before the harness installs** (the one guard this structurally applies to is `numpy.datetime64`'s disclosed rebind, cross-cutting ADR-002 rule 3),
When the judge spec §4, cross-cutting ADR-002 rule 3, and the `results.html` methods note are read together,
Then all three state, in plain words, that TC-JU12-01 is a best-effort tripwire and the ten-run byte-identity gate (TC-NF1-01, run across separate processes) is the load-bearing reproducibility backstop for the accidental case — so no reader concludes the purity harness is complete, and none of the three still names the now-closed `getrandom()`-via-`Generator(PCG64())` route as part of the residual. This is a documentation-honesty check (like TC-JU10-02), verifying the residual is disclosed accurately — neither wider nor narrower than what execution actually shows — rather than a mechanism that could pass or fail at runtime.

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

**TC-RP-CARD-01 (model card rendered — limitations and not-tested are first-class)** (new, design-review-005) · judged
Given a completed run's `out/results.html`,
When a reader opens it,
Then the verdict's seven `limitations` disclosures (JU-10 / judge ADR-J7) and its `not_tested` list are present and legible as a first-class section near the top of the page — not only inside `out/verdicts/*.json` (design-review-005 — Round 5 found these two mandatory fields, and the reporting Expert Panel's own Mitchell mandate "the not-tested list rendered as a first-class output, not a footnote," reached no rendered surface Dev actually opens). *Judged, per the plain-language/human-comprehension discipline the other JU-10/RP-5 cases use.*

**TC-RP7-01** · scenario test · executable
Given a clean checkout of the repository,
When the single documented command is run,
Then every chart and every verdict number is regenerated from scratch with no manual steps, on an ordinary laptop.

**TC-RP7-02 (SVG charts are byte-reproducible across processes)** (new, design-review-007) · scenario test · executable
Given the reporting style pinning both `matplotlib.rcParams['svg.hashsalt']` and `savefig(..., metadata={'Date': None})` (reporting ADR-R1),
When the same chart is rendered as SVG in two separate process invocations and the bytes are compared,
Then they are **identical** — Round 7 executed the gap: with `svg.hashsalt` alone the default `<dc:date>` element and element-ID salt still vary between processes, so a sceptic diffing SVG text (ADR-R1's stated purpose) would see spurious differences; both pins together give byte-identity. (Stated as a practical README property; NF-1's formal guarantee still names verdicts + manifest only.)

**TC-RP8-01** · scenario test · executable
Given a chart that includes any fixture model's output,
When the chart image is viewed on its own, with no surrounding caption or documentation,
Then the fixture label is visible on the image itself.

---

## Non-Functional Requirements

**TC-NF1-01** (separate-process requirement pinned, design-review-008 C9) · scenario test · executable
Given identical inputs and seed,
When the judge runs ten times **in ten separate process invocations** (not ten in-process calls),
Then the serialized verdict is byte-identical across all ten runs — the whole record compared, not selected fields. **Why separate processes, not just repetition (executed, design-review-008):** a stand-in judge with a lazily-initialized, process-global cache filled from an unseeded RNG draw on first use passes ten *in-process* "consecutive runs" every time (the cache fills once and is reused for the rest of that process's calls), while the underlying value differs freely across three fresh, separate processes — exactly the class of accidental impurity (NF-1's own "across processes" clause, matching TC-WD7-01's and TC-RP7-02's own separate-process wording) this gate exists to catch and the residual-narrowing in TC-JU12-04 depends on it catching.

**TC-NF1-02 (negative/phantom-gate)** · mutation test · executable
Given the NF-1 byte-identity check,
When a deliberately unseeded floating-point operation (e.g. unordered parallel summation) is injected into the judge,
Then the byte-identity check must fail. (Same phantom-gate proof as WD-3/WD-7 — this is the requirement the project's own Assumption 5 flags as most likely to bite, so it gets the same negative-case discipline.)

**TC-NF1-03 (seed derivation is content-addressed — no shift on roster change)** (design-review-005; verified design-review-006) · property-based test · executable
Given the content-addressed derivation via `SeedSource`/`component_key` (cross-cutting ADR-002 rule 2, `wmj.models.base`),
When the derived seeds for the seven-model roster are computed, then one new model whose name sorts alphabetically *before* existing names is added and the derivation is recomputed,
Then every pre-existing component's derived seed is **byte-identical** across the two computations — Round 6 executed this and it held (`C3 no-shift (add 'aardvark'): True`), while the rejected positional-`spawn` implementation reassigned every seed.

**TC-NF1-04 (component_key is pinned enough that two implementations converge)** (new, design-review-006) · property-based test · executable
Given the pinned `component_key` construction (blake2b of `":".join(str(p) for p in parts)`, `digest_size=8`, one big-endian int as a 1-tuple — cross-cutting ADR-002 rule 2, no "e.g."),
When two independent implementations written from the pinned text alone derive keys for the same `(name, world, region, purpose)` tuples,
Then they produce **identical** keys and therefore identical `SeedSequence` streams — Round 6 executed this (`C3 two-impl convergence: True`), closing the gap where the v1.5 "e.g." wording let four textually-conforming readings diverge (so a sceptic re-deriving a published seed from the spec gets the same bytes — the project's independent-verification mission).

**TC-NF1-05 (paired eval starts are identical across models)** (new, design-review-007) · property-based test · executable
Given the orchestration loop (cross-cutting ADR-004) drawing each (world, region)'s N=200 eval starts from `seeds.rng_for(world, region, "eval-starts")` — a function of the region's identity, not the model's,
When the starts seen by two different models on the same (world, region) are compared,
Then they are **byte-identical** — enforcing JU-8's paired-comparison requirement (every model faces the same trials) by construction. A mutation that keys the eval-start draw on the model name (breaking pairing) makes this fail.

**TC-NF1-06 — superseded (design-review-008)** · tombstone · not executable
Former content ("each JudgeInput carries exactly one region, n_trials = N") described the v1.7 orchestration design, which Round 8 found broken and rewrote: a `JudgeInput` now carries *every* region a world declares, in one call per model, matching the `region_labels` design judge §4/worlds §5 have held since Round 3 — the opposite claim. TC-NF1-09 replaces it. The ID is retained per this document's append-only discipline; it appears in no tally except the total.

**TC-NF1-08 (seed-key non-str rejection — the sibling collision the colon fix left open)** (new, design-review-008) · mutation test · executable
Given `component_key`'s pinned construction (cross-cutting ADR-002 rule 2, updated design-review-008),
When it is called with a part that is not already a `str` — e.g. `component_key(1, "a")` or `component_key("ensemble", 3)`,
Then it raises `SeedKeyError` — Round 8 executed the collision the colon-rejection fix (TC-NF1-07) left open: the old `str(p)` coercion inside `component_key` collapsed `component_key(1, "a")` and `component_key("1", "a")` to the identical joined text and therefore the identical seed stream. Rejecting any non-`str` part at construction closes it the same way TC-NF1-07 closes the colon case; the `str()` conversion, where needed, is now the caller's job (models spec's `seeds.rng("member", str(k), ...)` call sites).

**TC-NF1-09 (each JudgeInput carries every one of its world's regions, n_trials = N × region count)** (new, design-review-008 — replaces TC-NF1-06) · scenario test · executable
Given the rewritten orchestration loop (cross-cutting ADR-004) calling the judge once per (model, world),
When any assembled `JudgeInput` is inspected,
Then its arrays have `n_trials == N × |region_names|` for that world (not `N` alone), and its `region_labels` names every region the world declares at least once — matching judge §4's own rationale for the per-trial `{region_name, axis}` shape ("a boolean flag can't do that once a world declares more than one out-region," which requires a *single* call to carry more than one region) and `JudgedResult`'s "one per (model, world)" contract. Round 8 executed the previous design (TC-NF1-06) and found it contradicted this rationale; a mutation that splits one world's trials across two separate `JudgeInput` calls, or omits a declared region from `region_labels`, fails this case.

**TC-NF2-01** (placeholder filled, design-review-005) · scenario test · executable
Given the full result set,
When run on a 4-core consumer laptop CPU and timed with a wall-clock counter,
Then elapsed time is under **600 seconds** (judge ADR-J6 settled OQ-5's runtime half) — a mechanical bound check. *(Design-review-005: `<spec-value>` = 600 s, from judge ADR-J6.)*

**TC-NF3-01** (placeholder filled, design-review-005) · scenario test · executable
Given the project's dependency manifest,
When compared against the technical spec's named minimal package list — **runtime `{numpy, matplotlib}`, dev `{pytest}`** (cross-cutting ADR-001 / Dependency Budget),
Then no dependency outside that named list is present. *(Design-review-005: the list is filled from cross-cutting's Dependency Budget, which the v1.0 case was pending.)*

**TC-NF4-01** (scan-command corrected design-review-006) · scenario test · executable
Given the full repository text, its commit-message history, **and every object in the local database enumerated via `git cat-file --batch-all-objects --batch`** (design-review-006 repair — Round 6 *executed* the v1.5 `--batch-check` command and found it emits only `<sha> <type> <size>`, never content, so the scan searched nothing and would pass green over any planted term; `--batch` emits each object's full content, which is what the scan greps: executed, `--batch-check` found 0 matches for a planted term, `--batch` found it),
When scanned for a maintained list of forbidden terms (employer name, internal team/committee names — the list itself declared and kept current in the technical spec, in a gitignored local file per the mechanism, never in a tracked one),
Then no match is found. **The mechanism is proven can-fail (design-review-006): a companion check plants a known term into a throwaway commit, runs the exact `--batch` command, and asserts the scan detects it — so a green result means the scan actually looked, not that the command was inert** (the failure mode Round 6 executed against `--batch-check`). This is the one place this project's honesty requirements have real, checkable teeth on a public, first-commit repo.

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

**TC-NF6-02** (rewritten design-review-005 — reframed as lint, list widened) · property-based test (AST identifiers) · executable
Given the judge package's AST,
When scanned for the identifiers `exec`, `eval`, `compile`, `__import__`, `__builtins__`, `globals`, `vars`, `getattr`, `__globals__`, `__subclasses__`, `__bases__`, `__mro__`, or `__class__` appearing **anywhere** — as a `Name` id, an `Attribute` attr, or an import alias,
Then the file is **flagged for review** (a strong lint signal — the judge is pure arithmetic and none of these has a legitimate use in it). **This is fast lint, not a completeness proof, and its identifier list is explicitly not claimed complete (design-review-005 — Round 5 executed evasions past the v1.4 list: `getattr(np, "__builtins__")["eval"]`, string-concatenated identifiers, `().__class__.__base__.__subclasses__()`): no finite identifier list bounds Python reflection, which is why the load-bearing purity control is the runtime harness TC-JU12-01, not this scan.**

**TC-NF6-03** (rewritten design-review-004) · property-based test (import graph) · executable
Given the judge package's AST,
When scanned for any `Import`/`ImportFrom` whose dotted module path begins `numpy.random`, and any `Attribute` node whose attr is `random`,
Then none is found — `numpy.random` is banned **wholesale** in the judge, which computes deterministic metrics and needs no randomness at all (JU-12). (The previous Generator/PCG64 exception implied otherwise and forced a name-by-name pattern that `from numpy.random import rand` and `import numpy.random as npr` both walked straight past; a total ban has no pattern to evade. The bare-`attr == "random"` rule is deliberately over-broad; TC-NF6-06 guards the false-positive side.)

**TC-NF6-04 (negative/phantom-gate — the lint's regression corpus)** (updated design-review-005) · mutation test (fixture corpus) · executable
Given one fixture file under `tests/gates/fixtures/` for every concrete evasion any review round has documented — bare/aliased/from-imports of forbidden modules; `importlib` aliasing; `getattr` indirection; `__builtins__` attribute routes; `exec`/`eval`/`compile` payloads; both `numpy.random` import idioms; identifier rebinding; **and the Round-5 reflection routes: `<fn>.__globals__[...]`, `getattr(<allowlisted>, "__builtins__")[...]`, string-concatenated identifiers, and `().__class__.__base__.__subclasses__()`**,
When the static lint is run against each fixture,
Then it flags **every** fixture — keeping the lint from regressing on known tricks. **This is the lint's regression floor, explicitly NOT a completeness contract (design-review-005 correction — the v1.4 text called it "the gate's completeness claim, exactly this, no more," which Round 5 falsified by executing a route outside the corpus): completeness is delivered by the runtime harness TC-JU12-01; this corpus only pins the fast lint against known evasions.**

**TC-NF6-05 — superseded (design-review-004)** · tombstone · not executable
Former content (the `<numpy-alias>.random.<name>` attribute pattern) is absorbed and strengthened by TC-NF6-03's wholesale ban. The ID is retained per this document's append-only discipline so no ID silently vanishes; it appears in no tally except the total.

**TC-NF6-06 (clean-pass / false-positive guard)** (rewritten design-review-004) · scenario test · executable
Given the real, legitimate judge source,
When the full gate (TC-NF6-01 through -03) runs against it,
Then it passes — necessary because TC-NF6-02 and TC-NF6-03 are deliberately over-broad, and a gate that cries wolf gets disabled by the people it inconveniences.

**TC-NF6-07 (models→worlds import gate)** (new, design-review-005) · property-based test (import graph) · executable
Given the AST of every module under `wmj/models/`,
When each is scanned for an import of `wmj.worlds` or any `wmj.worlds` submodule — comparing, for an `ImportFrom`, the **joined** `node.module + "." + alias.name` as well as `node.module` itself,
Then none is found — enforcing "models never import world internals" (models §6), the claim the entire `WorldContext` ADR was built to satisfy and which Round 5 found had no gate (every existing gate pointed the other way). `from wmj.worlds import lv` is caught by the joined comparison, not only `import wmj.worlds.lv`. This is the models-side twin of TC-MU9-01(c).

**TC-NF6-08 (models→harness import gate)** (new, design-review-007) · property-based test (import graph) · executable
Given the AST of every module under `wmj/models/`,
When each is scanned for an import of `wmj.harness` or any `wmj.harness` submodule — the same joined `node.module + "." + alias.name` comparison as TC-NF6-07,
Then none is found — enforcing "models never import the harness," the invariant the `SeedSource`-in-`wmj.models.base` design depends on (a fixture rebuilds `direct`'s stream with no harness import). Round 7 found this direction asserted as settled fact but enforced by no gate (TC-NF6-07 checks models→worlds; TC-MU9-01(c) checks the other four packages' imports *toward* models; neither checks models' imports *out* toward the harness). Executed: catches `import wmj.harness` and `from wmj.harness.trials import build_roster`, passes on `from wmj.models.base import SeedSource, component_key`. (`wmj.models.base` is model-side, so the harness importing *it* is the allowed direction, admitted by TC-MU9-01(c)'s allowlist.)

**TC-NF6-09 (models→judge / models→reporting import gate)** (new, design-review-008) · property-based test (import graph) · executable
Given the AST of every module under `wmj/models/`,
When each is scanned for an import of `wmj.judge` or `wmj.reporting` (or any of their submodules) — the same joined `node.module + "." + alias.name` comparison as TC-NF6-07/08,
Then none is found — closing the same class of gap TC-NF6-07/08 closed for `worlds`/`harness`, but for the two remaining sideways directions, which Round 8 found undisclosed as well as unenforced (nothing checked what `wmj/models/*` imports toward `wmj.judge` or `wmj.reporting`, so a model file could in principle reach sideways into judge threshold constants or reporting internals — including hand-tuning a fixture's corruption against the judge's own thresholds, the exact prereg-gaming category MU-6/ADR-M5 otherwise police). `wmj/models/*`'s only sanctioned outward imports are, completely: `numpy`, `math`, `dataclasses`, `typing`, `__future__`, `wmj.errors`, `hashlib`, `wmj.models.base`, `wmj.models.registry` (**corrected design-review-009 I5** — the previous wording here, and cross-cutting ADR-003's own sentence, omitted `wmj.errors`/`hashlib`/`__future__`, which the already-built and already-reviewed `wmj/models/base.py` legitimately needs; a spec correction applied verbatim, not just enforced by a gate that quietly diverged from it).

**TC-NF6-10 (reporting's own outward-import gate)** (new, design-review-009 I3/A9) · property-based test (import graph) · executable
Given the AST of every module under `wmj/reporting/`,
When each is scanned against a full allowlist — `numpy`, `matplotlib`, `math`, `dataclasses`, `typing`, `pathlib`, `re`, `importlib.metadata`, `wmj.errors`, `wmj.judge` (read-only, the `Verdict` type — reporting.md §4), and reporting's own package,
Then any import outside that set is flagged. Reporting was the one package among the four (judge, models, worlds, reporting) whose ADR-003 layering claim had no mechanical gate — an inconsistency design-review-009's Structural and Security panels both flagged: reporting is the sole writer of every public `out/` artefact (design-review-008 C8), so it is the last boundary before content becomes public (NF-4), and the separate-process byte-identity gate (TC-NF1-01) structurally cannot catch a *deterministic* leak through it (a stable path/hostname/env-var read via an accidentally-imported `os`/`socket` is identical across all ten runs, so a gate that only flags variation never sees it). This is a fast lint, not a completeness proof — the same disclaimer as TC-NF6-01/07/08/09 — but it is, uniquely for reporting, the only mechanical control of any kind at this boundary.

**TC-NF6-11 (negative/phantom-gate — the metaclass structural ban)** (new, design-review-009 A3/I6) · mutation test (fixture) · executable
Given the `class X(metaclass=Y): ...` fixture already used to prove the judge gate's metaclass check fires (`tests/gates/fixtures/metaclass_namespace_capture_route.py`),
When the check is run in isolation against that fixture,
Then it is flagged specifically by the structural metaclass scan (`scan_metaclass_usage`), independent of any banned-identifier hit — closing the orphan design-review-009's Requirements Coverage panel found: the check was built and applied (judge and models gates) but traced to no test-case ID, and TC-NF6-02's "no finite identifier list bounds Python reflection" disclaimer does not cover it, since a `metaclass=` keyword argument is a structurally different AST shape (a `ClassDef` keyword, not a `Name`/`Attribute`/import-alias) than anything TC-NF6-02 enumerates.

**TC-NF1-07 (seed-key colon rejection)** (new, design-review-007) · mutation test · executable
Given `component_key`'s pinned construction (cross-cutting ADR-002 rule 2),
When it is called with a part containing the `:` delimiter — e.g. `component_key("a:b", "c")` — or the current colon-free vocabulary — e.g. `component_key("lv", "training", "eval-starts")`,
Then the colon-bearing call raises `SeedKeyError` (the join is not injective across parts straddling a `:`, executed: `component_key('a:b','c') == component_key('a','b:c')` before the guard), while the legitimate call succeeds — so "adding a model/world/region provably never shifts another's stream" is enforced by construction, not by luck of the current names never containing a colon.

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
| WD-7 | TC-WD7-01, -02 | JU-8 | TC-JU8-01, -02, -03; NF1-05/-06 (paired starts, per-region partitioning — ADR-004) |
| WD-8 | — (Won't; nothing to verify) | JU-9 | TC-JU9-01, -02, -03 |
| MU-1 | TC-MU1-01, -02, -03 | JU-10 | TC-JU10-01, -02 |
| MU-2 | TC-MU2-01, -02, -03; -MU6-05 (baseline-scope) | JU-11 | TC-JU11-01, -02 |
| MU-3 | TC-MU3-01, -02, -03 | JU-12 | TC-JU12-01, -02, -03, -04 |
| MU-4 | TC-MU4-01, -02 | JU-13 | TC-JU13-01 |
| MU-5 | TC-MU5-01, -02, -03 | RP-1 | TC-RP1-01 |
| MU-6 | TC-MU6-01, -02, -03, -04, -05 | RP-2 | TC-RP2-01 |
| MU-7 | TC-MU7-01, -02 | RP-3 | TC-RP3-01 |
| MU-8 | TC-MU8-01 | RP-4 | TC-RP4-01, -02 |
| MU-9 | TC-MU9-01, -02, -03 | RP-5 | TC-RP5-01 |
| MU-10 | — (Won't; nothing to verify) | RP-6 | TC-RP6-01, RP-CARD-01 |
| JU-1 | TC-JU1-01, -02 | RP-7 | TC-RP7-01, -02 |
| | | RP-8 | TC-RP8-01 |
| NF-1 | TC-NF1-01, -02, -03, -04, -05, -07, -08, -09 (-06 superseded tombstone) | NF-4 | TC-NF4-01, -02, -03 |
| NF-2 | TC-NF2-01 | NF-5 | TC-NF5-01, -02 |
| NF-3 | TC-NF3-01 | NF-6 | TC-NF6-01, -02, -03, -04, -06, -07, -08, -09, -10, -11 (-05 superseded tombstone) |

**Every Must and the one mechanically-testable Won't (JU-13) has at least one case. Zero orphan requirements, zero orphan cases.** WD-8 and MU-10 are the two Won'ts with nothing to verify by design (they describe absence, not behaviour).

**Totals:** 103 case IDs (101 live) across 45 requirements — grown round by round (70 after design-review-001/002; +9 after `/gvm-design-review` design-review-003 — Round 3, dual/blind — found: TC-MU9-02, a phantom-gate pairing TC-MU9-01 could not previously have, since its prior git-diff mechanism was structurally incapable of failing; TC-JU9-03, a property test enforcing the `exceptions.observed`/`trials.is_exception` invariant judge.md's prose only asserted; TC-NF6-02 through -06, splitting the single `TC-NF6-01` into one ID per AST-gate check plus a phantom-gate pairing, after the gate grew from one check to five across two rounds with no corresponding test-ID growth; TC-NF4-03, proving the run-time confidentiality scan is a real safety net independent of whether the pre-commit hook was ever installed; TC-NF5-02, covering the previously-orphaned `.md`/`.html` spec-parity hash check. Count independently verified via `grep -c '^\*\*TC-'` = **103**, not carried forward by arithmetic; **design-review-006 (v1.3) added four cases (TC-JU12-03, TC-NF1-04, TC-MU6-03, TC-MU7-02) to the 85; design-review-007 (v1.4) added eight (TC-NF6-08, TC-NF1-07, TC-NF1-05, TC-NF1-06, TC-MU6-04, TC-MU6-05, TC-JU12-04, TC-RP7-02); design-review-008 (v1.5) added three (TC-NF6-09, TC-NF1-08, TC-NF1-09) to the 97; design-review-009 (v1.6) added three (TC-MU2-03, TC-NF6-10, TC-NF6-11) to the 100 — the executed build's own review closing two orphan fail-loud mechanisms (the metaclass structural ban, the baseline spreads' `DegenerateSpreadError` guard) that had no phantom-gate case, plus reporting's previously-ungated outward imports; two IDs, TC-NF6-05 and TC-NF1-06, are superseded tombstones, so 101 cases are live**). 13 negative/phantom-gate cases (TC-WD3-02, TC-WD7-02, TC-NF1-02, TC-JU9-02, TC-JU11-02, TC-MU9-02, TC-NF6-04, TC-JU12-02, TC-JU12-03, TC-NF1-07, TC-MU6-04, TC-MU2-03, and TC-NF6-11 — the seed-key colon, prereg-content-drift, degenerate-spread, and metaclass can-fail proofs; TC-NF6-06 is the complementary clean-pass guard, not counted here). 17 property-based cases (TC-JU1-02, TC-JU4-02, TC-JU5-02, TC-JU12-01, TC-NF6-01, TC-NF6-02, TC-NF6-03, TC-NF6-07, TC-NF6-08, TC-NF6-09, TC-NF6-10, TC-JU9-03, TC-NF1-03, TC-NF1-04, TC-NF1-05, TC-NF1-08, TC-MU7-02). 7 cases marked *judged* rather than *executable* (TC-MU6-02, TC-JU10-02, TC-RP5-01, TC-NF4-02, TC-NF5-01, TC-RP-CARD-01, TC-JU12-04 — plain-language and human-comprehension checks that genuinely need a reader, not a script). **All three former `<spec-value>` placeholders are now filled** from the settled specs (TC-JU8-02 → N=200, TC-NF2-01 → 600 s, TC-NF3-01 → `{numpy, matplotlib}`+dev `{pytest}`; design-review-005 — no case now carries an unresolved `<spec-value>`). 2 cases marked *(supporting)* — they validate wiring between two requirements rather than being the primary coverage of either (TC-MU2-02, TC-JU8-03).

---

*Developed using the Grounded Vibe Methodology*
