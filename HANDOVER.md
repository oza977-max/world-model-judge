# Handover

## CURRENT STATE

**Phase:** Technical specification complete, **four times design-reviewed**, and
now **structurally repaired** — specs at **v1.4** (worlds.md at v1.3), test
cases at **v1.1** (79 IDs, 1 superseded tombstone, 78 live). Round 4 (second
dual/blind, 14 reviewers, targeted at pressure-testing Round 3's two flagship
fixes per BC-2) found both flagship fixes failed — and failed *into each other*
(the `fx-brittle` redesign and the new TC-MU9-01 mechanism were mutually
exclusive on the one fixture exercising both). Four consecutive "Do not build"
verdicts with a non-shrinking defect population led to a root-cause diagnosis
recorded in `reviews/calibration.md`: fixes were being made at the size of the
finding, not the size of the rule, and verified by prose re-reading, which
cannot check code-behaviour claims. The user approved a **structural repair
session** (31 Aug) executing all five moves of calibration.md's structural
recommendation: (1) one canonical Verdict-schema keying rule verified via a
block×axis table (judge v1.4); (2) a `WorldContext` + uniform
`factory(ctx, rng)` signature dissolving the models↔worlds layering question
(models v1.4); (3) the AST gate rebuilt as allowlist + banned identifiers +
wholesale `numpy.random` ban + an executable evasion-fixture corpus
(cross-cutting v1.4, TC-NF6 family restructured, TC-MU9-01 mechanism v3);
(4) a full implementation-guide reconciliation (10 orphaned test cases wired,
three-branch critical path, roster corrected 8→7); (5) the phantom CI layer
and GitHub-API dependency deleted from the design. **These repairs are
self-verified only — Round 5 (dual/blind, mandatory) is the pending next step
and its explicit job is to pressure-test the five repair groups.**

Earlier history: `/gvm-tech-spec` ran in full; Round 1 (quick-scan, 4
panels) found 10 Critical + 16 Important, patched into v1.1; Round 2 (full,
strict, +Security/Quality-Attribute panels) found 8 new Critical + 9 new
Important plus 3 partial/regressed Round-1 fixes, patched into v1.2.

**Round 3 was the first genuine dual/blind review** (shared rule 16 activates
automatically at 2+ prior rounds): 7 defect-class panels × 2 (one calibrated,
one fully blind, zero shared context) = 14 independent reviewers. **The two
fixes flagged after Round 2 as least trustworthy — the double-pendulum physics
correction and the NF-4 security-mechanism redesign — both held**, confirmed by
two independent from-scratch re-derivations/recounts each (the EOM was
re-derived from Euler–Lagrange twice, independently, and matched exactly; the
28-chunk count was recounted twice, independently, and matched exactly). But
dual/blind review surfaced a large new defect population anyway — ~12 Critical,
~12 Important, 13 of them reached by two or more of the 14 reviewers with zero
shared context. Two structural patterns dominated (both promoted to standing
**Build Checks** in `reviews/calibration.md`): (1) Verdict schema blocks kept
missing a task/region/step key their own defining ADR required — this is the
*third* round this exact shape has appeared, previously on Chart 1 (R1), Chart
2/3 (R2), now on `calibration`/`sharpness`/`climatology`/`region_labels`/Chart
4/Chart 1 again (R3); (2) every enforcement mechanism Round 2 introduced to fix
a prior finding had a real, adversarially-discoverable gap one level below what
it explicitly checked — most strikingly, the `test_registry_isolation.py`
mechanism written in Round 2 to close a "no enforcement" finding was found to
be **structurally incapable of ever failing** (it diffed a fixture it
manufactured itself). The user again chose "fix everything item-by-item"; all
of it is patched into v1.3 (see `design-review/design-review-003.html` and
`reviews/calibration.md` for the full record, including the two Build Checks).

**Code written:** none. Deliberately. Still true — specs only.
**Blocked on:** nothing. **The honesty gap is finally closed** — Round 3's
fixes are, as of this writing, self-verified again (not yet independently
re-checked), so whoever picks this up should treat a Round 4 the same way
Round 3 treated Round 2: check the fixes least likely to be self-verifiable
correctly first (the `fx-brittle` redesign and the AST-gate/registry-discovery
mechanisms are this round's equivalent of "a physics formula and a security
mechanism"). The two Build Checks in `reviews/calibration.md` — schema
granularity, and adversarial-pressure-testing of enforcement mechanisms — are
the standing process fix meant to catch the next instance of either pattern
without needing a full round to find it. The other live choice is the same one
Round 2's handover posed and Round 3 answered "keep reviewing": accept the
residual risk and proceed to `/gvm-build` at Phase 1 / P1-C01, 28 chunks per
the (now twice-independently-confirmed) implementation guide.

**Update — Round 5 complete, and its fixes applied ("fix all").** Specs are at
**v1.5** (worlds v1.4), test-cases at **v1.2** (85 IDs, 84 live). Round 5 (third
dual/blind, 14 reviewers) returned the fifth "Do not build" but was the first
round to *clear* prior defects rather than reshuffle them — the critical-path
arithmetic and BC-3 reconciliation held under independent re-derivation, and the
Round-4 fx-brittle/registry mutual exclusion is genuinely dissolved. What it then
reached, with executable adversarial review, was a deeper layer: three design-level
findings that the "fix all" pass *decided* rather than patched —
(1) the judge-purity gate cannot be a static-analysis boundary in Python, so
TC-JU12-01's **runtime effect-guarding harness is now the load-bearing control**
and the AST gate is lint; (2) the model interface gained a `TrainingData` channel
and `WorldContext.scale` (6 of 7 models were unbuildable without them);
(3) seed derivation is now content-addressed so adding a model shifts no other's
stream. Everything else (13 Critical + 15 Important) was fixed at rule-size; see
`reviews/calibration.md`'s "Round 5 fixes applied" entry and
`design-review/design-review-005.html`. The critical path was honestly
re-derived to **10 chunks** (the new file-ownership/prereg-count edges lengthen the
reporting branch — a real graph change, not a recount error).

**Update — Round 6 complete (dual/blind, 14 reviewers, ran the BC-2 pressure-test).**
Sixth "Do not build", but the failure shape **inverted** and this is the closest
to buildable. The design/contract/coverage surface **converged and verified clean**
(panels that scored 2–4 for five rounds scored 6–9), and **three v1.5 design
decisions were confirmed working by execution**: content-addressed seeding (no-shift,
3 panels), the TC-MU9-01 teardown (both R5 leaks reproduced as regressions), and the
import-join. Critical path (10 chunks) and count (28) correct on three independent
re-derivations. The remaining Criticals are concentrated in two mechanism-precision
clusters with known small fixes:
1. **Purity/scan mechanisms:** the runtime purity harness is executed-broken by 4
   panels — `sys.modules['os'] = guard` doesn't reach numpy's pre-bound `os`; the fix
   is **mutate-the-shared-object-in-place** (`setattr(os,'system',…)`), demonstrated
   by the panels; also guard numpy's C-level clock. And the NF-4 scan command is inert
   — `git cat-file … --batch-check` emits no content; needs **`--batch`**.
2. **Determinism precision:** pin `component_key`'s hash (it's an "e.g."), add a
   purpose discriminator to the seed key (train/eval/benchmark starts collide),
   disclose the `model_ref` roster-shift, spec the ensemble K=5 split, name the
   `meta.platform` source.
Plus prose/wiring fixes (the skill-step regression, the Verdict dp-control-in-lv
example, the fixture-seed channel, the baseline-classification contradiction, TC-MU9-03,
three misplaced wirings). **v1.5's own fix-all planted 5 of the 10 Criticals** (fast +
self-verified — the BC-2 pattern). See `design-review/design-review-006.html` and the
"Round 6 result" entry in `reviews/calibration.md`.

**Update — Round 6 fixes applied with execution-verification (31 Aug, specs → v1.6,
test-cases v1.3), then Round 7 dispatched.** The user chose "fix all with
execution-verification this time." The BC-2 discipline was finally honoured: every
enforcement-mechanism fix was written as a python3 script and **run before the spec was
written**, with the executed result cited in the text — the purity harness (mutate-in-place,
executed catching the escape the `sys.modules` swap missed; guards `numpy.datetime64`),
the NF-4 scan (`--batch` executed finding a planted term `--batch-check` missed), and the
seeding (`SeedSource`+pinned `component_key` in `models.base`: two-impl convergence, fixture
rebuild of `direct`, ensemble K=5, and train/eval/benchmark purpose-key independence all
executed True). Plus the prose/wiring fixes (skill one-step `h=1`, Verdict example, ninth
mandatory `limitations`, `interval_levels` single-producer, `meta.platform` composite,
`is_baseline` flag, per-model `check_prereg`, the three re-placed wirings, `wmj list-models`
for TC-MU9-03, `N_train=2000`, `model_ref`-shift disclosure). See the "Round 6 fixes applied"
entry in `reviews/calibration.md` and `design-review-006.html`.

**Update — Round 7 complete (dual/blind, 14 reviewers, independent execution re-run).**
The BC-2 discipline paid off *for the mechanisms actually run*: three of four determinism
repairs and the specific Round-6 purity escape are confirmed genuinely fixed by independent
running code (guard fires; `--batch` finds planted terms; `component_key` converges; fixture
rebuild is bit-exact). Interface contracts converged (B: 9/8, highest any dimension has
scored). **But three things did not clear:**

1. **BC-2/BC-3 recurred on the prose-only edits** — judge §4's purity *body* still describes
   the disproven `sys.modules` swap (only the changelog claimed the mutate-in-place fix);
   TC-JU12-03 and TC-MU7-02 are change-note-claimed-wired but in no chunk's `[Test:]` tag;
   architecture-overview.md + reporting.md have no v1.6 changelog row; the model_ref-shift
   disclosure promised in Round 6's plan was never written. All mechanical, ~1 hour to finish.
2. **BC-4 is now CONFIRMED, not a candidate** — Security scored 3/3 (both panels). The
   mutate-in-place harness closes the *specific* escape but is defeated one layer down by
   executed routes: `os.environ` pre-capture rebind (same bug class v1.6 fixed), `np.random.normal()`
   (the threat model's own example), `ctypes` resident via numpy reaching raw syscalls, and
   unseeded `Generator(PCG64())` → `getrandom()` (no Python object to mutate — architecturally
   uninterceptable). No finite enumeration closes Python's reflection/syscall graph. **This is a
   design decision** (narrow JU-12's claim to best-effort+empirical+disclosed, OR move purity
   out-of-process to an OS sandbox), not a patch — reserved for the user per Hard Gate 6.
3. **Two standing spec gaps** — the harness orchestration loop + JudgeInput per-region partitioning
   is still unwritten as an algorithm (open since Round 5); no gate enforces `models→harness`; and
   `check_prereg` verifies commit *order* not content-invariance (an honestly-dated second commit
   defeats prereg — a real MU-6 gap, executed with git).

Seventh "Do not build," but the first where the gap is one mechanical hour plus one honest design
fork. See `design-review/design-review-007.html` and the "Round 7 result" entry in
`reviews/calibration.md`.

**Update — Round 7 fixes applied (31 Aug, specs → v1.7, test-cases → v1.4, commit `db4a15e`).** The user chose "fix all with execution-verification" and delegated the C1 design decision to Claude. Every enforcement mechanism was run as a python3 script before the spec was written.
- **The C1/BC-4 decision (reversible in the spec text):** JU-12's completeness claim is **dropped**. The purity harness is now a best-effort tripwire over the known ambient surface; the **ten-run byte-identity gate TC-NF1-01 is the load-bearing backstop**; the `ctypes`/`getrandom` residual is disclosed (TC-JU12-04). Out-of-process seccomp was rejected (Linux-only, would break NF-1 cross-machine byte-identity). Accidental routes hardened: `os.environ` in-place `_Environ` mutation, all 63 `numpy.random` module names.
- **Mechanical cluster (all executed):** judge §4 body rewritten to mutate-in-place; TC-JU12-03/TC-MU7-02 wired; models→harness gate (TC-NF6-08); colon-rejection (TC-NF1-07); TC-MU9-03 pinned to `cli.main([...])`; prereg content-hash (TC-MU6-04); is_baseline scope + model_ref disclosure; orchestration loop pinned as cross-cutting ADR-004 (TC-NF1-05/06); SVG hashsalt+metadata (TC-RP7-02); run-manifest source; the two missing v1.6 changelog rows. Eight new cases (97 IDs, 96 live); all banners v1.7.

**Next: a Round 8 (dual/blind) is owed by BC-2** — self-verification, even execution-based, is not an independent re-run, and no external panel has yet seen the reframed purity claim or ADR-004. The user owns whether/when to run it. If Round 8 clears to "Build with caveats" or "Build from this," `/gvm-build` can start at P1-C01.

**Update — Round 8 complete (dual/blind, 14 reviewers, independent execution re-run of the v1.7 hardening).** Eighth "Do not build," but the finding shape is new: the Round 7 architecture call (drop JU-12 completeness, tripwire + TC-NF1-01 backstop) is **independently confirmed sound**. What broke is the execution under it, plus a brand-new design element:

- **Purity hardening: the SAME rebind-vs-mutate bug recurred a THIRD time, three different objects.** `os.environ`'s dunders (`os.environ["X"]`, `dict(os.environ)`) bypass the instance-level patch — Python resolves special methods against the *type*. `numpy.datetime64` **cannot** be mutated in place at all (immutable C type) — only a rebind works, silently reopening a pre-capture escape. `numpy.random`'s private submodules (`_pcg64`, `_generator`) hold separate bindings the top-level patch never reaches — an ordinary `import numpy.random._pcg64` bypasses it, and TC-JU12-04's disclosed residual is factually wrong about its own example. All three execution-confirmed (E-cal, E-blind).
- **TC-NF1-01 (the backstop) may itself be a phantom gate** — no "separate processes" requirement unlike its siblings; executed proof that an in-process cached value passes 10 "runs" while varying across processes (F1-blind).
- **ADR-004 needs rethinking, not patching** — 5+ panels independently found its pseudocode doesn't typecheck against the registry contract (executed `AttributeError`), baselines are consumed before they're populated (4 panels, same computed sorted order), and — the deepest finding — its "one region per JudgeInput" **contradicts** the multi-region `region_labels` design stable since Round 3 (C-cal, B-cal).
- **The wiring gap recurred a third consecutive round** on different IDs each time (6 panels: TC-MU6-04/05, TC-NF1-07, TC-RP7-02 unwired, misattribution repeated).

New standing pattern, **BC-5 (candidate)**: every "mutate in place" fix needs to generalize to *every reachable binding* of the guarded object — module-level, submodule-level, class-vs-instance — not just the first one found. See `design-review/design-review-008.html` and the "Round 8 result" entry in `reviews/calibration.md`. **Awaiting user triage before the next fix pass.**

**Known debt:** the `specs/*.html` twins carry a deterministic staleness banner (they
regenerate with the NF5-02 parity hash at build time, P6-C02). The older "What exists"
table below is itself stale (says v1.3/v1.0) — Round 6 flagged it; refresh when next
editing this file.

Last updated 2026-08-31 (design-review-008 — Round 8 complete; awaiting user triage).

---

## What exists

| File | What it is |
|---|---|
| `CLAUDE.md` | Project context and standing rules. Read this first. |
| `requirements/requirements.md` | v1.2. 45 requirements (42 Must, 0 Should, 3 Won't) across four domains plus non-functional. Revised after a six-expert GVM review board and two independent audit passes (technical-correctness, LeCun-grounded, novelty/prior-art). Every one carries an "In plain words" restatement. |
| `requirements/requirements.html` | Same content, Tufte-styled. |
| `requirements/wordsareamenu.html` | The source essay, now draft v2.7. Also revised post-review: narrowed novelty claim, corrected two factual errors (flight-simulator framing, SR 11-7 currency), added the employer disclaimer and an AI-assistance transparency note. |
| `risks/risk-assessment.md` | Four product risks, written before requirements. Still references essay draft v2.1 deliberately — the historical record of what it was written against. |
| `test-cases/test-cases.md` | v1.0 (content through Round 3). 79 cases, one `TC-{REQ-ID}-{NN}` per requirement minimum, more for credibility-fatal ones. 7 negative/phantom-gate cases, 10 property-based cases. Full traceability matrix at the end — zero orphan requirements, zero orphan cases, every count independently grep-verified against the document body (`grep -c '^\*\*TC-'` = 79). |
| `specs/cross-cutting.md` (+.html) | v1.3. Stack (pure NumPy, hand-rolled MLPs), the four-rule determinism strategy, five-package structure, error-handling conventions, the two-package dependency budget. The AST import-graph gate now does 5 individually-tested things (TC-NF6-01..05) — bans `wmj.*` imports, ambient-module imports, `importlib`/`__import__` outright, `exec`/`eval` outright, and `numpy.random.*` legacy calls. NF-4's scan now covers historical git blobs too, with a named pre-commit-hook installation path. `wmj.models.registry`'s `pkgutil`-based auto-discovery is now specified, and TC-MU9-01's mechanism is a structural test, not a tautological git-diff (all Round 3 fixes). |
| `specs/worlds.md` (+.html) | v1.2, unchanged since — Round 3 assigned it no fix. LV and double-pendulum constants fully pinned. Shared fixed-step RK4 with a 1e-6 relative drift bound (ADR-W1). Divergence benchmark: 64 seeded starts, median curve per region (ADR-W3). The double-pendulum EOM's Round 2 coefficient correction was independently re-derived from scratch twice in Round 3 and confirmed correct. |
| `specs/models.md` (+.html) | v1.3. Baselines with honest training-residual spreads. Three fixtures, `fx-brittle` redesigned in Round 3 as a genuine post-hoc wrapper keyed to the world's own declared region box (closing a wrapper-framing contradiction and a region-mismatch three reviewers independently found). The direct-vs-ensemble unrigged pair with pre-registered spread mapping and matching margin. Pre-registration enforced by git commit ordering; three disclosed residual risks, the third (prereg timestamp forgery) added in Round 3 with a partial GitHub-hosting-dependent mitigation. |
| `specs/judge.md` (+.html) | v1.3. CRPS as the scoring rule. **N=200 independent trials**, exact-binomial bands. Full verdict JSON schema: `calibration` and `sharpness` now keyed per task (Round 3 fix — both were the one metric family left flat despite their own ADRs requiring per-task granularity), `climatology.per_task` gained a region field, `region_labels` unified to one canonical shape across all three domain specs. |
| `specs/reporting.md` (+.html) | v1.3. All four required charts designed down to authored caption templates. Chart 1 now also panels by horizon_step; Chart 2 starts at step 1 (closing an undefined `log(0)` case); Chart 3 reads the new per-task `calibration` block; Chart 4's region-to-column mapping rule is now stated explicitly (all Round 3 fixes). |
| `specs/architecture-overview.md` (+.html, with an inline C4 container SVG) | v1.3. Synthesis of all five specs + a Brooks conceptual-integrity review. No new conceptual-integrity defect found in Round 2 or Round 3 — version references updated each round. |
| `specs/implementation-guide.md` (+.html) | v1.3. 6 phases, **28 chunks**, independently recounted and confirmed correct twice in Round 3 (previously "24"/"23" — both wrong since v1.0). The stated "critical path" was itself wrong (miscounted its own list, omitted a hard prerequisite) — corrected to 9 chunks across two equal-length branches. Full dependency network, parallel-work sets, and a complete wiring matrix. |
| `design-review/design-review-001.html` | Round 1 (quick-scan, 4 panels). 10 Critical + 16 Important, fixed into v1.1. Historical record. |
| `design-review/design-review-002.html` | Round 2 (full, strict, +Security/Quality-Attribute panels). 16/26 Round 1 findings held, 3 partial/regressed; 8 new Critical + 9 new Important, fixed into v1.2. Historical record. |
| `design-review/design-review-003.html` | Round 3 — the first genuine **dual/blind** review (14 independent reviewers). The two highest-risk Round 2 fixes (EOM, chunk count) both independently confirmed correct twice over. ~12 new Critical + ~12 new Important surfaced anyway, dominated by two recurring structural patterns, both promoted to standing Build Checks. All fixed into v1.3, same-session, again self-verified. Verdict was "Do not build" pre-fix. |
| `reviews/calibration.md` | GVM review-calibration record. Score history across all three rounds, dual-review metadata (Round 3), two promoted Build Checks (schema/consumer granularity mismatch; adversarial-pressure-test enforcement mechanisms before calling them resolved), and the still-open note that Round 3's own fixes are self-verified, not yet independently re-checked. |

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

Round 3 already happened — dual/blind, as Round 2's handover predicted it would need to be. The two fixes flagged as least trustworthy held under independent re-derivation; a large new defect population showed up anyway, dominated by two now-named-and-tracked structural patterns (see the two Build Checks in `reviews/calibration.md`). Two reasonable options remain, in order of recommendation:
1. **A Round 4 design review**, prioritising independent pressure-testing of Round 3's own least-self-verifiable fixes (the `fx-brittle` redesign; the registry auto-discovery mechanism and its new TC-MU9-01 structural test) ahead of fresh discovery. Genuine dual/blind review is now standing practice (shared rule 16 stays active at round 3+), so Round 4 would run the same way Round 3 did. If this shape keeps producing a comparable defect population a fourth time, that itself would be worth surfacing to the user as a question about whether continued rounds are the right lever, versus a deliberate one-time structural rewrite guided by the two Build Checks.
2. **`/gvm-build`**, accepting the residual self-verification risk, starting at Phase 1 / chunk P1-C01 (foundations: scaffold, serializer, seed plumbing), followed by P1-C02 (the walking-skeleton MVP slice) and P2-C05 (the second early real-chart slice) — 28 chunks total, independently confirmed twice, per the v1.3-corrected implementation guide's 9-chunk critical path.

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
