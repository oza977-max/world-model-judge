# World Model Judge — Judge Specification

Version 1.6 · 31 August 2026 · Domain 3 of Requirements v1.2 · References: cross-cutting spec v1.6, worlds spec v1.4, models spec v1.6

> **Change note (v1.6 — design-review-006 repair).** Five Round-6 findings closed: (1) `skill.per_task_region` is pinned to the **one-step `h=1`** horizon, not `h_task` — the v1.5 block×axis pass had reworded it in a way that contradicted the literal "one-step" MU-5's matching margin and TC-MU2-02/TC-JU2-01 depend on; (2) the §5 Verdict example no longer lists a `dp-control` (double-pendulum) entry inside an `"lv"` verdict — both entries are lv tasks, honouring the one-Verdict-per-(model,world) contract; (3) `limitations` is now the **ninth mandatory** JU-9 group (refuse-on-missing), reconciling judge.md's own enumeration with reporting.md's model-card claim; (4) `interval_levels` is removed from `JudgeInput` — the four levels are judge constants (`_normal.py`), one producer; (5) `meta.platform` is pinned to the coarse composite `f"{sys.platform}-{platform.machine()}"`, matching NF-1's "same platform" granularity (the natural `platform.platform()` embeds kernel/glibc and would break byte-identity). See design-review-006.html.

> **Change note (v1.5 — design-review-005 repair).** TC-JU12-01, the judge's runtime purity test, is now specified concretely as the **load-bearing** purity control (cross-cutting ADR-002 rule 3 demoted the static AST gate to lint after Round 5 executed reflection escapes): it guards *effects* by replacing the shared ambient capability objects (`os`, `open`, clock, network, subprocess, `numpy.random`) with raise-on-access guards, so an impure operation trips however the code reached it — with a phantom-gate case (TC-JU12-02) proving it catches a `__globals__`-routed `os` read. Also: calibration and sharpness are pinned to the task's *single* evaluation horizon `h_task` (Round 5 found "at its own declared horizon *steps*" unbuildable against the block×axis table's own "—" for `horizon_step`); the N=200 trial set is now pinned as **shared across models** (paired design) as well as tasks, so MU-5's margin and Chart 4's ranking are apples-to-apples; ADR-J6's runtime envelope corrected for the per-region 2× multiplier (~16M, not 8M — still inside budget); `error_vs_horizon` carries `dt` so Chart 2's world-time axis has a source. See design-review-005.html.

> **Change note (v1.4 — the design-review-004 structural repair).** The **canonical keying rule** replaces four rounds of block-by-block schema patching: every metric block carries explicit `task`/`region`/`horizon_step` key fields for exactly the axes its defining ADR says the fact varies over, suffix-encoding is banned, and a block×axis table in §5 makes the whole convention checkable by inspection. Applied this pass: `error_vs_horizon` became a `per_region` array (the flat object couldn't even hold two regions' curves), `calibration`/`sharpness` lost their `_in`/`_out` suffix fields for explicit region keys. Also: N=200 pinned as one shared trial set per (model, world, region) across tasks (the ambiguity ADR-J6's budget had silently resolved one way while ADR-J4 read the other); `natural_units` made nullable with the pendulum's `null` pinned; the roster count corrected to 7 (ADR-J4, ADR-J6); sharpness given its missing chart consumer (Chart 3, reporting spec). See design-review-004.html.

> **Change note (v1.3).** Revised after `/gvm-design-review` design-review-003 (Round 3, dual/blind). Fixed: `calibration` and `sharpness` restructured from flat, region-only blocks to `per_task` arrays, matching ADR-J4's own per-(task,region) 200-trial-set definition and ADR-J4's hedging cross-check text, which already read "that region/task's" width; `climatology.per_task` gained a `region` field, since `switch_step` is explicitly measured against "the trial's region's curve"; `region_labels` given one canonical `{region_name, axis}` shape, replacing three previously-inconsistent definitions across worlds/models/this document; the `trials.is_exception`/`exceptions.per_task.observed` reconciliation is now a build-enforced property test (TC-JU9-03), not only a stated rule. See design-review-003.html for the full findings.

> **Change note (v1.2).** Revised after `/gvm-design-review` design-review-002 (Round 2, independent re-check under strict criteria). Fixed: `trust_horizons.per_task` had no `region` field, unlike every other per-task block, which meant the headline comparison table couldn't show `fx-brittle`'s entire reason for existing (great at home, catastrophic away) — added, with the trust-horizon formula restated per (model, world, task, region). JU-9's mandatory-field list named six field groups but the schema had grown to eight (`trials`, `climatology` were both added in v1.1 but never added to the "mandatory" enumeration) — fixed, closing a fail-loudly gap on the primary chart's own data block. Added `calibration.n_trials`, the one number Chart 3's error bars need that reporting cannot compute itself. Clarified that `trials.is_exception` is copied from ADR-J4's actual per-dimension joint exception test, never derived from the scalar `outcome_distance`/`band_hi` comparison also carried in the same block for charting — and that `exceptions.per_task.observed` is a sum over the same `trials.is_exception` array, so the two blocks cannot silently disagree. See design-review-002.html for the full findings.

> **Change note (v1.1).** Revised after `/gvm-design-review` design-review-001 (four parallel defect-class panels). Fixed: the Verdict schema carried `is_fixture`, `model_ref`, and `meta` fields the judge structurally cannot produce (JU-1/JU-12 purity violations) — these move to a harness-owned envelope, and the pure Verdict now contains only what the judge itself computes. Added: a per-trial data block the exception chart needs (was missing entirely), an explicit `region` field on exception counts, fixed z_p constants (erf gives no inverse), a corrected φ formula and its array-vectorization strategy, an explicit restatement of the shared distance formula, concrete climatology bin-edge/reference-run numbers and edge-case behaviour (out-of-range invariant, no-switch-step), a pre-registered absolute sharpness threshold replacing an undefined "bottom-decile," and the seven JU-10 disclosures authored verbatim (ADR-J7, new) instead of left as a placeholder. See design-review-001.html for the full findings.

**What this document is.** The design of the product itself: every metric the judge computes, precisely enough to build from — the skill score, the calibration diagnostic, the scoring rule, sharpness, the climatology switch, the trust horizon, and the exception bands. This spec settles Open Questions 1 and 2 (bands and sample size, derived together) and the runtime half of Open Question 5.

**In plain words:** this is the arithmetic of the verdict. Every number the judge produces is defined here before any model is judged, down to the formula — so when the verdict says "amber", the only remaining question is what the data was, never what the rule was.

---

## Expert Panel

| Expert | Work | Role in This Document |
|--------|------|----------------------|
| Tilmann Gneiting & Adrian Raftery | *Strictly Proper Scoring Rules* (JASA 2007) | The CRPS choice (ADR-J1) and its anti-gaming property |
| Gneiting, Balabdaoui & Raftery | *Probabilistic Forecasts, Calibration and Sharpness* (JRSS-B 2007) | Calibration and sharpness as distinct, jointly required checks (ADR-J2, ADR-J3) |
| Allan Murphy | *What Is a Good Forecast?* (1993) | Skill-score form: always relative to a reference |
| A. Philip Dawid | The prequential principle (1984) | The judge's input contract: predictions and outcomes only |
| Edward Lorenz | *Deterministic Nonperiodic Flow* (1963) | The divergence-relative grading and the JU-6 switch |
| Jolliffe & Stephenson | *Forecast Verification* (2nd ed.) | Independent-trials discipline in exception counting |
| Federal Reserve / OCC | *SR 11-7* (2011) | The band structure: exceptions counted against ex-ante thresholds (public, generic description only — NF-4) |
| Michael Keeling | *Design It!* | ADR format; boundary-decision capture (design-review Panel B/C) |

---

## 1. Purpose

Covers requirements **JU-1 through JU-13**. The judge package (`wmj.judge`) is a set of pure functions (JU-12) over plain arrays (JU-1) producing one structured verdict (JU-9) with a limitations statement (JU-10), computed against thresholds fixed in advance (JU-11), containing no learned component (JU-13, Won't).

## 2. Architecturally Significant Requirements

- **JU-12 + NF-6**: the judge is a leaf package — pure functions, no I/O, imports stdlib + NumPy only. Everything it needs arrives as arguments; everything it produces returns as values. This now includes `meta` fields (platform, prereg commit) that are environment reads — they are explicitly **not** part of the judge's own output (§5).
- **JU-1**: the input type is defined *inside* the judge package and has no field that could carry identity — blindness by construction, not by discipline. The judge's *output* type carries the same discipline: it cannot contain `is_fixture` or a model name either, since those are exactly the facts JU-1 forbids the judge from having received in the first place.
- **JU-11 + OQ-1/OQ-2**: bands must be *derived* from a declared test and sample size. This spec fixes the derivation; the resulting integers are computed once and committed in `prereg/thresholds.json` (models spec ADR-M5 mechanics).
- **JU-9**: the verdict record is the reporting layer's entire input — its schema is the project's most important API boundary contract (§5).

## 3. Design Decisions

### ADR-J1 — Scoring rule: CRPS, closed-form Gaussian, skill relative to baselines

**Status:** Accepted. [Requirement: JU-2, JU-4(b)] [Test: TC-JU2-01, TC-JU4-02]

**Context:** JU-4(b) requires a strictly proper scoring rule so misstated confidence cannot improve the summary score. JU-2 requires accuracy always reported as skill relative to both baselines.

**Options considered:**
1. **CRPS (Continuous Ranked Probability Score)** — strictly proper for distributions with finite first moment; closed form for Gaussians: `CRPS(μ,σ;y) = σ·[z(2Φ(z)−1) + 2φ(z) − 1/√π]`, z=(y−μ)/σ. In the units of the state (we use normalised units), robust to outliers, degrades gracefully as σ→0.
2. **Logarithmic score (Gaussian NLL)** — strictly proper; but unbounded penalties explode on a single bad out-of-region prediction with small σ, letting one step dominate a summary — bad fit for a toy sample size.
3. **Interval/Winkler score** — proper for a single interval level, narrower in what it validates; retained implicitly via the calibration diagnostic instead.

**Decision:** CRPS (option 1), computed per dimension on normalised state (worlds-spec scale vectors), averaged over dimensions.

**Φ and φ, exactly (design-review fix — the original text imprecisely attributed both to erf, and never addressed array shape):**
- `Φ(z) = 0.5 · (1 + erf(z/√2))`, from stdlib `math.erf`.
- `φ(z) = (1/√(2π)) · exp(−z²/2)`, from stdlib `math.exp` and `math.pi` — **not** from `erf`; the earlier draft's claim that both come from erf was imprecise.
- **Vectorization:** `JudgeInput` arrays are NumPy-shaped `[n_trials, H, d]`, but `math.erf`/`math.exp` are scalar-only. Both are wrapped once, in `wmj/judge/_normal.py`, via `numpy.frompyfunc(math.erf, 1, 1)` (and a native `numpy.exp` call for φ, which *is* already array-native) — no SciPy dependency (NF-3).

**Skill form (Murphy, JU-2):** `skill = 1 − CRPS_model / CRPS_baseline`, reported against persistence and linear separately, per task, per region, **at the one-step-ahead horizon `h = 1` — a fixed single step, NOT the task's `h_task` (design-review-006 repair — Round 6 found the v1.5 block×axis table's gloss "the step is fixed by the task" reworded this as if it varied with the task like calibration/sharpness, contradicting the literal "one-step" that models spec ADR-M3's MU-5 matching margin, TC-MU2-02, and TC-JU2-01 all depend on; `h=1` is now stated explicitly here and in the §5 table).** The one-step CRPS is the apples-to-apples ranking quantity Chart 4 uses and the MU-5 margin gates. No absolute error is ever emitted without its skill counterpart (TC-JU2-01); the strictly-proper property is verified by TC-JU4-02's misstatement test.

### ADR-J2 — Calibration diagnostic: central-interval coverage at four declared levels

**Status:** Accepted. [Requirement: JU-4(a)] [Test: TC-JU4-01]

**Decision:** Predictions are Gaussian per dimension (MU-1 mean/spread). The diagnostic computes observed coverage of central intervals at levels **{50%, 80%, 90%, 95%}**: for level p, the interval is μ ± z_(p)·σ per dimension.

**z_p values (design-review fix — erf gives the forward normal CDF only; there is no general inverse available without an iterative routine this project doesn't otherwise need):** fixed constants, not computed by a general inverse-CDF, since exactly four levels are ever used:

| p | z_p |
|---|---|
| 0.50 | 0.6745 |
| 0.80 | 1.2816 |
| 0.90 | 1.6449 |
| 0.95 | 1.9600 |

(Each is Φ⁻¹((1+p)/2), the standard two-sided central-interval multiplier, taken from published normal-quantile tables and hardcoded in `wmj/judge/_normal.py` alongside the Φ/φ helpers.)

An outcome "covers" if *all* dimensions fall inside (the joint event is what a user of the prediction experiences; per-dimension coverage is also recorded for diagnosis). Coverage is reported as one entry per (task, region) — explicit key fields per the §5 canonical keying rule (design-review-004 repair: v1.3 keyed by task but smuggled region into `_in_region`/`_out_region` field-name suffixes, the exact suffix-encoding the keying rule now bans) — computed over that region's independent-trial set (§ADR-J4 sampling; the trial set is shared across the region's tasks). **Each (task, region) entry's coverage is evaluated at the task's single declared evaluation horizon `h_task` (the task's JU-6 switch step; control and planning differ precisely because their `h_task` differ), so per-task coverage genuinely differs while the block itself does not carry a `horizon_step` axis** — design-review-005 repair: Round 5 found the v1.4 text said "at its own declared horizon *steps*" (plural), which is unbuildable against the block×axis table's claim that calibration does not vary over `horizon_step`, and unreadable against Chart 3's one-coverage-line-per-(task,region) design. Pinning to the single `h_task` makes the "—" in the §5 table true and matches the chart consumer. (Exception counting, ADR-J4, separately reports both h=1 and h_task and does carry `horizon_step`; calibration is the distinct JU-4 diagnostic at the task's own horizon.) The diagnostic and the ADR-J1 summary are distinct fields in the verdict and never combined (JU-4's "never conflate", TC-JU4-01).

### ADR-J3 — Sharpness: mean 90% interval width, reported beside calibration

**Status:** Accepted. [Requirement: JU-5] [Test: TC-JU5-01, TC-JU5-02]

**Decision:** Sharpness = mean over trials and dimensions of the stated 90% central-interval width (2·z₀.₉₀·σ, using the z₀.₉₀ = 1.6449 constant from ADR-J2), **at the task's single evaluation horizon `h_task`** (design-review-005 — same pinning as calibration, ADR-J2: sharpness qualifies the coverage it sits beside, so it is measured at the same step; the block carries no `horizon_step` axis, matching the §5 table's "—"), in normalised units, one entry per (task, region) with explicit key fields (§5 canonical keying rule — design-review-004 repair replaced v1.3's `_in`/`_out` suffix fields; ADR-J4's own hedging cross-check reads "that region/task's mean 90%-interval width," which is now a direct lookup). Smaller is sharper; it is monotone in stated σ by construction (TC-JU5-02's property). Reported in the same verdict block as calibration so neither is readable without the other (GBR: maximise sharpness subject to calibration), **and rendered beside it on Chart 3 (reporting ADR-R2, design-review-004 repair — Round 4 found the sharpness numbers had no chart consumer anywhere, leaving JU-5's "reported alongside calibration" unmet at the surface Dev actually reads)**.

### ADR-J4 — Exception counting and bands: binomial acceptance regions at N = 200 (settles OQ-1 and OQ-2)

**Status:** Accepted. [Requirement: JU-8, JU-11] [Test: TC-JU8-01, TC-JU8-02, TC-JU8-03]

**Context:** JU-8 demands exceptions counted over statistically independent trials, compared to the model's implied rate with a declared test at a declared sample size, against bands fixed in advance with declared false-alarm probabilities. Open Questions 1 and 2 defer the numbers to this spec.

**The declared design:**
- **Independent trials: one shared set of N = 200 evaluation starts per (world, region), reused identically across every model and across that region's tasks** (design-review-004 pinned the shared-across-tasks half; design-review-005 pins the shared-across-*models* half — Round 5 found "per (model, world, region)" still read as each model drawing its own 200 starts, which makes MU-5's headline direct-vs-ensemble skill-margin an *unpaired* comparison with far higher sampling noise. The paired design is pinned: the harness generates the 200 starts once per (world, region) from a content-addressed seed (cross-cutting ADR-002 rule 2, keyed on world+region, **not** on model), and every model — baselines, unrigged pair, fixtures — is rolled out from those identical starts. This is what makes the matching-margin and the Chart-4 ranking apples-to-apples). Each task evaluates the same 200 trials at its own horizon steps and tolerance. JU-8's independence requirement is between *trials*, which holds; the two tasks' exception counts are consequently correlated *with each other* — disclosed here, and harmless, since the binomial test is within-task. Each trial is a rollout from its own evaluation starting condition (disjoint from training and from each other — models spec MU-7 mechanics) with its own seeded action sequence. Trials are represented as the first axis of every `[n_trials, H, d]` array in `JudgeInput` (§4) — a trial's own row *is* its boundary; there is no flat/concatenated representation and therefore no separate boundary-marker field anywhere in the contract (design-review fix: an earlier draft's cross-cutting/models documents named a `trial_boundaries` array that this pre-shaped-array design makes unnecessary and undefined — removed there, see cross-cutting and models changelogs).
- **Exception definition:** at each **pre-declared horizon step h ∈ {1, h_task}** (h_task = the task's JU-6 switch step, per task), the trial's outcome at step h either falls inside the model's stated 90% central interval (all dimensions, ADR-J2 joint convention) or is an exception. One observation per trial per horizon — never pooled within a rollout (TC-JU8-01 rejects correlated pooling by construction: each trial occupies its own row in the array's trial axis, and the counter consumes one designated step per trial, never steps from two different trials' rows treated as if independent).
- **The test:** exact two-sided binomial test against p = 0.10 at n = 200. Expected exceptions: 20.
- **Bands, derived (not chosen by eye — JU-11):**
  - **Green** — count inside the central 95% binomial acceptance region for Bin(200, 0.10): **[12, 29]**.
  - **Amber** — outside green but inside the central 99.9% region: **[8, 11] or [30, 35]**.
  - **Red** — **≤ 7 or ≥ 36**.
  - False-alarm probability for a perfectly calibrated model, from the exact binomial CDF: **3.3%** of landing outside green (declared bound ≤ 5%), **0.087%** outside amber (declared bound ≤ 0.1%) — declared per band, per JU-8. The integer boundaries above are computed from the exact binomial CDF by a build-time derivation script whose output is committed to `prereg/thresholds.json`; the committed file is what the judge reads (via its arguments), and TC-JU8-02 checks a known-calibrated fixture lands green at the declared rate.
  - **Low-side counts are flagged, not rewarded, against an absolute pre-registered threshold, not a decile (design-review fix):** a green-or-better count is cross-checked against sharpness (ADR-J3); if that region/task's mean 90%-interval width exceeds a fixed value declared in `prereg/thresholds.json` (`sharpness_hedge_threshold`, set per world from the world's own scale vector at pre-registration time, before any model is judged — worlds spec ADR-W3 defines the scale), the JU-5 cross-flag fires (TC-JU8-03). The original "bottom-decile" wording is removed: with 7 contestants per (world, region) — design-review-004 count correction, see models spec ADR-M1 — a decile has no statistical meaning; an absolute, pre-registered width bound does.

**Why N=200:** at p=0.10 the green region [12,29] distinguishes a true 90% model from a fixture at 70% true coverage (expected 60 exceptions — deep red) and from a padded 99% model (expected 2 — red on the low side) with power ≈ 1; it is the smallest round sample where the amber band is non-degenerate at both ends, and it fits the NF-2 budget with two orders of magnitude to spare (§ADR-J6). Assumption 2 of the requirements (independent trials shrink the usable sample) is answered by construction: the 200 are independent starts, not steps.

### ADR-J5 — The JU-6 switch and conditioned climatology

**Status:** Accepted. [Requirement: JU-6, JU-3, JU-7] [Test: TC-JU6-01, TC-JU6-02, TC-JU3-01, TC-JU7-01, TC-JU7-02]

**The shared distance metric (design-review fix — this was asserted in worlds.md and architecture-overview.md but never actually restated or cited here, despite every formula in this document otherwise being given in full):** every "normalised error" and "normalised distance" in this ADR is worlds spec ADR-W3's metric, restated here so the two specs cannot silently drift apart: RMS over state dimensions each divided by the world's declared scale vector (worlds.md §4) — `distance(a, b) = sqrt(mean_d(((a_d − b_d) / scale_d)²))`. This is the *same* function used for WD-4's divergence curves, JU-3's error-vs-horizon, and every task's tolerance τ. It is a different metric from CRPS (ADR-J1): CRPS scores the full predictive distribution for the anti-gaming skill summary; this RMS distance scores point predictions against tolerances and divergence. Both are deliberate and distinct — see architecture-overview v1.1's corrected conceptual-integrity note.

**Decision:**
- **Switch step per task and region (design-review-003 fix — Round 3 found `switch_step` was schema'd per-task only, even though this very sentence defines it against "the trial's region's curve," and divergence curves differ materially by region — ADR-W3's whole point):** the first step where the world's own divergence curve (WD-4, the trial's region's curve) exceeds the task tolerance τ. Before it: trajectory grading (normalised error vs the divergence reference, JU-3, using the distance metric above). At and after it: statistical grading. The verdict records the switch step per task **and region** (TC-JU6-01). **If the divergence curve never exceeds τ within the world's declared horizon (design-review fix — previously undefined):** `switch_step = null` in the verdict, the task is graded by trajectory error for its entire horizon, and the verdict states explicitly ("this task never left trajectory-grading within the horizon") rather than leaving `trust_horizon`'s formula (below) without an upper bound.
- **Conditioned climatology, concretely (design-review fix — bin-edge method and reference-run length were previously unstated numbers in a document that otherwise pins every constant):** the reference statistics (per-dimension mean and standard deviation of state) of the true world *restricted to the invariant's value*, computed by binning **one continuous 200,000-step reference trajectory per world** (harness-generated, null action, from the world's training-region centre — long enough that every bin below has ≥50 samples for both worlds, checked by an assertion at generation time) by conserved-quantity value into **16 equal-population bins** (not equal-width — equal-population guarantees every bin has comparable sample size regardless of how the invariant's long-run distribution is shaped, which an equal-width scheme cannot guarantee for the pendulum's energy, whose long-run density is not uniform). At each compared step, the invariant is *measured from the true trajectory at that step* (TC-JU6-02's re-measured-not-frozen check; both action impulses and bounded integrator drift move the bin, per WD-3/JU-6). **Out-of-range invariant values (design-review fix — a real case: WD-5's out-of-region starts and JU-6's own action-driven excursions can both reach invariant values the reference run never visited):** the outermost bins are unbounded on their open side (the lowest bin's floor and the highest bin's ceiling are ±∞), so every possible invariant value falls in exactly one of the 16 bins by construction — no value is ever "outside the bins."
- **Statistical agreement score:** past the switch, the model's predicted mean at each compared step is scored against the conditioned climatology as a standardised deviation `(pred_mean − clim_mean)/clim_sd`, averaged per dimension over the post-switch window; agreement holds when the average |z| ≤ 1. The threshold is part of `prereg/thresholds.json`.
- **Trust horizon (JU-7), now per region (design-review-002 fix — Round 2's Structural panel found the v1.1 formula and schema had no region axis, unlike `exceptions.per_task` and `error_vs_horizon`, which defeated the one thing `fx-brittle` exists to demonstrate: a region-conditioned collapse invisible in an aggregate number):** `trust_horizon(model, world, task, region) = the largest step h such that the model's median normalised error across trials in that region at every step ≤ h stays within τ, capped at switch_step if switch_step is not null (computed per region, since the divergence curve itself is per-region — worlds spec ADR-W3), else capped at the world's declared horizon H`. If the model fails tolerance even at step 1, `trust_horizon = 0` (an explicit, not implicit, convention — design-review fix). Always reported as (steps, steps × dt in world time units, and `natural_units` — **nullable, design-review-004 repair**: a fraction of the world's declared natural cycle where one exists (LV's near-equilibrium period, worlds spec §4.1) and `null` where none does (the double pendulum is chaotic and has no natural period — worlds spec §4.2 now states this absence explicitly rather than leaving the field's pendulum value undefined; Chart 4 renders `null` as "—", reporting ADR-R2). JU-7's own wording — "seconds, *or* a fraction of the world's natural cycle" — is satisfied for the pendulum by the seconds already present in `world_time`), and always attached to its task name, its region, and τ (TC-JU7-01/02, updated to check region presence). It is capped at the switch step by construction — beyond it, exact-path trust is not a claim anyone can earn (Lorenz).

### ADR-J6 — Runtime budget (settles OQ-5's runtime half)

**Status:** Accepted. [Requirement: NF-2] [Test: TC-NF2-01]

**Decision:** The full result set — benchmarks, training both unrigged models, all rollouts, judging, charts — completes in **under 600 seconds wall-clock on a 4-core consumer laptop CPU** (TC-NF2-01's `<spec-value>` = 600 s). Envelope (design-review-005 correction — the v1.4 figure omitted the per-region multiplier ADR-J4 requires: 200 trials are drawn *per region*, and each world has 2 regions, so the true rollout count is ~2× the v1.4 number): 200 trials × **2 regions** × (700 + 5000) steps × 7 models ≈ **16M** model-step calls, each a 2×64×64 MLP forward ≈ tens of kFLOPs — still well under a minute of compute; training two small MLPs for a pinned epoch count dominates and is budgeted at ≤ 6 minutes combined. Even at the corrected 16M the compute budget holds with two orders of magnitude to spare; the headroom claim (≥ 30% on the reference machine class) survives the correction.

### ADR-J7 — The seven JU-10 disclosures, authored verbatim

**Status:** Accepted. [Requirement: JU-10] [Test: TC-JU10-01, TC-JU10-02]

**Context (design-review fix):** JU-10 and TC-JU10-01 both name seven required disclosures, but every prior draft of this spec only *counted* them ("`limitations.py` # the seven JU-10 disclosures as fixed text constants") without ever writing the text — the Verdict schema literally showed a placeholder string. This project's own pattern (reporting.md ADR-R3 authors every chart caption verbatim, specifically so caption quality is reviewable before code exists) applies at least as strongly here, since JU-10 is one of the most credibility-critical requirements in the document.

**Decision — the seven fixed strings, stored as module-level constants in `wmj/judge/limitations.py` and emitted unchanged in every verdict's `limitations` array, in this order:**

1. `"This method validates the middle of the distribution, not the extremes — it says nothing about how the model behaves on rare, disaster-scale events."`
2. `"This toy world validates the judging harness, not any real-world model — a clean result here is not evidence that a real world model is trustworthy."`
3. `"The judge, the models, and the thresholds share a single author. The blinding in JU-1 means the judge cannot favour a model for its identity — it does not mean this evaluation has banking's organisational independence: a separate team, reporting separately, empowered to challenge the model builder."`
4. `"This project borrows banking's backtesting and ex-ante-threshold practices, not its ongoing monitoring or its power to force a stop. Deciding who has the authority to act on a bad verdict is exactly the gap this project's source essay names — it is not solved here."`
5. `"The judge's own thresholds and metric choices are modelling decisions. They were fixed in advance and are not arbitrary, but they are not beyond challenge either."`
6. `"The toy worlds contain no genuine off-model surprises by construction (no randomness, no hidden state, no high-dimensional input) — this judge's behaviour when a model meets a real surprise is untested."`
7. `"'Control' and 'planning' name the tolerance regime used to grade a passive prediction, not a closed-loop decision. This judge does not test whether a model can choose its own actions to reach a goal — the actions it is graded against were chosen by someone else."`

`limitations.py` exports these as a tuple `JU10_DISCLOSURES` with no parameters and no formatting logic — they are complete, final English sentences, identical in every verdict regardless of model or world, per JU-10's "at minimum" wording (a verdict may add more, per-run detail elsewhere, but never fewer or reworded versions of these seven).

## 4. Component Design

```
wmj/judge/
  types.py        # JudgeInput, Verdict, Task/ToleranceView — frozen dataclasses, arrays+floats only,
                  # no name/id/provenance fields anywhere (JU-1; TC-JU1-01)
  _normal.py      # Phi, phi, and the four fixed z_p constants (ADR-J1, ADR-J2)
  skill.py        # CRPS closed form, skill scores vs both baselines (ADR-J1)
  distance.py     # the shared normalised-RMS distance metric (ADR-J5), imported by skill/climatology/horizon
  calibration.py  # coverage at the four levels, in/out region (ADR-J2)
  sharpness.py    # interval widths (ADR-J3)
  exceptions.py   # trial-wise exception counts, binomial band assignment, hedge cross-flag (ADR-J4)
  climatology.py  # switch step, conditioned climatology agreement, out-of-range/no-switch handling (ADR-J5)
  horizon.py      # trust horizons (ADR-J5)
  verdict.py      # assembles the pure Verdict (JU-9); refuses on any missing input (TC-JU9-02)
  limitations.py  # the seven JU-10 disclosures, verbatim (ADR-J7)
```

Every public function: pure, typed, arrays in → values out. The package imports `numpy`, `math`, `dataclasses`, `typing` — nothing else. Nothing in this package reads a file, a clock, or an environment variable — `platform`, `prereg_commit`, and any other run-identifying metadata are the harness's concern, not the judge's (§5).

**Purity enforcement — TC-JU12-01 is the load-bearing control, specified concretely (design-review-005 repair — Round 5 executed reflection escapes of the static AST gate and found TC-JU12-01, its named backstop, had only a prose Given/When/Then that intercepted imports/builtins, not the *effects* those escapes reach; cross-cutting ADR-002 rule 3 now makes the runtime harness the load-bearing control and the static gate mere lint).** TC-JU12-01 runs the judge's full call graph — every public function, on a representative real `JudgeInput` — inside a context manager that replaces every ambient capability with an object raising `AmbientAccessError` on **any** use, so an impure operation fails at the point of its *effect*, however the code reached it:
- `builtins.open`, `io.open`, and `os` (the whole module object in `sys.modules`, so `os.environ`, `os.system`, `os.getcwd`, etc. all raise) — catches filesystem/OS access reached even via `().__class__.__base__.__subclasses__()` or `<fn>.__globals__['os']`, because the guarded object is the one real shared module, not a name the judge typed;
- `time`, `datetime` clock entry points; `socket` / any network entry; `subprocess`;
- **randomness:** `numpy.random`'s `default_rng`/`Generator` construction and the legacy globals are monkeypatched to raise (the judge needs no randomness — JU-12), and the run asserts the judge consumed none.
Because the guards are on the shared capability objects (patched in `sys.modules` / the real `builtins`), not on identifiers the AST could name, a reflection route that recovers `os` still ends at the same guarded `os` object and raises. The test asserts: the judge produces the correct verdict from its arguments alone, and no guard fired. Its own phantom-gate case (TC-JU12-02, new) injects a deliberate `os.environ` read reached via `__globals__` into a judge function and asserts the guard fires — proving the harness catches the exact Round-5 escape class, not just syntactic imports. This is what makes the runtime harness *complete* where the static lint (TC-NF6-01/02, which remains as fast pre-check) cannot be.

**JudgeInput (complete):** predictions (mean, spread) `[n_trials, H, d]`, outcomes `[n_trials, H, d]`, baseline predictions (same shapes, one set per baseline), **`region_labels: [n_trials]`, one `{region_name: str, axis: "state"|"action"|"both"|null}` per trial (design-review-003 fix — Round 3 found three mutually inconsistent definitions of this field across worlds.md, models.md, and this document's own prose; this is now the single, canonical shape all three cite: `region_name` is the join key reporting and the climatology lookup use to pick a specific named curve/table — a boolean in/out flag alone can't do that once a world declares more than one out-region, which WD-5 explicitly permits — and `axis` is WD-5's own "which axis" categorical field (worlds.md ADR-W4), `null` when the trial is fully in-region)**, divergence curves per region **`[H+1]`** (design-review fix: matches the worlds-spec artefact's own `H+1`-length, 0-indexed convention exactly — see worlds spec §5 — rather than the previously mismatched `[H]`), conditioned-climatology table (16 equal-population bins → mean[d], sd[d], each bin's invariant range) + per-step true invariant bins `[n_trials, H]`, task list (name, kind, τ, horizon), thresholds (bands, sharpness-hedge threshold, agreement threshold — **not** the interval levels: design-review-006 fix — Round 6 found §4 listed "interval levels" as harness-supplied `JudgeInput.thresholds` data while ADR-J2 defines the four levels {0.50, 0.80, 0.90, 0.95} and their z_p as constants hardcoded in `wmj/judge/_normal.py`; a fact must have one producer, so the levels are judge constants, removed from the input here), world time step dt and natural-cycle length. Nothing else — by type, nothing else *can* be passed (TC-JU1-01), and label-swap invariance is property-tested (TC-JU1-02).

## 5. API Boundary Contracts — the Verdict record (JU-9) and the harness envelope

**The judge produces exactly one thing: a pure `Verdict`, containing only what its own arithmetic computed.** Design-review Panels B and C both independently found the previous draft's Verdict schema carrying `is_fixture`, `model_ref`, and `meta.platform`/`meta.prereg_commit` — fields the judge cannot honestly produce (JU-1 forbids it from ever knowing `is_fixture` or a model's identity; JU-12 forbids it from reading the platform or git state). Those fields move to a **harness-owned envelope** that wraps the pure Verdict after the fact. There is now exactly one producer for each fact: the judge for everything inside `Verdict`, the harness for everything identifying *whose* verdict it is.

**The pure `Verdict` (serialized canonically, cross-cutting ADR-002):**

```json
{
  "schema": "wmj-verdict/1",
  "world": "lv",
  "skill": {"per_task_region": [{"task": "lv-control", "region": "training",
    "vs_persistence": 0.41, "vs_linear": 0.28, "crps": 0.031}]},
  "error_vs_horizon": {"dt": 0.02, "per_region": [{"region": "training",
    "steps": [0, 1, 2, "..."], "median_error": ["..."],
    "divergence_reference": ["..."]}]},
  "calibration": {"per_task": [{"task": "lv-control", "region": "training",
    "levels": [0.5, 0.8, 0.9, 0.95],
    "coverage": [0.52, 0.79, 0.9, 0.94],
    "n_trials": 200,
    "per_dimension": "..."}]},
  "sharpness": {"per_task": [{"task": "lv-control", "region": "training",
    "mean_width_90": 0.18}]},
  "exceptions": {"per_task": [{"task": "lv-control", "region": "training", "horizon_step": 1,
    "n_trials": 200, "expected": 20, "observed": 27, "band": "green",
    "bands": {"green": [12, 29], "amber": [[8, 11], [30, 35]], "red": "outside"},
    "low_side_sharpness_flag": false}]},
  "trials": {"per_task": [{"task": "lv-control", "region": "training", "horizon_step": 1,
    "distance_unit": "rms-normalised (ADR-W3/ADR-J5, scalar even for multi-dimensional state)",
    "outcome_distance": [0.08, "...(200 values, one per trial — distance from this trial's mean prediction to its true outcome; a scalar SUMMARY for charting only, see note below)"],
    "band_lo": [0.0, "...(always 0 — a distance is never negative)"],
    "band_hi": [0.16, "...(the trial's stated 90% interval width in the same distance unit, from spread via ADR-J2/J3's z-multiplier; also charting-only, see note below)"],
    "is_exception": [false, "...(the ADR-J4 per-dimension joint containment test's actual result for this trial — NOT derived from comparing outcome_distance to band_hi, see note below)"]}]},
  "climatology": {"per_task": [{"task": "lv-planning", "region": "training", "switch_step": 214,
    "agreement_mean_abs_z": 0.6, "agrees": true}]},
  "trust_horizons": {"per_task": [{"task": "lv-control", "region": "training", "tolerance": 0.1,
    "steps": 118, "world_time": 2.36, "natural_units": "0.34 cycles"},
    {"task": "lv-planning", "region": "training", "tolerance": 0.3,
    "steps": 41, "world_time": 0.82, "natural_units": "0.12 cycles"}]},
    // ^ design-review-006 fix: both entries are lv tasks — this whole example is ONE (model, world="lv")
    //   Verdict. The v1.5 example listed a "dp-control" (double-pendulum) entry inside an "lv" verdict,
    //   self-contradicting the one-Verdict-per-(model,world) contract (Round 6). The pendulum's own
    //   verdict is where natural_units is null (ADR-J5) — the pendulum has no natural cycle;
    //   that null case is illustrated in ADR-J5's own text, not smuggled into this lv example.
  "not_tested": ["genuine off-model surprises (WD-8)", "closed-loop action selection",
    "ongoing monitoring", "the seven JU-10 disclosures cover the rest of this list"],
  "limitations": ["<the seven ADR-J7 strings, verbatim, in order>"]
}
```

**Changes from the pre-review draft:** `model_ref`, `is_fixture`, and `meta` are removed (moved to the envelope, below). `error_vs_horizon.steps` now starts at 0, matching the worlds-spec divergence artefact's own indexing exactly (design-review Important #1) — there is one shared step-zero origin for every step-indexed curve in the system. `exceptions.per_task` entries now carry a `region` field, matching the convention `calibration` and `error_vs_horizon` already used (design-review Important #2). A new `trials` block carries the per-trial outcome/band/exception arrays reporting's Chart 1 (the primary chart) actually needs to draw the 200 individual points and their bands — the pre-review schema only had aggregate counts, which cannot render that chart (design-review Critical finding on RP-1). `not_tested`'s example list is fully stated rather than trailing off with "..." (design-review Minor). `trust_horizons.per_task` now carries a `region` field, and `calibration` now carries `n_trials` (design-review-002 fixes, below).

**`trials.is_exception` and `exceptions.per_task.observed` are one fact, not two (design-review-002 fix — Round 2's Structural panel found these were two independently-computable, unreconciled definitions of "exception": a scalar distance-vs-band-width comparison in `trials`, and a per-dimension joint containment test in ADR-J4's `exceptions`, with nothing stating they must agree).** They do not: `exceptions.py` computes ADR-J4's per-dimension joint containment test once per trial, per the ADR-J4 definition ("falls inside the model's stated 90% central interval, all dimensions, or is an exception"); that same per-trial boolean is what populates `trials.per_task[...].is_exception` — it is copied, never re-derived. `outcome_distance` and `band_lo`/`band_hi` in the same `trials` block are a *separate, scalar summary of the same trial*, computed only for Chart 1's visualisation (projecting a d-dimensional prediction/outcome pair onto one axis) and must never be used to compute `is_exception` by comparing them to each other — that comparison is not equivalent to the joint test and would silently diverge from it. `exceptions.per_task.observed` for a given (task, region, horizon_step) is defined as `sum(trials.per_task[same key].is_exception)` — a derived count, not an independent computation — so RP-1's on-chart misses and its header-strip count are structurally the same number by construction, not by convention. **This is now a build-enforced invariant, not only a stated one (design-review-003 fix — Round 3 found the v1.2 prose named the correct relationship but no test would catch a regression back to two independent computations):** TC-JU9-03 (new) asserts, for every (task, region, horizon_step) key present in a computed Verdict, `exceptions.per_task[key].observed == sum(t.is_exception for t in trials.per_task if same key)` — a property test, not an example-based one, so it holds for every verdict a run produces, not just a hand-picked case.

**The canonical keying rule (design-review-004 structural repair — this rule replaces four rounds of block-by-block patching with one convention, verified once, in the table below):** every metric block in the Verdict is an array of entries carrying an **explicit string key field for each axis its fact genuinely varies over** — `task`, `region`, `horizon_step` — as determined by the block's own defining ADR. No block may encode an axis in field-name suffixes (the `_in_region`/`_out_region` and `_in`/`_out` forms Rounds 3–4 caught are banned outright: a suffix hardcodes exactly one out-region, which worlds spec §5 explicitly promises not to assume). An axis a fact does *not* vary over is omitted, not faked — the table states why in each case, so "missing key" and "doesn't vary" are never confusable again.

| Block | task | region | horizon_step | Why, per the defining ADR |
|---|---|---|---|---|
| `skill.per_task_region` | ✓ | ✓ | — | ADR-J1: skill computed at the fixed one-step-ahead horizon **`h = 1`** for every task/region (a constant, not `h_task`, and not an axis — design-review-006: this is the literal "one-step" score MU-5's matching margin and TC-MU2-02/TC-JU2-01 require) |
| `error_vs_horizon.per_region` | — | ✓ | — (the curve *is* the step axis) | ADR-J5/JU-3: the median-error curve depends on region's trials only; τ (task) never enters its computation — the task-dependent facts (switch steps) live in `climatology`, and Chart 2 overlays them from there (reporting ADR-R2). **The block also carries a scalar `dt`** (the world's step size, design-review-005 repair — Round 5 found Chart 2's mandated "world time" second axis had no data source: the judge receives `dt` in `JudgeInput` and passes it through here so reporting can render world-time = step × dt without importing `wmj.worlds`; `dt` is a block-level constant, not a per-entry axis) |
| `calibration.per_task` | ✓ | ✓ | — | ADR-J2: computed over the region's trial set at the task's *single* evaluation horizon `h_task` — one coverage per (task, region), so `horizon_step` is not an axis (design-review-005: the step is fixed by the task, not varied) |
| `sharpness.per_task` | ✓ | ✓ | — | ADR-J3: mean 90%-width at the same `h_task`; ADR-J4's hedge cross-check reads "that region/task's" width as a direct lookup |
| `exceptions.per_task` | ✓ | ✓ | ✓ | ADR-J4: counts at h ∈ {1, h_task} per task per region |
| `trials.per_task` | ✓ | ✓ | ✓ | same trial-level facts behind `exceptions` |
| `climatology.per_task` | ✓ | ✓ | — | ADR-J5: switch step is defined against the region's divergence curve at the task's τ |
| `trust_horizons.per_task` | ✓ | ✓ | — | ADR-J5: per (task, region) by formula |

This table is the mechanical check BC-1 demanded: any future block, or edit to one, is verified against it by inspection, not re-derived chart-by-chart. Two consequences of applying the rule this pass: `error_vs_horizon` became a `per_region` array (the v1.3 flat single object could not even hold both regions' curves — a latent bug independent of keying style), and `calibration`/`sharpness` lost their suffix fields in favour of one `coverage`/`mean_width_90` value per (task, region) entry.

**The harness-owned envelope (`wmj/harness/results.py`, NOT part of the judge package):**

```json
{
  "model_ref": "opaque-index-3",
  "model_name": "ensemble",
  "is_fixture": false,
  "verdict": { "...the pure Verdict above..." },
  "meta": {"seed": 20260825, "platform": "linux-x86_64", "prereg_commit": "<sha>",
    "thresholds_file": "prereg/thresholds.json"}
    // platform is the PINNED coarse composite `f"{sys.platform}-{platform.machine()}"`
    //   (e.g. "linux-x86_64") — design-review-006 fix: Round 6 found the source API was unnamed
    //   and the natural `platform.platform()` embeds kernel release + glibc version, so two real
    //   machines of "the same platform" (NF-1's own in-scope case) would differ on this one field
    //   and fail byte-identity. The coarse composite matches NF-1's "same platform" granularity exactly.
}
```

The harness constructs one envelope per (model, world) immediately after calling the judge; this envelope — never the bare `Verdict` — is what's written to `out/verdicts/*.json` (via the canonical serializer, cross-cutting ADR-002 rule 4) and handed to reporting. `model_ref` is an opaque index; the judge never sees a name or a fixture flag at any point in its call graph. **All nine JU-9 field groups are mandatory inside the pure `Verdict`** (design-review-002 made it eight; design-review-006 adds `limitations` as the ninth — Round 6 found judge.md's own "eight" enumeration excluded `limitations` while reporting.md's model card called it "one of the eight mandatory groups"; since `limitations` is the JU-10 honesty field the whole project treats as load-bearing, it must be refuse-on-missing like the rest, not a silently-optional field): skill, error-vs-horizon, calibration+sharpness, exceptions, **trials**, **climatology**, trust horizons, not-tested, **limitations**. `verdict.py` raises rather than emitting a record with any of the nine missing or null (TC-JU9-01, TC-JU9-02). (`limitations` is a hardcoded constant tuple, ADR-J7, so it will not in practice go missing — but the refuse-guard now covers it by contract, matching reporting.md's consumer-side claim.)

## 6. Integration Points

- **← harness:** builds `JudgeInput` from world artefacts + model rollouts; owns the prereg check (models spec ADR-M5) and passes `thresholds` as data — the judge never reads files (JU-12).
- **→ harness:** returns the pure `Verdict`; the harness (not the judge) wraps it in the envelope described in §5 before it reaches reporting or disk.
- **→ reporting:** consumes only the harness's `{model_ref, model_name, is_fixture, verdict, meta}` envelopes — never a bare `Verdict` and never a separately-maintained side map (the previous draft's "index→name/fixture map" is retired; the envelope carries everything reporting needs in one object, per Keeling's boundary-decision-capture principle).
- **Bands derivation script** (`wmj/harness/derive_thresholds.py`): computes the binomial boundaries from (N, p, α-levels) and the per-world sharpness-hedge threshold, and writes `prereg/thresholds.json`; run once, committed, never re-run after judging begins (JU-11; TC-JU11-01/02).

## 7. Error Handling & Edge Cases

- Missing baselines → `MissingBaselineError`, no verdict (TC-MU2-01). Any uncomputable JU-9 field → abort, no partial record (TC-JU9-02).
- σ = 0 in a stated spread → `ModelContractError` upstream (models spec); the judge additionally guards CRPS/coverage against σ ≤ 0 with a hard error, never a clamp.
- A trial whose true trajectory left the declared task's domain (e.g. LV clamp event) was already aborted upstream (worlds spec §7); the judge asserts input completeness (`n_trials` consistent across all arrays).
- Bands file with boundaries inconsistent with the declared (N, p, α) → the harness's certification check fails the run before judging (TC-JU11-02's mechanism).
- A task whose divergence curve never crosses τ within the horizon → `switch_step = null`, not an error (ADR-J5); this is a reportable fact about the task, not a defect.

## 8. Testing Strategy

| Concern | Cases |
|---|---|
| Blindness by type + label-swap invariance | TC-JU1-01, TC-JU1-02 |
| Skill-only reporting | TC-JU2-01 |
| Error vs divergence on shared axes, shared distance metric | TC-JU3-01 |
| Calibration/summary distinct; anti-gaming property; fixed z_p constants | TC-JU4-01, TC-JU4-02 |
| Sharpness beats hedging; monotone; absolute hedge threshold (not decile) | TC-JU5-01, TC-JU5-02 |
| Switch step + re-measured invariant + no-switch and out-of-range-bin cases | TC-JU6-01, TC-JU6-02 |
| Trust horizon always task- and region-attached, dual units, h=0 convention | TC-JU7-01, TC-JU7-02 |
| Independent trials only; band correctness; hedging flag | TC-JU8-01, TC-JU8-02, TC-JU8-03 |
| Complete verdict or none; pure Verdict excludes identity fields | TC-JU9-01, TC-JU9-02 |
| Seven disclosures present verbatim; readable | TC-JU10-01, TC-JU10-02 (judged) |
| Prereg ordering | TC-JU11-01, TC-JU11-02 |
| Purity under blocked environment — effects guarded, reflection-route escape caught | TC-JU12-01, TC-JU12-02 (phantom-gate, design-review-005) |
| No learned component | TC-JU13-01 |

---

## Changelog

| Version | Date | Change |
|---|---|---|
| 1.0 | 2026-08-25 | Initial version. CRPS chosen (ADR-J1); OQ-1/OQ-2 settled: N=200 independent trials, 90% interval, exact-binomial bands green [12,29] / amber [8,11]∪[30,35] / red beyond, derived by exact binomial CDF (verified in-session) and committed. OQ-5 runtime half settled: 600 s budget. |
| 1.1 | 2026-08-25 | Design-review fixes (design-review-001): moved `is_fixture`/`model_ref`/`meta` out of the judge-produced Verdict into a harness-owned envelope (JU-1/JU-12 purity); added the `trials` per-trial data block RP-1's chart needs; added `region` to `exceptions.per_task`; fixed `error_vs_horizon.steps` to 0-index matching worlds' artefact; pinned z_p constants and corrected the φ formula/vectorization strategy; restated the shared distance metric explicitly (ADR-J5); pinned climatology bin-edge method (16 equal-population bins) and reference-run length (200,000 steps), and defined out-of-range-bin and no-switch-step behaviour; replaced the undefined "bottom-decile" sharpness flag with an absolute pre-registered threshold; authored the seven JU-10 disclosures verbatim (ADR-J7, new). |
| 1.2 | 2026-08-30 | Design-review fixes (design-review-002, Round 2): added `region` to `trust_horizons.per_task` and restated the trust-horizon formula per region (the v1.1 schema couldn't show `fx-brittle`'s in/out-region collapse — the phenomenon it exists to demonstrate); added `trials` and `climatology` to JU-9's mandatory-field list (grew to eight groups, both previously unlisted); added `calibration.n_trials`; clarified that `trials.is_exception` is copied from ADR-J4's joint test and `exceptions.per_task.observed` is a sum over it, so the two blocks cannot silently disagree; noted the harness envelope's serializer coverage explicitly. |
| 1.3 | 2026-08-30 | Design-review fixes (design-review-003, Round 3, dual/blind): restructured `calibration`/`sharpness` to `per_task` arrays; added `region` to `climatology.per_task`; unified `region_labels` to one canonical shape across all three domain specs; added TC-JU9-03 to build-enforce the `is_exception`/`observed` reconciliation. |
| 1.4 | 2026-08-30 | Design-review-004 structural repair: canonical keying rule + block×axis table (§5); `error_vs_horizon` → `per_region` array; `calibration`/`sharpness` region-suffix fields replaced with explicit keys; N=200 pinned as shared-per-region across tasks with the cross-task correlation disclosed; `natural_units` nullable (pendulum = null); roster count corrected 8→7 in ADR-J4/ADR-J6; sharpness wired to Chart 3. |
| 1.5 | 2026-08-31 | Design-review-005 repair: TC-JU12-01 specified as the load-bearing runtime purity control (guards effects — `os`/`open`/clock/network/subprocess/`numpy.random` shared objects raise on access — so reflection escapes trip at the effect; TC-JU12-02 phantom-gate proves it); calibration/sharpness pinned to the task's single `h_task` (reconciling the block×axis "—"); N=200 pinned shared across models (paired design); ADR-J6 envelope corrected for the 2× per-region multiplier; `error_vs_horizon` carries `dt` for Chart 2's world-time axis. |
| 1.6 | 2026-08-31 | Design-review-006 repair: `skill.per_task_region` pinned to one-step `h=1` (v1.5's "fixed by the task" gloss contradicted the MU-5 margin); Verdict example corrected (dp-control entry → lv-planning); `limitations` made the ninth mandatory JU-9 group; `interval_levels` removed from JudgeInput (judge constants, one producer); `meta.platform` pinned to `f"{sys.platform}-{platform.machine()}"`; TC-JU12-01 mechanism corrected to mutate-in-place (cross-cutting ADR-002 rule 3). |

---

*Developed using the Grounded Vibe Methodology*
