# World Model Judge — Judge Specification

Version 1.1 · 25 August 2026 · Domain 3 of Requirements v1.2 · References: cross-cutting spec v1.1, worlds spec v1.1, models spec v1.1

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

**Skill form (Murphy, JU-2):** `skill = 1 − CRPS_model / CRPS_baseline`, reported against persistence and linear separately, per task, per region, at one step. No absolute error is ever emitted without its skill counterpart (TC-JU2-01); the strictly-proper property is verified by TC-JU4-02's misstatement test.

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

An outcome "covers" if *all* dimensions fall inside (the joint event is what a user of the prediction experiences; per-dimension coverage is also recorded for diagnosis). Coverage is computed separately in-region and out-of-region (WD-5 labels), over the independent-trial set only (§ADR-J4 sampling). The diagnostic and the ADR-J1 summary are distinct fields in the verdict and never combined (JU-4's "never conflate", TC-JU4-01).

### ADR-J3 — Sharpness: mean 90% interval width, reported beside calibration

**Status:** Accepted. [Requirement: JU-5] [Test: TC-JU5-01, TC-JU5-02]

**Decision:** Sharpness = mean over trials and dimensions of the stated 90% central-interval width (2·z₀.₉₀·σ, using the z₀.₉₀ = 1.6449 constant from ADR-J2), in normalised units, per region. Smaller is sharper; it is monotone in stated σ by construction (TC-JU5-02's property). Reported in the same verdict block as calibration so neither is readable without the other (GBR: maximise sharpness subject to calibration).

### ADR-J4 — Exception counting and bands: binomial acceptance regions at N = 200 (settles OQ-1 and OQ-2)

**Status:** Accepted. [Requirement: JU-8, JU-11] [Test: TC-JU8-01, TC-JU8-02, TC-JU8-03]

**Context:** JU-8 demands exceptions counted over statistically independent trials, compared to the model's implied rate with a declared test at a declared sample size, against bands fixed in advance with declared false-alarm probabilities. Open Questions 1 and 2 defer the numbers to this spec.

**The declared design:**
- **Independent trials: N = 200** per (model, world, region, task). Each trial is a rollout from its own evaluation starting condition (disjoint from training and from each other — models spec MU-7 mechanics) with its own seeded action sequence. Trials are represented as the first axis of every `[n_trials, H, d]` array in `JudgeInput` (§4) — a trial's own row *is* its boundary; there is no flat/concatenated representation and therefore no separate boundary-marker field anywhere in the contract (design-review fix: an earlier draft's cross-cutting/models documents named a `trial_boundaries` array that this pre-shaped-array design makes unnecessary and undefined — removed there, see cross-cutting v1.1 and models v1.1 changelogs).
- **Exception definition:** at each **pre-declared horizon step h ∈ {1, h_task}** (h_task = the task's JU-6 switch step, per task), the trial's outcome at step h either falls inside the model's stated 90% central interval (all dimensions, ADR-J2 joint convention) or is an exception. One observation per trial per horizon — never pooled within a rollout (TC-JU8-01 rejects correlated pooling by construction: each trial occupies its own row in the array's trial axis, and the counter consumes one designated step per trial, never steps from two different trials' rows treated as if independent).
- **The test:** exact two-sided binomial test against p = 0.10 at n = 200. Expected exceptions: 20.
- **Bands, derived (not chosen by eye — JU-11):**
  - **Green** — count inside the central 95% binomial acceptance region for Bin(200, 0.10): **[12, 29]**.
  - **Amber** — outside green but inside the central 99.9% region: **[8, 11] or [30, 35]**.
  - **Red** — **≤ 7 or ≥ 36**.
  - False-alarm probability for a perfectly calibrated model, from the exact binomial CDF: **3.3%** of landing outside green (declared bound ≤ 5%), **0.087%** outside amber (declared bound ≤ 0.1%) — declared per band, per JU-8. The integer boundaries above are computed from the exact binomial CDF by a build-time derivation script whose output is committed to `prereg/thresholds.json`; the committed file is what the judge reads (via its arguments), and TC-JU8-02 checks a known-calibrated fixture lands green at the declared rate.
  - **Low-side counts are flagged, not rewarded, against an absolute pre-registered threshold, not a decile (design-review fix):** a green-or-better count is cross-checked against sharpness (ADR-J3); if that region/task's mean 90%-interval width exceeds a fixed value declared in `prereg/thresholds.json` (`sharpness_hedge_threshold`, set per world from the world's own scale vector at pre-registration time, before any model is judged — worlds spec ADR-W3 defines the scale), the JU-5 cross-flag fires (TC-JU8-03). The original "bottom-decile" wording is removed: with ~8 contestants per (model, world, region) a decile has no statistical meaning; an absolute, pre-registered width bound does.

**Why N=200:** at p=0.10 the green region [12,29] distinguishes a true 90% model from a fixture at 70% true coverage (expected 60 exceptions — deep red) and from a padded 99% model (expected 2 — red on the low side) with power ≈ 1; it is the smallest round sample where the amber band is non-degenerate at both ends, and it fits the NF-2 budget with two orders of magnitude to spare (§ADR-J6). Assumption 2 of the requirements (independent trials shrink the usable sample) is answered by construction: the 200 are independent starts, not steps.

### ADR-J5 — The JU-6 switch and conditioned climatology

**Status:** Accepted. [Requirement: JU-6, JU-3, JU-7] [Test: TC-JU6-01, TC-JU6-02, TC-JU3-01, TC-JU7-01, TC-JU7-02]

**The shared distance metric (design-review fix — this was asserted in worlds.md and architecture-overview.md but never actually restated or cited here, despite every formula in this document otherwise being given in full):** every "normalised error" and "normalised distance" in this ADR is worlds spec ADR-W3's metric, restated here so the two specs cannot silently drift apart: RMS over state dimensions each divided by the world's declared scale vector (worlds.md §4) — `distance(a, b) = sqrt(mean_d(((a_d − b_d) / scale_d)²))`. This is the *same* function used for WD-4's divergence curves, JU-3's error-vs-horizon, and every task's tolerance τ. It is a different metric from CRPS (ADR-J1): CRPS scores the full predictive distribution for the anti-gaming skill summary; this RMS distance scores point predictions against tolerances and divergence. Both are deliberate and distinct — see architecture-overview v1.1's corrected conceptual-integrity note.

**Decision:**
- **Switch step per task:** the first step where the world's own divergence curve (WD-4, the trial's region's curve) exceeds the task tolerance τ. Before it: trajectory grading (normalised error vs the divergence reference, JU-3, using the distance metric above). At and after it: statistical grading. The verdict records the switch step per task (TC-JU6-01). **If the divergence curve never exceeds τ within the world's declared horizon (design-review fix — previously undefined):** `switch_step = null` in the verdict, the task is graded by trajectory error for its entire horizon, and the verdict states explicitly ("this task never left trajectory-grading within the horizon") rather than leaving `trust_horizon`'s formula (below) without an upper bound.
- **Conditioned climatology, concretely (design-review fix — bin-edge method and reference-run length were previously unstated numbers in a document that otherwise pins every constant):** the reference statistics (per-dimension mean and standard deviation of state) of the true world *restricted to the invariant's value*, computed by binning **one continuous 200,000-step reference trajectory per world** (harness-generated, null action, from the world's training-region centre — long enough that every bin below has ≥50 samples for both worlds, checked by an assertion at generation time) by conserved-quantity value into **16 equal-population bins** (not equal-width — equal-population guarantees every bin has comparable sample size regardless of how the invariant's long-run distribution is shaped, which an equal-width scheme cannot guarantee for the pendulum's energy, whose long-run density is not uniform). At each compared step, the invariant is *measured from the true trajectory at that step* (TC-JU6-02's re-measured-not-frozen check; both action impulses and bounded integrator drift move the bin, per WD-3/JU-6). **Out-of-range invariant values (design-review fix — a real case: WD-5's out-of-region starts and JU-6's own action-driven excursions can both reach invariant values the reference run never visited):** the outermost bins are unbounded on their open side (the lowest bin's floor and the highest bin's ceiling are ±∞), so every possible invariant value falls in exactly one of the 16 bins by construction — no value is ever "outside the bins."
- **Statistical agreement score:** past the switch, the model's predicted mean at each compared step is scored against the conditioned climatology as a standardised deviation `(pred_mean − clim_mean)/clim_sd`, averaged per dimension over the post-switch window; agreement holds when the average |z| ≤ 1. The threshold is part of `prereg/thresholds.json`.
- **Trust horizon (JU-7):** `trust_horizon(model, world, task) = the largest step h such that the model's median normalised error across trials at every step ≤ h stays within τ, capped at switch_step if switch_step is not null, else capped at the world's declared horizon H`. If the model fails tolerance even at step 1, `trust_horizon = 0` (an explicit, not implicit, convention — design-review fix). Always reported as (steps, steps × dt in world time units, fraction of the world's natural cycle for LV / seconds for the pendulum), and always attached to its task name and τ (TC-JU7-01/02). It is capped at the switch step by construction — beyond it, exact-path trust is not a claim anyone can earn (Lorenz).

### ADR-J6 — Runtime budget (settles OQ-5's runtime half)

**Status:** Accepted. [Requirement: NF-2] [Test: TC-NF2-01]

**Decision:** The full result set — benchmarks, training both unrigged models, all rollouts, judging, charts — completes in **under 600 seconds wall-clock on a 4-core consumer laptop CPU** (TC-NF2-01's `<spec-value>` = 600 s). Envelope: 200 trials × (700 + 5000) steps × 8 models ≈ 9M model-step calls, each a 2×64×64 MLP forward ≈ tens of kFLOPs — well under a minute of compute; training two small MLPs full-batch for a pinned epoch count dominates and is budgeted at ≤ 6 minutes combined. The budget includes headroom ≥ 30% on the reference machine class.

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

Every public function: pure, typed, arrays in → values out. The package imports `numpy`, `math`, `dataclasses`, `typing` — nothing else (TC-NF6-01, TC-JU12-01, TC-JU13-01). Nothing in this package reads a file, a clock, or an environment variable — `platform`, `prereg_commit`, and any other run-identifying metadata are the harness's concern, not the judge's (§5).

**JudgeInput (complete):** predictions (mean, spread) `[n_trials, H, d]`, outcomes `[n_trials, H, d]`, baseline predictions (same shapes, one set per baseline), region labels per trial (state axis, action axis, **and the region's name string** — design-review fix: a boolean in/out flag alone gives reporting and the climatology lookup no way to pick which named curve/table a trial belongs to once a world declares more than one out-region, which WD-5 explicitly permits), divergence curves per region **`[H+1]`** (design-review fix: matches the worlds-spec artefact's own `H+1`-length, 0-indexed convention exactly — see worlds v1.1 §5 — rather than the previously mismatched `[H]`), conditioned-climatology table (16 equal-population bins → mean[d], sd[d], each bin's invariant range) + per-step true invariant bins `[n_trials, H]`, task list (name, kind, τ, horizon), thresholds (bands, sharpness-hedge threshold, agreement threshold, interval levels), world time step dt and natural-cycle length. Nothing else — by type, nothing else *can* be passed (TC-JU1-01), and label-swap invariance is property-tested (TC-JU1-02).

## 5. API Boundary Contracts — the Verdict record (JU-9) and the harness envelope

**The judge produces exactly one thing: a pure `Verdict`, containing only what its own arithmetic computed.** Design-review Panels B and C both independently found the previous draft's Verdict schema carrying `is_fixture`, `model_ref`, and `meta.platform`/`meta.prereg_commit` — fields the judge cannot honestly produce (JU-1 forbids it from ever knowing `is_fixture` or a model's identity; JU-12 forbids it from reading the platform or git state). Those fields move to a **harness-owned envelope** that wraps the pure Verdict after the fact. There is now exactly one producer for each fact: the judge for everything inside `Verdict`, the harness for everything identifying *whose* verdict it is.

**The pure `Verdict` (serialized canonically, cross-cutting ADR-002):**

```json
{
  "schema": "wmj-verdict/1",
  "world": "lv",
  "skill": {"per_task_region": [{"task": "lv-control", "region": "training",
    "vs_persistence": 0.41, "vs_linear": 0.28, "crps": 0.031}]},
  "error_vs_horizon": {"steps": [0, 1, 2, "..."], "median_error": ["..."],
    "divergence_reference": ["..."], "region": "training"},
  "calibration": {"levels": [0.5, 0.8, 0.9, 0.95],
    "coverage_in_region": [0.52, 0.79, 0.9, 0.94],
    "coverage_out_region": [0.44, 0.71, 0.82, 0.88],
    "per_dimension": "..."},
  "sharpness": {"mean_width_90_in": 0.18, "mean_width_90_out": 0.21},
  "exceptions": {"per_task": [{"task": "lv-control", "region": "training", "horizon_step": 1,
    "n_trials": 200, "expected": 20, "observed": 27, "band": "green",
    "bands": {"green": [12, 29], "amber": [[8, 11], [30, 35]], "red": "outside"},
    "low_side_sharpness_flag": false}]},
  "trials": {"per_task": [{"task": "lv-control", "region": "training", "horizon_step": 1,
    "distance_unit": "rms-normalised (ADR-W3/ADR-J5, scalar even for multi-dimensional state)",
    "outcome_distance": [0.08, "...(200 values, one per trial — distance from this trial's mean prediction to its true outcome)"],
    "band_lo": [0.0, "...(always 0 — a distance is never negative)"],
    "band_hi": [0.16, "...(the trial's stated 90% interval width in the same distance unit, from spread via ADR-J2/J3's z-multiplier)"],
    "is_exception": [false, "..."]}]},
  "climatology": {"per_task": [{"task": "lv-planning", "switch_step": 214,
    "agreement_mean_abs_z": 0.6, "agrees": true}]},
  "trust_horizons": {"per_task": [{"task": "lv-control", "tolerance": 0.1,
    "steps": 118, "world_time": 2.36, "natural_units": "0.34 cycles"}]},
  "not_tested": ["genuine off-model surprises (WD-8)", "closed-loop action selection",
    "ongoing monitoring", "the seven JU-10 disclosures cover the rest of this list"],
  "limitations": ["<the seven ADR-J7 strings, verbatim, in order>"]
}
```

**Changes from the pre-review draft:** `model_ref`, `is_fixture`, and `meta` are removed (moved to the envelope, below). `error_vs_horizon.steps` now starts at 0, matching the worlds-spec divergence artefact's own indexing exactly (design-review Important #1) — there is one shared step-zero origin for every step-indexed curve in the system. `exceptions.per_task` entries now carry a `region` field, matching the convention `calibration` and `error_vs_horizon` already used (design-review Important #2). A new `trials` block carries the per-trial outcome/band/exception arrays reporting's Chart 1 (the primary chart) actually needs to draw the 200 individual points and their bands — the pre-review schema only had aggregate counts, which cannot render that chart (design-review Critical finding on RP-1). `not_tested`'s example list is fully stated rather than trailing off with "..." (design-review Minor).

**The harness-owned envelope (`wmj/harness/results.py`, NOT part of the judge package):**

```json
{
  "model_ref": "opaque-index-3",
  "model_name": "ensemble",
  "is_fixture": false,
  "verdict": { "...the pure Verdict above..." },
  "meta": {"seed": 20260825, "platform": "linux-x86_64", "prereg_commit": "<sha>",
    "thresholds_file": "prereg/thresholds.json"}
}
```

The harness constructs one envelope per (model, world) immediately after calling the judge; this envelope — never the bare `Verdict` — is what's written to `out/verdicts/*.json` and handed to reporting. `model_ref` is an opaque index; the judge never sees a name or a fixture flag at any point in its call graph. All six JU-9 field groups (skill, error-vs-horizon, calibration+sharpness, exceptions, trust horizons, not-tested) are mandatory inside the pure `Verdict`: `verdict.py` raises rather than emitting a record with any of them missing or null (TC-JU9-01, TC-JU9-02).

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
| Trust horizon always task-attached, dual units, h=0 convention | TC-JU7-01, TC-JU7-02 |
| Independent trials only; band correctness; hedging flag | TC-JU8-01, TC-JU8-02, TC-JU8-03 |
| Complete verdict or none; pure Verdict excludes identity fields | TC-JU9-01, TC-JU9-02 |
| Seven disclosures present verbatim; readable | TC-JU10-01, TC-JU10-02 (judged) |
| Prereg ordering | TC-JU11-01, TC-JU11-02 |
| Purity under blocked environment (no meta fields computed in-package) | TC-JU12-01 |
| No learned component | TC-JU13-01 |

---

## Changelog

| Version | Date | Change |
|---|---|---|
| 1.0 | 2026-08-25 | Initial version. CRPS chosen (ADR-J1); OQ-1/OQ-2 settled: N=200 independent trials, 90% interval, exact-binomial bands green [12,29] / amber [8,11]∪[30,35] / red beyond, derived by exact binomial CDF (verified in-session) and committed. OQ-5 runtime half settled: 600 s budget. |
| 1.1 | 2026-08-25 | Design-review fixes (design-review-001): moved `is_fixture`/`model_ref`/`meta` out of the judge-produced Verdict into a harness-owned envelope (JU-1/JU-12 purity); added the `trials` per-trial data block RP-1's chart needs; added `region` to `exceptions.per_task`; fixed `error_vs_horizon.steps` to 0-index matching worlds' artefact; pinned z_p constants and corrected the φ formula/vectorization strategy; restated the shared distance metric explicitly (ADR-J5); pinned climatology bin-edge method (16 equal-population bins) and reference-run length (200,000 steps), and defined out-of-range-bin and no-switch-step behaviour; replaced the undefined "bottom-decile" sharpness flag with an absolute pre-registered threshold; authored the seven JU-10 disclosures verbatim (ADR-J7, new). |

---

*Developed using the Grounded Vibe Methodology*
