# World Model Judge — Judge Specification

Version 1.0 · 25 August 2026 · Domain 3 of Requirements v1.2 · References: cross-cutting spec v1.0, worlds spec v1.0, models spec v1.0

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
| Michael Keeling | *Design It!* | ADR format |

---

## 1. Purpose

Covers requirements **JU-1 through JU-13**. The judge package (`wmj.judge`) is a set of pure functions (JU-12) over plain arrays (JU-1) producing one structured verdict (JU-9) with a limitations statement (JU-10), computed against thresholds fixed in advance (JU-11), containing no learned component (JU-13, Won't).

## 2. Architecturally Significant Requirements

- **JU-12 + NF-6**: the judge is a leaf package — pure functions, no I/O, imports stdlib + NumPy only. Everything it needs arrives as arguments; everything it produces returns as values.
- **JU-1**: the input type is defined *inside* the judge package and has no field that could carry identity — blindness by construction, not by discipline.
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

**Decision:** CRPS (option 1), computed per dimension on normalised state (worlds-spec scale vectors), averaged over dimensions; Φ and φ implemented in-house from `math.erf` (stdlib) — no SciPy (NF-3).

**Skill form (Murphy, JU-2):** `skill = 1 − CRPS_model / CRPS_baseline`, reported against persistence and linear separately, per task, per region, at one step. No absolute error is ever emitted without its skill counterpart (TC-JU2-01); the strictly-proper property is verified by TC-JU4-02's misstatement test.

### ADR-J2 — Calibration diagnostic: central-interval coverage at four declared levels

**Status:** Accepted. [Requirement: JU-4(a)] [Test: TC-JU4-01]

**Decision:** Predictions are Gaussian per dimension (MU-1 mean/spread). The diagnostic computes observed coverage of central intervals at levels **{50%, 80%, 90%, 95%}**: for level p, the interval is μ ± z_(p)·σ per dimension; an outcome "covers" if *all* dimensions fall inside (the joint event is what a user of the prediction experiences; per-dimension coverage is also recorded for diagnosis). Coverage is computed separately in-region and out-of-region (WD-5 labels), over the independent-trial set only (§ADR-J4 sampling). The diagnostic and the ADR-J1 summary are distinct fields in the verdict and never combined (JU-4's "never conflate", TC-JU4-01).

### ADR-J3 — Sharpness: mean 90% interval width, reported beside calibration

**Status:** Accepted. [Requirement: JU-5] [Test: TC-JU5-01, TC-JU5-02]

**Decision:** Sharpness = mean over trials and dimensions of the stated 90% central-interval width (2·z₀.₉₀·σ), in normalised units, per region. Smaller is sharper; it is monotone in stated σ by construction (TC-JU5-02's property). Reported in the same verdict block as calibration so neither is readable without the other (GBR: maximise sharpness subject to calibration).

### ADR-J4 — Exception counting and bands: binomial acceptance regions at N = 200 (settles OQ-1 and OQ-2)

**Status:** Accepted. [Requirement: JU-8, JU-11] [Test: TC-JU8-01, TC-JU8-02, TC-JU8-03]

**Context:** JU-8 demands exceptions counted over statistically independent trials, compared to the model's implied rate with a declared test at a declared sample size, against bands fixed in advance with declared false-alarm probabilities. Open Questions 1 and 2 defer the numbers to this spec.

**The declared design:**
- **Independent trials: N = 200** per (model, world, region). Each trial is a rollout from its own evaluation starting condition (disjoint from training and from each other — models spec MU-7 mechanics) with its own seeded action sequence.
- **Exception definition:** at each **pre-declared horizon step h ∈ {1, h_task}** (h_task = the task's JU-6 switch step, per task), the trial's outcome at step h either falls inside the model's stated 90% central interval (all dimensions, ADR-J2 joint convention) or is an exception. One observation per trial per horizon — never pooled within a rollout (TC-JU8-01 rejects correlated pooling by construction: the input carries trial boundaries, and the counter consumes one designated step per trial).
- **The test:** exact two-sided binomial test against p = 0.10 at n = 200. Expected exceptions: 20.
- **Bands, derived (not chosen by eye — JU-11):**
  - **Green** — count inside the central 95% binomial acceptance region for Bin(200, 0.10): **[12, 29]**.
  - **Amber** — outside green but inside the central 99.9% region: **[8, 11] or [30, 35]**.
  - **Red** — **≤ 7 or ≥ 36**.
  - False-alarm probability for a perfectly calibrated model, from the exact binomial CDF: **3.3%** of landing outside green (declared bound ≤ 5%), **0.087%** outside amber (declared bound ≤ 0.1%) — declared per band, per JU-8. The integer boundaries above are computed from the exact binomial CDF by a build-time derivation script whose output is committed to `prereg/thresholds.json`; the committed file is what the judge reads (via its arguments), and TC-JU8-02 checks a known-calibrated fixture lands green at the declared rate.
  - **Low-side counts are flagged, not rewarded:** a green-or-better count with bottom-decile sharpness triggers the JU-5 cross-flag (TC-JU8-03) — too few misses plus wide intervals is hedging, and the verdict says so.

**Why N=200:** at p=0.10 the green region [12,28] distinguishes a true 90% model from a fixture at 70% true coverage (expected 60 exceptions — deep red) and from a padded 99% model (expected 2 — red on the low side) with power ≈ 1; it is the smallest round sample where the amber band is non-degenerate at both ends, and it fits the NF-2 budget with two orders of magnitude to spare (§ADR-J6). Assumption 2 of the requirements (independent trials shrink the usable sample) is answered by construction: the 200 are independent starts, not steps.

### ADR-J5 — The JU-6 switch and conditioned climatology

**Status:** Accepted. [Requirement: JU-6, JU-3, JU-7] [Test: TC-JU6-01, TC-JU6-02, TC-JU3-01, TC-JU7-01, TC-JU7-02]

**Decision:**
- **Switch step per task:** the first step where the world's own divergence curve (WD-4, the trial's region's curve) exceeds the task tolerance τ. Before it: trajectory grading (normalised error vs the divergence reference, JU-3). At and after it: statistical grading. The verdict records the switch step per task (TC-JU6-01).
- **Conditioned climatology:** the reference statistics (per-dimension mean and standard deviation of state) of the true world *restricted to the invariant's value* — computed by binning long true reference runs (harness-generated artefact, part of the benchmark data) by conserved-quantity value into **16 bins** per world, and at each compared step using the bin of the invariant *measured from the true trajectory at that step* (TC-JU6-02's re-measured-not-frozen check; both action impulses and bounded integrator drift move the bin, per WD-3/JU-6).
- **Statistical agreement score:** past the switch, the model's predicted mean at each compared step is scored against the conditioned climatology as a standardised deviation `(pred_mean − clim_mean)/clim_sd`, averaged per dimension over the post-switch window; agreement holds when the average |z| ≤ 1. The threshold is part of `prereg/thresholds.json`.
- **Trust horizon (JU-7):** `trust_horizon(model, world, task) = the largest step h ≤ switch_step such that the model's median normalised error across trials at every step ≤ h stays within τ`. Always reported as (steps, steps × dt in world time units, fraction of the world's natural cycle for LV / seconds for the pendulum), and always attached to its task name and τ (TC-JU7-01/02). It is capped at the switch step by construction — beyond it, exact-path trust is not a claim anyone can earn (Lorenz).

### ADR-J6 — Runtime budget (settles OQ-5's runtime half)

**Status:** Accepted. [Requirement: NF-2] [Test: TC-NF2-01]

**Decision:** The full result set — benchmarks, training both unrigged models, all rollouts, judging, charts — completes in **under 600 seconds wall-clock on a 4-core consumer laptop CPU** (TC-NF2-01's `<spec-value>` = 600 s). Envelope: 200 trials × (700 + 5000) steps × 8 models ≈ 9M model-step calls, each a 2×64×64 MLP forward ≈ tens of kFLOPs — well under a minute of compute; training two small MLPs full-batch for a pinned epoch count dominates and is budgeted at ≤ 6 minutes combined. The budget includes headroom ≥ 30% on the reference machine class.

## 4. Component Design

```
wmj/judge/
  types.py        # JudgeInput, Verdict, Task/ToleranceView — frozen dataclasses, arrays+floats only,
                  # no name/id/provenance fields anywhere (JU-1; TC-JU1-01)
  skill.py        # CRPS closed form, skill scores vs both baselines (ADR-J1)
  calibration.py  # coverage at the four levels, in/out region (ADR-J2)
  sharpness.py    # interval widths (ADR-J3)
  exceptions.py   # trial-wise exception counts, binomial band assignment (ADR-J4)
  climatology.py  # switch step, conditioned climatology agreement (ADR-J5)
  horizon.py      # trust horizons (ADR-J5)
  verdict.py      # assembles the JU-9 record; refuses on any missing input (TC-JU9-02)
  limitations.py  # the seven JU-10 disclosures as fixed text constants
```

Every public function: pure, typed, arrays in → values out. The package imports `numpy`, `math`, `dataclasses`, `typing` — nothing else (TC-NF6-01, TC-JU12-01, TC-JU13-01).

**JudgeInput (complete):** predictions (mean, spread) `[n_trials, H, d]`, outcomes `[n_trials, H, d]`, baseline predictions (same shapes, one set per baseline), region labels per trial (+ axis), divergence curves per region `[H]`, conditioned-climatology table `(bin → mean[d], sd[d])` + per-step true invariant bins `[n_trials, H]`, task list (name, kind, τ, horizon), thresholds (bands, agreement threshold, interval levels), world time step dt and natural-cycle length. Nothing else — by type, nothing else *can* be passed (TC-JU1-01), and label-swap invariance is property-tested (TC-JU1-02).

## 5. API Boundary Contracts — the Verdict record (JU-9)

The complete schema reporting consumes; serialized canonically (cross-cutting ADR-002). One verdict per (model, world):

```json
{
  "schema": "wmj-verdict/1",
  "world": "lv",
  "model_ref": "opaque-index-3",
  "is_fixture": false,
  "skill": {"per_task_region": [{"task": "lv-control", "region": "training",
    "vs_persistence": 0.41, "vs_linear": 0.28, "crps": 0.031}]},
  "error_vs_horizon": {"steps": [1, 2, "..."], "median_error": ["..."],
    "divergence_reference": ["..."], "region": "training"},
  "calibration": {"levels": [0.5, 0.8, 0.9, 0.95],
    "coverage_in_region": [0.52, 0.79, 0.9, 0.94],
    "coverage_out_region": [0.44, 0.71, 0.82, 0.88],
    "per_dimension": "..."},
  "sharpness": {"mean_width_90_in": 0.18, "mean_width_90_out": 0.21},
  "exceptions": {"per_task": [{"task": "lv-control", "horizon_step": 1, "n_trials": 200,
    "expected": 20, "observed": 27, "band": "green",
    "bands": {"green": [12, 29], "amber": [[8, 11], [30, 35]], "red": "outside"},
    "low_side_sharpness_flag": false}]},
  "climatology": {"per_task": [{"task": "lv-planning", "switch_step": 214,
    "agreement_mean_abs_z": 0.6, "agrees": true}]},
  "trust_horizons": {"per_task": [{"task": "lv-control", "tolerance": 0.1,
    "steps": 118, "world_time": 2.36, "natural_units": "0.34 cycles"}]},
  "not_tested": ["genuine off-model surprises (WD-8)", "closed-loop action selection",
    "ongoing monitoring", "..."],
  "limitations": ["<the seven JU-10 disclosures, fixed text>"],
  "meta": {"seed": 20260825, "platform": "linux-x86_64", "prereg_commit": "<sha>",
    "thresholds_file": "prereg/thresholds.json"}
}
```

`model_ref` is an opaque index assigned by the harness — the judge never sees a name; the harness's index→name map lets reporting label charts (models spec §5). All six JU-9 field groups are mandatory: `verdict.py` raises rather than emitting a record with any of them missing or null (TC-JU9-01, TC-JU9-02).

## 6. Integration Points

- **← harness:** builds `JudgeInput` from world artefacts + model rollouts; owns the prereg check (models spec ADR-M5) and passes `thresholds` as data — the judge never reads files (JU-12).
- **→ reporting:** consumes only `Verdict` records plus the harness's index→name/fixture map.
- **Bands derivation script** (`wmj/harness/derive_thresholds.py`): computes the binomial boundaries from (N, p, α-levels) and writes `prereg/thresholds.json`; run once, committed, never re-run after judging begins (JU-11; TC-JU11-01/02).

## 7. Error Handling & Edge Cases

- Missing baselines → `MissingBaselineError`, no verdict (TC-MU2-01). Any uncomputable JU-9 field → abort, no partial record (TC-JU9-02).
- σ = 0 in a stated spread → `ModelContractError` upstream (models spec); the judge additionally guards CRPS/coverage against σ ≤ 0 with a hard error, never a clamp.
- A trial whose true trajectory left the declared task's domain (e.g. LV clamp event) was already aborted upstream (worlds spec §7); the judge asserts input completeness (`n_trials` consistent across all arrays).
- Bands file with boundaries inconsistent with the declared (N, p, α) → the harness's certification check fails the run before judging (TC-JU11-02's mechanism).

## 8. Testing Strategy

| Concern | Cases |
|---|---|
| Blindness by type + label-swap invariance | TC-JU1-01, TC-JU1-02 |
| Skill-only reporting | TC-JU2-01 |
| Error vs divergence on shared axes | TC-JU3-01 |
| Calibration/summary distinct; anti-gaming property | TC-JU4-01, TC-JU4-02 |
| Sharpness beats hedging; monotone | TC-JU5-01, TC-JU5-02 |
| Switch step + re-measured invariant | TC-JU6-01, TC-JU6-02 |
| Trust horizon always task-attached, dual units | TC-JU7-01, TC-JU7-02 |
| Independent trials only; band correctness; hedging flag | TC-JU8-01, TC-JU8-02, TC-JU8-03 |
| Complete verdict or none | TC-JU9-01, TC-JU9-02 |
| Seven disclosures present; readable | TC-JU10-01, TC-JU10-02 (judged) |
| Prereg ordering | TC-JU11-01, TC-JU11-02 |
| Purity under blocked environment | TC-JU12-01 |
| No learned component | TC-JU13-01 |

---

## Changelog

| Version | Date | Change |
|---|---|---|
| 1.0 | 2026-08-25 | Initial version. CRPS chosen (ADR-J1); OQ-1/OQ-2 settled: N=200 independent trials, 90% interval, exact-binomial bands green [12,29] / amber [8,11]∪[30,35] / red beyond, derived by exact binomial CDF (verified in-session) and committed. OQ-5 runtime half settled: 600 s budget. |

---

*Developed using the Grounded Vibe Methodology*
