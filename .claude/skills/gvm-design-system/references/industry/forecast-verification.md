---
domain_name: forecast_verification
activation_signals:
  - calibration
  - sharpness
  - proper_scoring_rule
  - skill_score
  - baseline_forecast
  - persistence_baseline
  - climatology
  - trust_horizon
  - divergence_rate
strong_signals:
  - crps
  - brier_score
  - prequential
  - weather_vs_climate
---

# Industry Domain Specialists — Forecast Verification (Project: World Model Judge)

> **Project-scoped file.** Grounds this project's measurement layer — the techniques borrowed from meteorology's forecast-verification discipline, distinct from banking's institutional/enforcement contribution (see `model-risk.md`). Append-only within this project.

## Activation Signals

Activate this file for any application dealing with: scoring a probabilistic prediction against an outcome, calibration and sharpness of stated confidence, skill scores relative to a reference baseline, strictly proper scoring rules, or grading prediction quality as a function of how far ahead the prediction reaches.

---

## Tilmann Gneiting & Adrian Raftery

**Source:** Tilmann Gneiting & Adrian E. Raftery, "Strictly Proper Scoring Rules, Prediction, and Estimation," *Journal of the American Statistical Association*, 102(477), 359–378 (2007)

**Expert score:**
| Authority | Publication | Breadth | Adoption | Currency | Classification |
|-----------|-------------|---------|----------|----------|----------------|
| 5 | 5 | 4 | 5 | 4 | **Established** |

Evidence: Authority — Gneiting is a leading figure in statistical forecast verification (University of Heidelberg); the paper is one of the most cited works in probabilistic forecasting. Publication — JASA, the flagship journal of the American Statistical Association. Adoption — the standard reference for why scoring-rule choice matters, cited across meteorology, econometrics, and machine-learning uncertainty-quantification literature.

**Work score — "Strictly Proper Scoring Rules, Prediction, and Estimation":**
| Specificity | Depth | Currency | Influence | Classification |
|-------------|-------|----------|-----------|----------------|
| 5 | 5 | 4 | 5 | **Established** |

Evidence: Specificity — precisely defines the property (a forecaster cannot improve their expected score by misreporting their true belief) that this project's JU-4(b) requires directly. Influence — the anti-gaming property this project's entire calibration-honesty argument rests on; without it, an overconfident model could score well by being confident, the exact failure the project exists to catch.

**Activation signals:** Strictly proper scoring rules, anti-gaming scoring, CRPS, honest probabilistic reporting

**Key principles:**

- **A strictly proper scoring rule cannot be gamed** — a forecaster's expected score is maximised by reporting their true belief, never by hedging or overstating confidence. This is JU-4(b)'s load-bearing property, and TC-JU4-02's property-based test verifies it directly.
- **The choice of scoring rule matters, not just its existence** — a cheatable rule (e.g. naive squared error on point forecasts alone) lets an overconfident model score well by being confident. Picking the wrong rule reintroduces the exact failure the project is designed to catch.

---

## Gneiting, Balabdaoui & Raftery

**Source:** Tilmann Gneiting, Fadoua Balabdaoui & Adrian E. Raftery, "Probabilistic Forecasts, Calibration and Sharpness," *Journal of the Royal Statistical Society: Series B*, 69(2), 243–268 (2007)

**Expert score:**
| Authority | Publication | Breadth | Adoption | Currency | Classification |
|-----------|-------------|---------|----------|----------|----------------|
| 5 | 5 | 4 | 4 | 4 | **Established** |

Evidence: Authority — same research lineage as the JASA 2007 paper, extending it into the calibration/sharpness framework specifically. Publication — JRSS-B, a top statistics journal. Adoption — the standard citation for "calibration is necessary but not sufficient" in probabilistic forecasting.

**Work score — "Probabilistic Forecasts, Calibration and Sharpness":**
| Specificity | Depth | Currency | Influence | Classification |
|-------------|-------|----------|-----------|----------------|
| 5 | 5 | 4 | 5 | **Established** |

Evidence: Specificity — formally separates calibration (statistical consistency between stated and observed frequencies) from sharpness (concentration of the predictive distribution), and shows both are required. Influence — this project's JU-4/JU-5 pairing is a direct application: JU-4 checks calibration, JU-5 checks sharpness, and JU-5's rationale ("without this the judge would reward cowardice, and the vaguest model would win") restates this paper's central finding in plain language.

**Activation signals:** Calibration and sharpness distinction, "maximise sharpness subject to calibration," honest-but-useless forecasts

**Key principles:**

- **Calibration is necessary but not sufficient** — a forecast that is always "somewhere between zero and a million" is perfectly calibrated (it never gets caught out) and completely useless. JU-5 exists precisely to catch this.
- **Maximise sharpness subject to calibration** — the correct ordering of the two properties: never sacrifice honesty for confidence, but among honest forecasts, reward the more precise one. TC-JU5-02's monotonicity property test verifies sharpness strictly decreases as a range is artificially widened.

---

## Allan Murphy

**Source:** Allan H. Murphy, "What Is a Good Forecast? An Essay on the Nature of Goodness in Weather Forecasting," *Weather and Forecasting*, 8(2), 281–293 (1993)

**Expert score:**
| Authority | Publication | Breadth | Adoption | Currency | Classification |
|-----------|-------------|---------|----------|----------|----------------|
| 5 | 4 | 4 | 5 | 3 | **Established** |

Evidence: Authority — Murphy was a foundational figure in operational meteorological forecast verification at NOAA/NWS; this essay is widely regarded as the field's clearest statement of why a single accuracy number is insufficient. Publication — *Weather and Forecasting*, the standard journal of the American Meteorological Society's forecasting community. Adoption — routinely cited as the entry point for skill-score thinking in verification science.

**Work score — "What Is a Good Forecast?":**
| Specificity | Depth | Currency | Influence | Classification |
|-------------|-------|----------|-----------|----------------|
| 4 | 4 | 3 | 5 | **Established** |

Evidence: Specificity — introduces the skill-score framework (score relative to a reference forecast, not in absolute terms) precisely as this project's JU-2 requires. Influence — JU-2's rationale ("0.03 means nothing until you know that doing nothing scores 0.04") restates Murphy's own argument directly.

**Activation signals:** Skill scores, forecast goodness dimensions, reference-forecast comparison

**Key principles:**

- **Never report raw accuracy alone — always relative to a reference.** A forecast is only good relative to a baseline (persistence, climatology, or a naive extrapolation); an absolute error number is uninterpretable without one. This is MU-2/JU-2's entire justification.
- **Forecast quality is multi-dimensional** — no single number captures "goodness"; different dimensions (accuracy, skill, reliability, resolution) can and do disagree, which is why this project's RP-4 comparison table (ordinary error vs. trust horizon) is the actual headline result, not a redundancy.

---

## Edward Lorenz

**Source:** Edward N. Lorenz, "Deterministic Nonperiodic Flow," *Journal of the Atmospheric Sciences*, 20(2), 130–141 (1963)

**Expert score:**
| Authority | Publication | Breadth | Adoption | Currency | Classification |
|-----------|-------------|---------|----------|----------|----------------|
| 5 | 5 | 5 | 5 | 5 | **Canonical** |

Evidence: Authority — the founding paper of deterministic chaos theory, discovered while Lorenz was building a simplified numerical weather model and noticed rounding differences produced wildly divergent forecasts. Publication — *Journal of the Atmospheric Sciences*, one of the most consequential papers in 20th-century applied mathematics. Breadth — the paper's central finding (sensitive dependence on initial conditions, popularly "the butterfly effect") underlies chaos theory, dynamical systems, and predictability science broadly, well beyond meteorology. Adoption — foundational and unchallenged; cited across mathematics, physics, and every field that studies predictability limits.

**Work score — "Deterministic Nonperiodic Flow":**
| Specificity | Depth | Currency | Influence | Classification |
|-------------|-------|----------|-----------|----------------|
| 5 | 5 | 5 | 5 | **Canonical** |

Evidence: Specificity — the Lorenz system is the standard minimal example of sensitive dependence on initial conditions, directly grounding this project's WD-4 divergence-benchmark concept. Influence — WD-4's core idea (grade a model's error against how fast the true system diverges from itself, not against zero) is Lorenz's argument applied to model evaluation rather than weather forecasting.

**Activation signals:** Sensitive dependence on initial conditions, chaos, divergence benchmark, "some error is nobody's fault"

**Key principles:**

- **Past a certain point, even a perfect model loses the trajectory** — because the world itself amplifies tiny differences; grading against zero error would condemn a flawless model. This is WD-4's entire justification.
- **Divergence is not a single constant rate** — Lorenz's own system has energy/regime-dependent behaviour; this project's WD-4 correctly avoids claiming one number for either world, instead requiring a measured curve, and correctly notes Lotka-Volterra is not chaotic in Lorenz's sense (linear phase drift, not exponential divergence) while the double pendulum is.
- **Weather vs. climate** — past the predictability horizon, stop grading the exact trajectory and grade the statistical pattern instead. This is JU-6's switch mechanism, named directly after Lorenz's own field's practice.

---

## A. Philip Dawid

**Source:** A. Philip Dawid, "Present Position and Potential Developments: Some Personal Views — Statistical Theory: The Prequential Approach," *Journal of the Royal Statistical Society: Series A*, 147(2), 278–292 (1984)

**Expert score:**
| Authority | Publication | Breadth | Adoption | Currency | Classification |
|-----------|-------------|---------|----------|----------|----------------|
| 5 | 4 | 4 | 4 | 3 | **Established** |

Evidence: Authority — Dawid is a leading figure in statistical theory (University College London), and the prequential principle is a foundational concept in sequential forecast evaluation. Publication — JRSS-A, a top statistics journal. Adoption — the standard citation for "judge a forecaster only on realized predictions vs. outcomes, never on internal mechanism."

**Work score — "Statistical Theory: The Prequential Approach":**
| Specificity | Depth | Currency | Influence | Classification |
|-------------|-------|----------|-----------|----------------|
| 4 | 4 | 3 | 5 | **Established** |

Evidence: Specificity — precisely states the principle that a forecaster should be evaluated purely on the sequence of predictions and outcomes, not on how the forecasts were generated. Influence — this is JU-1's entire architecture: the judge sees only predictions, uncertainties, and outcomes, never model identity or internals.

**Activation signals:** Prequential principle, evaluate forecasts not mechanisms, blind evaluation

**Key principles:**

- **Judge a forecaster only on what it predicted and what happened** — never on how it works inside. This is JU-1's founding principle, stated almost verbatim in requirements.md.
- **Blindness is not the same as organisational independence** — the prequential principle guarantees the judge cannot favour a model for its architecture or reputation; it says nothing about whether the same person built both the model and the judge. This distinction is exactly what this project's JU-10 discloses (same-author blind-not-independent) — a genuinely honest reading of Dawid's principle, not an overclaim of it.

---

## Jolliffe & Stephenson

**Source:** Ian T. Jolliffe & David B. Stephenson (eds.), *Forecast Verification: A Practitioner's Guide in Atmospheric Science* (2nd ed.), Wiley-Blackwell (2012)

**Expert score:**
| Authority | Publication | Breadth | Adoption | Currency | Classification |
|-----------|-------------|---------|----------|----------|----------------|
| 4 | 4 | 5 | 4 | 4 | **Established** |

Evidence: Authority — the standard practitioner-level textbook for forecast verification methodology, widely used in operational meteorology training. Publication — Wiley-Blackwell, 2nd edition. Breadth — comprehensive catalogue of verification measures across the whole field, not a single technique. Adoption — the reference text cited when someone needs "which measure applies to which kind of forecast."

**Work score — *Forecast Verification* (2nd ed.):**
| Specificity | Depth | Currency | Influence | Classification |
|-------------|-------|----------|-----------|----------------|
| 4 | 5 | 4 | 4 | **Established** |

Evidence: Specificity — catalogues verification measures and their known failure traps in practical, applicable detail. Influence — the practitioner's-eye-view source behind this project's general awareness that verification measures have traps (e.g. JU-8's insistence on independent trials rather than pooled correlated steps is exactly the kind of trap this text catalogues).

**Activation signals:** Verification measure catalogue, practitioner traps in scoring, forecast evaluation methodology

**Key principles:**

- **Verification measures have known traps, catalogued across decades of operational practice** — e.g. pooling correlated observations inflates apparent sample size and biases exception-rate tests toward false alarms, exactly the trap JU-8 explicitly guards against.
- **Different forecast types need different verification approaches** — a categorical forecast, a probabilistic forecast, and a continuous forecast each need their own measures; this project's judge correctly varies its approach (JU-2 skill score, JU-4 calibration, JU-6 statistical-agreement past the divergence horizon) rather than applying one measure everywhere.

---

*Developed using the Grounded Vibe Methodology*
