---
domain_name: model_risk
activation_signals:
  - model_risk
  - model_validation
  - backtesting
  - exception_count
  - sr_11_7
  - sr_26_2
  - ss1_23
  - model_risk_management
  - independent_validation
strong_signals:
  - trust_horizon
  - challenger_model
  - effective_challenge
---

# Industry Domain Specialists — Model Risk (Project: World Model Judge)

> **Project-scoped file, named `model-risk-world-model-judge.md` to avoid colliding with the generic `model-risk.md` in this same directory.** Grounds the World Model Judge project's specific transplant of banking model-risk practice onto learned world models. Append-only within this project.

## Activation Signals

Activate this file for any application dealing with: model risk governance for learned/generative simulators, backtesting a learned model's stated confidence against outcomes, exception counting against pre-declared thresholds, independent (or blinded) validation of a model that shares an author with its evaluator, or importing SR 11-7/SS1/23-style institutional practice into a non-banking domain.

---

## Federal Reserve / OCC / FDIC

**Source:** Board of Governors of the Federal Reserve System, Office of the Comptroller of the Currency, Federal Deposit Insurance Corporation, *SR 11-7: Guidance on Model Risk Management* (2011), superseded April 2026 by *SR 26-2*

**Expert score:**
| Authority | Publication | Breadth | Adoption | Currency | Classification |
|-----------|-------------|---------|----------|----------|----------------|
| 5 | 5 | 4 | 5 | 3 | **Established** |

Evidence: Authority — joint supervisory guidance from the three U.S. federal banking regulators, binding on every regulated bank. Publication — formal interagency guidance, publicly issued. Adoption — became the de facto global template for model risk management programmes at banks worldwide, referenced far beyond U.S. jurisdiction. Currency — SR 11-7 itself is superseded (April 2026, by SR 26-2, which excludes generative/agentic AI from scope — directly relevant to this project's own gap claim); scored down one band on currency for this reason, not on the underlying practice's validity.

**Work score — *SR 11-7*:**
| Specificity | Depth | Currency | Influence | Classification |
|-------------|-------|----------|-----------|----------------|
| 4 | 4 | 2 | 5 | **Established** |

Evidence: Specificity — defines "effective challenge," independent validation, and ongoing monitoring precisely enough to audit against. Influence — the template this project explicitly borrows (backtesting, ex-ante thresholds), named directly in requirements.md JU-1/JU-10/JU-11. Currency scored low: literally superseded, and superseded specifically on the question this project asks (does it cover generative AI) — see SR 26-2 below, which this project should treat as the more current citation for scope claims.

**Activation signals:** Model risk governance, independent validation, effective challenge, backtesting discipline, ongoing monitoring

**Key principles:**

- **Effective challenge** — model risk management requires a party organisationally independent of the model's builders, empowered to challenge and potentially block. This project explicitly cannot claim this (JU-1/JU-10: same author, blind not independent) — the gap is disclosed, not solved.
- **Backtesting with counted exceptions** — outcomes checked against stated confidence after the fact, misses ("exceptions") counted against thresholds set in advance. This is the mechanism this project's JU-8 imports directly.
- **Ongoing monitoring, not a one-time check** — model performance is tracked continuously in production, not validated once at build time. This project does not attempt this (disclosed in JU-10) — a toy-scale, single-run demonstration is not ongoing monitoring.
- **SR 26-2 (2026) explicitly excludes generative and agentic AI from formal scope**, calling them "novel and rapidly evolving" — direct, current evidence that even the freshest version of this institution has not been pointed at the class of model this project judges. Cite SR 26-2 for scope claims about the current state of banking guidance; cite SR 11-7 for the specific mechanism (backtesting, thresholds) this project borrows, since that mechanism predates and survives the 2026 update.

---

## Prudential Regulation Authority

**Source:** Bank of England Prudential Regulation Authority, *SS1/23: Model Risk Management Principles for Banks* (2023)

**Expert score:**
| Authority | Publication | Breadth | Adoption | Currency | Classification |
|-----------|-------------|---------|----------|----------|----------------|
| 5 | 5 | 4 | 4 | 5 | **Established** |

Evidence: Authority — UK's prudential banking regulator, binding on UK-regulated banks. Publication — formal supervisory statement. Adoption — treated as a leading model risk framework alongside SR 11-7, cited internationally in model risk literature. Currency — 2023, current and not superseded as of this project's writing.

**Work score — *SS1/23*:**
| Specificity | Depth | Currency | Influence | Classification |
|-------------|-------|----------|-----------|----------------|
| 5 | 4 | 5 | 4 | **Established** |

Evidence: Specificity — explicit principles on risk appetite for model risk stated in advance, not reactively. Influence — this project's JU-11 ("all thresholds... fixed and recorded before the judge is run") is a direct application of SS1/23's ex-ante risk-appetite principle.

**Activation signals:** Ex-ante risk appetite, model risk principles, pre-declared thresholds, banking model governance

**Key principles:**

- **State model risk appetite in advance, not after seeing results** — the principle behind this project's JU-11 (thresholds fixed and recorded before the judge is first run against the unrigged models) and MU-6 (recipe and predicted ranking committed before judging).
- **Principles-based, not purely rules-based** — SS1/23 sets outcomes (e.g. "understand and manage model risk") rather than prescribing exact numeric thresholds, leaving implementation to the institution — consistent with this project's Open Questions 1/2 deferring exact band numbers to the technical spec rather than guessing them here.

---

## Emanuel Derman

**Source:** Emanuel Derman, *Models.Behaving.Badly: Why Confusing Illusion with Reality Can Lead to Disaster, on Wall Street and in Life*, Free Press (2011)

**Expert score:**
| Authority | Publication | Breadth | Adoption | Currency | Classification |
|-----------|-------------|---------|----------|----------|----------------|
| 4 | 4 | 4 | 4 | 4 | **Established** |

Evidence: Authority — former head quant at Goldman Sachs, co-author of the Black-Derman-Toy interest rate model; widely regarded voice on the philosophy of financial modelling. Publication — Free Press, general audience but well-reviewed. Adoption — frequently cited in model-risk and quant-finance discourse as the accessible statement of "a model is not the world."

**Work score — *Models.Behaving.Badly*:**
| Specificity | Depth | Currency | Influence | Classification |
|-------------|-------|----------|-----------|----------------|
| 3 | 4 | 3 | 4 | **Recognised** |

Evidence: Specificity — general epistemology of modelling rather than a technical methodology; scored down on specificity for this reason. Influence — the book's central distinction (model as analogy vs. model as truth) is directly load-bearing in this project's JU-10 self-limitations statement and the "scope honesty" callout in requirements.md's Purpose & Vision section.

**Activation signals:** Model risk philosophy, model-vs-reality distinction, unnoticed simplifying assumptions

**Key principles:**

- **A model is an analogy, not the truth** — the dangerous simplifications are the ones nobody notices they've made. Grounds this project's insistence that a toy world "validates the harness, not the field" (JU-10).
- **Know what you're ignoring** — Derman's rule, applied directly in JU-10's requirement that every verdict states the judge's own limitations, including what the toy world cannot represent (WD-8's exclusions: no randomness, no hidden state, no genuine off-model surprise).

---

## Riccardo Rebonato

**Source:** Riccardo Rebonato, *Plight of the Fortune Tellers: Why We Need to Manage Financial Risk Differently*, Princeton University Press (2007)

**Expert score:**
| Authority | Publication | Breadth | Adoption | Currency | Classification |
|-----------|-------------|---------|----------|----------|----------------|
| 4 | 4 | 3 | 3 | 3 | **Recognised** |

Evidence: Authority — global head of rates and FX options research at RBS at time of writing, PhD physicist; recognised voice on the limits of quantitative risk management. Publication — Princeton University Press, academic-adjacent. Adoption — cited in model-risk and financial-risk-management circles, less broadly known outside quant finance than Derman.

**Work score — *Plight of the Fortune Tellers*:**
| Specificity | Depth | Currency | Influence | Classification |
|-------------|-------|----------|-----------|----------------|
| 4 | 4 | 3 | 3 | **Recognised** |

Evidence: Specificity — direct argument that backtesting-style validation is structurally weakest exactly where the stakes are highest (tail events, structural breaks). Influence — this project's JU-10 disclosure ("this method validates the middle of the distribution rather than the extremes") states Rebonato's objection turned on the project's own method, per the citation already in requirements.md.

**Activation signals:** Tail risk validation gap, backtesting limitations, model risk in crisis conditions

**Key principles:**

- **Backtesting validates the ordinary case, not the disaster** — a model can pass every backtest and still be structurally wrong exactly when it matters most. This is the specific objection JU-10 requires every verdict to disclose about itself.
- **Structural breaks are outside what historical validation can see** — directly parallel to this project's WD-8 exclusion of genuine off-model surprises, now explicitly disclosed in JU-10 as an untested case.

---

## Nassim Nicholas Taleb

**Source:** Nassim Nicholas Taleb, *The Black Swan: The Impact of the Highly Improbable* (2nd ed.), Random House (2010)

**Expert score:**
| Authority | Publication | Breadth | Adoption | Currency | Classification |
|-----------|-------------|---------|----------|----------|----------------|
| 4 | 5 | 5 | 5 | 4 | **Established** |

Evidence: Authority — former derivatives trader, widely credited with popularising tail-risk thinking beyond quant finance into general risk discourse. Publication — Random House, international bestseller, 2nd edition. Breadth — the argument extends far beyond finance into any domain where a model's training distribution doesn't cover the event that eventually happens. Adoption — one of the most cited popular risk-management texts across finance, policy, and technology writing.

**Work score — *The Black Swan* (2nd ed.):**
| Specificity | Depth | Currency | Influence | Classification |
|-------------|-------|----------|-----------|----------------|
| 3 | 4 | 4 | 5 | **Established** |

Evidence: Specificity — general risk philosophy rather than a technical validation methodology, scored down accordingly. Influence — directly grounds this project's disclosure that the risks that matter most are the ones the model doesn't represent at all, a distinct point from Rebonato's tail-of-distribution objection (Taleb: the event isn't even in the distribution).

**Activation signals:** Unrepresented risk, model blind spots, out-of-distribution catastrophe

**Key principles:**

- **The risks that matter most are the ones the model doesn't represent at all** — distinct from ordinary tail risk (an unlikely outcome the model's distribution still covers); this is an event the model's frame excludes entirely. WD-8's deliberate exclusions (no randomness, no hidden state, no high-dimensional input) are exactly this kind of unrepresented risk, made explicit rather than hidden.

---

*Developed using the Grounded Vibe Methodology*
