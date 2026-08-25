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

> **Project-scoped file, named `model-risk-world-model-judge.md` to avoid colliding with the generic `model-risk.md` in this same directory.** Grounds the World Model Judge project's specific transplant of banking model-risk practice onto learned world models. Experts already scored in the generic library (Derman, Taleb in `model-risk.md`; Rebonato in `market-risk.md`) are cross-referenced, not re-scored, per the Single Canonical Source rule. Append-only within this project.

## Activation Signals

Activate this file for any application dealing with: model risk governance for learned/generative simulators, backtesting a learned model's stated confidence against outcomes, exception counting against pre-declared thresholds, independent (or blinded) validation of a model that shares an author with its evaluator, or importing SR 11-7/SS1/23-style institutional practice into a non-banking domain.

---

## Federal Reserve / OCC (SR 11-7), with later FDIC adoption

**Source:** Board of Governors of the Federal Reserve System & Office of the Comptroller of the Currency, *SR 11-7 / OCC 2011-12: Supervisory Guidance on Model Risk Management* (2011); adopted by the FDIC in 2017 (FIL-22-2017); superseded April 2026 by *SR 26-2*

**Expert score:**
| Authority | Publication | Breadth | Adoption | Currency | Classification |
|-----------|-------------|---------|----------|----------|----------------|
| 5 | 5 | 4 | 5 | 3 | **Established** |

Evidence: Authority — joint supervisory guidance from the Federal Reserve and OCC (2011), later adopted by the FDIC (2017), binding on every regulated U.S. bank. Publication — formal interagency guidance, publicly issued. Breadth — covers the full model lifecycle (development, implementation, use, validation, governance), not a single technique, though scoped to banking models rather than risk practice generally. Adoption — became the de facto global template for model risk management programmes at banks worldwide, referenced far beyond U.S. jurisdiction. Currency — SR 11-7 itself is superseded (April 2026, by SR 26-2, which excludes generative/agentic AI from scope — directly relevant to this project's own gap claim); scored down on currency for this reason, not on the underlying practice's validity.

**Work score — *SR 11-7*:**
| Specificity | Depth | Currency | Influence | Classification |
|-------------|-------|----------|-----------|----------------|
| 4 | 4 | 2 | 5 | **Established** |

Evidence: Specificity — defines "effective challenge," independent validation, and ongoing monitoring precisely enough to audit against. Depth — a full governance framework (roles, lifecycle stages, validation elements), not a checklist, though it deliberately stops short of prescribing numeric thresholds. Influence — the template this project explicitly borrows (backtesting, ex-ante thresholds), named directly in requirements.md JU-1/JU-10/JU-11. Currency scored low: literally superseded, and superseded specifically on the question this project asks (does it cover generative AI) — see SR 26-2 below, which this project should treat as the more current citation for scope claims.

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
| 5 | 5 | 4 | 4 | 5 | **Canonical** |

Evidence: Authority — UK's prudential banking regulator, binding on UK-regulated banks. Publication — formal supervisory statement. Breadth — five principles spanning identification, governance, development, validation, and risk mitigation across all model types including AI/ML, wider in stated scope than SR 11-7. Adoption — treated as a leading model risk framework alongside SR 11-7, cited internationally in model risk literature; scored below SR 11-7 on adoption because it is newer and UK-scoped. Currency — 2023, current and not superseded as of this project's writing. Computed average 4.6 — classified Canonical per the scoring bands.

**Work score — *SS1/23*:**
| Specificity | Depth | Currency | Influence | Classification |
|-------------|-------|----------|-----------|----------------|
| 5 | 4 | 5 | 4 | **Canonical** |

Evidence: Specificity — explicit principles on risk appetite for model risk stated in advance, not reactively. Depth — principle-by-principle expectations with sub-principles and implementation guidance, at supervisory rather than technical depth. Currency — in force, and explicitly written to cover AI/ML models. Influence — this project's JU-11 ("all thresholds... fixed and recorded before the judge is run") is a direct application of SS1/23's ex-ante risk-appetite principle. Computed average 4.5 — boundary classifies upward to Canonical per the scoring bands.

**Activation signals:** Ex-ante risk appetite, model risk principles, pre-declared thresholds, banking model governance

**Key principles:**

- **State model risk appetite in advance, not after seeing results** — the principle behind this project's JU-11 (thresholds fixed and recorded before the judge is first run against the unrigged models) and MU-6 (recipe and predicted ranking committed before judging).
- **Principles-based, not purely rules-based** — SS1/23 sets outcomes (e.g. "understand and manage model risk") rather than prescribing exact numeric thresholds, leaving implementation to the institution — consistent with this project's Open Questions 1/2 deferring exact band numbers to the technical spec rather than guessing them here.

---

## Emanuel Derman

*Expert scored in the generic `model-risk.md` (this directory). Classification: **Established** (avg 4.2). Work — *Models.Behaving.Badly*: **Established** (avg 3.75).*

**Project-specific key principles** (how this project uses Derman — no re-scoring):

- **A model is an analogy, not the truth** — the dangerous simplifications are the ones nobody notices they've made. Grounds this project's insistence that a toy world "validates the harness, not the field" (JU-10).
- **Know what you're ignoring** — Derman's rule, applied directly in JU-10's requirement that every verdict states the judge's own limitations, including what the toy world cannot represent (WD-8's exclusions: no randomness, no hidden state, no genuine off-model surprise).

---

## Riccardo Rebonato

*Expert scored in `market-risk.md` (this directory). Classification: **Established** (avg 4.2). Work — *Plight of the Fortune Tellers*: **Established** (avg 3.75).*

**Project-specific key principles** (how this project uses Rebonato — no re-scoring):

- **Backtesting validates the ordinary case, not the disaster** — a model can pass every backtest and still be structurally wrong exactly when it matters most. This is the specific objection JU-10 requires every verdict to disclose about itself ("this method validates the middle of the distribution rather than the extremes").
- **Structural breaks are outside what historical validation can see** — directly parallel to this project's WD-8 exclusion of genuine off-model surprises, now explicitly disclosed in JU-10 as an untested case.

---

## Nassim Nicholas Taleb

*Expert scored in the generic `model-risk.md` (this directory). Classification: **Canonical** (avg 4.8). Work — *The Black Swan* (2nd ed.): **Established** (avg 3.5).*

**Project-specific key principles** (how this project uses Taleb — no re-scoring):

- **The risks that matter most are the ones the model doesn't represent at all** — distinct from ordinary tail risk (an unlikely outcome the model's distribution still covers); this is an event the model's frame excludes entirely. WD-8's deliberate exclusions (no randomness, no hidden state, no high-dimensional input) are exactly this kind of unrepresented risk, made explicit rather than hidden.

---

*Developed using the Grounded Vibe Methodology*
