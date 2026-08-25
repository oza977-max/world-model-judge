---
domain_name: ai_evaluation
activation_signals:
  - ai_accountability
  - model_cards
  - benchmark_leakage
  - llm_as_judge
  - evaluation_coverage
strong_signals:
  - helm
  - internal_algorithmic_audit
---

# Industry Domain Specialists — AI Evaluation & Accountability (Project: World Model Judge)

> **Project-scoped file.** Grounds this project's honesty-in-reporting discipline (what to disclose, what leakage to avoid, why the judge avoids being itself an AI). Append-only within this project.

## Activation Signals

Activate this file for any application dealing with: auditing an AI system as an institution rather than a single metric, reporting a model's intended limits, train/test leakage in ML evaluation, or the documented biases of AI systems used to judge other AI systems.

---

## Deborah Raji

**Source:** Inioluwa Deborah Raji et al., "Closing the AI Accountability Gap: Defining an End-to-End Framework for Internal Algorithmic Auditing," *FAT\* '20: Proceedings of the 2020 Conference on Fairness, Accountability, and Transparency* (2020)

**Expert score:**
| Authority | Publication | Breadth | Adoption | Currency | Classification |
|-----------|-------------|---------|----------|----------|----------------|
| 4 | 4 | 4 | 4 | 4 | **Established** |

Evidence: Authority — Raji is a leading researcher in AI accountability and algorithmic auditing (Mozilla Foundation / UC Berkeley lineage); this paper is widely cited in the AI-governance literature as a framework paper, not a position piece. Publication — FAT* (now FAccT), the leading venue for AI fairness/accountability research. Breadth — her work spans auditing frameworks, facial-recognition evaluation (Gender Shades follow-ons), and evaluation policy. Adoption — widely cited as the closest existing academic relative to "auditing as an institution." Currency — actively publishing in this area; the audit-framework line remains current.

**Work score — "Closing the AI Accountability Gap":**
| Specificity | Depth | Currency | Influence | Classification |
|-------------|-------|----------|-----------|----------------|
| 4 | 4 | 4 | 4 | **Established** |

Evidence: Specificity — proposes a concrete internal-audit process (not just a call for accountability), the closest existing academic relative to this project's own institutional framing. Depth — a staged end-to-end framework (scoping, mapping, artifact collection, testing, reflection) with defined artefacts per stage. Influence — cited in requirements.md explicitly as "the closest existing relative to this thesis — auditing as an institution, not a metric," and this project's honest positioning ("cite the fragments that do exist rather than claiming a vacuum," per the risk assessment's mitigation for Value Risk) depends on this citation existing.

**Activation signals:** Auditing as institution, internal algorithmic audit framework, accountability gap

**Key principles:**

- **Auditing is an institutional process, not a metric.** A checklist of measurements is not the same as an organisation that owns, reports, and can act on the number. This is this project's own central distinction, independently arrived at and then correctly grounded in Raji's prior work rather than presented as novel.
- **Internal audits need a defined end-to-end process** — discovery, scoping, evaluation, reflection, post-audit — not an ad hoc check. This project's own MU-6/JU-11 pre-registration discipline (recipe and thresholds fixed before judging) mirrors this structured-process instinct.

---

## Margaret Mitchell et al.

**Source:** Margaret Mitchell et al., "Model Cards for Model Reporting," *FAT\* '19: Proceedings of the Conference on Fairness, Accountability, and Transparency* (2019)

**Expert score:**
| Authority | Publication | Breadth | Adoption | Currency | Classification |
|-----------|-------------|---------|----------|----------|----------------|
| 5 | 5 | 4 | 5 | 4 | **Canonical** |

Evidence: Authority — Mitchell led Google's AI ethics documentation work; "Model Cards" is one of the most widely adopted AI documentation standards in the field. Publication — FAT* '19, foundational venue paper. Breadth — her work spans documentation standards, bias measurement, and ML fairness, not one artefact. Adoption — Model Cards are now a standard feature on major model-hosting platforms (e.g. Hugging Face), making this one of the most operationally adopted papers in applied AI ethics. Currency — the template remains in active platform-level use. Computed average 4.6 — classified Canonical per the scoring bands.

**Work score — "Model Cards for Model Reporting":**
| Specificity | Depth | Currency | Influence | Classification |
|-------------|-------|----------|-----------|----------------|
| 5 | 4 | 4 | 5 | **Canonical** |

Evidence: Specificity — a concrete reporting template (intended use, limitations, evaluation data, ethical considerations) that directly parallels this project's JU-9 verdict-record structure. Depth — the template is worked through with per-section guidance and worked examples, not a bare checklist. Currency — still the reference format for model reporting. Influence — JU-9's rationale ("stating what wasn't evaluated matters more than the headline score") is Mitchell's central argument, restated for this project's domain. Computed average 4.5 — boundary classifies upward to Canonical per the scoring bands.

**Activation signals:** Model cards, intended-use disclosure, stating what wasn't evaluated

**Key principles:**

- **Stating what wasn't evaluated matters more than the headline score.** A report that only shows what was measured invites the reader to assume everything else was fine. This is JU-9's explicit requirement (the verdict record's not-tested list) and JU-10's limitations statement.
- **Documentation is part of the model, not an afterthought.** This project's insistence on plain-English "In plain words" restatements throughout requirements.md is the same instinct applied to a requirements document rather than a trained model.

---

## Percy Liang et al.

**Source:** Percy Liang et al., "Holistic Evaluation of Language Models," *Transactions on Machine Learning Research* (2023) / arXiv:2211.09110 (HELM, Stanford CRFM, 2022–)

**Expert score:**
| Authority | Publication | Breadth | Adoption | Currency | Classification |
|-----------|-------------|---------|----------|----------|----------------|
| 4 | 4 | 4 | 4 | 4 | **Established** |

Evidence: Authority — Liang leads Stanford's Center for Research on Foundation Models; HELM is a large, well-resourced, continuously maintained multi-institution benchmark effort. Publication — TMLR (2023) plus an ongoing public leaderboard. Breadth — HELM spans scenarios, metrics, and model families rather than one benchmark axis. Adoption — frequently cited as the standard example of multi-metric, coverage-honest LLM evaluation, as opposed to single-number leaderboards. Currency — continuously maintained and extended (e.g. domain-specific HELM variants).

**Work score — HELM:**
| Specificity | Depth | Currency | Influence | Classification |
|-------------|-------|----------|-----------|----------------|
| 4 | 4 | 4 | 4 | **Established** |

Evidence: Specificity — HELM's core design choice (report a matrix of scenarios × metrics, not one leaderboard number) directly parallels this project's refusal to reduce a verdict to a single trust number (JU-7: never report a trust horizon without its task and tolerance attached). Depth — dozens of scenarios crossed with seven metric categories, with explicit incompleteness reporting. Influence — the general principle "a single headline number misleads; report coverage honestly" is HELM's central contribution, cited in requirements.md's expert table.

**Activation signals:** Multi-metric evaluation, coverage honesty, no single leaderboard number

**Key principles:**

- **A single headline number misleads.** Report the full coverage matrix, and be explicit about what wasn't tested. This project's JU-9 (six required fields, not one score) and NF-5 (no claim beyond the evidence) both apply this directly.
- **Evaluation coverage itself needs to be reported, not just results.** What scenarios/models/metrics were and weren't run is itself information the reader needs — the same instinct behind this project's not-tested list.

---

## Kapoor & Narayanan

**Source:** Sayash Kapoor & Arvind Narayanan, "Leakage and the Reproducibility Crisis in Machine-Learning-Based Science," *Patterns*, 4(9) (2023)

**Expert score:**
| Authority | Publication | Breadth | Adoption | Currency | Classification |
|-----------|-------------|---------|----------|----------|----------------|
| 4 | 4 | 4 | 4 | 5 | **Established** |

Evidence: Authority — Narayanan (Princeton CITP) and Kapoor are widely recognised voices on ML reproducibility and evaluation rigor, including the book *AI Snake Oil*. Publication — *Patterns* (Cell Press), a peer-reviewed venue. Breadth — systematic review across 294 papers in 17 fields documenting leakage as pervasive. Currency — 2023, actively cited in ongoing ML-methodology discourse.

**Work score — "Leakage and the Reproducibility Crisis":**
| Specificity | Depth | Currency | Influence | Classification |
|-------------|-------|----------|-----------|----------------|
| 5 | 4 | 5 | 4 | **Canonical** |

Evidence: Specificity — catalogues concrete leakage types (e.g. temporal leakage, duplicate-record leakage) with worked examples across scientific fields. Depth — a systematic survey across 294 papers in 17 fields with a taxonomy and model-info-sheet remedy, not a commentary piece. Influence — this project's MU-7 (disjoint training/evaluation data) and its rationale ("a major, underappreciated cause" of overstated results — scoped this way rather than as definitively "the biggest") trace directly to this paper. Computed average 4.5 — boundary classifies upward to Canonical per the scoring bands.

**Activation signals:** Train/test leakage, reproducibility crisis, disjoint data requirement

**Key principles:**

- **Leakage is pervasive and underappreciated as a cause of overstated ML results** — the paper documents it across 294 papers in 17 fields; the claim should be scoped as "a major cause," not asserted as definitively "the biggest" — the scoping this project's MU-7 rationale uses.
- **Disjoint data is a structural requirement, not a courtesy** — a judge whose subject was trained on the test set measures nothing at all. Directly grounds MU-7.

---

## Zheng et al.

**Source:** Lianmin Zheng et al., "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena," *NeurIPS 2023 Datasets and Benchmarks Track* (2023)

**Expert score:**
| Authority | Publication | Breadth | Adoption | Currency | Classification |
|-----------|-------------|---------|----------|----------|----------------|
| 4 | 4 | 3 | 4 | 4 | **Established** |

Evidence: Authority — widely cited paper documenting systematic biases in using LLMs to evaluate other LLMs (verbosity bias, position bias, self-enhancement bias). Publication — NeurIPS, the leading ML conference. Adoption — the standard citation for "LLM-as-judge has documented, specific biases," frequently referenced in subsequent AI-evaluation methodology work.

**Work score — "Judging LLM-as-a-Judge":**
| Specificity | Depth | Currency | Influence | Classification |
|-------------|-------|----------|-----------|----------------|
| 5 | 4 | 4 | 5 | **Canonical** |

Evidence: Specificity — names and empirically measures specific bias types (position bias, verbosity bias, self-enhancement bias where a model favours its own outputs). Depth — controlled measurement across two benchmarks with agreement analysis against human judgments, not anecdote. Currency — remains the standard citation as LLM-as-judge use has grown. Influence — this project's JU-13 (no language model or learned component in the judge) exists specifically to avoid this entire class of problem, and the rationale cites this paper directly in requirements.md. Computed average 4.5 — boundary classifies upward to Canonical per the scoring bands.

**Activation signals:** LLM-as-judge bias, position bias, verbosity bias, self-enhancement bias

**Key principles:**

- **Automated judges built on language models have documented, specific biases** — favouring longer answers (verbosity bias), favouring whichever answer came first (position bias), and favouring their own outputs (self-enhancement bias). Not a vague worry — empirically measured.
- **The fix is architectural, not procedural** — a plain-arithmetic judge (JU-13) avoids this entire class of problem by construction, rather than trying to correct for the bias after the fact.

---

*Developed using the Grounded Vibe Methodology*
