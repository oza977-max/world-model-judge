---
domain_name: world_models
activation_signals:
  - world_model
  - action_conditioned_prediction
  - jepa
  - latent_imagination
  - model_based_rl
  - rollout_horizon
strong_signals:
  - compounding_error
  - genie_latent_actions
---

# Industry Domain Specialists — World Models & Model-Based RL (Project: World Model Judge)

> **Project-scoped file.** Grounds this project's technical account of what a world model is and why drift/compounding error is its characteristic failure mode. Append-only within this project.

## Activation Signals

Activate this file for any application dealing with: action-conditioned prediction of a system's next state, learned simulators used for planning, latent-space (JEPA-style) prediction, model-based reinforcement learning, or the compounding-error problem in multi-step rollouts.

---

## David Ha & Jürgen Schmidhuber

**Source:** David Ha & Jürgen Schmidhuber, "World Models," arXiv:1803.10122 / NeurIPS 2018

**Expert score:**
| Authority | Publication | Breadth | Adoption | Currency | Classification |
|-----------|-------------|---------|----------|----------|----------------|
| 5 | 5 | 4 | 5 | 3 | **Established** |

Evidence: Authority — the paper that fixed the term "world model" in its modern ML sense (Ha at Google Brain at time of writing; Schmidhuber a foundational RNN/RL researcher with the concept's roots in his own early-1990s controller/model work). Publication — NeurIPS 2018, top ML venue. Adoption — the canonical starting citation for any "world models" literature review since 2018. Currency scored down: the field has moved substantially past the paper's specific VAE+RNN architecture, though the naming and framing endure.

**Work score — "World Models":**
| Specificity | Depth | Currency | Influence | Classification |
|-------------|-------|----------|-----------|----------------|
| 4 | 4 | 3 | 5 | **Established** |

Evidence: Specificity — concretely demonstrates training a policy inside a learned, compressed simulation of an environment. Influence — the paper this essay's title and central framing trace to directly; "the name stuck in 2018" is a factual claim this citation supports.

**Activation signals:** World model definition, learned environment simulation, train-inside-a-dream

**Key principles:**

- **A world model predicts the environment's own dynamics, learned from data, that an agent can then plan or train inside** — distinct from a policy or a value function. This is the field-level definition this project's WD-2 operationalises (state + action → next state).
- **The name and framing are recent (2018) but the idea is older** — Schmidhuber's own early-1990s work on controller/model pairs predates the popularised term, correctly noted in the essay as "the idea runs back to the early 1990s."

---

## Yann LeCun

**Source:** Yann LeCun, "A Path Towards Autonomous Machine Intelligence," OpenReview preprint (2022)

**Expert score:**
| Authority | Publication | Breadth | Adoption | Currency | Classification |
|-----------|-------------|---------|----------|----------|----------------|
| 5 | 4 | 5 | 5 | 4 | **Established** |

Evidence: Authority — LeCun is a Turing Award laureate and one of the most prominent voices arguing that current LLM-scale approaches are insufficient for genuine world modelling; the paper is his fullest public statement of an alternative architecture (JEPA). Publication — widely circulated preprint, extensively discussed rather than formally peer-reviewed at time of writing, scored down slightly on publication accordingly. Breadth — the position spans architecture, training objective, and a broader theory of machine intelligence. Adoption — JEPA has since been implemented and extended in multiple follow-on papers (I-JEPA, V-JEPA), establishing real research uptake beyond the position paper itself.

**Work score — "A Path Towards Autonomous Machine Intelligence":**
| Specificity | Depth | Currency | Influence | Classification |
|-------------|-------|----------|-----------|----------------|
| 4 | 4 | 4 | 5 | **Established** |

Evidence: Specificity — proposes predicting in a learned abstract representation space rather than raw pixel/token space specifically because most high-dimensional perceptual detail is unpredictable and irrelevant. Influence — grounds the essay's central "menu" metaphor (language has a pre-built menu of meaningful tokens; the world does not, so a world model must learn what to discard) and is directly cited as the "predict-in-abstract-space position" in requirements.md's expert table.

**Activation signals:** JEPA, predict-in-representation-space, self-supervised world modelling, hierarchical planning

**Key principles:**

- **Predict in a learned abstract space, not raw observation space** — most of a high-dimensional signal (video, pixels) is unpredictable in detail and irrelevant to the task; a world model should discard it rather than model it. This critique targets high-dimensional perceptual domains specifically; it does not straightforwardly apply to this project's own low-dimensional toy worlds (population counts, joint angles), which is correctly noted as an independent finding by this project's own LeCun-grounded review.
- **A world model's real purpose is closed-loop planning** — simulating candidate action sequences and evaluating a cost to select one, with prediction instrumental to that loop rather than terminal in itself. This project's own JU-10 now discloses that its "control"/"planning" tasks (WD-6) test passive, open-loop prediction accuracy, not closed-loop action-selection competence — a distinction this project's LeCun-grounded review found and the project accepted rather than disputed.

---

## Danijar Hafner et al.

**Source:** Danijar Hafner et al., "Dream to Control: Learning Behaviors by Latent Imagination" (Dreamer, ICLR 2020) and successors (DreamerV2, DreamerV3)

**Expert score:**
| Authority | Publication | Breadth | Adoption | Currency | Classification |
|-----------|-------------|---------|----------|----------|----------------|
| 4 | 4 | 4 | 4 | 4 | **Established** |

Evidence: Authority — Hafner's Dreamer line is one of the most influential model-based RL research programmes, spanning multiple top-venue papers (ICLR, with DreamerV3 achieving broad benchmark success including Minecraft diamond collection). Publication — ICLR, a top ML venue, across three successive papers. Adoption — widely cited and reproduced as the reference architecture for "train a policy entirely inside a learned latent world model."

**Work score — Dreamer line:**
| Specificity | Depth | Currency | Influence | Classification |
|-------------|-------|----------|-----------|----------------|
| 3 | 4 | 4 | 4 | **Recognised** |

Evidence: Specificity — Dreamer's core contribution is training behaviour entirely inside imagined latent rollouts, with rollout-error-growth measurement appearing as supporting analysis rather than the paper's central claim; this project's own independent audit correctly flagged that crediting Dreamer as "prior art for the core measurement" somewhat overstates its centrality to that specific methodology relative to the paper's actual headline contribution — reflected here in a lower specificity score than the expert-level score, and by the essay's own softened wording ("has measured how error grows," not "measured exactly").

**Activation signals:** Latent imagination, train-inside-a-dream RL, Dreamer architecture

**Key principles:**

- **A policy can be trained entirely inside a learned latent simulator** — no real-environment interaction needed once the world model is trained. This is the strongest working demonstration that world models are useful for something beyond prediction accuracy alone.
- **Rollout-error growth is measurable and reported, though it is supporting analysis rather than Dreamer's central contribution** — cite this line for "a real system that measures compounding error as part of its evaluation," not as the originating source for the measurement technique itself; Janner et al. (below) is the more specific source for why compounding error justifies short rollouts.

---

## Sutton & Barto

**Source:** Richard S. Sutton & Andrew G. Barto, *Reinforcement Learning: An Introduction* (2nd ed.), MIT Press (2018)

**Expert score:**
| Authority | Publication | Breadth | Adoption | Currency | Classification |
|-----------|-------------|---------|----------|----------|----------------|
| 5 | 5 | 5 | 5 | 4 | **Canonical** |

Evidence: Authority — the standard, most widely used textbook in reinforcement learning; Sutton is a foundational figure in the field (temporal-difference learning). Publication — MIT Press, 2nd edition, the field's reference text. Breadth — covers the full landscape of RL including model-based methods and planning. Adoption — used in the overwhelming majority of university RL courses and cited as the default RL reference across the field.

**Work score — *Reinforcement Learning: An Introduction* (2nd ed.):**
| Specificity | Depth | Currency | Influence | Classification |
|-------------|-------|----------|-----------|----------------|
| 3 | 5 | 4 | 5 | **Established** |

Evidence: Specificity — foundational and comprehensive, but general RL theory rather than a specific claim about world-model rollout-length practice; scored down on specificity relative to Janner et al.'s targeted contribution. Influence — the field's standard grounding text for "planning with a learned model" as a concept, correctly re-scoped by this project's own review to a general foundational role rather than the specific source for the compounding-error/short-rollout argument (which belongs to Janner et al., below) — this correction was applied to both this file and requirements.md's expert table.

**Activation signals:** Model-based RL foundations, planning with a learned model, general RL theory

**Key principles:**

- **Model-based RL uses a learned model of the environment for planning** — the general framework this project's judged models sit within, distinct from model-free RL.
- **General foundational reference, not the source of the specific "why short rollouts" argument** — that argument belongs to Janner et al.; using this citation for that specific claim was a misattribution corrected during this project's own review process.

---

## Janner, Fu, Zhang & Levine

**Source:** Michael Janner, Justin Fu, Marvin Zhang & Sergey Levine, "When to Trust Your Model: Model-Based Policy Optimization" (MBPO), *NeurIPS 2019*

**Expert score:**
| Authority | Publication | Breadth | Adoption | Currency | Classification |
|-----------|-------------|---------|----------|----------|----------------|
| 4 | 4 | 3 | 4 | 4 | **Established** |

Evidence: Authority — Levine's lab (UC Berkeley) is a leading model-based RL research group; MBPO is a well-cited, influential paper specifically on the theoretical justification for rollout-length limits. Publication — NeurIPS 2019, top ML venue. Adoption — widely cited as the source of the theoretical bound relating model rollout length to policy-improvement guarantees.

**Work score — "When to Trust Your Model":**
| Specificity | Depth | Currency | Influence | Classification |
|-------------|-------|----------|-----------|----------------|
| 5 | 4 | 4 | 4 | **Established** |

Evidence: Specificity — the paper's core theoretical contribution is precisely a bound on model rollout length tied to policy-improvement guarantees under compounding model error — exactly the "why do researchers keep imagined rollouts short" claim this project needed a source for. Influence — this project's own review process identified this as the correct, specific attribution (replacing an earlier misattribution to Dreamer/Sutton & Barto), and both the essay and requirements.md were corrected accordingly.

**Activation signals:** Compounding rollout error, model rollout length bound, when to trust a learned model

**Key principles:**

- **Compounding model error is why imagined rollouts are kept short** — the specific, named theoretical justification, distinct from Dreamer's demonstration that long rollouts can be made to work in practice under favourable conditions.
- **A model should be trusted only as far as its own error growth permits** — the conceptual ancestor of this project's own WD-4 divergence-benchmark and JU-7 trust-horizon machinery, applied here to model-based RL policy improvement rather than to a governance verdict.

---

*Developed using the Grounded Vibe Methodology*
