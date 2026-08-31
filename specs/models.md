# World Model Judge — Models Specification

Version 1.5 · 31 August 2026 · Domain 2 of Requirements v1.2 · References: cross-cutting spec v1.5, worlds spec v1.4

> **Change note (v1.5 — design-review-005 repair).** Three interface gaps Round 5 found on the v1.4 flagship (WorldContext/uniform factory) are closed at the interface, not patched around: (1) `WorldContext` gains `scale` — ADR-M3's MLP normalises by "the world's scale vector," which was reachable only by the forbidden `wmj.worlds` import; (2) the factory signature gains a `TrainingData` argument — the interface had `reset`/`predict` but no channel to receive training data, leaving 6 of 7 models unbuildable; (3) fixtures now build their inner net from the same content-addressed seed as `direct`, so "a copy of Model A" is bit-identical, not a second independent training run. Also: the auto-discovery import path's arbitrary-code-on-import is disclosed with an explicit (out-of-scope) threat model; MU-9's "one file" claim is honestly scoped for a new *unrigged* contestant (which does require a `prereg/` entry — the derived baseline/fixture classification needs no `is_unrigged` field); `training_action_interval` shape corrected to `[a, 2]`. See design-review-005.html.

> **Change note (v1.4 — the design-review-004 structural repair).** This round repairs by rule, not by instance, per the four-round pattern recorded in `reviews/calibration.md`. ADR-M1 gains the **uniform factory signature** — `factory(ctx: WorldContext, rng) -> Model`, with `WorldContext` defined in `wmj.models.base` (the judge's own deliberate-duplication pattern) and constructed only by the harness — one decision that closes Round 4's three interlocking Criticals (the models→worlds import contradiction, the two-worlds selection gap, and the mutual exclusion with TC-MU9-01). The registry contract is pinned (`register(name, factory)`, `DuplicateModelError` on collision, `all_models()` returning sorted-name order as a stated contract). The seven canonical model names are pinned as literal join keys, and the roster count is corrected to **seven** everywhere ("eight" was never true against this document's own enumeration). `fx-brittle`'s trigger now checks both WD-5 axes (state box and action interval) via `ctx`. ADR-M5's GitHub-API mitigation is **removed** (undisclosed network dependency, silent ~90-day decay) in favour of plain disclosure. See design-review-004.html.

> **Change note (v1.3).** Revised after `/gvm-design-review` design-review-003 (Round 3, dual/blind). `fx-brittle` redesigned twice over: it is now a genuine post-hoc wrapper around a normally-trained Model A (no separate training run), and its in/out boundary is exactly the world's declared training-region box rather than an independently-chosen sub-region — closing both a wrapper-framing contradiction and a region-label mismatch that three independent reviewers converged on. Fixtures' registration path stated explicitly. `region_labels`' JSON example fixed to the canonical per-trial shape (judge spec v1.3). A third residual risk (prereg commit-timestamp forgery) disclosed, with a partial GitHub-hosting-dependent mitigation. See design-review-003.html for the full findings.

> **Change note (v1.2).** Revised after `/gvm-design-review` design-review-002 (Round 2, new Security panel). ADR-M5's residual-risk paragraph now names a second, distinct disclosed gap alongside the existing git-rewritability one: MU-6's "publish either way" clause has no mechanism proving a judged run of the unrigged models actually happened and was not quietly discarded before an unfavourable result could be published — nothing in the design detects non-publication. Named explicitly rather than left implicit, matching this project's own disclosure discipline. See design-review-002.html for the full findings.

**What this document is.** The design of everything the judge grades: the two dumb baselines, the three deliberately broken fixtures, and the two honest ("unrigged") models — plus the training recipe shape, the accuracy-matching rule, and the pre-registration mechanics that stop any of it being quietly tuned after the fact.

**In plain words:** this spec builds the contestants. The baselines set the floor, the fixtures prove the judge's instruments work, and the two honest models are the actual experiment — matched on accuracy, different only in how they work out their confidence, with everything about them written down before the judge ever sees them.

---

## Expert Panel

| Expert | Work | Role in This Document |
|--------|------|----------------------|
| Kapoor & Narayanan | *Leakage and the Reproducibility Crisis* (2023) | The disjoint train/eval rule (MU-7) and why it is structural, not courtesy |
| Balaji Lakshminarayanan, Alexander Pritzel & Charles Blundell | *Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles* (NeurIPS 2017) | The ensemble-disagreement uncertainty method Model B implements (discovered expert — see below) |
| D. A. Nix & A. S. Weigend | *Estimating the mean and variance of the target probability distribution* (IEEE ICNN 1994) | The direct variance-prediction method Model A implements (discovered expert — see below) |
| Kent Beck | *Test-Driven Development: By Example* | Gradient-check-first discipline for the hand-rolled MLP |
| Allan Murphy | *What Is a Good Forecast?* (1993) | Skill scores as the matching currency for MU-5's margin |
| Michael Keeling | *Design It!* | ADR format |

### Expert Discovery: Uncertainty Estimation

The roster has no specialist for neural-network uncertainty quantification — the core of MU-5. Per the discovery protocol:

**Lakshminarayanan, Pritzel & Blundell** — "Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles", *NeurIPS 2017*. The standard citation for ensemble disagreement as predictive uncertainty: independently initialised networks, trained identically, whose spread of predictions estimates epistemic uncertainty — typically better-calibrated out-of-distribution than a single network's self-estimate. This published finding is exactly why MU-6's pre-registered ranking prediction is "reasonably predictable" (requirements, MU-6 rationale).

**Nix & Weigend** — "Estimating the mean and variance of the target probability distribution", *IEEE ICNN 1994*. The original formulation of a network with a second output head predicting its own error variance, trained under Gaussian negative log-likelihood — Model A's method, in its simplest honest form.

---

## 1. Purpose

Covers requirements **MU-1 through MU-10**. The models package (`wmj.models`) provides: the single model interface in the MU-1 fixed uncertainty format, the two baselines (MU-2), three fixtures with specified failure modes, labelled everywhere (MU-3, MU-4), two unrigged models matched on accuracy and differing on uncertainty method (MU-5), the pre-registration artefacts and their enforcement (MU-6), disjoint train/eval data (MU-7), seeded training (MU-8), and registry-only extensibility (MU-9). Large/pretrained models are excluded (MU-10, Won't).

## 2. Architecturally Significant Requirements

- **MU-9** forces a *registry* architecture: a model is one file that registers a factory; the harness discovers models only through the registry, so adding one touches nothing else (TC-MU9-01's zero-diff check).
- **MU-1 + WD-2** force a small stateful-per-rollout interface (see ADR-M4): "carry on in a straight line" needs the previous state, and the interface must provide it without leaking anything to the judge.
- **MU-6 + JU-11** force pre-registration to be a *mechanically checkable* property of git history, implemented in the harness (the judge is pure and cannot read git — cross-cutting ADR-003).
- **MU-8 + NF-1** force training itself into the determinism regime: hand-rolled Adam, seeded init, fixed batch order.

## 3. Design Decisions

### ADR-M1 — The model interface: reset-then-predict, rollout-local memory allowed

**Status:** Accepted. [Requirement: MU-1, MU-9, WD-2] [Test: TC-MU1-01, TC-MU1-02, TC-MU9-01]

**Context:** MU-1 fixes the prediction format (per-dimension mean + one standard deviation — cross-cutting data model). The linear-extrapolation baseline needs the previous state; a bare `predict(state, action)` cannot supply it.

**Options considered:** (1) pass a history window in every call — bloats the interface every model must implement for the sake of one baseline; (2) **`reset()` at rollout start plus `predict(state, action) → Prediction`, with models permitted rollout-local memory** — the harness calls `reset()` before each rollout; a model may remember what it has seen *within* that rollout and nothing across rollouts.

**Decision:** Option 2. The interface, complete:

```
Model (protocol, wmj.models.base):
  name: str                 # harness/reporting only — never reaches the judge (JU-1)
  is_fixture: bool          # MU-4/RP-8 labelling flag, carried to every output surface
  reset() -> None           # called by the harness at the start of every rollout
  predict(state, action) -> Prediction   # mean: float64[d], spread: float64[d] (1 sd)
```

**The uniform factory signature and `WorldContext` (design-review-004 repair — this one decision closes three Round 4 Criticals at once).** Round 4 found that (a) `fx-brittle` needed the world's training-region box but "models never import world internals" (§6) forbids reaching into `wmj.worlds` for it; (b) no per-world selection mechanism existed for a fixture running against both worlds; and (c) the only workaround — harness special-casing `fx-brittle` by name — would trip the very isolation gate (TC-MU9-01) built to forbid such coupling. All three dissolve if world context reaches **every** model the same way, so no model is special:

```
WorldContext (frozen dataclass, wmj.models.base — defined HERE, not imported from worlds,
              following the same deliberate-duplication pattern the judge already uses
              for its own input types, cross-cutting ADR-003):
  world_name: str                          # "lv" | "pendulum" — for weights-file keying, never judged
  state_dim: int                           # d
  action_dim: int                          # a
  training_state_box: float64[d, 2]        # per-dimension [low, high] of the WD-5 training box
  training_action_interval: float64[a, 2]  # per-action-dimension [low, high] of the trained action range
  scale: float64[d]                        # the world's per-state-dimension normalisation vector
                                           #   (worlds spec §4.3 `scale`, ADR-W3) — design-review-005:
                                           #   ADR-M3's MLP normalises (state, action) by "the world's
                                           #   scale vector"; that vector must reach the model through
                                           #   this channel, since importing wmj.worlds is forbidden (§6).
                                           #   Action inputs are normalised by their training_action_interval
                                           #   half-width, already carried above — no separate action scale.

Factory (every registered model, no exceptions):
  factory(ctx: WorldContext, rng: numpy.random.Generator, training: TrainingData) -> Model

TrainingData (frozen dataclass, wmj.models.base — design-review-005 repair; see below):
  states:  float64[N, H+1, d]   # N seeded training trajectories for this world (models spec §4)
  actions: float64[N, H, a]     # the seeded action sequences that produced them
      # constructed once per world by the harness and handed to every factory identically;
      # the harness owns generation (seeded, MU-7-disjoint) — the model only fits to it.
```

The **harness** constructs one `WorldContext` and one `TrainingData` per world from the world's own declared constants (`World.regions()`/`World.scale`, worlds spec §4.3/ADR-W4) and its seeded training-trajectory generator (§4) — one producer, one construction site, no re-derived copies — and calls every registered factory with them identically, once per world. **The factory is where a model becomes ready-to-predict (design-review-005 repair — Round 5 found the interface had `reset`/`predict` but no channel to receive training data, leaving 6 of 7 models unbuildable): `persistence` and `linear` fit their spread constants from `training` inside the factory; `direct`/`ensemble` train their MLP weights from `training` inside the factory; the fixtures wrap a `direct` built from the same `training` (see ADR-M4). `persistence`'s mean rule needs nothing from `training`, but it still fits its spread from it — no model needs a channel the signature doesn't carry.** A model that doesn't need world *geometry* ignores `ctx` (persistence's mean does); `fx-brittle` and the two MLPs read what they need from `ctx`. No `models → worlds` import exists anywhere (enforced by TC-NF6-07, cross-cutting ADR-003 — design-review-005 repair added the gate for this direction); no factory is called differently from any other; the per-world question is answered structurally (one factory, one instance per world, each holding that world's context and fitted to that world's data).

**The registry contract, pinned (design-review-004 repair — Round 4 found the mechanism named but its interface unspecified, and no duplicate-name rule):**

```
wmj.models.registry:
  register(name: str, factory: Factory) -> None
      # module-level call at import time; raises DuplicateModelError (fail loudly,
      # cross-cutting Error-Handling rule 1) if the name is already registered
  all_models() -> dict[str, Factory]
      # triggers auto-discovery (cross-cutting ADR-003), then returns the registered
      # factories keyed by name, in SORTED-NAME order — the ordering is part of the
      # contract, stated here so determinism never silently depends on any discovery
      # mechanism's internal behaviour; model_ref indices (judge spec §5 envelope)
      # are assigned in this sorted order and are therefore reproducible by contract
```

**The seven canonical model names, pinned as literal join keys** (the same discipline worlds.md applies to region names — Round 4 found the harness must pick out specific models by name for MU-2's baseline extraction and MU-5's unrigged-pair check, with no pinned vocabulary to match against): `"persistence"`, `"linear"`, `"direct"`, `"ensemble"`, `"fx-overconfident"`, `"fx-honest-rough"`, `"fx-brittle"`. **The roster is seven contestants** — design-review-004 repair: "eight registered contestants"/"8 models" appeared in three specs and was never true against this document's own enumeration (2 baselines + direct + ensemble + 3 fixtures = 7), the same never-recounted-number defect class as the chunk-count error two rounds earlier.

**Auto-discovery runs a discovered module's top-level code on import — threat model, stated (design-review-005 repair — Round 5 Security noted this path executes arbitrary module code with no disclosure, unlike ADR-M5's prereg risks):** `all_models()` imports every file under `wmj/models/`, so any file placed there executes on the next call. **The adversary is explicitly out of scope:** this is a single-author project whose author writes every model file; the property auto-discovery protects is *extensibility* (MU-9), not *isolation from hostile code*. Anyone who can write a file into `wmj/models/` already controls the whole checkout. This is a convenience mechanism, not a security boundary — named here so no reader mistakes the AST gates (which harden the *judge's* purity, a different concern) for a sandbox around model code. The same disclosure discipline as ADR-M5's residual risks.

**Consequences:** Rollout-local state is invisible to the judge (which sees only arrays) and harmless to determinism (reset() makes every rollout self-contained) — **provided `reset()` actually clears it (design-review fix — this was previously asserted but never tested)**: §8 adds a cross-rollout isolation test, since a model that silently leaked state across JU-8's 200 independent trials would corrupt the independence assumption the whole exception-band derivation depends on, undetected by any other test in this document. TC-MU1-01/02 check the format and its cross-model identity; the registry is the only discovery path (MU-9), and harness code referencing the pinned name *strings* above (data, not Python identifiers) is explicitly permitted and does not violate MU-9's isolation — TC-MU9-01's import-graph check (cross-cutting ADR-003) forbids importing model *submodules*, not naming models by their registered string keys.

### ADR-M2 — Baselines: persistence and linear extrapolation, with honest training-residual spreads

**Status:** Accepted. [Requirement: MU-2, MU-1] [Test: TC-MU2-01, TC-MU2-02]

**Context:** MU-2 requires both baselines in every verdict. MU-1 requires *every* model under test to state uncertainty in the fixed format — baselines included, or the judge would need a special case, which JU-1's blindness forbids.

**Decision:**
- **Persistence:** mean = current state; spread = per-dimension standard deviation of one-step state *changes* over the training dataset (a constant vector, computed once at fit time). "Nothing changes, give or take how much things usually change."
- **Linear extrapolation:** mean = current + (current − previous) for the same dimension-wise step (first step of a rollout falls back to persistence, having no previous); spread = per-dimension standard deviation of that rule's own residuals on the training data.
- Both are fitted (their spread constants) on exactly the training dataset the learned models use, so their calibration is honest rather than decorative.

**Consequences:** Baselines participate in calibration and exception counting like everyone else — a learned model must beat persistence not only on error (TC-MU2-02's sanity floor) but visibly on the same charts. The judge refuses to run without both baselines present in its input (TC-MU2-01, cross-cutting error conventions).

### ADR-M3 — The two unrigged models: same network, two uncertainty methods

**Status:** Accepted. [Requirement: MU-5, MU-6, MU-8] [Test: TC-MU5-01, TC-MU5-02, TC-MU5-03, TC-MU8-01]

**Context:** MU-5 demands at least two non-engineered models, matched on accuracy to a pre-fixed margin, differing only in how they derive uncertainty: one predicting its own error bar, one small ensemble whose disagreement supplies it.

**Decision:**
- **Model A — "direct" (Nix & Weigend):** one MLP taking `(state, action)` — state normalised by `ctx.scale`, action by its `ctx.training_action_interval` half-width (design-review-005: both arrive through `WorldContext`, the one sanctioned channel; no `wmj.worlds` import) — outputting per-dimension `(Δmean, log σ)` — it predicts the state *change* plus its own per-dimension error bar. Trained with Gaussian negative log-likelihood. `mean = state + Δmean`, `spread = exp(log σ)`.
- **Model B — "ensemble" (Lakshminarayanan et al.):** **K = 5** MLPs with the same architecture minus the variance head (mean output only), identical training data, differing only in seeded initialisation and seeded batch shuffling. Pre-registered rules (the MU-5 clauses, fixed here and repeated in `prereg/recipe.md`):
  - **Point prediction:** the arithmetic mean of the K member means.
  - **Spread mapping:** per-dimension `spread = sqrt(1 + 1/K) × std(member_means, ddof=1)` — sample standard deviation with Bessel's correction, inflated by the standard `sqrt(1 + 1/K)` predictive-variance factor to correct small-ensemble underdispersion (the MU-5 correction clause; TC-MU5-03 checks it is applied).
- **Shared architecture (both models, both worlds):** 2 hidden layers × 64 units, tanh activations, hand-rolled NumPy forward/backward.
  - **Weight initialization (design-review fix — previously unspecified, and MU-8's seeded-reproducibility guarantee only holds if the init scheme is itself fixed):** each weight matrix `W ~ U(−1/√fan_in, 1/√fan_in)` (uniform, fan-in scaled — the standard scheme for tanh networks), all biases initialised to 0, drawn from the run's seeded `Generator` (cross-cutting ADR-002).
  - **Model A's loss, written out (design-review fix — previously named "Gaussian negative log-likelihood" with no formula, unlike CRPS in the judge spec which gets a full closed form):** per-dimension, per-example `NLL = 0.5·log(2π) + logσ + (y − μ)²/(2σ²)`, summed over the d dimensions and averaged over the batch.
  - **Training loop shape (design-review fix — v1.0 said "full-batch gradient checked... before any training run" and, separately, that Model B's members differ in "seeded batch shuffling," which only makes sense for mini-batches; the two statements were never reconciled):** training itself is **mini-batch**, batch size **32**, examples shuffled each epoch from the run's seeded `Generator`. The *gradient check* (below) is a separate, one-time, full-batch finite-difference check run once before training starts — it is not the training loop itself, and "batch shuffling" refers only to the mini-batch training loop's own epoch-order shuffle.
  - Adam optimizer: lr 1e-3, β₁ 0.9, β₂ 0.999, **ε = 1e-8** (design-review fix — previously omitted; two implementers picking different conventional defaults would get bit-different trained weights under this project's own determinism standard).
  - **Gradient check:** one full-batch pass, checked against finite differences to 1e-6 relative, run once before any training run begins (Beck: the failing test first — the gradient check is the MLP's first test; it validates the hand-rolled backprop implementation, and is not part of the mini-batch training loop above).
- **Stopping rule:** fixed epoch count (pinned in `prereg/recipe.md`), no early stopping — early stopping peeks at validation loss, which adds a tuning channel MU-6 exists to close.
- **Matching margin (the MU-5/TC-MU5-01 number):** the two models' one-step skill scores (judge-spec definition), per task and per region, must differ by **less than 0.05** — recorded in `prereg/recipe.md` before training; if any pair exceeds the margin, judging refuses to proceed (the pre-registration is "not yet satisfied", TC-MU5-01's second branch — the remedy is revising the *recipe* openly and re-running, never nudging a trained model).

**Options considered for Model B's spread:** raw `std(ddof=0)` — understates, exactly the artefact MU-5 forbids; `ddof=1` alone — better, still ignores mean-estimation variance; **`sqrt(1+1/K)`-inflated `ddof=1`** — accepted, standard, pre-registrable as one formula.

**Consequences:** The two models are identical everywhere except the uncertainty machinery — which is the entire experimental design. Both are deterministic under MU-8 (seeded init, seeded shuffle, single-threaded — TC-MU8-01's train-twice-identical check).

### ADR-M4 — Fixtures: three wrappers around one honest core

**Status:** Accepted. [Requirement: MU-3, MU-4] [Test: TC-MU3-01, TC-MU3-02, TC-MU3-03, TC-MU4-01]

**Context:** MU-3 specifies three failure modes; MU-4 demands the fixture label travel everywhere.

**Decision:** Each fixture wraps a copy of Model A (trained normally) and corrupts exactly one thing, so each failure mode is isolated and explainable. **"A copy of Model A", made buildable and identical (design-review-005 repair — Round 5 found the uniform factory gave a fixture no channel to obtain `direct`'s trained weights, so the only buildable reading retrained a fresh net, making each fixture differ from `direct` by two things, the corruption *and* an independent training run, contradicting "isolated"):** a fixture's factory calls the `direct` factory with the **same `(ctx, rng-derived-for-"direct", training)`** it would receive in a real run — the seed for `"direct"` is content-addressed (cross-cutting ADR-002 rule 2, keyed on the name string `"direct"`), so the fixture's inner network is bit-identical to the registered `direct` model's, and the fixture then applies its one corruption to that identical core. Weight identity between each fixture's inner model and `direct` is checkable (TC-MU4-02, new).

| Fixture | Corruption | What it proves the judge catches | Case |
|---|---|---|---|
| `fx-overconfident` | spread × 0.25, mean untouched | accurate but cocky → calibration fails despite good accuracy | TC-MU3-01 |
| `fx-honest-rough` | mean + seeded noise (σ = 2× the model's spread); spread widened by the exact formula `spread ← sqrt(base_spread² + noise_σ²)` (design-review fix — "widened to match the true resulting error" was prose; this closed form is exact because it's the true variance of `mean + independent Gaussian noise`, so the fixture is honest by construction rather than by an empirical post-hoc fit) | rougher but honest → calibration passes with worse accuracy | TC-MU3-02 |
| `fx-brittle` | **(design-review-002/003/004 fix — see below)** at each `predict(state, action)` call, checks whether `state` lies inside `ctx.training_state_box` **and** `action` lies inside `ctx.training_action_interval` (both halves of WD-5's own either-axis out-of-region rule, worlds spec ADR-W4 — design-review-004 repair: the v1.3 trigger checked the state box only, so an in-state/out-of-action trial would have been graded out-of-region by the harness but treated as in-region by the fixture); if both hold, returns Model A's unmodified prediction; if either fails, replaces the mean with `state` itself (a naive "nothing changes" prediction — badly wrong for a fast-moving or chaotic out-of-region trajectory) and leaves spread untouched | great at home, catastrophic away → only the in/out region split surfaces it | TC-MU3-03 |

**Design-review-003/004 fixes — `fx-brittle` redesigned, then re-wired.** Round 2's corruption ("half the training box") required a separate training run and misaligned with WD-5's world-level boundary; Round 3 made it a genuine post-hoc wrapper keyed to the world's declared box — but specified the box as "imported directly" from `wmj.worlds`, which Round 4 found contradicts this document's own §6 rule ("models never import world internals"), left the two-worlds selection question open, and collided with TC-MU9-01's isolation gate. **Round 4's repair (ADR-M1's uniform factory + `WorldContext`) resolves all of it structurally:** the box and action interval arrive through the same `ctx` argument every factory receives — no import, no per-world ambiguity (one instance per world, each holding its own context), no harness special-casing, and the trigger now covers both WD-5 axes, so the fixture's in/out boundary and the harness's region label are computed from the same constants delivered through one construction site. `fx-brittle` remains a pure wrapper: Model A trained exactly like every other contestant, no extra training-pipeline dependency. **Fixtures register via `wmj.models.registry.register()` exactly like every other model** (design-review-003 clarification).

All three set `is_fixture = True`; the flag flows through the harness into the verdict record and onto every chart (MU-4's three surfaces — code, record, rendered chart — are exactly TC-MU4-01's checklist; the rendering rule is the reporting spec's RP-8 section).

**Consequences:** All three fixtures are genuinely one small wrapper module each, with no dependency beyond an already-trained Model A and (for `fx-brittle`) the world's already-public region-box constant; their corruption parameters are constants in the spec (this table), not tunables.

### ADR-M5 — Pre-registration mechanics: committed files, harness-enforced git ordering

**Status:** Accepted. [Requirement: MU-6, JU-11 (mechanics shared)] [Test: TC-MU6-01, TC-MU6-02, TC-JU11-01, TC-JU11-02]

**Context:** MU-6 requires recipe, margins, formats, and the ranking prediction committed before the first judged run of unrigged models; the enforcement must be mechanical (commit timestamps), and the judge itself cannot read git (JU-12).

**Decision:** `prereg/` contains, each as one committed file: `recipe.md` (architecture including the ADR-M3 init/loss/batching/ε constants, training data counts, epochs, seeds policy, matching margin, ensemble rules, **and the MU-1 uncertainty format itself — per-dimension mean + one standard deviation, cross-referencing cross-cutting.md's data model** — design-review fix: MU-6's own text names the uncertainty format as one of the things that must be recorded before judging, but v1.0 left it implicitly satisfied elsewhere rather than traceably inside this file), `prediction.md` (the written ranking prediction), `thresholds.json` (JU-11's bands — owned by the judge spec, enforced by the same mechanism). The harness's `check_prereg` step (runs before any judged run of an unrigged model) asserts: every `prereg/` file is committed (not dirty), and its *first-added* commit timestamp strictly precedes the timestamp of any recorded judged-run artefact for unrigged models. Runs record their commit-of-record in the verdict metadata, making TC-JU11-02's after-the-fact edit detectable: a `prereg/` file whose first commit postdates a recorded run fails certification.

**How the harness knows a model is "unrigged" (design-review-005 repair — Round 5 found `check_prereg`'s trigger "before any judged run of an unrigged model" had no field to resolve, so MU-9's "adding a model touches nothing else" silently either skipped prereg for a new contestant or forced edits elsewhere):** a model is subject to pre-registration iff it is neither a pinned baseline (`"persistence"`, `"linear"`) nor a fixture (`is_fixture == True`). No `is_unrigged` field is added — the classification is derived from the two facts the harness already has (the pinned baseline names and the `is_fixture` flag), so it generalises to a newly-registered contestant with no edit to any classification list. **The honest scope of MU-9, stated:** adding a new *unrigged* contestant is the one case where "one file touches nothing else" does not fully hold — the model's *code* is one file, but making its pre-registration meaningful requires adding its recipe/prediction entry to `prereg/` and, if it is to be compared under MU-5's matching margin, naming it in `prereg/recipe.md`. This is a deliberate, disclosed tradeoff between modifiability (MU-9) and pre-registration integrity (MU-6): a contestant that could be added with *no* prereg entry would be a contestant that never had to declare itself in advance, which is the exact gaming MU-6 exists to prevent. Adding a *baseline* or a *fixture* remains genuinely one-file (neither passes through `check_prereg`).

**Consequences:** Pre-registration violations stop the pipeline before output exists (cross-cutting error conventions). MU-6's publish-either-way clause (TC-MU6-02, judged) is procedural, not code — the spec records it as an owner obligation. **A named residual risk (design-review addition):** "append-only" `prereg/` history is a convention this mechanism relies on, not a technical control — git history is rewritable by the same single author JU-10's disclosure #3 already discloses as a limitation. This is not a defect to fix (no technical control fully closes it without infrastructure this toy-scale project doesn't otherwise need) but is named here explicitly rather than left implicit, in the same spirit as JU-10's other disclosures.

**A second, distinct residual risk (design-review-002 addition — Round 2's Security panel):** MU-6's publish-either-way clause is procedural in a stronger sense than the paragraph above covers — nothing in this design detects *non-publication*. A judged run of the unrigged models could be executed, its result observed as unfavourable, and simply never committed; there is no artefact, log, or check anywhere in the pipeline that proves a judged run occurred if its output was withheld, since the only record of anything is what actually gets committed. This is a different failure mode from git-history rewriting (which alters a committed record) — this is a record that never gets made in the first place. No JU-10 disclosure covers this specific scenario (checked against all seven verbatim strings in judge spec ADR-J7). As with the git-rewritability risk, no technical control in a single-author, offline, no-server-side-witness project fully closes this; it is disclosed here rather than presented as solved.

**A third residual risk (design-review-003 addition; mitigation removed by design-review-004 repair):** a git commit's author/committer timestamp is ordinary metadata the committer supplies (`git commit --date=...`, or the `GIT_AUTHOR_DATE`/`GIT_COMMITTER_DATE` environment variables) — no history rewrite is needed to set a `prereg/` commit's *original* timestamp to any value at the moment it is first made, which is a simpler and easier-to-execute action than the rewritability risk disclosed above. Round 3 added a GitHub-API push-timestamp cross-check as a "partial mitigation"; Round 4 found it introduced an undisclosed network dependency contradicting the architecture's own "no external systems but git" claim (architecture-overview §1), specified no offline/failure behaviour for a pipeline whose target machine is an ordinary laptop, and — because GitHub's events API retains only a bounded recent window — would have silently decayed to no protection for any prereg commit older than roughly ninety days, undisclosed. **It is removed rather than patched.** Like the two risks above, this one has no technical control that a single-author, offline, no-server-side-witness project can honestly claim; a reader who wants stronger assurance can independently note the public repository's push time at the moment prereg lands (anyone watching the public repo can do this; the design just doesn't pretend the pipeline does it for them). Named, not solved — the same disclosure discipline as its siblings.

## 4. Component Design

```
wmj/models/
  base.py        # Model protocol, Prediction dataclass, WorldContext, TrainingData (frozen dataclasses)
  registry.py    # register(name, factory) / all_models() — the only discovery path (MU-9);
                 # full contract incl. DuplicateModelError and sorted-name ordering in ADR-M1
  baselines.py   # persistence, linear extrapolation (ADR-M2)
  mlp.py         # hand-rolled MLP: forward, backward, Adam, gradient check hook
  direct.py      # Model A (ADR-M3)
  ensemble.py    # Model B (ADR-M3)
  fixtures.py    # the three wrappers (ADR-M4)
tests/models/    # gradient check, format checks, fixture behaviour, determinism
```

**Training data (shape; final counts pinned in `prereg/recipe.md`):** for each world, N trajectories of full horizon from seeded training-region starts with seeded trained-range action sequences; the train/eval split is by *starting condition* — no evaluation rollout starts from any training start (MU-7, TC-MU7-01), with the harness storing both sets' start lists and asserting disjointness at run time. Resemblance between train and eval states is expected and fine (the systems cycle); only literal start reuse is forbidden, per MU-7's own wording.

## 5. API Boundary Contracts

What the harness extracts from models for the judge — the only model-shaped data the judge ever sees (JU-1). Arrays are pre-shaped `[n_trials, H, d]` (judge spec v1.5 §4); a trial's own row in that first axis is its boundary, so there is no separate boundary-marker field (design-review fix — v1.0 named a `trial_boundaries: [n_trials] int` field here that judge.md's own array shapes made undefined and redundant; removed):

```json
{
  "predictions": {"mean": "[n_trials, H, d] float64", "spread": "[n_trials, H, d] float64"},
  "outcomes": "[n_trials, H, d] float64",
  "region_labels": ["[n_trials] entries, one per trial — design-review-003 fix: this was previously shown as a single flat object, which cannot represent a batch mixing in- and out-of-region trials; the canonical per-trial shape (judge spec v1.3 §4) is used here verbatim, not restated independently",
    {"region_name": "training", "axis": null},
    {"region_name": "out-high-amplitude", "axis": "state"}
  ]
}
```

`is_fixture` and the model's `name` are *not* part of `JudgeInput` (the judge must stay blind) and are never computed by anything in this package's own model code — the harness assembles them into the harness-owned envelope (judge spec v1.5 §5) alongside the judge's pure `Verdict` once judging completes, which is what reporting actually consumes (design-review fix: v1.0 described this as the harness holding "a mapping" reporting separately "rejoins"; the envelope is the single object that replaces both, per Keeling's single-producer-per-fact principle applied in the judge spec).

## 6. Integration Points

- **← worlds:** training data generated through the world interface with the worlds spec's regions and action ranges; models never import world internals — they see `(state, action)` arrays, **plus the harness-constructed `WorldContext` (world geometry: boxes, action interval, `scale`) and `TrainingData` (the seeded trajectories) every factory receives uniformly (ADR-M1, design-review-004/005 — these two dataclasses are the only sanctioned channel for world facts and training data to reach a model, data handed in, never an import reaching out; the no-import direction is gate-enforced by TC-NF6-07)**.
- **→ judge:** the boundary contract above; the judge spec owns `JudgeInput`.
- **→ harness:** registry discovery, training orchestration, prereg checks, train/eval disjointness assertion.

## 7. Error Handling & Edge Cases

- A model returning a malformed `Prediction` (wrong shape, non-positive spread, NaN) → `ModelContractError` naming the model and MU-1; the run aborts (no partial verdicts, TC-JU9-02 discipline).
- Gradient check failure → training refuses to start.
- Prereg violations → pipeline stops before judging (ADR-M5).
- First-step linear extrapolation (no previous state) falls back to persistence — deterministic and documented, not an error.

## 8. Testing Strategy

| Concern | Cases |
|---|---|
| Uncertainty format fixed and identical | TC-MU1-01, TC-MU1-02 |
| `reset()` isolates rollouts — no cross-rollout state leak (design-review addition) | TC-MU1-03 (new) |
| Baselines present and non-trivial | TC-MU2-01, TC-MU2-02 |
| Fixtures fail as specified | TC-MU3-01, TC-MU3-02, TC-MU3-03 |
| Fixture labels on all three surfaces | TC-MU4-01 |
| Each fixture's inner net is weight-identical to `direct` (design-review-005) | TC-MU4-02 (new) |
| Matching margin + pre-registered ensemble rules | TC-MU5-01, TC-MU5-02, TC-MU5-03 |
| Prereg commit ordering, publish-either-way | TC-MU6-01, TC-MU6-02 (judged) |
| Train/eval disjointness | TC-MU7-01 |
| Deterministic training | TC-MU8-01 |
| Registry-only extensibility | TC-MU9-01 |
| No `models → worlds` import (design-review-005) | TC-NF6-07 (cross-cutting ADR-003) |
| Plus: MLP gradient check (this spec's own addition, pre-TC discipline) | — |

---

## Changelog

| Version | Date | Change |
|---|---|---|
| 1.0 | 2026-08-25 | Initial version. Interface (reset+predict), baselines with honest spreads, direct-vs-ensemble design with pre-registered spread mapping, three one-corruption fixtures, prereg mechanics. Matching margin pinned at 0.05 skill-score difference. |
| 1.1 | 2026-08-25 | Design-review fixes (design-review-001): pinned weight initialization, Model A's exact NLL formula, resolved the full-batch/mini-batch contradiction (mini-batch 32, gradient check separately full-batch), pinned Adam's ε; `fx-honest-rough`'s spread now an exact formula; `fx-brittle`'s "half the training box" now names the specific axis and split; added TC-MU1-03 (reset-isolation test); §5 dropped the undefined `trial_boundaries` field and now points at the judge spec's harness-owned envelope; ADR-M5 traces the MU-1 format explicitly into `prereg/recipe.md`. |
| 1.2 | 2026-08-30 | Design-review fix (design-review-002, Round 2): named a second residual risk in ADR-M5 — non-publication of an unfavourable judged run is undetectable by any mechanism in this design, distinct from the existing git-rewritability disclosure. |
| 1.3 | 2026-08-30 | Design-review fixes (design-review-003, Round 3, dual/blind): redesigned `fx-brittle` as a post-hoc wrapper keyed to the world's own declared region box (fixing a wrapper-framing contradiction and a region-label mismatch); stated fixtures' registration path explicitly; fixed `region_labels`' example to the canonical shape; named a third residual risk (prereg timestamp forgery) with a partial mitigation. |
| 1.4 | 2026-08-30 | Design-review-004 structural repair: uniform factory signature + `WorldContext` (ADR-M1) closing the models↔worlds layering contradiction, the two-worlds gap, and the TC-MU9-01 collision in one decision; registry contract pinned (named registration, duplicate error, sorted-order return); seven canonical names pinned; roster corrected 8→7; `fx-brittle` trigger covers both WD-5 axes; GitHub-API prereg mitigation removed in favour of plain disclosure. |
| 1.5 | 2026-08-31 | Design-review-005 repair: `WorldContext` gains `scale`; factory signature gains `TrainingData` (closing the no-training-channel gap that left 6 of 7 models unbuildable); fixtures build a bit-identical `direct` core via content-addressed seeding; auto-discovery arbitrary-code-on-import disclosed with an out-of-scope threat model; MU-9's "one file" honestly scoped for new unrigged contestants (derived baseline/fixture classification, no `is_unrigged` field); `training_action_interval` shape → `[a, 2]`; added TC-MU4-02 (fixture weight identity) and the TC-NF6-07 reference (models→worlds import gate). |

---

*Developed using the Grounded Vibe Methodology*
