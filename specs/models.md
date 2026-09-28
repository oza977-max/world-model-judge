# World Model Judge — Models Specification

Version 2.0 · 28 September 2026 · Domain 2 of Requirements v1.4 · References: cross-cutting spec v2.0, worlds spec v1.6

> **Change note (v2.0 — design-review-010, owner-triaged 2026-09-28).** The training recipe revision, with every value pinned (authoritative detail: `build/spec-corrections-backlog.md`, "Disposition (Round 10)"). **(1) Batched prediction** (ADR-M1): stateless models add `predict_batch`, row-for-row bit-identical to `predict` (TC-MU1-04), through a batch-size-invariant MLP forward — measured: NumPy's BLAS multiply changes the last bit for odd batch sizes, `einsum` does not. **(2) Training data built once by the harness** (ADR-M1 `TrainingData`, §4): 2,000 kicked trajectories, then a seeded training set of M = 50,000 pairs with exactly 12,500 kick pairs, a 10,000-pair held-out set and a 64-pair gradient-check batch, handed identically to every factory. **(3) Model A trains with β-NLL, β = 0.5** (ADR-M3), its gradient written out (TC-MU5-04). **(4) Batch size 256**, and the gradient check runs on 64 pairs, not the full set (ADR-M3; A10's floor and 1e-5 tolerance ratified). **(5) "Is M enough?" is decided by a pre-registered, outcome-independent test** (ADR-M3) — never by the MU-5 margin. **(6) The freeze** (ADR-M5): pre-registration is certified against the one commit that added `prereg/FREEZE` — one freeze, ever — replacing each file's first-added commit; A13's residual and the no-early-evaluation rule are disclosed. **In plain words:** the two practice models now have a recipe a builder can follow without guessing, that runs in minutes, and whose lock cannot be quietly moved.

> **Change note (v1.8 — design-review-008 repair, execution-verified).** Three fixes Round 8 found: (1) **`check_prereg`'s first-added-commit resolution is pinned to `--reverse --diff-filter=A`, first line** (ADR-M5) — Round 8 executed the gap: an unpinned, newest-first `git log` order resolves a delete-then-honestly-re-add to the *re-add* commit, silently passing the exact gaming scenario the content-hash check exists to catch; (2) **MU-5's margin-generalization overclaim corrected** — ADR-M5 had implied naming a third unrigged model in `prereg/recipe.md` brings it under MU-5's margin check; no mechanism does, and MU-5/ADR-M3's margin is a fixed pairwise comparison between the two named models by design, not a generic N-model rule; (3) **`seeds.rng("member", k, ...)`'s call sites now pass `str(k)` explicitly** — `component_key` no longer coerces non-`str` parts itself (cross-cutting ADR-002 rule 2, TC-NF1-08), closing a type-representation collision the caller must now avoid by contract. See design-review-008.html.

> **Change note (v1.7 — design-review-007 repair, execution-verified).** Three model-side gaps Round 7 found: (1) **`check_prereg` now enforces content-invariance** — it hashes each `prereg/` file's blob at its first-added commit and refuses if the working-tree content differs (ADR-M5, TC-MU6-04), closing the gap Round 7 executed where an ordinary honestly-dated second commit tuning `matching_margin` after seeing results passed the order-only check; (2) **`is_baseline`'s scope is stated honestly** — its only consumer is prereg-exemption; a new `is_baseline` model is judged like any contestant and does *not* become a third comparator (the Verdict's `skill` block names exactly `vs_persistence`/`vs_linear`), TC-MU6-05; (3) **`model_ref`'s cross-roster shift is disclosed** in ADR-M1 (a sorted-position index shifts when an earlier-sorting model is added; use `model_name` for cross-run joins). The `models→harness` no-import direction is now gate-enforced (TC-NF6-08, cross-cutting). See design-review-007.html.

> **Change note (v1.6 — design-review-006 repair, execution-verified).** Round 6 (dual/blind, executed) found the v1.5 model-provisioning fixes buildable for the four non-fixture models but with real holes, each now closed and the fix *run* before writing it: (1) the factory's bare `rng` becomes a **`SeedSource`** (defined in `wmj.models.base`, cross-cutting ADR-002 rule 2) — this gives a **fixture the sanctioned channel to rebuild `direct` bit-identically** via `seeds.rng_for("direct", …)` with no `models→harness` import (executed: `C7 fixture rebuild == direct weights: True`), and pins the **ensemble's K=5 per-member split** (`seeds.rng("member", k, …)`, executed distinct+reproducible); (2) a structural **`is_baseline` flag** replaces name-based baseline classification, so "adding a baseline is one file" (ADR-M5) is actually true; (3) `check_prereg` now verifies the **specific unrigged model's own `prereg/` entry** exists, not just that the files exist (TC-MU6-03); (4) the training count is **pinned (`N_train = 2000`)** with a machine-readable recipe format. See design-review-006.html.

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

**Nix & Weigend** — "Estimating the mean and variance of the target probability distribution", *IEEE ICNN 1994*. The original formulation of a network with a second output head predicting its own error variance, trained under Gaussian negative log-likelihood — Model A's method in its original form. Model A now trains with the β-NLL refinement of the same loss (ADR-M3, design-review-010); it is still a network predicting its own error bar.

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
  is_baseline: bool         # structural marker (design-review-006 repair): a new baseline
                            #   declares itself with this flag, exactly as a fixture does with
                            #   is_fixture — so classification is flag-based, not name-based, and
                            #   "adding a baseline is one file" (ADR-M5) is actually true. The
                            #   pinned {"persistence","linear"} names stay for MU-2 extraction
                            #   convenience only, not as the classification mechanism.
                            #   SCOPE (design-review-007 I5): the flag's ONLY consumer is
                            #   prereg-exemption (check_prereg skips is_baseline/is_fixture). A
                            #   model with is_baseline=True is still trained (if it trains) and
                            #   judged like any contestant; it does NOT become a reference the
                            #   others are scored against. The Verdict's skill block names exactly
                            #   two comparators (vs_persistence, vs_linear, judge §5), and the
                            #   blind judge cannot add a third. "Adding a baseline is one file" is
                            #   true for classification/provisioning, not for the MU-2 comparison
                            #   set — MU-2's two reference baselines are fixed. TC-MU6-05 (new)
                            #   asserts a new is_baseline model is prereg-exempt but not a comparator.
  reset() -> None           # called by the harness at the start of every rollout
  predict(state, action) -> Prediction   # mean: float64[d], spread: float64[d] (1 sd)
  stateless: bool           # design-review-010: True iff predict() depends only on (state, action) —
                            #   no rollout-local memory. persistence, direct, ensemble and all four
                            #   fixtures are stateless; linear (keeps the previous state) is not.
  predict_batch(states: float64[n, d], actions: float64[n, a]) -> (means: float64[n, d], spreads: float64[n, d])
                            # REQUIRED iff stateless (design-review-010, R10-P2). Row i must be
                            #   bit-identical to predict(states[i], actions[i]) — TC-MU1-04, every
                            #   registered stateless model, n in {1, 2, 7, 64, 200}. The harness rolls
                            #   all N trials of a region forward together through it (one call per step);
                            #   stateful models are rolled out per trial with reset() as before.
```

**Why batched prediction, and why it needs care (design-review-010, measured 2026-09-28).** One-at-a-time network calls cost ~11–14 µs each; a full evaluation makes ~34 million of them (three regions per world) — about six to eight minutes on its own, which ADR-J6's old FLOP-count estimate had called "well under a minute". Batched across the 200 trials the same work costs well under a minute. But "same work" must mean *same bits*: NumPy's BLAS matrix multiply is not batch-size invariant here (odd batch sizes differed from one-at-a-time in the last bit, ~2e-16; even sizes matched), while `np.einsum('bi,io->bo', A, W, optimize=False)` was row-invariant at every size tested at ~1.7× the cost. So `wmj.models.mlp` gains **`forward_invariant(X)`** — per layer `einsum` + bias, tanh hidden, linear output — and every MLP-backed `predict` and `predict_batch` uses it; training keeps the BLAS `forward`. The trained weights are the same either way; only the prediction arithmetic is pinned. TC-MU1-04 is the guard: the invariance is tested on every run of the suite, not assumed from one machine.

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
  factory(ctx: WorldContext, seeds: SeedSource, training: TrainingData) -> Model
      # design-review-006 repair: the bare `rng` argument is replaced by SeedSource
      # (cross-cutting ADR-002 rule 2, defined in wmj.models.base), which the harness
      # binds to this model's name and hands in as DATA — no models→harness import.
      # A model draws its own streams by purpose: seeds.rng("weights"), seeds.rng("shuffle", epoch),
      # seeds.rng("member", k) for ensemble member k. A FIXTURE rebuilds `direct` bit-identically
      # by drawing direct's own streams via seeds.rng_for("direct", "weights") etc. — this is the
      # sanctioned channel Round 6 found missing (the fixture had no way to reach direct's seed).
      # Executed and confirmed: `C7 fixture rebuild == direct weights: True`.

TrainingData (frozen dataclass, wmj.models.base — design-review-005 repair; extended design-review-010):
  states:  float64[N, H+1, d]   # N seeded training trajectories for this world (models spec §4)
  actions: float64[N, H, a]     # the seeded kick sequences that produced them (worlds ADR-W2)
  train_pairs:   Pairs          # design-review-010 (R10-P5): the M-pair training set every MLP trains on
  heldout_pairs: Pairs          #   10,000 held-out pairs — sufficiency check + kick report ONLY
  gradcheck_index: int64[64]    #   indices into train_pairs for the one-time gradient check
Pairs (frozen dataclass, wmj.models.base):
  state: float64[m, d]; action: float64[m, a]; next_state: float64[m, d]; is_kick: bool[m]
      # constructed once per world by the harness and handed to every factory identically;
      # the harness owns generation AND subsampling (seeded, MU-7-disjoint) — the model only
      # fits to it. No factory draws its own subsample: that is what keeps the ensemble's five
      # members on "identical training data" (ADR-M3) and a fixture's rebuilt core bit-identical
      # to direct (ADR-M4, TC-MU4-02).
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
      # WITHIN ONE FIXED ROSTER. Disclosed limit (design-review-007 I3): because
      # model_ref is a sorted-position index, adding a model that sorts earlier shifts
      # every later model's model_ref across roster versions. On-disk verdict files are
      # keyed by model NAME (out/verdicts/{world}-{model}.json), so the artefact set is
      # not corrupted, but a model_ref value must never be joined across two runs with
      # different rosters — use model_name for any cross-run join (see cross-cutting
      # Data-Model Overview, JudgedResult row).
```

**The seven canonical model names, pinned as literal join keys** (the same discipline worlds.md applies to region names — Round 4 found the harness must pick out specific models by name for MU-2's baseline extraction and MU-5's unrigged-pair check, with no pinned vocabulary to match against): `"persistence"`, `"linear"`, `"direct"`, `"ensemble"`, `"fx-overconfident"`, `"fx-honest-rough"`, `"fx-brittle"`. **The roster is seven contestants** — design-review-004 repair: "eight registered contestants"/"8 models" appeared in three specs and was never true against this document's own enumeration (2 baselines + direct + ensemble + 3 fixtures = 7), the same never-recounted-number defect class as the chunk-count error two rounds earlier.

**Auto-discovery runs a discovered module's top-level code on import — threat model, stated (design-review-005 repair — Round 5 Security noted this path executes arbitrary module code with no disclosure, unlike ADR-M5's prereg risks):** `all_models()` imports every file under `wmj/models/`, so any file placed there executes on the next call. **The adversary is explicitly out of scope:** this is a single-author project whose author writes every model file; the property auto-discovery protects is *extensibility* (MU-9), not *isolation from hostile code*. Anyone who can write a file into `wmj/models/` already controls the whole checkout. This is a convenience mechanism, not a security boundary — named here so no reader mistakes the AST gates (which harden the *judge's* purity, a different concern) for a sandbox around model code. The same disclosure discipline as ADR-M5's residual risks.

**Consequences:** Rollout-local state is invisible to the judge (which sees only arrays) and harmless to determinism (reset() makes every rollout self-contained) — **provided `reset()` actually clears it (design-review fix — this was previously asserted but never tested)**: §8 adds a cross-rollout isolation test, since a model that silently leaked state across JU-8's 200 independent trials would corrupt the independence assumption the whole exception-band derivation depends on, undetected by any other test in this document. TC-MU1-01/02 check the format and its cross-model identity; the registry is the only discovery path (MU-9), and harness code referencing the pinned name *strings* above (data, not Python identifiers) is explicitly permitted and does not violate MU-9's isolation — TC-MU9-01's import-graph check (cross-cutting ADR-003) forbids importing model *submodules*, not naming models by their registered string keys.

### ADR-M2 — Baselines: persistence and linear extrapolation, with honest training-residual spreads

**Status:** Accepted. [Requirement: MU-2, MU-1] [Test: TC-MU2-01, TC-MU2-02, TC-MU2-03]

**Context:** MU-2 requires both baselines in every verdict. MU-1 requires *every* model under test to state uncertainty in the fixed format — baselines included, or the judge would need a special case, which JU-1's blindness forbids.

**Decision:**
- **Persistence:** mean = current state; spread = per-dimension standard deviation of one-step state *changes* over the training dataset (a constant vector, computed once at fit time). "Nothing changes, give or take how much things usually change."
- **Linear extrapolation:** mean = current + (current − previous) for the same dimension-wise step (first step of a rollout falls back to persistence, having no previous); spread = per-dimension standard deviation of that rule's own residuals on the training data.
- Both are fitted (their spread constants) on exactly the training dataset the learned models use, so their calibration is honest rather than decorative.
- **`ddof=1` (sample standard deviation), pinned for both fits (corrected design-review-009 A8/C1):** the training set is a sample of the world, not its complete population, so the unbiased (Bessel-corrected) estimator is the statistically defensible one — the same convention ADR-M3 already pins for the ensemble spread below. This was left unpinned through Phase 2 and an interim build applied it to persistence only, silently leaving linear on the population default (`ddof=0`) with no guard — a live inconsistency between two baselines under one ADR, caught by design-review-009 before pre-registration. `ddof` changes the emitted spread's bytes and therefore every downstream CRPS, skill score, and Verdict byte (NF-1), so it cannot be a renderer's-discretion detail: both baselines' fits go through one shared, guarded implementation.
- **Fail loudly on a degenerate fit:** a dimension whose training changes/residuals have zero variance, or too few samples to estimate a sample standard deviation from (`<2`), would otherwise emit a zero-width or non-finite spread the CRPS rightly refuses (judge ADR-J1) — the fit refuses first, instead (`DegenerateSpreadError`), rather than the judge failing later with a less legible error (TC-MU2-03).

**Consequences:** Baselines participate in calibration and exception counting like everyone else — a learned model must beat persistence not only on error (TC-MU2-02's sanity floor) but visibly on the same charts. The judge refuses to run without both baselines present in its input (TC-MU2-01, cross-cutting error conventions).

### ADR-M3 — The two unrigged models: same network, two uncertainty methods

**Status:** Accepted. [Requirement: MU-5, MU-6, MU-8] [Test: TC-MU5-01, TC-MU5-02, TC-MU5-03, TC-MU8-01]

**Context:** MU-5 demands at least two non-engineered models, matched on accuracy to a pre-fixed margin, differing only in how they derive uncertainty: one predicting its own error bar, one small ensemble whose disagreement supplies it.

**Decision:**
- **Model A — "direct" (Nix & Weigend):** one MLP taking `(state, action)` — state normalised by `ctx.scale`, action by its `ctx.training_action_interval` half-width (design-review-005: both arrive through `WorldContext`, the one sanctioned channel; no `wmj.worlds` import) — outputting per-dimension `(Δmean, log σ)` — it predicts the state *change* plus its own per-dimension error bar. Trained with **β-NLL, β = 0.5** (Seitzer et al., ICLR 2022 — design-review-010 R10-P8: plain Gaussian NLL down-weights the squared error where the predicted variance is high, degrading the mean fit, which would leave the result open to "you picked a known-weak Model A"; β-NLL is still a self-predicted-error-bar model, so MU-5's contrast is unchanged). `mean = state + Δmean`, `spread = exp(log σ)`.
- **Model B — "ensemble" (Lakshminarayanan et al.):** **K = 5** MLPs with the same architecture minus the variance head (mean output only), identical training data, differing only in seeded initialisation and seeded batch shuffling. **Each member's streams are content-addressed by member index (design-review-006 repair — Round 6 found the single `rng` gave the K=5 members no specified split, so two builds diverged in Model B's weights): member `k` initialises from `seeds.rng("member", str(k), "weights")` and shuffles from `seeds.rng("member", str(k), "shuffle", str(epoch))` (design-review-008 I5 — `component_key` now rejects non-`str` parts, so every caller passes its own `str()` form explicitly, per cross-cutting ADR-002 rule 2). Executed: the five member streams are distinct and reproducible (`ensemble K=5 members distinct: True | reproducible: True`).** Pre-registered rules (the MU-5 clauses, fixed here and repeated in `prereg/recipe.md`):
  - **Point prediction:** the arithmetic mean of the K member means.
  - **Spread mapping:** per-dimension `spread = sqrt(1 + 1/K) × std(member_means, ddof=1)` — sample standard deviation with Bessel's correction, inflated by the standard `sqrt(1 + 1/K)` predictive-variance factor to correct small-ensemble underdispersion (the MU-5 correction clause; TC-MU5-03 checks it is applied).
- **Shared architecture (both models, both worlds):** 2 hidden layers × 64 units, tanh activations, hand-rolled NumPy forward/backward.
  - **Weight initialization (design-review fix — previously unspecified, and MU-8's seeded-reproducibility guarantee only holds if the init scheme is itself fixed):** each weight matrix `W ~ U(−1/√fan_in, 1/√fan_in)` (uniform, fan-in scaled — the standard scheme for tanh networks), all biases initialised to 0, drawn from `seeds.rng("weights")` for `direct` (and `seeds.rng("member", str(k), "weights")` for ensemble member k), cross-cutting ADR-002 rule 2.
  - **Model A's loss, written out (design-review fix; replaced design-review-010 R10-P8 — the network has no automatic differentiation, so the gradient is written out too):** per dimension, per example, with `r = y − μ`, `s = log σ` and weight `w = σ^(2β)` with **β = 0.5** (so `w = σ`), `w` computed from the current forward pass and **held constant** (the "stop-gradient"): `L = w·[0.5·log(2π) + s + r²/(2σ²)]`, summed over the d dimensions and averaged over the batch. The `d_output` handed to backprop is `∂L/∂μ = −w·r/σ²` and `∂L/∂s = w·(1 − r²/σ²)`, each divided by the batch size. TC-MU5-04 (new) checks this closure against central finite differences taken over the network *outputs* with `w` held fixed. The ensemble members train on mean-squared error of the state change.
  - **Training loop shape (design-review fix — v1.0 said "full-batch gradient checked... before any training run" and, separately, that Model B's members differ in "seeded batch shuffling," which only makes sense for mini-batches; the two statements were never reconciled):** training itself is **mini-batch**, batch size **256** (design-review-010: was 32; measured 2.84 µs vs 7.05 µs per example, and pinned as `batch_size: 256` in `prereg/recipe.md`), examples shuffled each epoch from the run's seeded `Generator`, over `training.train_pairs` (the M-pair set the harness built). The *gradient check* (below) is a separate, one-time, full-batch finite-difference check run once before training starts — it is not the training loop itself, and "batch shuffling" refers only to the mini-batch training loop's own epoch-order shuffle.
  - Adam optimizer: lr 1e-3, β₁ 0.9, β₂ 0.999, **ε = 1e-8** (design-review fix — previously omitted; two implementers picking different conventional defaults would get bit-different trained weights under this project's own determinism standard).
  - **Gradient check (revised design-review-010 R10-P4; A10 ratified):** run once before each training run on the fixed **64-pair** batch `training.gradcheck_index` (the "one full-batch pass" wording is retired — measured, a finite-difference check over 50,000 examples would take ≈7–10 minutes, over the whole run budget; 64 pairs take well under a second). It validates **backprop**, not the loss, so `direct`'s network is checked under plain Gaussian NLL and each ensemble member's under mean-squared error (finite differences through the weights of β-NLL would also differentiate the held-constant `w`). Metric: relative error with its denominator floored at `GRADIENT_SCALE_FLOOR = 1e-3` × the largest gradient magnitude; **tolerance 1e-5** (the earlier 1e-6 tripped on float64 finite-difference noise in the real 2×64 net; 1e-5 still sits four orders of magnitude below a real backprop bug). Disclosed blind spot: a bug confined to a near-dead unit (gradient below 0.1% of the peak) is judged by absolute agreement and can pass. (Beck: the failing test first — the gradient check is the MLP's first test; it is not part of the mini-batch training loop above.)
  - **Is M enough? — pre-registered, outcome-independent (design-review-010 R10-P10; Round 10 found "fit vs the MU-5 margin" would have tuned a data setting against the very comparison it must stay independent of).** Checked once at build time (P3-C03 for `direct`, P3-C04 for `ensemble`), before the freeze, never in a judged run and never against the MU-5 margin: train at M = 50,000 and at 2M = 100,000 (the same 12,500 kick pairs plus the first 87,500 non-kick pairs of the same permutation; everything else identical) and compare held-out error — mean over `heldout_pairs` of mean over dimensions of `((mean − y)/scale)²`. M is sufficient iff `err(M) ≤ 1.10 × err(2M)` for **both** models; if either fails, `subsample_pairs` becomes **100,000** for both (the one pre-registered fallback, recorded in the recipe's revision log), with no further iteration. The same build step reports each model's held-out one-step error split by kick and non-kick pairs, because training pairs are 25% kicks while evaluation steps are ~1% (LV) / ~0.2% (pendulum) kicks — a shift both models share, disclosed rather than hidden.
- **Stopping rule:** fixed epoch count (pinned in `prereg/recipe.md`), no early stopping — early stopping peeks at validation loss, which adds a tuning channel MU-6 exists to close.
- **Matching margin (the MU-5/TC-MU5-01 number):** the two models' one-step skill scores (judge-spec definition), per task and per region, must differ by **less than 0.05** — recorded in `prereg/recipe.md` before training; if any pair exceeds the margin, judging refuses to proceed (the pre-registration is "not yet satisfied", TC-MU5-01's second branch — the remedy is revising the *recipe* openly and re-running, never nudging a trained model).

**Options considered for Model B's spread:** raw `std(ddof=0)` — understates, exactly the artefact MU-5 forbids; `ddof=1` alone — better, still ignores mean-estimation variance; **`sqrt(1+1/K)`-inflated `ddof=1`** — accepted, standard, pre-registrable as one formula.

**Consequences:** The two models are identical everywhere except the uncertainty machinery — which is the entire experimental design. Both are deterministic under MU-8 (seeded init, seeded shuffle, single-threaded — TC-MU8-01's train-twice-identical check).

### ADR-M4 — Fixtures: four wrappers around one honest core (three until design-review-009 B1 added `fx-action-blind`)

**Status:** Accepted. [Requirement: MU-3, MU-4] [Test: TC-MU3-01, TC-MU3-02, TC-MU3-03, TC-MU4-01]

**Context:** MU-3 specifies three failure modes; MU-4 demands the fixture label travel everywhere.

**Decision:** Each fixture wraps a copy of Model A (trained normally) and corrupts exactly one thing, so each failure mode is isolated and explainable. **"A copy of Model A", made buildable and identical (design-review-005 opened this; design-review-006 made it actually buildable — Round 6 found the v1.5 wording said the fixture "calls the direct factory with rng-derived-for-direct" but gave the fixture no sanctioned channel to that seed, since it lives in the harness and models cannot import the harness):** the fixture's factory builds its inner core by calling the `direct` construction with **`seeds.rng_for("direct", "weights")`** (and direct's other purposes), where `seeds: SeedSource` is the object handed to every factory (cross-cutting ADR-002 rule 2, defined in `wmj.models.base`, so no `models→harness` import is needed — the derivation function is pure and model-facing). Because `SeedSource` is content-addressed on the name `"direct"`, the fixture's inner network is **bit-identical** to the registered `direct` model's — executed and confirmed (`C7 fixture rebuild == direct weights: True`) — and the fixture then applies its one corruption to that identical core. Weight identity between each fixture's inner model and `direct` is checkable (TC-MU4-02).

| Fixture | Corruption | What it proves the judge catches | Case |
|---|---|---|---|
| `fx-overconfident` | spread × 0.25, mean untouched | accurate but cocky → calibration fails despite good accuracy | TC-MU3-01 |
| `fx-honest-rough` | mean + seeded noise (σ = 2× the model's spread); spread widened by the exact formula `spread ← sqrt(base_spread² + noise_σ²)` (design-review fix — "widened to match the true resulting error" was prose; this closed form is exact because it's the true variance of `mean + independent Gaussian noise`, so the fixture is honest by construction rather than by an empirical post-hoc fit) | rougher but honest → calibration passes with worse accuracy | TC-MU3-02 |
| `fx-brittle` | **(design-review-002/003/004 fix — see below)** at each `predict(state, action)` call, checks whether `state` lies inside `ctx.training_state_box` **and** `action` lies inside `ctx.training_action_interval` (both halves of WD-5's own either-axis out-of-region rule, worlds spec ADR-W4 — design-review-004 repair: the v1.3 trigger checked the state box only, so an in-state/out-of-action trial would have been graded out-of-region by the harness but treated as in-region by the fixture); if both hold, returns Model A's unmodified prediction; if either fails, replaces the mean with `state` itself (a naive "nothing changes" prediction — badly wrong for a fast-moving or chaotic out-of-region trajectory) and leaves spread untouched | great at home, catastrophic away → only the in/out region split surfaces it | TC-MU3-03 |

**Design-review-003/004 fixes — `fx-brittle` redesigned, then re-wired.** Round 2's corruption ("half the training box") required a separate training run and misaligned with WD-5's world-level boundary; Round 3 made it a genuine post-hoc wrapper keyed to the world's declared box — but specified the box as "imported directly" from `wmj.worlds`, which Round 4 found contradicts this document's own §6 rule ("models never import world internals"), left the two-worlds selection question open, and collided with TC-MU9-01's isolation gate. **Round 4's repair (ADR-M1's uniform factory + `WorldContext`) resolves all of it structurally:** the box and action interval arrive through the same `ctx` argument every factory receives — no import, no per-world ambiguity (one instance per world, each holding its own context), no harness special-casing, and the trigger now covers both WD-5 axes, so the fixture's in/out boundary and the harness's region label are computed from the same constants delivered through one construction site. `fx-brittle` remains a pure wrapper: Model A trained exactly like every other contestant, no extra training-pipeline dependency. **Fixtures register via `wmj.models.registry.register()` exactly like every other model** (design-review-003 clarification).

All four set `is_fixture = True` (`fx-action-blind`, MU-3's fourth failure mode, is specified with P3-C08 and TC-MU3-04); the flag flows through the harness into the verdict record and onto every chart (MU-4's three surfaces — code, record, rendered chart — are exactly TC-MU4-01's checklist; the rendering rule is the reporting spec's RP-8 section).

**Consequences:** All four fixtures are genuinely one small wrapper module each, with no dependency beyond an already-trained Model A and (for `fx-brittle`) the world's already-public region-box constant; their corruption parameters are constants in the spec (this table), not tunables.

### ADR-M5 — Pre-registration mechanics: committed files, harness-enforced git ordering

**Status:** Accepted. [Requirement: MU-6, JU-11 (mechanics shared)] [Test: TC-MU6-01, TC-MU6-02, TC-MU6-03, TC-JU11-01, TC-JU11-02]

**Context:** MU-6 requires recipe, margins, formats, and the ranking prediction committed before the first judged run of unrigged models; the enforcement must be mechanical (commit timestamps), and the judge itself cannot read git (JU-12).

**Decision:** `prereg/` contains, each as one committed file: `recipe.md` (architecture including the ADR-M3 init/loss/batching/ε constants, training data counts, epochs, seeds policy, matching margin, ensemble rules, **and the MU-1 uncertainty format itself — per-dimension mean + one standard deviation, cross-referencing cross-cutting.md's data model** — design-review fix: MU-6's own text names the uncertainty format as one of the things that must be recorded before judging, but v1.0 left it implicitly satisfied elsewhere rather than traceably inside this file), `prediction.md` (the written ranking prediction), `thresholds.json` (JU-11's bands — owned by the judge spec, enforced by the same mechanism). The harness's `check_prereg` step (runs before any judged run of an unrigged model) asserts: every `prereg/` file is committed (not dirty); its *first-added* commit timestamp strictly precedes the timestamp of any recorded judged-run artefact for unrigged models; **and — design-review-007 I8 — the file's *content* has not changed since that first-added commit: `check_prereg` hashes the blob at the first-added commit (`git show <first-add-sha>:prereg/<file>`) and asserts it equals the hash of the working-tree file being judged against. If they differ, certification fails. The first-added commit is resolved with `git log --reverse --diff-filter=A --format=%H -- prereg/<file>`, taking the FIRST line of that output (design-review-008 I6 repair — Round 8 executed the gap: a file that is deleted and later honestly re-added under the same path produces *two* `A` (added) events in its history; `git log`'s default order is newest-first, so a natural "take the first line" implementation without `--reverse` resolves to the *re-add* commit, whose content matches the working tree by construction, silently passing the exact gaming scenario this check exists to catch — `--reverse` pins the correct, oldest-first resolution, executed and confirmed against both the ordinary-edit and the delete-then-readd scenarios).** Runs record their commit-of-record in the verdict metadata, making TC-JU11-02's after-the-fact edit detectable: a `prereg/` file whose first commit postdates a recorded run fails certification. **Why the content check was added (executed, design-review-007):** Round 7 exhibited, with real git, that the order-only check is defeated by an ordinary honestly-dated second commit — a day-1 placeholder `recipe.md`, then a day-243 commit writing `matching_margin: 0.049 # tuned after seeing results`, clean working tree, no history rewrite, no `--amend`, all timestamps genuine — which passes "first-added precedes run" while the content in force at judging time was written after results were seen, exactly the MU-6 gaming ("thresholds moved after seeing results are not thresholds") the mechanism exists to prevent. The content-hash check turns that from an undisclosed gap into a hard failure (TC-MU6-04, new). This does not close the deliberate-history-rewrite risk disclosed below — a determined single author can still rewrite the first-add commit itself — but it closes the *un*rewritten, honestly-dated drift, which is the easier and more likely accident.

**How the harness knows a model is "unrigged" — flag-based classification (design-review-005 introduced the derived classification; design-review-006 fixed a contradiction Round 6 found in it):** a model is subject to pre-registration iff it is **neither `is_baseline` nor `is_fixture`**. Both are structural flags the Model instance carries (ADR-M1) — design-review-006 added `is_baseline` for exactly this: Round 6 found the v1.5 rule keyed "baseline" off the hardcoded name pair, so a genuinely new third baseline (one new file) would be misclassified as unrigged and blocked by `check_prereg`, directly contradicting the same ADR's "adding a baseline remains one-file" claim. With a flag, a new baseline self-declares `is_baseline=True` and the claim is true. **The honest scope of MU-9, stated:** adding a *baseline* or a *fixture* is genuinely one file (the flag self-classifies; neither passes through `check_prereg`). Adding a new *unrigged* contestant is the one disclosed exception — its code is one file, but making its pre-registration meaningful requires a `prereg/` entry; a contestant addable with *no* prereg entry is one that never declared itself in advance, the exact gaming MU-6 exists to prevent. **Naming a third unrigged model in `prereg/recipe.md` does NOT, by itself, bring it under MU-5's margin check (design-review-008 I3 correction — Round 8 found the previous wording here implied it would, but no mechanism does):** MU-5/ADR-M3's margin is a fixed pairwise comparison between the two named models, `direct` and `ensemble` — the project's own deliberately-scoped experiment (CLAUDE.md: "at least two unrigged models, matched on accuracy, differing only in how they derive uncertainty"), not a generic N-model rule. A third unrigged model is judged, and is prereg-checked for its own entry, but is not automatically compared against anything by MU-5's mechanism as pinned; extending the margin check to more than the two matched models would be a deliberate scope change to MU-5 itself, not a side effect of adding a `prereg/` entry.

**`check_prereg` verifies the specific new model's entry, not just that `prereg/` files exist (design-review-006 repair — Round 6 found the v1.5 check was roster-agnostic: it asserted the `prereg/` files as a whole were committed and pre-dated the run, so a new unrigged model passed as "pre-registered" without any text about it ever being written into `prereg/`).** For every model classified unrigged (`not is_baseline and not is_fixture`), `check_prereg` additionally asserts that model's `name` appears in the committed `prereg/recipe.md` and `prereg/prediction.md` (a named entry, its commit predating the run) — so a new unrigged contestant with no prereg entry of its own is refused, not silently blessed. This is what makes MU-6's "mechanically checkable" language true per-model, matching the WD-3/JU-11 enforcement pattern used elsewhere (TC-MU6-01 extended; TC-MU6-03 new — the per-model presence check).

**The freeze — superseding the per-file first-added anchor above (design-review-010 R10-P11, owner-triaged; the text above is kept as the history of why).** Round 10 found the first-added-commit anchor makes the spec's own remedy ("revise the recipe openly and re-run") fail certification forever, and that the first replacement proposed — a declared freeze commit named in a file plus a git tag — could be moved silently. The mechanism now:
- `prereg/FREEZE` is a short human-readable declaration (date, a sentence); it contains **no** commit id (a commit cannot name itself).
- **The freeze commit is the commit that added `prereg/FREEZE`**, read from git history, never chosen or typed. `check_prereg` refuses unless `git log --diff-filter=A --format=%H -- prereg/FREEZE` prints **exactly one line** and `git log --diff-filter=D --format=%H -- prereg/FREEZE` prints **none** — **one freeze, ever**; a second freeze, or a delete-and-re-add, is refused (TC-MU6-06). No `FREEZE` → refused as "not frozen yet" (TC-MU6-08).
- Every certified file — `recipe.md`, `prediction.md`, `thresholds.json` and `FREEZE` itself — must be clean in the working tree and byte-equal to its blob **at the freeze commit** (TC-MU6-04, rewritten); the freeze commit's timestamp must strictly precede the run (TC-MU6-01, rewritten).
- `check_prereg` returns the **freeze commit SHA**; the harness records it as `meta.prereg_commit` (P6-C01 — TC-MU6-07). The mechanism is built at P3-C10, before P6-C01.
- The git tag `prereg-freeze` is an optional human label; `check_prereg` never reads it.
- Before the freeze, `recipe.md` and `prediction.md` may be revised openly, each revision dated in the recipe's revision log; earlier versions stay in history.
- The recipe's kick and action-interval keys must equal the world modules' constants (TC-MU6-09) — the frozen recipe pins them, and code cannot drift away silently.

**Consequences:** Pre-registration violations stop the pipeline before output exists (cross-cutting error conventions). MU-6's publish-either-way clause (TC-MU6-02, judged) is procedural, not code — the spec records it as an owner obligation. **A named residual risk (design-review addition):** "append-only" `prereg/` history is a convention this mechanism relies on, not a technical control — git history is rewritable by the same single author JU-10's disclosure #3 already discloses as a limitation. This is not a defect to fix (no technical control fully closes it without infrastructure this toy-scale project doesn't otherwise need) but is named here explicitly rather than left implicit, in the same spirit as JU-10's other disclosures.

**A second, distinct residual risk (design-review-002 addition — Round 2's Security panel):** MU-6's publish-either-way clause is procedural in a stronger sense than the paragraph above covers — nothing in this design detects *non-publication*. A judged run of the unrigged models could be executed, its result observed as unfavourable, and simply never committed; there is no artefact, log, or check anywhere in the pipeline that proves a judged run occurred if its output was withheld, since the only record of anything is what actually gets committed. This is a different failure mode from git-history rewriting (which alters a committed record) — this is a record that never gets made in the first place. No JU-10 disclosure covers this specific scenario (checked against all seven verbatim strings in judge spec ADR-J7). As with the git-rewritability risk, no technical control in a single-author, offline, no-server-side-witness project fully closes this; it is disclosed here rather than presented as solved.

**A third residual risk (design-review-003 addition; mitigation removed by design-review-004 repair):** a git commit's author/committer timestamp is ordinary metadata the committer supplies (`git commit --date=...`, or the `GIT_AUTHOR_DATE`/`GIT_COMMITTER_DATE` environment variables) — no history rewrite is needed to set a `prereg/` commit's *original* timestamp to any value at the moment it is first made, which is a simpler and easier-to-execute action than the rewritability risk disclosed above. Round 3 added a GitHub-API push-timestamp cross-check as a "partial mitigation"; Round 4 found it introduced an undisclosed network dependency contradicting the architecture's own "no external systems but git" claim (architecture-overview §1), specified no offline/failure behaviour for a pipeline whose target machine is an ordinary laptop, and — because GitHub's events API retains only a bounded recent window — would have silently decayed to no protection for any prereg commit older than roughly ninety days, undisclosed. **It is removed rather than patched.** Like the two risks above, this one has no technical control that a single-author, offline, no-server-side-witness project can honestly claim; a reader who wants stronger assurance can independently note the public repository's push time at the moment prereg lands (anyone watching the public repo can do this; the design just doesn't pretend the pipeline does it for them). Named, not solved — the same disclosure discipline as its siblings.

**A fourth residual risk (design-review-010, ratifying backlog A13):** `check_prereg`'s per-model entry check confirms that a model's *name* appears as a whole token in the committed recipe and prediction; it cannot confirm that the surrounding text is a genuine recipe or prediction for that model — a hollow mention passes. Checking substance would need an understanding of prose that no mechanism has. Disclosed, not "fixed" with a fake check.

**A fifth residual risk (design-review-010, R10-P11 — Round 10's blind panel found re-freezing *before* a judged run was not refused by the first proposal):** the rule is that **no evaluation-trial metric of either unrigged model is computed before the freeze**. One freeze, ever, stops the lock being moved afterwards; it cannot stop an unrecorded informal run *before* the freeze from influencing the recipe, because such a run leaves no trace — the same family as the non-publication residual above. The build-time sufficiency check (ADR-M3) is designed to stay on the right side of this line: it uses held-out training-region pairs only, never evaluation trials, never the margin.

## 4. Component Design

```
wmj/models/
  base.py        # Model protocol, Prediction, WorldContext, TrainingData, SeedSource + component_key
                 #   (all frozen/pure — SeedSource/component_key moved here design-review-006 so fixtures
                 #    can derive direct's stream without importing the harness)
  registry.py    # register(name, factory) / all_models() — the only discovery path (MU-9);
                 # full contract incl. DuplicateModelError and sorted-name ordering in ADR-M1
  baselines.py   # persistence, linear extrapolation (ADR-M2)
  mlp.py         # hand-rolled MLP: forward, forward_invariant (prediction path, design-review-010), backward, Adam, gradient check
  direct.py      # Model A (ADR-M3)
  ensemble.py    # Model B (ADR-M3)
  fixtures.py    # the fixture wrappers (ADR-M4; four since design-review-009 B1)
tests/models/    # gradient check, format checks, fixture behaviour, determinism
```

**Training data (design-review-006 repair — Round 6 found the training count `N` had no value, formula, or machine-readable source anywhere, unlike the evaluation N=200 which ADR-J4 pins with a power rationale):** the training count is **`N_train = 2000` trajectories per world** (pinned here, and mirrored in `prereg/recipe.md` — a round figure ample for a 2-hidden-layer 64-unit MLP on a 2–4-dimensional system, well inside the NF-2 budget; the number is a modelling choice, fixed in advance, not derived, and stated as such). `prereg/recipe.md` records it as a machine-readable key-value line (`training_trajectories: 2000`, `epochs: 100` — design-review-010 ratified A12's value) that `check_prereg`/the training chunks parse — the parse format is pinned so P3-C06/C03/C04 have a defined source, not free prose. Each world draws `N_train` trajectories of full horizon from seeded training-region starts (`seeds.rng_for(world, "training", "train-starts")`, cross-cutting ADR-002 rule 2) with seeded kick sequences (worlds ADR-W2; trajectory `i` from `seeds.rng_for(world, "training", "train-kicks", str(i))`, band `"in"`), generated with the batched world step (worlds ADR-W1). **Subsampling (design-review-010, R10-P5 — the harness, never a factory):** a *pair* is `(state_t, action_t, state_{t+1})`; a *kick pair* has `action_t ≠ 0`. The training set is exactly **12,500 kick pairs** (the first of a permutation of all kick-pair indices drawn with `seeds.rng_for(world, "training", "subsample-kick")`; fewer available → `TrainingDataError`, never a silent shrink; expected available: LV ≈ 14,000, pendulum ≈ 20,000) plus **M − 12,500 non-kick pairs** (first of a permutation drawn with `"subsample-nonkick"`), M = `subsample_pairs` (50,000; 100,000 only via ADR-M3's fallback). The held-out set is 10,000 pairs from those *not* selected (first of a permutation drawn with `"heldout"`), and the gradient-check batch is the first 64 of a permutation of the training set drawn with `"gradcheck-batch"`. All counts are machine-readable keys in `prereg/recipe.md`. the train/eval split is by *starting condition* — no evaluation rollout starts from any training start (MU-7, TC-MU7-01; the distinct `"train-starts"`/`"eval-starts"` seed purposes make the two sets independent by construction, not just by a post-hoc disjointness check), with the harness also asserting disjointness at run time. Resemblance between train and eval states is expected and fine (the systems cycle); only literal start reuse is forbidden, per MU-7's own wording.

## 5. API Boundary Contracts

What the harness extracts from models for the judge — the only model-shaped data the judge ever sees (JU-1). Arrays are pre-shaped `[n_trials, H, d]` (judge spec v1.8 §4); a trial's own row in that first axis is its boundary, so there is no separate boundary-marker field (design-review fix — v1.0 named a `trial_boundaries: [n_trials] int` field here that judge.md's own array shapes made undefined and redundant; removed):

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

`is_fixture` and the model's `name` are *not* part of `JudgeInput` (the judge must stay blind) and are never computed by anything in this package's own model code — the harness assembles them into the harness-owned envelope (judge spec v1.8 §5) alongside the judge's pure `Verdict` once judging completes, which is what reporting actually consumes (design-review fix: v1.0 described this as the harness holding "a mapping" reporting separately "rejoins"; the envelope is the single object that replaces both, per Keeling's single-producer-per-fact principle applied in the judge spec).

## 6. Integration Points

- **← worlds:** training data generated through the world interface with the worlds spec's regions and action ranges; models never import world internals — they see `(state, action)` arrays, **plus the harness-constructed `WorldContext` (world geometry: boxes, action interval, `scale`) and `TrainingData` (the seeded trajectories) every factory receives uniformly (ADR-M1, design-review-004/005 — these two dataclasses are the only sanctioned channel for world facts and training data to reach a model, data handed in, never an import reaching out; the no-import direction is gate-enforced by TC-NF6-07)**.
- **↛ harness:** models never import the harness — the invariant the `SeedSource`-in-`wmj.models.base` design depends on (a fixture rebuilds `direct`'s stream via `seeds.rng_for("direct", …)` with no harness import). Gate-enforced by **TC-NF6-08** (cross-cutting ADR-003, design-review-007 repair — Round 7 found this direction asserted but unenforced). `SeedSource`/`component_key` live in `wmj.models.base` precisely so the shared seed utility is model-side, consumed by the harness (an allowed harness→models.base import) rather than pulling models toward the harness.
- **→ judge:** the boundary contract above; the judge spec owns `JudgeInput`.
- **→ harness:** registry discovery, training orchestration, prereg checks, train/eval disjointness assertion.

## 7. Error Handling & Edge Cases

- A model returning a malformed `Prediction` (wrong shape, non-positive spread, NaN) → `ModelContractError` naming the model and MU-1; the run aborts (no partial verdicts, TC-JU9-02 discipline).
- Gradient check failure → training refuses to start.
- Fewer than 12,500 kick pairs available → `TrainingDataError`; the harness never shrinks the count (design-review-010).
- A stateless model whose `predict_batch` row differs from `predict` → `ModelContractError` (TC-MU1-04).
- Prereg violations → pipeline stops before judging (ADR-M5).
- First-step linear extrapolation (no previous state) falls back to persistence — deterministic and documented, not an error.

## 8. Testing Strategy

| Concern | Cases |
|---|---|
| Uncertainty format fixed and identical | TC-MU1-01, TC-MU1-02 |
| `reset()` isolates rollouts — no cross-rollout state leak (design-review addition) | TC-MU1-03 (new) |
| `predict_batch` rows bit-identical to `predict`, every stateless model (design-review-010) | TC-MU1-04 |
| Baselines present and non-trivial | TC-MU2-01, TC-MU2-02 |
| Fixtures fail as specified | TC-MU3-01, TC-MU3-02, TC-MU3-03 |
| Fixture labels on all three surfaces | TC-MU4-01 |
| Each fixture's inner net is weight-identical to `direct` (design-review-005) | TC-MU4-02 (new) |
| Matching margin + pre-registered ensemble rules | TC-MU5-01, TC-MU5-02, TC-MU5-03 |
| β-NLL gradient matches its written formula (design-review-010) | TC-MU5-04 |
| M sufficiency, outcome-independent (design-review-010) | TC-MU5-05 |
| Prereg freeze: ordering, content at the freeze commit, one freeze ever, not-frozen refusal, SHA recorded, recipe/world constants agree (design-review-010) | TC-MU6-01, TC-MU6-04, TC-MU6-06, TC-MU6-07, TC-MU6-08, TC-MU6-09; publish-either-way TC-MU6-02 (judged) |
| Train/eval disjointness | TC-MU7-01 |
| Training subsample: one harness draw, shared, exact kick count, shortfall refused (design-review-010) | TC-MU7-03 |
| Deterministic training | TC-MU8-01 |
| Registry-only extensibility | TC-MU9-01 |
| No `models → worlds` import (design-review-005) | TC-NF6-07 (cross-cutting ADR-003) |
| Plus: MLP gradient check (this spec's own addition, pre-TC discipline; 64 pairs, tolerance 1e-5 — design-review-010) | — |

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
| 1.6 | 2026-08-31 | Design-review-006 repair (execution-verified): factory `rng` → `SeedSource` (models.base), giving fixtures a real channel to rebuild `direct` (executed weight-identity) and pinning the ensemble K=5 split; `is_baseline` flag added so a new baseline is genuinely one-file; `check_prereg` verifies the specific unrigged model's `prereg/` entry (TC-MU6-03); training count pinned `N_train=2000` with a machine-readable recipe format; distinct seed *purposes* for train/eval/benchmark starts (closing the collision Round 6 found). |
| 1.7 | 2026-08-31 | Design-review-007 repair (execution-verified): `check_prereg` enforces prereg content-invariance (first-added blob hash == working-tree hash — TC-MU6-04, closing the honestly-dated-second-commit drift Round 7 executed with git); `is_baseline` scope stated (prereg-exempt only, not a comparator — TC-MU6-05); `model_ref` cross-roster shift disclosed (ADR-M1); `models→harness` no-import direction gate-enforced (TC-NF6-08). |
| 1.8 | 2026-08-31 | Design-review-008 repair (execution-verified): `check_prereg`'s first-added-commit resolution pinned to `--reverse` (closing a delete-then-readd bypass Round 8 executed); MU-5's margin-generalization overclaim in ADR-M5 corrected (fixed pairwise comparison, not N-model); `component_key` call sites now pass `str(k)` explicitly (TC-NF1-08). |
| 1.9 | 2026-09-04 | Design-review-009 repair (post-construction, verdict: Build with caveats): ADR-M2 pins `ddof=1` for **both** baselines' spread fits, not persistence alone — an interim build had pinned it for persistence only, silently leaving linear on NumPy's population default with no degeneracy guard; both now share one guarded implementation and a fail-loud `DegenerateSpreadError` (A8/C1, TC-MU2-03 added). |
| 2.0 | 2026-09-28 | Design-review-010 (owner-triaged): `stateless` + `predict_batch` with bit-identical rows and a batch-invariant MLP prediction forward (ADR-M1); `TrainingData` gains the harness-drawn training/held-out/grad-check pairs (ADR-M1, §4); β-NLL (β=0.5) with its gradient written out; batch 256; 64-pair gradient check, A10 ratified; outcome-independent M-sufficiency test (ADR-M3); the one-time freeze replaces the first-added anchor, residuals #4 (A13) and #5 added (ADR-M5); §7/§8 updated. |

---

*Developed using the Grounded Vibe Methodology*
