# World Model Judge — Worlds Specification

Version 1.0 · 25 August 2026 · Domain 1 of Requirements v1.2 · References: cross-cutting spec v1.0

**What this document is.** The complete design of the two reference worlds — Lotka–Volterra and the double pendulum — including the shared integrator, the action levers, the divergence benchmark, the declared regions, and the evaluation tasks. Everything the judge later grades against is defined here.

**In plain words:** this spec builds the pretend worlds where every correct answer is knowable. It fixes the exact equations, the exact numbers, and the exact meaning of "familiar territory" and "the world's own unpredictability" — so no later stage can fudge them.

---

## Expert Panel

| Expert | Work | Role in This Document |
|--------|------|----------------------|
| Edward Lorenz | *Deterministic Nonperiodic Flow* (1963) | The divergence benchmark's whole rationale: grade against the world's own drift, not zero |
| Ernst Hairer, Nørsett & Wanner | *Solving Ordinary Differential Equations I* | Integrator selection (ADR-W1) and the drift-is-systematic argument behind WD-3's bound |
| David Goldberg | *Floating-Point Arithmetic* (1991) | Why trajectory determinism (WD-7) is an ordering discipline |
| Jolliffe & Stephenson | *Forecast Verification* | Regime-dependence of predictability — the reason WD-4 is a curve per region, not a number |
| Michael Keeling | *Design It!* | ADR format |

---

## 1. Purpose

Covers requirements **WD-1 through WD-8**. The worlds package (`wmj.worlds`, per cross-cutting ADR-003) provides: two dynamical systems with exact discretised ground truth (WD-1), a single state-and-action interface (WD-2), the shared integrator with measured drift (WD-3), the empirical divergence benchmark (WD-4), declared training/out-of-training regions for states *and* actions (WD-5), two tasks per world with different tolerances (WD-6), and byte-identical seeded trajectories (WD-7). Randomness, hidden state, and high-dimensional input are excluded (WD-8, Won't).

## 2. Architecturally Significant Requirements

- **WD-3** forces a *single shared integrator module*: truth and models call the same `step()` code object, and the gate test compares identity, not configuration values alone.
- **WD-4** forces divergence to be *measured artefacts, computed once per world per region and cached as data* — the judge receives curves as input (JU-1), so they must exist as plain arrays before judging.
- **WD-7 + NF-1** force the worlds to be pure functions of (state, action): all sampling randomness lives in the harness, none in the worlds.
- **WD-6** forces tasks to be world-owned data (name + tolerance + horizon), not judge logic — the judge reads tolerances; it never defines them.

## 3. Design Decisions

### ADR-W1 — Integrator: shared fixed-step RK4, drift measured and bounded

**Status:** Accepted. [Requirement: WD-1, WD-3] [Test: TC-WD3-01, TC-WD3-02, TC-WD3-03]

**Context:** WD-3 requires ground truth and every model to advance on the same integrator and step size, test-enforced, with the integrator's own drift in each world's conserved quantity measured and bounded over the JU-6 horizons.

**Options considered:**
1. **Fixed-step classical RK4, one shared module** — O(h⁴) local error, well-characterised (Hairer), trivially deterministic, same code for both worlds.
2. **Symplectic integrator for the pendulum (leapfrog), RK4 for LV** — near-zero energy drift for the pendulum; but two integrator families to gate, no comparable structure-preserving choice for LV's non-canonical invariant, and WD-3 demands drift be *measured and bounded*, not eliminated.
3. **Adaptive-step methods (e.g. RK45)** — rejected outright: adaptive steps make "the same step size" (WD-3) meaningless and break byte-determinism across state regions.

**Decision:** Option 1. One module `wmj.worlds.integrator` exposing `rk4_step(deriv_fn, state, dt)`. The WD-3 gate asserts that the world's and the model-rollout driver's integrator are the *same function object* and the same `dt` constant (TC-WD3-01), and the negative test proves the gate fails on a mismatched `dt` (TC-WD3-02).

**Drift bound (the TC-WD3-03 numbers):** over each world's full JU-6 horizon (§4), the relative drift of the conserved quantity under the true dynamics with null action shall stay below **1e-6 (relative)**. RK4 at the step sizes below sits orders of magnitude under this in preliminary hand calculation; the bound is deliberately loose enough to be stable across platforms and tight enough that JU-6's conditioned climatology cannot be corrupted by integrator error. The measured drift curve is stored alongside the divergence benchmark and re-checked by the gate on every full run.

**Consequences:** Energy/orbit drift exists (RK4 is not structure-preserving — Hairer) but is measured, bounded, and small relative to every task tolerance. If a future world violates the bound, the gate fails the run loudly rather than producing an untrustworthy climatology (TC-WD3-03's "fails loudly" branch).

### ADR-W2 — Action semantics: impulse at step boundary

**Status:** Accepted. [Requirement: WD-2] [Test: TC-WD2-01]

**Context:** WD-2 requires `(state, action) → next_state`. The action must genuinely change the successor state, and the integrator must stay shared and autonomous (ADR-W1).

**Options considered:** (1) action as a forcing term inside the ODE right-hand side — makes the derivative function action-dependent and complicates the shared-integrator gate; (2) **action as an instantaneous impulse applied to the state at the step boundary, followed by one null-action integration step** — keeps the ODE autonomous, the integrator identical for truth and models, and the action's effect exact and explainable.

**Decision:** Option 2. `transition(state, action) = rk4_step(deriv, apply_action(state, action), dt)`. A null action (`0.0`) makes `apply_action` the identity, so TC-WD2-01's non-null-action-changes-outcome check is exact.

**Consequences:** The action's physical meaning is a discrete intervention ("remove rabbits now", "kick the pivot now") — which matches the essay's lever language and makes "what happens if I do this" literal.

### ADR-W3 — Divergence benchmark: empirical median curve per region

**Status:** Accepted. [Requirement: WD-4] [Test: TC-WD4-01, TC-WD4-02]

**Context:** WD-4 requires each world to report separation growth between nearby trajectories as a measured curve, for a declared perturbation size, distance measure, and starting region — explicitly not a single rate.

**Decision — the declared quantities:**
- **Perturbation:** relative size **δ₀ = 1e-6** applied to each state dimension (state × (1 + δ₀ per-dimension, sign-alternating)).
- **Distance measure:** RMS over *normalised* state dimensions — each dimension divided by its world's declared scale vector (§4), so populations and radians are comparable. This same normalised distance is the error metric the judge uses (one metric everywhere; the judge spec references this definition).
- **Sampling:** **64 starting states** per declared region (training region and each out-region), sampled by the harness's seeded generator; null actions throughout (the benchmark measures the *world's* drift, not a policy's).
- **The curve:** median separation at each step out to the world's JU-6 horizon, reported per region, stored as arrays `(region, step) → separation` in the divergence artefact. Median, not mean, because chaotic separations are heavy-tailed and one saturated trajectory would swamp a mean.
- **Sanity assertions built into the artefact:** LV's curve grows sub-exponentially (log-separation vs step is concave/linear — TC-WD4-01's linear-in-phase check), and the pendulum's low-energy and high-energy curves differ by a declared factor (≥5× separation at half-horizon — TC-WD4-02's regime check).

**Consequences:** The benchmark is data, computed by one harness command, cached under `out/benchmarks/`, and handed to the judge as arrays. Regenerating it is deterministic (seeded).

### ADR-W4 — Region declarations: closed boxes, in-region on the boundary

**Status:** Accepted. [Requirement: WD-5] [Test: TC-WD5-01, TC-WD5-02]

**Decision:** Regions are axis-aligned boxes over state dimensions and an interval over the action. Membership is **closed on the training region**: a point exactly on the training boundary is in-region (TC-WD5-01's boundary determinism). An evaluation is labelled out-of-region if *either* its start state is outside the training state box *or* any action in its rollout is outside the trained action interval — and the label records *which axis* (state, action, or both), per WD-5's "say which" clause (TC-WD5-02).

## 4. Component Design — the two worlds, exactly

### 4.1 Lotka–Volterra (foxes and rabbits)

**Equations:** dx/dt = αx − βxy, dy/dt = δxy − γy, with x = prey, y = predator.

| Constant | Value | Meaning |
|---|---|---|
| α, β, γ, δ | 1.0, 0.4, 0.8, 0.2 | growth/predation/death/conversion; equilibrium at (γ/δ, α/β) = (4.0, 2.5) |
| dt | 0.02 | step size (shared, WD-3); near-equilibrium period ≈ 2π/√(αγ) ≈ 7.0 time units ≈ 351 steps |
| Horizon H | 700 steps (14.0 time units, ≈2 cycles) | rollout + JU-6 + drift-measurement horizon |
| Scale vector | (4.0, 2.5) | normalisation for the shared distance measure (equilibrium values) |
| Conserved quantity | V(x,y) = δx − γ·ln x + βy − α·ln y | the "orbit" WD-3 bounds and JU-6 conditions on |
| Action | u ∈ [−1.0, 1.0], prey impulse: x ← max(x + u, 0.05) | the lever: remove (u<0) or add (u>0) rabbits; floor keeps populations positive |
| State floor | both dimensions clamped ≥ 0.05 after every step | LV populations are positive by construction; the clamp is a guard, and any clamp activation is logged as an integrator-anomaly event |

**Regions (WD-5):** training states x ∈ [2.0, 6.0] × y ∈ [1.0, 4.0]; out-region states x ∈ [8.0, 12.0] × y ∈ [4.0, 6.0] (high-amplitude orbits). Trained actions u ∈ [−0.5, 0.5]; out-of-range actions |u| ∈ (0.5, 1.0].

**Tasks (WD-6):**
| Task | Kind | Tolerance τ (normalised distance, §3 ADR-W3) | Description |
|---|---|---|---|
| `lv-control` | control (tight) | 0.10 | hold prey inside the band [3.5, 4.5] by working the lever; predictions graded to tight tolerance |
| `lv-planning` | planning (loose) | 0.40 | will prey crash below 1.0 within the horizon; predictions graded to loose tolerance |

### 4.2 Double pendulum

**Equations:** standard two-link frictionless double pendulum, state (θ₁, θ₂, ω₁, ω₂), the textbook equations of motion.

| Constant | Value | Meaning |
|---|---|---|
| m₁ = m₂ | 1.0 kg | link masses |
| l₁ = l₂ | 1.0 m | link lengths |
| g | 9.81 m/s² | gravity |
| dt | 0.002 s | step size (shared, WD-3) |
| Horizon H | 5000 steps (10.0 s) | rollout + JU-6 + drift-measurement horizon |
| Scale vector | (π, π, 2π, 2π) | normalisation for the shared distance measure |
| Conserved quantity | total mechanical energy E(θ₁, θ₂, ω₁, ω₂) | the "energy shell" WD-3 bounds and JU-6 conditions on |
| Action | u ∈ [−2.0, 2.0] rad/s, pivot impulse: ω₁ ← ω₁ + u | the lever: kick the first joint's angular velocity |
| Angles | stored unwrapped (not mod 2π) | distance and the flip task need the winding; reporting may display wrapped |

**Regions (WD-5):** training states θ₁, θ₂ ∈ [−0.3, 0.3] rad, ω₁, ω₂ ∈ [−0.5, 0.5] rad/s (low-energy, near-regular); out-region states θ₁ ∈ [2.5, π] (near-inverted, chaotic regime), θ₂ ∈ [−0.3, 0.3], ω ∈ [−0.5, 0.5]. Trained actions u ∈ [−1.0, 1.0]; out-of-range |u| ∈ (1.0, 2.0].

**Tasks (WD-6):**
| Task | Kind | Tolerance τ | Description |
|---|---|---|---|
| `dp-control` | control (tight) | 0.05 | hold the tip near a target angle; tight grading |
| `dp-planning` | planning (loose) | 0.30 | will the first link flip (unwrapped |θ₁| exceeds π) within the horizon; loose grading |

Tolerances are quantitatively different within each world by construction (TC-WD6-01); band-edge classification uses ≤ (closed), so exact-boundary values pass deterministically (TC-WD6-02).

### 4.3 The world interface (WD-2)

```
World (protocol, wmj.worlds.base):
  d: int                      # state dimensionality
  dt: float                   # shared step size (the WD-3 constant)
  transition(state, action) -> next_state        # pure; ADR-W2 semantics
  conserved(state) -> float                      # V or E; used by WD-3 gate and JU-6
  regions() -> RegionSpec                        # training + out boxes, state and action
  tasks() -> tuple[Task, ...]                    # name, kind, tolerance, horizon
  scale: float64[d]                              # normalisation vector
```

`Task` and `RegionSpec` are frozen dataclasses in `wmj.worlds.base`. Worlds hold **no mutable state and no RNG** — `transition` is a pure function, which is what makes TC-WD7-01's cross-process byte-identity achievable and TC-WD7-02's injected-RNG mutation detectable.

## 5. API Boundary Contracts

The divergence artefact — the shape the judge and reporting consume (produced by `wmj.harness.benchmarks`, serialized canonically per cross-cutting ADR-002):

```json
{
  "world": "lv",
  "perturbation": 1e-06,
  "distance": "rms-normalised",
  "regions": {
    "training": {"steps": [0, 1, "...", 700], "median_separation": [0.0, 1.1e-06, "..."]},
    "out-high-amplitude": {"steps": ["..."], "median_separation": ["..."]}
  },
  "drift": {"conserved_rel_drift_max": 3.2e-09, "bound": 1e-06, "within_bound": true},
  "seed": 20260825, "n_starts": 64
}
```

Field names, nesting, and units are fixed here; the judge spec references this contract rather than restating it. A trajectory artefact is simply `float64[H+1, d]` plus the action sequence `float64[H, a]`.

## 6. Integration Points

- **→ models:** models are trained on trajectory datasets the harness generates by sampling training-region starts and trained-range actions (models spec owns the recipe; the *regions* are owned here).
- **→ judge:** the judge receives task tolerances, the divergence curves, and region labels as plain data (JU-1) — never a `World` object.
- **→ harness:** the WD-3 gate, the drift check, and benchmark generation run in `wmj.harness`; worlds expose the pieces (`conserved`, `dt`, integrator identity) the gates inspect.

## 7. Error Handling & Edge Cases

- LV state-floor clamp activations and pendulum energy-drift bound violations raise/log per the cross-cutting error conventions — a clamped trajectory in a *benchmark or judged run* aborts that run loudly (it means the region/action declarations allowed an unphysical excursion, which is a spec bug, not data).
- `transition` called with an action outside the world's declared full range → `ActionRangeError` (the declared range is the world's contract; out-of-*trained*-range is legitimate and labelled, out-of-*declared*-range is a caller bug).
- Region boxes are validated at construction: training box strictly inside the state-floor-safe domain; out-region disjoint from training box on at least one axis.

## 8. Testing Strategy

| Concern | Cases |
|---|---|
| Ground truth correctness | TC-WD1-01 (hand-verified reference values, both worlds) |
| Action lever real | TC-WD2-01 |
| Integrator gate + drift | TC-WD3-01, TC-WD3-02 (negative), TC-WD3-03 |
| Divergence curve shape + regime dependence | TC-WD4-01, TC-WD4-02 |
| Region labelling incl. action axis | TC-WD5-01, TC-WD5-02 |
| Task distinctness + boundary determinism | TC-WD6-01, TC-WD6-02 |
| Byte-identical trajectories | TC-WD7-01, TC-WD7-02 (negative) |

Hand-verified reference values for TC-WD1-01: one LV step from (4.0, 2.5) with u=0 and one pendulum step from (0.1, 0.1, 0, 0) with u=0, computed independently (symbolic/high-precision) and embedded as constants with 15 significant digits.

---

## Changelog

| Version | Date | Change |
|---|---|---|
| 1.0 | 2026-08-25 | Initial version. All world constants pinned; integrator ADR decided (RK4 fixed-step, drift bound 1e-6 relative). |

---

*Developed using the Grounded Vibe Methodology*
