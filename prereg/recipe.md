# Pre-registration recipe — World Model Judge

**In plain words:** this file is written and committed *before* any unrigged
model is judged. It fixes, in advance, every choice that could otherwise be
nudged after seeing results — the model architecture, how much data and how
long we train, how the two models derive their uncertainty, the seeds policy,
and the margin at which we will call the two models "accuracy-matched." The
harness (`check_prereg`) refuses to judge an unrigged model unless this file
was committed first and has not changed since (models ADR-M5, MU-6). If any
of these numbers turns out to need changing, the honest remedy is to revise
this recipe **openly and re-run** — never to nudge a trained model.

The two unrigged contestants this recipe governs are **direct** and
**ensemble** (named here so `check_prereg` can confirm each was declared in
advance — TC-MU6-03).

---

## Machine-readable pinned counts

These lines are parsed by the training chunks and the margin check; the
format (a `key: value` line) is fixed so there is one defined source.

```
training_trajectories: 2000
epochs: 100
batch_size: 32
matching_margin: 0.05
ensemble_members: 5
```

- `training_trajectories: 2000` — trajectories drawn per world for training
  (models §4; a round figure, ample for a 2-hidden-layer 64-unit MLP on a
  2–4-dimensional system, fixed in advance, not derived).
- `epochs: 100` — fixed epoch count, **no early stopping** (early stopping
  peeks at validation loss, a tuning channel MU-6 exists to close). This
  value is a pre-registered modelling choice; its convergence-and-runtime
  feasibility against the NF-2 budget is validated when training is built
  (P3-C03/C06) — see `build/spec-corrections-backlog.md` A12.
- `batch_size: 32` — mini-batch training (models ADR-M3).
- `matching_margin: 0.05` — the MU-5 headline: the two unrigged models' one-
  step skill scores, per task and region, must differ by **less than 0.05**.
  If any pair exceeds it, pre-registration is "not yet satisfied" and judging
  does not proceed (TC-MU5-01).
- `ensemble_members: 5` — K for Model B.

## Shared MLP architecture (models ADR-M3)

- 2 hidden layers × 64 units, tanh hidden, **linear output** (outputs are a
  state change / log-spread / mean — unbounded), hand-rolled in NumPy.
- Weight init: `W ~ U(-1/√fan_in, 1/√fan_in)`, biases 0, drawn from a seeded
  `Generator`.
- Optimiser: Adam, lr `1e-3`, β₁ `0.9`, β₂ `0.999`, ε `1e-8` (all pinned — two
  implementers picking different defaults would get bit-different weights
  under NF-1 determinism).
- Gradient check: one full-batch finite-difference check run once before
  training (models §8; see backlog A10 for the tolerance correction).

## Model A — "direct" (Nix & Weigend)

One MLP taking `(state, action)` — state normalised by the world's `scale`,
action by its training half-width — and outputting per-dimension `(Δmean,
log σ)`. `mean = state + Δmean`, `spread = exp(log σ)`. Trained with Gaussian
negative log-likelihood: `NLL = 0.5·log(2π) + log σ + (y − μ)² / (2σ²)`,
summed over dimensions, averaged over the batch.

## Model B — "ensemble" (Lakshminarayanan et al.)

`K = 5` MLPs, same architecture minus the variance head (mean output only),
identical training data, differing only in seeded initialisation and seeded
batch shuffling (member `k` draws from `seeds.rng("member", str(k), …)`).

- **Point prediction:** the arithmetic mean of the K member means.
- **Spread mapping:** per-dimension `spread = sqrt(1 + 1/K) × std(member_means,
  ddof=1)` — sample standard deviation with Bessel's correction, inflated by
  the standard `sqrt(1 + 1/K)` small-ensemble underdispersion factor.

These two rules are fixed here and checked against this text at judging time
(TC-MU5-02, TC-MU5-03), not against what "seems reasonable" then.

## Baselines (models ADR-M2)

- **persistence:** mean = current state; spread = sample std (`ddof=1`) of
  one-step training changes.
- **linear:** mean = current + (current − previous), persistence on the first
  step; spread = sample std (`ddof=1`) of that rule's own training residuals.

Baselines declare `is_baseline=True` and are prereg-exempt (TC-MU6-05).

## Seeds policy (cross-cutting ADR-002)

Every random draw comes from a content-addressed `SeedSource` keyed on stable
name strings, never iteration position — so adding a model never shifts
another model's stream. Training starts and evaluation starts use distinct
seed purposes (`"train-starts"` / `"eval-starts"`), making the two sets
disjoint by construction (MU-7).

## The uncertainty format (MU-1)

Every model — baseline, unrigged, or fixture — states its uncertainty in one
fixed format: **per-dimension mean + one standard deviation** (cross-cutting
data model, `Prediction`). This is recorded here because MU-6's own text
names the uncertainty format as one of the things fixed before judging.

## Thresholds

The judge's exception bands, the sharpness-hedge threshold, and the
climatology agreement threshold are derived by `derive_thresholds.py` and
committed alongside this file as `prereg/thresholds.json` (JU-11).
