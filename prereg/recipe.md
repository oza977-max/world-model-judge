# Pre-registration recipe — World Model Judge

**In plain words:** this file is written and committed *before* any unrigged
model is judged. It fixes, in advance, every choice that could otherwise be
nudged after seeing results — the model architecture, how much data and how
long we train, how the two models derive their uncertainty, the seeds policy,
and the margin at which we will call the two models "accuracy-matched." The
harness (`check_prereg`) refuses to judge an unrigged model unless this file
was **frozen** first — locked by the one commit that adds `prereg/FREEZE` —
and has not changed since (models ADR-M5, MU-6). Until the freeze, a number
that turns out to need changing is revised **openly**, with a dated entry in
the revision log at the end of this file — never by nudging a trained model.
After the freeze, nothing here changes.

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
batch_size: 256
subsample_pairs: 100000
subsample_pairs_fallback: 100000
kick_pairs: 12500
heldout_pairs: 10000
gradcheck_pairs: 64
heldout_kick_pairs: 1000
lr_initial: 0.001
lr_final: 0.00001
lr_schedule: cosine
sufficiency_seeds: 5
sufficiency_tolerance: 0.10
beta_nll: 0.5
matching_margin: 0.05
ensemble_members: 5
lv_kick_rate_per_s: 0.5
lv_action_max: 0.1
pendulum_kick_rate_per_s: 1.0
pendulum_action_max: 1.0
```

- `training_trajectories: 2000` — trajectories drawn per world for training
  (models §4; a round figure, ample for a 2-hidden-layer 64-unit MLP on a
  2–4-dimensional system, fixed in advance, not derived).
- `epochs: 100` — fixed epoch count, **no early stopping** (early stopping
  peeks at validation loss, a tuning channel MU-6 exists to close). This
  value is a pre-registered modelling choice; its convergence-and-runtime
  feasibility against the NF-2 budget is validated when training is built
  (P3-C03/C06) — see `build/spec-corrections-backlog.md` A12.
- `batch_size: 256` — mini-batch training (models ADR-M3). Was 32 until the
  2026-09-28 revision; measured 2.84 µs vs 7.05 µs per example.
- `subsample_pairs: 100000` (was 50,000 until the 2026-10-04 fallback — see the
  revision log) — each MLP trains on this many one-step pairs,
  drawn once per world by the harness from the 2,000 trajectories and shared
  by every network (models §4). The full trajectories are kept; training on
  all ~1.4M (LV) / ~10M (pendulum) highly correlated pairs would take ~14
  hours, against a run meant to finish in minutes.
- `subsample_pairs_fallback: 100000` — the **only** other value
  `subsample_pairs` may take, and only if the sufficiency test below fails.
- `kick_pairs: 12500` — exactly this many of the training pairs are kick
  pairs (a non-null action), so the models see the lever often enough to
  learn it. Fewer available → the harness refuses; it never shrinks the count.
- `heldout_pairs: 10000` — pairs *not* used for training, kept for the
  sufficiency test and the kick/non-kick report only — never for early
  stopping or tuning.
- `heldout_kick_pairs: 1000` — exactly this many of the `heldout_pairs` are kick
  pairs (added 2026-10-04, owner decision D17, before any model was judged).
  Without it the held-out set held 6 (predator–prey) / 10 (pendulum) kick pairs
  out of 10,000, because the training quota takes most of the kicks the worlds
  supply, so the "held-out error split by kick and non-kick" report would have
  rested on a handful of examples. They are drawn from the kick pairs the
  training set did not take (predator–prey has about 1,370 of them, the
  pendulum about 7,400); fewer available → the harness refuses, as for the
  training quota. Held-out kick pairs are 10% of the held-out set against ~1%
  (predator–prey) / ~0.2% (pendulum) of evaluation steps — a shift disclosed,
  not hidden, and the split report names both counts.
- `gradcheck_pairs: 64` — the one-time backprop check runs on this many
  pairs (models ADR-M3; tolerance 1e-5, backlog A10 ratified).
- `sufficiency_tolerance: 0.10` — see "Is 50,000 pairs enough?" below.
- `sufficiency_seeds: 5` — how many training seeds the sufficiency test takes
  the median over (D18).
- `lr_initial`, `lr_final`, `lr_schedule` — the cosine learning-rate decay (D18;
  see "Shared MLP architecture").
- `beta_nll: 0.5` — Model A's loss weight (β-NLL; see Model A).
- `lv_kick_rate_per_s`, `lv_action_max`, `pendulum_kick_rate_per_s`,
  `pendulum_action_max` — how often the lever is pulled (kicks per second
  of world time) and its trained half-width. The per-step kick probability
  is `p_step = kick_rate_per_s × dt`: **LV 0.5 × 0.02 = 0.01, pendulum
  1.0 × 0.002 = 0.002**. These must equal the world modules' own constants
  (TC-MU6-09), so the frozen recipe pins them.
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
- Optimiser: Adam, β₁ `0.9`, β₂ `0.999`, ε `1e-8` (all pinned — two
  implementers picking different defaults would get bit-different weights
  under NF-1 determinism), with a **cosine learning-rate decay** from
  `lr_initial: 0.001` to `lr_final: 0.00001` over the fixed epochs
  (`lr_schedule: cosine`; the rate is constant within an epoch, epoch `e` of `E`
  uses `lr_final + ½(lr_initial − lr_final)(1 + cos(π·e/(E−1)))`). Added
  2026-10-04 (owner decision D18): with a constant rate the final error of the
  *same* network on the *same* data swung by a factor of 20 from the random seed
  alone; with the decay every seed lands within a factor of ~1.3
  (`build/measurements/p3-c03-model-a-real-run.md`, section 5).
- Gradient check: one finite-difference check of backprop on the fixed
  64-pair batch, run once before training — `direct`'s network under plain
  Gaussian NLL, each ensemble member under MSE; relative-error denominator
  floored at 1e-3 of the largest gradient, tolerance 1e-5 (models ADR-M3).

## Model A — "direct" (Nix & Weigend)

One MLP taking `(state, action)` — state normalised by the world's `scale`,
action by its training half-width — and outputting per-dimension `(Δmean,
log σ)`. `mean = state + Δmean`, `spread = exp(log σ)`. Trained with
**β-NLL, β = 0.5** (Seitzer et al., ICLR 2022): per dimension, with
`r = y − μ`, `s = log σ` and weight `w = σ^(2β)` computed from the current
forward pass and **held constant**,
`L = w · [0.5·log(2π) + s + r² / (2σ²)]`, summed over dimensions, averaged
over the batch. Gradients handed to backprop: `∂L/∂μ = −w·r/σ²`,
`∂L/∂s = w·(1 − r²/σ²)`, each divided by the batch size. *Why not plain
NLL:* plain NLL down-weights the squared error wherever the model predicts a
large variance, which degrades its mean fit — leaving any result open to
"you picked a known-weak Model A". β-NLL keeps Model A a self-predicted-
error-bar model, so the comparison with the ensemble is unchanged.

## Model B — "ensemble" (Lakshminarayanan et al.)

`K = 5` MLPs, same architecture minus the variance head (mean output only),
identical training data (the same `subsample_pairs` set), trained on
mean-squared error of the state change, differing only in seeded
initialisation and seeded batch shuffling (member `k` draws from
`seeds.rng("member", str(k), …)`).

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

## How the lever is pulled (worlds ADR-W2)

Training trajectories, evaluation trials and the divergence benchmark all
draw their actions from one generator: at each step a kick with probability
`p_step` (above), otherwise no action. In-range kicks are uniform on
`[−max, max]`; the action-axis test region `out-large-action` uses kicks
with `|u|` uniform on `(max, 2·max]`. Predator–prey kicks are small (trained
`[−0.1, 0.1]`) because larger ones were measured to crash the prey
population through its floor on full-length runs.

**Disclosed:** training pairs are 25% kicks (`kick_pairs / subsample_pairs`)
while evaluation steps are about 1% (LV) and 0.2% (pendulum) kicks. Both
unrigged models share this. The build reports each model's held-out error
separately on kick and non-kick pairs.

## Is 50,000 pairs enough? (fixed now, checked once, before the freeze)

Decided by a test that never looks at the comparison it must stay
independent of. For each unrigged model, train at `subsample_pairs` and at
twice that (the same 12,500 kick pairs plus more non-kick pairs, everything
else identical); held-out error = mean over the held-out pairs of the mean
over dimensions of `((prediction − truth) / scale)²`. **Enough iff
`err(50,000) ≤ 1.10 × err(100,000)` for both models, each error being the
**median over `sufficiency_seeds: 5` training seeds** (seed `k` is the run seed
plus `k`; added 2026-10-04, owner decision D18 — a single run is partly luck).**
Both versions are scored on the held-out set of the 100,000-pair build (disjoint
from both training sets). If either fails,
`subsample_pairs` becomes 100,000 for both — the only fallback — recorded
in the revision log below; there is no further iteration. The test never
reads evaluation trials, skill scores or `matching_margin`.

## Seeds policy (cross-cutting ADR-002)

Every random draw comes from a content-addressed `SeedSource` keyed on stable
name strings, never iteration position — so adding a model never shifts
another model's stream. Training starts and evaluation starts use distinct
seed purposes (`"train-starts"` / `"eval-starts"`), making the two sets
disjoint by construction (MU-7). The 2026-09-28 revision adds its own
purposes, each distinct: `"train-kicks"`, `"eval-kicks"`, `"benchmark-kicks"`
(one stream per trajectory, trial or start), `"subsample-kick"`,
`"subsample-nonkick"`, `"heldout"`, `"gradcheck-batch"`.

## The uncertainty format (MU-1)

Every model — baseline, unrigged, or fixture — states its uncertainty in one
fixed format: **per-dimension mean + one standard deviation** (cross-cutting
data model, `Prediction`). This is recorded here because MU-6's own text
names the uncertainty format as one of the things fixed before judging.

## Thresholds

The judge's exception bands, the sharpness-hedge threshold, and the
climatology agreement threshold are derived by `derive_thresholds.py` and
committed alongside this file as `prereg/thresholds.json` (JU-11).

## Revision log

Every change to this file before the freeze is listed here, dated, with its
reason. Earlier versions stay readable in git history.

- **2026-09-11 — first committed** (P3-C07).
- **2026-09-28 — one open revision, before any model was trained or
  judged** (design-review-010, owner-triaged; detail in
  `build/spec-corrections-backlog.md`, "Disposition (Round 10)"). Why: the
  first version would have taken ~14 hours to train against a run meant to
  finish in minutes, and never said how actions are chosen. Changed:
  `batch_size` 32 → 256; added `subsample_pairs`, its fallback,
  `kick_pairs`, `heldout_pairs`, `gradcheck_pairs`, `sufficiency_tolerance`,
  `beta_nll` and the four kick keys; Model A's loss from plain NLL to
  β-NLL; the gradient check from "one full batch" to 64 pairs at tolerance
  1e-5; the kick generator and the sufficiency test added. Unchanged:
  `training_trajectories`, `epochs`, `matching_margin`, `ensemble_members`,
  the architecture, the ensemble rules, the baselines, the uncertainty
  format.
- **2026-10-04 — one open revision, before any model was judged** (owner
  decision D17; `build/spec-corrections-backlog.md` A19, A23). Added
  `heldout_kick_pairs: 1000`. Why: the measured held-out kick share was 6 / 10
  in 10,000. Unchanged: every other key.
- **2026-10-04 — a second open revision the same day, before any model was
  judged** (owner decision D18; `build/spec-corrections-backlog.md` A22, A23).
  Added `lr_initial`, `lr_final`, `lr_schedule: cosine` (Model A and every
  ensemble member) and `sufficiency_seeds: 5`; the sufficiency test now uses the
  median over seeds and the larger build's held-out set. Why: measured on
  identical data, the ratio err(50,000)/err(100,000) ranged from 0.15 to 39 by
  seed alone, and error was still falling at 200 epochs. Epochs stay at 100
  (with the decay, 200 epochs lowered error a little more at twice the cost).
  Unchanged: every other key.
- **2026-10-04 — the pre-registered fallback is enacted: `subsample_pairs`
  50,000 → 100,000** (the rule in "Is 50,000 pairs enough?"; no owner choice
  involved). The check was run once for Model A with the final settings (held-out
  kick quota, cosine decay, median of 5 seeds): predator–prey ratio
  err(50,000)/err(100,000) = 1.29 and pendulum 2.75, both above 1.10, every seed
  agreeing in direction (`build/measurements/p3-c03-model-a-real-run.md`, section
  6). The rule says one failing model switches both worlds and both models, so
  the ensemble's later check cannot change the outcome; there is no further
  iteration. `subsample_pairs_fallback: 100000` stays as the record of the only
  other value. Unchanged: every other key. Consequence: one Model A training now
  takes about 35 s instead of 17 s, and the held-out kick quota (1,000) plus the
  training kick quota (12,500) use 13,500 of predator–prey's ~13,870 kick pairs.
