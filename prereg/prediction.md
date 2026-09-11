# Pre-registered ranking prediction

**In plain words:** this is the prediction we commit to *before* seeing any
judged result — which of the two unrigged models the judge will rank better,
and why. Writing it down in advance is the whole point: if the judge later
contradicts it, both the prediction and the result are published unchanged
(MU-6 / TC-MU6-02). A prediction edited after the fact is not a prediction.

The two unrigged contestants are **direct** and **ensemble** (named here so
`check_prereg` can confirm each was declared in advance — TC-MU6-03).

## The prediction

By ordinary one-step error, **direct** and **ensemble** are expected to be a
statistical tie — they share the same architecture, data, and training, and
differ only in how they derive their uncertainty (direct predicts its own
error bar from a variance head; ensemble reads its spread from the
disagreement among five members). That tie is the point: the pre-registered
`matching_margin` of 0.05 is what we require their one-step skill scores to
fall within, per task and region (TC-MU5-01).

**Where we predict they diverge — the headline (MU-5):** the two will *not*
be ranked equal by the judge, because the judge grades calibration and
sharpness, not just error. We predict that **ensemble's disagreement-based
spread is better calibrated than direct's self-predicted error bar**, and so
**ensemble earns the longer trust horizon** — most visibly on the predator–
prey control task, where direct's variance head is expected to be
overconfident (its intervals too narrow) while ensemble's spread more
honestly covers the truth.

We record this now, before any judged run, precisely so that a contrary
result (direct better calibrated, or the two genuinely indistinguishable on
calibration too) cannot be quietly reinterpreted as what we "expected." Both
outcomes will be published.

## What would make this prediction wrong

- If the two models are *not* accuracy-matched (one-step skill difference
  ≥ 0.05), pre-registration is not yet satisfied and judging does not
  proceed — the remedy is revising the recipe openly and re-running, not
  re-tuning either model.
- If the judge ranks direct's calibration at or above ensemble's, the
  prediction is contradicted — and is published as contradicted.
