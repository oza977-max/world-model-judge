# P3-C03 measurements — Model A ("direct"), real recipe, 2026-10-03

**In plain words:** what happened when Model A was trained for real on the
frozen recipe's 50,000 (and 100,000) training examples, on both worlds. The
headline: how well it does depends a lot on the random seed, so the
pre-registered "is 50,000 enough?" test — which compares one 50,000-example run
with one 100,000-example run — mostly measures that luck, not the amount of
data. Nothing in the recipe was changed because of this; it is the owner's
decision (REMEMBER.md D18).

Setup: seed 20260825 for the data; `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`;
the training set, held-out set and kicks exactly as `build_training_data`
makes them from `prereg/recipe.md`; Model A as built in `models/direct.py`
(2×64 tanh, β-NLL β=0.5, batch 256, Adam 1e-3). "Error" is the held-out error
of ADR-M3: mean over held-out pairs and dimensions of ((mean − y)/scale)².
Runtime: one 100-epoch training ≈ 17 s (either world); a 100,000-pair training
≈ 35 s.

## 1. The pre-registered sufficiency test, one run per world (seed 20260825)

```
lv: err(M)=4.9051e-08 err(2M)=1.9482e-07 ratio=0.252 sufficient=True time=52s
   split M : KickSplit(n_kick=10, n_plain=9990, error_kick=1.6246712254372518e-08, error_plain=4.908373359525273e-08)
   split 2M: KickSplit(n_kick=10, n_plain=9990, error_kick=1.6227634630330138e-07, error_plain=1.948550332494074e-07)
pendulum: err(M)=8.3938e-08 err(2M)=1.3580e-08 ratio=6.181 sufficient=False time=76s
   split M : KickSplit(n_kick=7, n_plain=9993, error_kick=8.199106150145154e-08, error_plain=8.393887774174443e-08)
   split 2M: KickSplit(n_kick=7, n_plain=9993, error_kick=1.82119243077415e-08, error_plain=1.3576529835424038e-08)
```

By the pre-registered rule alone: LV → "M = 50,000 is enough" (ratio 0.25 ≤
1.10); pendulum → "not enough" (ratio 6.2). Either world failing switches both
worlds to 100,000 — but only once the ensemble (P3-C04) has also been checked.
The recipe has not been touched.

## 2. Held-out error against number of epochs (one seed, 50,000 pairs)

```
lv 10 5.3846e-07
lv 25 6.8472e-07
lv 50 3.4523e-07
lv 100 5.4208e-08
lv 200 2.8977e-08
pendulum 10 9.1392e-08
pendulum 25 7.8187e-08
pendulum 50 9.8096e-08
pendulum 100 8.7380e-08
pendulum 200 1.5487e-08
```

Not monotone, and still falling at 200 epochs (the recipe pins 100).

## 3. The same comparison with different seeds (identical data)

```
lv epochs 100 seed 1 err(M)=9.347e-08 err(2M)=2.616e-07 ratio=0.36
lv epochs 100 seed 2 err(M)=1.699e-06 err(2M)=4.344e-08 ratio=39.12
lv epochs 100 seed 3 err(M)=1.049e-07 err(2M)=7.181e-07 ratio=0.15
lv epochs 200 seed 1 err(M)=3.898e-08 err(2M)=2.305e-08 ratio=1.69
lv epochs 200 seed 2 err(M)=2.866e-08 err(2M)=2.754e-08 ratio=1.04
lv epochs 200 seed 3 err(M)=1.480e-07 err(2M)=9.009e-08 ratio=1.64
pendulum epochs 100 seed 1 err(M)=3.215e-07 err(2M)=9.134e-09 ratio=35.20
pendulum epochs 100 seed 2 err(M)=2.999e-08 err(2M)=1.930e-08 ratio=1.55
pendulum epochs 100 seed 3 err(M)=5.421e-08 err(2M)=8.814e-09 ratio=6.15
pendulum epochs 200 seed 1 err(M)=1.775e-08 err(2M)=3.487e-09 ratio=5.09
pendulum epochs 200 seed 2 err(M)=4.366e-07 err(2M)=2.215e-09 ratio=197.09
pendulum epochs 200 seed 3 err(M)=1.347e-08 err(2M)=2.803e-09 ratio=4.81
```

On identical data the ratio err(M)/err(2M) ranges from 0.15 to 39 (LV, 100
epochs) and 1.6 to 35 (pendulum, 100 epochs); at 200 epochs 1.0 to 197. A single
run therefore cannot tell "not enough data" from "an unlucky seed". The 2M model
also gets twice as many optimisation steps at the same epoch count, which is a
second confound. Over three seeds the pendulum median ratio is still above 1.10
(6.2 at 100 epochs, 5.1 at 200) — a real effect is probably in there for the
pendulum — while LV's medians are 0.36 (100) and 1.64 (200), i.e. not stable.

## 4. Model A's own error bars (single run, real recipe)

Spread is roughly 2× the real error on LV (z std 0.38 / 0.27) and mixed on the
pendulum (ω₁ z mean −1.8: biased and overconfident there). Practice model, not
a defect — this is the material the judge exists to grade.
