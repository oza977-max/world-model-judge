# P3-C03 measurements — Model A ("direct"), real recipe, 2026-10-03

**In plain words:** what happened when Model A was trained for real on the
frozen recipe's 50,000 (and 100,000) training examples, on both worlds. The
headline: how well it does depends a lot on the random seed, so the
pre-registered "is 50,000 enough?" test — which compares one 50,000-example run
with one 100,000-example run — mostly measures that luck, not the amount of
data. The recipe was changed afterwards, openly, on the owner's decision (D18, 2026-10-04):
sections 5 and 6 below record what changed and the re-measurement.

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

## 5. With the cosine learning-rate decay (the D18 change), five seeds, identical data

Same data and seeds 1–5 as section 3, but the learning rate decays from 1e-3 to
1e-5 (cosine) over the epochs. Held-out set: the 100,000-pair build's, drawn
before the D17 kick quota (so the numbers are comparable with sections 1–3).

```
lv cosine 100 seed 1 M=1.596e-08 2M=1.576e-08 ratio=1.01
lv cosine 100 seed 2 M=1.981e-08 2M=1.730e-08 ratio=1.15
lv cosine 100 seed 3 M=1.609e-08 2M=1.446e-08 ratio=1.11
lv cosine 100 seed 4 M=1.501e-08 2M=1.313e-08 ratio=1.14
lv cosine 100 seed 5 M=1.776e-08 2M=1.585e-08 ratio=1.12
lv cosine 100 SUMMARY median M 1.609e-08 median 2M 1.576e-08 median-ratio 1.02  spread(max/min) M 1.3 2M 1.3
pendulum cosine 100 seed 1 M=5.425e-08 2M=2.064e-08 ratio=2.63
pendulum cosine 100 seed 2 M=5.301e-08 2M=1.812e-08 ratio=2.93
pendulum cosine 100 seed 3 M=4.963e-08 2M=9.602e-09 ratio=5.17
pendulum cosine 100 seed 4 M=5.377e-08 2M=2.237e-08 ratio=2.40
pendulum cosine 100 seed 5 M=4.872e-08 2M=1.194e-08 ratio=4.08
pendulum cosine 100 SUMMARY median M 5.301e-08 median 2M 1.812e-08 median-ratio 2.93  spread(max/min) M 1.1 2M 2.3
lv cosine 200 seed 1 M=1.486e-08 2M=1.023e-08 ratio=1.45
lv cosine 200 seed 2 M=1.752e-08 2M=7.623e-09 ratio=2.30
lv cosine 200 seed 3 M=1.336e-08 2M=1.091e-08 ratio=1.22
lv cosine 200 seed 4 M=1.286e-08 2M=8.216e-09 ratio=1.57
lv cosine 200 seed 5 M=1.511e-08 2M=7.852e-09 ratio=1.92
lv cosine 200 SUMMARY median M 1.486e-08 median 2M 8.216e-09 median-ratio 1.81  spread(max/min) M 1.4 2M 1.4
pendulum cosine 200 seed 1 M=2.166e-08 2M=4.528e-09 ratio=4.78
pendulum cosine 200 seed 2 M=1.757e-08 2M=7.592e-09 ratio=2.31
pendulum cosine 200 seed 3 M=1.130e-08 2M=1.369e-09 ratio=8.26
pendulum cosine 200 seed 4 M=2.360e-08 2M=8.276e-09 ratio=2.85
pendulum cosine 200 seed 5 M=2.197e-08 2M=1.421e-09 ratio=15.46
pendulum cosine 200 SUMMARY median M 2.166e-08 median 2M 4.528e-09 median-ratio 4.78  spread(max/min) M 2.1 2M 6.0
```

The seed-to-seed spread of the held-out error fell from up to ~20× (constant
rate) to **1.1–1.4×** at 50,000 pairs (LV 1.3×, pendulum 1.1×); the error
itself fell to ~1.6e-8 (LV) and ~5.3e-8 (pendulum) at 100 epochs. 200 epochs lower
the error further (LV 1.5e-8 → ~8e-9 at 100,000 pairs; pendulum 2.2e-8) but make the
M-vs-2M gap larger and cost twice as much; epochs stay at 100.

## 6. The official sufficiency check at the final settings (D17 + D18)

Held-out set with the 1,000-kick quota (D17); median over 5 seeds (D18); cosine
decay; both versions scored on the 100,000-pair build's held-out set. One call
of `check_world(..., direct_factory, recipe, 20260825, "direct")` per world
(≈4–5 min each).

```
lv: median err(M)=1.7054e-08 err(2M)=1.3239e-08 ratio=1.288 sufficient=False seeds=5 time=244s
  errs_m  ['1.497e-08', '1.852e-08', '1.569e-08', '1.705e-08', '2.087e-08']
  errs_2m ['1.324e-08', '1.284e-08', '1.388e-08', '1.272e-08', '1.719e-08']
  split M : KickSplit(n_kick=1000, n_plain=9000, error_kick=2.3840149070454372e-08, error_plain=1.6448906508526735e-08)
  split 2M: KickSplit(n_kick=1000, n_plain=9000, error_kick=1.973885040028899e-08, error_plain=1.2517216847601596e-08)
pendulum: median err(M)=5.6582e-08 err(2M)=2.0578e-08 ratio=2.750 sufficient=False seeds=5 time=280s
  errs_m  ['5.352e-08', '6.069e-08', '5.859e-08', '5.321e-08', '5.658e-08']
  errs_2m ['2.058e-08', '3.125e-08', '1.986e-08', '1.766e-08', '2.104e-08']
  split M : KickSplit(n_kick=1000, n_plain=9000, error_kick=5.121222143571175e-08, error_plain=5.717889435062468e-08)
  split 2M: KickSplit(n_kick=1000, n_plain=9000, error_kick=3.0992576676503744e-08, error_plain=1.963262880210123e-08)
```

Both worlds say **"50,000 is not enough"** under the pre-registered rule
(`err(M) ≤ 1.10 × err(2M)`): LV median ratio 1.29, pendulum 2.75, and every
individual seed agrees in direction. With the held-out set including 1,000 kicked
pairs, the 100,000-pair models are better on kicked and un-kicked pairs alike.
The rule says "if either model fails, `subsample_pairs` becomes 100,000 for
both" — a single failure decides it, so the ensemble's later check cannot change
the outcome. The recipe's revision log records the switch (2026-10-04).
Kick-split report (D17): error on the 1,000 held-out kick pairs vs the 9,000
others — LV 2.4e-8 vs 1.6e-8 (50,000 pairs) and 2.0e-8 vs 1.3e-8 (100,000);
pendulum 5.1e-8 vs 5.7e-8 and 3.1e-8 vs 2.0e-8.
