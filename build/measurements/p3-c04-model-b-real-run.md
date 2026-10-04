# P3-C04 measurements — Model B ("ensemble"), real recipe, 2026-10-04

**In plain words:** what happened when the second practice model — five networks
whose disagreement is the error bar — was trained for real on the frozen recipe
(100,000 training pairs per world, 100 epochs, cosine learning-rate decay), next
to Model A. Headline: **the ensemble is about as accurate as Model A (in fact
slightly more accurate), but its error bars are far too small** — exactly the
"equally accurate, differently honest about uncertainty" pair the experiment
needs. Nothing was tuned to produce this.

Setup: seed 20260825; `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`; data exactly as
`build_training_data` makes it from `prereg/recipe.md` (including the held-out
kick quota); the held-out set is the 10,000 pairs never trained on, 1,000 of them
kicked. "Error" is the ADR-M3 held-out error (mean of the scaled squared error).
"z" is (truth − prediction) / predicted error bar: a calibrated model has z with
standard deviation ≈ 1 and ≈ 68% of |z| below 1.

## 1. Accuracy, error bars and time — ensemble vs Model A, one run each

```
lv ensemble: train 142s  held-out err 5.106e-09  kick 6.346e-09 plain 4.968e-09
   spread median per dim [7.24216253e-05 4.67244780e-05]  z std per dim [3.76264932 2.28367609]  z mean [-0.18235422  0.20626919]  coverage |z|<1: [0.2242 0.2859]
lv direct: train 32s  held-out err 1.324e-08  kick 1.974e-08 plain 1.252e-08
   spread median per dim [0.00034416 0.00024134]  z std per dim [0.84998935 0.89827151]  z mean [-0.05179489  0.02591444]  coverage |z|<1: [0.8238 0.8395]
   persistence err 4.979e-05
pendulum ensemble: train 148s  held-out err 8.910e-09  kick 1.043e-08 plain 8.741e-09
   spread median per dim [1.63272930e-05 1.22853836e-05 1.43268815e-04 2.26956671e-04]  z std per dim [0.51398172 0.52417924 1.79258865 1.88928588]  z mean [-0.01215392 -0.02714177 -0.12152551  0.02352528]  coverage |z|<1: [0.9533 0.9423 0.4556 0.4837]
pendulum direct: train 34s  held-out err 2.058e-08  kick 2.909e-08 plain 1.963e-08
   spread median per dim [5.70345766e-05 4.69023258e-05 9.04041152e-04 1.34757032e-03]  z std per dim [0.80427424 0.86429355 0.95152745 0.97810994]  z mean [ 0.05383921 -0.08493661  0.03540752 -0.07399304]  coverage |z|<1: [0.9005 0.8879 0.8686 0.8455]
   persistence err 2.085e-04
```

Reading it: both models beat "nothing changes" by three to four orders of magnitude
(persistence error 5.0e-5 on predator–prey, 2.1e-4 on the pendulum). The ensemble's
held-out error is **lower than Model A's** (5.1e-9 vs 1.3e-8; 8.9e-9 vs 2.1e-8) on
both kicked and un-kicked pairs. Skill against persistence is ≈ 0.9999 for both, so
the pre-registered accuracy-matching margin (0.05 in skill) is met with huge room.

Error bars: **Model A is well calibrated** (z std 0.85–0.98, 82–90% inside one error
bar). **The ensemble is badly over-confident on predator–prey** (z std 3.8 and 2.3;
only 22–29% inside one bar) **and on the pendulum's angular speeds** (z std 1.8–1.9,
46–48% inside), while slightly over-cautious on the pendulum's angles (z std 0.5,
95% inside). Five networks trained the same way agree with each other far more than
they agree with the truth — the known failure of small ensembles, here visible
without any deliberate sabotage. This is the contrast the judge is built to measure.

**Runtime (matters for NF-2).** Training takes **≈ 145–150 s per world for the
ensemble** (5 members × 100 epochs × 100,000 pairs) and **≈ 33 s per world for
Model A**: about 6 minutes of training for both models on both worlds, against an NF-2
target of 600 s for the whole run (which also has to evaluate ≈ 5–6 min by the
earlier estimate). The sum exceeds the target: **P6-C01 must re-measure end to end and
NF-2's number must be revised openly if it still does not fit** (the requirement's
purpose is re-runnability by anyone, REMEMBER.md).

## 2. The ensemble's own sufficiency check at the original comparison (50,000 vs 100,000)

Run for completeness at the pre-registered M (not at the post-fallback 100,000 vs
200,000): median over 5 seeds, cosine decay, held-out set with the kick quota.

```
lv: median err(50,000)=1.8486e-08 err(100,000)=4.0745e-09 ratio=4.537 sufficient=False seeds=5 time=1079s
  errs_m  ['1.878e-08', '1.432e-08', '2.222e-08', '1.646e-08', '1.849e-08']
  errs_2m ['5.106e-09', '3.362e-09', '5.357e-09', '4.074e-09', '3.606e-09']
  split M : KickSplit(n_kick=1000, n_plain=9000, error_kick=1.890213158482515e-08, error_plain=1.843296262806768e-08)
  split 2M: KickSplit(n_kick=1000, n_plain=9000, error_kick=5.372084607628291e-09, error_plain=3.970016478800977e-09)
pendulum: median err(50,000)=4.2311e-08 err(100,000)=8.3297e-09 ratio=5.079 sufficient=False seeds=5 time=1135s
  errs_m  ['5.060e-08', '3.954e-08', '4.553e-08', '4.231e-08', '4.008e-08']
  errs_2m ['8.910e-09', '7.395e-09', '9.588e-09', '8.330e-09', '7.774e-09']
  split M : KickSplit(n_kick=1000, n_plain=9000, error_kick=4.3868108159888514e-08, error_plain=4.216092481066408e-08)
  split 2M: KickSplit(n_kick=1000, n_plain=9000, error_kick=1.1104182222289037e-08, error_plain=8.02147447635121e-09)
```

The ensemble also says **"50,000 is not enough"** (ratios 4.5 and 5.1; every seed
agrees), the same direction as Model A (1.29 and 2.75). The fallback to 100,000 had
already been enacted by Model A's failure; this result confirms it and cannot change
it. About 18–19 minutes per world.
