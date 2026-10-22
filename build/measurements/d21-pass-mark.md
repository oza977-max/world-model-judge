# D21 — the climatology-agreement pass mark, measured from a perfect model

**In plain words:** the judge's climatology-agreement test asks, for long horizons, "do this model's
forecasts still look like the kind of states the real world visits?" It scores a forecast by how
many typical spreads it sits from the typical state for its energy level (mean |z|). The pass mark
was set at 1.0 by reasoning, and when we measured it, the *true world itself* failed it in the one
place the test runs. A test that fails the truth is broken. The owner chose (2026-10-22) to set the
pass mark from what a perfect model actually scores, plus stated headroom, and then to check that
the test still catches forecasts that are plainly wrong. **This file was committed with the recipe
below before anything was measured**, so the recipe cannot have been chosen to fit the numbers.
No model is involved at any point: the "perfect model" is the true world, and the bad forecasts
are built from the truth alone.

## The recipe (fixed before measuring — owner decisions D21, 2026-10-22)

1. **Cells.** Every (world, task, region) whose switch step exists. The switch step comes from the
   world's divergence curve exactly as the real run builds it (`build_divergence_artefact`, run seed
   20260825). The judge scores agreement only in those cells. A world with no such cell keeps the
   mark 1.0, recorded as inert, because the judge never uses it there.
2. **Tables.** One table per region, built from that region's own starts: 50 starts × 20,000
   null-action steps, 16 equal-population bins (`build_climatology_table(..., region_name=region,
   n_starts=50, steps_per_start=20000)`). There are 20 table seeds: run seeds 1 to 20.
3. **Perfect-model trials.** There are 5 trial sets: run seeds 101 to 105, purpose key
   `"d21-starts"`. Each set holds 200 starts (the real trial count per region) drawn uniformly in the
   region's box. Each start is kicked with the evaluation kick generator (`benchmark_kicks`) and run for
   the task's horizon. The bins are measured from the true state at every step
   (`measure_invariant_bins`).
4. **Score.** Mean |state − bin mean| / bin sd over trials, quantities and steps from the switch step
   to the task's horizon. This is the judge's own window and arithmetic, and it is scale-free, so it
   is the same in raw and in normalised units. That gives 20 × 5 = 100 perfect-model scores per cell.
5. **Mark.** For each world, take the largest of the 100 scores over every cell where the test runs,
   multiply by 1.10, and round **up** to two decimal places. That number replaces 1.0 for the world
   in `prereg/thresholds.json`, as a dated pre-freeze revision.
6. **Bad-forecast check (afterwards, and never used to move the mark).** On the same 100 combinations,
   two model-free bad forecasts are scored against the truth's bins:
   (a) **persistence**, which repeats the start state at every step;
   (b) **energy-inflated truth**, which is the true path with each state's angular velocities scaled so
   that its energy is the true energy + 10% of |true energy|. Where that is impossible (no kinetic
   energy to scale), the true state is kept.
   We report whether each lands above the mark. If one does not, the report says the test cannot
   tell that bad forecast from the truth in that cell. The mark is not changed to make it work.

## Results (measured 2026-10-22, after the recipe above was committed in 4bd2728)

Run: `PYTHONPATH=src:. OMP_NUM_THREADS=1 python build/measurements/d21-reproduce.py` (169 s). The script
checks that its shortcut bin lookup and its score equal the harness's own `measure_invariant_bins` and
`truth_mean_abs_z` exactly on the first combination.

**Cells.** The test runs in exactly one cell: pendulum, `out-near-inverted`, `dp-control`, switch step
4,750, window 4,750–5,000. Predator–prey has no switch step in any region or task, so its mark stays
1.0 and is inert.

**The perfect model (the true world), 100 scores** (20 tables × 5 trial sets of 200 kicked trials):

| | min | median | 95th percentile | max |
|---|---|---|---|---|
| true world | 0.821 | 1.049 | 1.371 | 1.479 |

**The mark, by the pinned rule:** 1.10 × 1.4785 = 1.6263, rounded up to **1.63**.

**The bad-forecast check (step 6), on the same 100 combinations:**

| bad forecast | min | median | max | above 1.63? |
|---|---|---|---|---|
| persistence (start state held) | 0.323 | 0.459 | 0.796 | never |
| true path, energy +10% (checked: exactly +10% of \|E\|) | 0.839 | 1.068 | 1.497 | never |

**In plain words — what this shows:** the measured mark does what it was asked to: the true world never
fails it. But neither plainly wrong forecast fails it either. A forecast that freezes the pendulum at its
start scores *better* than the truth (0.46 against 1.05), and one with 10% too much energy is
indistinguishable from the truth (1.07 against 1.05). The statistic, mean |state − typical state for
this energy| in typical spreads, rewards forecasts that sit near the middle of the energy level's spread
of states. A frozen state is a perfectly ordinary state of its energy, so it is not caught. And the
spread of states at one energy level is so wide that a 10% energy error moves the score by less than the
table-to-table noise. **As specified, the test cannot tell these bad forecasts from the truth in the one
place it runs.** Following the recipe, the mark has not been changed to make it work; the finding goes
to the owner.
