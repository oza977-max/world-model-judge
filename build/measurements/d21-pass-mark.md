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

## Results

(Measured after this recipe was committed; see below.)
