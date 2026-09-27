# Round 10 — proposal under review: the training-recipe revision

**In plain words:** before the two practice models are built, the written
recipe for how they learn has turned out to be unworkable in two ways: it
would take ~14 hours instead of minutes, and it never says how the "kicks"
(the action lever) are chosen. Nothing has been trained and no result
exists, so the recipe can still be honestly changed — but only once, openly,
after an independent check. This document is what that check reviews. The
owner asked that the "10-minute" rule itself be challenged too.

Scope is deliberately narrow (a "quick check"). Out of scope and still owed
(see `REMEMBER.md` §3): the independent re-check of the purity guards and
the ADR-004 orchestration loop (BC-2).

All measurements below were taken 2026-09-27 on this environment,
single-threaded (the project's own thread guard), before any model existed.

---

## R10-1 — The training budget (backlog A12, A14)

**Measured.** One world step ≈ 26–30 µs. One Adam step on the real 2×64 net:
226 µs at batch 32 (7.05 µs per example), 727 µs at batch 256 (2.84 µs per
example). The recipe as committed — 2,000 full-horizon trajectories
(LV 700 steps, pendulum 5,000), 100 epochs, batch 32, six networks per world
(direct + five ensemble members) — needs ≈ **13.7 h** of training. ADR-J6
allows ≤ 6 min of a 600 s run and undercounts the networks ("two small
MLPs"; really 12) and the roster ("× 7 models"; now 8). Generating the
pendulum data one trajectory at a time adds ≈ 263 s.

**Options.**
- **(a) Train once offline, cache the weights, restate NF-2.** Keep the
  full data; the recurring run loads weights and only judges. Needs a
  requirements change (NF-2 is an approved Must) — owner decision.
- **(b) Keep the full data, train on a pinned subsample (recommended).**
  Keep 2,000 full-horizon trajectories per world (the spec's "full
  horizon" stands; memory ≈ 0.4 GB for the pendulum, fine), generated with
  a batched world step. Train each network on a **fixed, seeded subsample
  of M = 50,000 transition pairs** per world, drawn across the whole
  trajectories (so late-trajectory states are covered), at **batch 256**,
  **100 epochs**. Estimated ≈ 15 s per network, ≈ 3 min for all 12 — inside
  the 6-minute training allowance with headroom. The models are one-step
  predictors; 10 M highly correlated pendulum pairs add little the
  subsample lacks. Whether 50,000 pairs is *enough* is an empirical
  question P3-C03 must answer and report (fit vs the MU-5 margin) — the
  review should say whether that is acceptable or M should be set higher.
- **(c) Only raise the batch / vectorise.** Gets ~2.5× — nowhere near 140×.

**Implementation constraint for (b).** A batched world step must produce
**bit-identical** results to the single-state step (WD-3: truth and model
share one integrator; the training-data generator is one of TC-WD3-01's
three generators). Also fix: `lv.transition` given a batch silently returns
a wrong-shaped array instead of refusing.

## R10-2 — Challenge: why "10 minutes" (NF-2, ADR-J6)

**Where it comes from.** The approved requirement (NF-2, Must) says only:
"runs to completion on an ordinary laptop … in minutes rather than hours."
**No purpose is stated.** The 600 s figure was chosen by the tech spec
(judge ADR-J6) to make "minutes" testable, on an estimate now shown wrong.

**What it is plausibly for.** Re-runnability: anyone can re-run the judge on
a laptop and check the verdict for themselves — the essay's "independent
institution" argument in miniature. The review should decide:
1. Is re-runnability NF-2's purpose? If so, write the purpose into NF-2's
   "In plain words" line (owner approves).
2. Does the purpose need the **whole** pipeline in minutes, or only the
   **judging**? Training is seeded and deterministic (TC-MU8-01), so a
   cached, hash-published set of weights can be re-checked separately.
3. Is 600 s the right figure, or is it arbitrary precision on "minutes"?
   (10 vs 20 minutes serves the same purpose.)

**Recommendation.** Keep the whole run in minutes (option R10-1(b) makes
that achievable without splitting verification), state the purpose in
NF-2, and treat 600 s as a revisable engineering target recorded in ADR-J6
with its corrected envelope — not a sacred number.

## R10-3 — How kicks are chosen (backlog A15)

**Measured** (20 random training-region starts per world):

| Style | LV (10 s) | Pendulum (5 s) |
|---|---|---|
| Kick every step (i.i.d.) | 16/20 crash through the prey floor | driven ~13× the world's scale |
| Kick every step, held 0.2 s | 20/20 crash | ~118× scale |
| Occasional kick (~1 per s of world time) | 0 crash; ~0.19× scale effect | 0 crash; ~0.16× scale effect |

**Proposal.** Sparse discrete kicks — matching worlds ADR-W2's own intent
("a discrete intervention: 'remove rabbits now', 'kick the pivot now'").
Each step independently carries a kick with probability `p` (pinned per
world as a rate per second of world time, proposed ≈ 1/s), magnitude drawn
uniformly from the training action interval; otherwise null. One generator
serves training trajectories and evaluation trials (so train and test
actions are alike). **Consequence:** kick transitions are rare (~0.2 % of
pendulum steps), so the R10-1(b) subsample must be **stratified** to include
a pinned share of kick transitions (proposed: all kick pairs up to 25 % of
M), or the models cannot learn the lever and P3-C08's action-blind check
means little. The review should check the rate, the share, and that the
out-of-range action region (WD-5) still gets its own actions outside the
training interval.

## R10-4 — Model A's loss (REMEMBER D1)

Model A (direct) trains a variance head with plain Gaussian NLL. Seitzer et
al. (ICLR 2022, arXiv 2203.09168) show NLL down-weights the squared error
where predicted variance is high, degrading the mean fit. If kept, and the
ensemble wins MU-5, the result is open to "you picked a known-weak Model A".
**Proposal:** β-NLL with β = 0.5 (Seitzer et al.'s recommended setting),
loss = stop-gradient(σ^(2β)) × NLL. Model A remains a self-predicted-error-bar
model (Nix & Weigend family), so MU-5's contrast (self-predicted vs ensemble
disagreement) is unchanged. The prediction in `prereg/prediction.md` stays as
written. The review should check this doesn't tilt MU-5 either way.

## R10-5 — The freeze point (REMEMBER D2)

`check_prereg` compares each prereg file to its **first** commit, so the
spec's own remedy ("revise the recipe openly and re-run") makes the
judged run fail certification forever. **Proposal:** pre-registration is
frozen at a **declared freeze commit** — a signed-off, dated commit
recorded in `prereg/FREEZE` (and a `prereg-freeze` git tag) made before the
first judged run. `check_prereg` compares working-tree content to the blob
**at the freeze commit** and requires the freeze commit to precede the run;
earlier recipe versions stay visible in history and are listed openly in
`prereg/recipe.md`'s revision log. A second freeze after any judged run is
refused. The review should check this re-opens no gaming route that
TC-MU6-04 / the `--reverse` pin closed (e.g. re-freezing after results).

## R10-6 — Owed ratifications (short)

- **A10:** gradient-check metric floor 1e-3 and tolerance 1e-5 (spec 1e-6),
  including the disclosed near-dead-unit blind spot.
- **A12:** `sharpness_hedge_threshold = scale`, per world (not per region —
  a DR-008 minor).
- **A13:** add "entry substance not verifiable" to ADR-M5's disclosed
  residuals.
- **B1 text:** MU-3's fourth fixture / TC-MU3-04 / P3-C08 entered the
  requirements after Round 9; its route promised a design review.
- **Wiring gaps found 2026-09-27:** TC-MU6-05(b) owned by no chunk (propose
  P4-C02); recording `check_prereg`'s commit SHA in verdict metadata owned by
  no chunk (propose P6-C01); the Round-9 P4-C05 climatology/drift note
  missing from the guide.

## What happens after the review

The owner triages the findings. Accepted changes go into the specs and
requirements (NF-2's wording needs the owner's approval), then **one** open,
dated revision of `prereg/recipe.md` carrying R10-1, R10-3 and R10-4,
followed by the freeze commit mechanism (R10-5) when P6-C03 approaches.
Then P3-C06 is built against the revised recipe.
