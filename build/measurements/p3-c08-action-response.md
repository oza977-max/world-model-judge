# P3-C08 — the action-response check, run for real (seed 20260825)

**In plain words:** every model was given 16 starting states and, for each, three
actions from the range the model was trained on (the low end, the middle, the high end).
The check asks whether the model's answer moves when only the action changes. "Moves"
means by more than a billionth of the world's own scale (the tolerance, backlog A26).
The models are the ones trained on the frozen recipe (100 epochs, 100,000 examples).

**The question carried since Round 10: is the smaller predator–prey lever (kicks of at
most 0.1) still enough for the action-blind fixture to be caught?** Yes. The real Model A
changes its one-step answer by up to 5% of the world's scale between the two ends of the
trained action range, on *every one* of the 48 probes — more than seven orders of magnitude
above the tolerance. Its action-blind twin changes by exactly zero and is flagged. (The
check works on a single step, so it does not depend on the lever being large *over a
trajectory* — the 0.12-unit drift at the horizon quoted in the worlds spec is a different
quantity.)

| World | Model | Flagged action-blind | Probes that responded | Largest change (× world scale) |
|---|---|---|---|---|
| LV | direct (Model A) | no | 48/48 | 5.04e-2 |
| LV | ensemble | no | 48/48 | 5.05e-2 |
| LV | fx-overconfident / fx-honest-rough / fx-brittle | no | 48/48 | 5.04e-2 |
| LV | **fx-action-blind** | **yes** | 0/48 | 0 |
| LV | linear (baseline) | yes | 0/48 | 0 |
| LV | persistence (baseline) | yes | 0/48 | 0 |
| Pendulum | direct (Model A) | no | 48/48 | 3.18e-1 |
| Pendulum | ensemble | no | 48/48 | 3.19e-1 |
| Pendulum | fx-overconfident / fx-brittle | no | 48/48 | 3.18e-1 |
| Pendulum | fx-honest-rough | no | 48/48 | 3.19e-1 |
| Pendulum | **fx-action-blind** | **yes** | 0/48 | 0 |
| Pendulum | linear / persistence (baselines) | yes | 0/48 | 0 |

**Cross-check against the world itself:** calling the world's own transition at the low and high ends of the action range gives a one-step change of 5.05e-2 (LV) and 3.18e-1 (pendulum) of scale — Model A's response matches the truth, so it has learned the lever rather than reacting to noise. (The fixture's own flag is nearly tautological — an exactly blind model gives exactly 0; this cross-check is the real evidence.)

**Two baselines are flagged, correctly.** `persistence` ("nothing changes") and `linear`
(extrapolate the last step) take no account of the action *by construction*, so the check
flags them. That is a true statement about them, not a defect to fix and not a discovery;
how the verdict presents a baseline's flag is P6's decision (REMEMBER §3). This is also why
the check cannot be read as "the model is bad": it asks only whether the lever is used.

**What was and was not measured.** One seed, one set of probes; the chaos in training means
the exact numbers move a little from machine to machine, which is why the real-scale gate
(`tests/gates/test_action_response_real_scale.py`) asserts only the machine-independent
facts: Model A responds on all 48 probes by more than 1e5 × the tolerance; the action-blind
fixture's change is exactly zero; the other fixtures built on Model A respond.
