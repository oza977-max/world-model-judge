"""wmj.judge.limitations — what this judge says about itself, word for word (JU-10, ADR-J7).

In plain words: a verdict that only reports good news is a sales pitch. Every verdict
this judge issues carries the same ten plain sentences about what it cannot do —
that it validates the middle of the distribution and not the extremes, that a toy
world validates the harness and not a real model, that the judge's author also wrote the
models (blind is not independent), that it borrows banking's backtesting but not its
power to force a stop, that its thresholds are modelling decisions, that the worlds hold
no genuine surprises, that "control" and "planning" grade passive prediction, not
decisions, that the 90% region reads each model's stated spreads as independent, and that the author
looked at both practice models' held-out calibration before the judged run (owner decision D19), and
that the climatology-agreement test is weak (owner decision D21). The last three were added after
code-review-003, its fix-check and the D21 measurement (backlog A39). They are fixed text: no formatting, no parameters, no softening per run.
A verdict may add detail elsewhere, never fewer or reworded versions of these.
"""

from __future__ import annotations

JU10_DISCLOSURES: tuple[str, ...] = (
    "This method validates the middle of the distribution, not the extremes — it says nothing about how the model behaves on rare, disaster-scale events.",
    "This toy world validates the judging harness, not any real-world model — a clean result here is not evidence that a real world model is trustworthy.",
    "The judge, the models, and the thresholds share a single author. The blinding in JU-1 means the judge cannot favour a model for its identity — it does not mean this evaluation has banking's organisational independence: a separate team, reporting separately, empowered to challenge the model builder.",
    "This project borrows banking's backtesting and ex-ante-threshold practices, not its ongoing monitoring or its power to force a stop. Deciding who has the authority to act on a bad verdict is exactly the gap this project's source essay names — it is not solved here.",
    "The judge's own thresholds and metric choices are modelling decisions. They were fixed in advance and are not arbitrary, but they are not beyond challenge either.",
    "The toy worlds contain no genuine off-model surprises by construction (no randomness, no hidden state, no high-dimensional input) — this judge's behaviour when a model meets a real surprise is untested.",
    "'Control' and 'planning' name the tolerance regime used to grade a passive prediction, not a closed-loop decision. This judge does not test whether a model can choose its own actions to reach a goal — the actions it is graded against were chosen by someone else.",
    "The 90% region takes each model at its word: its stated spreads are read as independent of one another, because the uncertainty format has no way to say how errors move together. A model whose errors are correlated misses that region more often than one time in ten even when each quantity's spread is honest, and so reads as somewhat overconfident; the per-quantity coverage is reported alongside so a reader can see it.",
    "Before the judged run, the author looked at both practice models' calibration on held-out training pairs (one step, inside the training region): the self-predicted error bar was close to honest and the ensemble's too narrow, the opposite of the pre-registered prediction. The prediction was deliberately left unchanged, and the look is recorded in prereg/prediction.md.",
    "The climatology-agreement test's pass mark was set from the true world's own measured scores, because the reference table is itself sampled and even the truth scores near, not at, the ideal. Measured before any model was judged, the test could not tell a forecast frozen at its start, or one with 10% too much energy, from the truth: it catches only forecasts that leave the world's typical states altogether.",
)

NOT_TESTED: tuple[str, ...] = (
    "genuine off-model surprises (WD-8)",
    "closed-loop action selection",
    "ongoing monitoring",
    "the JU-10 disclosures cover the rest of this list",
)
