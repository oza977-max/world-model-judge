"""wmj.judge.limitations — what this judge says about itself, word for word (JU-10, ADR-J7).

In plain words: a verdict that only reports good news is a sales pitch. Every verdict
this judge issues carries the same seven plain sentences about what it cannot do —
that it validates the middle of the distribution and not the extremes, that a toy
world validates the harness and not a real model, that the judge's author also wrote the
models (blind is not independent), that it borrows banking's backtesting but not its
power to force a stop, that its thresholds are modelling decisions, that the worlds hold
no genuine surprises, and that "control" and "planning" grade passive prediction, not
decisions. They are fixed text: no formatting, no parameters, no softening per run.
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
)

NOT_TESTED: tuple[str, ...] = (
    "genuine off-model surprises (WD-8)",
    "closed-loop action selection",
    "ongoing monitoring",
    "the seven JU-10 disclosures cover the rest of this list",
)
