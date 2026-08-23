# World Model Judge

**A checker that decides how far ahead a learned simulator can be believed.**

> **Status: requirements only. No code has been written yet.**

---

## The problem, plainly

Imagine a machine that has never been told how foxes and rabbits work, but has
watched them for a while and now tries to guess what happens next — including
what happens *if you interfere*, say by removing some rabbits.

That is a miniature version of what the AI field calls a **world model**:
something that predicts what the world does next, rather than what word comes
next in a sentence.

It is easy to check whether one guess was right. It is much harder to check what
happens when the machine feeds its own guesses back into itself — guessing
tomorrow from today's guess, then the day after from tomorrow's — because small
errors pile up. Ten steps out it can be confidently describing a world that does
not exist.

**This project builds the checker, not the guessing machine.**

## The four questions the judge asks

1. Is this machine better than doing nothing at all?
2. How fast does it go wrong as it predicts further ahead — and is that worse
   than the world's own natural unpredictability?
3. When it says it is confident, is it actually right that often?
4. At what point should you stop believing it — and stop believing it *for which
   purpose*?

## Where the method comes from

Banks have asked these questions about their own models for decades, because
confident models once lost them enormous amounts of money. They have independent
teams who check, they count the times a model was wrong when it claimed to be
sure, and they have limits that force action when the count gets too high.

Weather forecasters built the measuring techniques — never scoring a forecast
against perfection, always against a reference; separating whether a stated
confidence was honest from whether it was useful; and switching from grading
individual forecasts to grading statistics once prediction becomes impossible.

Neither discipline has been pointed at world models. That gap is what this fills,
at the smallest honest scale we could find.

## What it will and will not do

**Will:** grade simple simulators of two toy systems — predator-prey populations
and a double pendulum — where the true answer is always known, and issue a
verdict stating how far ahead each can be trusted, for which task, against which
baseline, and with an explicit list of what was never tested.

**Will not:** build or improve a world model, touch video or robotics data, or
say anything about whether any real world model is trustworthy. The models being
judged here are deliberately trivial, because the judging is the scarce thing.

## Honest limits

- A toy world validates the checker. It proves nothing about the field.
- This kind of checking is strongest on ordinary cases and weakest on rare
  disasters — which is exactly where the damage happens.
- The judge is plain arithmetic. Its only judgement calls are where its
  thresholds sit, and those are written down before any results are seen.

## Repository

```
requirements/   45 requirements, each with a plain-English restatement
risks/          the four ways this project could fail, assessed up front
CLAUDE.md       project context and standing rules
HANDOVER.md     current state and what comes next
```

Built with the Grounded Vibe Methodology.
