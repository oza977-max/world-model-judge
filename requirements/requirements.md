# World Model Judge — Requirements

**A governance harness for learned simulators, at toy scale.**

Version 1.1 · 23 August 2026 · Derived from the essay *Words Are a Menu. The World Is Not.* (draft v2.2)

> **Change note (v1.1).** The essay was revised to draft v2.2 after its first reader
> feedback: simpler words, same argument. No requirement changed its meaning. One
> alignment worth recording: draft v2.1 called the double pendulum a "stretch goal";
> the requirements had already committed to it (WD-1), and draft v2.2 now commits to
> it too, so the essay and the requirements finally promise the same thing.

---

## How to read this document

Every numbered item below is a **requirement** — one thing the finished project must do. Each one has three parts:

- **The requirement itself**, written precisely, because later stages of the build read these and turn them into tests.
- **In plain words** — the same thing said simply. If the precise version and the plain version ever disagree, the plain version is what was meant.
- **Why** — a short note explaining the reasoning, usually naming the person or field the idea came from.

Each requirement is tagged **Must**, **Should**, or **Won't**:

- **Must** — if this is missing, the project has failed at its purpose.
- **Should** — genuinely wanted, but the project still works without it.
- **Won't** — deliberately excluded. Written down so the exclusion is a decision, not an oversight.

---

## What we are building, in one page

Imagine a machine that has never been told how foxes and rabbits work, but has watched them for a while and now tries to guess what happens next — including what happens *if you interfere*, say by removing some rabbits.

That guessing machine is a miniature version of what the AI field calls a **world model**: something that predicts what the world does next, rather than predicting the next word in a sentence.

Now the important part. It is easy to check whether one guess was right. It is much harder to check what happens when the machine feeds its own guesses back into itself — guessing tomorrow from today's guess, then the day after from tomorrow's guess — because small errors pile up. Ten steps out it can be confidently describing a world that does not exist.

**This project builds the checker, not the guessing machine.** The checker — we call it the judge — asks four questions:

1. Is this machine better than doing nothing at all?
2. How fast does it go wrong as it predicts further ahead — and is that worse than the world's own natural unpredictability?
3. When it says it is confident, is it actually right that often?
4. At what point should you stop believing it — and stop believing it *for which purpose*?

Banks have asked exactly these questions about their own models for decades, because confident models once lost them enormous amounts of money. They have independent teams who check, they count the times a model was wrong when it claimed to be sure, and they have limits that force action when the count gets too high. Weather forecasters built the measuring techniques. Nobody has pointed either discipline at world models.

That gap is what this project fills, at the smallest honest scale we could find.

---

## Expert Panel

The requirements below are grounded in published work rather than invented. This table records whose ideas shaped which parts.

| Expert | Work | What they contribute here |
|--------|------|---------------------------|
| Karl Wiegers | *Software Requirements* (3rd ed.) | How to write a requirement so it can actually be tested |
| Gause & Weinberg | *Exploring Requirements* | Forcing vague words to become specific ones |
| Alan Cooper | *About Face* (4th ed.) | Deciding who the verdict is written for |
| Clayton Christensen | *Competing Against Luck* | Framing what job this project is hired to do |
| Robertson & Robertson | *Mastering the Requirements Process* | Making "it must be reproducible" into something measurable |
| Federal Reserve / OCC | *SR 11-7* | The governance shape being copied: independent checking, ongoing monitoring, and the phrase "effective challenge" |
| Prudential Regulation Authority | *SS1/23* | Stating in advance how much model risk is acceptable, rather than reacting later |
| Emanuel Derman | *Models.Behaving.Badly* | A model is an analogy, not the truth; the dangerous simplifications are the ones you didn't notice |
| Riccardo Rebonato | *Plight of the Fortune Tellers* | The objection to our own method: this kind of checking tests the ordinary cases, not the disasters |
| Nassim Nicholas Taleb | *The Black Swan* | The risks that matter most are the ones the model doesn't represent at all |
| Tilmann Gneiting & Adrian Raftery | *Strictly Proper Scoring Rules* (JASA 2007) | How to score a confident prediction so the model can't cheat by lying about its confidence |
| Gneiting, Balabdaoui & Raftery | *Probabilistic forecasts, calibration and sharpness* | Being honest isn't enough — a useless "anything could happen" forecast is perfectly honest |
| Allan Murphy | *What is a good forecast?* (1993) | Forecast quality is multi-dimensional and depends on who's using it; one number won't do |
| Edward Lorenz | *Deterministic Nonperiodic Flow* (1963) | Some error is the world's fault, not the model's — grade against that, not against perfection |
| A. Philip Dawid | The prequential principle (1984) | Judge a forecaster only on what it predicted and what happened, never on how it works inside |
| Jolliffe & Stephenson | *Forecast Verification* | The practitioner's catalogue of verification measures and their traps |
| Deborah Raji | *Closing the AI Accountability Gap* (2020) | The closest existing relative to this thesis — auditing as an institution, not a metric |
| Margaret Mitchell et al. | *Model Cards for Model Reporting* (2019) | Stating a model's intended limits matters more than stating its score |
| Percy Liang et al. | *HELM* | A single headline number misleads; report coverage honestly |
| Kapoor & Narayanan | *Leakage and the Reproducibility Crisis* | How evaluations quietly cheat by testing on training data |
| Zheng et al. | *Judging LLM-as-a-Judge* (2023) | Automated judges have their own documented biases — a reason ours is plain arithmetic |
| David Ha & Jürgen Schmidhuber | *World Models* (2018) | The paper that fixed the term |
| Yann LeCun | *A Path Towards Autonomous Machine Intelligence* (2022) | The predict-in-abstract-space position |
| Danijar Hafner et al. | The Dreamer line | Measured how prediction error grows with rollout length — prior art for the core measurement |
| Sutton & Barto | *Reinforcement Learning: An Introduction* | Why practitioners keep imagined rollouts short |
| Daniel Wolpert | Internal models for motor control | The brain runs forward models predicting the consequences of its own actions |
| Elizabeth Spelke | Core knowledge | Infants arrive expecting objects to persist — the pre-wiring case |

> **Honesty note on this table.** These experts were scored against the methodology's rubric in a single pass, not the full two-reviewer cross-check. The scores are good enough to ground design decisions and not strong enough to cite as settled authority rankings.

---

## Purpose & Vision

**The job statement.** *When I have a learned simulator and someone is about to make a decision based on what it predicts, I want an independent, repeatable check on how far ahead it can be believed, so I can say plainly where trust ends instead of guessing.*

The essay this comes from makes one narrow claim: the measurements exist, scattered across research papers and separate fields, but the **institution** does not. No independent body owns the number, has to report it, or can force a stop when it crosses a line.

This project does not try to build a world model, invent a new measurement, or advance the state of the art. It builds the smallest honest version of the judge.

**In plain words:** the models here are deliberately simple, because the checking is the part nobody has built.

> **Scope honesty (Derman).** A toy world proves the checker works. It proves nothing about whether any real world model is trustworthy. Every requirement below treats that limit as something to state clearly, not something to hide.

---

## Target User

**Dev, the essay's reader.** Works somewhere near AI — engineer, researcher, or risk professional. Read the essay, followed the link, has about ten minutes, will look at one or two charts and will not read the source code. Wants to leave able to explain the idea to someone else. Hates projects that hide a thin result behind heavy machinery.

**Mor, the validator.** Has a learned simulator of their own and wants to know whether this checker could be pointed at it. Will read the model interface and the verdict format, and will judge the project on whether adapting it looks like an afternoon or a rewrite.

Every requirement traces to one of two goals: **Dev must understand the verdict without a risk background**, and **Mor must be able to adapt it without rewriting the judge.**

---

## Domain 1 — The Worlds

We need pretend worlds where we already know every correct answer, so "was the model right" is never a matter of opinion.

Two worlds are in scope, chosen because they differ in the one property that matters most here: how quickly two almost-identical starting points drift apart.

**Foxes and rabbits** (the Lotka–Volterra equations) is predictable and repeating — populations rise and fall in a steady cycle, and two near-identical starts stay close for a long time. **A double pendulum** — a pendulum with a second one swinging off its end — is chaotic. Release it twice from almost the same position and within seconds the two swings look nothing alike.

Having both is what forces the judge to tell apart *the model is wrong* from *the world is genuinely unpredictable*. It also stops the judge being secretly built around one world.

**WD-1 (Must):** The harness shall provide two reference worlds — a Lotka–Volterra predator–prey system and a double pendulum — each computing exact ground-truth trajectories from known equations.

**In plain words:** build two toy worlds where we can always work out the true answer.

> Both are standard textbook systems. Neither is novel and neither is meant to be.

**WD-2 (Must):** Every world shall expose a single common interface in which a transition takes a state and an action and returns the next state.

**In plain words:** every world has a lever you can pull, so we're always asking "what happens *if I do this*", not just "what happens next".

> Without the lever this is ordinary forecasting, which plenty of tools already check. The lever is what makes it a world model at all.
>
> For foxes and rabbits, the lever removes or adds animals. For the pendulum, it's a push at the pivot.

**WD-3 (Must):** Ground truth and every model under test shall be advanced using the same numerical integrator and the same step size, enforced by an automated test rather than by convention.

**In plain words:** the true world and the model being tested must use the same maths engine and the same size time-step — and a test has to check it, not a promise.

> This is the trap named in the essay. If the two differ, every chart secretly measures the difference between two maths engines instead of the quality of the model. The results still look completely plausible, which is exactly why a written convention isn't enough protection.

**WD-4 (Must):** Every world shall report its own intrinsic divergence rate — how quickly two trajectories starting a small distance apart separate under exact dynamics.

**In plain words:** each world measures how fast it drifts away from itself, so we know how much error is nobody's fault.

> Lorenz's argument: past a certain point even a perfect model loses the trajectory, because the world amplifies tiny differences. Grading against zero error would condemn a flawless model. This number is the fair yardstick.

**WD-5 (Must):** Every world shall declare a training region and at least one starting region outside it.

**In plain words:** each world marks out a "familiar" area and an "unfamiliar" area, so we can test whether a model admits it's out of its depth.

> "Does it know when it doesn't know" can only be answered somewhere the model has never been. Without a declared outside, we'd only ever test the easy case.

**WD-6 (Must):** Every world shall declare at least two evaluation tasks with different error tolerances: a tight-tolerance control task and a loose-tolerance planning task.

**In plain words:** each world names two jobs — a fussy one and a rough one — because "trustworthy for 14 steps" is meaningless until you say what for.

> Foxes and rabbits: hold the rabbit population inside a target band by working the lever (fussy), versus will the rabbits crash below a danger line within N steps (rough). Pendulum: hold the tip near a target angle (fussy), versus will it flip over within N steps (rough).
>
> Tasks belong to the world, not to the judge. That keeps the judge general, and makes it impossible to quote a trust number without naming its job.

**WD-7 (Must):** Given the same seed, every world shall produce byte-identical trajectories across runs and across processes.

**In plain words:** run it twice with the same starting settings and you get exactly the same numbers, down to the last digit.

**WD-8 (Won't):** Worlds with randomness, hidden state, or high-dimensional input such as video or images are out of scope for this round.

**In plain words:** no randomness, nothing hidden, nothing picture-shaped.

> All three are real features of the systems the essay is ultimately about. All three would also make "the true answer" ambiguous — which is precisely what this round trades away in return for a checker we can actually verify.

---

## Domain 2 — The Models Under Test

The models here are deliberately unimpressive. Their only job is to give the judge something to grade.

Two distinctions carry most of this project's honesty. First, the difference between a model that is **wrong** and one that is **overconfident** — the second is far more dangerous, and ordinary accuracy scores cannot see it at all. Second, the difference between a **test fixture** and a **finding**: a failure we deliberately built and then detected is a passing test, not a discovery.

**MU-1 (Must):** All models under test shall implement one interface: given a state and an action, return a predicted next state together with a stated uncertainty.

**In plain words:** every model plugs into the same socket, and must say not just what it predicts but how sure it is.

> The confidence is not decoration. A model that only ever says "12.4", and never "12.4, give or take 3", cannot be asked the one question this whole project exists to ask.

**MU-2 (Must):** The harness shall provide two reference baselines — persistence and linear extrapolation — and no verdict shall be issued without comparing the model against them.

**In plain words:** every model has to be measured against two stupid ones — "nothing changes" and "carry on in a straight line" — and no verdict is allowed without that comparison.

> Taken straight from weather forecasting: a score with nothing to compare it to is uninterpretable. A model that can't beat "nothing changes" hasn't earned anything, no matter how small its error looks.

**MU-3 (Must):** The harness shall provide fixture models failing in specified ways, covering at minimum: accurate but overconfident; less accurate but honestly uncertain; and accurate inside the training region but catastrophically wrong outside it.

**In plain words:** we build three broken models on purpose — one that's right but cocky, one that's rougher but honest, and one that's excellent at home and disastrous away from home.

> These are the judge's own test suite. The cocky one matters most: it's the case where ranking models by average error actively misleads you.

**MU-4 (Must):** Fixture models shall be labelled as test fixtures wherever they appear — in code, documentation, and any published output — and never presented as findings.

**In plain words:** the deliberately broken models get labelled as test equipment everywhere, never dressed up as discoveries.

> Catching a fault you planted yourself proves your instrument works. It proves nothing about the world. Blurring those two would be the fastest way to lose the credibility this project is trying to build.

**MU-5 (Must):** The harness shall include at least two models whose behaviour is not engineered, matched as closely as practicable on accuracy and differing instead in how they derive their stated uncertainty: at minimum, one network predicting its own error bar directly, and one small ensemble whose disagreement supplies the error bar.

**In plain words:** at least two genuine models that we haven't rigged — roughly as accurate as each other, but working out their confidence in different ways. One guesses its own error bar; the other is a small group of models whose disagreement *is* the error bar.

> Varying accuracy proves nothing — everyone accepts a worse model is worse. Varying only the confidence machinery is what makes the interesting comparison possible: ordinary error scores can't separate these two, and the judge is designed to.

**MU-6 (Must):** The architecture, training recipe, stopping rule, and an explicit written prediction of how the unrigged models will rank shall be recorded in the repository before the judge is first run against them; the resulting verdicts shall be published unchanged whether or not the prediction holds.

**In plain words:** write down how the honest models are built *and* which one we expect to win — before running the judge. Then publish the result either way.

> This is the only place a genuine surprise is possible, and writing it down first is what protects it. A model tuned until its verdict looks good is a rigged model in disguise, and a prediction written afterwards isn't a prediction. Being wrong here is a better result than being right.

**MU-7 (Must):** Models shall be trained and evaluated on disjoint data.

**In plain words:** no model is ever tested on the data it learned from.

> Kapoor and Narayanan document this kind of leakage as the biggest cause of overstated results across machine-learning research. A judge whose subject was trained on the test set measures nothing at all.

**MU-8 (Must):** Model training shall be seeded and reproducible, such that re-training from the same seed produces the same model and therefore the same verdict.

**In plain words:** training the model again from the same starting point gives you the identical model, and so the identical verdict.

**MU-9 (Should):** Adding a new model shall require implementing the MU-1 interface and nothing else — no changes to the judge, the worlds, or the reporting layer.

**In plain words:** plugging in someone else's model should mean writing one small adapter and touching nothing else.

> This is what turns "reusable checker" from a claim into something the code can be held to — and it's what Mor will judge the project on.

**MU-10 (Won't):** Large pretrained models, foundation models, and anything requiring specialised hardware are out of scope.

**In plain words:** no big AI models. We make no attempt to judge a serious world model.

---

## Domain 3 — The Judge

This is the product. Everything else exists to give it something to grade.

The judge takes what a model predicted and what actually happened, and returns a verdict: how far ahead this model can be believed, for which job, measured against what, and with an explicit list of what was never tested. It never looks inside the model.

**JU-1 (Must):** The judge shall receive only a model's stated predictions and uncertainties, the outcomes that followed, and the world's declared divergence rate and tasks. It shall not receive the model's identity, internals, architecture, or training history.

**In plain words:** the judge only ever sees what the model predicted and what actually happened. It is never told whose model it is or how it works inside.

> Dawid's prequential principle. It's also what makes the judge independent in the banking sense — someone objective checking the work, rather than the builder marking their own homework.

**JU-2 (Must):** The judge shall report one-step accuracy as a skill score relative to the persistence and linear baselines, never as an absolute error alone.

**In plain words:** never report "the error was 0.03". Report "it beat doing-nothing by this much".

> Murphy's framework: 0.03 means nothing until you know that doing nothing scores 0.04.

**JU-3 (Must):** The judge shall report prediction error as a function of rollout length, benchmarked against the world's own divergence rate rather than against zero.

**In plain words:** show how wrong it gets the further ahead it predicts — and compare that against how fast the world itself becomes unpredictable, not against perfection.

> The question is never "is there error". It's "is there more error than the world itself produces".

**JU-4 (Must):** The judge shall assess calibration using a strictly proper scoring rule, reported separately for starts inside and outside the training region.

**In plain words:** check whether "I'm 90% sure" turns out to be right about 90% of the time — and check it separately on familiar and unfamiliar ground. The scoring maths must be the kind a model cannot cheat.

> Gneiting and Raftery: the wrong choice of scoring rule can be gamed by a model that misstates its own confidence. Picking a cheatable rule would let the overconfident model score well by being confident — the exact failure this project exists to catch.

**JU-5 (Must):** The judge shall report sharpness alongside calibration, and shall not treat calibration alone as sufficient.

**In plain words:** also check the model isn't just hedging. "Somewhere between zero and a million" is technically always right and completely useless.

> Without this the judge would reward cowardice, and the vaguest model would win.

**JU-6 (Must):** Beyond the point where the world's own divergence exceeds the task tolerance, the judge shall stop grading individual trajectories and grade statistical agreement instead.

**In plain words:** past the point where nobody could predict the details, stop marking the exact path and start marking whether the general pattern is right.

> This is weather versus climate. Nobody grades a forecast for a specific day nine months out; they check whether the predicted climate is right. The judge must switch the same way — and must say where it switched.

**JU-7 (Must):** The judge shall compute a trust horizon for every combination of model, world, and task, and shall never report one without naming the task and tolerance it belongs to.

**In plain words:** every "trust it for N steps" comes stapled to the job it applies to. There is no single trust number.

**JU-8 (Must):** The judge shall count exceptions — outcomes falling outside the model's stated confidence — compare the count against the rate the model's own confidence implies, and assign a status from a fixed set of bands using thresholds declared in advance.

**In plain words:** count how often reality landed outside the model's stated confidence. A model claiming 95% confidence should be caught out about one time in twenty. Far more than that means its confidence was fiction — however good its average looked.

> This is the banking mechanism, imported directly.

**JU-9 (Must):** Every verdict shall be one structured record containing: skill scores against both baselines, error against horizon with the divergence benchmark, calibration and sharpness inside and outside the training region, exception counts against thresholds, per-task trust horizons, and an explicit list of what was not tested.

**In plain words:** the verdict is one tidy record with everything in it — including a list of what we never checked.

> Mitchell's model cards: stating what wasn't evaluated matters more than the headline score. A verdict reporting only what was measured invites the reader to assume everything else was fine.

**JU-10 (Must):** Every verdict shall carry a statement of the judge's own limitations, including that this method validates the middle of the distribution rather than the extremes, and that a toy world validates the harness rather than the field.

**In plain words:** every verdict says what the judge itself can't see — mainly that this kind of checking is weakest on rare disasters, which is exactly where the damage happens.

> Rebonato's objection turned on ourselves: a model can pass every backtest and still be structurally wrong. Derman's rule — know what you're ignoring — makes saying so mandatory.

**JU-11 (Must):** All thresholds, band definitions, and metric choices shall be fixed and recorded before the judge is run against the unrigged models.

**In plain words:** decide where the pass/fail lines sit before seeing any results, so nobody can move the goalposts once the answers are in.

> The judge has exactly one subjective part: where its thresholds sit. Fixing them in advance is what stops them being nudged until the answer looks interesting.

**JU-12 (Should):** The judge shall be a pure function of its inputs, with no file, network, clock, or random-number access anywhere in its call graph.

**In plain words:** the judge does arithmetic on what it's handed and nothing else — it never reads files, checks the time, or rolls dice.

**JU-13 (Won't):** The judge will not use a language model or any learned component.

**In plain words:** the judge is plain arithmetic, not an AI. Its only judgement calls are the thresholds, which are written down.

> Automated judges built on language models have documented biases — favouring longer answers, favouring whatever came first, favouring their own outputs. Ours avoids that entire class of problem by not being one.

---

## Domain 4 — Showing the Result

A verdict nobody can read is not a verdict. This domain covers what comes out of the judge and how it is presented.

The audience test is simple: Dev has ten minutes, no risk background, and will look at two charts. If the point does not land in that time, the project has failed regardless of how correct the arithmetic is.

**RP-1 (Must):** The primary chart shall be a backtesting exception plot: predictions with their stated confidence range, actual outcomes drawn over them, and every outcome falling outside its range marked.

**In plain words:** the main picture shows what the model expected, what actually happened, and circles every time reality landed outside what the model said was likely.

> Anyone from a risk background recognises this chart instantly, which is the point — it is the visual proof that the banking method has been transplanted. Everyone else gets it from the caption.

**RP-2 (Must):** A second chart shall show error growing with prediction distance, with the world's own divergence rate drawn on the same axes as a reference line.

**In plain words:** a picture of how wrong the model gets the further ahead it guesses — with a second line showing how fast the world becomes unguessable anyway.

> The gap between the two lines is the model's fault. Everything under the reference line is nobody's fault. Drawing them together is what makes that distinction visible rather than argued.

**RP-3 (Must):** A third chart shall show calibration — stated confidence against how often that confidence was justified — plotted separately for familiar and unfamiliar starting points.

**In plain words:** a picture answering "when it said 90% sure, was it right 90% of the time?" — shown separately for familiar and unfamiliar territory.

**RP-4 (Must):** The results shall include a comparison table showing, for every model, its ranking by ordinary error alongside its trust horizon for each task.

**In plain words:** one table putting the usual accuracy score next to the trust horizon, so you can see where they disagree.

> This table is the actual result of the project. If ordinary error ranks two models the same and the judge separates them, that difference is the entire argument, and it needs to be visible in one glance rather than reconstructed from three charts.

**RP-5 (Must):** Every chart shall carry a one-sentence plain-language caption stating what it shows and what a reader should conclude.

**In plain words:** every picture comes with one ordinary sentence saying what it means.

**RP-6 (Must):** Every verdict shall also be written in a structured machine-readable form, not only as charts.

**In plain words:** the verdict is saved as data as well as pictures, so it can be checked, compared, and re-read by other tools.

**RP-7 (Must):** The complete result set shall be reproducible from a single documented command on an ordinary laptop.

**In plain words:** one command rebuilds every number and every chart from scratch.

> A governance claim that cannot be re-run by a sceptic is a claim on trust, which is precisely what this project says should not be accepted.

**RP-8 (Must):** Output involving fixture models shall visibly mark them as fixtures at the point of display, not only in surrounding documentation.

**In plain words:** the deliberately broken models are labelled as such right on the chart, not just in the notes underneath.

> Charts get screenshotted and pasted somewhere else without their context. The label has to travel with the image.

---

## Non-Functional Requirements

These describe how the whole system must behave rather than what it must do.

**NF-1 (Must):** Given identical inputs and seeds, the judge shall produce a byte-identical verdict across runs, across processes, and across machines of the same platform.

**In plain words:** the same inputs always produce exactly the same verdict, character for character — not merely a very similar one.

> **How this gets checked:** serialise the entire verdict and compare it across ten consecutive runs, byte for byte. Comparing the whole record rather than selected fields means any new field is covered automatically, and anything non-deterministic added later fails loudly instead of quietly.
>
> This is not tidiness. A checker whose answer wobbles between runs cannot be the basis of a decision, and everything this project claims about governance rests on it.

**NF-2 (Must):** The full result set shall run to completion on an ordinary laptop with no specialised hardware, in minutes rather than hours.

**In plain words:** it runs on a normal laptop in a few minutes.

**NF-3 (Must):** The judge shall depend only on the standard library and a small, named set of well-established scientific packages.

**In plain words:** almost no external code, and every piece of it common and long-established.

**NF-4 (Must):** The repository shall contain nothing originating from the author's professional context — no internal figures, no employer name, no internal team or committee names. Banking practice shall be described in publicly documented, generic terms only.

**In plain words:** nothing from the author's day job goes in the repository, ever. Banking is described only from published public sources.

> The repository is public from its first commit, so there is no window in which a mistake could be quietly fixed before the world sees it.

**NF-5 (Must):** No output shall claim more than the evidence supports. Where the harness cannot demonstrate something, the output shall say so plainly rather than omitting it.

**In plain words:** the project never says more than it can prove, and says out loud where it falls short.

> This follows from the value and viability risks. Overclaiming on a public repository attached to a public essay costs more credibility than the result could ever earn.

**NF-6 (Should):** The judge, the worlds, the models, and the reporting layer shall be separable, with the judge importing nothing from the others.

**In plain words:** the four parts stay in their own boxes, and the judge never reaches into any of them.

> This is what makes NF-1 achievable and what lets someone else point the judge at their own model.

---

## Assumptions

1. Exact ground truth from known equations is an acceptable substitute for real-world observation, for the purpose of validating the harness rather than the field.
2. A model's stated uncertainty can be meaningfully compared against observed outcomes over the sample sizes a toy world produces.
3. The two chosen worlds differ enough in predictability to exercise the divergence-benchmark logic.
4. The reader of the results has no risk-management background and no intention of reading source code.
5. Standard scientific Python produces reproducible floating-point results on a fixed platform when seeded and single-threaded.

> Assumption 5 is the one most likely to bite. If it fails, NF-1 becomes considerably harder and the technical approach may need revisiting.

---

## Constraints

1. **Public repository from the first commit.** Nothing can be quietly corrected before it is world-readable.
2. **Solo, unfunded, evenings.** Scope discipline is a delivery control, not a preference.
3. **Ordinary consumer hardware.** No GPU, no cluster.
4. **The essay is already published in draft.** The build must make good on what draft v2.2 promised, or the difference must be stated openly.

---

## Out of Scope

- Building, improving, or competing with any real world model.
- Video, images, robotics data, or any real-world dataset.
- A web application, hosted service, or interactive dashboard.
- Any claim about whether a specific published world model is trustworthy.
- Regulatory or standards advocacy. The project demonstrates a method; it does not propose a rule.

---

## Open Questions

These are unresolved and carried forward to the next stage rather than guessed at now.

1. **Where do the exception thresholds sit?** Banking uses established supervisory bands. Whether to adopt an analogous banding or derive thresholds from each task's tolerance is a design decision for the technical specification.
2. **How many rollouts make a sound sample?** Exception counting needs enough trials for the count to mean something. The required number depends on the confidence level being tested.
3. **Is foxes-and-rabbits too well-behaved?** Its divergence is slow, so its trust horizons may be long and uninteresting. The double pendulum exists partly to cover this, but if the tame world produces nothing worth reporting, that should be stated rather than padded.
4. **What exactly is the ensemble's stated uncertainty?** Spread across members is the obvious choice, but the mapping from spread to a confidence range needs to be fixed before any judging happens, per JU-11.

---

## Requirements Index

| ID | Domain | Summary | Priority |
|----|--------|---------|----------|
| WD-1 | Worlds | Two reference worlds with exact ground truth | Must |
| WD-2 | Worlds | Common state-and-action interface | Must |
| WD-3 | Worlds | Shared integrator and step size, test-enforced | Must |
| WD-4 | Worlds | Each world reports its own divergence rate | Must |
| WD-5 | Worlds | Declared training and out-of-training regions | Must |
| WD-6 | Worlds | At least two tasks with different tolerances | Must |
| WD-7 | Worlds | Seeded, byte-identical trajectories | Must |
| WD-8 | Worlds | No randomness, hidden state, or high-dimensional input | Won't |
| MU-1 | Models | One interface returning prediction and uncertainty | Must |
| MU-2 | Models | Persistence and linear baselines, always compared | Must |
| MU-3 | Models | Three fixture models with specified failure modes | Must |
| MU-4 | Models | Fixtures labelled as fixtures everywhere | Must |
| MU-5 | Models | At least two unrigged models differing on uncertainty method | Must |
| MU-6 | Models | Recipe and expected ranking recorded before judging | Must |
| MU-7 | Models | Training and evaluation data disjoint | Must |
| MU-8 | Models | Seeded, reproducible training | Must |
| MU-9 | Models | New models need only the interface | Should |
| MU-10 | Models | No large or pretrained models | Won't |
| JU-1 | Judge | Sees only predictions and outcomes | Must |
| JU-2 | Judge | Skill score against baselines, never absolute error alone | Must |
| JU-3 | Judge | Error against horizon, benchmarked on divergence | Must |
| JU-4 | Judge | Calibration via a strictly proper scoring rule, in and out of region | Must |
| JU-5 | Judge | Sharpness reported alongside calibration | Must |
| JU-6 | Judge | Switch to statistical grading past the predictability limit | Must |
| JU-7 | Judge | Trust horizon per model, world, and task | Must |
| JU-8 | Judge | Exception counting against pre-declared thresholds | Must |
| JU-9 | Judge | One structured verdict record including what was not tested | Must |
| JU-10 | Judge | Every verdict states the judge's own limitations | Must |
| JU-11 | Judge | Thresholds fixed before judging the unrigged models | Must |
| JU-12 | Judge | Pure function, no I/O, clock, or randomness | Should |
| JU-13 | Judge | No language model or learned component in the judge | Won't |
| RP-1 | Reporting | Backtesting exception plot as the primary chart | Must |
| RP-2 | Reporting | Error against horizon with divergence reference line | Must |
| RP-3 | Reporting | Calibration chart, familiar and unfamiliar separately | Must |
| RP-4 | Reporting | Comparison table: ordinary error against trust horizon | Must |
| RP-5 | Reporting | Plain-language caption on every chart | Must |
| RP-6 | Reporting | Machine-readable verdict alongside charts | Must |
| RP-7 | Reporting | Full reproduction from one documented command | Must |
| RP-8 | Reporting | Fixtures marked as fixtures on the output itself | Must |
| NF-1 | Non-functional | Byte-identical verdicts across runs | Must |
| NF-2 | Non-functional | Runs on an ordinary laptop in minutes | Must |
| NF-3 | Non-functional | Minimal, well-established dependencies | Must |
| NF-4 | Non-functional | Nothing from the author's professional context | Must |
| NF-5 | Non-functional | No claim beyond the evidence | Must |
| NF-6 | Non-functional | Judge separable, importing nothing from other layers | Should |

**Totals:** 45 requirements — 39 Must, 3 Should, 3 Won't.

---

## Priority Model

| Priority | Meaning |
|----------|---------|
| **Must** | Without it the project fails its purpose. There is no version of this worth shipping that omits a Must. |
| **Should** | Genuinely wanted and expected to be built, but the project still stands without it. |
| **Could** | Desirable if time allows. Nothing in this round is a Could — anything that weak was moved to Won't instead. |
| **Won't** | Deliberately excluded from this round. Recorded so the exclusion is a decision that can be revisited, not an oversight. |

> The high proportion of Musts is a consequence of deliberately narrow scope rather than a lack of prioritisation. Everything genuinely optional was cut before this document rather than downgraded within it.

---

*Developed using the Grounded Vibe Methodology*
