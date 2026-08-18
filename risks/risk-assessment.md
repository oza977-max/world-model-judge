---
schema_version: 1
---

# Risk Assessment — World Model Judge

Cagan's four product risks, assessed before requirements elicitation.
Source material: the essay *Words Are a Menu. The World Is Not.* (draft v2.1)
and the scoping conversation that preceded this document.

## Value Risk

Will anyone want this? The essay's claim is that evaluation metrics for learned
simulators exist in fragments while the governance function — an independent
party who owns the number, reports it, and can force a stop — exists nowhere.
The value therefore rests on the packaging, not on inventing a metric, and that
is a narrower claim than it first appears.

Three ways the value fails. First and most serious: on a clean toy system every
model may pass, or the judge's verdicts may rank models exactly as ordinary
error metrics already do. A judge that never disagrees with RMSE is ceremony,
not an institution, and the project would have demonstrated nothing. This is the
dominant risk and the requirements must be shaped around it — the judge earns
its existence only by separating models that a naive metric ranks the same.

Second: the world-model research audience may reject a predator-prey ODE as too
trivial to be evidence about anything, reading it as a toy that dodges every
hard part of the real problem. Third: practitioners may point at existing work
(uncertainty calibration, model-based reinforcement-learning rollout evaluation,
closed-loop simulation in autonomous driving) and say the governance already
exists under other names, contesting the essay's central absence claim.

Mitigation is honesty about scope rather than a bigger toy: state plainly that
the toy validates the harness, not the field, and cite the fragments that do
exist rather than claiming a vacuum.

questioner: Kshitij Oza

## Usability Risk

Can the intended people use it? There are two distinct audiences and the risk is
different for each.

The essay's readers are the first audience, and most of them have no risk-
management background. The essay promises that a lay reader will recognise a
backtesting exception plot and have the one sentence that explains it. If the
judge's output is a wall of numbers, or if reading it requires knowing what a
confidence interval or a Lyapunov exponent is, the demonstration fails at
exactly the point it is supposed to land.

The second audience is anyone who later points the judge at their own model. For
them the risk is a model interface so entangled with predator-prey specifics
that adapting it is a rewrite. Since the product is explicitly a reusable
checker rather than a one-off analysis, an interface that only ever fits the toy
would quietly falsify the project's own claim about itself.

A third, subtler failure: the trust horizon reads as a single universal number
and gets quoted out of context, as though a model were trustworthy for fourteen
steps full stop. It is always relative to a task and a tolerance. The design
answer is to define more than one task from the start so the same model visibly
earns different horizons, which kills the misreading before anyone makes it.

questioner: Kshitij Oza

## Feasibility Risk

Can it be built? Most of it is ordinary numerical work, and the deliberate
choice to keep the judged models simple removes the largest source of technical
difficulty. Three specific hazards remain.

The first is the one the essay already names: if the true system and the learned
model are integrated with different solvers or step sizes, every curve measures
the integrator rather than the model. This is a design error rather than a
coding error, which means it must be settled in the specification and enforced
by a test, not caught during debugging.

The second is that separating model error from genuine unpredictability requires
a benchmark for how fast the true system diverges from itself under a tiny
perturbation. Lotka-Volterra is periodic rather than chaotic, so that divergence
is mild — which raises a real possibility that the toy world is too well-behaved
to exercise the most interesting logic in the judge, pushing the honest
demonstration toward a chaotic system such as a double pendulum and enlarging
scope.

The third is determinism. Byte-identical verdicts across runs are a stated
requirement, and in a numerical Python stack that is not free: seeding, floating
point accumulation order, dictionary and set iteration, and threaded linear
algebra libraries can all introduce run-to-run variation. It is achievable, but
only if it is designed in from the first commit and pinned by a test rather than
asserted in prose.

questioner: Kshitij Oza

## Viability Risk

Does this work for its owner? This is not a commercial product. It is a public
repository under the author's own name, attached to a public essay, whose
purpose is to make the essay's final promise good and to open conversations with
people working on world models, evaluation, and model risk.

That framing creates its own risks. Confidentiality is the hardest constraint:
the repository is public from the first commit, so nothing from the author's
professional context may appear in it — no internal figures, no employer name,
no internal team or committee names — and there is no window in which a mistake
can be quietly corrected before it is world-readable. Banking practice must be
described in street-generic terms only.

The second risk is credibility damage through overclaiming. Models deliberately
built to fail in specific ways are test fixtures that prove the judge can detect
each failure class; they are not discoveries. If the repository presents rigged
failures as findings, a knowledgeable reader will notice immediately and the
project will subtract from the author's credibility rather than adding to it.
The distinction must be stated explicitly in the documentation, not merely
understood by the author.

The third is sustained effort. This is solo, unfunded, and competes with other
projects for the same evenings. The scope discipline the essay itself sets —
the smallest honest version — is therefore a viability control and not a
stylistic preference. Scope growth here does not produce a better project; it
produces an unfinished one.

questioner: Kshitij Oza

---

*Developed using the Grounded Vibe Methodology*
