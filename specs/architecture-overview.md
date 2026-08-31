# World Model Judge — Architecture Overview

Version 1.4 · 31 August 2026 · Synthesises: cross-cutting v1.4, worlds v1.3, models v1.4, judge v1.4, reporting v1.4

> **Change note (v1.4).** Revised in the design-review-004 repair session. Version references updated to the v1.4 spec suite (worlds v1.3). The NF-4 credibility line in §4 no longer mentions a "CI-vs-local enforcement gap": Round 4 found the CI layer was a phantom — no CI system is built, configured, or listed anywhere in this design, and this document's own §1 System Context says there are no external systems beyond git — so cross-cutting v1.4 deleted the claim rather than building the contradiction; the two real enforcement layers (pre-commit hook, `wmj run` startup gates) are what §4 now names. See design-review-004.html and the repair entry in reviews/calibration.md.

> **Change note (v1.3).** Revised after `/gvm-design-review` design-review-003 (Round 3, dual/blind — 14 independent reviewers). Version references updated to the v1.3 spec suite. No new conceptual-integrity finding this round — Rounds 1–2's fixes to this document held under independent re-check. See design-review-003.html for the full findings, including the two Build Checks promoted to `reviews/calibration.md` (schema/consumer granularity mismatch; self-verified enforcement-mechanism fixes requiring adversarial pressure-testing).

> **Change note (v1.2).** Revised after `/gvm-design-review` design-review-002 (Round 2). No conceptual-integrity claim in §6 was found false this round (Round 1's correction held under independent re-check). Updated version references throughout to the v1.2 spec suite, and the NF-4 mention in §4 now reflects the redesigned local-terms-file scan mechanism (cross-cutting v1.3) rather than the v1.1 mechanism Round 2 found self-defeating. See design-review-002.html for the full findings.

**What this document is.** The one-page-up view: how the pieces fit, the decisions that matter most, and an honest check that the whole thing coheres as one design rather than five documents stapled together.

**In plain words:** if you read only one spec, read this one — it says what the system is, then points into the four domain specs for every detail.

---

## Expert Panel

| Expert | Work | Role in This Document |
|--------|------|----------------------|
| Simon Brown | C4 Model / *Software Architecture for Developers* | Context and container views |
| Frederick Brooks | *The Mythical Man-Month* / *The Design of Design* | The conceptual-integrity review (§6) |
| Bass, Clements & Kazman | *Software Architecture in Practice* (4th ed.) | Quality-attribute framing (§4) |

---

## 1. System Context (C4 Level 1)

**The system:** one Python command-line pipeline (`python -m wmj run`) that builds two toy worlds, trains and rolls out models against them, judges every model blind, and writes verdicts and charts.

**People:**
- **Dev** (the essay's reader) — opens `out/results.html`, reads two charts and their captions in ten minutes.
- **Mor** (the validator) — reads the model interface (models spec ADR-M1) and the verdict schema (judge spec §5) to judge adaptability.
- **The sceptic** — clones the public repo, runs `python -m wmj verify`, and byte-compares the published verdicts.

**External systems:** exactly one — **the git repository itself**, which serves as the pre-registration ledger: commit timestamps of `prereg/` files are the mechanical evidence that recipes, margins, and thresholds predate judging (MU-6/JU-11; models spec ADR-M5). There are no other external systems: no network, no services, no data sources beyond the equations.

## 2. Container View (C4 Level 2)

One deployable unit; the containers are the five packages plus two artefact stores. Data flows one way:

```
[worlds]────trajectories, tasks,──────▶┌─────────┐
            divergence curves          │ harness │────JudgeInput────▶[judge]
[models]────predictions + spreads─────▶│  (hub)  │◀───pure Verdict───  pure,
                                       └─────────┘                    no I/O
[prereg/]──recipe, thresholds,             │ harness wraps the pure
           git timestamps─────────────────▶│ Verdict in a {model_ref, name,
                                            │ is_fixture, verdict, meta} envelope
                                            └────────────▶[reporting]───▶[out/]
```

(design-review fix: the judge returns only a pure `Verdict`; the harness — never the judge — attaches identity and run metadata, per judge spec v1.4 §5, and writes the resulting envelope through the same canonical serializer as every other byte-compared artefact, cross-cutting v1.4 ADR-002 rule 4.)

- **`wmj.worlds`** — LV + pendulum, shared RK4 integrator, divergence benchmark, regions, tasks. Pure functions of (state, action).
- **`wmj.models`** — baselines, fixtures, the direct/ensemble pair; registry-only discovery. Sees `(state, action)` arrays, never world internals.
- **`wmj.judge`** — pure functions, arrays in → verdict out; imports stdlib + NumPy only; cannot represent a model's identity in its types.
- **`wmj.reporting`** — verdict records in → charts, captions, results page out; computes no metric of its own.
- **`wmj.harness`** — the only package that imports everything: seeding, orchestration, gates, prereg checks, CLI.
- **`prereg/`** (committed) and **`out/`** (generated) — the ledger and the product.

The judge's isolation is the load-bearing wall: everything else may know about the judge; the judge knows about nothing (cross-cutting ADR-003, enforced by AST test).

## 3. Key Decisions (the ADRs that shape everything)

| ADR | Decision | Spec |
|---|---|---|
| ADR-001 | Pure NumPy stack; hand-rolled MLPs; deps = {numpy, matplotlib} | cross-cutting |
| ADR-002 | Determinism via single-thread + explicit seeds + no ambient inputs + canonical serialization | cross-cutting |
| ADR-003 | Five packages; judge imports nothing; blindness by type | cross-cutting |
| ADR-W1 | Shared fixed-step RK4; drift measured, bounded at 1e-6 relative | worlds |
| ADR-W3 | Divergence benchmark: empirical median curve per region, 64 seeded starts | worlds |
| ADR-M3 | Same network twice: direct variance head vs 5-member ensemble with pre-registered `sqrt(1+1/K)` spread correction; 0.05 matching margin | models |
| ADR-M5 | Pre-registration enforced by git commit ordering | models |
| ADR-J1 | CRPS (closed-form Gaussian) as the strictly proper score; skill vs both baselines | judge |
| ADR-J4 | N=200 independent trials; exact-binomial bands green [12,29] / amber / red, derived and committed | judge |
| ADR-J5 | Per-task climate switch at divergence-exceeds-tolerance; conditioned climatology re-measured from the true trajectory; 16 equal-population bins, out-of-range and no-switch cases defined | judge |
| ADR-J7 | The seven JU-10 limitations disclosures, authored verbatim (design-review addition) | judge |
| ADR-R3 | Captions authored as templates in the spec, not improvised at render time | reporting |
| ADR-R5 | `wmj run` / `wmj verify` command pair; byte-identity scoped to verdicts + manifest | reporting |

## 4. Quality Attributes (how the ASRs are answered)

- **Reproducibility (NF-1, WD-7, MU-8):** the four ADR-002 rules + the ten-run byte gate + `wmj verify`. Platform-scoped, stated in the manifest.
- **Credibility (NF-4, NF-5, MU-4, JU-10):** fixture labels burned into images; limitations as fixed constants in the judge; the forbidden-terms scan (a gitignored local terms file, enforced at exactly two local layers — the pre-commit hook and the `wmj run` startup gates, which also verify the hook is actually activated — with the residual gap named rather than hidden — cross-cutting v1.4); the agreement-case caption that declines to manufacture a punchline.
- **Separability (NF-6, JU-12, JU-1):** enforced by import-graph AST test, purity-under-blocked-environment test, and identity-free types.
- **Adaptability (MU-9):** registry + one-interface rule; the zero-diff-outside-own-file test is the contract Mor checks.
- **Performance (NF-2):** 600-second budget with envelope math (judge spec ADR-J6); single-threaded by design and still two orders of magnitude inside budget.
- **Comprehensibility (RP-5, NF-5):** captions specified as content; plain-words docstrings; the ten-minute reader is a named quality attribute, not a hope.

## 5. Domain Spec Index

| Spec | Covers | In one line |
|---|---|---|
| `cross-cutting.md` | stack, determinism, structure, errors, deps | The rulebook: NumPy-only, four determinism rules, judge-imports-nothing. |
| `worlds.md` | WD-1..8 | Two pinned worlds, one integrator, measured divergence, declared regions and tasks. |
| `models.md` | MU-1..10 | Baselines with honest spreads, three one-corruption fixtures, the direct-vs-ensemble experiment, prereg mechanics. |
| `judge.md` | JU-1..13 | The arithmetic of the verdict: CRPS, coverage, sharpness, derived bands, climate switch, trust horizons, full schema. |
| `reporting.md` | RP-1..8 | Four charts with authored captions, fixture marking, machine-readable verdicts, one-command reproduction. |

## 6. Conceptual Integrity Review (Brooks)

Does it cohere as one design? The checks, run across all five specs:

- **One distance for error/tolerance/divergence, and one separate, deliberately different metric for anti-gaming scoring — corrected (design-review fix: the previous "no second metric exists anywhere" line was checked against the specs by Panel C and found false).** The normalised RMS distance defined in worlds ADR-W3 is the divergence measure, the tolerance unit, the trust-horizon cutoff, and the chart axis — one metric, restated explicitly in judge spec ADR-J5 rather than only asserted here. CRPS (judge spec ADR-J1) is a genuinely different metric, used only for the JU-4(b) anti-gaming skill summary, because that job specifically needs a strictly proper scoring rule over the full predictive distribution — a job RMS point-distance cannot do. Two metrics, two distinct and non-overlapping jobs, both named here so the claim is checkable rather than asserted. ✓ (corrected)
- **One uncertainty vocabulary.** Per-dimension mean + one standard deviation (MU-1), from the baselines to the fixtures to both unrigged models to every judge computation. ✓
- **One serializer.** Every byte-compared artefact goes through the cross-cutting canonical serializer; reporting adds none of its own. ✓
- **One refusal discipline.** Missing baseline, missing verdict field, prereg violation, gate failure — all stop the run before output exists; nothing degrades silently. ✓
- **One pre-registration mechanism.** MU-6 and JU-11 share the same git-timestamp machinery rather than two bespoke checks. ✓
- **Tension found and resolved during synthesis:** cross-cutting ADR-002 said canonical serialization covers "every artefact NF-1 compares"; reporting ADR-R5 scopes byte-identity to verdicts + manifest, excluding PNG encoding. These are consistent — NF-1's own wording covers *verdicts* — but the exclusion is now stated openly in the reporting spec and the README obligation, rather than discovered by a sceptic. Resolution: reporting's scoping stands; no spec change needed beyond the disclosure it already carries.
- **Residual asymmetry, accepted:** the judge defines its own input dataclasses instead of importing shared types — a deliberate duplication paid for NF-6 (cross-cutting ADR-003 records it). It is the design's one repetition, and it is load-bearing. ✓

Verdict: the system reads as one mind's design. The single idea it expresses everywhere: **decide the rules first, in writing, then let arithmetic apply them — and refuse loudly rather than improvise.**

---

## Changelog

| Version | Date | Change |
|---|---|---|
| 1.0 | 2026-08-25 | Initial version, synthesised after all four domain specs were individually approved. |
| 1.1 | 2026-08-25 | Design-review fixes (design-review-001): corrected the "no second metric exists anywhere" overclaim (CRPS and RMS distance are both genuinely in use, for distinct purposes); updated the container view to the harness-owned envelope; added ADR-J7 to Key Decisions. |
| 1.2 | 2026-08-30 | Design-review-002 (Round 2): version references updated to the v1.2 spec suite; NF-4 description updated to the redesigned local-terms-file scan; no new conceptual-integrity finding — Round 1's fix held under independent re-check. |
| 1.3 | 2026-08-30 | Design-review-003 (Round 3, dual/blind): version references updated to the v1.3 spec suite; no new conceptual-integrity finding. |
| 1.4 | 2026-08-31 | Design-review-004 repair session: version references updated to the v1.4 spec suite (worlds v1.3); §4's NF-4 line rewritten — the "CI" enforcement layer was a phantom contradicted by this document's own System Context and is deleted from the design, leaving the two real local layers. |

---

*Developed using the Grounded Vibe Methodology*
