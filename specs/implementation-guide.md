# World Model Judge — Implementation Guide

Version 1.0 · 25 August 2026 · Bridges: all five specs v1.0 → `/gvm-build` · Requirements v1.2, Test Cases v1.0

**What this document is.** The build plan: six phases, twenty-three chunks, each sized to one context window, each with its tests co-located, with the dependency network, the critical path, the parallelism map, and the wiring matrix proving every built module has a path into the running product.

**In plain words:** this is the order of work. Every chunk says what it builds, which spec section it reads, which test cases it must pass, and which later chunk would fail without it — so nothing gets built that nothing uses, and nothing gets used that nothing built.

---

## Build Phases

**MVP-1 check:** the first user-facing chunk is **P1-C02**, which delivers a runnable end-to-end product (`python -m wmj run --skeleton`: one world, one baseline, one skill score, one serialized output file, deterministic) — the smallest honest slice of the whole pipeline. No exemption needed.

**Phase numbering:** starts at P1 — `build/handovers/` does not exist and no prior implementation guide exists in this repository (checked at write time).

### Phase 1 — Walking skeleton (deliverable: one command produces a deterministic skill report)

- **P1-C01 · Foundations.** `src` scaffold, `pyproject.toml` with the pinned two-package dependency set + lockfile, canonical serializer, seed plumbing (spawned `PCG64` streams), single-thread env guard, `errors.py` conventions. Tests: serializer canonical-bytes round-trip; seed-spawn stability; thread-guard assertion. *Spec: cross-cutting ADR-002/003, Dependency Budget.* [Test: groundwork for TC-NF1-01, TC-NF3-01] ~1 session.
- **P1-C02 · MVP slice (user-facing).** Minimal LV world (`transition` + shared RK4), persistence + linear baselines, judge `skill.py` (CRPS + skill), `python -m wmj run --skeleton` writing a `wmj-skeleton/0` JSON report. Acceptance: clean run from checkout; two consecutive runs byte-identical. *Spec: worlds §4.1/4.3, models ADR-M2, judge ADR-J1.* [Test: TC-WD1-01 (LV half), TC-MU2-02 seed] ~1 session.
- **P1-C03 · Gates.** `tests/gates/`: judge import-graph AST test; WD-3 integrator-identity gate + its negative (mismatched dt must fail); ten-run byte-identity harness over the skeleton output. *Spec: cross-cutting ADR-003, worlds ADR-W1.* [Test: TC-NF6-01, TC-WD3-01, TC-WD3-02, TC-NF1-01 (skeleton scope)] ~1 session.

### Phase 2 — Worlds complete (deliverable: both worlds + benchmark artefacts)

- **P2-C01 · LV full.** Regions, tasks, conserved V, clamp guard, action-range validation. [Test: TC-WD1-01, TC-WD2-01, TC-WD6-01, TC-WD6-02] ∥ *parallel with P2-C02.*
- **P2-C02 · Pendulum full.** EOM, energy, regions, tasks, unwrapped angles. [Test: TC-WD1-01, TC-WD2-01, TC-WD7-01, TC-WD7-02] ∥
- **P2-C03 · Divergence + drift artefacts.** Benchmark generator (64 starts/region, median curves), drift measurement vs the 1e-6 bound, curve sanity assertions. *Needs C01+C02.* [Test: TC-WD3-03, TC-WD4-01, TC-WD4-02]
- **P2-C04 · Region labelling.** In/out labelling with axis attribution, boundary determinism. *Needs C01+C02.* [Test: TC-WD5-01, TC-WD5-02] ∥ *parallel with P2-C03.*

### Phase 3 — Models (deliverable: all eight registered contestants + prereg tooling)

- **P3-C01 · MLP core.** Forward/backward/Adam, gradient-check-first (the failing finite-difference test precedes backprop). *Spec: models ADR-M3 shared architecture.*
- **P3-C02 · Baseline spreads.** Training-residual spread fitting for both baselines; registry wiring. [Test: TC-MU1-01/02 (baselines), TC-MU2-01] ∥ *parallel with P3-C01.*
- **P3-C03 · Model A (direct).** Variance head, Gaussian NLL training. *Needs C01.* [Test: TC-MU1-01/02] ∥
- **P3-C04 · Model B (ensemble).** K=5, pre-registered point rule + `sqrt(1+1/K)`·std(ddof=1) spread mapping. *Needs C01.* [Test: TC-MU5-02, TC-MU5-03] ∥ *parallel with C03.*
- **P3-C05 · Fixtures.** The three one-corruption wrappers, `is_fixture` flag. *Needs C03.* [Test: TC-MU3-01/02/03 (behavioural halves), TC-MU4-01 (code surface)]
- **P3-C06 · Training data + determinism.** Trajectory dataset generation, start-disjointness assertion, train-twice-identical test. *Needs C01, P2-C01/02.* [Test: TC-MU7-01, TC-MU8-01]
- **P3-C07 · Prereg tooling.** `derive_thresholds.py` (exact binomial → `prereg/thresholds.json`), `check_prereg` git-ordering checker; commit `prereg/recipe.md` + `prediction.md` + `thresholds.json`. *Independent of C03–C06.* [Test: TC-MU6-01, TC-JU11-01, TC-JU11-02, TC-MU5-01 (margin read)] ∥

### Phase 4 — Judge complete (deliverable: full verdicts from arrays)

- **P4-C01 · Types + verdict assembly.** `JudgeInput`, `Verdict`, limitations constants, refuse-on-missing-field. [Test: TC-JU1-01, TC-JU9-01, TC-JU9-02, TC-JU10-01]
- **P4-C02 · Skill (extend P1-C02).** Per-task/region skill vs both baselines; anti-gaming property test. [Test: TC-JU2-01, TC-JU4-02] ∥ *C02–C05 parallel after C01.*
- **P4-C03 · Calibration + sharpness.** Four-level coverage in/out region; width monotonicity property. [Test: TC-JU4-01, TC-JU5-01, TC-JU5-02] ∥
- **P4-C04 · Exceptions + bands.** Trial-wise counting, band assignment from thresholds data, hedging cross-flag. *Needs P3-C07 format.* [Test: TC-JU8-01, TC-JU8-02, TC-JU8-03] ∥
- **P4-C05 · Climatology + horizons.** Switch step, conditioned climatology (16 bins, re-measured invariant), trust horizons in dual units. *Needs P2-C03 artefact shape.* [Test: TC-JU6-01, TC-JU6-02, TC-JU7-01, TC-JU7-02] ∥
- **P4-C06 · Blindness/purity properties.** Label-swap invariance, blocked-environment purity, no-learned-component static check. *Needs C01–C05.* [Test: TC-JU1-02, TC-JU12-01, TC-JU13-01]

### Phase 5 — Reporting (deliverable: charts + page from verdicts)

- **P5-C01 · Style + captions.** Shared Matplotlib style, colour semantics, `mark_fixture`, caption templates. *Spec: reporting ADR-R1/R3/R4.*
- **P5-C02 · Exception plot.** [Test: TC-RP1-01, TC-RP8-01] ∥ *C02–C04 parallel after C01.*
- **P5-C03 · Horizon + calibration plots.** [Test: TC-RP2-01, TC-RP3-01] ∥
- **P5-C04 · Comparison table + page + writer.** Table with disagreement marking, `results.html`, verdict/manifest writer via the canonical serializer. [Test: TC-RP4-01, TC-RP4-02, TC-RP6-01] ∥

### Phase 6 — Integration closure and the judged run (deliverable: the published result)

- **P6-C01 · Full wiring.** `python -m wmj run`: gates → benchmarks → training/load → rollouts (all models × worlds × regions, 200 trials) → judging → reporting; `python -m wmj verify` byte-comparison mode. Closes every seam left open by P2–P5 (this is the integration wiring chunk; no deferred seam survives it). [Test: TC-MU3-01/02/03 (judged halves), TC-MU4-01 (all surfaces), TC-MU9-01, full-scope TC-NF1-01/02]
- **P6-C02 · Startup verification + NF gates.** Clean-checkout single-command test, wall-clock budget check (< 600 s), dependency-manifest gate, forbidden-terms scan. This chunk's acceptance test is the product startup verification. [Test: TC-RP7-01, TC-NF2-01, TC-NF3-01, TC-NF4-01]
- **P6-C03 · The pre-registered judged run.** `check_prereg` certification → full pipeline on the unrigged models → publish `out/` → record the MU-6 either-way publication. Procedural checklist + mechanical certification. [Test: TC-MU5-01, TC-MU6-02 (judged), TC-JU10-02 / TC-RP5-01 / TC-NF4-02 / TC-NF5-01 (judged, human pass)]

---

## Dependency Network

| Chunk | Depends on | Enables | Parallel with |
|---|---|---|---|
| P1-C01 | — | everything | — |
| P1-C02 | P1-C01 | P1-C03, P2, P3, P4-C02 | — |
| P1-C03 | P1-C02 | (gate coverage for all) | — |
| P2-C01 | P1-C02 | P2-C03/04, P3-C06 | P2-C02 |
| P2-C02 | P1-C02 | P2-C03/04, P3-C06 | P2-C01 |
| P2-C03 | P2-C01, P2-C02 | P4-C05, P6-C01 | P2-C04 |
| P2-C04 | P2-C01, P2-C02 | P6-C01 | P2-C03 |
| P3-C01 | P1-C01 | P3-C03/04 | P3-C02, P3-C07 |
| P3-C02 | P1-C02 | P6-C01 | P3-C01 |
| P3-C03 | P3-C01, P3-C06 | P3-C05 | P3-C04 |
| P3-C04 | P3-C01, P3-C06 | P6-C01 | P3-C03 |
| P3-C05 | P3-C03 | P6-C01 | P4-C01 |
| P3-C06 | P3-C01, P2-C01/02 | P3-C03/04 | P3-C07 |
| P3-C07 | P1-C01 | P4-C04, P6-C03 | P3-C01..06 |
| P4-C01 | P1-C02 | P4-C02..06 | P3-C05 |
| P4-C02..C05 | P4-C01 (+C04: P3-C07; +C05: P2-C03) | P4-C06, P6-C01 | each other |
| P4-C06 | P4-C01..05 | P6-C01 | P5-C01 |
| P5-C01 | P1-C01 | P5-C02..04 | P4-C06 |
| P5-C02..C04 | P5-C01, P4-C01 (Verdict type) | P6-C01 | each other |
| P6-C01 | all P2–P5 | P6-C02/03 | — |
| P6-C02 | P6-C01 | P6-C03 | — |
| P6-C03 | P6-C02, P3-C07 | done | — |

**Two-track network (backend = worlds/models/judge; frontend = reporting):**

```
P1-C01 ─ P1-C02 ─ P1-C03
            │
   ┌────────┼──────────────┐
   ▼        ▼              ▼
P2-C01 ∥ P2-C02        P3-C01 ∥ P3-C02 ∥ P3-C07
   └───┬────┘              │                │
 P2-C03 ∥ P2-C04       P3-C06               │
   │                       │                │
   │                 P3-C03 ∥ P3-C04        │
   │                       │                │
   │                    P3-C05              │
   │                       │                │
   └──────────► P4-C01 ◄───┘                │
        P4-C02 ∥ C03 ∥ C04 ∥ C05  ◄─────────┘ (C04)
                   │                     [reporting track]
                P4-C06                   P5-C01 → C02 ∥ C03 ∥ C04
                   └──────────┬──────────────┘
                           P6-C01 ─ P6-C02 ─ P6-C03
```

**Critical path (11 chunks):** P1-C01 → P1-C02 → P2-C01 → P2-C03 → *(with P3-C01→P3-C06→P3-C03 running in parallel)* → P4-C01 → P4-C05 → P4-C06 → P6-C01 → P6-C02 → P6-C03.

---

## Wiring Matrix

| Entry point | Consumed modules | Wiring chunk | Demanded by |
|---|---|---|---|
| `wmj run --skeleton` (harness.cli) | `worlds.lv`, `worlds.integrator`, `models.baselines`, `judge.skill`, `harness.serialize` | P1-C02 | P1-C03 (ten-run byte gate consumes the skeleton output) |
| `wmj run` (harness.cli) | `worlds.lv`, `worlds.pendulum`, `harness.benchmarks`, `harness.regions`, `models.registry` (baselines, direct, ensemble, fixtures), `harness.trials`, `judge.verdict` (skill, calibration, sharpness, exceptions, climatology, horizon, limitations), `reporting.*`, `harness.serialize` | P6-C01 | P6-C02 (clean-checkout startup acceptance test) |
| `wmj verify` (harness.cli) | full `wmj run` path + byte comparison | P6-C01 | P6-C02 (TC-RP7-01 acceptance) |
| `harness.derive_thresholds` | exact-binomial derivation, `harness.serialize` | P3-C07 | P4-C04 (band tests read the committed thresholds format) |
| `harness.check_prereg` | git-timestamp inspection | P3-C07 | P6-C03 (judged-run certification refuses without it) |
| `worlds.divergence` (producer) | consumed by `harness.benchmarks` → `judge.climatology`, `reporting.horizon_plot` | P2-C03, P6-C01 | P4-C05 (climatology tests demand the curve artefact shape) |
| `models.mlp` (producer) | consumed by `models.direct`, `models.ensemble` | P3-C03, P3-C04 | P3-C03 (direct's failing NLL-training test demanded the MLP) |
| `models.fixtures` (producer) | consumed by `harness.trials` roster | P6-C01 | P6-C01 (TC-MU3 judged-half acceptance runs fixtures through the full pipeline) |
| `judge.limitations` (producer) | consumed by `judge.verdict` | P4-C01 | P4-C01 (TC-JU10-01 field-presence test) |
| `reporting.captions` (producer) | consumed by all four chart modules | P5-C02..C04 | P6-C02 (startup acceptance asserts rendered captions exist per chart) |
| `reporting.style.mark_fixture` (producer) | consumed by every chart renderer | P5-C02..C04 | P6-C01 (TC-RP8-01 fixture-label acceptance) |
| `harness.serialize` (producer) | consumed by skeleton, thresholds writer, verdict writer, manifest | P1-C02, P3-C07, P5-C04 | P1-C03 (byte-identity gate is the serializer's failing consumer test) |
| `judge.types` (producer) | consumed by every judge module + reporting reader | P4-C01, P5-C04 | P4-C01 (TC-JU1-01 structural-blindness test) |
| `wmj.models.registry` (producer) | consumed by `harness.trials` discovery | P6-C01 | P6-C01 (TC-MU9-01 zero-diff acceptance uses registry discovery) |

No row has an empty wiring-chunk or `Demanded by` cell; no exemptions claimed.

---

## Claude-Specific Chunking

- **Context per chunk:** this guide's chunk entry + cross-cutting spec + the one relevant domain-spec section + the named TC-IDs. Never load all five specs into a build session.
- **Chunk size:** each chunk above is scoped to one session (~1–3 hours of agent work). If a chunk runs long, split by data variation (Cohn): e.g. P2-C01/C02 are already the worked example — same structure, different world.
- **Strict TDD:** the chunk's named TC-IDs are translated to failing pytest tests first; implementation follows. Negative/phantom-gate cases (TC-WD3-02, TC-WD7-02, TC-NF1-02, TC-JU9-02, TC-JU11-02) are written as mutation tests in the same chunk as their gate.
- **Test co-location:** every chunk includes its tests; there are no "write tests" chunks anywhere in this guide.

## Parallel Work Identification (share-nothing)

Safe parallel sets (no shared files): {P2-C01, P2-C02} · {P3-C01, P3-C02, P3-C07} · {P3-C03, P3-C04} · {P4-C02, P4-C03, P4-C04, P4-C05} · {P5-C02, P5-C03, P5-C04}. Interface contracts at the boundaries are the specs' own (MU-1 Prediction, the divergence artefact JSON, the Verdict schema). Merge strategy: sequential merge in chunk-ID order within each parallel set; parallel chunks never modify the same file (each owns its module + its test file).

## Integration Closure

Deferred seams and their closing chunks: model training deferred from P3 rollout wiring → closed by P6-C01; chart rendering against real verdicts (P5 builds against fixture verdict records) → closed by P6-C01; runtime/dependency/confidentiality gates → P6-C02; prereg certification → P6-C03. A scan of this guide for "deferred"/"wired later" phrases resolves each to one of those three chunks. P6-C02 is the product startup verification chunk.

---

## Changelog

| Version | Date | Change |
|---|---|---|
| 1.0 | 2026-08-25 | Initial version. Starts at P1 — no prior build phases or implementation guide exist in this repository. 6 phases, 23 chunks, MVP-1 satisfied by P1-C02, wiring matrix complete with no exemptions. |

---

*Developed using the Grounded Vibe Methodology*
