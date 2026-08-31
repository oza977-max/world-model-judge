# World Model Judge — Reporting Specification

Version 1.6 · 31 August 2026 · Domain 4 of Requirements v1.2 · References: cross-cutting spec v1.6, judge spec v1.6 (verdict schema + harness envelope)

> **Change note (v1.6 — design-review-006 repair).** The model card's rationale is reconciled with judge.md: `limitations` is now the *ninth* mandatory JU-9 group (v1.5 had called it one of "the eight" while judge.md's own list excluded it — Round 6). No chart logic changed. See design-review-006.html.

> **Change note (v1.5 — design-review-005 repair).** Three consumer-side gaps Round 5 found: (1) `results.html` now renders a **model card** — the verdict's seven `limitations` disclosures and its `not_tested` list as a first-class section near the top, not buried in a JSON file Dev never opens (ADR-R5), finally meeting the Expert Panel's own Mitchell mandate; (2) Chart 4's ordinary-error-rank column now names its source explicitly (`skill.per_task_region` at the training region's control task) and shows the `vs_persistence`/`vs_linear` **skill counterpart beside every rank**, so the headline table no longer displays the bare CRPS ADR-J1 forbids; (3) Chart 2's world-time axis now sources `dt` from the `error_vs_horizon` block the judge added, closing a mandated axis that had no data. See design-review-005.html.

> **Change note (v1.4).** Revised in the design-review-004 repair session — this round's fixes were applied at the size of the *rule*, not the size of the finding (the failure mode reviews/calibration.md diagnosed after four rounds). Chart 2 is now one panel per region, reading `error_vs_horizon.per_region` (the flat object it previously read could not carry two regions' curves), with both tasks' JU-6 switch steps drawn as separately labelled lines from `climatology.per_task`. Chart 3 is one panel per task with one line per region present in `calibration.per_task` — regions follow the data, none are hardcoded — and its header strip now displays each region's `sharpness` number, giving JU-5's "reported alongside calibration" its first actual chart consumer. Chart 2's caption template was cut to three sentences (it violated RP-5's own limit and would have failed TC-RP5-01 on day one). Chart 4 renders a `natural_units: null` cell as "—" (declared absent, not forgotten — the pendulum has no natural cycle, judge spec ADR-J5). See design-review-004.html and the repair entry in reviews/calibration.md.

> **Change note (v1.3).** Revised after `/gvm-design-review` design-review-003 (Round 3, dual/blind). Chart 1 now panels by horizon_step too (a task genuinely has two, h=1 and the switch step, each with its own exception band). Chart 2 now explicitly starts plotting at step 1, not step 0, closing an undefined `log(0)` case at the schema's own guaranteed-zero starting value. Chart 3 now reads a per-task `calibration` block; the "reporting never computes a metric" invariant is restated precisely (no judgment metric, but standard display arithmetic like Chart 3's SE calculation is fine) after Round 3 found the old absolute wording self-contradicted by Chart 3. Chart 4's region-to-column mapping rule is now stated explicitly. `wmj chart-preview` is now explicitly scoped as an internal-only command, not part of the public interface. See design-review-003.html for the full findings.

> **Change note (v1.2).** Revised after `/gvm-design-review` design-review-002 (Round 2, independent re-check under strict criteria). Fixed three data gaps Round 2 found on the remaining charts, all the same defect class as v1.1's Chart 1 fix: Chart 1's field name and X-axis ordering rule cited data the schema doesn't carry (fixed: the field name is `outcome_distance` throughout, and the X-ordering rule now uses trial-generation index, which needs no additional data); Chart 2's baseline curves need the baseline models' own error-vs-horizon data, which was never stated to be reachable (fixed: reporting's input is the full envelope set for a run, not one envelope at a time — Chart 4 already required this, it was just never said); Chart 3's error bars need N, which reporting cannot compute itself (fixed: judge spec v1.2 added `calibration.n_trials`). Chart 4 now shows trust horizon per region (judge spec v1.2's `trust_horizons.per_task` gained a `region` field specifically so `fx-brittle`'s in/out-region collapse is visible on the headline chart, not just present in the data). See design-review-002.html for the full findings.

> **Change note (v1.1).** Revised after `/gvm-design-review` design-review-001. Chart 1 (the primary chart) was designed to scatter 200 individual trial outcomes and bands, but its only declared input, the pre-review Verdict schema, carried aggregate counts only — no per-trial data existed to draw from. Judge spec v1.1 added a `trials` block for exactly this; ADR-R2's Chart 1 description below now cites it, and three further undefined drawing rules (how a multi-dimensional prediction collapses to one Y-value, which region(s) the 200-trial x-axis covers, and what "ordered by start-state position" actually projects onto) are now pinned. Chart 4 now renders all three of JU-7's mandated trust-horizon units, not two. §5/§6 now describe the harness-owned envelope (judge spec v1.1) instead of a separately-maintained "index→name/fixture map." See design-review-001.html for the full findings.

**What this document is.** The design of what a reader actually sees: the four required charts and their captions, the comparison table, the fixture labelling, the machine-readable verdict files, and the single command that rebuilds everything.

**In plain words:** the judge's arithmetic is worthless if Dev can't get the point in ten minutes from two charts. This spec designs those charts sentence by sentence — including the captions — and wires the "run one command, get everything" promise.

---

## Expert Panel

| Expert | Work | Role in This Document |
|--------|------|----------------------|
| Edward Tufte | *The Visual Display of Quantitative Information* (2nd ed.) | Data-ink ratio, small multiples, no chartjunk |
| Stephen Few | *Show Me the Numbers* (2nd ed.) | Tables where precision matters; comparison as the core task (RP-4) |
| Margaret Mitchell et al. | *Model Cards for Model Reporting* (2019) | The not-tested list rendered as a first-class output, not a footnote |
| Federal Reserve / OCC | *SR 11-7* (2011, public description) | The exception plot's visual grammar a risk reader recognises on sight |
| Michael Keeling | *Design It!* | ADR format |

---

## 1. Purpose

Covers requirements **RP-1 through RP-8**. The reporting package (`wmj.reporting`) consumes the harness's `{model_ref, model_name, is_fixture, verdict, meta}` envelopes (judge spec v1.6 §5) — the full set for a run, one object per (model, world), rather than a `Verdict` plus a separately-maintained side map — and produces: the four required charts (RP-1..RP-4) with plain-language captions (RP-5), machine-readable verdict files (RP-6), and fixture labels burned into every image (RP-8) — all regenerated by one documented command (RP-7).

## 2. Architecturally Significant Requirements

- **RP-7 + NF-1**: chart *data* must be byte-reproducible; chart *images* are regenerated deterministically (fixed Matplotlib style, fixed figure sizes, no timestamps) but the byte-identity guarantee (NF-1) attaches to the serialized verdicts, not PNG encoding.
- **RP-8**: fixture labelling must survive screenshots — so it is drawn *inside* the figure axes, not in surrounding page text.
- **RP-5 + the ten-minute reader**: captions are content, not decoration — they are authored in this spec as templates with slots, so their quality is reviewable now, before any code exists.

## 3. Design Decisions

### ADR-R1 — Chart engine and style: Matplotlib, one shared style module, data-ink discipline

**Status:** Accepted. [Requirement: RP-1..RP-5, NF-3] [Test: TC-RP1-01..TC-RP5-01]

**Decision:** All charts through `wmj.reporting.style`: a single Matplotlib style (white background, no chart borders beyond the two axes, light dotted gridlines only where values are read off, colour-blind-safe palette, direct line labelling instead of floating legends where possible — Tufte's data-ink rule). Figure size 8×5 in at 150 dpi PNG. **SVG duplicates are also written** (charts as text — diff-able and crisp) **and `results.html` (ADR-R5) is the static page bundling all of it — both trace to RP-7's reproducibility rationale specifically (a sceptic diffing SVG text needs no image tooling) rather than to any single RP-ID** (design-review Minor fix: these two artefacts previously appeared with no traceability note at all). No 3D, no gradients, no decorative colour.

**Fixed colour semantics** (colour conveys meaning, consistently across all charts): model under discussion = dark blue; baselines = mid grey; world's own divergence reference = black dashed; exceptions/misses = red circles; bands: green/amber/red as muted fills; fixtures = every fixture line/marker rendered in orange *and* labelled (colour alone never carries the fixture flag).

### ADR-R2 — The four charts, exactly

**Status:** Accepted. [Requirement: RP-1, RP-2, RP-3, RP-4] [Test: TC-RP1-01, TC-RP2-01, TC-RP3-01, TC-RP4-01, TC-RP4-02]

**Chart 1 — Backtesting exception plot (RP-1, the primary chart).**
One panel per (model, world, region, task, **horizon_step** — design-review-003 fix: ADR-J4 defines exceptions at two horizon steps per task, h=1 and h=h_task/the switch step, each with its own `trials.per_task` entry and its own exception count/band; a task therefore contributes up to two Chart-1 panels per region, not one, matching the pattern already established for region itself) — design-review fix: JU-8's 200 trials are defined per-region, so one panel never mixes two regions; a world with both an in-region and an out-region panel for a task gets two Chart-1 panels, both captioned, side by side (Tufte: small multiples). Data source: the Verdict's `trials.per_task` block (judge spec v1.6 §5) — `outcome_distance`, `band_lo`, `band_hi`, `is_exception`, one entry per trial (design-review-002 fix: this paragraph previously named the field `outcome`, which does not exist in the schema — `outcome_distance` is the only correct name and is used consistently below); this block did not exist before design-review-001, and reporting has no other source for per-trial data (it computes no metric of its own, §4).

X: trial index, 0..199, **ordered by trial-generation index — the order in which the harness's seeded sampler drew each trial's starting condition** (design-review-002 fix: the v1.1 rule, "RMS normalised distance of the trial's start state from the region's box centre," named a real, well-defined projection, but no start-state or region-centre value exists anywhere in the Verdict for reporting to compute it from — reporting has no other input than the envelope, §5, and computes no metric of its own. Trial-generation index needs no additional data: it is simply the position each trial already occupies in `trials.per_task[...]`'s arrays, which is fixed at generation time by the harness's seeded sampler and therefore deterministic and reproducible under NF-1 like everything else). Y: `trials.outcome_distance` directly from the Verdict — the shared RMS normalised distance (judge spec ADR-J5) between the trial's true outcome and the model's stated mean at the task's declared horizon step, already computed by the judge (design-review fix — collapsing a per-dimension Gaussian prediction and a d-dimensional outcome into one scalar needs a stated rule; the judge computes it once, in the one place the project's distance metric is defined, rather than reporting re-deriving it). The drawn band is `[trials.band_lo, trials.band_hi]` (always `[0, width]` in this same distance unit — a distance is never negative) directly from the Verdict; reporting never derives a joint-region boundary itself. **`is_exception` is drawn as-is from the Verdict and never re-derived by comparing `outcome_distance` to `band_hi`** (judge spec v1.6 §5's explicit warning) — the two are separate facts about the same trial that happen to both appear in this block.

Misses (`trials.is_exception[i] == true`) circled in red. A header strip states: `region: training · observed exceptions: 27 · expected: 20 of 200 · band: GREEN`, with the JU-8 band boundaries drawn as a small inset scale showing where the observed count sits inside green/amber/red — the expected-vs-actual pairing RP-1 demands, on the image itself (TC-RP1-01).

**Chart 2 — Error against horizon (RP-2).**
X: rollout step **starting at step 1, not step 0** (design-review-003 fix — Round 3 found the schema's own zero-based convention guarantees `median_error[0] == 0.0` by construction, the same way the divergence artefact's `median_separation[0]` is exactly `0.0`: at step 0 a rollout hasn't advanced yet, so there is trivially no error to plot, and `log(0)` is undefined. Step 0 carries no information and is dropped from this chart specifically — the underlying array is untouched, only the plotted range excludes index 0) (dual axis label: steps and world time — **world time = step × `error_vs_horizon.dt`, the world step size the judge now passes through in that block (design-review-005 repair — Round 5 found the mandated world-time axis had no data source: reporting cannot import `wmj.worlds` for `dt`, and it was absent from the schema; judge spec §5 adds it to the block)**); Y: median normalised error, **log scale** — the scale choice that keeps the early-horizon gap legible when late-horizon error blows up (TC-RP2-01's legibility clause; on a linear scale the pendulum's late divergence flattens the first 100 steps into invisibility). Model curve solid blue, world divergence reference black dashed, **baselines grey — drawn from the two baseline models' own `error_vs_horizon` blocks (design-review-002 fix: baselines are registered models like any other, per models spec ADR-M2, so the harness produces a Verdict/envelope for them in the same run; reporting's input is the full envelope set for that (world, task) — every registered model including baselines — not a single envelope in isolation. Chart 4 already required exactly this [multiple models on one table]; this paragraph previously implied reporting only ever sees one envelope at a time, which cannot be true given Chart 4's own requirements. §5 restates the input contract explicitly.)**. **One Chart-2 panel per region**, reading that region's entry from `error_vs_horizon.per_region` (design-review-004 repair — judge spec v1.4 made this block a per-region array; the previous flat object couldn't even carry both regions' curves). **Both tasks' JU-6 switch steps are marked, each as its own labelled vertical line** ("`lv-control` switches to climate here" / "`lv-planning` switches here"), sourced from `climatology.per_task` filtered to this panel's region (design-review-004 repair — Round 4 found "the" switch step undefined for a world with two tasks whose tolerances cross the divergence curve at different steps; the error curve itself is task-independent, so the two lines overlay one curve rather than duplicating it). Caption states the reading rule verbatim (template below).

**Chart 3 — Calibration (RP-3).**
One Chart-3 panel per task. X: stated confidence level (the four declared levels); Y: observed coverage — one line per region present in `calibration.per_task` for that task, the `"training"` region solid and every other region dashed with a direct label (design-review-004 repair: judge spec v1.4 replaced the `_in_region`/`_out_region` suffix fields with explicit region keys, so the chart's line set now follows the data instead of hardcoding exactly one out-region — the same dynamic-regions rule Chart 4 already uses). The perfect-calibration diagonal drawn and labelled "perfectly honest". Y-error bars from the binomial standard error, computed from each entry's `n_trials` (judge spec §5). **The panel's header strip states each region's sharpness — `sharpness.per_task[same task, same region].mean_width_90`, e.g. "sharpness (mean 90% width): training 0.18 · out-high-amplitude 0.21" — design-review-004 repair: Round 4 found the sharpness numbers had no chart consumer anywhere in this document, leaving JU-5's Must-level "reported alongside calibration" satisfied in the JSON but invisible to Dev; it is now on the calibration chart itself, beside the coverage lines it qualifies.** **This is display arithmetic, not a judgment metric (design-review-003 fix — Round 3 found this contradicted §4's blanket "reporting... never computes a metric" claim, which Chart 3 has always been an exception to):** reporting never computes a *judgment* about model quality (skill, coverage, exceptions, trust horizon — the judge's exclusive domain, per JU-1/JU-12) — but it does apply small, standard, fully-specified formulas to already-judge-supplied numbers purely for rendering (this SE calculation; the exception-band inset's positioning). §4's invariant is restated below to say this precisely. Caption in natural frequencies (template below; TC-RP3-01).

**Chart 4 — The comparison table (RP-4, the headline result).**
A rendered table (Few: precision task → table, not chart), one row per model, columns: ordinary-error rank **with its skill counterpart beside it, from a single explicitly-named source (design-review-005 repair — Round 5 found the rank was "by one-step CRPS" with no stated (task, region) selection against a block that has up to 4 entries per world, and that `skill.per_task_region`'s `vs_persistence`/`vs_linear` — ADR-J1's mandated skill numbers — had no chart consumer anywhere, so the headline table showed the exact bare-CRPS ADR-J1 forbids): the rank is computed from `skill.per_task_region` filtered to `region == "training"` at the world's control task (the in-region, short-horizon entry — the fair apples-to-apples ranking surface, and the one every model has), and each row shows both the ordinary-error rank (ordering by that entry's `crps`) AND that entry's `vs_persistence`/`vs_linear` skill scores in the adjacent cell, so no absolute-error number ever appears without its skill counterpart (ADR-J1, TC-JU2-01). The chosen (task, region) is stated in the column header, not left implicit.** Then trust horizon **per (task, region) pair** — each task contributes one sub-column per region present in `trust_horizons.per_task` for that task, rendered as all three of JU-7's mandated units — **steps + world time + natural-cycle fraction**; when a world declares no natural cycle, the Verdict carries `natural_units: null` (judge spec ADR-J5, worlds spec ADR-W4 — the pendulum, chaotic, has none) **and Chart 4 renders that cell as "—"** — an em-dash, not a zero and not a blank, so the table says "declared absent" rather than "forgot to fill in" (design-review-004 repair) (design-review fix — v1.0's column spec named only two of the three fields already present in the Verdict's `trust_horizons.natural_units`, silently dropping the third from "the headline result"; **design-review-002 fix: the trust-horizon column is now split by region, matching `trust_horizons.per_task`'s new `region` field, since a single aggregate number cannot show `fx-brittle`'s entire reason for existing — great in-region, catastrophic out-of-region — which was otherwise present in the data but invisible on the one chart built to be "the headline result"**). **The region-to-column mapping rule, stated explicitly (design-review-003 fix — Round 3 found the v1.2 fix added the data but not the rule that turns it into columns):** the entry whose `region == "training"` is always the in-region column ("training" is the one reserved region name every world uses for its training region, worlds spec ADR-W4/§4.1/§4.2); every other distinct region name present for that task becomes its own additional out-of-region column, added dynamically — matching worlds spec's own statement that the region mechanism "generalises without change if a world later declares more [out-regions]." Rows where error-rank and any trust-horizon rank disagree get a filled marker in a "⇄ disagree" column and a light highlight; agreement rows get nothing (TC-RP4-02's not-always-on check). Fixture rows are prefixed `FIXTURE:` and tinted (ADR-R1 orange). The caption slot renders the one-sentence disagreement statement (template below), generated from the actual ranking comparison.

### ADR-R3 — Captions as authored templates with computed slots

**Status:** Accepted. [Requirement: RP-5] [Test: TC-RP5-01 (judged)]

**Decision:** Captions live in `wmj.reporting.captions` as fixed English templates with numeric slots — authored here, reviewed as part of this spec, and never improvised at render time:

- **Chart 1:** "Each vertical bar is what the model said was 90% likely for one test run; the dot is what actually happened. Red circles are the misses. A model this confident should miss about {expected} of {n} times — it missed {observed}, which lands it in the {band} band drawn on the right."
- **Chart 2:** "How wrong the model gets as it predicts further ahead. The dashed black line is how fast the world drifts away from itself — only the gap above that line is the model's fault. Past each task's marked step, exact paths stop being gradable and the judge switches to checking the overall pattern." *(design-review-004 repair: the previous template was four sentences against RP-5's own two-to-three limit — it would have failed its own acceptance test, TC-RP5-01, on day one; also updated for the two per-task switch lines.)*
- **Chart 3:** "When the model said it was {level}% sure, was it right that often? Of every 100 ranges it drew at 90% confidence, about 90 should contain the true outcome. The solid line is familiar territory; the dashed line is territory it never saw in training — the honest model's lines both hug the diagonal."
- **Chart 4:** "Ordinary accuracy and the judge's trust horizon, side by side. {disagreement_sentence}" — where the sentence is one of two fixed forms: "Ordinary error says {A} and {B} are equals; the judge does not." (disagreement) or "On this run the two rankings agree — the interesting case did not occur, and saying so plainly beats manufacturing one." (agreement — NF-5's no-overclaim rule applied to the punchline itself).

All captions ≤ 3 sentences (RP-5), natural frequencies not percentages where RP-3 demands, and rendered into both the image sidecar text file and the results page.

### ADR-R4 — Fixture labelling that survives a screenshot

**Status:** Accepted. [Requirement: RP-8, MU-4] [Test: TC-RP8-01, TC-MU4-01]

**Decision:** Any figure containing fixture output draws, inside the axes (top-left, semi-opaque box): `TEST FIXTURE — deliberately broken model; a detected fault here is the instrument working, not a finding.` Fixture curves additionally use the fixture colour and a `FIXTURE:` name prefix in any on-chart label. The rule is implemented once in `wmj.reporting.style.mark_fixture(ax)` and the renderer calls it whenever the envelope's `is_fixture` field says so (judge spec v1.6 §5) — one code path, no per-chart discretion (TC-RP8-01 checks the pixels' source: the label call is asserted present in the figure object before save).

### ADR-R5 — Output layout and the single command (RP-6, RP-7)

**Status:** Accepted. [Requirement: RP-6, RP-7, NF-1] [Test: TC-RP6-01, TC-RP7-01, TC-NF1-01]

**Decision:** `python -m wmj run` (documented in the README as *the* command) executes: gates → benchmarks → training (or load of committed trained weights per prereg) → rollouts → judging → reporting, writing:

```
out/
  verdicts/{world}-{model}.json      # canonical serialization — the NF-1 byte-compared artefacts (RP-6)
  charts/{world}-{model}-{chart}.png + .svg
  charts/{world}-comparison.png + .svg
  captions/{...}.txt                 # each chart's rendered caption
  results.html                       # one static page: all charts, captions, AND the model card (below)
  run-manifest.json                  # seed, platform, package versions, prereg commit — canonical
```

**The model card — `results.html` renders the verdict's `limitations` and `not_tested` as a first-class section, not a link (design-review-005 repair — Round 5 found both fields — both now mandatory JU-9 groups (`not_tested` always was; `limitations` was made the ninth mandatory group in judge spec v1.6, design-review-006, closing a miscount where this document called it one of "the eight" while judge.md's own enumeration excluded it) and the project's core honesty requirement — existed only inside `out/verdicts/*.json`, a file Dev is never told to open; the Expert Panel's own Mitchell mandate, "the not-tested list rendered as a first-class output, not a footnote," was unmet at the surface Dev actually reads).** `page.py` renders, for the run, a **"What this verdict does not tell you"** section near the top of `results.html` (before or immediately after the headline comparison table, never below the fold): the seven `limitations` strings (ADR-J7, identical across models) as a bulleted list under that heading, and the `not_tested` list under a **"Never tested"** sub-heading. These are read verbatim from the verdict's `limitations`/`not_tested` arrays — reporting adds no wording of its own (the strings are authored in judge spec ADR-J7 precisely so they are reviewable there). This is display of judge-supplied text, consistent with §4's "reporting computes no judgment" rule. TC-RP-CARD-01 (new, judged — a human confirms the seven disclosures and the not-tested list are present and legible on the rendered page).

`python -m wmj verify` re-runs the pipeline and byte-compares `out/verdicts/` and `run-manifest.json` against the committed published copies — the sceptic's one command (RP-7's rationale). `results.html` is a static file with no JavaScript dependencies — openable from disk, matching the no-hosted-service scope exclusion.

**`wmj chart-preview` is a third, internal-only command (design-review-003 clarification — Round 3 found it named in the implementation guide's Wiring Matrix with no specification anywhere else), not part of the public `run`/`verify` pair this section otherwise describes.** It exists solely so P2-C05's early tracer-bullet chunk (implementation guide) has something to run before the full pipeline exists; it is not RP-6/RP-7's documented interface, is not mentioned in the README, and Dev/the sceptic never need it.

## 4. Component Design

```
wmj/reporting/
  style.py       # shared Matplotlib style, colour semantics, mark_fixture (ADR-R1, ADR-R4)
  captions.py    # the ADR-R3 templates + slot filling
  exception_plot.py, horizon_plot.py, calibration_plot.py, comparison_table.py
  page.py        # results.html assembly (static, stdlib templating)
  writer.py      # verdict JSON + manifest via the cross-cutting canonical serializer
```

Reporting imports the judge's `Verdict` type (via the envelope) read-only and Matplotlib. **It never computes a judgment metric — no skill score, no coverage figure, no exception count, no trust horizon is derived here; every one of those numbers comes from the verdict record verbatim, including the per-trial `trials` block Chart 1 draws from (single source of truth; a chart/verdict disagreement about a judgment is structurally impossible).** (design-review-003 clarification — Round 3 found the previous absolute "never computes a metric" wording was contradicted by Chart 3's own binomial-SE calculation.) Reporting *does* apply small, fully-specified, standard formulas to judge-supplied numbers purely to render them (Chart 3's SE from `n_trials`; the exception-band inset's scale) — this is display arithmetic, not a second, competing computation of anything the judge itself decides.

## 5. API Boundary Contracts

**Input (design-review-002 clarification): the full set of the harness's envelopes for one run** — `{model_ref, model_name, is_fixture, verdict, meta}` (judge spec v1.6 §5), one per (model, world), **as a list covering every registered model for that world, baselines and fixtures included, not a single envelope processed in isolation**. This was always required by Chart 4 (which compares every model on one table) but v1.1's wording described the input as "one per (model, world)" in a way that read as a single-envelope-at-a-time contract; Chart 2's baseline curves make the same requirement explicit for a chart that renders one model at a time but still needs its siblings' data. Reporting reads `verdict` for every number it draws and `model_name`/`is_fixture` for labelling, and adds nothing and drops nothing from any `verdict` in the set. Output contract: the `out/` layout above; `results.html` embeds charts by relative path so the folder is self-contained and copyable.

## 6. Integration Points

- **← harness:** the envelope (§5) — this is reporting's only input; it supersedes any separate "index→name/fixture map" (retired in judge spec v1.1, unchanged in v1.2) and run-manifest data (seed, platform, prereg commit) lives in `envelope.meta`.
- **→ the reader:** `results.html` and the four chart families; **→ other tools:** `out/verdicts/*.json` (RP-6, now storing the full envelope, not a bare Verdict).

## 7. Error Handling & Edge Cases

- A verdict record failing schema validation → `VerdictSchemaError`, no charts rendered (partial chart sets would misrepresent the run — same no-partial-output discipline as TC-JU9-02).
- A chart whose data is degenerate (e.g. all 200 outcomes inside bands — zero exceptions) renders normally with the true zero count; the JU-5 hedging cross-flag, if set, is printed in the header strip ("0 misses — but see sharpness: possible hedging").
- Font/rendering nondeterminism: PNGs are excluded from byte-identity claims (ADR-R5); SVGs are generated with fixed Matplotlib `svg.hashsalt` so they are reproducible in practice, but NF-1's guarantee names verdicts + manifest only — stated in the README rather than silently implied.

## 8. Testing Strategy

| Concern | Cases |
|---|---|
| Exception plot: expected-vs-actual + band on image | TC-RP1-01 |
| Horizon chart: legible early gap + reading-rule caption | TC-RP2-01 |
| Calibration chart: diagonal, regions, natural frequencies | TC-RP3-01 |
| Comparison table: disagreement marked, not always-on | TC-RP4-01, TC-RP4-02 |
| Captions readable in isolation | TC-RP5-01 (judged) |
| Machine-readable verdicts exist and match schema | TC-RP6-01 |
| One command, clean checkout, full regeneration | TC-RP7-01 |
| Fixture label on the image itself | TC-RP8-01, TC-MU4-01 |

---

## Changelog

| Version | Date | Change |
|---|---|---|
| 1.0 | 2026-08-25 | Initial version. Four charts fully designed including caption templates; fixture marking centralised; `python -m wmj run` / `verify` command pair; NF-1 byte-identity scoped to verdicts + manifest with the PNG exclusion stated openly. |
| 1.1 | 2026-08-25 | Design-review fixes (design-review-001): Chart 1 now sources per-trial data from the judge's new `trials` block (didn't exist before — the chart was undrawable), pinned its Y-value/band definition and x-axis ordering to the project's one shared distance metric, and scoped one panel to one region. Chart 4 now renders all three JU-7 trust-horizon units. §5/§6 rewritten around the judge spec's harness-owned envelope, retiring the separately-described "index→name/fixture map." |
| 1.2 | 2026-08-30 | Design-review fixes (design-review-002, Round 2): fixed Chart 1's field name (`outcome` → `outcome_distance`) and replaced its undrawable X-ordering rule (distance from a region centre the schema doesn't carry) with trial-generation index; clarified reporting's input is the full envelope set for a run, not one at a time, so Chart 2 can draw baseline curves from the baselines' own envelopes; added Chart 3's binomial-SE input (`calibration.n_trials`); split Chart 4's trust-horizon column by region, matching judge spec v1.2's new `region` field, so `fx-brittle`'s story is visible on the headline chart. |
| 1.3 | 2026-08-30 | Design-review fixes (design-review-003, Round 3, dual/blind): Chart 1 now panels by horizon_step; Chart 2 starts at step 1 (closing a `log(0)` gap); Chart 3 reads the new per-task `calibration` block; the "computes no metric" invariant restated precisely; Chart 4's region-to-column mapping rule stated explicitly; `wmj chart-preview` scoped as internal-only. |
| 1.4 | 2026-08-31 | Design-review-004 repair session (rule-sized fixes): Chart 2 panels per region from `error_vs_horizon.per_region` with two labelled per-task switch lines from `climatology.per_task`; Chart 3 panels per task with dynamic region lines and a sharpness header strip (JU-5's first chart consumer); Chart 2 caption cut to three sentences (RP-5 compliance); Chart 4 renders `natural_units: null` as "—". |
| 1.5 | 2026-08-31 | Design-review-005 repair: `results.html` renders the model card (seven `limitations` + `not_tested`, first-class, ADR-R5); Chart 4's ordinary-error rank names its `skill.per_task_region` source and shows the skill counterpart beside it (ADR-J1 compliance); Chart 2's world-time axis sources `dt` from `error_vs_horizon`. |

---

*Developed using the Grounded Vibe Methodology*
