# Project final status

Date: 2026-10-27. Written after a full verification pass. Nothing was trained,
and nothing was staged, committed or pushed.

## Executive summary

**What the project does.** It forecasts the excess path loss (EPL, in dB) of a
39 GHz Earth-LEO satellite link 1-5 hours ahead.
* Data: hourly ERA5 weather for six European sites.
* Labels: reconstructed with ITU-R propagation models.
* Model: a multi-task LSTM with Gaussian uncertainty.

It is an independent implementation inspired by El-Sayed et al. (EURASIP JWCN
2026:102), **not an exact reproduction**.

**What is complete:**
* the data and label pipeline;
* the strictly causal evaluation protocol, with leakage tests;
* one trained main model (seed 42);
* baselines on identical keys;
* an analysis of the gap to the paper;
* 13 figures;
* a full documentation set (README, visual guide, explanation, presentation
  guide, checklist, this report).

**What was verified in this session:**
* the main metrics, recomputed from the saved predictions;
* 41/41 unit tests;
* 127/127 audit checks (earlier protocol);
* that all 13 figures and PDFs are readable;
* that all Markdown links resolve;
* that no protected artefact changed.

**Readiness.** Ready for demonstration and submission as an honest, documented
study, provided the stated limitations are presented: reconstructed labels,
one seed, heavy-rain weakness, and the non-comparable paper value.

## Verified quantitative results (causal hourly evaluation)

* **Source of truth:** `results/causal_hourly_fig1heads_seed42/metrics_test.json`
  and `run_config.json`.
* **Verification method:** `evaluation/plot_research_results.py` recomputes
  RMSE and MAE from `test_predictions.npz` and the baseline file
  `results/causal_baseline_audit/aligned_keys_hourly.npz`. It stops if they
  differ by more than 1e-9 or if the keys differ. It ran successfully on
  2026-10-27.
* **Units:** the targets were recomputed independently from the physics
  tables and match to within 9e-7 dB, confirming physical dB
  (`results/why_results_differ/units_check.json`).

| Method | RMSE (dB) | MAE (dB) |
|---|---|---|
| LSTM `multitask_fig1heads`, seed 42 | **0.8592** | **0.3350** |
| Persistence | 1.0181 | 0.3461 |
| Recent 24-h mean | 1.1261 | 0.4940 |
| Linear extrapolation | 2.5186 | 0.7182 |

**Improvement**, computed as 100 x (persistence - LSTM) / persistence:

| Metric | Calculation | Improvement |
|---|---|---|
| RMSE | (1.0181 - 0.8592) / 1.0181 | **15.6 %** |
| MAE | (0.3461 - 0.3350) / 0.3461 | **3.2 %** |

`metrics_test.json` stores these as -15.60 % and -3.20 %.

**Sample counts.**
* 39,000 forecast origins (6 cities x 6,500, 2024-04-04 to 2024-12-31) x 5
  horizons = **195,000 target values**.
* Pooled metrics average over the 195,000 values; per-horizon metrics over
  39,000 each.

**Model and training** (from `run_config.json`):

| Item | Value |
|---|---|
| Parameters | 1,468,707 (checked by hand from the layer sizes) |
| Input | 60 hourly steps x 14 features |
| Horizons | +1 to +5 h |
| Epochs | 11 run, best epoch 5 (validation loss -0.3006) |
| Training time | 3.08 h on CPU |
| Software | TensorFlow 2.21.0 |

**Weaknesses (must stay visible):**

| Item | LSTM | Persistence |
|---|---|---|
| Heavy rain (>= 2.5 mm/h at the target hour, 1,475 values), RMSE | **6.704 dB** | 5.845 dB |
| Heavy rain, MAE | 5.660 dB | 4.495 dB |
| MAE at +1 h | 0.211 dB | 0.191 dB |
| MAE at +2 h | 0.302 dB | 0.295 dB |

The LSTM is worse than persistence in every row of this table.

Coverage of the 95 % prediction interval:

| Group | Coverage |
|---|---|
| All values | 97.4 % |
| Dry | 99.5 % |
| Light rain | 91.0 % |
| Heavy rain | **46.0 %** |
| True EPL >= 10 dB | 28.6 % |

The 50 % interval covers 81.9 % overall. Overall coverage averages two
opposite failures: intervals too wide in clear sky and too narrow in rain.

**Minor numerical note.** At the nominal 50 % level, `metrics_test.json`
reports 81.927 % and Figure 9 shows 81.944 %. The evaluation script uses
z = 0.674 (rounded) and the plotting script the exact normal quantile
0.6745. This is not an error, and it does not affect any rounded value.

## Deliverables inventory

| Path | Purpose | Status | Evidence |
|---|---|---|---|
| `results/causal_hourly_fig1heads_seed42/` | Main run: metrics, predictions, log, config, scalers | Verified | Metrics recomputed; MD5 unchanged |
| `models/causal_hourly_fig1heads_seed42/` | `best.keras`, `model.keras` | Present, unverified | MD5 unchanged; not reloaded this session |
| `results/causal_baseline_audit/` | Baselines and reference test keys | Verified | Keys equal the model's keys (checked by the plot script) |
| `evaluation/plot_research_results.py` | 13 figures and manifest; metric re-check | Verified | Ran OK; `--help` matches the documented flags; PNGs byte-identical across runs |
| `results/figures/` (13 PNG, 13 PDF, `figures_manifest.json`) | Presentation figures | Verified | All PNGs open; PDFs have valid header and trailer; manifest is valid JSON with 13 entries |
| `docs/assets/figures/causal/` (5 PNG) | Figures embedded in the README | Verified | Byte-identical to `results/figures/` |
| `README.md` | Two-minute overview | Verified | Rewritten; metrics cross-checked; links checked |
| `docs/VISUAL_RESULTS_GUIDE.md` | Figure-by-figure guide | Verified | Numbers match the manifest |
| `docs/PROJECT_EXPLAINED_FROM_SCRATCH.md` | Learning and viva guide | Verified | Written from the code; numbers cross-checked |
| `docs/PRESENTATION_GUIDE.md` | 9-slide, 9-minute talk | Verified | Figure paths exist; numbers cross-checked |
| `docs/FINALIZATION_CHECKLIST.md` | Pre-submission checklist | Verified | Commands match the scripts |
| `docs/WHY_RESULTS_DIFFER_FROM_PAPER.md` and `results/why_results_differ/` | Paper-gap analysis | Verified | Outputs present; one reference wording fixed |
| `docs/CAUSAL_HOURLY_MODEL_EXPERIMENT.md`, `docs/CAUSAL_FORECASTING_INVESTIGATION.md` | Experiment and protocol reports | Verified | Numbers match `metrics_test.json`; date corrected |
| `tests/` (41 tests) | Unit and leakage tests | Verified | 41 tests, OK |
| `evaluation/audit_results.py` | Audit of the earlier stratified results | Verified | 127/127 PASS via a read-only wrapper |
| `docs/FINAL_RESEARCH_REPORT.md`, `REPLICATION.md`, `docs/REPLICATION_RESULTS.md`, `STATUS.md` | Earlier-protocol reports and log | Present | Now marked as earlier-protocol documents; content otherwise unchanged |
| `docs/assets/figures/fig1_architecture.png`, `fig2_pipeline.png` | Older diagrams | Present, **outdated for the main result** | They show the earlier main model and the stratified dataset; the presentation guide provides replacement diagrams |
| Full rebuild (`run_pipeline.ps1 -Download`, causal training) | Data and training reproducibility | Not run | Expensive; would overwrite outputs; needs internet |

## Tests and validation

All commands were run from the repository root on 2026-10-27, with
`.venv` (Python 3.12).

| Command | Result |
|---|---|
| `.venv\Scripts\python -m unittest discover -s tests` | `Ran 41 tests ... OK` |
| `evaluation\audit_results.py` via a read-only wrapper (report redirected to a temporary folder) | `127 passed, 0 failed`; `results\paper\audit_report.json` unchanged (MD5) |
| `.venv\Scripts\python -m py_compile` on `evaluation\plot_research_results.py`, `why_results_differ.py`, `evaluate_causal_model.py`, `training\train_causal_hourly.py`, `data_pipeline\causal_forecast_source.py` | OK |
| `.venv\Scripts\python evaluation\plot_research_results.py --help` | Shows `--overwrite`, `--export-docs` |
| `.venv\Scripts\python evaluation\plot_research_results.py --overwrite --export-docs` | 13 figures written, 5 copied, metric assertions passed; about 11 s |
| PNG and PDF readability (Pillow `verify`, PDF header and trailer) | 13/13 and 13/13 OK |
| JSON validity (`figures_manifest.json`, `metrics_test.json`, `run_config.json`) | OK |
| Markdown link and image check (README, STATUS, REPLICATION, all `docs/*.md`) | 0 broken links |
| MD5 of 635 protected files (`results/` except `results/figures/`, `models/`, `data/processed/`, LWP coefficients) | All unchanged |

**Skipped on purpose:**
* model training;
* the data download;
* `run_pipeline.ps1`, which would overwrite `results\paper\`;
* a fresh environment install;
* re-running the scripts that refuse to overwrite their outputs.

**Reproducibility levels:**
* **Saved evaluation and plots:** reproducible and verified.
* **Model training and data acquisition:** deterministic settings and a
  documented procedure, but **not re-run** in this session.

## Scientific limitations

1. **Reconstructed labels.**
   * EPL is ITU-R models applied to ERA5 reanalysis, not measured received
     power.
   * The elevation (30 degrees) and the use of hourly rain as R0.01 in P.618
     are our choices.
   * The cloud liquid-water proxy is weak (held-out R^2 about 0).
2. **Single seed** for the causal experiment: no robustness or significance
   claim. On the earlier protocol, three seeds varied with a standard
   deviation of 0.004-0.007 dB.
3. **Heavy rain:** worse than persistence. Peaks are under-predicted (largest
   prediction 8.9 dB vs a true maximum of 25.6 dB).
4. **Conditional calibration is poor:** 46 % coverage in heavy rain, despite
   97.4 % overall.
5. **Retrospective inputs:** ERA5 is not available in real time, so this is
   causal but not operational.
6. **Not equivalent to the paper.** The 0.026 dB is not comparable:
   * the protocol probably differs (the paper's figures suggest a
     minute-interpolated series, where trivial baselines score 0.01-0.05 dB);
   * the labels differ (ours reach 25.6 dB, the paper's plots about 6.7 dB);
   * the paper's Table 7 overall does not follow from its per-city values.

   These are research observations, not proof of error or misconduct.
7. **Unseen cities:** evaluated only on the earlier stratified protocol.

## Discrepancies found and how they were handled

| Discrepancy | Files | Type | Action |
|---|---|---|---|
| The causal run was dated 2026-10-28; its artefacts are timestamped 2026-10-27 | `STATUS.md`, `docs/CAUSAL_HOURLY_MODEL_EXPERIMENT.md` | Documentation | Corrected to 2026-10-27 |
| The README's main table showed the earlier stratified results | `README.md` | Documentation | Rewritten: the causal result is primary; the stratified results are labelled historical |
| Resume bullets said "replicated" and "reported accuracy did not hold", using the old numbers | `docs/RESUME_BULLETS.md` | Documentation (overclaim) | Rewritten with the verified causal numbers and hedged wording |
| Older reports read as the current result | `docs/FINAL_RESEARCH_REPORT.md`, `REPLICATION.md`, `docs/PROJECT_INTERVIEW_NOTES.md` | Documentation | Added an "earlier protocol" note at the top |
| Figure 10 coverage labels touched the 95 % line | `evaluation/plot_research_results.py` | Presentation | Labels moved inside the bars; figures regenerated |
| A test reference used a non-existent pytest-style path | `docs/WHY_RESULTS_DIFFER_FROM_PAPER.md` | Documentation | Fixed |
| The older architecture and pipeline diagrams do not show the main model | `docs/assets/figures/fig1_architecture.png`, `fig2_pipeline.png` | Outdated artefact | Not regenerated (they belong to the earlier results); flagged in the presentation guide and the checklist |
| The paper's protocol, labels and Table 7 aggregation | paper vs `docs/WHY_RESULTS_DIFFER_FROM_PAPER.md` | Research ambiguity | Unresolvable without the authors' data or code |

## Unresolved issues (need your attention)

1. **No up-to-date architecture or pipeline image for the slides.** Draw them
   from the text diagrams in `docs/PRESENTATION_GUIDE.md` (slides 3-5), or ask
   for a new diagram script.
2. **Single seed.** Seeds 1 and 2 would take about 3 h each on mains power.
   This is your decision; it was not started.
3. **Git decisions.** The 10 modified prototype files and `.gitignore` were
   your earlier edits; review them, choose a licence (`GITHUB_CHECKLIST.md`),
   and decide whether to commit.
4. **Fresh-clone reproducibility is unverified.** `data/`, `models/` and the
   prediction files in `results/` are git-ignored. A reader of the repository
   sees `docs/assets/` and the small result files listed in
   `GITHUB_CHECKLIST.md` (metrics, configs, the 13 figures, audit outputs).

## Next actions (prioritised)

1. Read the three files below and rehearse the 9-minute talk.
2. Build the slides from `results/figures/`, plus simple diagrams for slides
   3-5.
3. Optional: train seeds 1 and 2 on the causal protocol before making any
   robustness statement.
4. Before sharing: choose a licence, review `git status`, and commit when
   ready.

## Recommended reading order

1. `docs/PROJECT_EXPLAINED_FROM_SCRATCH.md`: teaches the whole project and
   contains the viva questions.
2. `docs/PRESENTATION_GUIDE.md`: the talk itself, with speaker notes and
   likely questions.
3. `docs/VISUAL_RESULTS_GUIDE.md`: what each figure shows and how to explain
   it.
