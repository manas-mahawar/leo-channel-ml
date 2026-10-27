# Replication experiment plan

Date: 2026-10-26. Follows `docs/REPLICATION_GAP_AUDIT.md`. Commands are for
Windows PowerShell from the repository root. Stages 1-6 are complete; stage
7 is the only remaining experiment that could change the headline
comparison, and it is optional.

**Measured compute basis.** Uninterrupted training on this laptop (16-core
CPU, no GPU; TensorFlow 2.21) took **3.2-4.1 min per epoch** for 42,000
training + 8,898 validation windows of 60 x 14 (runs `multitask_seed1`,
`weather_only`, `multitask_fig1heads`, `multitask_msl`, `epl_only_seed2`;
`run_config.json`). Runs stop after 14-36 epochs (about 1-2.5 h). Slower
entries in the logs include standby pauses or shared CPU and are not used
for estimates.

## Stage 1: physical components and target units

* Commands:
  `.venv\Scripts\python -m unittest tests.test_pipeline.TestPhysicsLabels -v`
  `.venv\Scripts\python evaluation\label_sensitivity.py`
* Artefacts: test output; `results/replication_audit/label_sensitivity.csv`.
* Validation: cloud labels reproduced from the committed coefficients; EPL =
  sum of components; components non-negative; P.618 steps equal `itur` (earlier
  scratch test, recorded in REPLICATION.md Sect. 9).
* GPU: no. Cost: seconds.
* Stop if: any component is negative, EPL is not the exact sum, or the
  coefficients do not reproduce the stored cloud labels.
* **Status: done, passed.** Finding: elevation changes the label level,
  nothing tested reproduces the paper's small rain tail.

## Stage 2: features, scaling, sequences, splits

* Command: `.venv\Scripts\python -m unittest tests.test_pipeline.TestSplits -v`
  (plus `evaluation\audit_results.py`).
* Validation: no hour shared between splits; scalers fitted on training data;
  inputs t-59..t, targets t+1..t+5 (earlier window-level checks).
* GPU: no. Cost: seconds. **Status: done, passed.**

## Stage 3: loss and aggregation on synthetic cases

* Command: `.venv\Scripts\python -m unittest tests.test_pipeline.TestMetrics tests.test_pipeline.TestPaperConsistencyChecks -v`
* Validation: RMSE, PICP, NLL (Eq. 10) on hand-computable inputs; paper-table
  arithmetic reproduced (`evaluation\replication_audit_checks.py`).
* GPU: no. Cost: seconds. **Status: done, passed.**

## Stage 4: smoke test of the pipeline

* Command: `.\run_pipeline.ps1 -EvaluateOnly` (evaluation of existing
  checkpoints, figures, tables, audit, tests; no training).
* Validation: audit 127/127 PASS, tests pass.
* GPU: no. Cost: about 10-15 min (unseen-city evaluation dominates).
* **Status: each step run individually and passing; the script as a whole
  has not been run in one go.**

## Stage 5: paper-faithful model, seed 42

* Paper-faithful architecture is `multitask_fig1heads` (Fig. 1 heads); it is
  already trained (`models/paper/multitask_fig1heads/`, plus seeds 1 and 2).
* No retraining needed. Re-running would be:
  `.venv\Scripts\python training\train_paper_model.py multitask_fig1heads`
  (skipped automatically because the run exists; a new seed writes to
  `models\paper\seeds\`).
* **Status: done.**

## Stage 6: trained-city tables and figures

* Commands: `.venv\Scripts\python evaluation\make_tables.py`,
  `.venv\Scripts\python evaluation\make_figures.py`,
  `.venv\Scripts\python evaluation\make_paper_comparison.py`
* Artefacts: `results/paper/tables.md`, `results/paper/figures/`,
  `results/replication_audit/paper_vs_ours.md`, `fig6_regions_paper_grouping.png`.
* **Status: done.**

## Stage 7 (optional): reconstruction of the paper's minute protocol

Purpose: test the main hypothesis from the audit (Sect. B) with a trained
model, not only with baselines. This is a **separately named
reconstruction**, not a replacement of the hourly results.

* Protocol: each city's full 2020-2024 hourly series (inputs and EPL)
  linearly interpolated to 1 minute; 60-minute windows every 5 minutes;
  targets +1..+5 minutes; chronological 70/15/15 per city with a 65-minute
  gap; scalers on training windows; same model as `multitask_fig1heads`,
  seed 42.
* Size [Measured from the baseline script]: 473,484 test windows; about
  2.2 million training windows (70 % of about 3.16 million).
* Code needed (not written yet): a windowed `tf.data` generator that cuts
  windows on the fly from per-city minute arrays (a materialised training
  array would be about 7 GB of float32), and a variant name such as
  `minute_full_fig1heads` writing to its own folder.
* Cost on this CPU (scaled from 3.2-4.1 min per 42,000 windows): about
  **2.8-3.6 h per epoch**, i.e. about 55-70 h for 20 epochs. Not practical
  on this laptop.
* GPU: **justified for this stage only.** A Colab GPU may or may not be
  available and its speed was not measured; first time 1 epoch on
  a 10 % subset there before committing.
* Validation criteria: the model must be compared with persistence (0.035 dB)
  and linear extrapolation (0.011 dB) on the *same* windows; beating
  persistence but not linear extrapolation would mean the network has
  learned interpolation behaviour, not forecasting skill.
* Stop and investigate if: the generator output differs from a direct
  interpolation of a few hand-checked windows, or validation loss diverges.
* Expected scientific value: it can show whether the paper's 0.026 dB is
  reachable under this reading. It cannot show what the authors actually did.

## Stage 8 (optional, only if a question needs it)

* Per-city normalisation (Fig. 2 "normalized by location"): one retrain of
  `multitask_fig1heads` with per-city scalers, seed 42, about 1-1.5 h CPU.
  Answers: does the normalisation reading matter? Expected impact low.
* Unseen cities for seeds 1-2: requires a model-folder option in
  `evaluate_unseen_cities.py`; about 4 min per model.
* More seeds: only if a comparison below 0.01 dB must be decided.

## Not planned

* Changing elevation, rain model or cloud proxy to match the paper's
  numbers: these choices are unspecified, and fitting them to the published
  results would be tuning to the target.
