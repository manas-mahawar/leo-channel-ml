# Visual results guide (causal hourly experiment)

This guide covers the presentation figures for the project's main result: the
`multitask_fig1heads` LSTM trained once (seed 42) on the strictly causal
hourly protocol `causal_hourly_v1`. It is meant for presenting the work.

* **Figures:** `results/figures/` (PNG at 200 dpi plus vector PDF). Five key
  PNGs are copied to `docs/assets/figures/causal/` for the README.
* **Script:** `evaluation/plot_research_results.py`. It is read-only: it
  never trains a model and never modifies predictions, labels or metrics.
* **Numbers behind each figure:** `results/figures/figures_manifest.json`.
* **Data sources:**
  * predictions: `results/causal_hourly_fig1heads_seed42/test_predictions.npz`;
  * baselines on identical keys:
    `results/causal_baseline_audit/aligned_keys_hourly.npz`;
  * labels: `data/processed/physics_paper/<city>_physics.csv`;
  * training log: `results/causal_hourly_fig1heads_seed42/training_log.csv`.
* **Checks built into the script:**
  * Before plotting, it recomputes every overall RMSE and MAE from the saved
    predictions. It stops if they differ from `metrics_test.json` by more
    than 1e-9.
  * It stops if the model and baseline test keys differ.
* **No sampling and no exclusions:**
  * Scatter, residual and calibration figures use all 195,000 test values,
    heavy rain included.
  * The two time-series periods are chosen by fixed rules (Figure 2), not by
    hand.
  * Re-running the script gives byte-identical PNGs (verified).

Test set for all model figures: 6 cities, 2024-04-04 to 2024-12-31, 39,000
forecast origins x 5 horizons (+1 to +5 h) = 195,000 values.

## Overall results

| Method (identical keys) | RMSE (dB) | MAE (dB) |
|---|---|---|
| **LSTM, multitask_fig1heads, seed 42** | **0.859** | **0.335** |
| Persistence (repeat the current EPL) | 1.018 | 0.346 |
| Recent 24-h mean | 1.126 | 0.494 |
| Linear extrapolation | 2.519 | 0.718 |

**Comparison with persistence:**

| | Change vs persistence |
|---|---|
| RMSE | -0.159 dB (**-15.6 %**) |
| MAE | -0.011 dB (**-3.2 %**) |
| RMSE by horizon | from -0.021 dB (+1 h) to -0.223 dB (+5 h) |
| MAE at +1 h / +2 h | **worse**: 0.211 vs 0.191 dB, 0.302 vs 0.295 dB |
| RMSE by city | better in all six cities |
| MAE by city | slightly worse in Athens and Warsaw |
| Heavy rain (>= 2.5 mm/h at the target hour) | **worse**: RMSE 6.70 vs 5.85 dB |

* **Training:** 11 epochs. The selected checkpoint is epoch 5 (lowest
  validation loss), restored by early stopping.
* **Uncertainty:** the nominal 95 % interval covers 97.4 % of values overall,
  but only **46.0 %** in heavy rain.

## Figures

### 1. Training curves

`results/figures/01_training_curves.png`

* **Shows:** total loss and EPL loss (Gaussian NLL) per epoch, for training
  and validation. A dashed line marks the selected checkpoint.
* **Key result:** validation loss is lowest at epoch 5 (-0.301). Afterwards,
  the training EPL loss keeps falling (-0.74 -> -1.01) while the validation
  EPL loss rises (-0.55 -> -0.28).
* **Interpretation:** this is overfitting. The model becomes over-confident
  on training data after epoch 5, and early stopping keeps the epoch 5
  weights. NLL values can be negative; lower is better.

### 2. Time-series examples

`results/figures/02_timeseries_examples.png`

* **Shows:** true EPL, the LSTM mean with its 95 % interval, and
  persistence, at +1 h and +5 h, for two periods chosen by fixed rules:
  * **A, typical:** the 14-day block, out of all cities' non-overlapping
    blocks, whose +1 h LSTM RMSE is the median. This is Bergen, 15-28 Nov 2024.
  * **B, hardest:** the 7 days around the largest true EPL in the test set.
    This is Vigo, 21-28 Nov 2024, with a peak of 25.6 dB.
* **Key result:**
  * In period A, the LSTM follows the series closely at +1 h (RMSE 0.47 vs
    0.50 dB) and is smoother but better at +5 h (0.91 vs 1.09 dB).
  * In period B, it predicts only about 8 dB (+1 h) or about 4.5 dB (+5 h)
    for the 25.6 dB peak. At +1 h, persistence wins (1.44 vs 2.23 dB),
    because it repeats the peak one hour late.
* **Interpretation:** typical rainy days are tracked well. Extreme peaks are
  strongly under-predicted. The Gaussian interval can extend below 0 dB,
  which is physically impossible; this is a known limitation of a Gaussian
  output.

### 3. Predicted vs true scatter

`results/figures/03_pred_vs_true_scatter.png`

* **Shows:** a density (hexbin, log colour scale) of all 195,000 (true,
  predicted) pairs, with a y = x line. There is a full-range panel and a
  0-7 dB zoom; the zoom only changes the axis limits and removes no values.
* **Key result:** Pearson r = 0.71. The largest prediction is 8.9 dB, against
  a largest true value of 25.6 dB.
* **Interpretation:**
  * Clear-sky values (about 0.5-1.5 dB) lie on the diagonal.
  * Above about 5 dB, predictions flatten: rain peaks are under-predicted.
  * Values above the diagonal at low true EPL are false alarms after rain ends.

### 4. Residuals

`results/figures/04_residuals.png`

* **Shows:** a histogram of the residuals (predicted - true, log counts), and
  the residuals plotted against the predicted EPL.
* **Key result:**
  * Median residual +0.041 dB; mean -0.002 dB.
  * The 1st and 99th percentiles are -3.50 and +1.75 dB.
  * The smallest residual is -23.3 dB.
* **Interpretation:** the distribution is strongly asymmetric. A long negative
  tail (missed rain peaks) dominates the squared error, while the typical
  error is small. This is why RMSE (0.859) is much larger than MAE (0.335).

### 5. RMSE and MAE by horizon

`results/figures/05_horizon_rmse_mae.png`

* **Shows:** LSTM vs persistence RMSE and MAE at +1 to +5 h.
* **Key result:**
  * LSTM RMSE grows from 0.627 to 0.985 dB; persistence from 0.648 to
    1.209 dB.
  * The LSTM's MAE is higher than persistence at +1 h (0.211 vs 0.191) and
    +2 h (0.302 vs 0.295), and lower from +3 h onward.
* **Interpretation:** the model's value is at longer lead times. One hour
  ahead, simply repeating the current value is hard to beat.

### 6. Baseline comparison

`results/figures/06_baseline_comparison.png`

* **Shows:** overall RMSE and MAE for all four methods on identical keys.
* **Key result:** the LSTM is best on both metrics: RMSE 0.859 vs 1.018
  (persistence), 1.126 (24-h mean) and 2.519 (linear extrapolation).
* **Interpretation:** linear extrapolation amplifies hourly jumps and is a
  poor baseline here. Persistence is the meaningful reference.

### 7. City comparison

`results/figures/07_city_comparison.png`

* **Shows:** RMSE and MAE per city, LSTM vs persistence, with 6,500 origins
  per city.
* **Key result:**
  * RMSE is lower in every city: Athens 0.547 vs 0.665, Bergen 0.830 vs
    0.959, Budapest 1.043 vs 1.285, Chilbolton 0.789 vs 0.962, Vigo 1.090
    vs 1.216, Warsaw 0.737 vs 0.894 dB.
  * MAE is slightly worse in Athens (0.153 vs 0.150) and Warsaw (0.312 vs
    0.304).
* **Interpretation:** the gain is consistent across climates. Vigo and
  Budapest are hardest because their rain events are the most intense; Athens
  is easiest because it is driest.

### 8. Precipitation classes

`results/figures/08_precipitation_classes.png`

* **Shows:** RMSE and MAE for dry, light rain (0-2.5 mm/h) and heavy rain
  (>= 2.5 mm/h), with sample sizes. Classes use ERA5 precipitation observed at
  the target hour. They are outcome-based, so they serve as diagnostics only.
* **Key result:**

  | Class | Values | LSTM RMSE | Persistence RMSE | LSTM MAE | Persistence MAE |
  |---|---|---|---|---|---|
  | Dry | 153,853 | 0.328 | 0.510 | 0.159 | 0.129 |
  | Light rain | 39,672 | 1.241 | 1.678 | 0.819 | 1.035 |
  | Heavy rain | 1,475 | 6.704 | 5.845 | 5.660 | 4.495 |

* **Interpretation:**
  * Most of the RMSE gain comes from dry and light-rain hours, where
    persistence carries a stale rain value.
  * In heavy rain, the hardest and most important case for link margins,
    the LSTM is **worse** than persistence.

### 9. Calibration reliability

`results/figures/09_calibration_reliability.png`

* **Shows:** nominal vs empirical coverage of the Gaussian interval
  mu +- z sigma, at 50-99 % nominal coverage, overall and per precipitation
  class.
* **Key result:**

  | Group | Covered at nominal 50 % | Covered at nominal 95 % |
  |---|---|---|
  | All values | 81.9 % | 97.4 % |
  | Dry | 87.6 % | 99.5 % |
  | Light rain | 62.7 % | 91.0 % |
  | Heavy rain | 7.9 % | **46.0 %** |

* **Interpretation:** the overall curve hides two opposite failures. The
  intervals are too wide for typical hours and much too narrow in heavy rain.
  Overall coverage near 95 % is not evidence of good calibration.

### 10. Interval width and coverage by group

`results/figures/10_interval_width_coverage.png`

* **Shows:** 95 % coverage and mean 95 % interval width, by true EPL range
  and by precipitation class. These groups use the true outcome (diagnostic).
* **Key result:**

  | Group | Coverage | Mean width |
  |---|---|---|
  | Dry | 99.5 % | 1.66 dB |
  | Heavy rain | 46.0 % | 9.45 dB |
  | True EPL 0-1 dB | 99.4 % | 1.51 dB |
  | True EPL 5-10 dB | 67.0 % | 8.15 dB |
  | True EPL >= 10 dB | 28.6 % | 11.72 dB |

* **Interpretation:** the model does widen its intervals in rain, but not
  enough to cover the errors of missed peaks.

### 11. EPL label distribution

`results/figures/11_epl_distribution.png`

* **Shows:** a histogram of all 263,088 hourly labels (6 cities, 2020-2024,
  log counts), and exceedance curves for the full series and the test targets.
  A grey band marks the approximate largest true EPL visible in the paper's
  Figs. 3-4 (6.2-6.7 dB). It was read by eye and comes from a different
  dataset, so it is context only.
* **Key result:**
  * Median 0.91 dB; 99th percentile 6.76 dB; maximum 25.6 dB.
  * Share of test targets above each threshold: 2 dB 10.43 %, 5 dB 2.21 %,
    6.7 dB 0.98 %, 10 dB 0.26 %, 20 dB 0.013 %.
  * The test set follows the full series closely, with a slightly heavier
    tail above 15 dB.
* **Interpretation:** the forecasting problem is mostly clear sky with rare,
  extreme rain events. The rare tail carries most of the squared error: the
  1 % of test values above 6.7 dB cause 53 % of it
  (`results/why_results_differ/error_budget.json`).

### 12. Attenuation components

`results/figures/12_attenuation_components.png`

* **Shows:** mean gas, cloud and rain attenuation per city (stacked), and the
  distribution of each component over all hourly labels.
* **Key result:**

  | Component | Value |
  |---|---|
  | Gas mean | 0.73-0.90 dB per city (range 0.46-1.26 dB) |
  | Cloud mean | 0.06-0.22 dB (maximum 4.1 dB) |
  | Rain mean | 0.11 dB (Athens) to 0.58 dB (Vigo, Bergen); maximum 20.6 dB |
  | Hours with zero rain attenuation | 79.0 % |

* **Interpretation:** gas sets a stable floor of about 0.75 dB. Rain, present
  in about one hour in five, causes almost all of the variability. The cloud
  values rest on our fitted liquid-water proxy, since the paper gives no
  formula.

### 13. Context: our result and the paper's 0.026 dB

`results/figures/13_paper_context.png`

* **Shows:** reported RMSE values on a log scale, in three clearly separated
  groups:
  * the paper's reported 0.026 dB;
  * trivial baselines on **our** labels under the minute-interpolated
    reading of the paper (+1 to +5 *minutes*): linear extrapolation 0.011,
    persistence 0.035 dB;
  * our causal hourly results (+1 to +5 *hours*).
* **Key result:** the paper's 0.026 dB lies in the range of no-model
  baselines on a minute-interpolated series, not near any hourly forecast.
* **Interpretation:** this is **not a controlled comparison**. The tasks,
  labels and possibly the metric definition differ, and the authors' data
  are unavailable. Present it only as context for why the numbers cannot be
  compared directly.

## Strongest findings

1. **The LSTM beats every simple baseline on RMSE** on a strictly causal test
   set with identical keys: -15.6 % vs persistence, consistently in all six
   cities (Figs. 6, 7).
2. **The advantage grows with lead time**, from -0.02 dB at +1 h to
   -0.22 dB at +5 h (Fig. 5).
3. **The evaluation is leakage-checked.** Inputs use only data up to the
   forecast time; scalers are fitted on training data; and model selection
   used validation data only (`docs/CAUSAL_HOURLY_MODEL_EXPERIMENT.md`).
4. **The discrepancy with the paper is explained and documented.** No
   implementation bug was found, and units were verified in dB
   (`docs/WHY_RESULTS_DIFFER_FROM_PAPER.md`).

## Main weaknesses

1. **Heavy rain:** the LSTM is worse than persistence there (RMSE 6.70 vs
   5.85 dB), and peaks are strongly under-predicted (Figs. 2, 3, 8).
2. **MAE gain is small:** only 3.2 % overall, and MAE is worse than
   persistence at +1 h and +2 h and in dry hours (Figs. 5, 8).
3. **Uncertainty is miscalibrated in both directions:** intervals are too
   wide in clear sky and cover only 46 % in heavy rain (Figs. 9, 10).
4. **Single seed:** this run has not been repeated with other seeds. Earlier
   seed spread on the stratified hourly protocol was 0.004-0.007 dB, so the
   0.159 dB gain over persistence is unlikely to be seed noise, but this is
   not verified on this protocol.
5. **Overfitting after epoch 5** (Fig. 1).

## Why our results differ from the paper

The full analysis is in `docs/WHY_RESULTS_DIFFER_FROM_PAPER.md`. In brief,
ranked by estimated effect:

1. **A different forecasting task (factor of about 20-35).** The paper's
   figures (1-minute sampling, a 5-minute stride, about 470,000 test samples)
   fit a minute-interpolated series forecast minutes ahead. On that task,
   no-model baselines already reach 0.01-0.05 dB. We forecast hourly values
   1-5 h ahead. This is the leading hypothesis; it cannot be proven without
   the authors' code.
2. **A label tail from the rain model (factor of about 1.5).** Our labels
   reach 25.6 dB; the paper's plots stop near 6.7 dB. How P.618 is applied to
   hourly rain is unspecified in the paper.
3. **Inconsistencies inside the paper.** Table 7's overall 0.026 does not
   follow from its own per-city values (their mean is 0.036, their RMS 0.063).
4. **No confirmed implementation bug.** RMSE is in physical dB, and the
   targets were recomputed independently.

## What we can and cannot claim

**We can claim** (observed):
* On reconstructed ITU-R labels from ERA5 hourly weather, a multi-task LSTM
  forecasts 39 GHz EPL 1-5 h ahead with RMSE 0.859 dB, 15.6 % below
  persistence, on a causal, leakage-checked test set of 195,000 values.
* The gain is largest at longer horizons and in light rain. The model fails
  on heavy rain and on extreme-event uncertainty.

**We cannot claim:**
* A reproduction of the paper's 0.026 dB, or any controlled comparison with
  it.
* Performance on real satellite links. The labels are ITU-R model outputs
  from reanalysis weather, not measured received power.
* That the result is robust across seeds (not yet repeated), or that the
  uncertainty is calibrated.
* Anything about the paper's unseen-city claims from this experiment. Those
  were evaluated only on the earlier stratified protocol
  (`docs/FINAL_RESEARCH_REPORT.md`).

**Hypotheses, not findings:** the minute-interpolation explanation of the
paper's number, and the rain-label explanation of its smaller tail.

## Suggested presentation order

1. **Fig. 12** (components) and **Fig. 11** (label distribution): what EPL
   is, and why rain dominates the problem.
2. **Fig. 1** (training): how the model was trained and selected.
3. **Fig. 6** (baselines): the headline result.
4. **Fig. 5** (horizon) and **Fig. 7** (cities): where the gain comes from.
5. **Fig. 2** (time series) and **Fig. 3** (scatter): what the forecasts look
   like, both a typical period and the worst event.
6. **Fig. 8** (precipitation classes) and **Fig. 4** (residuals): the
   heavy-rain weakness.
7. **Fig. 9** and **Fig. 10**: uncertainty calibration.
8. **Fig. 13**: context with the paper, stated as not comparable.

## Regenerate

From the repository root (Windows PowerShell):

```powershell
.venv\Scripts\python evaluation\plot_research_results.py                  # first run (refuses if results\figures exists)
.venv\Scripts\python evaluation\plot_research_results.py --overwrite      # replace this script's own figures
.venv\Scripts\python evaluation\plot_research_results.py --overwrite --export-docs   # also refresh docs\assets\figures\causal
```

The script needs the existing artefacts listed at the top of this guide and
does not need TensorFlow. Unavailable plots: none. All 13 requested figures
were generated from saved data.
