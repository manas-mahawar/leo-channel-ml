# Why our results differ from the paper (0.859 dB vs 0.026 dB)

Date: 2026-10-27. Question: why does our causal hourly model
(`results/causal_hourly_fig1heads_seed42/`, test RMSE **0.859 dB**) differ
from the **0.026 dB** reported by El-Sayed et al., EURASIP JWCN 2026:102
(Table 7, p. 13)?

No model was trained and no existing file or result was changed. New files:
this report, `evaluation/why_results_differ.py` (read-only measurements) and
its outputs in `results/why_results_differ/` (`epl_distribution.csv`,
`error_budget.json`, `aggregation.json`, `units_check.json`,
`city_ranking.csv`). The report builds on `docs/REPLICATION_GAP_AUDIT.md`
and `docs/RAIN_LABEL_DIAGNOSTIC.md` and does not repeat their full tables.

Evidence labels: **[Paper]** stated or plotted in the paper; **[Code]** read
in our source; **[Measured]** computed from files in this repository;
**[Hypothesis]** plausible but not proven.

## Summary

* **No implementation bug was found** that could explain the gap. RMSE is
  computed in physical dB. The targets were recomputed independently from the
  physics tables and match to 9e-7 dB, and nothing is scaled or
  inverse-transformed on the EPL path (Sect. 5).
* The gap is a factor of about **33**. Most of it is explained by **what is
  being forecast**: an hourly forecast 1-5 h ahead in our setup, versus,
  most probably, a minute-interpolated series 1-5 minutes ahead in the
  paper. On the latter, simple baselines already reach 0.01-0.05 dB.
* A second, smaller factor is the **label tail**. Our rain labels reach
  25.6 dB, while the paper's plots stop at about 6.2-6.7 dB. The 1 % of our
  test values above 6.7 dB cause 53 % of the model's squared error.
  Removing them lowers RMSE from 0.859 to 0.591 dB, which is still 23 times
  the paper.
* Some evidence **cannot be explained by any hypothesis**:
  * the paper's own Table 7 "Overall" (0.026) does not follow from its
    per-city values (mean 0.036, RMS 0.063);
  * its city ranking is roughly the reverse of ours (Spearman -0.55);
  * its error is almost flat from hour 1 to hour 5.

  Without the authors' data or code, these points cannot be settled.

## Ranked explanation

| Rank | Cause | Type | Estimated effect on RMSE | Evidence (Sect.) | Confidence |
|---|---|---|---|---|---|
| 1 | **Different forecasting task**: the paper most probably forecasts a 1-minute-interpolated series 1-5 *steps* ahead (minutes, whatever the "Hour" labels say), with a 5-min stride over the full series. We forecast hourly values 1-5 h ahead | Unspecified or contradictory method | **Factor about 20-35** (persistence 1.018 -> 0.028-0.048 dB) | 3.1 | High that the published numbers fit this reading; which reading the authors intended cannot be proven |
| 2 | **Label tail from the rain model**: P.618 applied with the hourly rate as R0.01. Our maximum is 25.6 dB; the paper's plotted maximum is about 6.2-6.7 dB | Unspecified method (not a coding error) | Factor about 1.5 at our horizon (0.859 -> 0.591 dB on targets <= 6.7 dB) | 1, 2, 4.3 | Medium |
| 3 | **Paper-internal inconsistencies** (Table 7 overall vs cities, Table 9 "improvement", 99.03 % vs 99.23 %): the 0.026 may not be reproducible even by the authors' own aggregation | Reporting | Unknown | 3.2 | High (arithmetic) |
| 4 | **Aggregation or scaling of the reported metric** (min-max-scaled RMSE, or mean of city mean errors) | Speculative | Could give numbers of the same order (0.029-0.034 for our model) | 3.3 | **Low**: contradicts the dB units stated throughout the paper, and does not reproduce the city pattern |
| 5 | Test-set composition: 10,000 stratified hours per city vs every hour | Protocol | Small (0.916 stratified vs 0.859 causal, same architecture) | 3.4 | High |
| 6 | Model or training differences | Implementation | Small; the architecture matches Fig. 1 (`multitask_fig1heads`) | gap audit E-F | High |
| - | Implementation bug in labels, units or metric | - | **None found** | 4, 5 | High |

## 1. EPL label definition and plotted ranges, city by city

**Definition.**
* [Paper] Eq. 6-7 (p. 6-7): EPL = Pr,fs - Pr,meas = L_atm + epsilon.
* [Paper] Sect. 3.1 (p. 8): "the excess path loss was computed as the sum of
  gaseous, cloud, and rain attenuation components".
* [Code] `physics/build_attenuation.py:156-169`:
  `epl_dB = gas + cloud + rain`, with epsilon = 0.
* [Measured] The sum is exact to 7e-15 dB (`units_check.json`).

These match. A non-zero epsilon (Eq. 7) would add noise to the labels and
make RMSE *larger*, so it cannot explain a smaller published error.

**What the paper shows.** The paper gives **no per-city label range for the
six training cities**:
* Fig. 3 (p. 13) pools all cities: true EPL runs from about **0.5** to about
  **6.2 dB**, and predictions reach about 6.7 dB.
* Fig. 4 (p. 14) pools about 470,000 test samples: values run from about
  0.6 to about 6.7 dB.
* Only the unseen cities have per-city values (Table 8 and Fig. 5): mean
  true EPL of 0.79-1.23 dB.

**Our labels** [Measured, `results/why_results_differ/epl_distribution.csv`],
full 2020-2024 hourly series, elevation 30 degrees:

| City | Mean | Min | p50 | p99 | Max | Gas / cloud / rain mean | Hours with rain |
|---|---|---|---|---|---|---|---|
| Athens | 0.92 | 0.47 | 0.78 | 4.07 | 18.6 | 0.76 / 0.06 / 0.11 | 7.6 % |
| Bergen | 1.54 | 0.51 | 0.99 | 6.64 | 16.7 | 0.75 / 0.22 / 0.57 | 42.5 % |
| Budapest | 1.11 | 0.46 | 0.85 | 6.14 | 23.0 | 0.75 / 0.11 / 0.24 | 13.7 % |
| Chilbolton | 1.21 | 0.53 | 0.94 | 5.95 | 18.8 | 0.78 / 0.16 / 0.27 | 19.1 % |
| Vigo | 1.67 | 0.60 | 1.08 | 9.92 | 25.6 | 0.90 / 0.19 / 0.58 | 27.1 % |
| Warsaw | 1.10 | 0.50 | 0.86 | 5.55 | 19.5 | 0.73 / 0.13 / 0.23 | 16.0 % |
| **All** | **1.26** | **0.46** | **0.91** | **6.76** | **25.6** | | |
| Paper (pooled, Figs. 3-4) | - | about 0.5 | - | - | about 6.2-6.7 | - | - |
| Paper unseen cities (Table 8) | 0.79-1.23 | - | - | - | - | - | - |

**Comparison:**
* **The clear-sky floor agrees.**
  * Our minimum EPL of 0.46-0.60 dB is gas plus a little cloud, and it
    matches Fig. 3's lowest values of about 0.5 dB.
  * Gas scales roughly with 1/sin(elevation): at 45 degrees our floor would
    be about 0.33-0.42 dB, and at 20 degrees about 0.67-0.88 dB.
  * So the floor is consistent with an elevation near our 30 degrees
    [Hypothesis; Fig. 3 is read by eye, and the paper does not state the
    elevation].
* **The tail disagrees.** Our pooled 99th percentile (6.76 dB) is about the
  paper's *maximum*. Our maximum is about 4 times larger.
* **The mean is somewhat higher.** Our unseen-city mean of period means is
  1.153 dB vs 0.911 dB in Table 8 (`docs/REPLICATION_GAP_AUDIT.md` Sect. C;
  the periods differ). The trained-city means of 0.92-1.67 dB are mostly
  driven by rain.
* **Conclusion:** the difference sits in the **rain (and possibly cloud)
  component**, not in gas or elevation.

## 2. Distribution of our EPL targets (causal test set)

[Measured] 39,000 origins x 5 horizons = 195,000 target values
(`epl_distribution.csv`).

| Set | Mean | Std | p1 | p5 | p25 | p50 | p75 | p90 | p95 | p99 | p99.9 | Max |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Causal test targets | 1.264 | 1.209 | 0.58 | 0.64 | 0.77 | 0.93 | 1.15 | 2.07 | 3.39 | 6.62 | 12.66 | 25.63 |
| Full series, 6 cities | 1.257 | 1.208 | 0.56 | 0.61 | 0.75 | 0.91 | 1.14 | 2.17 | 3.39 | 6.76 | 12.44 | 25.63 |

Share of values above a threshold:

| Set | > 2 dB | > 5 dB | > 6.2 dB | > 6.7 dB | > 10 dB | > 20 dB |
|---|---|---|---|---|---|---|
| Causal test targets | 10.43 % | 2.21 % | 1.23 % | 0.98 % | 0.26 % | 0.013 % |
| Full series | 11.10 % | 2.24 % | 1.28 % | 1.02 % | 0.27 % | 0.005 % |

Per city, the share of test targets above 6.7 dB is:

| City | Share above 6.7 dB |
|---|---|
| Athens | 0.38 % |
| Warsaw | 0.43 % |
| Chilbolton | 0.86 % |
| Budapest | 1.02 % |
| Bergen | 1.22 % |
| Vigo | 1.98 % |

The test set is representative of the full series: their distributions are
nearly identical.

## 3. Could 0.026 dB come from a different protocol, aggregation, scaling or label?

### 3.1 Sampling and forecast horizon (strongest evidence)

[Paper] Statements that point to a minute-level task:
* "Sequence length 60 min" (Tables 5-6).
* Fig. 2: "1-minute uniform sampling", "Prediction Horizon: 5 steps",
  "Stride: 5 minutes".
* Table 4: a "Minute-of-hour" feature.
* The Fig. 4 x-axis runs to about 470,000 samples.

[Paper] Statements that point to an hourly task: "hourly samples",
"Prediction horizon 5 h", "Hour 1 ... Hour 5" (Table 7).

[Measured] (`results/replication_audit/minute_protocol_baselines.csv`):
* Interpolating the full 2020-2024 series to 1 minute with a 5-minute stride
  gives 473,484 test windows, which matches Fig. 4.
* On that task, no-model baselines score:
  * persistence: 0.028-0.048 dB per city;
  * linear extrapolation: 0.008-0.014 dB per city.
* Both are the same order as the paper's 0.026 dB. On our hourly task,
  persistence scores 1.018 dB.
* Interpolated minute inputs also contain the next hourly value, so the
  target is partly visible in the input (documented leak,
  `tests/test_causal.py`, `TestCausalMinuteProxy.test_existing_interpolation_uses_the_next_hour`).

**This one difference accounts for a factor of about 20-35.** It is the
leading explanation. Two caveats:
* It is unprovable without the authors' code.
* It does **not** reproduce two features of the paper:
  * Table 7's flat horizon profile: 0.026 -> 0.027 from step 1 to 5, while
    minute persistence grows 5-fold (0.009 -> 0.042 in Athens);
  * the city ranking (Sect. 3.5).

### 3.2 The paper's own numbers are not internally consistent

[Measured from published values]

* **Table 7 (p. 13).** The six cities have equal sample sizes (10,000 each,
  Table 3), and their "Average" values are 0.017, 0.148, 0.005, 0.005, 0.003
  and 0.039. Their mean is 0.036, their RMS 0.063 and their median 0.011.
  **None of these equals the reported overall 0.026.** A pooled 0.026 would
  need Bergen to hold at most about 3 % of the test values (gap audit, H).
* **Table 9.** The single-task EPL model (0.015) is *better* than the
  multi-task model (0.026). Yet the row "Improvement 40 %" and the text
  credit the multi-task model.
* **Calibration.** It is reported as 99.03 % (Sect. 4.1, Table 11) and as
  99.23 % (PICP, Sect. 4.1).
* **Figure 3 vs the stated sample size** [Hypothesis, read by eye]. Fig. 3
  shows dense arms with errors of about 1-1.4 dB, for example true about
  1.5-2.9 dB predicted about 1.5 dB. With the stated 10,000 samples per
  city, the test set would hold about 9,000 windows x 5 = 45,000 values. An
  RMSE of 0.026 dB allows a total squared error of only about 30 dB², which
  means at most about 30 such points. With about 2.4 million values (the
  Fig. 4 reading), about 1,600 such points would be allowed. Fig. 3 is
  therefore more consistent with the large minute test set than with the
  10,000-sample description.

### 3.3 Alternative aggregations and scalings of our own predictions

[Measured] `results/why_results_differ/aggregation.json`. These are the same
model predictions scored in different ways (no retraining):

| Aggregation | Model | Persistence | Near 0.026? |
|---|---|---|---|
| Pooled RMSE, dB (ours, standard) | 0.859 | 1.018 | no |
| Mean of the six city RMSEs | 0.839 | 0.997 | no |
| RMS of city RMSEs | 0.859 | 1.018 | no |
| RMSE of the 5-step-averaged forecast | 0.672 | 0.851 | no |
| RMSE / training std (EPL z-scored) | 0.734 | 0.870 | no |
| MAE | 0.335 | 0.346 | no |
| Median absolute error | 0.089 | 0.055 | no |
| RMSE / target range (min-max scaled to [0, 1]) | **0.034** | 0.041 | same order |
| Mean of abs(city mean error) (Table 8 style) | **0.029** | 0.0006 | same order |
| abs(overall mean error) | 0.002 | 0.0005 | no |

Two non-standard metrics land near 0.026. Neither is supported by evidence:

* **Min-max scaling.**
  * Against it: the paper labels Table 7 "RMSE in dB" and gives Sharpness
    as "0.0350 dB", and Fig. 3 shows errors of about 1 dB on a dB axis.
  * Against it: per-city min-max-scaled RMSE spans only a factor of 2
    (0.035-0.074), while Table 7 spans a factor of 50 (0.003-0.148).
* **A bias-type metric** (abs(mean pred - mean true)). The paper does use
  this in Table 8 for unseen cities. Against it for Table 7:
  * persistence would score almost 0 on it;
  * our per-city biases (Vigo -0.057, Warsaw +0.052, Bergen -0.027) do not
    follow the paper's pattern.

Both remain **[Hypothesis, low confidence]**. The coincidence in magnitude
alone is not evidence.

### 3.4 Sampling of the test set

On the same architecture:
* the stratified hourly test set (10,000 sampled hours per city) gave
  0.916 +- 0.006 dB (3 seeds);
* the complete causal test set gave 0.859 dB.

The choice of sampling changes RMSE by under 10 %.

### 3.5 City ranking: unexplained

[Measured] `results/why_results_differ/city_ranking.csv`. Spearman rank
correlation with the paper's Table 7:

| Our quantity | Spearman with Table 7 |
|---|---|
| Model RMSE | -0.55 |
| Persistence RMSE | -0.67 |
| EPL standard deviation | -0.35 |
| Share of rainy hours | +0.09 |

* For us, Vigo (1.09 dB) and Budapest (1.04 dB) are the hardest cities and
  Athens is the easiest (0.55 dB).
* In the paper, Vigo (0.003) and Budapest (0.005) are the easiest and Bergen
  the hardest (0.148).
* Bergen has the most rainy hours in our data (42.5 %). Vigo is the
  second-rainiest (27 %), yet it is the easiest city in the paper.

No reading of protocol, label or aggregation that we tested reproduces this
pattern. The paper attributes Bergen's error to terrain (Sect. 4.7), but no
terrain information enters the inputs or the labels (Table 2).

### 3.6 Label definition

* An elevation above 30 degrees lowers the level but leaves a maximum of
  16 dB even at 90 degrees.
* Liquid rain vs total precipitation changes the label statistics by less
  than 1 %.
* A different P.618 time percentage would shrink the tail. For example, at
  p = 0.1 % Vigo's maximum falls from 20.6 to 7.4 dB
  (`docs/RAIN_LABEL_DIAGNOSTIC.md` Sect. 3) [Hypothesis].

However, even if the label tail were removed entirely, the **hourly** task
could not reach 0.026 dB:

| Subset of our test set | Model RMSE (dB) | Persistence RMSE (dB) |
|---|---|---|
| Rain-free windows (no rain at the origin or at any target hour; 67 % of origins) | 0.179 | 0.083 |
| Targets <= 6.7 dB only | 0.591 | 0.853 |

In the rain-free windows, the 1-5 h variation of gas and cloud attenuation
alone already gives persistence an error 3 times the paper's. The label
definition therefore explains part of the gap, but not its magnitude
[Measured, `error_budget.json`].

## 4. Paper equations, tables and figures vs our implementation

| Item | Paper | Ours | Verdict |
|---|---|---|---|
| Gas, Eq. 1-2 (p. 4) | gamma = gamma_o + gamma_w. Eq. 2 is printed as 0.1820 f [N''_o/(N_o^2 + N''_o^2) + ...] (garbled in the PDF) | P.676-12 Annex 1 line-by-line gamma_o, gamma_w with dry-air pressure, Annex 2 equivalent heights, / sin(el) (`physics/build_attenuation.py:55-80`) | The printed Eq. 2 is **not** the P.676 formula, which is gamma = 0.1820 f N''(f). We follow the Recommendation the paper cites; our dry-sky floor matches Fig. 3 (Sect. 1). Agrees with `itur` to 0.005 dB (`RAIN_LABEL_DIAGNOSTIC.md`) |
| Gas, Eq. 8 (p. 9) | mu_a = gamma_oxygen x L_eff, sigma_a = f(T, P, RH) x L_eff | mu: full gas attenuation; sigma: trailing 24-h std (`data_pipeline/causal_forecast_source.py`) | Ambiguous: Eq. 8 mentions oxygen only, and f is unspecified. Input features only, no effect on labels |
| Rain, Eq. 3 (p. 4) | gamma_R = k R^alpha, k and alpha from P.838 | P.838-3, 39 GHz, 30 degrees, tau = 45 degrees; k = 0.4136, alpha = 0.8617 (`build_attenuation.py:121-123`) | Match (checked by hand, 4e-16) |
| Rain path (Sect. 3.1) | "P.838 and P.618, rain height from P.839" | P.618-13 steps 1-9, hourly R used as R0.01, no step 10 (`build_attenuation.py:109-150`) | Equations exact (vs `itur` 0.0 dB). **How** P.618 is applied to hourly rain is unspecified; this is the main label ambiguity |
| Cloud, Eq. 4 (p. 5) | A_c = LWP K_l / sin(theta) | Same formula; K_l(39 GHz, 0 degC) = 1.233 from P.840-8; LWP from a fitted RH + precipitation proxy (`build_attenuation.py:91-103`, `physics/lwp_proxy_calibration.json`) | Formula matches. The proxy is unspecified ("simple proxy derived from relative humidity and precipitation"); our fit to ERA5 water content is our choice |
| Residual epsilon, Eq. 7 | EPL = L_atm + epsilon | epsilon = 0 | Matches Sect. 3.1 (sum of three). A non-zero epsilon would *raise* RMSE |
| Elevation, Eq. 9 (p. 9) | Geometric formula; Sect. 3.1 says "fixed elevation angle"; no value, altitude or distance given | 30 degrees (`config.py`, `ELEVATION_DEG`) | Unspecified. The Fig. 3 floor (about 0.5 dB) is consistent with about 30 degrees [Hypothesis] |
| NLL, Eq. 10 | mean 0.5(log sigma^2 + (y - mu)^2/sigma^2) | `training/train_paper_model.py:53-57` | Match |
| Total loss, Eq. 11 | 1.0 MSE + 1.0 NLL | Same | Match |
| Normalisation (p. 9; Fig. 2) | "standardized using the statistics of the training set only"; Fig. 2 "normalized by location" | One feature scaler on training input rows; weather targets on training target rows; **EPL not scaled** | Ambiguous (global vs per location). Even if the paper had reported RMSE on z-scored EPL, our equivalent would be 0.73, not 0.026 (Sect. 3.3) |
| Table 7 | "RMSE in dB", hours 1-5 | dB, horizons +1..+5 h | Same units. The paper's overall row is inconsistent (Sect. 3.2) |
| Fig. 3 / Fig. 4 | True EPL about 0.5-6.7 dB | Ours 0.46-25.6 dB | Floor matches, tail differs (Sect. 1) |

## 5. Is RMSE in physical dB? Any scaling or inverse-transform mismatch?

[Code + Measured] **The RMSE is in dB, and there is no mismatch.**

* **Targets.** `CausalHourlySource.windows` reads `epl_dB` directly from the
  physics table (`data_pipeline/causal_forecast_source.py:59, 89`), and no
  scaler is applied to EPL (`training/train_causal_hourly.py`; the scaler is
  used only for X and the weather targets).
* **Model output.** The EPL head is a linear `Dense(10)` with no activation
  (`training/train_paper_model.py:79`), so the predicted means are already in
  dB. `epl_mean = pred[:, :5]` is saved without any transform.
* **Independent recomputation** (`units_check.json`):
  * the stored targets equal the physics-CSV `epl_dB` at origin + 1..5 h to
    within 8.7e-7 dB (float32 rounding);
  * persistence equals the CSV value at the origin to the same precision;
  * RMSE recomputed from the CSV targets is 0.85922 dB, identical to the
    report.
* **Scale sanity.**
  * The prediction mean (1.2615 dB) is close to the target mean (1.2635 dB).
  * The prediction range is 0.49-8.90 dB, against a target range of
    0.50-25.63 dB. The model under-predicts the extreme tail, which matches
    the poor heavy-rain results and is not a unit error.
* **Metric.** `evaluation/evaluate_causal_model.py:37-45` computes
  sqrt(mean squared error) over all (origin, horizon) values in dB. Model
  and baselines are scored on identical keys.

## 6. Confirmed bugs vs unavoidable differences

**Confirmed implementation bugs in our current pipeline: none.**
* Checked in this report: units, target alignment, the EPL sum, the metric,
  and the equations in Sect. 4.
* Bugs found and fixed earlier, all now covered by tests:
  * P.840 temperature given in Kelvin instead of Celsius;
  * P.676 ignoring the measured temperature and pressure;
  * total instead of dry-air pressure;
  * snow counted as rain;
  * the minute-split boundary leak.
* One known minor imprecision is kept deliberately: our water-vapour
  pressure omits P.453's enhancement factor, a 0.005 dB effect.

**Unavoidable differences** (need the authors' data or code):

| Difference | Why we cannot resolve it |
|---|---|
| Forecasting task (minute vs hour steps; full series vs 10,000 samples) | The paper's text, tables and Fig. 2/Fig. 4 contradict each other |
| How P.618 is applied to hourly rain (R0.01, time percentage, path factors) | Not described |
| LWP proxy formula | "Simple proxy" not given |
| Elevation value, polarisation | Not given |
| sigma feature definition (Eq. 8, f unspecified) | Not given |
| Overall-metric definition (Table 7) | Not derivable from the published city values |
| Data period ("five consecutive years") | Years not given |
| The authors' label table | "Available on request" |

**Paper-side inconsistencies** (not ours to fix): Table 7 overall vs cities,
the Table 9 improvement sign, 99.03 % vs 99.23 %, the printed Eq. 2, the
Table 8 vs Fig. 5 errors for Berlin and Stockholm, and surface vs mean
sea-level pressure.

## 7. What would settle it

1. **The authors' label table for one city**, or the EPL for a single hour
   with a known rain rate. This identifies the rain-model reading (Sect. 3.6)
   in one row.
2. **The authors' test-window definition** (step length and stride). This
   resolves cause 1.
3. **Their evaluation script**, to settle the Table 7 overall definition.

Without these, our numbers should be reported as an independent
reconstruction and never as a replication of 0.026 dB. The closest
measurable comparison already in the repository is the minute-protocol
baseline set (`results/replication_audit/minute_protocol_baselines.csv`).
A model trained on that protocol remains an optional GPU-scale experiment
(`docs/MINUTE_REPLICATION_IMPLEMENTATION.md`). It would measure
interpolation, not forecasting skill.

No physical parameter was tuned in this investigation.

Reproduce (writes only to a new folder; refuses to overwrite):

```powershell
.venv\Scripts\python evaluation\why_results_differ.py
```
