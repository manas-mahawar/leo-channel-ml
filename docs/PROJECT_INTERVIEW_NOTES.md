# Project interview notes: LEO satellite path-loss forecasting (replication study)

All numbers are from this repository's results (see FINAL_RESEARCH_REPORT.md).
"Paper" values are the published reference values, not ours.

> **Historical notes (earlier stratified protocol).** The current main result is
> the causal hourly evaluation: RMSE 0.859 vs persistence 1.018 dB on 195,000
> test values. For interviews, use
> [PROJECT_EXPLAINED_FROM_SCRATCH.md](PROJECT_EXPLAINED_FROM_SCRATCH.md), Sect. I.
> Describe the work as an independent implementation, not a replication or an
> exact reproduction.

## 30-second pitch

I independently implemented a 2026 journal paper that uses a multi-task LSTM to forecast how
much weather weakens a 39 GHz satellite link over the next five hours. I
built the whole pipeline from public data: ERA5 weather, physics-based ITU-R
attenuation labels, the paper's dataset and model, baselines, ablations,
repeated-seed experiments and an automated results audit. The paper reports
0.026 dB error; my implementation gives about 0.92-0.94 dB, roughly 10 %
better than simply repeating the current value, and I traced why the
published number is not achievable for a genuine 5-hour forecast.

## Two-minute technical explanation

Signals from low-Earth-orbit satellites at Q-band are attenuated by gases,
clouds and rain. The paper predicts this excess path loss (EPL) from the past
60 hours of weather at a location. I downloaded five years of hourly ERA5
weather for six European cities, computed EPL labels with ITU-R models
(P.676 gas, P.840 cloud, P.838/P.618 rain) and built 60-hour input windows
with 14 features, including the physics-based attenuation terms. I split each
city's data chronologically with a purge gap so no hour leaks between
training, validation and test, and fitted scalers on training data only.

The model is a three-layer LSTM with two heads: one forecasts five weather
variables, the other outputs a mean and variance for EPL at each of the next
five hours, trained with Gaussian negative log-likelihood. On the April-December
2024 test period it reaches 0.940 +- 0.004 dB RMSE over three seeds; a variant
with an extra dense layer per head, as drawn in the paper's figure, reaches
0.916 +- 0.006 dB. Persistence gives 1.042 dB and is still better one hour
ahead. I ran the paper's ablations with three seeds and found no measurable
benefit from the physics features and only a tiny one from multi-task
training. On ten unseen cities, the paper's metric looked similar to theirs,
but hourly errors were worse than persistence; switching to sea-level
pressure fixed two high-altitude cities. Finally, I showed the paper's 0.026 dB
is reachable trivially if the data are interpolated to one-minute steps.

## Problem, data, pipeline, model

* **Problem:** forecast EPL (dB) 1-5 h ahead at 39 GHz, 30 deg elevation, with uncertainty.
* **Data:** ERA5 via Open-Meteo, 2020-2024, six training cities; 2023-2024 for
  ten unseen cities; 10,000 stratified anchor hours per city.
* **Labels:** EPL = gas + cloud + rain attenuation from ITU-R models (`itur`);
  a fitted relative-humidity/precipitation proxy for cloud liquid water.
* **Pipeline:** download -> labels -> windows (60 h in, 5 h out) ->
  chronological 70/15/15 split with 65-h purge -> train -> evaluate ->
  figures/tables -> audit.
* **Model:** 3 x LSTM(256), dropout 0.2; weather head (MSE) and Gaussian EPL
  head (NLL); Adam 1e-4, batch 128, early stopping (patience 6).
* **Metrics:** RMSE and MAE in dB per city and horizon; 95 % interval coverage,
  NLL, interval width; persistence and climatology baselines on identical windows.

## Strongest verified results

* Main model 0.940 +- 0.004 dB, Fig. 1-head model 0.916 +- 0.006 dB,
  persistence 1.042 dB (test set, three seeds).
* The Fig. 1-head gain is consistent: better in all three seeds by
  0.021-0.026 dB (0.064 dB at hour 1).
* Physics (ITU-R) input features: no measurable effect (paired differences
  change sign across seeds).
* MSL pressure: Davos error 2.25 -> 0.05 dB, Madrid 0.99 -> 0.04 dB (own weather).
* All metrics independently recomputed by an audit script (127 checks pass);
  12 unit and data-integrity tests pass; training is deterministic.

## Why the extra-head model did better (what I can and cannot say)

Measured: it wins in every seed, on validation as well as test, and the gain
is largest at hour 1 and (in the seed-42 runs) in Vigo, where it turns a
loss to persistence into a win; it also predicts higher peaks (8.3 vs 5.8 dB,
seed 42). Plausible explanation (hypothesis, not tested): the extra ReLU layer
gives each task its own non-linear transformation of the shared LSTM state,
so the EPL head can represent sharp rain-driven changes without competing
with the weather head. It still under-predicts heavy rain.

## Why the published RMSE was not reproduced

* Even persistence has 1.04 dB RMSE on these labels, because rain changes EPL
  by several dB within hours; 0.026 dB is not plausible for a real 5-hour forecast.
* On minute-interpolated data, linear extrapolation reaches 0.013 dB and
  persistence 0.033 dB, so the paper's number fits a minutes-ahead reading of
  its Fig. 2. This is a hypothesis; the authors' data were not available.
* Other differences: unspecified cloud proxy and feature definitions,
  different ITU-R versions, possibly a different pressure variable.

## What repeated seeds and ablations taught me

* Seed-to-seed spread is about 0.004-0.007 dB, larger than several of the
  paper's reported ablation gains (0.002 dB for ITU-R features).
* Seed 42 was the best seed for three of four variants, so single-run
  numbers are slightly optimistic.
* Paired comparisons (same seed, only the variant changes) are more
  informative than comparing single runs.

## The minute-level baseline lesson

When hourly data are linearly interpolated to minutes, the series is a
straight line within each hour. Linear extrapolation from the last two
minutes is then exact for any anchor not on the hour (92.5 % of windows), so
a "good" score measures the interpolation, not forecasting skill. Lesson:
always ask how the data were constructed before trusting a low error, and
always compare against simple baselines on the same samples.

## Limitations and improvements

* Labels are model outputs, not link measurements; the cloud proxy is weak.
* Heavy rain is rare (66 of 42,000 training windows above 14 dB) and badly
  under-predicted; intervals under-cover rain (37 % at >= 2.5 mm/h).
* One test period, few cities, three seeds, no significance tests.
* Improvements: heavy-tailed or quantile losses, re-weighting rare events,
  post-hoc recalibration on validation data, more seeds, real link data.

## Likely interview questions

1. **What exactly does the model predict?** The excess path loss in dB at each
   of the next five hours (mean and variance of a Gaussian) plus five weather
   variables for those hours.
2. **What is EPL and how did you get labels?** Gas + cloud + rain attenuation
   on a 39 GHz, 30 deg slant path, computed from weather with ITU-R models
   (P.676, P.840, P.838/P.618/P.839) through the `itur` library.
3. **How did you prevent data leakage?** Chronological split per city, a
   65-hour purge gap (60 input + 5 target hours) so no hour appears in two
   splits, scalers fitted on training windows only, sigma features computed
   from past hours only; all checked by automated tests.
4. **Why not a random split?** Neighbouring windows overlap in time, so a
   random split would put almost identical windows in training and test and
   overstate accuracy.
5. **Why persistence as a baseline?** It is the simplest forecast and very
   strong at short horizons; a model that cannot beat it adds little. Ours
   loses to it at hour 1 and wins from hour 2.
6. **How do you know the model is not overfitting?** Early stopping on a
   separate validation period, best weights restored; validation and test
   rankings of the variants agree; test is a later, unseen period.
7. **Why a Gaussian NLL loss?** It learns a mean and a variance, giving
   prediction intervals. Its weakness here: the model can widen the variance
   instead of fitting rare large values.
8. **Is the model well calibrated?** Overall 95 % coverage is 96.6 %, but this
   hides over-wide intervals in dry hours (99.4 %) and too-narrow ones in rain
   (86 %, 37 % in heavy rain); nominal 50 % intervals cover 78 %.
9. **What did the ablations show?** Removing the ITU-R features changed
   nothing measurable; multi-task vs EPL-only differed by 0.006 dB; the
   weather-only model forecast weather better than the multi-task model.
10. **Does "no effect" mean the physics features are useless?** No. It means
    this model did not benefit; the information may be redundant with the raw
    weather inputs for this network. A rain-only ablation or a smaller model
    would test that.
11. **Why three seeds and paired comparisons?** To measure run-to-run noise;
    pairing by seed removes shared variation so small consistent effects are
    visible. Three seeds are still few, so I avoid strong significance claims.
12. **How does the unseen-city method work?** Inputs for a new city are a
    distance-weighted average (softmax(-d/500 km)) of its three nearest
    training cities' weather, with the new city's coordinates.
13. **Why did Davos and Madrid fail?** Their surface pressure is far below
    anything seen in training; with sea-level pressure they are in range and
    the errors almost vanish. I cannot say which variable the authors used.
14. **Why is the paper's unseen-city metric misleading?** It compares
    period means, so hourly errors cancel; hourly MAE was worse than local
    persistence.
15. **How did you make results reproducible?** Fixed seeds and TensorFlow op
    determinism (retrain matches to 1e-7), per-run config files, pinned
    requirements, a queue that never overwrites finished runs, an independent
    audit that recomputes all metrics, and unit tests.
16. **What would you do with more time?** Get real link measurements or the
    authors' data, model heavy rain better, recalibrate the intervals, and run
    more seeds.
17. **What was the hardest bug?** The cloud model received temperature in
    Kelvin where the library expects Celsius, making cloud attenuation about
    5.6 times too small; I caught it by checking one coefficient by hand.
18. **Why might an LSTM not be the best choice here?** Most skill comes from
    the current state; heavy rain is rare and abrupt. Simpler models or
    hybrid persistence-plus-correction approaches could be competitive.
