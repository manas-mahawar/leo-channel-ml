# Resume bullet options

Pick one version. Each bullet is supported by the repository's verified
results (`docs/PROJECT_FINAL_STATUS.md`). Do not describe the work as
published, deployed, an exact reproduction, or validated on measured satellite
data.

## Option A (three bullets)

* Built an end-to-end Python/TensorFlow pipeline that forecasts 39 GHz LEO
  satellite excess path loss 1-5 h ahead:
  * hourly ERA5 weather for 6 European sites, 2020-2024;
  * ITU-R gas, cloud and rain attenuation labels;
  * a 1.47M-parameter multi-task LSTM with Gaussian uncertainty.
* Designed a strictly causal evaluation: chronological gapped splits,
  train-only scaling, and unit tests that perturb future data.
  * The LSTM reduced RMSE by 15.6 % against persistence (0.859 vs 1.018 dB)
    on 195,000 test values.
  * Failure analysis documented heavy-rain under-prediction and only 46 %
    conditional interval coverage.
* Independently implemented a 2026 EURASIP JWCN model and audited why its
  reported 0.026 dB RMSE could not be matched. The evidence points to a
  minute-interpolated protocol on which trivial baselines score 0.01-0.05 dB.
  No implementation bug was found (127 automated audit checks, 41 unit tests).

## Option B (two bullets)

* Developed a leakage-checked deep-learning forecaster (multi-task LSTM with
  uncertainty) for Q-band satellite link attenuation from ERA5 weather and
  ITU-R physics. It cut RMSE by 15.6 % against a persistence baseline in a
  causal, multi-city evaluation (195,000 forecasts).
* Built reproducible evaluation tooling: identical-key baselines, calibration
  analysis, an audit script recomputing every metric, unit tests and 13
  publication-quality figures. Documented the limitations (simulated labels,
  single seed, heavy-rain weakness) and the discrepancies with the reference
  paper.

## Option C (one bullet)

* Implemented and causally evaluated a published multi-task LSTM for 39 GHz
  satellite path-loss forecasting (ERA5 -> ITU-R labels -> LSTM). It is
  15.6 % better than persistence in RMSE, but worse in heavy rain. Also
  explained why the paper's 0.026 dB error is not comparable.

## Earlier stratified-protocol facts (usable as extra detail)

15 training runs on the earlier stratified protocol (ablations and three
seeds per key variant): Fig. 1-head model 0.916 +- 0.006 dB vs persistence
1.042 dB, on 8,916 test windows. These are not the same test set as the causal
result; do not combine the two numbers.
