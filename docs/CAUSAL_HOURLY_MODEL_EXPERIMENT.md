# Causal hourly model experiment (causal_hourly_v1, seed 42)

Date: 2026-10-27. One controlled training run of the paper-faithful
architecture `multitask_fig1heads` on the strictly causal protocol
`causal_hourly_v1` (`docs/CAUSAL_FORECASTING_INVESTIGATION.md`). This is an
independent reconstruction; it is **not comparable with the paper's 0.026 dB**
(different and partly unknown protocol and labels).

## 1. Audit before training (verdict: pass)

| Check | Result |
|---|---|
| Split separation | train / validation / test origins are chronological per city; the next split's first input row is after the previous split's last target row (>= 65 h); tested on synthetic data (`tests/test_causal.py`) |
| Feature timing | all 14 features use rows <= the origin; sigma features are trailing 24-h windows; future-perturbation test passes |
| Scalers / imputation | feature scaler on training input rows only; weather-target scaler on training target rows only; no imputation (windows with missing rows are excluded; none occur) |
| Labels | unchanged ITU-R labels; rain, gas and cloud reproduce independent references (`docs/RAIN_LABEL_DIAGNOSTIC.md`); the label-tail ambiguity versus the paper remains, but no construction error |
| Test keys | the model's 39,000 test origins and targets are identical to `results/causal_baseline_audit/aligned_keys_hourly.npz` (checked in the dry run and again at evaluation) |
| Output compatibility | EPL head (b, 10) = 5 means + 5 log-variances for +1..+5 h; weather head (b, 5, 5); losses Gaussian NLL (Eq. 10) and MSE, unchanged |
| Pipeline mismatch | the existing scripts read the stratified hourly arrays; a new adapter `training/train_causal_hourly.py` builds the causal arrays (no change to existing code) |
| Resources | arrays 0.91 GB; peak working set 1.76 GB after data build, 3.21 GB during training (10.8 GB free); disk about 50 MB |
| Power | the laptop was on battery (34 %); training started only after the charger was connected (your choice) |

The dry run printed: train 183,798, validation 39,000, test 39,000 windows;
X (n, 60, 14), weather (n, 5, 5), EPL (n, 5); all finite; estimate 16 min
per epoch on mains power.

## 2. Configuration

* Data: six training cities, every hourly origin with a complete window;
  train 2020-01-04 10:00..2023-07-03 18:00 (30,633 origins per city),
  validation 2023-07-06 11:00..2024-04-02 06:00 (6,500), test
  2024-04-04 23:00..2024-12-31 18:00 (6,500).
* Model: 3 x LSTM(256), dropout 0.2, heads Dense(256, ReLU) -> Dense(25) /
  Dense(10); 1,468,707 parameters.
* Training: Adam 1e-4, batch 128, up to 50 epochs, early stopping on
  validation loss (patience 6, best weights restored), loss weights 1.0/1.0,
  log-variance clip [-12, 6], seed 42, TensorFlow op determinism.
* Differences from the earlier stratified hourly runs (documented, not
  tuned): all hourly origins instead of 10,000 stratified anchors per city
  (4.4x more training windows); scalers fitted on training rows instead of
  overlapping training windows.
* Model selection used validation data only; the test set was predicted once,
  after training. Outputs: `models/causal_hourly_fig1heads_seed42/`,
  `results/causal_hourly_fig1heads_seed42/`.

## 3. Training

11 epochs, best epoch 5 (validation loss -0.3006), 3.08 h on mains power
(about 15-19 min per epoch, 0.63-0.78 s per batch), peak memory 3.21 GB.
After epoch 5 the training EPL loss kept falling (-0.75 -> -1.01) while the
validation EPL loss rose (-0.56 -> -0.28): the model becomes overconfident on
training data, and early stopping restored epoch 5.
Figure: `results/causal_hourly_fig1heads_seed42/training_curves.png`.

## 4. Test results (39,000 origins x 5 horizons, identical keys for all methods)

| Method | RMSE (dB) | MAE (dB) | RMSE +1 / +2 / +3 / +4 / +5 h | MAE +1 .. +5 h |
|---|---|---|---|---|
| **Model** | **0.859** | **0.335** | 0.627 / 0.796 / 0.893 / 0.946 / 0.985 | 0.211 / 0.302 / 0.351 / 0.391 / 0.422 |
| Persistence | 1.018 | 0.346 | 0.648 / 0.921 / 1.066 / 1.147 / 1.209 | 0.191 / 0.295 / 0.368 / 0.418 / 0.459 |
| Recent 24-h mean | 1.126 | 0.494 | 1.065 / 1.102 / 1.131 / 1.155 / 1.174 | 0.463 .. 0.521 |
| Linear extrapolation | 2.519 | 0.718 | 0.913 .. 3.647 | 0.267 .. 1.154 |

Model vs persistence: RMSE -0.159 dB (**-15.6 %**), MAE -0.011 dB (**-3.2 %**).
By horizon the RMSE gain grows from -0.021 dB (+1 h) to -0.223 dB (+5 h); the
MAE is *worse* than persistence at +1 h (0.211 vs 0.191) and +2 h (0.302 vs
0.295) and better from +3 h.

Per city (model / persistence, RMSE and MAE in dB):

| City | RMSE | MAE |
|---|---|---|
| Athens | 0.547 / 0.665 | 0.153 / 0.150 |
| Bergen | 0.830 / 0.959 | 0.458 / 0.474 |
| Budapest | 1.043 / 1.285 | 0.338 / 0.363 |
| Chilbolton | 0.789 / 0.962 | 0.350 / 0.372 |
| Vigo | 1.090 / 1.216 | 0.399 / 0.413 |
| Warsaw | 0.737 / 0.894 | 0.312 / 0.304 |

By precipitation at the target hour (observed in the reanalysis; outcome-based,
diagnostic only), RMSE / MAE:

| Class (values) | Model | Persistence |
|---|---|---|
| Dry (153,853) | 0.328 / 0.159 | 0.510 / 0.129 |
| Light 0-2.5 mm/h (39,672) | 1.241 / 0.819 | 1.678 / 1.035 |
| Heavy >= 2.5 mm/h (1,475) | **6.704 / 5.660** | **5.845 / 4.495** |

Uncertainty: nominal 50 / 80 / 90 / 95 / 99 % intervals cover 81.9 / 94.1 /
96.3 / 97.4 / 98.5 %; mean 95 % width 2.39 dB; NLL -0.734. Coverage of the
95 % interval: dry 99.5 % (width 1.66 dB), light rain 91.0 % (4.95 dB),
heavy rain **46.0 %** (9.45 dB); by true EPL 0-1 dB 99.4 %, 5-10 dB 67.0 %,
>= 10 dB 28.6 %; by horizon 96.8-97.8 %. Weather head: standardised RMSE 0.479.

## 5. Interpretation

* The model beats persistence on pooled RMSE, in every city, and at every
  horizon, with the largest gains at +3..+5 h, where persistence degrades.
* The gain is much smaller in MAE (3 %), MAE is worse than persistence for
  the first two hours and for dry hours, and the model is **worse than
  persistence in heavy rain** (RMSE 6.70 vs 5.85 dB). Much of the RMSE gain
  comes from light-rain and dry hours where persistence carries a stale rain
  value; the hard part of the problem (onset and peak of heavy rain) is not
  solved.
* Intervals are too wide for typical hours and much too narrow in heavy
  rain; overall 95 % coverage (97.4 %) hides this.
* One seed only: the earlier three-seed spread on the stratified hourly
  protocol was 0.004-0.007 dB, so a 0.159 dB gain over persistence is very
  unlikely to be seed noise, but this run has not been repeated.
* Context, not a comparison: on the stratified hourly protocol the same
  architecture reached 0.916 +- 0.006 dB vs persistence 1.042 dB (12 %);
  here 0.859 vs 1.018 dB (16 %), on a different (complete, causal) test set.
  The paper's 0.026 dB is not comparable (protocol and labels differ).

## 6. Integrity

* Tests: 41/41 pass. Results audit: 127/127 (run through a wrapper, so
  `results/paper/audit_report.json` was not rewritten).
* All 366 protected files (results, models including the minute smoke test,
  datasets) match their pre-training MD5; the 90-model manifest matches; no
  file outside the two new experiment folders changed.
* No test data were used for training or model selection: the test arrays
  were only checked against the baseline keys before training and predicted
  once after training (`training/train_causal_hourly.py`).
* Git: only new untracked files (`training/train_causal_hourly.py`,
  `evaluation/evaluate_causal_model.py`, this report); nothing staged,
  committed or pushed.

## 7. Recommended next step

Repeat the run with seeds 1 and 2 (about 3 h each on mains power) to measure
seed variance on this protocol before drawing conclusions. After that, the
most informative model-side work is the heavy-rain weakness: evaluate a
heavy-tailed or quantile loss, or re-weighting of rare rain hours, selected
on the validation set only.
