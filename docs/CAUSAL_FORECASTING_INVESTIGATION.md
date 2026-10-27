# Causal forecasting investigation

Date: 2026-10-27. Scope: decide which forecasting protocol is defensible,
build a strictly causal dataset path, and evaluate simple baselines on
identical keys. **No model was trained or run.** Paper: El-Sayed et al.,
EURASIP J. Wireless Commun. Netw. 2026:102 (PDF read directly; page
numbers are the journal's "Page x of 21").

## 1. Starting state (recorded before any change)

* Git: 49 status entries (pre-existing modified prototypes and untracked
  pipeline files), nothing staged.
* Unit tests: 28/28 pass. Results audit: 127/127 pass (run through a wrapper
  that redirected its report file to the scratchpad, so
  `results/paper/audit_report.json` was not rewritten).
* Fingerprints: MD5 of 355 protected files (all of `results/`,
  `models/paper/`, `models/minute_full_smoke/`, `data/processed/paper*`,
  `physics_paper/`) taken at the start; the 90-model and 214-file manifests
  of 2026-10-26/27 also matched.
* Hourly reference (unchanged): stratified dataset `data/processed/paper/`,
  8,916 test windows; persistence 1.042 dB, climatology 1.235 dB;
  `multitask_fig1heads` 0.916 +- 0.006 dB (three seeds).
* Minute source (unchanged): train 2,208,678, validation 473,076, test
  473,112 windows; smoke test in `results/minute_full_smoke/` (one epoch, 1 %).

## 2. What the paper says about sampling and horizon

| # | Paper statement | Where | Supports | Strength |
|---|---|---|---|---|
| 1 | "historical sequences ... spanning 60 min to simultaneously predict weather conditions and excess path loss (EPL) for 5 h" | Abstract, p. 1 | 60-min input, 5-h horizon | explicit |
| 2 | "Whereas prior work focused on short-term, minute-level forecasting with limited temporal context, 60 min historical sequences are utilized to model the full progression of atmospheric conditions" | Sect. 1, p. 3 | the paper sees itself as *not* minute-level forecasting; input is 60 min | explicit but internally odd (60 min of history is short) |
| 3 | "60 min sequences of 14 input features" | Sect. 2.3, p. 5 | 60-min input | explicit |
| 4 | "Input features (14 dimensions over 60 time steps) include location, weather, and physical parameters from the past hour"; heads output "over 5 h" | Fig. 1 caption, p. 7 | 60 one-minute steps covering the past hour; 5-h output | explicit |
| 5 | "Hourly weather data were obtained from the ERA5 reanalysis database through the Open-Meteo archive"; "exactly 10,000 hourly samples for each city" | Sect. 3.1, p. 8 | source resolution is hourly | explicit |
| 6 | "The dataset was built using real atmospheric measurements combined with standard ITU-R propagation models" | Sect. 3.1, p. 8 | calls reanalysis "measurements" | contradictory with "ERA5 reanalysis" |
| 7 | "For each selected timestamp, the excess path loss was computed as the sum of gaseous, cloud, and rain attenuation components" | Sect. 3.1, p. 8 | EPL target is reconstructed from ITU-R models | explicit |
| 8 | Eq. 6: "EPL = Pr,fs - Pr,measured"; Eq. 7: "EPL = L_atm + epsilon"; "y_i is the actual measured EPL value" | pp. 6, 7, 9 | EPL as measured power difference | contradicts #7 (no received-power data are described) |
| 9 | "fixed elevation angle representative of LEO satellite downlinks" (value not given) | Sect. 3.1, p. 8 | static elevation | explicit, unspecified value |
| 10 | "chronologically divided into training (70%), validation (15%), and testing (15%) subsets for each location" | Sect. 3.1, p. 9 | per-city chronological split | explicit |
| 11 | "All input features were standardized using the statistics of the training set only" | Sect. 3.1, p. 9 | global train-only scaling | explicit |
| 12 | Table 4: temporal features "Minute-of-hour, Hour-of-day, Day-of-year" | p. 9 | sub-hourly time stamps | contradicts the 14 features of Table 2 / Fig. 1 |
| 13 | Fig. 2 caption: "aligned in time, normalized by location, and transformed into fixed-length sequences of 60-time steps ... A sliding window is used"; "real-time inference" | p. 10 | sliding windows; location-wise normalisation | contradicts #11 |
| 14 | Fig. 2 boxes: "Temporal Alignment 1-minute uniform sampling", "Input Sequence: 60 time step", "Prediction Horizon: 5 steps", "Stride: 5 minutes", "Time-based Split 70%:15%:15% Training:Testing:Validation", "Live Open-Meteo Validation" | Fig. 2 image, p. 10 | 1-min sampling, 5 output steps of unstated length | explicit for sampling; step length not stated |
| 15 | Tables 5 and 6: "Sequence length 60 min", "Prediction horizon 5 h" | pp. 10, 12 | 60-min input, 5-h horizon | explicit |
| 16 | Table 7 columns "Hour 1 ... Hour 5"; RMSE 0.026-0.027 for all hours | p. 13 | hourly steps (but almost no growth with lead time) | explicit labels; flat errors are unusual for hours |
| 17 | Fig. 4: "EPL prediction over a 5 h forecast horizon (Hour + 1 to Hour + 5)"; x-axis about 470,000 samples | p. 14 | hourly steps; very many test windows | labels explicit; count matches a 5-minute stride over the full series (gap audit B) |
| 18 | Table 11: prediction horizon "1 min" (Bai et al.) vs "5 h" (proposed) | p. 17 | 5-h horizon, contrasted with 1-min prior work | explicit |
| 19 | No statement that weather inputs are forecasts or that the future-hour value is excluded from inputs; operational availability not discussed | - | - | absent |

**Explicit:** the text, tables and figure captions consistently state a
60-minute input and a **5-hour** horizon (#1, #4, #15-18), on **hourly**
source data (#5), with a chronological per-city split (#10).
**Ambiguous or contradictory:** how 60 one-minute steps are built from hourly
data (#4, #14; not explained); the length of the "5 steps" in Fig. 2 (#14);
measured vs reconstructed EPL (#7 vs #8); global vs per-location
normalisation (#11 vs #13); temporal features (#12); the number of test
samples (#17 vs 10,000 hourly samples per city). The evidence **cannot
determine** the authors' exact protocol.

## 3. Candidate protocols

| | A: hourly five-step | B: minute five-step | C: 1-min input, 5-h horizon |
|---|---|---|---|
| Definition | 60 hourly inputs, targets t+1..t+5 h | 60 one-minute inputs, targets t+1..t+5 min | 60 one-minute inputs (past hour), targets at +1..+5 h |
| Paper support | horizon "5 h", "Hour 1-5", hourly data (#1, #5, #15-18); but "60 min" input contradicts 60 hourly steps | Fig. 2 sampling and stride (#14), Fig. 4 sample count (#17), flat Table 7; contradicts every "5 h" statement | matches the text literally (#1, #4, #15); Fig. 2 "5 steps" could be 5 hourly outputs |
| Data needed | hourly weather and labels (available) | genuine 1-minute weather and EPL (not available) | genuine 1-minute inputs (not available), hourly targets (available) |
| Implementable from hourly ERA5? | yes, exactly | only with interpolated or held minutes; targets would be estimates | inputs only as held values (at most 2 distinct hourly values in 60 minutes) or leaky interpolation |
| Interpolation effect | none | linear interpolation leaks hour h+1 into inputs (Sect. 5); holding makes the 60 inputs nearly constant | linear interpolation leaks hour h+1, i.e. the +1-h target region; holding leaves almost no history |
| Tests genuine forecasting? | yes (retrospectively) | no with interpolation; with held inputs, forecasting of an interpolated proxy | partly; information content is about one hourly observation |
| Comparable with 0.026 dB? | only as "same task as described in the text"; labels differ (rain diagnostic) | numerically closer, but the task is different and trivial under interpolation | no; unclear what the paper computed |

**Decision:** investigate **Protocol A first**. It is the only protocol whose
inputs and targets both exist in the available data without
reconstruction, it matches the paper's repeated "5 h" horizon, and it tests
genuine (retrospective) forecasting. B and C cannot be implemented
faithfully because no minute-level observations exist; B is used only to
*measure* the interpolation leak (Sect. 7). C is not implemented: with
hourly data, a causal 60-minute history holds one or two hourly
observations, so it would be a degenerate version of A.

## 4. Feature availability (protocol causal_hourly_v1)

| Feature | Source | Timestamp / resolution | Type | Available at origin t? | Operations | Future information? |
|---|---|---|---|---|---|---|
| latitude, longitude | Table 3 coordinates | static | metadata | yes | none | no |
| temperature_C, relative_humidity_percent, pressure_hPa (surface), cloud_cover_percent | ERA5 via Open-Meteo | value valid at each hour, hourly | reanalysis | retrospectively yes; operationally only after ERA5 latency (days) | none | no |
| precipitation_mm | ERA5 via Open-Meteo | sum over the hour ending at the timestamp | reanalysis | same | none | no |
| gas_mean, cloud_mean, rain_mean | ITU-R models (`physics/build_attenuation.py`) | same hour | derived | same | functions of that hour's weather (rain uses `rain_mm`, also preceding-hour sum) | no |
| gas_std, cloud_std, rain_std | derived | trailing 24 h (t-23..t) | derived | same | `pandas.rolling(24).std()` (right-aligned, no `center`) | no (tested by perturbation) |
| elevation_deg | config (30 deg) | static | assumption | yes | none | no |
| Target EPL(t+k) | ITU-R labels | hours t+1..t+5 | derived from reanalysis | target (future by design) | none | targets only |
| y(t), y(t-1), mean y(t-23..t) for baselines | same labels | hours <= t | derived | same as features | none | no |

All 14 features are computed per city before splitting, but every operation
is row-wise or trailing, so this introduces no future values (verified by
the future-perturbation tests). Scalers are fitted on training input rows
only. Validation/test inputs may look back into training-period rows (24-h
sigma, 60-h window): this uses the past, not the future. ERA5 is a
retrospective reanalysis, so the protocol is **causal retrospective
evaluation, not operational forecasting**.

## 5. The interpolation leak

The existing minute source interpolates linearly between hourly values: a
minute m with h = floor(m/60) gets (1-w) x(h) + w x(h+1). At a forecast origin
inside hour h, x(h+1) lies up to 60 minutes in the future but already enters
the inputs, and the 5 minute targets lie on the same straight segment.
Linear extrapolation from minutes t-1 and t therefore recovers the targets
exactly unless they cross the hour boundary. This is a property of the
protocol, not a coding error in that source. The test
`tests/test_causal.py::TestCausalMinuteProxy::test_existing_interpolation_uses_the_next_hour`
documents it; the new causal path never uses x(h+1) for an origin before
hour h+1 (`test_minute_inputs_never_use_the_next_hour`).

## 6. Implementation (new files only)

* `data_pipeline/causal_forecast_source.py`: `CausalHourlySource` (protocol
  `causal_hourly_v1`) and `causal_minute_proxy_baselines` (zero-order hold of
  the latest hourly value; used only for Sect. 7).
* `evaluation/evaluate_causal_baselines.py`: baselines on identical keys;
  writes `results/causal_baseline_audit/` (refuses to overwrite).
* `tests/test_causal.py`: 13 tests on synthetic data (future perturbation;
  trailing rolling features; scaler ignores validation/test; inputs at or
  before the origin; targets exactly +1..+5 h; no shared rows across splits;
  sorted unique deterministic origins; missing past values exclude windows
  without filling; too-short series rejected; metadata does not claim
  minute measurements; identical baseline keys and deterministic metrics;
  causal minute inputs never use the next hour; the existing interpolation
  does).

**Dataset `causal_hourly_v1`:** each city's full 2020-2024 hourly series;
every hourly origin with a complete 60-hour input window (including the
24-hour sigma history) and five target hours; no stratified subsampling;
chronological 70/15/15 of each city's origins, dropping the first origins of
the next split until its first input row is after the previous split's last
target row (a gap of at least 65 hours); windows with missing values are
excluded (none occur in these data; 23 early origins per city lack
sigma history). Per city: train 30,633 origins (2020-01-04 10:00 to
2023-07-03 18:00), validation 6,500 (2023-07-06 11:00 to 2024-04-02 06:00),
test 6,500 (2024-04-04 23:00 to 2024-12-31 18:00); identical for all six
cities. Scaling: global `StandardScaler` on training input rows (computed
and stored in `config.json`; the baselines do not use it). Windows are cut
by array indexing from the hourly table; nothing is materialised.

## 7. Baseline results

### 7.1 causal_hourly_v1, test set

39,000 origins (6 x 6,500) x 5 horizons = 195,000 values per baseline; the
same (city, origin, horizon) keys for every baseline
(`results/causal_baseline_audit/aligned_keys_hourly.npz`). Formulas: persistence
y(t); linear extrapolation y(t) + k (y(t) - y(t-1)), k in hours; recent mean
mean(y(t-23..t)).

| Baseline | RMSE (dB) | MAE (dB) | RMSE +1 / +2 / +3 / +4 / +5 h | MAE +1 .. +5 h |
|---|---|---|---|---|
| Persistence | **1.018** | **0.346** | 0.648 / 0.921 / 1.066 / 1.147 / 1.209 | 0.191 / 0.295 / 0.368 / 0.418 / 0.459 |
| Linear extrapolation | 2.519 | 0.718 | 0.913 / 1.672 / 2.375 / 3.024 / 3.647 | 0.267 / 0.500 / 0.726 / 0.943 / 1.154 |
| Recent 24-h mean | 1.126 | 0.494 | 1.065 / 1.102 / 1.131 / 1.155 / 1.174 | 0.463 / 0.481 / 0.496 / 0.509 / 0.521 |

By city (persistence RMSE): Athens 0.665, Bergen 0.959, Budapest 1.285,
Chilbolton 0.962, Vigo 1.216, Warsaw 0.894. By precipitation at the target
hour (observed in the reanalysis at that hour; outcome-based grouping, for
diagnosis only), persistence RMSE: dry 0.510 (153,853 values), light 1.678
(39,672), heavy >= 2.5 mm/h 5.845 (1,475).

Context (different sample, not directly comparable): on the existing
stratified test set (8,916 windows, `results/paper/`) persistence is 1.042
dB and `multitask_fig1heads` 0.916 +- 0.006 dB.

### 7.2 Quantifying the leak: identical minute anchors and targets

473,112 test anchors of `data_pipeline/minute_full_source.py` (5-minute
stride, 2024-04-02 to 2024-12-31), the same linearly interpolated minute
targets t+1..t+5 for all four rows. Only the inputs differ: "leaky" uses the
interpolated EPL at t and t-1 (contains hour h+1); "causal" uses hourly
values up to hour h (zero-order hold; linear trend from hours h-1 and h).
The targets are interpolated estimates, not observations.

| Inputs | Baseline | RMSE (dB) | MAE (dB) | RMSE +1 .. +5 min |
|---|---|---|---|---|
| Leaky (interpolated) | Persistence | 0.0358 | 0.0096 | 0.011 / 0.022 / 0.032 / 0.043 / 0.054 |
| Leaky (interpolated) | Linear extrapolation | 0.0145 | 0.0011 | 0.004 / 0.009 / 0.013 / 0.018 / 0.022 |
| Causal (held hourly) | Persistence | 0.379 | 0.097 | 0.360 / 0.369 / 0.378 / 0.388 / 0.397 |
| Causal (held hourly) | Linear extrapolation | 0.532 | 0.136 | 0.506 / 0.519 / 0.532 / 0.545 / 0.559 |

With everything else identical, removing the future hourly value raises the
persistence error **10.6x** and the linear-extrapolation error **37x**. By
city, causal persistence ranges 0.30-0.51 dB and leaky persistence
0.029-0.048 dB. The minute protocol's very small errors are therefore an
artefact of interpolation, not forecasting skill.

The hourly (7.1) and minute (7.2) results differ in horizon (hours vs
minutes), target type (hourly labels vs interpolated estimates) and sample
(39,000 hourly origins vs 473,112 minute anchors), so they must not be
compared as if only leakage differed.

## 8. Comparison with the paper's 0.026 dB

A like-for-like comparison is **not possible**: the authors' labels and
protocol are unavailable, and the paper's statements conflict. Under the
protocol its text describes (5-hour horizon), the simplest causal baseline
already has 1.018 dB RMSE on our labels, and the trained model 0.916 dB on
the stratified test set; under the interpolated minute protocol, numbers of
the paper's size are reached by baselines that exploit the interpolation
leak (0.0145-0.036 dB) and vanish when the leak is removed (0.38-0.53 dB).
Our labels also have a much larger rain tail than the paper's (see
`docs/RAIN_LABEL_DIAGNOSTIC.md`), which affects any comparison.

## 9. Limitations and open questions

* ERA5 is retrospective: "causal" here means no future timestamp is used, not
  that the inputs would be available in real time. Operational forecasting
  would need weather forecasts or near-real-time observations as inputs.
* Labels are ITU-R model outputs from reanalysis weather, not received-power
  measurements, in both this work and (apparently, #7) the paper.
* No minute-level observations exist; any minute protocol uses interpolated
  or held hourly values. Genuine minute-level validation is not possible with
  the current data.
* Unresolved from the paper: how 60 one-minute steps were built; the length
  of Fig. 2's "5 steps"; per-location vs global normalisation; measured vs
  reconstructed EPL; the source of about 470,000 test samples.
* Precipitation classes are defined at the target hour (outcome-based) and
  are diagnostic groupings only.

## 10. Recommended next step (before any neural-network training)

Train nothing yet. The decision to make first is which protocol the next
model run should use. Recommended: **`causal_hourly_v1` with all hourly
origins** (as built here), training `multitask_fig1heads` on it and
comparing against persistence on the same 39,000 test origins. It needs a
small training adapter (array-indexed windows from `CausalHourlySource`, which
fit in memory: 183,798 training windows of 60 x 14 float32 is about 0.6 GB),
and about 4.4 x the cost of the stratified hourly runs per epoch (about
16 min per epoch on mains power, about 32 min on battery, scaled from the
measured 3.65 min per 42,000 windows). This needs your approval.
