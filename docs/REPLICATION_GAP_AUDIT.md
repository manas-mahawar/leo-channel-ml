# Replication gap audit: El-Sayed et al. (2026) vs this implementation

Audit date: 2026-10-26. Paper: *EURASIP J. Wireless Commun. Netw.* 2026:102,
doi:10.1186/s13638-026-02667-w (PDF on the user's Desktop, not in the
repository; text extracted and figures viewed for this audit).

Evidence labels used throughout: **[Paper]** stated in the paper (page,
table, figure or equation given); **[Code]** verified in our source;
**[Measured]** computed from files in this repository; **[Hypothesis]**
plausible but not proven.

Status values: MATCH, MISMATCH (we differ from an explicit statement),
AMBIGUOUS (the paper is unspecified or contradicts itself), NOT VERIFIABLE
(would need the authors' data or code).

Supporting computations: `evaluation/replication_audit_checks.py` and
`evaluation/make_paper_comparison.py` (outputs in `results/replication_audit/`).

## 0. Inventory (Phase 1)

* Git: branch `main`, two commits; nothing staged. Modified tracked files are
  older prototypes (`data_pipeline/build_dataset.py`, `create_features.py`,
  `create_sequences.py`, `download_era5.py`, `prepare_era5_weather.py`,
  `physics/*_attenuation.py`) plus `.gitignore` and `data_pipeline/cities.py`;
  the replication pipeline itself is untracked. Nothing was reset or discarded.
* Pipeline in use: `config.py`, `physics/build_attenuation.py`,
  `data_pipeline/build_paper_dataset.py` (+ `_minute_`), `training/train_paper_model.py`,
  `evaluation/*.py`. The prototype modules `physics/gaseous_attenuation.py`,
  `rain_attenuation.py`, `cloud_attenuation.py`, `data_pipeline/build_dataset.py`,
  `create_features.py`, `create_sequences.py`, `prepare_era5_weather.py` are
  standalone scripts from the earlier Athens prototype; no pipeline file
  imports them [Code]. They are excluded from this audit's MATCH/MISMATCH
  verdicts (one of them, `gaseous_attenuation.py`, uses `itur`'s exact
  P.676 slant-path mode, which ignores measured T/P; this is why the
  pipeline re-implemented it).
* Runs: 15 complete (`models/paper/<variant>/`, seeds 42; `models/paper/seeds/`,
  seeds 1-2), each with `best.keras`, `model.keras`, `run_config.json`,
  `training_log.csv`, val/test predictions. Model artefact MD5 list:
  `backups/2026-10-26/model_artifacts_md5.txt` (unchanged during this audit).
* Datasets: `data/processed/paper/`, `paper_msl/`, `paper_minute/` with
  `manifest.json`; physics tables in `data/processed/physics_paper/`.
* Results: `results/paper/` (metrics, tables, figures, audit report);
  backups in `backups/2026-10-23..26/`.
* Not in the review bundle: datasets, weights, arrays, per-window prediction
  CSVs (12 files, 4.7-5.6 MB), backups. Everything below that needs them was
  computed here, not from the bundle.

## A. Dataset construction

| Item | Paper | Our code | Status | Impact | Action |
|---|---|---|---|---|---|
| Training cities | 6, coordinates Table 3 (p. 8) | `data_pipeline/cities.py`, same coordinates | MATCH | - | - |
| Period / source | "five consecutive years", ERA5 via Open-Meteo, hourly (Sect. 3.1, p. 8) | 2020-2024, Open-Meteo archive (`download_openmeteo.py`) | MATCH (years not stated) | LOW | - |
| Samples | "exactly 10,000 hourly samples for each city", stratified over months and hours (p. 8) | `build_paper_dataset.stratified_anchors`: 10,000 per city over 288 (month, hour) strata, 33-36 each [Measured] | MATCH | - | - |
| What is windowed | Fig. 2 (p. 10): "1-minute uniform sampling", "sliding window", "Stride: 5 minutes"; Fig. 4 x-axis about 470,000 test samples (p. 14) | Main: windows around the 10,000 hourly anchors (8,916 test windows). Minute variant: same anchors + random minute offset | AMBIGUOUS, our main protocol differs from Fig. 2/Fig. 4 | **HIGH** | See B |
| Split | chronological 70/15/15 per location (p. 9); Fig. 2: "70%:15%:15% Training:Testing:Validation" | `split_anchors`: chronological 70/15/15 per city, 65-h purge gap [Code, tested] | MATCH (order of the last two splits ambiguous in Fig. 2) | LOW | - |
| Windows crossing splits | not discussed | purge gap = SEQ_LEN + HORIZON; no shared hour [tested] | MATCH (stricter) | - | - |
| Normalisation | "standardized using the statistics of the training set only" (p. 9); Fig. 2: "Geographic Normalization, City-specific processing", caption "normalized by location" | one `StandardScaler` over all six cities' training windows | AMBIGUOUS | MEDIUM | Optional per-city-scaler experiment (plan stage 8) |
| Missing values | not discussed | 0 missing hours in all 30 city-years [Measured] | MATCH | - | - |
| Unseen cities | 10 cities, inputs synthesized from K=3 neighbours, Eq. 12-13 (p. 11) | `evaluate_unseen_cities.py`: weighted (tau 500 km) and median modes, own-weather mode as extra | MATCH in form; tau unspecified; evaluation period unspecified ("Live Open-Meteo Validation", Fig. 2) | MEDIUM | - |
| Unseen-city truth | "reference physics-based values" (Fig. 5 caption) | ITU-R labels from each city's own ERA5 weather | MATCH | - | - |

## B. Sampling ambiguity (the decisive gap)

Evidence for each reading [Paper]:

| Statement | Location | Supports |
|---|---|---|
| "Hourly weather data were obtained from ERA5" ; "10,000 hourly samples" | Sect. 3.1, p. 8 | hourly data |
| "Sequence length 60 min", "Prediction horizon 5 h" | Tables 5-6, p. 10/12 | 60 one-minute steps in, 5 h out |
| "60 min sequences of 14 input features" | Sect. 2.3, p. 5 | minute steps |
| "physical parameters from the past hour", "over 5 h" | Fig. 1 caption, p. 7 | 60 min in, 5 h out |
| "Temporal Alignment 1-minute uniform sampling", "Input Sequence: 60 time step", "Prediction Horizon: 5 steps", "Stride: 5 minutes" | Fig. 2, p. 10 | minute steps, 5 one-minute steps out |
| Temporal features "Minute-of-hour, Hour-of-day, Day-of-year" | Table 4, p. 9 | sub-hourly sampling |
| Table 7 RMSE almost flat from hour 1 to 5 (0.026 to 0.027) | p. 13 | very short effective horizon |
| Fig. 4 x-axis "Samples" reaching about 470,000 | p. 14 | many more windows than 10,000 per city |

**[Measured]** Interpolating each city's full 2020-2024 hourly series to 1
minute, taking 60-minute windows every 5 minutes and keeping the last 15 %
gives 6 x 78,914 = **473,484 test windows**, matching Fig. 4's axis. Under
that protocol, trivial baselines need no model at all:

| Horizon reading | Persistence RMSE | Linear extrapolation RMSE |
|---|---|---|
| +1..+5 minutes | 0.035 dB | 0.011 dB |
| +1..+5 hours | 0.945 dB | 2.217 dB |
| Paper (reference) | 0.026 dB | |

**Reconciliation [Hypothesis]:** the paper's numbers fit "full series
interpolated to 1 min, 60-min input, 5 one-minute steps out, 5-min stride".
Under this reading the "5 h" horizon and "Hour 1-5" labels would be
mislabelled minutes. The text's "hourly samples / 5 h" and the 10,000
stratified samples cannot both be literally true together with Fig. 4.
Linear interpolation makes every 1-minute series piecewise linear, so
linear extrapolation is exact except across hour boundaries (unit test
`TestMinuteInterpolation`; `results/paper/minute_diagnostics.json`: exact
for 92.5 % of windows). Such a test is easy by construction and measures
interpolation, not forecasting skill.

**Our code [Code]:** the main dataset implements the hourly reading (60 h
in, 5 h out); `build_minute_dataset.py` implements a minute reading but
only around the 10,000 stratified anchors (8,916 test windows), not the
full series. **Status: AMBIGUOUS; our main protocol is one of two
incompatible readings.** **Impact: HIGH** (factor about 30 in RMSE). The
per-city pattern is *not* reproduced by the minute baselines (paper Bergen
0.148 vs others 0.003-0.039; our minute persistence 0.028-0.048 for every
city), so the hypothesis explains the magnitude but not Table 7's city
ranking. NOT VERIFIABLE without the authors' code or data.

## C. Target construction and physical models

| Item | Paper | Our code | Status | Impact |
|---|---|---|---|---|
| EPL definition | Eq. 6-7: EPL = Pr,fs - Pr,meas = L_atm + epsilon (p. 6-7); Sect. 3.1: EPL "computed as the sum of gaseous, cloud, and rain attenuation" | `compute_attenuation`: gas + cloud + rain, epsilon = 0 | MATCH with Sect. 3.1 (epsilon is a modelling term, not added to labels) | - |
| Gas | P.676 from T, P, RH, fixed elevation (p. 8); Eq. 8 mu_a = gamma_oxygen x L_eff (p. 9) | `gaseous_attenuation`: P.676-12 Annex 1 gamma_o + gamma_w at the surface (dry pressure), Annex 2 equivalent heights, / sin(el) | MATCH with text; Eq. 8 mentions oxygen only (AMBIGUOUS) | LOW (about 0.75 dB nearly constant) |
| P.676 version | P.676-13 cited | P.676-12 (`itur` 0.4.0) | MISMATCH (version) | LOW [Hypothesis] |
| Cloud | P.840, LWP "simple proxy derived from RH and precipitation" (p. 8); Eq. 4 A_c = LWP K_l / sin(theta) | `cloud_attenuation`: P.840-8 K_l(39 GHz, 0 degC) = 1.233; proxy a*clip((RH-65)/35)+b*precip fitted to ERA5 tclw (`physics/lwp_proxy_calibration.json`) | AMBIGUOUS (formula not given) | MEDIUM (affects level, about 0.1-0.5 dB) |
| Rain | P.838 + P.618, rain height P.839 (p. 8); Table 2: mu_r "from ITU-R P.838" | `rain_attenuation`: P.838-3 gamma(R), P.618-13 steps 1-9 with the **hourly** R in place of R_0.01; equals `itur.rain_attenuation(R001=R, p=0.01)` to 0.0 dB [tested] | AMBIGUOUS: P.618 is a statistical method (A exceeded 0.01 % of the time); applying it to instantaneous hourly R is our interpretation, the paper does not say how | **HIGH** for the label tail |
| P.618 version | P.618-14 cited | P.618-13 | MISMATCH (version) | LOW |
| Rain input / units | "hourly precipitation", Table 2 "precipitation rate" (mm) | ERA5 liquid `rain_mm` per hour = mm/h (snow excluded) | AMBIGUOUS (paper may use total precipitation) | MEDIUM in winter cities |
| Elevation | "fixed elevation angle representative of LEO" (p. 8); Eq. 9 geometric formula; no value | 30 deg (`config.ELEVATION_DEG`) | AMBIGUOUS | **HIGH** for level (all components scale about 1/sin(el)) |
| Polarisation | not given | circular (tau = 45 deg) | AMBIGUOUS | LOW |
| Label level [Measured] | Table 8 true mean 0.911 dB (unseen cities); Fig. 3/4 true EPL up to about 6.2-6.7 dB | unseen-city mean 1.153 dB; trained-city test targets up to 25.6 dB; 2.5 % of test windows exceed 6.7 dB and carry 67 % of our squared error | NOT VERIFIABLE (different periods; paper data unavailable) | HIGH for RMSE |

**Label sensitivity [Measured, `evaluation/label_sensitivity.py`,
`results/replication_audit/label_sensitivity.csv`]** (no retraining):

| Elevation | Rain input | Train-city mean / p99 / max EPL (dB) | Unseen-city mean of period means (dB) |
|---|---|---|---|
| 30 deg (ours) | liquid rain (ours) | 1.257 / 6.76 / 25.6 | 1.153 |
| 30 deg | total precipitation | 1.264 / 6.77 / 25.6 | 1.160 |
| 45 deg | liquid rain | 0.911 / 5.10 / 20.8 | 0.833 |
| 60 deg | liquid rain | 0.758 / 4.37 / 17.9 | 0.691 |
| 90 deg | liquid rain | 0.667 / 3.94 / 16.2 | 0.607 |
| Paper (reference) | - | max about 6.2-6.7 (Figs. 3-4) | 0.911 (Table 8) |

* Liquid rain vs total precipitation changes nothing material.
* Elevation shifts the level: the paper's unseen-city mean lies between our
  30 deg and 45 deg values (periods differ, so this is only indicative).
* No elevation reproduces the paper's maximum of about 6.7 dB: even at
  90 deg our maximum is 16 dB. The paper's much smaller rain tail therefore
  points to a different rain-attenuation computation (or rain-rate input),
  which the paper does not specify [Hypothesis]. We do **not** change the
  elevation or rain model to match the published values; that would be
  tuning to the target numbers.

No formula, unit or conversion error was found in the pipeline's physics
[Code + tests]: P.840 K_l (degC, fixed earlier), dry-air pressure for P.676,
mm/h rain, the P.618 step equations (checked against the Recommendation
and `itur`), EPL = exact sum (unit test).

## D. Input features and target

| Item | Paper | Our code | Status | Impact |
|---|---|---|---|---|
| 14 features, order | Table 2 (p. 6), Fig. 1 "60*14" | `config.FEATURES`: lat, lon, T, RH, P, precip, cloud, mu/sigma gas, cloud, rain, elevation | MATCH | - |
| Pressure | Table 2: "at mean sea level"; Sect. 3.1: "surface pressure" | surface (main); MSL in `multitask_msl` | AMBIGUOUS (paper contradicts itself) | LOW trained cities, HIGH for Davos/Madrid with own inputs [Measured] |
| mu features | expected attenuation (Table 2), Eq. 8 mu = gamma x L_eff | instantaneous component at each hour | MATCH | - |
| sigma features | "uncertainty in ... attenuation prediction" (Table 2); Eq. 8 sigma_a = f(T, P, RH) x L_eff, f unspecified | trailing 24-h standard deviation of the component (`add_features`) | AMBIGUOUS / likely MISMATCH in definition | LOW-MEDIUM (`no_itu` ablation shows these inputs barely matter) |
| Temporal features | Table 4: minute-of-hour, hour-of-day, day-of-year | not used (14 features) | AMBIGUOUS (Table 4 vs Table 2 / Fig. 1) | LOW |
| Future leakage | - | inputs use hours t-59..t only; sigma uses t-23..t; targets t+1..t+5 [tested] | MATCH | - |
| Target | EPL in dB, 5 steps (Fig. 1) | EPL dB at t+1..t+5, same construction for all cities | MATCH (step size: see B) | - |

## E. Model architecture

| Item | Paper (Fig. 1, p. 7; Tables 5-6) | Our code (`train_paper_model.build_model`) | Status |
|---|---|---|---|
| Encoder | 3 LSTM layers x 256, dropout 0.2 | same; dropout between LSTM layers | MATCH (dropout placement unspecified) |
| Input | 60 x 14 | 60 x 14 | MATCH |
| Weather head | "Dense: 256 -> 256 -> 25", reshape 5 x 5 | main: Dense(25); `multitask_fig1heads`: Dense(256, ReLU) -> Dense(25) | main MISMATCH; fig1heads MATCH (activation unspecified) |
| EPL head | "Dense: 256 -> 256 -> 10", split mu + log sigma^2 | main: Dense(10); fig1heads: Dense(256, ReLU) -> Dense(10) | main MISMATCH; fig1heads MATCH |
| Parameters | not given | 1,337,123 (main), 1,468,707 (fig1heads) | - |

**Conclusion:** `multitask_fig1heads` is the architecture closest to the paper;
the "main" model is a deviation kept for continuity. That fig1heads is
also the best variant is a separate, measured fact (three seeds, validation
and test), not the reason for this conclusion.

## F. Training procedure and loss

| Item | Paper | Our code | Status |
|---|---|---|---|
| Optimiser | Adam, lr 1e-4, batch 128, up to 50 epochs (p. 9, Table 5) | same | MATCH |
| Early stopping | "early stopping (patience = 6)" (p. 9); Table 6: "Early stopping 23 epochs" | patience 6 on val loss, best weights restored | AMBIGUOUS (Table 6 gives an epoch count, not a patience; we follow the text) |
| Seed | 42 (p. 9) | 42 (+ 1, 2 repeats) | MATCH |
| Loss | Eq. 11: 1.0 MSE(weather) + 1.0 NLL(EPL); Eq. 10: mean of 0.5(log sigma^2 + (y-mu)^2/sigma^2) | `gaussian_nll` = Eq. 10 exactly (mean over batch and 5 steps), Keras MSE on standardised weather, weights 1.0 | MATCH (equals NumPy Eq. 10, tested) |
| Variance parameterisation | log sigma^2 output (Fig. 1) | log-variance, clipped to [-12, 6] | MATCH; clip is our numerical safeguard |
| Scaling of targets | weather targets standardised? not stated; EPL in dB | weather standardised (train stats), EPL raw dB | AMBIGUOUS |

## G. Evaluation metrics

| Item | Paper | Our code | Status |
|---|---|---|---|
| RMSE / MAE | in dB, per city and hour (Table 7) | `evaluate_paper_model.rmse`, `per_city_table`; predictions already in dB | MATCH |
| Overall RMSE | "Overall" row; definition not given | sqrt(mean of all squared errors) over windows and steps | AMBIGUOUS. [Measured] for our main model: pooled 0.936, mean of city RMSEs 0.910, RMS of city RMSEs 0.936 (equal city sizes, 1,481-1,490 windows each). Differences of at most 0.03 dB; cannot explain the gap to 0.026 |
| NLL | Eq. 10 | `uncertainty_metrics` NLL_eq10 (and with constant) | MATCH |
| PICP 95 % | mu +- 1.96 sigma | same | MATCH |
| Sharpness | "mean predictive standard deviation" (p. 13) | mean sigma | MATCH |
| Unseen-city error | "average absolute error", Table 8 | abs(mean pred - mean true) per city, averaged; also hourly MAE/RMSE | MATCH (plus extra metrics) |
| Weather "accuracy" per condition | Sect. 4.6, Fig. 7 "Accuracy %" | not computed: undefined in the paper | NOT VERIFIABLE |

## H. Numerical consistency of the paper [Measured from published values]

| Check | Result | Interpretation |
|---|---|---|
| Table 7 overall vs city averages | mean 0.036, pooled RMS (equal weights) 0.063, median 0.011, reported 0.026 | Inconsistent under equal city sizes for both usual definitions. A pooled RMS of 0.026 would need Bergen to hold at most 3 % of test samples. Possibly a different overall definition or test set; cannot be decided. |
| Table 7 hour 1 | mean 0.034, RMS 0.061, reported 0.026 | same |
| Table 8 averages | true 0.911 and pred 1.089 reproduce; listed errors average 0.211, stated 0.213 | The 0.213 equals the average of the **Fig. 5** bars (0.2127), where Berlin is 0.097 and Stockholm 0.171 instead of Table 8's 0.087 and 0.162. Table and figure disagree for two cities. |
| Fig. 7 | "Avg Error 0.212", "Std Dev 0.082" | std of Fig. 5 errors: 0.080 (population), 0.085 (sample); close but not exact |
| Fig. 6 regions | 0.277 / 0.177 / 0.167 / 0.140 / 0.293 | Reproduced exactly from Fig. 5 values if Berlin is grouped in Western Europe (our original code grouped it as Eastern). |
| Table 9 | EPL-only 0.015 vs multi-task 0.026: EPL-only is 42 % better (multi-task 73 % worse); weather 14.7 % better with multi-task; time +31.7 % | The "40 %" in the Improvement row matches a *reduction from multi-task to EPL-only*, contradicting the text's "substantial gains in EPL prediction accuracy" for multi-task. 14.7 % and +32 % reproduce. |
| Table 10 | RMSE +7.7 %, MAE +33 % without ITU-R features | arithmetic consistent; interpretation (no leakage) is the authors' |
| Coverage | 99.03 % (Sect. 4.1, Table 11) vs PICP 99.23 % (Sect. 4.1) | two different values for the same quantity |

## Prioritised discrepancies (Phase 4)

| # | Discrepancy | Evidence | Impact | Confidence | Fixable without author data? | Smallest test |
|---|---|---|---|---|---|---|
| 1 | Sampling/horizon protocol (hourly 5 h vs interpolated minutes, 5-min stride, full series) | B; Fig. 4 count 473,484 reproduced | HIGH | High that the paper's numbers fit the minute reading; unprovable which the authors intended | Yes, as a separately named reconstruction | Baselines done (no training); a model run needs about 3 h/epoch on this CPU (plan stage 7) |
| 2 | Label tail / level (rain model use of P.618 with hourly R, elevation, rain input, cloud proxy) | C; our max 25.6 vs paper about 6.7 dB; unseen mean 1.15 vs 0.91 dB | HIGH for RMSE | Medium (different periods, unknown authors' choices) | No: the authors' choices are unspecified; alternatives can only be documented | Done (label sensitivity table in C): elevation explains level, not tail |
| 3 | Head architecture of "main" model | E; Fig. 1 | MEDIUM (0.024 dB measured) | High | Yes: already trained (`multitask_fig1heads`, 3 seeds) | Done; designate as paper-faithful variant |
| 4 | Paper-internal inconsistencies (Tables 7-9, Fig. 5-7) | H | affects comparability, not our code | High (arithmetic) | n/a | Done |
| 5 | Normalisation per location vs global | A; Fig. 2 | MEDIUM-LOW [Hypothesis] | Low | Yes | One retrain with per-city scalers (plan stage 8, optional) |
| 6 | sigma feature definition | D; Eq. 8 | LOW (no_itu shows little effect) | Medium | No (f unspecified) | Not worth a run |
| 7 | Pressure variable | D | LOW trained / HIGH 2 unseen cities | Medium | Done (`multitask_msl`) | Done |
| 8 | Fig. 6 regional grouping | H | cosmetic | High | Yes | Done (`results/replication_audit/fig6_regions_paper_grouping.png`) |
| 9 | ITU-R versions (P.676-13, P.618-14 cited) | C | LOW [Hypothesis] | Medium | Partly (not in `itur` 0.4.0) | Not worth it |
| 10 | Early-stopping description | F | LOW | High | n/a | Documented |

**Confirmed implementation bugs found in this audit: none** in the current
pipeline (earlier bugs, e.g. P.840 Kelvin vs Celsius and P.676 ignoring T/P,
were fixed on 2026-10-23 and are covered by tests). The large RMSE gap is
explained by protocol and label differences, not by a coding error we could
find.

## Follow-up (2026-10-27)

The rain-label question is examined in `docs/RAIN_LABEL_DIAGNOSTIC.md`:
the rain, gas and cloud implementations reproduce independent ITU-R
references; the label differences are due to unspecified choices, not to a
coding error.
